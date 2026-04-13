from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


POSITIVE_LABEL = "positive"
NEGATIVE_LABEL = "negative"


@dataclass(frozen=True)
class PredictionRow:
    model_id: str
    stage: str
    branch: str
    family: str
    label_fraction: str
    source_key: str
    source_id: str
    split: str
    actual_label: str
    predicted_label: str
    max_probability_sickled: float
    decisive_x: int
    decisive_y: int
    decisive_size: int
    n_windows: int
    raw_path: str
    overlay_path: str
    gradcam_path: str


@dataclass
class BoxCluster:
    x1: int
    y1: int
    x2: int
    y2: int
    labels: list[str]
    probabilities: list[float]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_predictions(path: Path) -> list[PredictionRow]:
    rows: list[PredictionRow] = []
    for row in read_csv(path):
        rows.append(
            PredictionRow(
                model_id=row["model_id"],
                stage=row["stage"],
                branch=row["branch"],
                family=row["family"],
                label_fraction=row["label_fraction"],
                source_key=row["source_key"],
                source_id=row["source_id"],
                split=row["split"],
                actual_label=row["actual_label"],
                predicted_label=row["predicted_label"],
                max_probability_sickled=float(row["max_probability_sickled"]),
                decisive_x=int(row["decisive_x"]),
                decisive_y=int(row["decisive_y"]),
                decisive_size=int(row["decisive_size"]),
                n_windows=int(row["n_windows"]),
                raw_path=row["raw_path"],
                overlay_path=row["overlay_path"],
                gradcam_path=row["gradcam_path"],
            )
        )
    return rows


def load_metrics(path: Path) -> list[dict[str, str]]:
    return read_csv(path)


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def open_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def get_value(row: PredictionRow | dict[str, str], key: str) -> str:
    if isinstance(row, dict):
        return row[key]
    return str(getattr(row, key))


def model_short_label(row: PredictionRow | dict[str, str]) -> str:
    stage_value = get_value(row, "stage")
    branch_value = get_value(row, "branch")
    family_value = get_value(row, "family")
    fraction_value = get_value(row, "label_fraction")
    stage = "S5" if stage_value == "stage5" else "S6"
    branch = "RGB" if branch_value == "rgb" else "Gray"
    family_map = {
        "supervised_scratch": "Scratch",
        "imagenet_finetune": "ImgNet",
        "simclr_finetune": "SimCLR",
    }
    family = family_map.get(family_value, family_value)
    if stage_value == "stage5":
        return f"{stage} {branch} {family}"
    return f"{stage} {branch} {fraction_value} {family}"


def model_sort_key(row: PredictionRow | dict[str, str]) -> tuple[int, int, str, str]:
    stage_order = 0 if get_value(row, "stage") == "stage5" else 1
    branch_order = 0 if get_value(row, "branch") == "rgb" else 1
    fraction_order = int(get_value(row, "label_fraction"))
    return (stage_order, branch_order, fraction_order, get_value(row, "family"))


def prediction_state(row: PredictionRow) -> str:
    if row.actual_label == POSITIVE_LABEL and row.predicted_label == POSITIVE_LABEL:
        return "TP"
    if row.actual_label == NEGATIVE_LABEL and row.predicted_label == NEGATIVE_LABEL:
        return "TN"
    if row.actual_label == NEGATIVE_LABEL and row.predicted_label == POSITIVE_LABEL:
        return "FP"
    return "FN"


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(a_area + b_area - inter)


def wrap_labels(labels: list[str], per_line: int = 3) -> list[str]:
    return [" | ".join(labels[i : i + per_line]) for i in range(0, len(labels), per_line)]


def cluster_rows(rows: list[PredictionRow], overlap_threshold: float) -> list[BoxCluster]:
    clusters: list[BoxCluster] = []
    sorted_rows = sorted(rows, key=lambda row: row.max_probability_sickled, reverse=True)
    for row in sorted_rows:
        box = (
            row.decisive_x,
            row.decisive_y,
            row.decisive_x + row.decisive_size,
            row.decisive_y + row.decisive_size,
        )
        label = f"{model_short_label(row)} ({row.max_probability_sickled:.2f})"
        matched = False
        for cluster in clusters:
            cluster_box = (cluster.x1, cluster.y1, cluster.x2, cluster.y2)
            if iou(box, cluster_box) >= overlap_threshold:
                cluster.x1 = min(cluster.x1, box[0])
                cluster.y1 = min(cluster.y1, box[1])
                cluster.x2 = max(cluster.x2, box[2])
                cluster.y2 = max(cluster.y2, box[3])
                cluster.labels.append(label)
                cluster.probabilities.append(row.max_probability_sickled)
                matched = True
                break
        if not matched:
            clusters.append(
                BoxCluster(
                    x1=box[0],
                    y1=box[1],
                    x2=box[2],
                    y2=box[3],
                    labels=[label],
                    probabilities=[row.max_probability_sickled],
                )
            )
    return clusters


def draw_master_image(
    image_path: Path,
    source_key: str,
    actual_label: str,
    rows: list[PredictionRow],
    output_path: Path,
    overlap_threshold: float,
    max_display_size: int,
) -> None:
    base_image = open_rgb_image(image_path)
    scale = min(1.0, max_display_size / max(base_image.width, base_image.height))
    display = base_image.copy()
    if scale < 1.0:
        display = display.resize((int(round(display.width * scale)), int(round(display.height * scale))))

    positive_rows = [row for row in rows if row.predicted_label == POSITIVE_LABEL]
    clusters = cluster_rows(positive_rows, overlap_threshold)
    draw = ImageDraw.Draw(display)
    font = ImageFont.load_default()

    title = f"{source_key} | actual={actual_label} | positive detections={len(positive_rows)} | clusters={len(clusters)}"
    title_width = min(display.width, max(220, len(title) * 7 + 12))
    draw.rectangle((0, 0, title_width, 24), fill="black")
    draw.text((6, 6), title, fill="white", font=font)

    for cluster in clusters:
        x1 = int(round(cluster.x1 * scale))
        y1 = int(round(cluster.y1 * scale))
        x2 = int(round(cluster.x2 * scale))
        y2 = int(round(cluster.y2 * scale))
        color = "lime" if actual_label == POSITIVE_LABEL else "red"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, int(round(3 * scale))))

        lines = wrap_labels(cluster.labels, per_line=3)
        text_x = min(display.width - 5, x2 + 6)
        text_y = max(28, y1)
        if text_x + 220 > display.width:
            text_x = max(5, x1 - 220)
        box_height = 14 * len(lines) + 8
        box_width = min(display.width - text_x - 5, max(len(line) * 6 + 10 for line in lines))
        draw.rectangle((text_x, text_y, text_x + box_width, text_y + box_height), fill="black")
        for idx, line in enumerate(lines):
            draw.text((text_x + 4, text_y + 4 + idx * 14), line, fill=color, font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    display.save(output_path)


def plot_model_leaderboard(metrics: list[dict[str, str]], output_path: Path) -> None:
    rows = sorted(metrics, key=lambda row: (-float(row["balanced_accuracy"]), row["model_id"]))
    labels = [model_short_label(row) for row in rows]
    values = [float(row["balanced_accuracy"]) for row in rows]
    colors = ["#1f77b4" if row["branch"] == "rgb" else "#ff7f0e" for row in rows]
    plt.figure(figsize=(12, max(8, len(rows) * 0.28)))
    y = np.arange(len(rows))
    plt.barh(y, values, color=colors)
    plt.yticks(y, labels, fontsize=8)
    plt.xlabel("Balanced Accuracy")
    plt.title("Full-Image Detection Leaderboard")
    plt.xlim(0.0, 1.0)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_prediction_matrix(predictions: list[PredictionRow], output_path: Path) -> None:
    image_keys = sorted({row.source_key for row in predictions})
    model_ids = [model_id for model_id in sorted({row.model_id for row in predictions})]
    order_lookup = {
        model_id: model_sort_key(next(row for row in predictions if row.model_id == model_id))
        for model_id in model_ids
    }
    model_ids.sort(key=lambda model_id: order_lookup[model_id])
    state_map = {"TN": 0, "FP": 1, "FN": 2, "TP": 3}
    matrix = np.zeros((len(model_ids), len(image_keys)), dtype=np.int64)
    for i, model_id in enumerate(model_ids):
        lookup = {row.source_key: row for row in predictions if row.model_id == model_id}
        for j, image_key in enumerate(image_keys):
            matrix[i, j] = state_map[prediction_state(lookup[image_key])]

    cmap = matplotlib.colors.ListedColormap(["#d9d9d9", "#d62728", "#ffbf00", "#2ca02c"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(max(12, len(image_keys) * 0.6), max(10, len(model_ids) * 0.3)))
    plt.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    plt.xticks(np.arange(len(image_keys)), image_keys, rotation=60, ha="right", fontsize=8)
    plt.yticks(np.arange(len(model_ids)), [model_short_label(next(row for row in predictions if row.model_id == model_id)) for model_id in model_ids], fontsize=7)
    plt.title("Model Outcome Matrix Across Sampled Full Images")
    cbar = plt.colorbar(ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["TN", "FP", "FN", "TP"])
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_per_image_positive_votes(predictions: list[PredictionRow], output_path: Path) -> None:
    by_image: dict[str, list[PredictionRow]] = defaultdict(list)
    for row in predictions:
        by_image[row.source_key].append(row)
    image_keys = sorted(by_image)
    counts = [sum(1 for row in by_image[key] if row.predicted_label == POSITIVE_LABEL) for key in image_keys]
    labels = [by_image[key][0].actual_label for key in image_keys]
    colors = ["#2ca02c" if label == POSITIVE_LABEL else "#d62728" for label in labels]
    plt.figure(figsize=(12, 5))
    plt.bar(np.arange(len(image_keys)), counts, color=colors)
    plt.xticks(np.arange(len(image_keys)), image_keys, rotation=60, ha="right", fontsize=8)
    plt.ylabel("Number of Models Predicting Positive")
    plt.title("Per-Image Positive Vote Count Across All Models")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_branch_family_summary(metrics: list[dict[str, str]], output_path: Path) -> None:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in metrics:
        grouped[(row["branch"], row["family"])].append(float(row["balanced_accuracy"]))
    keys = sorted(grouped)
    labels = [f"{'RGB' if branch == 'rgb' else 'Gray'} | {family}" for branch, family in keys]
    values = [float(np.mean(grouped[key])) for key in keys]
    colors = ["#1f77b4" if key[0] == "rgb" else "#ff7f0e" for key in keys]
    plt.figure(figsize=(10, 4.5))
    plt.bar(np.arange(len(keys)), values, color=colors)
    plt.xticks(np.arange(len(keys)), labels, rotation=35, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Mean Balanced Accuracy")
    plt.title("Average Full-Image Accuracy By Branch And Family")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def build_verdict_report(
    predictions: list[PredictionRow],
    metrics: list[dict[str, str]],
    output_path: Path,
    visuals_dir: Path,
) -> None:
    best_rows = sorted(metrics, key=lambda row: (-float(row["balanced_accuracy"]), row["model_id"]))[:5]
    by_image: dict[str, list[PredictionRow]] = defaultdict(list)
    for row in predictions:
        by_image[row.source_key].append(row)
    image_vote_rows = []
    for source_key, rows in sorted(by_image.items()):
        n_positive = sum(1 for row in rows if row.predicted_label == POSITIVE_LABEL)
        actual_label = rows[0].actual_label
        image_vote_rows.append((source_key, actual_label, n_positive, len(rows) - n_positive))

    worst_negative = sorted(
        (row for row in image_vote_rows if row[1] == NEGATIVE_LABEL),
        key=lambda item: item[2],
        reverse=True,
    )[:5]
    weakest_positive = sorted(
        (row for row in image_vote_rows if row[1] == POSITIVE_LABEL),
        key=lambda item: item[2],
    )[:5]

    lines = [
        "# Full-Image Sliding-Window Verdict",
        "",
        "## High-Level Verdict",
        "- This box-free full-image test is much harder than the crop task.",
        "- The dominant failure mode is false positives on negative full images, not missed positives.",
        "- Most models detected all 10 positive images, but many also triggered on most or all 10 negative images.",
        "- That means the current crop classifiers do not yet transfer cleanly to real whole-image detection under dense sliding windows.",
        "",
        "## Best Models By Balanced Accuracy",
    ]
    for row in best_rows:
        lines.append(
            f"- `{model_short_label(row)}`: balanced_accuracy `{float(row['balanced_accuracy']):.3f}`, "
            f"sensitivity `{float(row['sensitivity']):.3f}`, specificity `{float(row['specificity']):.3f}`."
        )

    lines.extend(
        [
            "",
            "## Most Problematic Negative Images",
            "These are negative full images that the largest number of models still called positive.",
        ]
    )
    for source_key, actual_label, n_positive, n_negative in worst_negative:
        lines.append(f"- `{source_key}`: `{n_positive}` models predicted positive, `{n_negative}` predicted negative.")

    lines.extend(
        [
            "",
            "## Least Supported Positive Images",
            "These are positive full images with the weakest model agreement.",
        ]
    )
    for source_key, actual_label, n_positive, n_negative in weakest_positive:
        lines.append(f"- `{source_key}`: `{n_positive}` models predicted positive, `{n_negative}` predicted negative.")

    lines.extend(
        [
            "",
            "## Useful Visuals",
            f"- Leaderboard: [{(visuals_dir / 'model_leaderboard.png').name}]({(visuals_dir / 'model_leaderboard.png').resolve()})",
            f"- Outcome matrix: [{(visuals_dir / 'prediction_matrix.png').name}]({(visuals_dir / 'prediction_matrix.png').resolve()})",
            f"- Per-image vote counts: [{(visuals_dir / 'per_image_positive_votes.png').name}]({(visuals_dir / 'per_image_positive_votes.png').resolve()})",
            f"- Branch/family summary: [{(visuals_dir / 'branch_family_summary.png').name}]({(visuals_dir / 'branch_family_summary.png').resolve()})",
        ]
    )
    write_markdown(output_path, "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create analysis visuals for full-image sliding-window evaluation outputs.")
    parser.add_argument("--input-dir", type=Path, default=Path("full_image_sliding_window_outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("full_image_sliding_window_outputs/analysis_visuals"))
    parser.add_argument("--max-display-size", type=int, default=1600)
    parser.add_argument("--overlap-threshold", type=float, default=0.3)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions(input_dir / "image_level_predictions.csv")
    metrics = load_metrics(input_dir / "image_level_metrics.csv")

    plot_model_leaderboard(metrics, output_dir / "model_leaderboard.png")
    plot_prediction_matrix(predictions, output_dir / "prediction_matrix.png")
    plot_per_image_positive_votes(predictions, output_dir / "per_image_positive_votes.png")
    plot_branch_family_summary(metrics, output_dir / "branch_family_summary.png")

    grouped_rows: dict[str, list[PredictionRow]] = defaultdict(list)
    for row in predictions:
        grouped_rows[row.source_key].append(row)
    for source_key, rows in grouped_rows.items():
        rows = sorted(rows, key=model_sort_key)
        image_path = input_dir.parent / rows[0].raw_path
        # input_dir.parent is the project root because input-dir defaults to repo/local path.
        if not image_path.exists():
            image_path = Path.cwd().resolve() / rows[0].raw_path
        draw_master_image(
            image_path=image_path,
            source_key=source_key,
            actual_label=rows[0].actual_label,
            rows=rows,
            output_path=output_dir / "master_images" / f"{source_key.replace(':', '_')}__master.png",
            overlap_threshold=args.overlap_threshold,
            max_display_size=args.max_display_size,
        )

    build_verdict_report(
        predictions=predictions,
        metrics=metrics,
        output_path=output_dir / "final_verdict.md",
        visuals_dir=output_dir,
    )
    print(f"Wrote {output_dir / 'final_verdict.md'}")


if __name__ == "__main__":
    main()
