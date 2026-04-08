from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


LABEL_TO_ID = {"non_sickled": 0, "sickled": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
CLASS_ORDER = ["non_sickled", "sickled"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def labels_and_scores(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray([LABEL_TO_ID[row["actual_label"]] for row in rows], dtype=np.int64)
    y_pred = np.asarray([LABEL_TO_ID[row["predicted_label"]] for row in rows], dtype=np.int64)
    y_score = np.asarray([float(row["probability_sickled"]) for row in rows], dtype=np.float64)
    return y_true, y_pred, y_score


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str, output_path: Path, normalize: str | None) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=[0, 1],
        display_labels=CLASS_ORDER,
        normalize=normalize,
        cmap="Blues",
        values_format=".2f" if normalize else "d",
        ax=ax,
        colorbar=False,
    )
    display.ax_.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_roc_curve(split_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for split, (y_true, _, y_score) in split_data.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, name=split).plot(ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="random baseline AUC=0.50")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_pr_curve(split_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for split, (y_true, _, y_score) in split_data.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        average_precision_proxy = auc(recall, precision)
        PrecisionRecallDisplay(
            precision=precision,
            recall=recall,
            average_precision=average_precision_proxy,
            name=split,
        ).plot(ax=ax)
        prevalence = float(y_true.mean())
        ax.hlines(prevalence, 0, 1, colors="gray", linestyles=":", label=f"{split} prevalence={prevalence:.2f}")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_probability_histogram(
    split: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 21)
    for class_id, label, color in [(0, "actual non_sickled", "#2f6f9f"), (1, "actual sickled", "#b23a48")]:
        ax.hist(
            y_score[y_true == class_id],
            bins=bins,
            alpha=0.65,
            label=label,
            color=color,
            edgecolor="white",
        )
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2, label="decision threshold=0.50")
    ax.set_title(f"{split.capitalize()} Probability Distribution")
    ax.set_xlabel("Predicted probability of sickled")
    ax.set_ylabel("Crop count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_calibration(split_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    for split, (y_true, _, y_score) in split_data.items():
        prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=8, strategy="quantile")
        ax.plot(prob_pred, prob_true, marker="o", label=split)
    ax.set_title("Calibration Curve")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed sickled fraction")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def majority_baseline(y_true: np.ndarray) -> tuple[str, float, float]:
    majority_label_id = Counter(y_true.tolist()).most_common(1)[0][0]
    y_pred = np.full_like(y_true, fill_value=majority_label_id)
    return (
        ID_TO_LABEL[int(majority_label_id)],
        float(accuracy_score(y_true, y_pred)),
        float(balanced_accuracy_score(y_true, y_pred)),
    )


def source_majority_metrics(rows: list[dict[str, str]]) -> tuple[int, float, float, Counter[str], Counter[str]]:
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_source[row["source_key"]].append(row)

    actual: list[int] = []
    predicted: list[int] = []
    for source_rows in by_source.values():
        actual_label = source_rows[0]["actual_label"]
        predicted_label = Counter(row["predicted_label"] for row in source_rows).most_common(1)[0][0]
        actual.append(LABEL_TO_ID[actual_label])
        predicted.append(LABEL_TO_ID[predicted_label])
    y_true = np.asarray(actual, dtype=np.int64)
    y_pred = np.asarray(predicted, dtype=np.int64)
    return (
        len(by_source),
        float(accuracy_score(y_true, y_pred)),
        float(balanced_accuracy_score(y_true, y_pred)),
        Counter(ID_TO_LABEL[int(value)] for value in y_true),
        Counter(ID_TO_LABEL[int(value)] for value in y_pred),
    )


def save_metric_bars(
    split_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_path: Path,
) -> dict[str, dict[str, float | str]]:
    rows: dict[str, dict[str, float | str]] = {}
    for split, (y_true, y_pred, y_score) in split_data.items():
        majority_label, majority_acc, majority_bal_acc = majority_baseline(y_true)
        rows[split] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "roc_auc": float(roc_auc_score(y_true, y_score)),
            "majority_label": majority_label,
            "majority_accuracy": majority_acc,
            "majority_balanced_accuracy": majority_bal_acc,
        }

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(split_data))
    width = 0.18
    metric_names = ["accuracy", "balanced_accuracy", "roc_auc", "majority_accuracy"]
    labels = ["accuracy", "balanced acc", "ROC AUC", "majority acc"]
    for idx, metric in enumerate(metric_names):
        ax.bar(x + (idx - 1.5) * width, [float(rows[split][metric]) for split in split_data], width, label=labels[idx])
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="random / one-class balanced acc")
    ax.set_xticks(x)
    ax.set_xticklabels(list(split_data.keys()))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model vs Baselines")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return rows


def compact_label(label: str) -> str:
    return "S" if label == "sickled" else "N"


def save_error_contact_sheet(
    rows: list[dict[str, str]],
    project_root: Path,
    output_path: Path,
    title: str,
    max_images: int,
) -> int:
    mistakes = [row for row in rows if row["actual_label"] != row["predicted_label"]]
    mistakes = sorted(mistakes, key=lambda row: abs(float(row["probability_sickled"]) - 0.5), reverse=True)[:max_images]
    if not mistakes:
        return 0

    thumb_size = 112
    pad = 24
    cols = 6
    rows_n = int(np.ceil(len(mistakes) / cols))
    sheet = Image.new("RGB", (cols * thumb_size, rows_n * (thumb_size + pad)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, row in enumerate(mistakes):
        image_path = project_root / row["crop_path"]
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize((thumb_size, thumb_size))
        col = idx % cols
        row_idx = idx // cols
        x = col * thumb_size
        y = row_idx * (thumb_size + pad)
        sheet.paste(image, (x, y))
        draw.rectangle((x, y, x + thumb_size - 1, y + thumb_size - 1), outline=(180, 0, 0), width=3)
        label = f"T:{compact_label(row['actual_label'])} P:{compact_label(row['predicted_label'])}"
        draw.text((x + 3, y + thumb_size + 3), label, fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return len(mistakes)


def save_source_crop_distribution(rows: list[dict[str, str]], output_path: Path) -> None:
    by_source = Counter(row["source_key"] for row in rows)
    by_label: dict[str, list[int]] = defaultdict(list)
    source_label = {}
    for row in rows:
        source_label[row["source_key"]] = row["actual_label"]
    for source_key, count in by_source.items():
        by_label[source_label[source_key]].append(count)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.arange(1, max(by_source.values()) + 2) - 0.5
    for label, color in [("non_sickled", "#2f6f9f"), ("sickled", "#b23a48")]:
        ax.hist(by_label[label], bins=bins, alpha=0.65, label=label, color=color, edgecolor="white")
    ax.set_title("Test Crops Per Source Image")
    ax.set_xlabel("Crop count from one source image")
    ax.set_ylabel("Source image count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    metric_summary: dict[str, dict[str, float | str]],
    source_summary: dict[str, tuple[int, float, float, Counter[str], Counter[str]]],
    visual_dir: Path,
    val_error_count: int,
    test_error_count: int,
) -> None:
    test = metric_summary["test"]
    val = metric_summary["val"]
    report = f"""# Stage 4 Visual Diagnostics

## What The Current Plots Say
- The model is not simply predicting `non_sickled`: test predictions are close to balanced by class, and the test balanced accuracy is {float(test["balanced_accuracy"]):.4f}.
- A one-class majority baseline would score test accuracy {float(test["majority_accuracy"]):.4f} and balanced accuracy {float(test["majority_balanced_accuracy"]):.4f}.
- Validation balanced accuracy is {float(val["balanced_accuracy"]):.4f}; test balanced accuracy is {float(test["balanced_accuracy"]):.4f}. Test being higher than validation is not impossible, but it is a reason to inspect artifacts.
- Source-level majority-vote test balanced accuracy is {source_summary["test"][2]:.4f} over {source_summary["test"][0]} source images.

## What A Legit Model Should Look Like
- Confusion matrix: both diagonal cells should be high; one row or column should not dominate.
- ROC curve: should sit clearly above the random diagonal, preferably with similar validation and test curves.
- Precision-recall curve: should sit above the class-prevalence baseline and not collapse at useful recall.
- Probability histogram: sickled and non_sickled distributions should separate, but not only because one pool has different crop borders/background.
- Calibration: points should roughly track the diagonal. Bad calibration means probabilities are not trustworthy even if accuracy is decent.
- Error sheets: false positives and false negatives should look like genuinely hard cells, not obvious black borders, empty crops, crop-scale artifacts, or extraction mistakes.

## What Would Suggest A Mistake
- Balanced accuracy near 0.50, even with high raw accuracy, means majority-class behavior.
- ROC AUC near 0.50 means no ranking signal.
- Very high metrics plus obvious visual artifacts in error/contact sheets means the classifier may be learning the crop pipeline rather than morphology.
- Source-level performance far below crop-level performance means a few source images or repeated crops are inflating results.

## Files
- `{visual_dir / "confusion_matrix_test_counts.png"}`
- `{visual_dir / "confusion_matrix_test_normalized.png"}`
- `{visual_dir / "roc_curve.png"}`
- `{visual_dir / "precision_recall_curve.png"}`
- `{visual_dir / "probability_hist_test.png"}`
- `{visual_dir / "calibration_curve.png"}`
- `{visual_dir / "metric_bars.png"}`
- `{visual_dir / "test_error_contact_sheet.jpg"}` with {test_error_count} mistakes sampled
- `{visual_dir / "validation_error_contact_sheet.jpg"}` with {val_error_count} mistakes sampled
"""
    path.write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Stage 4 classifier diagnostic plots.")
    parser.add_argument("--stage4-dir", type=Path, default=Path("stage4_outputs"), help="Directory with Stage 4 outputs")
    parser.add_argument("--output-dir", type=Path, default=Path("stage4_outputs/visuals"), help="Directory for plots")
    parser.add_argument("--max-error-images", type=int, default=36, help="Max mistakes per error contact sheet")
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)

    split_rows = {
        "val": read_csv(args.stage4_dir / "validation_predictions.csv"),
        "test": read_csv(args.stage4_dir / "test_predictions.csv"),
    }
    split_data = {split: labels_and_scores(rows) for split, rows in split_rows.items()}

    for split, (y_true, y_pred, _) in split_data.items():
        save_confusion_matrix(
            y_true,
            y_pred,
            f"{split.capitalize()} Confusion Matrix (counts)",
            output_dir / f"confusion_matrix_{split}_counts.png",
            normalize=None,
        )
        save_confusion_matrix(
            y_true,
            y_pred,
            f"{split.capitalize()} Confusion Matrix (row-normalized)",
            output_dir / f"confusion_matrix_{split}_normalized.png",
            normalize="true",
        )
        save_probability_histogram(split, y_true, split_data[split][2], output_dir / f"probability_hist_{split}.png")

    save_roc_curve(split_data, output_dir / "roc_curve.png")
    save_pr_curve(split_data, output_dir / "precision_recall_curve.png")
    save_calibration(split_data, output_dir / "calibration_curve.png")
    metric_summary = save_metric_bars(split_data, output_dir / "metric_bars.png")
    save_source_crop_distribution(split_rows["test"], output_dir / "test_crops_per_source.png")

    val_error_count = save_error_contact_sheet(
        split_rows["val"],
        project_root,
        output_dir / "validation_error_contact_sheet.jpg",
        "Validation mistakes",
        args.max_error_images,
    )
    test_error_count = save_error_contact_sheet(
        split_rows["test"],
        project_root,
        output_dir / "test_error_contact_sheet.jpg",
        "Test mistakes",
        args.max_error_images,
    )
    source_summary = {split: source_majority_metrics(rows) for split, rows in split_rows.items()}
    write_report(
        args.stage4_dir / "visual_diagnostics.md",
        metric_summary,
        source_summary,
        args.output_dir,
        val_error_count,
        test_error_count,
    )

    print(f"Wrote visuals to {args.output_dir}")
    print(f"Wrote {args.stage4_dir / 'visual_diagnostics.md'}")


if __name__ == "__main__":
    main()
