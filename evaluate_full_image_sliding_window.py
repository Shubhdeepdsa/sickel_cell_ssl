from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

import train_stage5 as stage5


DEFAULT_SETUPS = ("supervised_scratch", "imagenet_finetune", "simclr_finetune")
DEFAULT_WINDOW_SIZES = (128, 160, 224)
POSITIVE_LABEL = "positive"
NEGATIVE_LABEL = "negative"


@dataclass(frozen=True)
class SourceImageRecord:
    split: str
    source_key: str
    source_id: str
    label: str
    raw_path: Path


@dataclass(frozen=True)
class WindowRecord:
    window_id: str
    x: int
    y: int
    size: int


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    stage: str
    branch: str
    family: str
    label_fraction: str
    checkpoint_path: Path


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module: torch.nn.Module, inputs, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _backward_hook(self, module: torch.nn.Module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def remove(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __call__(self, input_tensor: torch.Tensor, class_index: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        logits[:, class_index].sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)
        cam = cam[0].detach().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def open_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def read_source_records(path: Path, project_root: Path, split: str) -> list[SourceImageRecord]:
    records: list[SourceImageRecord] = []
    for row in read_csv(path):
        if split != "all" and row["split"] != split:
            continue
        records.append(
            SourceImageRecord(
                split=row["split"],
                source_key=row["source_key"],
                source_id=row["source_id"],
                label=row["label"],
                raw_path=project_root / row["raw_path"],
            )
        )
    return records


def sample_source_records(
    records: list[SourceImageRecord],
    n_positive: int,
    n_negative: int,
    seed: int,
) -> list[SourceImageRecord]:
    rng = random.Random(seed)
    positive_records = [record for record in records if record.label == POSITIVE_LABEL]
    negative_records = [record for record in records if record.label == NEGATIVE_LABEL]
    if len(positive_records) < n_positive:
        raise ValueError(f"Requested {n_positive} positive images, but only found {len(positive_records)}")
    if len(negative_records) < n_negative:
        raise ValueError(f"Requested {n_negative} negative images, but only found {len(negative_records)}")
    rng.shuffle(positive_records)
    rng.shuffle(negative_records)
    selected = positive_records[:n_positive] + negative_records[:n_negative]
    selected.sort(key=lambda record: (record.label, record.source_key))
    return selected


def discover_models(stage5_root: Path, stage5_gray_root: Path, stage6_root: Path, stage6_gray_root: Path) -> list[ModelSpec]:
    specs: list[ModelSpec] = []

    def add_stage5(root: Path, branch: str) -> None:
        for family in DEFAULT_SETUPS:
            checkpoint_path = root / family / "best_model.pt"
            if checkpoint_path.exists():
                specs.append(
                    ModelSpec(
                        model_id=f"stage5__{branch}__full__{family}",
                        stage="stage5",
                        branch=branch,
                        family=family,
                        label_fraction="100",
                        checkpoint_path=checkpoint_path,
                    )
                )

    def add_stage6(root: Path, branch: str) -> None:
        for fraction_dir in sorted(path for path in root.glob("fraction_*") if path.is_dir()):
            fraction = fraction_dir.name.split("_", 1)[1]
            for family in DEFAULT_SETUPS:
                checkpoint_path = fraction_dir / family / "best_model.pt"
                if checkpoint_path.exists():
                    specs.append(
                        ModelSpec(
                            model_id=f"stage6__{branch}__fraction_{fraction}__{family}",
                            stage="stage6",
                            branch=branch,
                            family=family,
                            label_fraction=str(int(fraction)),
                            checkpoint_path=checkpoint_path,
                        )
                    )

    add_stage5(stage5_root, "rgb")
    add_stage5(stage5_gray_root, "grayscale")
    add_stage6(stage6_root, "rgb")
    add_stage6(stage6_gray_root, "grayscale")
    if not specs:
        raise FileNotFoundError("No model checkpoints were discovered under the supplied Stage 5/Stage 6 roots.")
    return specs


def clip_window_sizes(window_sizes: list[int], width: int, height: int) -> list[int]:
    shorter_side = min(width, height)
    clipped = [max(64, min(int(size), shorter_side)) for size in window_sizes]
    deduped = sorted(set(clipped))
    return deduped or [shorter_side]


def positions(limit: int, size: int, stride: int) -> list[int]:
    if size >= limit:
        return [0]
    max_start = limit - size
    values = list(range(0, max_start + 1, stride))
    if values[-1] != max_start:
        values.append(max_start)
    return values


def build_windows(width: int, height: int, window_sizes: list[int], stride_ratio: float) -> list[WindowRecord]:
    windows: list[WindowRecord] = []
    for size in clip_window_sizes(window_sizes, width, height):
        stride = max(1, int(round(size * stride_ratio)))
        for y in positions(height, size, stride):
            for x in positions(width, size, stride):
                windows.append(
                    WindowRecord(
                        window_id=f"s{size}_x{x}_y{y}",
                        x=x,
                        y=y,
                        size=size,
                    )
                )
    return windows


def load_model(spec: ModelSpec, device: torch.device) -> tuple[torch.nn.Module, int, bool]:
    checkpoint = stage5.load_torch_checkpoint(spec.checkpoint_path)
    args = checkpoint.get("args", {})
    image_size = int(args.get("image_size", 224))
    grayscale = bool(args.get("grayscale", False))
    model = stage5.build_resnet18_classifier(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, image_size, grayscale


def predict_window_probabilities(
    model: torch.nn.Module,
    base_image: Image.Image,
    windows: list[WindowRecord],
    image_size: int,
    grayscale: bool,
    batch_size: int,
    device: torch.device,
) -> list[float]:
    transform = stage5.eval_transform(image_size, grayscale=grayscale)
    probabilities: list[float] = []
    for start in range(0, len(windows), batch_size):
        batch_windows = windows[start : start + batch_size]
        tensors = []
        for window in batch_windows:
            crop = base_image.crop((window.x, window.y, window.x + window.size, window.y + window.size))
            tensors.append(transform(crop))
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = F.softmax(logits, dim=1)[:, stage5.LABEL_TO_ID["sickled"]]
        probabilities.extend(float(value) for value in probs.detach().cpu().tolist())
    return probabilities


def save_heatmap_overlay(base_image: np.ndarray, heatmap: np.ndarray, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(base_image)
    plt.imshow(heatmap, cmap="turbo", alpha=0.45, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def crop_for_display(base_image: Image.Image, window: WindowRecord, image_size: int, grayscale: bool) -> Image.Image:
    crop = base_image.crop((window.x, window.y, window.x + window.size, window.y + window.size)).resize((image_size, image_size))
    if grayscale:
        crop = crop.convert("L").convert("RGB")
    return crop


def render_gradcam_for_window(
    model: torch.nn.Module,
    base_image: Image.Image,
    window: WindowRecord,
    image_size: int,
    grayscale: bool,
    output_path: Path,
    title: str,
    device: torch.device,
) -> None:
    transform = stage5.eval_transform(image_size, grayscale=grayscale)
    crop = base_image.crop((window.x, window.y, window.x + window.size, window.y + window.size))
    input_tensor = transform(crop).unsqueeze(0).to(device)
    grad_cam = GradCAM(model, model.layer4)
    cam_map = grad_cam(input_tensor, stage5.LABEL_TO_ID["sickled"])
    grad_cam.remove()
    cam_resized = np.asarray(
        Image.fromarray((cam_map * 255).astype(np.uint8)).resize((image_size, image_size), Image.Resampling.BILINEAR)
    ) / 255.0
    base_display = np.asarray(crop_for_display(base_image, window, image_size, grayscale)) / 255.0
    save_heatmap_overlay(base_display, cam_resized, output_path, title)


def render_full_image_overlay(
    base_image: Image.Image,
    window: WindowRecord,
    output_path: Path,
    title_lines: list[str],
    color: str,
    max_display_size: int,
) -> None:
    display = base_image.copy()
    scale = min(1.0, max_display_size / max(display.width, display.height))
    if scale < 1.0:
        display = display.resize((int(round(display.width * scale)), int(round(display.height * scale))))
    draw = ImageDraw.Draw(display)
    x1 = int(round(window.x * scale))
    y1 = int(round(window.y * scale))
    x2 = int(round((window.x + window.size) * scale))
    y2 = int(round((window.y + window.size) * scale))
    line_width = max(2, int(round(4 * scale)))
    draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
    if title_lines:
        text = " | ".join(title_lines)
        draw.rectangle((0, 0, max(1, min(display.width, len(text) * 7 + 18)), 24), fill="black")
        draw.text((8, 6), text, fill="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display.save(output_path)


def compute_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    y_true = np.asarray([1 if row["actual_label"] == POSITIVE_LABEL else 0 for row in rows], dtype=np.int64)
    y_pred = np.asarray([1 if row["predicted_label"] == POSITIVE_LABEL else 0 for row in rows], dtype=np.int64)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    sensitivity = float(tp / max(tp + fn, 1))
    specificity = float(tn / max(tn + fp, 1))
    return {
        "n_images": int(len(rows)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def window_prediction_fieldnames() -> list[str]:
    return [
        "model_id",
        "stage",
        "branch",
        "family",
        "label_fraction",
        "source_key",
        "source_id",
        "split",
        "actual_label",
        "raw_path",
        "window_id",
        "window_x",
        "window_y",
        "window_size",
        "probability_sickled",
    ]


def write_markdown_report(
    output_path: Path,
    sampled_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    image_rows: list[dict[str, object]],
    threshold: float,
    split: str,
    window_sizes: list[int],
    stride_ratio: float,
) -> None:
    metric_rows = sorted(metric_rows, key=lambda row: (-float(row["balanced_accuracy"]), row["model_id"]))
    image_rows = sorted(image_rows, key=lambda row: (str(row["model_id"]), str(row["source_key"])))
    lines = [
        "# Full-Image Sliding-Window Evaluation",
        "",
        "## Protocol",
        f"- Source split used: `{split}`.",
        f"- Sampled images: `{len(sampled_rows)}` total.",
        f"- Positive images: `{sum(1 for row in sampled_rows if row['label'] == POSITIVE_LABEL)}`.",
        f"- Negative images: `{sum(1 for row in sampled_rows if row['label'] == NEGATIVE_LABEL)}`.",
        f"- Window sizes: `{', '.join(str(size) for size in window_sizes)}` pixels.",
        f"- Stride ratio: `{stride_ratio}`.",
        f"- Image-level decision rule: positive if any window has `P(sickled) >= {threshold}`.",
        "",
        "## Model Metrics",
        "",
        "| Model | Stage | Branch | Fraction | Accuracy | Balanced Acc | Sensitivity | Specificity | TN | FP | FN | TP |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            "| {model_id} | {stage} | {branch} | {label_fraction}% | {accuracy:.3f} | {balanced_accuracy:.3f} | "
            "{sensitivity:.3f} | {specificity:.3f} | {true_negative} | {false_positive} | {false_negative} | {true_positive} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Sampled Images",
            "",
            "| Source Key | Label | Split | Raw Path |",
            "|---|---|---|---|",
        ]
    )
    for row in sampled_rows:
        lines.append(f"| {row['source_key']} | {row['label']} | {row['split']} | {row['raw_path']} |")

    lines.extend(
        [
            "",
            "## Image-Level Predictions",
            "",
            "| Model | Source Key | Actual | Predicted | Max P(sickled) | Decisive Window | Full Image Overlay | Grad-CAM |",
            "|---|---|---|---|---:|---|---|---|",
        ]
    )
    for row in image_rows:
        overlay_link = f"[overlay]({row['overlay_path']})" if row.get("overlay_path") else ""
        gradcam_link = f"[gradcam]({row['gradcam_path']})" if row.get("gradcam_path") else ""
        decisive_window = (
            f"({row['decisive_x']}, {row['decisive_y']}, {row['decisive_size']})"
            if row.get("decisive_x") is not None
            else "n/a"
        )
        lines.append(
            f"| {row['model_id']} | {row['source_key']} | {row['actual_label']} | {row['predicted_label']} | "
            f"{float(row['max_probability_sickled']):.4f} | {decisive_window} | {overlay_link} | {gradcam_link} |"
        )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate full-image sickle detection using dense sliding-window crops.")
    parser.add_argument("--source-manifest", type=Path, default=Path("split_outputs/source_image_splits.csv"))
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--n-positive", type=int, default=10)
    parser.add_argument("--n-negative", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage5-root", type=Path, default=Path("stage5_outputs"))
    parser.add_argument("--stage5-gray-root", type=Path, default=Path("stage5_grayscale_outputs"))
    parser.add_argument("--stage6-root", type=Path, default=Path("stage6_outputs"))
    parser.add_argument("--stage6-gray-root", type=Path, default=Path("stage6_grayscale_outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("full_image_sliding_window_outputs"))
    parser.add_argument("--window-sizes", nargs="+", type=int, default=list(DEFAULT_WINDOW_SIZES))
    parser.add_argument("--stride-ratio", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-display-size", type=int, default=1400)
    parser.add_argument("--max-models", type=int, default=None, help="Optional cap for quick smoke tests.")
    args = parser.parse_args()

    if args.stride_ratio <= 0.0 or args.stride_ratio > 1.0:
        raise ValueError(f"--stride-ratio must be in (0, 1]. Received: {args.stride_ratio}")

    project_root = Path.cwd().resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stage5.set_seed(args.seed)
    device = stage5.select_device(args.device)

    source_records = read_source_records(args.source_manifest, project_root, args.split)
    sampled_records = sample_source_records(source_records, args.n_positive, args.n_negative, args.seed)
    model_specs = discover_models(
        args.stage5_root.resolve(),
        args.stage5_gray_root.resolve(),
        args.stage6_root.resolve(),
        args.stage6_gray_root.resolve(),
    )
    if args.max_models is not None:
        model_specs = model_specs[: max(args.max_models, 0)]
    if not model_specs:
        raise ValueError("No models selected for evaluation.")

    sampled_rows = [
        {
            "source_key": record.source_key,
            "source_id": record.source_id,
            "label": record.label,
            "split": record.split,
            "raw_path": str(record.raw_path.relative_to(project_root)),
        }
        for record in sampled_records
    ]
    model_registry_rows = [
        {
            "model_id": spec.model_id,
            "stage": spec.stage,
            "branch": spec.branch,
            "family": spec.family,
            "label_fraction": spec.label_fraction,
            "checkpoint_path": str(spec.checkpoint_path.relative_to(project_root)),
        }
        for spec in model_specs
    ]
    write_csv(output_dir / "sampled_source_images.csv", sampled_rows)
    write_csv(output_dir / "model_registry.csv", model_registry_rows)

    source_window_bank: dict[str, tuple[list[WindowRecord], int, int]] = {}
    for record in sampled_records:
        image = open_rgb_image(record.raw_path)
        windows = build_windows(image.width, image.height, list(args.window_sizes), args.stride_ratio)
        source_window_bank[record.source_key] = (windows, image.width, image.height)
        image.close()

    image_level_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    window_predictions_path = output_dir / "window_predictions.csv"
    with window_predictions_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=window_prediction_fieldnames())
        writer.writeheader()

        for spec_index, spec in enumerate(model_specs, start=1):
            print(f"[{spec_index}/{len(model_specs)}] Evaluating {spec.model_id}", flush=True)
            model, image_size, grayscale = load_model(spec, device)
            per_model_image_rows: list[dict[str, object]] = []

            for record in sampled_records:
                base_image = open_rgb_image(record.raw_path)
                windows, _, _ = source_window_bank[record.source_key]
                probabilities = predict_window_probabilities(
                    model,
                    base_image,
                    windows,
                    image_size=image_size,
                    grayscale=grayscale,
                    batch_size=args.batch_size,
                    device=device,
                )

                for window, probability in zip(windows, probabilities):
                    writer.writerow(
                        {
                            "model_id": spec.model_id,
                            "stage": spec.stage,
                            "branch": spec.branch,
                            "family": spec.family,
                            "label_fraction": spec.label_fraction,
                            "source_key": record.source_key,
                            "source_id": record.source_id,
                            "split": record.split,
                            "actual_label": record.label,
                            "raw_path": str(record.raw_path.relative_to(project_root)),
                            "window_id": window.window_id,
                            "window_x": window.x,
                            "window_y": window.y,
                            "window_size": window.size,
                            "probability_sickled": round(probability, 6),
                        }
                    )

                decisive_index = int(np.argmax(probabilities)) if probabilities else 0
                decisive_window = windows[decisive_index]
                max_probability = float(probabilities[decisive_index]) if probabilities else 0.0
                predicted_label = POSITIVE_LABEL if max_probability >= args.threshold else NEGATIVE_LABEL
                model_dir = output_dir / "visuals" / record.source_key.replace(":", "_")
                overlay_path = model_dir / f"{spec.model_id}__overlay.png"
                gradcam_path = model_dir / f"{spec.model_id}__gradcam.png"
                render_full_image_overlay(
                    base_image,
                    decisive_window,
                    overlay_path,
                    title_lines=[
                        spec.model_id,
                        f"actual={record.label}",
                        f"pred={predicted_label}",
                        f"p={max_probability:.3f}",
                    ],
                    color="lime" if predicted_label == record.label else "red",
                    max_display_size=args.max_display_size,
                )
                render_gradcam_for_window(
                    model,
                    base_image,
                    decisive_window,
                    image_size=image_size,
                    grayscale=grayscale,
                    output_path=gradcam_path,
                    title=f"{spec.model_id} Grad-CAM",
                    device=device,
                )
                base_image.close()

                row = {
                    "model_id": spec.model_id,
                    "stage": spec.stage,
                    "branch": spec.branch,
                    "family": spec.family,
                    "label_fraction": spec.label_fraction,
                    "source_key": record.source_key,
                    "source_id": record.source_id,
                    "split": record.split,
                    "actual_label": record.label,
                    "predicted_label": predicted_label,
                    "max_probability_sickled": round(max_probability, 6),
                    "decisive_window_id": decisive_window.window_id,
                    "decisive_x": decisive_window.x,
                    "decisive_y": decisive_window.y,
                    "decisive_size": decisive_window.size,
                    "n_windows": len(windows),
                    "raw_path": str(record.raw_path.relative_to(project_root)),
                    "overlay_path": str(overlay_path.resolve()),
                    "gradcam_path": str(gradcam_path.resolve()),
                }
                image_level_rows.append(row)
                per_model_image_rows.append(row)

            metrics = compute_metrics(per_model_image_rows)
            metric_rows.append(
                {
                    "model_id": spec.model_id,
                    "stage": spec.stage,
                    "branch": spec.branch,
                    "family": spec.family,
                    "label_fraction": spec.label_fraction,
                    **metrics,
                }
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(output_dir / "image_level_predictions.csv", image_level_rows)
    write_csv(output_dir / "image_level_metrics.csv", metric_rows)
    write_markdown_report(
        output_dir / "summary_report.md",
        sampled_rows=sampled_rows,
        metric_rows=metric_rows,
        image_rows=image_level_rows,
        threshold=args.threshold,
        split=args.split,
        window_sizes=list(args.window_sizes),
        stride_ratio=args.stride_ratio,
    )
    summary = {
        "split": args.split,
        "n_positive": args.n_positive,
        "n_negative": args.n_negative,
        "seed": args.seed,
        "window_sizes": list(args.window_sizes),
        "stride_ratio": args.stride_ratio,
        "threshold": args.threshold,
        "n_models": len(model_specs),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {output_dir / 'summary_report.md'}", flush=True)


if __name__ == "__main__":
    main()
