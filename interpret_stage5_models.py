from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

import evaluate_stage7_robustness as stage7
import train_stage5 as stage5


DEFAULT_SETUPS = ("supervised_scratch", "imagenet_finetune", "simclr_finetune")
DIAGNOSTIC_PERTURBATIONS = (
    "clean",
    "forced_grayscale",
    "blurred",
    "lower_brightness",
    "mild_compression",
    "mild_stain_color",
)


@dataclass(frozen=True)
class PredictionRecord:
    crop_path: str
    actual_label: str
    predicted_label: str
    probability_sickled: float


@dataclass
class LoadedSetup:
    name: str
    model: nn.Module
    image_size: int
    grayscale: bool
    prediction_lookup: dict[str, PredictionRecord]
    train_features: torch.Tensor | None = None
    train_records: list[stage5.CropRecord] | None = None


class FeatureDataset(Dataset):
    def __init__(self, records: list[stage5.CropRecord], transform) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        with Image.open(record.crop_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, index


def read_prediction_csv(path: Path) -> dict[str, PredictionRecord]:
    lookup: dict[str, PredictionRecord] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            lookup[row["crop_path"]] = PredictionRecord(
                crop_path=row["crop_path"],
                actual_label=row["actual_label"],
                predicted_label=row["predicted_label"],
                probability_sickled=float(row["probability_sickled"]),
            )
    return lookup


def extract_backbone_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    x = model.conv1(images)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return x


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module: nn.Module, inputs, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _backward_hook(self, module: nn.Module, grad_input, grad_output) -> None:
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


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(stage5.IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(stage5.IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    image = tensor.detach().cpu() * std + mean
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return image


def resize_pil_for_display(path: Path, image_size: int, grayscale: bool) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        if grayscale:
            image = image.convert("L").convert("RGB")
        return image


def save_heatmap_overlay(
    base_image: np.ndarray,
    heatmap: np.ndarray,
    output_path: Path,
    title: str,
    cmap: str = "turbo",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(base_image)
    plt.imshow(heatmap, cmap=cmap, alpha=0.45, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def save_neighbor_grid(
    query_image: Image.Image,
    query_title: str,
    neighbor_images: list[Image.Image],
    neighbor_titles: list[str],
    output_path: Path,
) -> None:
    n_cols = 1 + len(neighbor_images)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.6))
    if n_cols == 1:
        axes = [axes]
    axes[0].imshow(query_image)
    axes[0].axis("off")
    axes[0].set_title(query_title, fontsize=9)
    for axis, image, title in zip(axes[1:], neighbor_images, neighbor_titles):
        axis.imshow(image)
        axis.axis("off")
        axis.set_title(title, fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_perturbation_grid(
    perturbation_images: list[Image.Image],
    perturbation_titles: list[str],
    output_path: Path,
) -> None:
    n_cols = len(perturbation_images)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.0 * n_cols, 3.6))
    if n_cols == 1:
        axes = [axes]
    for axis, image, title in zip(axes, perturbation_images, perturbation_titles):
        axis.imshow(image)
        axis.axis("off")
        axis.set_title(title, fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_sensitivity_plot(
    probabilities_by_setup: dict[str, dict[str, float]],
    output_path: Path,
    title: str,
) -> None:
    x = np.arange(len(DIAGNOSTIC_PERTURBATIONS))
    plt.figure(figsize=(9, 4.8))
    for setup_name, values in probabilities_by_setup.items():
        y = [values[name] for name in DIAGNOSTIC_PERTURBATIONS]
        plt.plot(x, y, marker="o", linewidth=2, label=setup_name)
    plt.xticks(x, DIAGNOSTIC_PERTURBATIONS, rotation=20)
    plt.ylim(-0.02, 1.02)
    plt.ylabel("Predicted P[sickled]")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def build_model_for_setup(setup_name: str) -> nn.Module:
    # Load architecture only. Checkpoints already contain the full fine-tuned weights.
    return stage5.build_resnet18_classifier(pretrained=False)


def load_setup(model_root: Path, setup_name: str, device: torch.device) -> LoadedSetup:
    setup_dir = model_root / setup_name
    checkpoint = stage5.load_torch_checkpoint(setup_dir / "best_model.pt")
    args = checkpoint.get("args", {})
    grayscale = bool(args.get("grayscale", False))
    image_size = int(args.get("image_size", 224))
    model = build_model_for_setup(setup_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    prediction_lookup = read_prediction_csv(setup_dir / "test_predictions.csv")
    return LoadedSetup(
        name=setup_name,
        model=model,
        image_size=image_size,
        grayscale=grayscale,
        prediction_lookup=prediction_lookup,
    )


def predict_probabilities(model: nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits = model(input_tensor)
        return F.softmax(logits, dim=1)[0]


def forced_grayscale(image: Image.Image, _: argparse.Namespace) -> Image.Image:
    return image.convert("L").convert("RGB")


def diagnostic_perturbations() -> dict[str, Callable[[Image.Image, argparse.Namespace], Image.Image]]:
    return {
        "clean": stage7.clean,
        "forced_grayscale": forced_grayscale,
        "blurred": stage7.blurred,
        "lower_brightness": stage7.lower_brightness,
        "mild_compression": stage7.mild_compression,
        "mild_stain_color": stage7.mild_stain_color,
    }


def occlusion_map(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_index: int,
    patch_size: int,
    stride: int,
) -> np.ndarray:
    with torch.no_grad():
        baseline = F.softmax(model(input_tensor), dim=1)[0, class_index].item()
    _, _, height, width = input_tensor.shape
    importance = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    for top in range(0, height, stride):
        for left in range(0, width, stride):
            bottom = min(top + patch_size, height)
            right = min(left + patch_size, width)
            occluded = input_tensor.clone()
            occluded[:, :, top:bottom, left:right] = 0.0
            with torch.no_grad():
                prob = F.softmax(model(occluded), dim=1)[0, class_index].item()
            drop = baseline - prob
            importance[top:bottom, left:right] += drop
            counts[top:bottom, left:right] += 1.0
    counts[counts == 0] = 1.0
    importance = importance / counts
    importance = importance - importance.min()
    if importance.max() > 0:
        importance = importance / importance.max()
    return importance


def compute_feature_bank(
    setup: LoadedSetup,
    records: list[stage5.CropRecord],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> None:
    dataset = FeatureDataset(records, stage5.eval_transform(setup.image_size, grayscale=setup.grayscale))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    features: list[torch.Tensor] = []
    ordered_records: list[stage5.CropRecord] = [None] * len(records)  # type: ignore[list-item]
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device)
            batch_features = extract_backbone_features(setup.model, images)
            batch_features = F.normalize(batch_features, dim=1).cpu()
            features.append(batch_features)
            for local_position, record_index in enumerate(indices.tolist()):
                ordered_records[record_index] = records[record_index]
    setup.train_features = torch.cat(features, dim=0)
    setup.train_records = ordered_records


def nearest_neighbors(
    setup: LoadedSetup,
    query_tensor: torch.Tensor,
    top_k: int,
) -> list[tuple[stage5.CropRecord, float]]:
    if setup.train_features is None or setup.train_records is None:
        raise RuntimeError(f"Feature bank has not been computed for {setup.name}")
    with torch.no_grad():
        query_feature = extract_backbone_features(setup.model, query_tensor)
        query_feature = F.normalize(query_feature, dim=1).cpu()
    similarities = torch.matmul(setup.train_features, query_feature[0])
    top_values, top_indices = torch.topk(similarities, k=min(top_k, len(setup.train_records)))
    return [
        (setup.train_records[index], float(value))
        for value, index in zip(top_values.tolist(), top_indices.tolist())
    ]


def auto_select_samples(
    model_root: Path,
    setup_name: str,
    max_samples: int,
) -> list[str]:
    prediction_path = model_root / setup_name / "test_predictions.csv"
    with prediction_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    def confidence(row: dict[str, str]) -> float:
        prob = float(row["probability_sickled"])
        return prob if row["predicted_label"] == "sickled" else (1.0 - prob)

    selected: list[str] = []
    for label in ("sickled", "non_sickled"):
        candidates = [
            row
            for row in rows
            if row["actual_label"] == label and row["predicted_label"] == label
        ]
        candidates.sort(key=confidence, reverse=True)
        if candidates:
            selected.append(candidates[0]["crop_path"])

    correct_rows = [row for row in rows if row["actual_label"] == row["predicted_label"]]
    correct_rows.sort(key=lambda row: abs(float(row["probability_sickled"]) - 0.5))
    if correct_rows:
        selected.append(correct_rows[0]["crop_path"])

    error_rows = [row for row in rows if row["actual_label"] != row["predicted_label"]]
    error_rows.sort(key=confidence, reverse=True)
    if error_rows:
        selected.append(error_rows[0]["crop_path"])

    if len(selected) < max_samples:
        rows.sort(key=confidence, reverse=True)
        for row in rows:
            selected.append(row["crop_path"])
            if len(selected) >= max_samples:
                break

    deduped: list[str] = []
    seen: set[str] = set()
    for path in selected:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
        if len(deduped) >= max_samples:
            break
    return deduped


def sample_summary_line(setup: LoadedSetup, crop_path: str) -> str:
    record = setup.prediction_lookup.get(crop_path)
    if record is None:
        return f"- `{setup.name}`: prediction not found in `test_predictions.csv`."
    return (
        f"- `{setup.name}`: predicted `{record.predicted_label}` "
        f"(P[sickled]={record.probability_sickled:.4f}), actual `{record.actual_label}`."
    )


def infer_class_index(prediction: PredictionRecord) -> int:
    return stage5.LABEL_TO_ID[prediction.predicted_label]


def render_interpretations_for_sample(
    sample_record: stage5.CropRecord,
    setups: list[LoadedSetup],
    output_dir: Path,
    device: torch.device,
    occlusion_patch: int,
    occlusion_stride: int,
    top_k_neighbors: int,
    perturbation_args: argparse.Namespace,
) -> list[str]:
    relative_lines = [
        f"### Sample `{sample_record.crop_path.name}`",
        "",
        f"- Source: `{sample_record.source_key}`",
        f"- Actual label: `{sample_record.label}`",
        f"- Crop path: `{sample_record.crop_path}`",
    ]
    sample_dir = output_dir / "samples" / sample_record.crop_path.stem
    sample_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_probabilities: dict[str, dict[str, float]] = {}

    for setup in setups:
        transform = stage5.eval_transform(setup.image_size, grayscale=setup.grayscale)
        with Image.open(sample_record.crop_path) as image:
            image = image.convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)

        prediction = setup.prediction_lookup[str(sample_record.crop_path)]
        class_index = infer_class_index(prediction)
        class_name = stage5.ID_TO_LABEL[class_index]

        grad_cam = GradCAM(setup.model, setup.model.layer4)
        cam_map = grad_cam(input_tensor, class_index)
        grad_cam.remove()
        cam_resized = np.asarray(
            Image.fromarray((cam_map * 255).astype(np.uint8)).resize(
                (setup.image_size, setup.image_size), Image.Resampling.BILINEAR
            )
        ) / 255.0

        occlusion = occlusion_map(
            setup.model,
            input_tensor,
            class_index=class_index,
            patch_size=occlusion_patch,
            stride=occlusion_stride,
        )

        base_image = np.asarray(resize_pil_for_display(sample_record.crop_path, setup.image_size, setup.grayscale)) / 255.0
        gradcam_path = sample_dir / f"{setup.name}_gradcam.png"
        occlusion_path = sample_dir / f"{setup.name}_occlusion.png"
        save_heatmap_overlay(
            base_image,
            cam_resized,
            gradcam_path,
            title=f"{setup.name} Grad-CAM ({class_name})",
        )
        save_heatmap_overlay(
            base_image,
            occlusion,
            occlusion_path,
            title=f"{setup.name} occlusion ({class_name})",
        )

        neighbors = nearest_neighbors(setup, input_tensor, top_k=top_k_neighbors)
        query_title = (
            f"Query\nactual={sample_record.label}\n"
            f"pred={prediction.predicted_label}\n"
            f"P[sickled]={prediction.probability_sickled:.3f}"
        )
        neighbor_images: list[Image.Image] = []
        neighbor_titles: list[str] = []
        for neighbor_record, similarity in neighbors:
            neighbor_images.append(
                resize_pil_for_display(neighbor_record.crop_path, setup.image_size, setup.grayscale)
            )
            neighbor_titles.append(
                f"{neighbor_record.label}\n{neighbor_record.source_key}\ncos={similarity:.3f}"
            )
        neighbor_path = sample_dir / f"{setup.name}_neighbors.png"
        save_neighbor_grid(
            resize_pil_for_display(sample_record.crop_path, setup.image_size, setup.grayscale),
            query_title=query_title,
            neighbor_images=neighbor_images,
            neighbor_titles=neighbor_titles,
            output_path=neighbor_path,
        )

        probabilities_for_setup: dict[str, float] = {}
        perturbation_images: list[Image.Image] = []
        perturbation_titles: list[str] = []
        with Image.open(sample_record.crop_path) as raw_image:
            base_rgb = raw_image.convert("RGB")
            for perturbation_name, perturbation_fn in diagnostic_perturbations().items():
                perturbed_image = perturbation_fn(base_rgb.copy(), perturbation_args)
                perturbed_tensor = transform(perturbed_image).unsqueeze(0).to(device)
                probs = predict_probabilities(setup.model, perturbed_tensor)
                probabilities_for_setup[perturbation_name] = float(probs[stage5.LABEL_TO_ID["sickled"]].item())
                display_image = perturbed_image.resize((setup.image_size, setup.image_size)).convert("RGB")
                if setup.grayscale:
                    display_image = display_image.convert("L").convert("RGB")
                perturbation_images.append(
                    display_image
                )
                perturbation_titles.append(
                    f"{perturbation_name}\nP[sickled]={probabilities_for_setup[perturbation_name]:.3f}"
                )

        diagnostic_probabilities[setup.name] = probabilities_for_setup
        perturbation_grid_path = sample_dir / f"{setup.name}_perturbations.png"
        save_perturbation_grid(
            perturbation_images,
            perturbation_titles,
            perturbation_grid_path,
        )

        relative_lines.extend(
            [
                f"- `{setup.name}` prediction: `{prediction.predicted_label}` "
                f"with `P[sickled]={prediction.probability_sickled:.4f}`.",
                f"- `{setup.name}` Grad-CAM: [{gradcam_path.name}]({gradcam_path.resolve()})",
                f"- `{setup.name}` occlusion: [{occlusion_path.name}]({occlusion_path.resolve()})",
                f"- `{setup.name}` nearest neighbors: [{neighbor_path.name}]({neighbor_path.resolve()})",
                f"- `{setup.name}` perturbation grid: [{perturbation_grid_path.name}]({perturbation_grid_path.resolve()})",
            ]
        )
    sensitivity_path = sample_dir / "shortcut_sensitivity.png"
    save_sensitivity_plot(
        diagnostic_probabilities,
        sensitivity_path,
        title=f"Shortcut diagnostic: {sample_record.crop_path.name}",
    )
    relative_lines.append(f"- Sample-level shortcut sensitivity: [{sensitivity_path.name}]({sensitivity_path.resolve()})")
    relative_lines.append("")
    return relative_lines


def build_report(
    output_dir: Path,
    model_root: Path,
    selected_samples: list[str],
    setups: list[LoadedSetup],
    sample_records: list[stage5.CropRecord],
) -> None:
    lines = [
        "# Stage 5 Interpretability Report",
        "",
        f"- Model root: `{model_root}`",
        f"- Setups: `{', '.join(setup.name for setup in setups)}`",
        f"- Samples analyzed: `{len(selected_samples)}`",
        "",
        "## Prediction Summary",
    ]
    for crop_path, sample_record in zip(selected_samples, sample_records):
        lines.extend(
            [
                f"### `{sample_record.crop_path.name}`",
                f"- Actual label: `{sample_record.label}`",
                f"- Source: `{sample_record.source_key}`",
            ]
        )
        for setup in setups:
            lines.append(sample_summary_line(setup, crop_path))
        lines.append("")

    lines.extend(
        [
            "## Visual Outputs",
            "",
            "Each sample directory contains three artifacts per model:",
            "- `*_gradcam.png`: Grad-CAM heatmap over the predicted class.",
            "- `*_occlusion.png`: probability-drop map when local regions are masked.",
            "- `*_neighbors.png`: nearest training crops in the learned feature space.",
            "",
        ]
    )
    for sample_record in sample_records:
        lines.extend(render_stub(sample_record, output_dir, setups))
    report_path = output_dir / "interpretability_report.md"
    report_path.write_text("\n".join(lines) + "\n")


def render_stub(
    sample_record: stage5.CropRecord,
    output_dir: Path,
    setups: list[LoadedSetup],
) -> list[str]:
    sample_dir = output_dir / "samples" / sample_record.crop_path.stem
    lines = [
        f"### Sample `{sample_record.crop_path.name}`",
        "",
    ]
    for setup in setups:
        gradcam_path = sample_dir / f"{setup.name}_gradcam.png"
        occlusion_path = sample_dir / f"{setup.name}_occlusion.png"
        neighbor_path = sample_dir / f"{setup.name}_neighbors.png"
        perturbation_grid_path = sample_dir / f"{setup.name}_perturbations.png"
        lines.extend(
            [
                f"- `{setup.name}` Grad-CAM: [{gradcam_path.name}]({gradcam_path.resolve()})",
                f"- `{setup.name}` occlusion: [{occlusion_path.name}]({occlusion_path.resolve()})",
                f"- `{setup.name}` neighbors: [{neighbor_path.name}]({neighbor_path.resolve()})",
                f"- `{setup.name}` perturbation grid: [{perturbation_grid_path.name}]({perturbation_grid_path.resolve()})",
            ]
        )
    sensitivity_path = sample_dir / "shortcut_sensitivity.png"
    lines.append(f"- Sample-level shortcut sensitivity: [{sensitivity_path.name}]({sensitivity_path.resolve()})")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM, occlusion, and nearest-neighbor views for Stage 5 models.")
    parser.add_argument("--pool-a-manifest", type=Path, default=Path("crop_outputs/pool_a_labeled_sickled_manifest.csv"))
    parser.add_argument("--pool-b-manifest", type=Path, default=Path("crop_outputs/pool_b_labeled_non_sickled_manifest.csv"))
    parser.add_argument("--model-root", type=Path, default=Path("stage5_outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("interpretability_outputs/stage5"))
    parser.add_argument("--setups", nargs="+", default=list(DEFAULT_SETUPS), choices=list(DEFAULT_SETUPS))
    parser.add_argument("--sample-path", nargs="*", default=None, help="Specific crop paths to analyze.")
    parser.add_argument("--reference-setup", default="simclr_finetune", choices=list(DEFAULT_SETUPS))
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--top-k-neighbors", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--occlusion-patch", type=int, default=32)
    parser.add_argument("--occlusion-stride", type=int, default=32)
    parser.add_argument("--blur-radius", type=float, default=1.25)
    parser.add_argument("--brightness-factor", type=float, default=0.75)
    parser.add_argument("--jpeg-quality", type=int, default=45)
    parser.add_argument("--red-scale", type=float, default=1.08)
    parser.add_argument("--green-scale", type=float, default=0.97)
    parser.add_argument("--blue-scale", type=float, default=1.06)
    parser.add_argument("--color-factor", type=float, default=1.08)
    parser.add_argument("--contrast-factor", type=float, default=1.03)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model_root = args.model_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = stage5.select_device(args.device)
    project_root = Path.cwd().resolve()
    labeled_records = stage5.load_labeled_records(args.pool_a_manifest, args.pool_b_manifest, project_root)
    stage5.validate_records(labeled_records)
    lookup_by_crop_path = {str(record.crop_path): record for record in labeled_records}

    if args.sample_path:
        selected_samples = [str((project_root / path).resolve()) if not Path(path).is_absolute() else str(Path(path).resolve()) for path in args.sample_path]
    else:
        selected_samples = auto_select_samples(model_root, args.reference_setup, args.max_samples)
        selected_samples = [str((project_root / path).resolve()) for path in selected_samples]

    sample_records: list[stage5.CropRecord] = []
    for crop_path in selected_samples:
        record = lookup_by_crop_path.get(crop_path)
        if record is None:
            raise KeyError(f"Selected crop path is not present in the labeled manifests: {crop_path}")
        sample_records.append(record)

    setups = [load_setup(model_root, setup_name, device) for setup_name in args.setups]
    train_records = [record for record in labeled_records if record.split == "train"]
    for setup in setups:
        compute_feature_bank(
            setup,
            train_records,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    sample_sections: list[str] = []
    for sample_record in sample_records:
        sample_sections.extend(
            render_interpretations_for_sample(
                sample_record,
                setups,
                output_dir,
                device=device,
                occlusion_patch=args.occlusion_patch,
                occlusion_stride=args.occlusion_stride,
                top_k_neighbors=args.top_k_neighbors,
                perturbation_args=args,
            )
        )

    report_lines = [
        "# Stage 5 Interpretability Report",
        "",
        f"- Model root: `{model_root}`",
        f"- Output directory: `{output_dir}`",
        f"- Device: `{device}`",
        f"- Setups: `{', '.join(setup.name for setup in setups)}`",
        f"- Samples analyzed: `{len(sample_records)}`",
        "",
        "## Prediction Summary",
    ]
    for crop_path, sample_record in zip(selected_samples, sample_records):
        report_lines.extend(
            [
                f"### `{sample_record.crop_path.name}`",
                f"- Actual label: `{sample_record.label}`",
                f"- Source: `{sample_record.source_key}`",
            ]
        )
        for setup in setups:
            report_lines.append(sample_summary_line(setup, crop_path))
        report_lines.append("")
    report_lines.extend(
        [
            "## Visual Outputs",
            "",
            "For each sample and setup, the script writes:",
            "- `Grad-CAM`: which spatial regions most support the predicted class.",
            "- `Occlusion`: where masking causes the largest drop in predicted-class probability.",
            "- `Nearest neighbors`: closest training crops in each model's penultimate feature space.",
            "- `Perturbation grid`: the same crop under grayscale/color/quality perturbations with updated probabilities.",
            "- `Shortcut sensitivity`: sample-level probability curves across nuisance perturbations for all setups.",
            "",
        ]
    )
    report_lines.extend(sample_sections)
    (output_dir / "interpretability_report.md").write_text("\n".join(report_lines) + "\n")

    manifest = {
        "model_root": str(model_root),
        "output_dir": str(output_dir),
        "device": str(device),
        "setups": [setup.name for setup in setups],
        "selected_samples": selected_samples,
        "occlusion_patch": args.occlusion_patch,
        "occlusion_stride": args.occlusion_stride,
        "top_k_neighbors": args.top_k_neighbors,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {output_dir / 'interpretability_report.md'}")


if __name__ == "__main__":
    main()
