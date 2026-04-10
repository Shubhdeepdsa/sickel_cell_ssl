from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

import train_stage5 as stage5


PERTURBATION_ORDER = [
    "clean",
    "blurred",
    "lower_brightness",
    "mild_compression",
    "mild_stain_color",
]


def clean(image: Image.Image, _: argparse.Namespace) -> Image.Image:
    return image


def blurred(image: Image.Image, args: argparse.Namespace) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=args.blur_radius))


def lower_brightness(image: Image.Image, args: argparse.Namespace) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(args.brightness_factor)


def mild_compression(image: Image.Image, args: argparse.Namespace) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=args.jpeg_quality, optimize=True)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB")


def mild_stain_color(image: Image.Image, args: argparse.Namespace) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    array[..., 0] *= args.red_scale
    array[..., 1] *= args.green_scale
    array[..., 2] *= args.blue_scale
    array = np.clip(array, 0, 255).astype(np.uint8)
    adjusted = Image.fromarray(array, mode="RGB")
    adjusted = ImageEnhance.Color(adjusted).enhance(args.color_factor)
    adjusted = ImageEnhance.Contrast(adjusted).enhance(args.contrast_factor)
    return adjusted


PERTURBATIONS = {
    "clean": clean,
    "blurred": blurred,
    "lower_brightness": lower_brightness,
    "mild_compression": mild_compression,
    "mild_stain_color": mild_stain_color,
}


class PerturbedLabeledCropDataset(Dataset):
    def __init__(
        self,
        records: list[stage5.CropRecord],
        transform,
        perturbation_name: str,
        args: argparse.Namespace,
    ) -> None:
        self.records = records
        self.transform = transform
        self.perturbation_name = perturbation_name
        self.args = args

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        if record.label is None:
            raise ValueError("PerturbedLabeledCropDataset received an unlabeled record")
        with Image.open(record.crop_path) as image:
            image = image.convert("RGB")
            image = PERTURBATIONS[self.perturbation_name](image, self.args)
            tensor = self.transform(image)
        return tensor, torch.tensor(stage5.LABEL_TO_ID[record.label], dtype=torch.long), record.source_key, str(record.crop_path)


def setup_names(requested: str) -> list[str]:
    if requested == "all":
        return ["supervised_scratch", "imagenet_finetune", "simclr_finetune"]
    return [requested]


def validate_checkpoints(model_root: Path, setups: list[str]) -> None:
    missing = [str(model_root / setup / "best_model.pt") for setup in setups if not (model_root / setup / "best_model.pt").exists()]
    if missing:
        preview = ", ".join(missing)
        raise FileNotFoundError(f"Missing trained model checkpoints: {preview}")


def load_model_for_setup(setup: str, model_root: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = stage5.load_torch_checkpoint(model_root / setup / "best_model.pt")
    pretrained = setup == "imagenet_finetune"
    model = stage5.build_resnet18_classifier(pretrained=pretrained)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate_setup(
    setup: str,
    model_root: Path,
    test_records: list[stage5.CropRecord],
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    model = load_model_for_setup(setup, model_root, device)
    setup_dir = output_dir / setup
    setup_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_perturbation: dict[str, dict[str, object]] = {}
    comparison_rows: list[dict[str, object]] = []

    for perturbation_name in PERTURBATION_ORDER:
        dataset = PerturbedLabeledCropDataset(
            test_records,
            stage5.eval_transform(args.image_size, grayscale=args.grayscale),
            perturbation_name,
            args,
        )
        loader = stage5.make_loader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
        y_true, y_score, source_keys, paths = stage5.predict(model, loader, device)
        crop_metrics = stage5.metric_dict(y_true, y_score)
        source_metrics = stage5.source_level_metrics(y_true, y_score, source_keys)
        metrics_by_perturbation[perturbation_name] = {
            "crop_metrics": crop_metrics,
            "source_metrics": source_metrics,
        }
        stage5.write_csv(
            setup_dir / f"{perturbation_name}_test_predictions.csv",
            stage5.prediction_rows(perturbation_name, y_true, y_score, source_keys, paths),
        )

    clean_crop_bal_acc = float(metrics_by_perturbation["clean"]["crop_metrics"]["balanced_accuracy"])
    clean_source_bal_acc = float(metrics_by_perturbation["clean"]["source_metrics"]["balanced_accuracy"])
    for perturbation_name in PERTURBATION_ORDER:
        crop_metrics = metrics_by_perturbation[perturbation_name]["crop_metrics"]
        source_metrics = metrics_by_perturbation[perturbation_name]["source_metrics"]
        comparison_rows.append(
            {
                "setup": setup,
                "perturbation": perturbation_name,
                "test_crop_balanced_accuracy": crop_metrics["balanced_accuracy"],
                "test_crop_accuracy": crop_metrics["accuracy"],
                "test_crop_roc_auc": crop_metrics["roc_auc"],
                "test_source_balanced_accuracy": source_metrics["balanced_accuracy"],
                "test_source_accuracy": source_metrics["accuracy"],
                "drop_from_clean_crop_balanced_accuracy": clean_crop_bal_acc - float(crop_metrics["balanced_accuracy"]),
                "drop_from_clean_source_balanced_accuracy": clean_source_bal_acc - float(source_metrics["balanced_accuracy"]),
            }
        )

    summary = {
        "setup": setup,
        "model_root": str(model_root),
        "n_test_crops": len(test_records),
        "n_test_sources": len({record.source_key for record in test_records}),
        "perturbations": metrics_by_perturbation,
    }
    (setup_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return metrics_by_perturbation, comparison_rows


def write_plots(output_dir: Path, rows: list[dict[str, object]]) -> None:
    visuals_dir = output_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    setups = sorted({str(row["setup"]) for row in rows})
    x = np.arange(len(PERTURBATION_ORDER))
    width = 0.24

    plt.figure(figsize=(11, 5))
    for index, setup in enumerate(setups):
        y = [
            next(
                float(row["test_crop_balanced_accuracy"])
                for row in rows
                if row["setup"] == setup and row["perturbation"] == perturbation
            )
            for perturbation in PERTURBATION_ORDER
        ]
        plt.bar(x + (index - (len(setups) - 1) / 2) * width, y, width=width, label=setup)
    plt.xticks(x, PERTURBATION_ORDER, rotation=20)
    plt.ylim(0.0, 1.02)
    plt.ylabel("Test Crop Balanced Accuracy")
    plt.title("Stage 7 Robustness Across Test Perturbations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(visuals_dir / "test_crop_balanced_accuracy_by_perturbation.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 5))
    for setup in setups:
        y = [
            next(
                float(row["drop_from_clean_crop_balanced_accuracy"])
                for row in rows
                if row["setup"] == setup and row["perturbation"] == perturbation
            )
            for perturbation in PERTURBATION_ORDER
        ]
        plt.plot(PERTURBATION_ORDER, y, marker="o", linewidth=2, label=setup)
    plt.ylabel("Drop From Clean Crop Balanced Accuracy")
    plt.title("Robustness Drop Relative To Clean Test Crops")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(visuals_dir / "robustness_drop_from_clean.png", dpi=180)
    plt.close()


def write_report(output_dir: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    report_lines = [
        "# Stage 7 Robustness Protocol",
        "",
        "## Protocol",
        "- Evaluate trained Stage 5 classifiers on the fixed clean test split and four deterministic perturbations.",
        "- Perturbations are designed to approximate common smartphone-microscope capture issues in the dataset.",
        f"- Inputs converted to grayscale before evaluation: `{args.grayscale}`.",
        f"- Blur radius: `{args.blur_radius}`.",
        f"- Brightness factor: `{args.brightness_factor}`.",
        f"- JPEG quality: `{args.jpeg_quality}`.",
        f"- Color scales (R,G,B): `({args.red_scale}, {args.green_scale}, {args.blue_scale})` with color factor `{args.color_factor}` and contrast factor `{args.contrast_factor}`.",
        "",
        "## Expected Interpretation",
        "- A robust model should keep most of its performance under these mild perturbations.",
        "- Large drops on blur, compression, or brightness suggest the model depends on capture-specific image quality rather than stable cell morphology.",
        "- Large drops on color perturbation suggest stain/color dependence.",
        "",
        "## Results",
    ]
    for setup in sorted({str(row["setup"]) for row in rows}):
        report_lines.append(f"### {setup}")
        for perturbation in PERTURBATION_ORDER:
            row = next(row for row in rows if row["setup"] == setup and row["perturbation"] == perturbation)
            report_lines.append(
                "- {perturbation}: crop_bal_acc={test_crop_balanced_accuracy}, "
                "crop_roc_auc={test_crop_roc_auc}, source_bal_acc={test_source_balanced_accuracy}, "
                "drop_from_clean={drop_from_clean_crop_balanced_accuracy}".format(**row)
            )
        worst = max(
            (row for row in rows if row["setup"] == setup and row["perturbation"] != "clean"),
            key=lambda row: float(row["drop_from_clean_crop_balanced_accuracy"]),
        )
        report_lines.append(
            f"- biggest crop-level drop: {worst['perturbation']} ({worst['drop_from_clean_crop_balanced_accuracy']})"
        )
        report_lines.append("")
    (output_dir / "stage7_report.md").write_text("\n".join(report_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 7: robustness evaluation on clean and perturbed test crops.")
    parser.add_argument("--pool-a-manifest", type=Path, default=Path("crop_outputs/pool_a_labeled_sickled_manifest.csv"))
    parser.add_argument("--pool-b-manifest", type=Path, default=Path("crop_outputs/pool_b_labeled_non_sickled_manifest.csv"))
    parser.add_argument("--model-root", type=Path, default=Path("stage5_outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("stage7_outputs"))
    parser.add_argument(
        "--setup",
        choices=["all", "supervised_scratch", "imagenet_finetune", "simclr_finetune"],
        default="all",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, etc.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Convert every crop to grayscale before evaluation while keeping the 3-channel model interface unchanged.",
    )
    parser.add_argument("--blur-radius", type=float, default=1.25)
    parser.add_argument("--brightness-factor", type=float, default=0.75)
    parser.add_argument("--jpeg-quality", type=int, default=45)
    parser.add_argument("--red-scale", type=float, default=1.08)
    parser.add_argument("--green-scale", type=float, default=0.97)
    parser.add_argument("--blue-scale", type=float, default=1.06)
    parser.add_argument("--color-factor", type=float, default=1.08)
    parser.add_argument("--contrast-factor", type=float, default=1.03)
    args = parser.parse_args()

    device = stage5.select_device(args.device)
    project_root = Path.cwd().resolve()
    model_root = args.model_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    setups = setup_names(args.setup)
    validate_checkpoints(model_root, setups)
    labeled_records = stage5.load_labeled_records(args.pool_a_manifest, args.pool_b_manifest, project_root)
    stage5.validate_records(labeled_records)
    stage5.validate_source_split_integrity(labeled_records)
    test_records = [record for record in labeled_records if record.split == "test"]

    comparison_rows: list[dict[str, object]] = []
    for setup in setups:
        _, rows = evaluate_setup(setup, model_root, test_records, args, device, output_dir)
        comparison_rows.extend(rows)

    stage5.write_csv(output_dir / "stage7_robustness_comparison.csv", comparison_rows)
    write_plots(output_dir, comparison_rows)
    write_report(output_dir, comparison_rows, args)
    print(f"Wrote {output_dir / 'stage7_report.md'}")


if __name__ == "__main__":
    main()
