from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import evaluate_stage7_robustness as stage7
import train_stage5 as stage5


def fraction_display_name(fraction_dir: Path) -> str:
    suffix = fraction_dir.name.removeprefix("fraction_")
    return f"{int(suffix)}%"


def discover_fraction_dirs(stage6_root: Path) -> list[Path]:
    fraction_dirs = sorted(
        [path for path in stage6_root.iterdir() if path.is_dir() and path.name.startswith("fraction_")],
        key=lambda path: path.name,
        reverse=True,
    )
    if not fraction_dirs:
        raise FileNotFoundError(f"No fraction_* directories found in {stage6_root}")
    return fraction_dirs


def write_fraction_plots(output_dir: Path, rows: list[dict[str, object]], fraction_dirs: list[Path], setups: list[str]) -> None:
    visuals_dir = output_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    fraction_labels = [fraction_display_name(path) for path in fraction_dirs]
    x = np.arange(len(fraction_labels))
    width = 0.24

    for perturbation in stage7.PERTURBATION_ORDER:
        plt.figure(figsize=(10, 5))
        for index, setup in enumerate(setups):
            y = [
                next(
                    float(row["test_crop_balanced_accuracy"])
                    for row in rows
                    if row["fraction"] == fraction_dir.name and row["setup"] == setup and row["perturbation"] == perturbation
                )
                for fraction_dir in fraction_dirs
            ]
            plt.bar(x + (index - (len(setups) - 1) / 2) * width, y, width=width, label=setup)
        plt.xticks(x, fraction_labels)
        plt.ylim(0.0, 1.02)
        plt.ylabel("Test Crop Balanced Accuracy")
        plt.title(f"Stage 7 Low-Label Robustness: {perturbation}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(visuals_dir / f"{perturbation}_by_fraction.png", dpi=180)
        plt.close()

    plt.figure(figsize=(11, 6))
    for perturbation in stage7.PERTURBATION_ORDER:
        y = [
            np.mean(
                [
                    float(row["test_crop_balanced_accuracy"])
                    for row in rows
                    if row["fraction"] == fraction_dir.name and row["perturbation"] == perturbation
                ]
            )
            for fraction_dir in fraction_dirs
        ]
        plt.plot(fraction_labels, y, marker="o", linewidth=2, label=perturbation)
    plt.ylabel("Mean Test Crop Balanced Accuracy Across Setups")
    plt.title("Average Robustness Across Label Fractions")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(visuals_dir / "mean_accuracy_by_fraction_and_perturbation.png", dpi=180)
    plt.close()


def write_report(output_dir: Path, rows: list[dict[str, object]], fraction_dirs: list[Path], setups: list[str], args: argparse.Namespace) -> None:
    report_lines = [
        "# Stage 7 Low-Label Robustness",
        "",
        "## Goal",
        "- Evaluate whether SSL retains value when labels are scarce and capture quality varies.",
        "- Each model family is evaluated on the Stage 6 low-label checkpoints under the same deterministic perturbations as Stage 7.",
        "",
        "## Protocol",
        f"- Stage 6 model root: `{args.stage6_root}`.",
        f"- Fractions evaluated: `{', '.join(fraction_display_name(path) for path in fraction_dirs)}`.",
        f"- Setups: `{', '.join(setups)}`.",
        f"- Inputs converted to grayscale before evaluation: `{args.grayscale}`.",
        f"- Perturbations: `{', '.join(stage7.PERTURBATION_ORDER)}`.",
        "- Validation/test splits remain fixed; only the amount of labeled train data differs across fractions.",
        "",
        "## Results",
    ]

    for fraction_dir in fraction_dirs:
        report_lines.append(f"### {fraction_display_name(fraction_dir)}")
        for setup in setups:
            report_lines.append(f"- {setup}:")
            for perturbation in stage7.PERTURBATION_ORDER:
                row = next(
                    row
                    for row in rows
                    if row["fraction"] == fraction_dir.name and row["setup"] == setup and row["perturbation"] == perturbation
                )
                report_lines.append(
                    "  - {perturbation}: crop_bal_acc={test_crop_balanced_accuracy}, "
                    "crop_roc_auc={test_crop_roc_auc}, source_bal_acc={test_source_balanced_accuracy}, "
                    "drop_from_clean={drop_from_clean_crop_balanced_accuracy}".format(**row)
                )

            if setup == "simclr_finetune":
                simclr_clean = next(
                    float(row["test_crop_balanced_accuracy"])
                    for row in rows
                    if row["fraction"] == fraction_dir.name and row["setup"] == "simclr_finetune" and row["perturbation"] == "clean"
                )
                scratch_clean = next(
                    float(row["test_crop_balanced_accuracy"])
                    for row in rows
                    if row["fraction"] == fraction_dir.name and row["setup"] == "supervised_scratch" and row["perturbation"] == "clean"
                )
                report_lines.append(f"  - simclr_minus_scratch_clean_bal_acc={simclr_clean - scratch_clean}")
        report_lines.append("")

    report_lines.extend(
        [
            "## Interpretation",
            "- If SimCLR stays above supervised scratch at 10% and 20% under perturbation, that supports SSL value in the low-label regime.",
            "- If SimCLR collapses under blur, brightness, compression, or color shifts while scratch does not, then its gains may be fragile.",
            "- If ImageNet remains stronger than SimCLR, then SSL is useful but not yet the strongest pretraining strategy on this dataset.",
        ]
    )
    (output_dir / "stage7_low_label_report.md").write_text("\n".join(report_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 7: robustness evaluation for Stage 6 low-label models.")
    parser.add_argument("--pool-a-manifest", type=Path, default=Path("crop_outputs/pool_a_labeled_sickled_manifest.csv"))
    parser.add_argument("--pool-b-manifest", type=Path, default=Path("crop_outputs/pool_b_labeled_non_sickled_manifest.csv"))
    parser.add_argument("--stage6-root", type=Path, default=Path("stage6_outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("stage7_low_label_outputs"))
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
    stage6_root = args.stage6_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    setups = stage7.setup_names(args.setup)
    fraction_dirs = discover_fraction_dirs(stage6_root)
    labeled_records = stage5.load_labeled_records(args.pool_a_manifest, args.pool_b_manifest, project_root)
    stage5.validate_records(labeled_records)
    stage5.validate_source_split_integrity(labeled_records)
    test_records = [record for record in labeled_records if record.split == "test"]

    comparison_rows: list[dict[str, object]] = []
    for fraction_dir in fraction_dirs:
        stage7.validate_checkpoints(fraction_dir, setups)
        fraction_output_dir = output_dir / fraction_dir.name
        fraction_output_dir.mkdir(parents=True, exist_ok=True)
        for setup in setups:
            _, rows = stage7.evaluate_setup(setup, fraction_dir, test_records, args, device, fraction_output_dir)
            for row in rows:
                row["fraction"] = fraction_dir.name
                row["fraction_display"] = fraction_display_name(fraction_dir)
            comparison_rows.extend(rows)

    stage5.write_csv(output_dir / "stage7_low_label_robustness_comparison.csv", comparison_rows)
    write_fraction_plots(output_dir, comparison_rows, fraction_dirs, setups)
    write_report(output_dir, comparison_rows, fraction_dirs, setups, args)
    print(f"Wrote {output_dir / 'stage7_low_label_report.md'}")


if __name__ == "__main__":
    main()
