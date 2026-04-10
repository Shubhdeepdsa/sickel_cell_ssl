from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SETUP_LABELS = {
    "supervised_scratch": "Scratch",
    "imagenet_finetune": "ImageNet",
    "simclr_finetune": "SimCLR",
}
SETUP_ORDER = ["supervised_scratch", "imagenet_finetune", "simclr_finetune"]
PERTURBATION_ORDER = [
    "clean",
    "blurred",
    "lower_brightness",
    "mild_compression",
    "mild_stain_color",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_stage5_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    for row in read_csv_rows(path):
        rows.append(
            {
                "setup": row["setup"],
                "test_crop_balanced_accuracy": float(row["test_crop_balanced_accuracy"]),
                "test_crop_accuracy": float(row["test_crop_accuracy"]),
                "test_crop_roc_auc": float(row["test_crop_roc_auc"]),
                "test_source_balanced_accuracy": float(row["test_source_balanced_accuracy"]),
                "test_source_accuracy": float(row["test_source_accuracy"]),
                "best_val_balanced_accuracy": float(row["best_val_balanced_accuracy"]),
            }
        )
    return rows


def load_stage6_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    for row in read_csv_rows(path):
        rows.append(
            {
                "setup": row["setup"],
                "label_fraction": float(row["label_fraction"]),
                "label_fraction_display": row["label_fraction_display"],
                "n_train_sources": int(row["n_train_sources"]),
                "train_source_counts": ast.literal_eval(row["train_source_counts"]),
                "n_train_crops": int(row["n_train_crops"]),
                "train_crop_counts": ast.literal_eval(row["train_crop_counts"]),
                "best_val_balanced_accuracy": float(row["best_val_balanced_accuracy"]),
                "test_crop_balanced_accuracy": float(row["test_crop_balanced_accuracy"]),
                "test_crop_accuracy": float(row["test_crop_accuracy"]),
                "test_crop_roc_auc": float(row["test_crop_roc_auc"]),
                "test_source_balanced_accuracy": float(row["test_source_balanced_accuracy"]),
                "test_source_accuracy": float(row["test_source_accuracy"]),
            }
        )
    return rows


def load_stage7_rows(path: Path, low_label: bool = False) -> list[dict[str, object]]:
    rows = []
    for row in read_csv_rows(path):
        parsed = {
            "setup": row["setup"],
            "perturbation": row["perturbation"],
            "test_crop_balanced_accuracy": float(row["test_crop_balanced_accuracy"]),
            "test_crop_accuracy": float(row["test_crop_accuracy"]),
            "test_crop_roc_auc": float(row["test_crop_roc_auc"]),
            "test_source_balanced_accuracy": float(row["test_source_balanced_accuracy"]),
            "test_source_accuracy": float(row["test_source_accuracy"]),
            "drop_from_clean_crop_balanced_accuracy": float(row["drop_from_clean_crop_balanced_accuracy"]),
            "drop_from_clean_source_balanced_accuracy": float(row["drop_from_clean_source_balanced_accuracy"]),
        }
        if low_label:
            parsed["fraction"] = row["fraction"]
            parsed["fraction_display"] = row["fraction_display"]
        rows.append(parsed)
    return rows


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stage5_plots(rows: list[dict[str, object]], outdir: Path) -> list[str]:
    ensure_dir(outdir)
    setups = [row["setup"] for row in rows if row["setup"] in SETUP_ORDER]
    x = np.arange(len(setups))
    width = 0.24
    metrics = [
        ("test_crop_balanced_accuracy", "Test Crop Balanced Accuracy"),
        ("test_source_balanced_accuracy", "Test Source Balanced Accuracy"),
        ("test_crop_roc_auc", "Test Crop ROC AUC"),
    ]

    plt.figure(figsize=(10, 5))
    for index, (metric_key, label) in enumerate(metrics):
        y = [float(next(row[metric_key] for row in rows if row["setup"] == setup)) for setup in setups]
        plt.bar(x + (index - 1) * width, y, width=width, label=label)
    plt.xticks(x, [SETUP_LABELS[setup] for setup in setups])
    plt.ylim(0.0, 1.02)
    plt.ylabel("Score")
    plt.title("Stage 5 Baseline Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "stage5_baseline_metrics.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    crop_bal = [float(next(row["test_crop_balanced_accuracy"] for row in rows if row["setup"] == setup)) for setup in setups]
    plt.bar([SETUP_LABELS[setup] for setup in setups], crop_bal)
    plt.ylim(0.0, 1.02)
    plt.ylabel("Test Crop Balanced Accuracy")
    plt.title("Stage 5: Who Works Best On Full Labels")
    plt.tight_layout()
    plt.savefig(outdir / "stage5_test_crop_balanced_accuracy.png", dpi=180)
    plt.close()

    return ["stage5_baseline_metrics.png", "stage5_test_crop_balanced_accuracy.png"]


def stage6_plots(rows: list[dict[str, object]], outdir: Path) -> list[str]:
    ensure_dir(outdir)
    fractions = sorted({float(row["label_fraction"]) for row in rows}, reverse=False)
    fraction_labels = [next(str(row["label_fraction_display"]) for row in rows if float(row["label_fraction"]) == fraction) for fraction in fractions]

    def values(setup: str, metric_key: str) -> list[float]:
        return [
            float(
                next(
                    row[metric_key]
                    for row in rows
                    if row["setup"] == setup and float(row["label_fraction"]) == fraction
                )
            )
            for fraction in fractions
        ]

    for metric_key, filename, title, ylabel in [
        ("test_crop_balanced_accuracy", "stage6_test_crop_balanced_accuracy.png", "Stage 6: Test Crop Balanced Accuracy", "Balanced Accuracy"),
        ("test_crop_roc_auc", "stage6_test_crop_roc_auc.png", "Stage 6: Test Crop ROC AUC", "ROC AUC"),
        ("test_source_balanced_accuracy", "stage6_test_source_balanced_accuracy.png", "Stage 6: Test Source Balanced Accuracy", "Balanced Accuracy"),
    ]:
        plt.figure(figsize=(9, 5))
        for setup in SETUP_ORDER:
            plt.plot(fraction_labels, values(setup, metric_key), marker="o", linewidth=2, label=SETUP_LABELS[setup])
        plt.ylim(0.0, 1.02)
        plt.ylabel(ylabel)
        plt.xlabel("Labeled Train Fraction")
        plt.title(title)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / filename, dpi=180)
        plt.close()

    plt.figure(figsize=(9, 5))
    simclr_minus_scratch = [
        values("simclr_finetune", "test_crop_balanced_accuracy")[i] - values("supervised_scratch", "test_crop_balanced_accuracy")[i]
        for i in range(len(fractions))
    ]
    imagenet_minus_scratch = [
        values("imagenet_finetune", "test_crop_balanced_accuracy")[i] - values("supervised_scratch", "test_crop_balanced_accuracy")[i]
        for i in range(len(fractions))
    ]
    plt.plot(fraction_labels, simclr_minus_scratch, marker="o", linewidth=2, label="SimCLR - Scratch")
    plt.plot(fraction_labels, imagenet_minus_scratch, marker="o", linewidth=2, label="ImageNet - Scratch")
    plt.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    plt.ylabel("Balanced Accuracy Gain")
    plt.xlabel("Labeled Train Fraction")
    plt.title("Stage 6: Gain Over Scratch")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "stage6_gain_over_scratch.png", dpi=180)
    plt.close()

    heatmap = np.zeros((len(SETUP_ORDER), len(fractions)), dtype=float)
    for r, setup in enumerate(SETUP_ORDER):
        heatmap[r, :] = values(setup, "test_crop_balanced_accuracy")
    plt.figure(figsize=(8, 4))
    image = plt.imshow(heatmap, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(image, label="Test Crop Balanced Accuracy")
    plt.xticks(np.arange(len(fractions)), fraction_labels)
    plt.yticks(np.arange(len(SETUP_ORDER)), [SETUP_LABELS[setup] for setup in SETUP_ORDER])
    plt.title("Stage 6: Low-Label Accuracy Heatmap")
    for r in range(len(SETUP_ORDER)):
        for c in range(len(fractions)):
            plt.text(c, r, f"{heatmap[r, c]:.3f}", ha="center", va="center", color="white", fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "stage6_accuracy_heatmap.png", dpi=180)
    plt.close()

    return [
        "stage6_test_crop_balanced_accuracy.png",
        "stage6_test_crop_roc_auc.png",
        "stage6_test_source_balanced_accuracy.png",
        "stage6_gain_over_scratch.png",
        "stage6_accuracy_heatmap.png",
    ]


def stage7_plots(rows: list[dict[str, object]], outdir: Path) -> list[str]:
    ensure_dir(outdir)
    x = np.arange(len(PERTURBATION_ORDER))
    width = 0.24

    plt.figure(figsize=(11, 5))
    for index, setup in enumerate(SETUP_ORDER):
        y = [
            float(next(row["test_crop_balanced_accuracy"] for row in rows if row["setup"] == setup and row["perturbation"] == perturbation))
            for perturbation in PERTURBATION_ORDER
        ]
        plt.bar(x + (index - 1) * width, y, width=width, label=SETUP_LABELS[setup])
    plt.xticks(x, PERTURBATION_ORDER, rotation=20)
    plt.ylim(0.0, 1.02)
    plt.ylabel("Test Crop Balanced Accuracy")
    plt.title("Stage 7: Robustness Across Perturbations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "stage7_balanced_accuracy_by_perturbation.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 5))
    for setup in SETUP_ORDER:
        y = [
            float(next(row["drop_from_clean_crop_balanced_accuracy"] for row in rows if row["setup"] == setup and row["perturbation"] == perturbation))
            for perturbation in PERTURBATION_ORDER
        ]
        plt.plot(PERTURBATION_ORDER, y, marker="o", linewidth=2, label=SETUP_LABELS[setup])
    plt.ylabel("Drop From Clean")
    plt.title("Stage 7: Robustness Drop Relative To Clean")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "stage7_drop_from_clean.png", dpi=180)
    plt.close()

    return [
        "stage7_balanced_accuracy_by_perturbation.png",
        "stage7_drop_from_clean.png",
    ]


def stage7_low_label_plots(rows: list[dict[str, object]], outdir: Path) -> list[str]:
    ensure_dir(outdir)
    fraction_keys = sorted({str(row["fraction"]) for row in rows}, key=lambda value: int(value.removeprefix("fraction_")))
    fraction_labels = [next(str(row["fraction_display"]) for row in rows if row["fraction"] == key) for key in fraction_keys]

    generated: list[str] = []

    for perturbation in PERTURBATION_ORDER:
        plt.figure(figsize=(9, 5))
        for setup in SETUP_ORDER:
            y = [
                float(
                    next(
                        row["test_crop_balanced_accuracy"]
                        for row in rows
                        if row["fraction"] == fraction_key and row["setup"] == setup and row["perturbation"] == perturbation
                    )
                )
                for fraction_key in fraction_keys
            ]
            plt.plot(fraction_labels, y, marker="o", linewidth=2, label=SETUP_LABELS[setup])
        plt.ylim(0.0, 1.02)
        plt.ylabel("Test Crop Balanced Accuracy")
        plt.xlabel("Labeled Train Fraction")
        plt.title(f"Stage 7 Low-Label: {perturbation}")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        filename = f"stage7_low_label_{perturbation}.png"
        plt.savefig(outdir / filename, dpi=180)
        plt.close()
        generated.append(filename)

    for perturbation in PERTURBATION_ORDER:
        simclr_advantage = [
            float(
                next(
                    row["test_crop_balanced_accuracy"]
                    for row in rows
                    if row["fraction"] == fraction_key and row["setup"] == "simclr_finetune" and row["perturbation"] == perturbation
                )
            )
            - float(
                next(
                    row["test_crop_balanced_accuracy"]
                    for row in rows
                    if row["fraction"] == fraction_key and row["setup"] == "supervised_scratch" and row["perturbation"] == perturbation
                )
            )
            for fraction_key in fraction_keys
        ]
        plt.figure(figsize=(8, 4.5))
        plt.plot(fraction_labels, simclr_advantage, marker="o", linewidth=2)
        plt.axhline(0.0, color="black", linewidth=1, alpha=0.5)
        plt.ylabel("SimCLR - Scratch")
        plt.xlabel("Labeled Train Fraction")
        plt.title(f"Stage 7 Low-Label Gain Under {perturbation}")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        filename = f"stage7_low_label_gain_simclr_vs_scratch_{perturbation}.png"
        plt.savefig(outdir / filename, dpi=180)
        plt.close()
        generated.append(filename)

    return generated


def combined_summary_plot(stage6_rows: list[dict[str, object]], stage7_rows: list[dict[str, object]] | None, outdir: Path) -> list[str]:
    ensure_dir(outdir)
    fractions = sorted({float(row["label_fraction"]) for row in stage6_rows}, reverse=False)
    fraction_labels = [next(str(row["label_fraction_display"]) for row in stage6_rows if float(row["label_fraction"]) == fraction) for fraction in fractions]

    plt.figure(figsize=(10, 5))
    for setup in SETUP_ORDER:
        clean_stage6 = [
            float(
                next(
                    row["test_crop_balanced_accuracy"]
                    for row in stage6_rows
                    if row["setup"] == setup and float(row["label_fraction"]) == fraction
                )
            )
            for fraction in fractions
        ]
        plt.plot(fraction_labels, clean_stage6, marker="o", linewidth=2, label=f"{SETUP_LABELS[setup]} clean")
    plt.ylim(0.0, 1.02)
    plt.ylabel("Test Crop Balanced Accuracy")
    plt.xlabel("Labeled Train Fraction")
    plt.title("Overall Low-Label Performance")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "overall_low_label_clean_performance.png", dpi=180)
    plt.close()

    generated = ["overall_low_label_clean_performance.png"]

    if stage7_rows:
        plt.figure(figsize=(10, 5))
        for setup in SETUP_ORDER:
            clean = float(next(row["test_crop_balanced_accuracy"] for row in stage7_rows if row["setup"] == setup and row["perturbation"] == "clean"))
            stain = float(next(row["test_crop_balanced_accuracy"] for row in stage7_rows if row["setup"] == setup and row["perturbation"] == "mild_stain_color"))
            compression = float(next(row["test_crop_balanced_accuracy"] for row in stage7_rows if row["setup"] == setup and row["perturbation"] == "mild_compression"))
            plt.bar(
                [f"{SETUP_LABELS[setup]} clean", f"{SETUP_LABELS[setup]} compression", f"{SETUP_LABELS[setup]} stain"],
                [clean, compression, stain],
            )
        plt.ylim(0.0, 1.02)
        plt.ylabel("Test Crop Balanced Accuracy")
        plt.title("Overall Robustness Snapshot")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(outdir / "overall_robustness_snapshot.png", dpi=180)
        plt.close()
        generated.append("overall_robustness_snapshot.png")

    return generated


def write_report(
    output_dir: Path,
    stage5_rows: list[dict[str, object]] | None,
    stage6_rows: list[dict[str, object]] | None,
    stage7_rows: list[dict[str, object]] | None,
    stage7_low_rows: list[dict[str, object]] | None,
) -> None:
    lines = [
        "# Stage 5-7 Visual Summary",
        "",
        "This report collects the main plots needed to understand what worked, where performance comes from, and how the three model families compare.",
        "",
    ]

    if stage5_rows:
        best_stage5 = max(stage5_rows, key=lambda row: float(row["test_crop_balanced_accuracy"]))
        lines.extend(
            [
                "## Stage 5",
                f"- Best full-label test crop balanced accuracy: `{best_stage5['test_crop_balanced_accuracy']:.4f}` from `{best_stage5['setup']}`.",
                f"- Main comparison plot: [stage5_baseline_metrics.png]({(output_dir / 'stage5' / 'stage5_baseline_metrics.png').as_posix()})",
                f"- Simple winner plot: [stage5_test_crop_balanced_accuracy.png]({(output_dir / 'stage5' / 'stage5_test_crop_balanced_accuracy.png').as_posix()})",
                "",
            ]
        )

    if stage6_rows:
        ten_percent = [row for row in stage6_rows if float(row["label_fraction"]) == 0.1]
        best_low_label = max(ten_percent, key=lambda row: float(row["test_crop_balanced_accuracy"]))
        simclr_10 = next(float(row["test_crop_balanced_accuracy"]) for row in ten_percent if row["setup"] == "simclr_finetune")
        scratch_10 = next(float(row["test_crop_balanced_accuracy"]) for row in ten_percent if row["setup"] == "supervised_scratch")
        lines.extend(
            [
                "## Stage 6",
                f"- At `10%` labels, best test crop balanced accuracy: `{best_low_label['test_crop_balanced_accuracy']:.4f}` from `{best_low_label['setup']}`.",
                f"- SimCLR minus Scratch at `10%`: `{simclr_10 - scratch_10:.4f}` balanced accuracy.",
                f"- Main low-label curve: [stage6_test_crop_balanced_accuracy.png]({(output_dir / 'stage6' / 'stage6_test_crop_balanced_accuracy.png').as_posix()})",
                f"- Gain-over-scratch plot: [stage6_gain_over_scratch.png]({(output_dir / 'stage6' / 'stage6_gain_over_scratch.png').as_posix()})",
                f"- Heatmap: [stage6_accuracy_heatmap.png]({(output_dir / 'stage6' / 'stage6_accuracy_heatmap.png').as_posix()})",
                "",
            ]
        )

    if stage7_rows:
        worst_rows = [row for row in stage7_rows if row["perturbation"] != "clean"]
        worst_stage7 = max(worst_rows, key=lambda row: float(row["drop_from_clean_crop_balanced_accuracy"]))
        lines.extend(
            [
                "## Stage 7",
                f"- Biggest robustness failure on full-label models: `{worst_stage7['setup']}` under `{worst_stage7['perturbation']}` with drop `{worst_stage7['drop_from_clean_crop_balanced_accuracy']:.4f}`.",
                f"- Perturbation comparison: [stage7_balanced_accuracy_by_perturbation.png]({(output_dir / 'stage7' / 'stage7_balanced_accuracy_by_perturbation.png').as_posix()})",
                f"- Drop-from-clean plot: [stage7_drop_from_clean.png]({(output_dir / 'stage7' / 'stage7_drop_from_clean.png').as_posix()})",
                "",
            ]
        )

    if stage7_low_rows:
        ten_clean = [row for row in stage7_low_rows if row["fraction_display"] == "10%" and row["perturbation"] == "clean"]
        simclr_10 = next(float(row["test_crop_balanced_accuracy"]) for row in ten_clean if row["setup"] == "simclr_finetune")
        scratch_10 = next(float(row["test_crop_balanced_accuracy"]) for row in ten_clean if row["setup"] == "supervised_scratch")
        lines.extend(
            [
                "## Stage 7 Low-Label",
                f"- At `10%` labels on clean test crops, SimCLR minus Scratch: `{simclr_10 - scratch_10:.4f}` balanced accuracy.",
                f"- Most important plot set lives in [stage7_low_label]({(output_dir / 'stage7_low_label').as_posix()})",
                f"- Example clean-fraction comparison: [stage7_low_label_clean.png]({(output_dir / 'stage7_low_label' / 'stage7_low_label_clean.png').as_posix()})",
                f"- Example low-label SimCLR-vs-Scratch gain: [stage7_low_label_gain_simclr_vs_scratch_clean.png]({(output_dir / 'stage7_low_label' / 'stage7_low_label_gain_simclr_vs_scratch_clean.png').as_posix()})",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Stage 7 Low-Label",
                "- No `stage7_low_label_outputs/stage7_low_label_robustness_comparison.csv` was found yet, so those plots were skipped.",
                "",
            ]
        )

    lines.extend(
        [
            "## Overall",
            f"- Overall clean low-label view: [overall_low_label_clean_performance.png]({(output_dir / 'summary' / 'overall_low_label_clean_performance.png').as_posix()})",
        ]
    )
    if stage7_rows:
        lines.append(f"- Robustness snapshot: [overall_robustness_snapshot.png]({(output_dir / 'summary' / 'overall_robustness_snapshot.png').as_posix()})")

    (output_dir / "visual_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visual summaries for Stages 5, 6, and 7.")
    parser.add_argument("--stage5-csv", type=Path, default=Path("stage5_outputs/stage5_baseline_comparison.csv"))
    parser.add_argument("--stage6-csv", type=Path, default=Path("stage6_outputs/stage6_low_label_comparison.csv"))
    parser.add_argument("--stage7-csv", type=Path, default=Path("stage7_outputs/stage7_robustness_comparison.csv"))
    parser.add_argument(
        "--stage7-low-label-csv",
        type=Path,
        default=Path("stage7_low_label_outputs/stage7_low_label_robustness_comparison.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("visual_reports/stage5_stage6_stage7"))
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)

    stage5_rows = load_stage5_rows(args.stage5_csv) if args.stage5_csv.exists() else None
    stage6_rows = load_stage6_rows(args.stage6_csv) if args.stage6_csv.exists() else None
    stage7_rows = load_stage7_rows(args.stage7_csv) if args.stage7_csv.exists() else None
    stage7_low_rows = load_stage7_rows(args.stage7_low_label_csv, low_label=True) if args.stage7_low_label_csv.exists() else None

    if stage5_rows:
        stage5_plots(stage5_rows, output_dir / "stage5")
    if stage6_rows:
        stage6_plots(stage6_rows, output_dir / "stage6")
    if stage7_rows:
        stage7_plots(stage7_rows, output_dir / "stage7")
    if stage7_low_rows:
        stage7_low_label_plots(stage7_low_rows, output_dir / "stage7_low_label")
    if stage6_rows:
        combined_summary_plot(stage6_rows, stage7_rows, output_dir / "summary")

    write_report(output_dir, stage5_rows, stage6_rows, stage7_rows, stage7_low_rows)
    print(f"Wrote {output_dir / 'visual_summary.md'}")


if __name__ == "__main__":
    main()
