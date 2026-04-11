from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SETUPS = ["supervised_scratch", "imagenet_finetune", "simclr_finetune"]
SETUP_LABELS = {
    "supervised_scratch": "Scratch",
    "imagenet_finetune": "ImageNet",
    "simclr_finetune": "SimCLR",
}
PERTURBATIONS = ["clean", "blurred", "lower_brightness", "mild_compression", "mild_stain_color"]
FRACTIONS = [1.0, 0.5, 0.2, 0.1]
FRACTION_LABELS = {1.0: "100%", 0.5: "50%", 0.2: "20%", 0.1: "10%"}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_stage5(path: Path) -> dict[str, dict[str, float]]:
    data: dict[str, dict[str, float]] = {}
    for row in read_csv_rows(path):
        data[row["setup"]] = {
            "test_crop_balanced_accuracy": float(row["test_crop_balanced_accuracy"]),
            "test_crop_accuracy": float(row["test_crop_accuracy"]),
            "test_crop_roc_auc": float(row["test_crop_roc_auc"]),
            "test_source_balanced_accuracy": float(row["test_source_balanced_accuracy"]),
            "best_val_balanced_accuracy": float(row["best_val_balanced_accuracy"]),
        }
    return data


def load_stage6(path: Path) -> dict[tuple[str, float], dict[str, float]]:
    data: dict[tuple[str, float], dict[str, float]] = {}
    for row in read_csv_rows(path):
        fraction = float(row["label_fraction"])
        data[(row["setup"], fraction)] = {
            "test_crop_balanced_accuracy": float(row["test_crop_balanced_accuracy"]),
            "test_crop_accuracy": float(row["test_crop_accuracy"]),
            "test_crop_roc_auc": float(row["test_crop_roc_auc"]),
            "test_source_balanced_accuracy": float(row["test_source_balanced_accuracy"]),
            "best_val_balanced_accuracy": float(row["best_val_balanced_accuracy"]),
        }
    return data


def load_stage7(path: Path, low_label: bool = False) -> dict[tuple[str, str] | tuple[str, float, str], dict[str, float]]:
    data: dict[tuple[str, str] | tuple[str, float, str], dict[str, float]] = {}
    for row in read_csv_rows(path):
        if low_label:
            key = (row["setup"], float(row["fraction_display"].rstrip("%")) / 100.0, row["perturbation"])
        else:
            key = (row["setup"], row["perturbation"])
        data[key] = {
            "test_crop_balanced_accuracy": float(row["test_crop_balanced_accuracy"]),
            "test_crop_accuracy": float(row["test_crop_accuracy"]),
            "test_crop_roc_auc": float(row["test_crop_roc_auc"]),
            "test_source_balanced_accuracy": float(row["test_source_balanced_accuracy"]),
            "drop_from_clean_crop_balanced_accuracy": float(row["drop_from_clean_crop_balanced_accuracy"]),
            "drop_from_clean_source_balanced_accuracy": float(row["drop_from_clean_source_balanced_accuracy"]),
        }
    return data


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def bar_positions(n_groups: int, n_series: int, width: float) -> list[np.ndarray]:
    x = np.arange(n_groups)
    offsets = [x + (idx - (n_series - 1) / 2) * width for idx in range(n_series)]
    return offsets


def plot_stage5_rgb_vs_gray(stage5_rgb: dict, stage5_gray: dict, outdir: Path) -> list[str]:
    ensure_dir(outdir)
    labels = [SETUP_LABELS[setup] for setup in SETUPS]
    width = 0.35
    x = np.arange(len(SETUPS))

    plt.figure(figsize=(9, 5))
    rgb_vals = [stage5_rgb[setup]["test_crop_balanced_accuracy"] for setup in SETUPS]
    gray_vals = [stage5_gray[setup]["test_crop_balanced_accuracy"] for setup in SETUPS]
    plt.bar(x - width / 2, rgb_vals, width=width, label="RGB")
    plt.bar(x + width / 2, gray_vals, width=width, label="Grayscale")
    plt.xticks(x, labels)
    plt.ylim(0.0, 1.02)
    plt.ylabel("Test Crop Balanced Accuracy")
    plt.title("Stage 5: RGB vs Grayscale")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "stage5_rgb_vs_grayscale_balanced_accuracy.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    rgb_vals = [stage5_rgb[setup]["test_crop_roc_auc"] for setup in SETUPS]
    gray_vals = [stage5_gray[setup]["test_crop_roc_auc"] for setup in SETUPS]
    plt.bar(x - width / 2, rgb_vals, width=width, label="RGB")
    plt.bar(x + width / 2, gray_vals, width=width, label="Grayscale")
    plt.xticks(x, labels)
    plt.ylim(0.5, 1.02)
    plt.ylabel("Test Crop ROC AUC")
    plt.title("Stage 5: RGB vs Grayscale ROC AUC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "stage5_rgb_vs_grayscale_roc_auc.png", dpi=180)
    plt.close()

    return [
        "stage5_rgb_vs_grayscale_balanced_accuracy.png",
        "stage5_rgb_vs_grayscale_roc_auc.png",
    ]


def plot_stage6_low_label(stage6_rgb: dict, stage6_gray: dict, outdir: Path) -> list[str]:
    ensure_dir(outdir)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, setup in zip(axes, SETUPS):
        rgb_vals = [stage6_rgb[(setup, fraction)]["test_crop_balanced_accuracy"] for fraction in FRACTIONS]
        gray_vals = [stage6_gray[(setup, fraction)]["test_crop_balanced_accuracy"] for fraction in FRACTIONS]
        ax.plot([FRACTION_LABELS[f] for f in FRACTIONS], rgb_vals, marker="o", linewidth=2, label="RGB")
        ax.plot([FRACTION_LABELS[f] for f in FRACTIONS], gray_vals, marker="o", linewidth=2, label="Grayscale")
        ax.set_title(SETUP_LABELS[setup])
        ax.set_xlabel("Labeled Train Fraction")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Test Crop Balanced Accuracy")
    axes[0].set_ylim(0.0, 1.02)
    axes[-1].legend()
    fig.suptitle("Stage 6: Low-Label Performance")
    fig.tight_layout()
    fig.savefig(outdir / "stage6_rgb_vs_grayscale_low_label_curves.png", dpi=180)
    plt.close(fig)

    plt.figure(figsize=(9, 5))
    rgb_gain = [
        stage6_rgb[("simclr_finetune", fraction)]["test_crop_balanced_accuracy"]
        - stage6_rgb[("supervised_scratch", fraction)]["test_crop_balanced_accuracy"]
        for fraction in FRACTIONS
    ]
    gray_gain = [
        stage6_gray[("simclr_finetune", fraction)]["test_crop_balanced_accuracy"]
        - stage6_gray[("supervised_scratch", fraction)]["test_crop_balanced_accuracy"]
        for fraction in FRACTIONS
    ]
    plt.plot([FRACTION_LABELS[f] for f in FRACTIONS], rgb_gain, marker="o", linewidth=2, label="RGB")
    plt.plot([FRACTION_LABELS[f] for f in FRACTIONS], gray_gain, marker="o", linewidth=2, label="Grayscale")
    plt.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    plt.ylabel("SimCLR - Scratch")
    plt.xlabel("Labeled Train Fraction")
    plt.title("Stage 6: SimCLR Gain Over Scratch")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "stage6_simclr_gain_over_scratch_rgb_vs_grayscale.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    fraction = 0.1
    x = np.arange(len(SETUPS))
    width = 0.35
    rgb_vals = [stage6_rgb[(setup, fraction)]["test_crop_balanced_accuracy"] for setup in SETUPS]
    gray_vals = [stage6_gray[(setup, fraction)]["test_crop_balanced_accuracy"] for setup in SETUPS]
    plt.bar(x - width / 2, rgb_vals, width=width, label="RGB")
    plt.bar(x + width / 2, gray_vals, width=width, label="Grayscale")
    plt.xticks(x, [SETUP_LABELS[s] for s in SETUPS])
    plt.ylim(0.0, 1.02)
    plt.ylabel("Test Crop Balanced Accuracy")
    plt.title("Stage 6: 10% Labels, RGB vs Grayscale")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "stage6_fraction_10_rgb_vs_grayscale.png", dpi=180)
    plt.close()

    return [
        "stage6_rgb_vs_grayscale_low_label_curves.png",
        "stage6_simclr_gain_over_scratch_rgb_vs_grayscale.png",
        "stage6_fraction_10_rgb_vs_grayscale.png",
    ]


def plot_stage7_full(stage7_rgb: dict, stage7_gray: dict, outdir: Path) -> list[str]:
    ensure_dir(outdir)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for ax, dataset_name, stage7 in [
        (axes[0], "RGB", stage7_rgb),
        (axes[1], "Grayscale", stage7_gray),
    ]:
        x = np.arange(len(PERTURBATIONS))
        width = 0.24
        for idx, setup in enumerate(SETUPS):
            vals = [stage7[(setup, perturbation)]["test_crop_balanced_accuracy"] for perturbation in PERTURBATIONS]
            ax.bar(x + (idx - 1) * width, vals, width=width, label=SETUP_LABELS[setup])
        ax.set_xticks(x)
        ax.set_xticklabels(PERTURBATIONS, rotation=20)
        ax.set_ylim(0.0, 1.02)
        ax.set_title(dataset_name)
    axes[0].set_ylabel("Test Crop Balanced Accuracy")
    axes[1].legend()
    fig.suptitle("Stage 7: Full-Label Robustness")
    fig.tight_layout()
    fig.savefig(outdir / "stage7_full_rgb_vs_grayscale_robustness.png", dpi=180)
    plt.close(fig)

    plt.figure(figsize=(9, 5))
    x = np.arange(len(SETUPS))
    width = 0.35
    rgb_vals = [stage7_rgb[(setup, "mild_stain_color")]["drop_from_clean_crop_balanced_accuracy"] for setup in SETUPS]
    gray_vals = [stage7_gray[(setup, "mild_stain_color")]["drop_from_clean_crop_balanced_accuracy"] for setup in SETUPS]
    plt.bar(x - width / 2, rgb_vals, width=width, label="RGB")
    plt.bar(x + width / 2, gray_vals, width=width, label="Grayscale")
    plt.xticks(x, [SETUP_LABELS[s] for s in SETUPS])
    plt.ylabel("Drop From Clean")
    plt.title("Stage 7: Stain/Color Dependence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "stage7_stain_color_drop_rgb_vs_grayscale.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    x = np.arange(len(SETUPS))
    width = 0.35
    rgb_vals = [stage7_rgb[(setup, "lower_brightness")]["drop_from_clean_crop_balanced_accuracy"] for setup in SETUPS]
    gray_vals = [stage7_gray[(setup, "lower_brightness")]["drop_from_clean_crop_balanced_accuracy"] for setup in SETUPS]
    plt.bar(x - width / 2, rgb_vals, width=width, label="RGB")
    plt.bar(x + width / 2, gray_vals, width=width, label="Grayscale")
    plt.xticks(x, [SETUP_LABELS[s] for s in SETUPS])
    plt.ylabel("Drop From Clean")
    plt.title("Stage 7: Brightness Sensitivity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "stage7_brightness_drop_rgb_vs_grayscale.png", dpi=180)
    plt.close()

    return [
        "stage7_full_rgb_vs_grayscale_robustness.png",
        "stage7_stain_color_drop_rgb_vs_grayscale.png",
        "stage7_brightness_drop_rgb_vs_grayscale.png",
    ]


def plot_stage7_low_label(stage7_low_rgb: dict, stage7_low_gray: dict, outdir: Path) -> list[str]:
    ensure_dir(outdir)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, setup in zip(axes, SETUPS):
        rgb_vals = [stage7_low_rgb[(setup, fraction, "clean")]["test_crop_balanced_accuracy"] for fraction in FRACTIONS]
        gray_vals = [stage7_low_gray[(setup, fraction, "clean")]["test_crop_balanced_accuracy"] for fraction in FRACTIONS]
        ax.plot([FRACTION_LABELS[f] for f in FRACTIONS], rgb_vals, marker="o", linewidth=2, label="RGB")
        ax.plot([FRACTION_LABELS[f] for f in FRACTIONS], gray_vals, marker="o", linewidth=2, label="Grayscale")
        ax.set_title(SETUP_LABELS[setup])
        ax.set_xlabel("Labeled Train Fraction")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Test Crop Balanced Accuracy")
    axes[0].set_ylim(0.0, 1.02)
    axes[-1].legend()
    fig.suptitle("Stage 7 Low-Label: Clean Performance")
    fig.tight_layout()
    fig.savefig(outdir / "stage7_low_label_clean_rgb_vs_grayscale.png", dpi=180)
    plt.close(fig)

    plt.figure(figsize=(9, 5))
    rgb_gain = [
        stage7_low_rgb[("simclr_finetune", fraction, "clean")]["test_crop_balanced_accuracy"]
        - stage7_low_rgb[("supervised_scratch", fraction, "clean")]["test_crop_balanced_accuracy"]
        for fraction in FRACTIONS
    ]
    gray_gain = [
        stage7_low_gray[("simclr_finetune", fraction, "clean")]["test_crop_balanced_accuracy"]
        - stage7_low_gray[("supervised_scratch", fraction, "clean")]["test_crop_balanced_accuracy"]
        for fraction in FRACTIONS
    ]
    plt.plot([FRACTION_LABELS[f] for f in FRACTIONS], rgb_gain, marker="o", linewidth=2, label="RGB")
    plt.plot([FRACTION_LABELS[f] for f in FRACTIONS], gray_gain, marker="o", linewidth=2, label="Grayscale")
    plt.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    plt.ylabel("SimCLR - Scratch")
    plt.xlabel("Labeled Train Fraction")
    plt.title("Stage 7 Low-Label: SimCLR Gain On Clean Data")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "stage7_low_label_clean_simclr_gain_rgb_vs_grayscale.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    for setup in SETUPS:
        rgb_vals = [stage7_low_rgb[(setup, fraction, "mild_stain_color")]["drop_from_clean_crop_balanced_accuracy"] for fraction in FRACTIONS]
        gray_vals = [stage7_low_gray[(setup, fraction, "mild_stain_color")]["drop_from_clean_crop_balanced_accuracy"] for fraction in FRACTIONS]
        plt.plot([FRACTION_LABELS[f] for f in FRACTIONS], rgb_vals, marker="o", linewidth=2, label=f"{SETUP_LABELS[setup]} RGB")
        plt.plot([FRACTION_LABELS[f] for f in FRACTIONS], gray_vals, marker="o", linewidth=2, linestyle="--", label=f"{SETUP_LABELS[setup]} Gray")
    plt.ylabel("Drop From Clean")
    plt.xlabel("Labeled Train Fraction")
    plt.title("Stage 7 Low-Label: Stain/Color Sensitivity")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "stage7_low_label_stain_drop_rgb_vs_grayscale.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    for setup in SETUPS:
        rgb_vals = [stage7_low_rgb[(setup, fraction, "lower_brightness")]["drop_from_clean_crop_balanced_accuracy"] for fraction in FRACTIONS]
        gray_vals = [stage7_low_gray[(setup, fraction, "lower_brightness")]["drop_from_clean_crop_balanced_accuracy"] for fraction in FRACTIONS]
        plt.plot([FRACTION_LABELS[f] for f in FRACTIONS], rgb_vals, marker="o", linewidth=2, label=f"{SETUP_LABELS[setup]} RGB")
        plt.plot([FRACTION_LABELS[f] for f in FRACTIONS], gray_vals, marker="o", linewidth=2, linestyle="--", label=f"{SETUP_LABELS[setup]} Gray")
    plt.ylabel("Drop From Clean")
    plt.xlabel("Labeled Train Fraction")
    plt.title("Stage 7 Low-Label: Brightness Sensitivity")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "stage7_low_label_brightness_drop_rgb_vs_grayscale.png", dpi=180)
    plt.close()

    return [
        "stage7_low_label_clean_rgb_vs_grayscale.png",
        "stage7_low_label_clean_simclr_gain_rgb_vs_grayscale.png",
        "stage7_low_label_stain_drop_rgb_vs_grayscale.png",
        "stage7_low_label_brightness_drop_rgb_vs_grayscale.png",
    ]


def md_link(path: Path) -> str:
    return f"[{path.name}]({path})"


def build_report(
    outdir: Path,
    stage5_rgb: dict,
    stage5_gray: dict,
    stage6_rgb: dict,
    stage6_gray: dict,
    stage7_rgb: dict,
    stage7_gray: dict,
    stage7_low_rgb: dict,
    stage7_low_gray: dict,
) -> None:
    report_path = outdir / "grayscale_truth_test_report.md"
    stage5_best_rgb = max(SETUPS, key=lambda setup: stage5_rgb[setup]["test_crop_balanced_accuracy"])
    stage5_best_gray = max(SETUPS, key=lambda setup: stage5_gray[setup]["test_crop_balanced_accuracy"])
    simclr_gain_10_rgb = stage6_rgb[("simclr_finetune", 0.1)]["test_crop_balanced_accuracy"] - stage6_rgb[("supervised_scratch", 0.1)]["test_crop_balanced_accuracy"]
    simclr_gain_10_gray = stage6_gray[("simclr_finetune", 0.1)]["test_crop_balanced_accuracy"] - stage6_gray[("supervised_scratch", 0.1)]["test_crop_balanced_accuracy"]
    stain_drop_rgb = {setup: stage7_rgb[(setup, "mild_stain_color")]["drop_from_clean_crop_balanced_accuracy"] for setup in SETUPS}
    stain_drop_gray = {setup: stage7_gray[(setup, "mild_stain_color")]["drop_from_clean_crop_balanced_accuracy"] for setup in SETUPS}

    lines = [
        "# Grayscale Truth Test",
        "",
        "This report explains the RGB-vs-grayscale rerun in simple terms. The main question is:",
        "",
        "> If we remove color and stain cues, do the models still work well, especially when labels are limited?",
        "",
        "If the answer is yes, that supports the idea that the models are learning real cell shape and morphology, not only color/stain shortcuts.",
        "",
        "## How To Read The Numbers",
        "",
        "- `Balanced accuracy`: higher is better. `1.0` is perfect. Around `0.5` means the model is close to guessing on a balanced two-class task.",
        "- `ROC AUC`: higher is better. `1.0` is perfect. Around `0.5` means no useful ranking ability.",
        "- `Drop from clean`: lower is better. `0.0` means the perturbation did not hurt. Bigger positive numbers mean the perturbation damaged performance more.",
        "- A small negative drop can happen. It means the perturbed version happened to score slightly better than clean on that run. Treat that as \"no meaningful harm.\"",
        "",
        "## Main Takeaway",
        "",
        f"- Full-label best RGB model: `{stage5_best_rgb}` at `{stage5_rgb[stage5_best_rgb]['test_crop_balanced_accuracy']:.4f}`.",
        f"- Full-label best grayscale model: `{stage5_best_gray}` at `{stage5_gray[stage5_best_gray]['test_crop_balanced_accuracy']:.4f}`.",
        f"- At `10%` labels, SimCLR minus Scratch in RGB: `{simclr_gain_10_rgb:.4f}`.",
        f"- At `10%` labels, SimCLR minus Scratch in grayscale: `{simclr_gain_10_gray:.4f}`.",
        f"- Scratch stain/color drop at full labels: RGB `{stain_drop_rgb['supervised_scratch']:.4f}` vs grayscale `{stain_drop_gray['supervised_scratch']:.4f}`.",
        f"- SimCLR stain/color drop at full labels: RGB `{stain_drop_rgb['simclr_finetune']:.4f}` vs grayscale `{stain_drop_gray['simclr_finetune']:.4f}`.",
        "",
        "In simpler terms: removing color did **not** make the whole pipeline collapse. That is evidence that morphology is carrying real signal. The grayscale rerun also removes most of the color/stain fragility we saw in the RGB robustness test.",
        "",
        "## Stage 5: Full-Label Comparison",
        "",
        f"{md_link((outdir / 'stage5' / 'stage5_rgb_vs_grayscale_balanced_accuracy.png').resolve())}",
        "",
        "What this plot shows:",
        "- Each group is one model family: Scratch, ImageNet, SimCLR.",
        "- Blue/orange bars compare RGB and grayscale on the same full-label task.",
        "- Higher bars are better.",
        "",
        "What to look for:",
        "- If grayscale bars were much lower, color was doing too much of the work.",
        "- If grayscale bars stay close, morphology is still strong.",
        "",
        "What it means here:",
        f"- Scratch improved slightly from `{stage5_rgb['supervised_scratch']['test_crop_balanced_accuracy']:.4f}` to `{stage5_gray['supervised_scratch']['test_crop_balanced_accuracy']:.4f}`.",
        f"- ImageNet improved slightly from `{stage5_rgb['imagenet_finetune']['test_crop_balanced_accuracy']:.4f}` to `{stage5_gray['imagenet_finetune']['test_crop_balanced_accuracy']:.4f}`.",
        f"- SimCLR improved slightly from `{stage5_rgb['simclr_finetune']['test_crop_balanced_accuracy']:.4f}` to `{stage5_gray['simclr_finetune']['test_crop_balanced_accuracy']:.4f}`.",
        "- So the grayscale rerun does not show collapse. That is the opposite of what we would expect if stain/color were the only real signal.",
        "",
        f"{md_link((outdir / 'stage5' / 'stage5_rgb_vs_grayscale_roc_auc.png').resolve())}",
        "",
        "This second plot is the same idea, but for ROC AUC. Again, higher is better.",
        "",
        "## Stage 6: Low-Label Comparison",
        "",
        f"{md_link((outdir / 'stage6' / 'stage6_rgb_vs_grayscale_low_label_curves.png').resolve())}",
        "",
        "What this plot shows:",
        "- Each panel is one model family.",
        "- The x-axis is how many labeled training images were kept: `100%`, `50%`, `20%`, `10%`.",
        "- The y-axis is test balanced accuracy. Higher is better.",
        "- RGB and grayscale are overlaid directly.",
        "",
        "What to look for:",
        "- Lines that stay high as labels shrink are stronger.",
        "- If the grayscale line stays close to RGB, the model still works without color.",
        "- The most important region is `20%` and `10%`, because that is where SSL is supposed to help.",
        "",
        "What it means here in simple terms:",
        f"- At `10%`, Scratch gets `{stage6_rgb[('supervised_scratch', 0.1)]['test_crop_balanced_accuracy']:.4f}` in RGB and `{stage6_gray[('supervised_scratch', 0.1)]['test_crop_balanced_accuracy']:.4f}` in grayscale.",
        f"- At `10%`, SimCLR gets `{stage6_rgb[('simclr_finetune', 0.1)]['test_crop_balanced_accuracy']:.4f}` in RGB and `{stage6_gray[('simclr_finetune', 0.1)]['test_crop_balanced_accuracy']:.4f}` in grayscale.",
        f"- At `10%`, ImageNet gets `{stage6_rgb[('imagenet_finetune', 0.1)]['test_crop_balanced_accuracy']:.4f}` in RGB and `{stage6_gray[('imagenet_finetune', 0.1)]['test_crop_balanced_accuracy']:.4f}` in grayscale.",
        "- So the low-label benefit of SimCLR is still there even after removing color.",
        "",
        f"{md_link((outdir / 'stage6' / 'stage6_simclr_gain_over_scratch_rgb_vs_grayscale.png').resolve())}",
        "",
        "How to read this plot:",
        "- This is not raw accuracy. It is `SimCLR minus Scratch`.",
        "- Higher is better for SimCLR here.",
        "- Values above zero mean SimCLR is helping over training from scratch.",
        "",
        "What it means here:",
        f"- At `10%`, SimCLR beats Scratch by `{simclr_gain_10_rgb:.4f}` in RGB and `{simclr_gain_10_gray:.4f}` in grayscale.",
        "- That is strong evidence that SSL still carries useful value when labels are limited, even after removing color.",
        "",
        f"{md_link((outdir / 'stage6' / 'stage6_fraction_10_rgb_vs_grayscale.png').resolve())}",
        "",
        "This is the same low-label story, but only for `10%` labels. It is the simplest plot to show someone if you want one fast summary of the SSL question.",
        "",
        "## Stage 7: Full-Label Robustness",
        "",
        f"{md_link((outdir / 'stage7' / 'stage7_full_rgb_vs_grayscale_robustness.png').resolve())}",
        "",
        "What this plot shows:",
        "- Left side: RGB robustness.",
        "- Right side: grayscale robustness.",
        "- Each group is a perturbation: clean, blur, lower brightness, JPEG compression, and mild stain/color shift.",
        "- Higher bars are better.",
        "",
        "What to look for:",
        "- If the bar under a perturbation drops a lot from clean, the model is sensitive to that problem.",
        "- The `mild_stain_color` column is especially important here because this whole experiment is about removing color dependence.",
        "",
        f"{md_link((outdir / 'stage7' / 'stage7_stain_color_drop_rgb_vs_grayscale.png').resolve())}",
        "",
        "How to read this plot:",
        "- This uses `drop from clean`, so lower is better.",
        "- A tall bar means the model was hurt a lot by stain/color change.",
        "- A near-zero bar means the model is basically insensitive to that perturbation.",
        "",
        "What it means here:",
        f"- Scratch stain/color drop fell from `{stain_drop_rgb['supervised_scratch']:.4f}` in RGB to `{stain_drop_gray['supervised_scratch']:.4f}` in grayscale.",
        f"- ImageNet stain/color drop fell from `{stain_drop_rgb['imagenet_finetune']:.4f}` in RGB to `{stain_drop_gray['imagenet_finetune']:.4f}` in grayscale.",
        f"- SimCLR stain/color drop fell from `{stain_drop_rgb['simclr_finetune']:.4f}` in RGB to `{stain_drop_gray['simclr_finetune']:.4f}` in grayscale.",
        "- This is one of the strongest pieces of evidence in the whole study. The huge stain/color fragility in RGB mostly disappears in grayscale.",
        "",
        f"{md_link((outdir / 'stage7' / 'stage7_brightness_drop_rgb_vs_grayscale.png').resolve())}",
        "",
        "This plot is also `drop from clean`, so lower is better.",
        "",
        "What it means:",
        "- Brightness still matters, especially for Scratch.",
        "- But the stain/color problem is much more clearly solved by the grayscale rerun.",
        "",
        "## Stage 7 Low-Label Robustness",
        "",
        f"{md_link((outdir / 'stage7_low_label' / 'stage7_low_label_clean_rgb_vs_grayscale.png').resolve())}",
        "",
        "What this plot shows:",
        "- This is the low-label version of Stage 7, on clean test crops.",
        "- Higher lines are better.",
        "- It tells you whether low-label gains survive into the grayscale branch.",
        "",
        f"{md_link((outdir / 'stage7_low_label' / 'stage7_low_label_clean_simclr_gain_rgb_vs_grayscale.png').resolve())}",
        "",
        "How to read this plot:",
        "- Again, this is `SimCLR minus Scratch`.",
        "- Higher is better for SimCLR.",
        "- Values above zero mean SSL is still helping in the low-label regime.",
        "",
        "What it means here:",
        f"- At `10%`, RGB SimCLR gain is `{stage7_low_rgb[('simclr_finetune', 0.1, 'clean')]['test_crop_balanced_accuracy'] - stage7_low_rgb[('supervised_scratch', 0.1, 'clean')]['test_crop_balanced_accuracy']:.4f}`.",
        f"- At `10%`, grayscale SimCLR gain is `{stage7_low_gray[('simclr_finetune', 0.1, 'clean')]['test_crop_balanced_accuracy'] - stage7_low_gray[('supervised_scratch', 0.1, 'clean')]['test_crop_balanced_accuracy']:.4f}`.",
        "- So SSL is still valuable after color removal.",
        "",
        f"{md_link((outdir / 'stage7_low_label' / 'stage7_low_label_stain_drop_rgb_vs_grayscale.png').resolve())}",
        "",
        "How to read this plot:",
        "- This is `drop from clean` under the stain/color perturbation.",
        "- Lower is better.",
        "- Solid lines are RGB; dashed lines are grayscale.",
        "",
        "What it means:",
        "- In RGB, stain/color shifts can badly hurt low-label models, especially Scratch and sometimes SimCLR.",
        "- In grayscale, those stain/color drops become tiny or disappear, which is exactly what we would hope if color shortcuts were removed.",
        "",
        f"{md_link((outdir / 'stage7_low_label' / 'stage7_low_label_brightness_drop_rgb_vs_grayscale.png').resolve())}",
        "",
        "This is also `drop from clean`, so lower is better.",
        "",
        "What it means:",
        "- Brightness remains a real capture-quality issue even in grayscale.",
        "- That tells us the grayscale rerun removed color dependence, but did not magically solve every acquisition problem.",
        "",
        "## Bottom Line",
        "",
        "In plain language:",
        "",
        "- The grayscale rerun does **not** collapse. That means color/stain is not the whole story.",
        "- The low-label SimCLR advantage remains positive in grayscale. That supports the idea that SSL is learning something useful beyond color shortcuts.",
        "- The biggest RGB fragility was stain/color perturbation. That fragility mostly disappears in grayscale.",
        "- Brightness still hurts weaker models, especially Scratch, so capture quality still matters.",
        "- ImageNet remains the strongest overall baseline on this dataset, but SimCLR still shows real value over Scratch when labels are scarce.",
        "",
        "So the simplest scientific reading is:",
        "",
        "> there is real morphology signal here, and SSL still helps in the low-label setting, but ImageNet pretraining is currently the strongest overall starting point.",
    ]
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RGB-vs-grayscale visuals and explanation report.")
    parser.add_argument("--stage5-rgb", type=Path, default=Path("stage5_outputs/stage5_baseline_comparison.csv"))
    parser.add_argument("--stage5-gray", type=Path, default=Path("stage5_grayscale_outputs/stage5_baseline_comparison.csv"))
    parser.add_argument("--stage6-rgb", type=Path, default=Path("stage6_outputs/stage6_low_label_comparison.csv"))
    parser.add_argument("--stage6-gray", type=Path, default=Path("stage6_grayscale_outputs/stage6_low_label_comparison.csv"))
    parser.add_argument("--stage7-rgb", type=Path, default=Path("stage7_outputs/stage7_robustness_comparison.csv"))
    parser.add_argument("--stage7-gray", type=Path, default=Path("stage7_grayscale_outputs/stage7_robustness_comparison.csv"))
    parser.add_argument("--stage7-low-rgb", type=Path, default=Path("stage7_low_label_outputs/stage7_low_label_robustness_comparison.csv"))
    parser.add_argument("--stage7-low-gray", type=Path, default=Path("stage7_low_label_grayscale_outputs/stage7_low_label_robustness_comparison.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("visual_reports/grayscale_truth_test"))
    args = parser.parse_args()

    outdir = args.output_dir.resolve()
    ensure_dir(outdir)

    stage5_rgb = load_stage5(args.stage5_rgb)
    stage5_gray = load_stage5(args.stage5_gray)
    stage6_rgb = load_stage6(args.stage6_rgb)
    stage6_gray = load_stage6(args.stage6_gray)
    stage7_rgb = load_stage7(args.stage7_rgb, low_label=False)
    stage7_gray = load_stage7(args.stage7_gray, low_label=False)
    stage7_low_rgb = load_stage7(args.stage7_low_rgb, low_label=True)
    stage7_low_gray = load_stage7(args.stage7_low_gray, low_label=True)

    plot_stage5_rgb_vs_gray(stage5_rgb, stage5_gray, outdir / "stage5")
    plot_stage6_low_label(stage6_rgb, stage6_gray, outdir / "stage6")
    plot_stage7_full(stage7_rgb, stage7_gray, outdir / "stage7")
    plot_stage7_low_label(stage7_low_rgb, stage7_low_gray, outdir / "stage7_low_label")
    build_report(outdir, stage5_rgb, stage5_gray, stage6_rgb, stage6_gray, stage7_rgb, stage7_gray, stage7_low_rgb, stage7_low_gray)
    print(f"Wrote {outdir / 'grayscale_truth_test_report.md'}")


if __name__ == "__main__":
    main()
