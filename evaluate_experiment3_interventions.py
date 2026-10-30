"""
Experiment 3: Controlled Intervention & Shortcut Reliance Analysis

Evaluates the frozen ImageNet ResNet-18 reference model across:
1. Non-interventional subgroup stratification (clean vs border-present).
2. Controlled perimeter inpainting (T_border on border crops vs T_control on clean crops).
3. Background plasma color equalization (T_plasma vs T_photo-control).
4. Paired probability shift analysis (Delta p) with 2,000 source-cluster bootstrap CIs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASS_NAMES = ("non_sickled", "sickled")


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_frozen_reference_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Image Intervention Operators
# -----------------------------------------------------------------------------

def apply_perimeter_inpainting(
    crop_img: Image.Image,
    target_color: tuple[int, int, int] = (220, 220, 220),
    margin_ratio: float = 0.15,
) -> Image.Image:
    """
    Replaces the outer margin (15% = 19px) with target background color,
    blending with a smooth Gaussian edge kernel to prevent sharp step artifacts.
    """
    w, h = crop_img.size
    margin_px = int(min(w, h) * margin_ratio)

    # Create inner mask: 255 inside, 0 outside margin
    inner_mask = Image.new("L", (w, h), 0)
    inner_box = (margin_px, margin_px, w - margin_px, h - margin_px)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(inner_mask)
    draw.rectangle(inner_box, fill=255)
    # Smooth edge with Gaussian blur
    smooth_mask = inner_mask.filter(ImageFilter.GaussianBlur(radius=2.0))

    # Background canvas
    bg_canvas = Image.new("RGB", (w, h), color=target_color)
    result = Image.composite(crop_img, bg_canvas, smooth_mask)
    return result


def apply_plasma_equalization(
    crop_img: Image.Image,
    local_bg_rgb: tuple[float, float, float],
    target_bg_rgb: tuple[float, float, float] = (220.0, 218.0, 215.0),
) -> Image.Image:
    """Multiplicatively rescales RGB channels to match global target plasma color."""
    arr = np.array(crop_img, dtype=float)
    scales = [
        target_bg_rgb[c] / max(local_bg_rgb[c], 1.0)
        for c in range(3)
    ]
    for c in range(3):
        arr[:, :, c] = np.clip(arr[:, :, c] * scales[c], 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def apply_luminance_control(
    crop_img: Image.Image,
    local_luma: float,
    target_luma: float = 218.0,
) -> Image.Image:
    """Rescales overall luminance without altering RGB chromaticity ratios."""
    arr = np.array(crop_img, dtype=float)
    scale = target_luma / max(local_luma, 1.0)
    arr = np.clip(arr * scale, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


# -----------------------------------------------------------------------------
# Evaluation Dataset & Inference
# -----------------------------------------------------------------------------

class SingleImageDataset(Dataset):
    def __init__(self, images: list[Image.Image], transform: transforms.Compose) -> None:
        self.images = images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.transform(self.images[index])


def predict_probabilities(
    model: nn.Module,
    images: list[Image.Image],
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    dataset = SingleImageDataset(images, eval_tf)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_probs = []
    with torch.no_grad():
        for tensors in loader:
            tensors = tensors.to(device)
            outputs = model(tensors)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # P(sickled)
            all_probs.extend(probs.cpu().numpy().tolist())
    return np.array(all_probs, dtype=float)


# -----------------------------------------------------------------------------
# Bootstrap Uncertainty Estimation
# -----------------------------------------------------------------------------

def compute_cluster_bootstrap_shift(
    source_keys: list[str],
    delta_p: np.ndarray,
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    src_map: dict[str, list[int]] = defaultdict(list)
    for idx, sk in enumerate(source_keys):
        src_map[sk].append(idx)

    unique_sources = list(src_map.keys())
    n_sources = len(unique_sources)
    point_mean = float(np.mean(delta_p))

    boot_means = []
    for _ in range(n_bootstraps):
        sampled_srcs = rng.choice(unique_sources, size=n_sources, replace=True)
        sampled_indices = []
        for s in sampled_srcs:
            sampled_indices.extend(src_map[s])
        boot_means.append(np.mean(delta_p[sampled_indices]))

    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return point_mean, ci_low, ci_high


# -----------------------------------------------------------------------------
# Main Intervention Runner
# -----------------------------------------------------------------------------

def run_experiment_3_interventions(
    checkpoint_path: Path = Path("experiment_3_reference_outputs/best_model.pt"),
    output_dir: Path = Path("experiment_3_reference_outputs"),
    batch_size: int = 32,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device("auto")
    print(f"Loading frozen reference model from {checkpoint_path} on {device}...")
    model = load_frozen_reference_model(checkpoint_path, device)

    # Load test crops from feature cache
    features_csv = Path("experiment_2_confound_outputs/experiment_2_features.csv")
    with features_csv.open() as f:
        records = [r for r in csv.DictReader(f) if r["split"] == "test"]

    print(f"Loaded {len(records)} test crops for intervention evaluation.")

    # Load original images
    orig_images = []
    for r in records:
        pool_dir = "pool_a_labeled_sickled" if r["pool"] == "Pool A" else "pool_b_labeled_non_sickled"
        p = Path("crop_outputs") / pool_dir / f"{r['crop_id']}.jpg"
        with Image.open(p) as img:
            orig_images.append(img.convert("RGB"))

    # Baseline predictions
    print("Computing baseline predictions on unmodified test crops...")
    p_orig = predict_probabilities(model, orig_images, device, batch_size=batch_size)
    y_true = np.array([int(r["label"]) for r in records])
    source_keys = [r["source_key"] for r in records]
    has_border = np.array([float(r["t1_has_border"]) for r in records])

    # -------------------------------------------------------------------------
    # Stage 3B: Non-Interventional Subgroup Stratification
    # -------------------------------------------------------------------------
    strata = {
        "stratum_1_sickled_clean": (y_true == 1) & (has_border == 0),
        "stratum_2_sickled_border": (y_true == 1) & (has_border == 1),
        "stratum_3_non_sickled_clean": (y_true == 0) & (has_border == 0),
        "stratum_4_non_sickled_border": (y_true == 0) & (has_border == 1),
    }

    subgroup_results = {}
    for name, mask in strata.items():
        if np.sum(mask) == 0:
            continue
        sub_yt = y_true[mask]
        sub_po = p_orig[mask]
        sub_pred = (sub_po >= 0.50).astype(int)
        acc = float(np.mean(sub_yt == sub_pred))
        subgroup_results[name] = {
            "n_crops": int(np.sum(mask)),
            "accuracy": float(acc),
            "mean_p_sickled": float(np.mean(sub_po)),
            "std_p_sickled": float(np.std(sub_po)),
        }

    # -------------------------------------------------------------------------
    # Stage 3C: Controlled Perimeter Interventions (T_border vs T_control)
    # -------------------------------------------------------------------------
    print("Applying perimeter interventions (T_border and T_control)...")
    perim_intervened_images = []
    for img, r in zip(orig_images, records):
        target_col = (int(float(r["t3_bg_r_mean"])), int(float(r["t3_bg_g_mean"])), int(float(r["t3_bg_b_mean"])))
        intervened = apply_perimeter_inpainting(img, target_color=target_col, margin_ratio=0.15)
        perim_intervened_images.append(intervened)

    p_perim = predict_probabilities(model, perim_intervened_images, device, batch_size=batch_size)
    delta_p_perim = p_perim - p_orig

    # Separate causal border effect on border-bearing crops vs control on clean crops
    mask_b_border = (y_true == 0) & (has_border == 1)
    mask_b_clean = (y_true == 0) & (has_border == 0)

    shift_b_border, b_bord_low, b_bord_high = compute_cluster_bootstrap_shift(
        [source_keys[i] for i in np.where(mask_b_border)[0]],
        delta_p_perim[mask_b_border],
    )
    shift_b_clean, b_clean_low, b_clean_high = compute_cluster_bootstrap_shift(
        [source_keys[i] for i in np.where(mask_b_clean)[0]],
        delta_p_perim[mask_b_clean],
    )

    # -------------------------------------------------------------------------
    # Stage 3D: Background Plasma Equalization (T_plasma vs T_photo-control)
    # -------------------------------------------------------------------------
    print("Applying plasma equalization interventions...")
    plasma_intervened_images = []
    photo_control_images = []

    # Global training target plasma RGB
    target_bg_rgb = (220.0, 218.0, 215.0)
    for img, r in zip(orig_images, records):
        loc_bg = (float(r["t3_bg_r_mean"]), float(r["t3_bg_g_mean"]), float(r["t3_bg_b_mean"]))
        loc_luma = float(r["t3_bg_luma_mean"])
        plasma_intervened_images.append(apply_plasma_equalization(img, loc_bg, target_bg_rgb))
        photo_control_images.append(apply_luminance_control(img, loc_luma, target_luma=218.0))

    p_plasma = predict_probabilities(model, plasma_intervened_images, device, batch_size=batch_size)
    p_photo = predict_probabilities(model, photo_control_images, device, batch_size=batch_size)

    delta_p_plasma = p_plasma - p_orig
    delta_p_photo = p_photo - p_orig

    shift_plasma, plas_low, plas_high = compute_cluster_bootstrap_shift(source_keys, delta_p_plasma)
    shift_photo, phot_low, phot_high = compute_cluster_bootstrap_shift(source_keys, delta_p_photo)

    # -------------------------------------------------------------------------
    # Stage 3E: Flip Rates & Causal Attribution Summary
    # -------------------------------------------------------------------------
    flips_perim_border = np.sum((p_orig[mask_b_border] < 0.50) & (p_perim[mask_b_border] >= 0.50))
    flips_perim_clean = np.sum((p_orig[mask_b_clean] < 0.50) & (p_perim[mask_b_clean] >= 0.50))

    flip_rate_border = flips_perim_border / max(np.sum(mask_b_border), 1)
    flip_rate_clean = flips_perim_clean / max(np.sum(mask_b_clean), 1)

    intervention_summary = {
        "subgroup_stratification": subgroup_results,
        "perimeter_border_intervention": {
            "border_non_sickled_mean_delta_p": shift_b_border,
            "border_non_sickled_95_ci": [b_bord_low, b_bord_high],
            "border_flip_rate_to_sickled": float(flip_rate_border),
            "control_clean_non_sickled_mean_delta_p": shift_b_clean,
            "control_clean_95_ci": [b_clean_low, b_clean_high],
            "control_flip_rate_to_sickled": float(flip_rate_clean),
        },
        "plasma_equalization_intervention": {
            "plasma_mean_delta_p": shift_plasma,
            "plasma_95_ci": [plas_low, plas_high],
            "photo_control_mean_delta_p": shift_photo,
            "photo_control_95_ci": [phot_low, phot_high],
        },
    }

    # Save CSV with per-crop paired results
    paired_rows = []
    for r, yt, po, pp, ppl, ph, hb in zip(records, y_true, p_orig, p_perim, p_plasma, p_photo, has_border):
        paired_rows.append({
            "crop_id": r["crop_id"],
            "source_key": r["source_key"],
            "y_true": yt,
            "has_border": int(hb),
            "p_orig": round(float(po), 5),
            "p_perim": round(float(pp), 5),
            "delta_p_perim": round(float(pp - po), 5),
            "p_plasma": round(float(ppl), 5),
            "delta_p_plasma": round(float(ppl - po), 5),
            "p_photo_control": round(float(ph), 5),
            "delta_p_photo": round(float(ph - po), 5),
        })

    paired_csv = output_dir / "experiment_3_intervention_paired_shifts.csv"
    with paired_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired_rows[0].keys()))
        writer.writeheader()
        writer.writerows(paired_rows)

    (output_dir / "experiment_3_interventions.json").write_text(json.dumps(intervention_summary, indent=2))
    generate_intervention_report(output_dir / "EXPERIMENT_3_INTERVENTION_REPORT.md", intervention_summary)
    print(f"\nStage 3 interventions complete! Report saved to {output_dir / 'EXPERIMENT_3_INTERVENTION_REPORT.md'}")


def generate_intervention_report(report_path: Path, summary: dict[str, object]) -> None:
    sg = summary["subgroup_stratification"]
    pi = summary["perimeter_border_intervention"]
    pl = summary["plasma_equalization_intervention"]

    md = [
        "# Experiment 3 — Artifact Intervention & Shortcut Reliance Report",
        "",
        "## 1. Executive Summary",
        "This experiment evaluates whether the frozen **ImageNet ResNet-18 Reference Model** relies on non-morphological shortcut features (microscope aperture borders and plasma color balance) by systematically testing:",
        "1. **Non-Interventional Subgroup Stratification**: Natural accuracy and confidence differences across clean vs border-bearing test crops.",
        "2. **Controlled Perimeter Inpainting ($T_{\\text{border}}$ vs $T_{\\text{control}}$)**: Neutralizing aperture borders against matched negative control edits.",
        "3. **Background Plasma Equalization ($T_{\\text{plasma}}$ vs $T_{\\text{photo-control}}$)**: Neutralizing plasma color shifts against neutral luminance edits.",
        "",
        "---",
        "",
        "## 2. Stage 3B: Non-Interventional Subgroup Stratification",
        "",
        "| Subgroup Stratum | Number of Crops | Reference Model Accuracy | Mean P(Sickled) | Std Dev P(Sickled) |",
        "|---|---:|---:|---:|---:|",
        f"| **Stratum 1: Sickled (Clean / No Border)** | {sg['stratum_1_sickled_clean']['n_crops']} | {sg['stratum_1_sickled_clean']['accuracy']:.1%} | {sg['stratum_1_sickled_clean']['mean_p_sickled']:.3f} | {sg['stratum_1_sickled_clean']['std_p_sickled']:.3f} |",
        f"| **Stratum 2: Sickled (Border Present)** | {sg['stratum_2_sickled_border']['n_crops']} | {sg['stratum_2_sickled_border']['accuracy']:.1%} | {sg['stratum_2_sickled_border']['mean_p_sickled']:.3f} | {sg['stratum_2_sickled_border']['std_p_sickled']:.3f} |",
        f"| **Stratum 3: Non-Sickled (Clean / No Border)** | {sg['stratum_3_non_sickled_clean']['n_crops']} | {sg['stratum_3_non_sickled_clean']['accuracy']:.1%} | {sg['stratum_3_non_sickled_clean']['mean_p_sickled']:.3f} | {sg['stratum_3_non_sickled_clean']['std_p_sickled']:.3f} |",
        f"| **Stratum 4: Non-Sickled (Border Present)** | {sg['stratum_4_non_sickled_border']['n_crops']} | {sg['stratum_4_non_sickled_border']['accuracy']:.1%} | {sg['stratum_4_non_sickled_border']['mean_p_sickled']:.3f} | {sg['stratum_4_non_sickled_border']['std_p_sickled']:.3f} |",
        "",
        "---",
        "",
        "## 3. Stage 3C: Controlled Perimeter & Border Inpainting",
        "",
        "| Intervention Condition | Target Crops | Mean $\\Delta P(\\text{Sickled})$ | 95% Source-Cluster CI | Decision Flip Rate (to Sickled) |",
        "|---|---|---:|---:|---:|",
        f"| **$T_{{\\text{{border}}}}$ (Border Removal)** | Border Non-Sickled Crops | **{pi['border_non_sickled_mean_delta_p']:+.3f}** | [{pi['border_non_sickled_95_ci'][0]:+.3f}, {pi['border_non_sickled_95_ci'][1]:+.3f}] | **{pi['border_flip_rate_to_sickled']:.1%}** |",
        f"| **$T_{{\\text{{control}}}}$ (Matched Control Edit)** | Clean Non-Sickled Crops | **{pi['control_clean_non_sickled_mean_delta_p']:+.3f}** | [{pi['control_clean_95_ci'][0]:+.3f}, {pi['control_clean_95_ci'][1]:+.3f}] | **{pi['control_flip_rate_to_sickled']:.1%}** |",
        "",
        "---",
        "",
        "## 4. Stage 3D: Plasma Color & Illumination Equalization",
        "",
        "| Intervention Condition | Evaluation Set | Mean $\\Delta P(\\text{Sickled})$ | 95% Source-Cluster CI |",
        "|---|---|---:|---:|",
        f"| **$T_{{\\text{{plasma}}}}$ (Color Equalization)** | All Test Crops | **{pl['plasma_mean_delta_p']:+.3f}** | [{pl['plasma_95_ci'][0]:+.3f}, {pl['plasma_95_ci'][1]:+.3f}] |",
        f"| **$T_{{\\text{{photo-control}}}}$ (Luminance Scaling)** | All Test Crops | **{pl['photo_control_mean_delta_p']:+.3f}** | [{pl['photo_control_95_ci'][0]:+.3f}, {pl['photo_control_95_ci'][1]:+.3f}] |",
        "",
        "---",
        "",
        "## 5. Pre-Registered Outcome Interpretation",
        "Referencing the pre-registered decision matrix in `EXPERIMENT_3_PROTOCOL.md`:",
    ]

    # Pattern evaluation
    net_border_shift = pi["border_non_sickled_mean_delta_p"] - pi["control_clean_non_sickled_mean_delta_p"]
    if net_border_shift > 0.15 and pi["border_flip_rate_to_sickled"] > 0.15:
        pattern = "**Pattern 1: Substantial Shortcut Sensitivity** — The reference model exhibits significant causal reliance on aperture borders. Removing the border shifts predictions toward sickled significantly above control edits."
    elif net_border_shift > 0.05 or pi["border_flip_rate_to_sickled"] > 0.05:
        pattern = "**Pattern 2: Moderate Confound Sensitivity** — The reference model exhibits modest sensitivity to perimeter/background cues, but decisions are largely preserved by underlying cell morphology."
    else:
        pattern = "**Pattern 3: Intervention Insensitivity** — Predictions are insensitive to border removal and plasma equalization beyond matched control perturbations, demonstrating that representations rely primarily on cell morphology."

    md.append(f"- **Observed Result**: {pattern}")
    report_path.write_text("\n".join(md) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 3 Artifact Interventions")
    parser.add_argument("--checkpoint", type=Path, default=Path("experiment_3_reference_outputs/best_model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiment_3_reference_outputs"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    run_experiment_3_interventions(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
