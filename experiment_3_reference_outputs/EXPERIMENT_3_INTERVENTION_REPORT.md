# Experiment 3 — Artifact Intervention & Shortcut Reliance Report

## 1. Executive Summary
This experiment evaluates whether the frozen **ImageNet ResNet-18 Reference Model** relies on non-morphological shortcut features (microscope aperture borders and plasma color balance) by systematically testing:
1. **Non-Interventional Subgroup Stratification**: Natural accuracy and confidence differences across clean vs border-bearing test crops.
2. **Controlled Perimeter Inpainting ($T_{\text{border}}$ vs $T_{\text{control}}$)**: Neutralizing aperture borders against matched negative control edits.
3. **Background Plasma Equalization ($T_{\text{plasma}}$ vs $T_{\text{photo-control}}$)**: Neutralizing plasma color shifts against neutral luminance edits.

---

## 2. Stage 3B: Non-Interventional Subgroup Stratification

| Subgroup Stratum | Number of Crops | Reference Model Accuracy | Mean P(Sickled) | Std Dev P(Sickled) |
|---|---:|---:|---:|---:|
| **Stratum 1: Sickled (Clean / No Border)** | 168 | 99.4% | 0.994 | 0.075 |
| **Stratum 2: Sickled (Border Present)** | 14 | 100.0% | 0.996 | 0.012 |
| **Stratum 3: Non-Sickled (Clean / No Border)** | 152 | 99.3% | 0.013 | 0.063 |
| **Stratum 4: Non-Sickled (Border Present)** | 70 | 100.0% | 0.003 | 0.023 |

---

## 3. Stage 3C: Controlled Perimeter & Border Inpainting

| Intervention Condition | Target Crops | Mean $\Delta P(\text{Sickled})$ | 95% Source-Cluster CI | Decision Flip Rate (to Sickled) |
|---|---|---:|---:|---:|
| **$T_{\text{border}}$ (Border Removal)** | Border Non-Sickled Crops | **+0.045** | [+0.012, +0.089] | **2.9%** |
| **$T_{\text{control}}$ (Matched Control Edit)** | Clean Non-Sickled Crops | **+0.047** | [+0.020, +0.079] | **3.9%** |

---

## 4. Stage 3D: Plasma Color & Illumination Equalization

| Intervention Condition | Evaluation Set | Mean $\Delta P(\text{Sickled})$ | 95% Source-Cluster CI |
|---|---|---:|---:|
| **$T_{\text{plasma}}$ (Color Equalization)** | All Test Crops | **+0.004** | [-0.001, +0.012] |
| **$T_{\text{photo-control}}$ (Luminance Scaling)** | All Test Crops | **+0.001** | [-0.000, +0.003] |

---

## 5. Pre-Registered Outcome Interpretation
Referencing the pre-registered decision matrix in `EXPERIMENT_3_PROTOCOL.md`:
- **Observed Result**: **Pattern 3: Intervention Insensitivity** — Predictions are insensitive to border removal and plasma equalization beyond matched control perturbations, demonstrating that representations rely primarily on cell morphology.
