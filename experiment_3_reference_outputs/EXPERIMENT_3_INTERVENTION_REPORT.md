# Experiment 3 — Artifact Intervention & Shortcut Reliance Report

## 1. Executive Summary & Defensible Findings
This experiment investigates whether the frozen **ImageNet ResNet-18 Reference Model** relies on unintended non-morphological image artifacts (specifically microscope aperture borders and background plasma color shifts) identified in Experiments 1 and 2.

### Authoritative Scientific Conclusion
> [!IMPORTANT]
> **Primary Finding**: The reference model showed **limited sensitivity to the tested border and color interventions on average**, despite the presence of label-predictive peripheral and background features in the dataset.
>
> However, perimeter inpainting is **not strictly border-specific**: it also alters peripheral cellular context. In approximately 14% of non-sickled crops (both clean and border-bearing), perimeter replacement caused substantial individual prediction swings ($|\Delta P| > 0.10$).
>
> **Methodological Boundary**: While these results demonstrate that the reference model does not strictly require the microscope aperture border to achieve high classification accuracy, they **do not prove that the model relies exclusively or primarily on biological sickle-cell morphology**. Untested acquisition characteristics, staining nuances, or subtle contextual cues could also influence predictions.

---

## 2. Authoritative Dataset Specification

Ground truth established directly from the frozen physical manifests (`crop_outputs/pool_a_labeled_sickled_manifest.csv` and `crop_outputs/pool_b_labeled_non_sickled_manifest.csv`):

| Partition | Pool A (Sickled) | Pool B (Non-Sickled) | Total Crops | Candidate-Bearing Source Images | Source Split Violations |
|---|---:|---:|---:|---:|:---:|
| **Training** | 917 | 1,014 | **1,931** | 382 | **0 (PASS)** |
| **Validation** | 150 | 232 | **382** | 80 | **0 (PASS)** |
| **Test** | 182 | 222 | **404** | 81 | **0 (PASS)** |
| **Total Labeled Pool** | **1,249** | **1,468** | **2,717** | **543** | **0 (PASS)** |

---

## 3. Stage 3B: Non-Interventional Subgroup Stratification

How accurately does the frozen model classify natural, unmodified crops when partitioned by the presence of dark aperture borders?

| Subgroup Stratum | Number of Crops | Reference Model Accuracy | Mean P(Sickled) | Std Dev P(Sickled) |
|---|---:|---:|---:|---:|
| **Stratum 1: Sickled (Clean / No Border)** | 168 | **99.4%** | 0.994 | 0.075 |
| **Stratum 2: Sickled (Border Present)** | 14 | **100.0%** | 0.996 | 0.012 |
| **Stratum 3: Non-Sickled (Clean / No Border)** | 152 | **99.3%** | 0.013 | 0.063 |
| **Stratum 4: Non-Sickled (Border Present)** | 70 | **100.0%** | 0.003 | 0.023 |

> **Key Observation**: On the 152 natural non-sickled crops with **zero aperture border**, the model achieves **99.3% accuracy** (151/152 correct) with mean predicted sickled probability of only **0.013**. The dark border artifact is not required for the classifier to recognize non-sickled cells.

---

## 4. Stage 3C: Controlled Perimeter Interventions ($T_{\text{border}}$ vs $T_{\text{control}}$)

The outer 15% margin (19 pixels) was replaced with local background color using a smooth Gaussian edge blend ($T_{\text{border}}$ on border crops, $T_{\text{control}}$ on clean crops):

### A. Signed Directional vs Absolute Shift Metrics

| Target Group | Condition | Number of Crops | Directional Mean $\Delta P$ | Absolute Mean $|\Delta P|$ | Max $|\Delta P|$ | $|\Delta P| > 0.10$ Rate | Decision Flip Rate |
|---|---|---:|---:|---:|---:|---:|---:|
| **Non-Sickled (Border)** | $T_{\text{border}}$ | 70 | **+0.0447** | **0.0447** | 0.9639 | **14.3%** (10 / 70) | **2.9%** (2 / 70) |
| **Non-Sickled (Clean)** | $T_{\text{control}}$ | 152 | **+0.0473** | **0.0620** | 0.9897 | **14.5%** (22 / 152) | **3.9%** (6 / 152) |
| **Sickled (Border)** | $T_{\text{border}}$ | 14 | -0.0020 | 0.0021 | 0.0218 | 0.0% | 0.0% |
| **Sickled (Clean)** | $T_{\text{control}}$ | 168 | -0.0049 | 0.0058 | 0.6758 | 0.6% | 0.6% (1 / 168) |

### B. Direct Source-Cluster Bootstrap Contrast (2,000 Resamples)

To test whether the border-removal effect exceeds the control edit, we directly estimate the contrast difference:
$$\Delta_{\text{contrast}} = \mathbb{E}[\Delta P \mid \text{Border}] - \mathbb{E}[\Delta P \mid \text{Clean}]$$
$$\Delta_{|\text{contrast}|} = \mathbb{E}[|\Delta P| \mid \text{Border}] - \mathbb{E}[|\Delta P| \mid \text{Clean}]$$

| Contrast Hypothesis | Point Estimate | 95% Source-Cluster Bootstrap CI | Null Value | Statistical Inference |
|---|---:|---:|:---:|---|
| **Directional Contrast Difference** | **-0.0026** | **[-0.0474, +0.0470]** | 0.0000 | **Cannot reject null ($p = 0.91$)** — Border removal does not produce a greater positive shift than the matched control edit. |
| **Absolute Contrast Difference** | **-0.0173** | **[-0.0640, +0.0339]** | 0.0000 | **Cannot reject null ($p = 0.49$)** — Perturbation magnitude is indistinguishable between border and clean crops. |

---

## 5. Stage 3D: Plasma Color & Illumination Equalization

Background color was equalized using both the empirical training-set median background RGB (`(216.13, 217.12, 209.12)`) and the fixed benchmark target (`(220.0, 218.0, 215.0)`):

| Intervention Condition | Evaluation Set | Mean $\Delta P(\text{Sickled})$ | Mean $|\Delta P|$ | 95% Source-Cluster CI |
|---|---|---:|---:|---:|
| **$T_{\text{plasma-exact}}$ (Training Median RGB)** | All Test Crops (n=404) | **+0.0011** | **0.0048** | [-0.001, +0.004] |
| **$T_{\text{plasma-fixed}}$ (Fixed Benchmark RGB)** | All Test Crops (n=404) | **+0.0044** | **0.0061** | [-0.001, +0.012] |
| **$T_{\text{photo-control}}$ (Luminance Rescaling)** | All Test Crops (n=404) | **+0.0010** | **0.0018** | [-0.000, +0.003] |

> **Key Observation**: Equalizing background plasma color to the exact training median shifted predictions by a negligible $+0.0011$ on average, demonstrating that the reference model does not rely on global plasma color variations to distinguish classes.

---

## 6. Failure Analysis: Why Do Individual Predictions Shift?

Although the average shift across all test crops is small, 32 of 222 non-sickled crops experienced $|\Delta P| > 0.10$. Diagnostic panels were rendered for the top 20 outliers in `experiment_3_reference_outputs/outlier_inspections/`:

1. **Peripheral Cellular Truncation**: In 80% of extreme outliers (16/20), the crop contained **no aperture border (border = 0)**. In these crops, neighboring red blood cells touched or extended into the 15% outer margin. Inpainting this margin clipped peripheral cellular structures, leaving partial or deformed erythrocyte silhouettes that the model misclassified.
2. **Contextual Inpainting Perturbations**: The inpainting operator is an intervention on the entire perimeter, not an isolated surgical removal of only dark aperture pixels. The observed variance reflects sensitivity to peripheral cellular information rather than selective reliance on the microscope aperture.

---

## 7. Pre-Registered Decision Synthesis

Referencing `EXPERIMENT_3_PROTOCOL.md`:
- **Directional Net Shift**: $\Delta_{\text{contrast}} = -0.0026$ (below the 0.05 threshold).
- **Decision Flip Rate**: 2.9% on border crops vs 3.9% on clean crops (both below 5.0%).
- **Plasma Shift**: $+0.0011$ (below 0.05 threshold).

**Final Synthesis**: The reference ResNet-18 model demonstrates **limited sensitivity to the tested border and plasma color interventions on average**. The model's high test accuracy is preserved on border-free crops and remains stable under color equalization. However, individual predictions can be perturbed when peripheral cellular context is modified, underscoring that average intervention insensitivity cannot be equated with complete proof of biological morphology learning.
