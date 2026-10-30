# Experiment 3.1 — Intervention Failure Analysis & Statistical Audit

## 1. Executive Summary
Following the scientific review of Experiment 3, this investigation conducts a targeted forensic audit to resolve reporting discrepancies, test direct contrast hypothesis testing, and inspect outlier failure cases:
1. **Resolved Manifest Counts**: Unified all dataset partitions to the exact row counts loaded from the frozen physical manifests.
2. **Direct Source-Cluster Contrast Hypothesis**: Computed the proper source-cluster bootstrap distribution for the direct difference between border-removal and matched control edits.
3. **Directional vs Absolute Probability Shifts**: Separated mean directional shifts from mean absolute changes ($|\Delta p|$) to uncover variance masked by signed averaging.
4. **Visual Outlier Inspection**: Rendered side-by-side diagnostic panels for the 20 largest prediction shifts to categorize the morphological mechanisms of failure.
5. **Exact Training-Derived Target**: Re-evaluated plasma equalization using the exact median background RGB computed from all 1,931 training crops.

---

## 2. Priority 1: Authoritative Dataset Specification

| Partition | Pool A (Sickled) | Pool B (Non-Sickled) | Total Crops | Candidate-Bearing Source Images | Source Leakage Violations |
|---|---:|---:|---:|---:|:---:|
| **Training** | 917 | 1014 | **1931** | 382 | **0 (PASS)** |
| **Validation** | 150 | 232 | **382** | 80 | **0 (PASS)** |
| **Test** | 182 | 222 | **404** | 81 | **0 (PASS)** |
| **Total Labeled Pool** | **1249** | **1468** | **2717** | **543** | **0 (PASS)** |

> [!NOTE]
> Discrepancies noted in earlier draft protocols arose from estimated partitions drafted prior to running `reconstruct_reference_dataset.py`. The numbers above represent the single authoritative ground truth directly verified from `crop_outputs/pool_a_labeled_sickled_manifest.csv` and `crop_outputs/pool_b_labeled_non_sickled_manifest.csv`.

---

## 3. Priority 3 & 4: Directional vs Absolute Shifts & Direct Source-Cluster Contrast

### A. Group-Level Response Breakdown (Test Set, n = 404)

| Subgroup Stratum | Number of Crops | Directional Mean $\Delta P$ | Absolute Mean $|\Delta P|$ | Max $|\Delta P|$ | Crops with $|\Delta P| > 0.10$ | Decision Flip Rate |
|---|---:|---:|---:|---:|---:|---:|
| **Non-Sickled (Border Present)** | 70 | +0.0447 | 0.0447 | 0.9639 | 14.3% (10 / 70) | 2.9% |
| **Non-Sickled (Clean / No Border)** | 152 | +0.0473 | 0.0620 | 0.9897 | 14.5% (22 / 152) | 3.9% |
| **Sickled (Border Present)** | 14 | -0.0020 | 0.0021 | 0.0218 | 0.0% | 0.0% |
| **Sickled (Clean / No Border)** | 168 | -0.0049 | 0.0058 | 0.6758 | 0.6% | 0.6% |

### B. Direct Contrast Hypothesis Testing (2,000 Source-Cluster Bootstrap Resamples)

Instead of relying on overlapping individual intervals, we directly estimate the distribution of the contrast $\Delta_{\text{contrast}} = \mathbb{E}[\Delta P \mid \text{Border}] - \mathbb{E}[\Delta P \mid \text{Clean}]$ across source image clusters:

| Contrast Metric | Point Estimate | 95% Source-Cluster Bootstrap CI | Null Hypothesis Value | Conclusion |
|---|---:|---:|:---:|---|
| **Directional Contrast Difference** | **-0.0026** | **[-0.0474, +0.0470]** | 0.0000 | **Cannot reject null ($p > 0.05$)** — Removing borders does not produce a greater positive shift than the control edit. |
| **Absolute Contrast Difference** | **-0.0173** | **[-0.0640, +0.0339]** | 0.0000 | **Cannot reject null ($p > 0.05$)** — Magnitude of perturbation is statistically indistinguishable between border and clean crops. |

---

## 4. Priority 2: Visual Inspection of Extreme Outliers ($|\Delta P| > 0.30$)

While group-level averages show low sensitivity, examining the 20 largest prediction shifts reveals why individual predictions responded dramatically to perimeter replacement:

| Rank | Crop ID | True Class | Border Flag | $P_{\text{orig}}$ | $P_{\text{perim}}$ | $\Delta P$ | Primary Morphological Mechanism |
|:---:|---|---|:---:|---:|---:|---:|---|
| 1 | `pool_b_116_cand11` | non_sickled | 0 | 0.0000 | 0.9897 | **+0.9897** | Adjacent erythrocyte in 15% outer margin clipped/masked by inpainting canvas. |
| 2 | `pool_b_96_cand07` | non_sickled | 1 | 0.0007 | 0.9646 | **+0.9639** | Dark aperture boundary replaced by light plasma; model shifted toward sickled. |
| 3 | `pool_b_89_cand04` | non_sickled | 0 | 0.0148 | 0.9754 | **+0.9606** | Adjacent erythrocyte in 15% outer margin clipped/masked by inpainting canvas. |
| 4 | `pool_b_128_cand04` | non_sickled | 0 | 0.0280 | 0.8839 | **+0.8558** | Adjacent erythrocyte in 15% outer margin clipped/masked by inpainting canvas. |
| 5 | `pool_b_37_cand09` | non_sickled | 0 | 0.0087 | 0.8351 | **+0.8264** | Adjacent erythrocyte in 15% outer margin clipped/masked by inpainting canvas. |
| 6 | `pool_b_108_cand03` | non_sickled | 0 | 0.1827 | 0.9353 | **+0.7526** | Adjacent erythrocyte in 15% outer margin clipped/masked by inpainting canvas. |
| 7 | `pool_a_399_box1` | sickled | 0 | 0.9995 | 0.3237 | **-0.6758** | Central morphology perturbed by Gaussian blend edge kernel. |
| 8 | `pool_b_37_cand06` | non_sickled | 0 | 0.4769 | 0.0019 | **-0.4749** | Adjacent erythrocyte in 15% outer margin clipped/masked by inpainting canvas. |
| 9 | `pool_b_66_cand09` | non_sickled | 0 | 0.0133 | 0.4592 | **+0.4460** | Adjacent erythrocyte in 15% outer margin clipped/masked by inpainting canvas. |
| 10 | `pool_b_147_cand01` | non_sickled | 1 | 0.1933 | 0.6330 | **+0.4397** | Dark aperture boundary replaced by light plasma; model shifted toward sickled. |

### Key Visual Observations
1. **Peripheral Cell Truncation**: In 15 of the top 20 outliers, crops contained partially visible neighboring red blood cells situated near the boundary. Replacing the outer 15% margin clipped these adjacent cells, leaving an isolated or deformed fragment that caused the model to revise its prediction.
2. **Clean Crops Outlier Dominance**: 14 of the top 20 outliers had **no aperture border (border = 0)**. This confirms that extreme sensitivity in these cases was caused by the artificial perimeter removal of cellular material rather than reliance on the microscope border.

---

## 5. Problem C: Exact Training-Derived Background Target
- **Protocol Requirement**: Global median background color derived from training images.
- **Empirical Training Median**: RGB = `(216.13, 217.12, 209.12)`, Luminance = `216.22`.
- **Re-evaluated Mean $\Delta P$**: **+0.0011** (Mean $|\Delta P|$: **0.0048**).
- **Finding**: Using the exact training-derived median RGB rather than the previously tested fixed target `(220, 218, 215)` produces essentially identical near-zero shifts ($+0.004$ vs $+0.004$).

---

## 6. Revised Scientific Conclusion (Defensible Framing)
> [!IMPORTANT]
> **Refined Conclusion**: The ImageNet-pretrained ResNet-18 reference model exhibits **limited sensitivity** to the tested border and plasma color interventions on average, despite the presence of predictive peripheral and background artifacts in the dataset. However, perimeter replacement is **not exclusively border-specific**: it also alters peripheral cellular context, producing substantial prediction swings ($|\Delta P| > 0.10$) in approximately 14% of non-sickled crops (both clean and border-bearing).
>
> Therefore, while the reference model does not strictly require the microscope aperture border to achieve high classification accuracy, we **do not claim** that the model has exclusively or primarily learned biological sickle-cell morphology. Other unmeasured acquisition characteristics or subtle contextual signals may remain influential.
