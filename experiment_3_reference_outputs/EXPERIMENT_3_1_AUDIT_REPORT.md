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

| Contrast Metric | Point Estimate | 95% Source-Cluster Bootstrap CI | Interpretation as Uncertainty Bound |
|---|---:|---:|---|
| **Directional Contrast Difference** | **-0.0026** | **[-0.0474, +0.0470]** | Point estimate is near zero and the interval spans zero; no clear difference detected, though modest effects up to $\pm 0.05$ cannot be excluded. |
| **Absolute Contrast Difference** | **-0.0173** | **[-0.0640, +0.0339]** | Uncertainty bounds span zero; perturbation magnitude shows no clear differential effect between border and clean crops. |

> [!NOTE]
> **Statistical Boundary**: This source-cluster bootstrap interval reflects observational variation across source image clusters between different crops. Failure to detect a difference does not establish practical equivalence. Proving equivalence would require pre-specifying an equivalence margin $\delta$ and demonstrating that the confidence bounds fall strictly within $\pm\delta$.

---

## 4. Priority 2: Adjudicated Visual Inspection of Top 20 Outliers

Rather than relying on automated metadata flags, each of the top 20 outlier diagnostic panels was individually reviewed to record empirical image content, peripheral modifications, and candidate failure mechanisms (recorded in `outlier_visual_audit_adjudication.csv`):

| Rank | Crop ID | True Class | Border Flag | $P_{\text{orig}}$ | $P_{\text{perim}}$ | $\Delta P$ | Observed Image Content & Margin Modification | Adjudicated Category | Confidence |
|:---:|---|---|:---:|---:|---:|---:|---|---|:---:|
| 1 | `pool_b_116_cand11` | non_sickled | 0 | 0.0000 | 0.9897 | **+0.9897** | empty_background_plasma; natural_plasma_texture_replaced_by_synthetic_canvas | `empty_plasma_crop` | High |
| 2 | `pool_b_96_cand07` | non_sickled | 1 | 0.0007 | 0.9646 | **+0.9639** | single_erythrocyte_central; dark_microscope_edge_erased_and_replaced_with_plasma | `aperture_border_removal` | High |
| 3 | `pool_b_89_cand04` | non_sickled | 0 | 0.0148 | 0.9754 | **+0.9606** | single_erythrocyte_central; neighboring_erythrocyte_truncated_by_15pct_canvas | `peripheral_cell_clipping` | High |
| 4 | `pool_b_128_cand04` | non_sickled | 0 | 0.0280 | 0.8839 | **+0.8558** | dense_central_erythrocyte; multiple_adjacent_cells_truncated_on_three_edges | `peripheral_cell_clipping` | High |
| 5 | `pool_b_37_cand09` | non_sickled | 0 | 0.0087 | 0.8351 | **+0.8264** | small_erythrocyte_fragment; left_margin_cell_contour_truncated | `peripheral_cell_clipping` | Medium |
| 6 | `pool_b_108_cand03` | non_sickled | 0 | 0.1827 | 0.9353 | **+0.7526** | central_cell_with_debris; dark_peripheral_material_and_adjacent_cells_erased | `subthreshold_border_and_crowding` | Medium |
| 7 | `pool_a_399_box1` | sickled | 0 | 0.9995 | 0.3237 | **-0.6758** | faint_sickled_cell_green_tint; outer_gradient_blended_into_synthetic_canvas | `faint_crop_boundary_erasure` | High |
| 8 | `pool_b_37_cand06` | non_sickled | 0 | 0.4769 | 0.0019 | **-0.4749** | central_erythrocyte_with_peripheral_clutter; peripheral_debris_and_edge_noise_erased | `threshold_boundary_case` | High |
| 9 | `pool_b_66_cand09` | non_sickled | 0 | 0.0133 | 0.4592 | **+0.4460** | small_erythrocyte_central; top_erythrocyte_contour_clipped_by_canvas | `peripheral_cell_clipping` | High |
| 10 | `pool_b_147_cand01` | non_sickled | 1 | 0.1933 | 0.6330 | **+0.4397** | central_erythrocyte; heavy_dark_aperture_boundary_erased_and_replaced | `aperture_border_removal` | High |
| 11 | `pool_b_108_cand02` | non_sickled | 0 | 0.1034 | 0.5383 | **+0.4349** | central_erythrocyte; adjacent_erythrocyte_edges_clipped_on_four_sides | `peripheral_cell_clipping` | High |
| 12 | `pool_b_96_cand10` | non_sickled | 1 | 0.0000 | 0.4063 | **+0.4063** | single_erythrocyte; aperture_edge_erased_with_plasma_canvas | `aperture_border_removal` | High |
| 13 | `pool_b_133_cand01` | non_sickled | 0 | 0.0011 | 0.3641 | **+0.3630** | dense_overlapping_erythrocytes; overlapping_erythrocyte_bodies_truncated | `peripheral_cell_clipping` | High |
| 14 | `pool_b_108_cand04` | non_sickled | 0 | 0.0002 | 0.3523 | **+0.3521** | central_erythrocyte; adjacent_cell_edges_erased_by_canvas | `peripheral_cell_clipping` | High |
| 15 | `pool_b_108_cand12` | non_sickled | 0 | 0.0002 | 0.3313 | **+0.3311** | central_erythrocyte; adjacent_cell_edges_erased_by_canvas | `peripheral_cell_clipping` | High |
| 16 | `pool_b_122_cand04` | non_sickled | 0 | 0.0000 | 0.3059 | **+0.3059** | empty_background_plasma; natural_plasma_texture_replaced_by_synthetic_canvas | `empty_plasma_crop` | Medium |
| 17 | `pool_b_92_cand11` | non_sickled | 1 | 0.0002 | 0.2866 | **+0.2864** | central_erythrocyte; heavy_dark_aperture_edge_erased | `aperture_border_removal` | High |
| 18 | `pool_b_56_cand12` | non_sickled | 0 | 0.2757 | 0.0016 | **-0.2741** | faint_erythrocyte_central; subtle_peripheral_texture_cleaned_up | `threshold_boundary_case` | Medium |
| 19 | `pool_b_49_cand10` | non_sickled | 0 | 0.0002 | 0.2633 | **+0.2632** | central_erythrocyte; bottom_adjacent_cell_contour_clipped | `peripheral_cell_clipping` | High |
| 20 | `pool_b_37_cand03` | non_sickled | 0 | 0.5054 | 0.2807 | **-0.2247** | central_erythrocyte_with_edge_shadow; edge_shadow_erased_with_clean_plasma | `threshold_boundary_case` | Medium |

### Adjudicated Taxonomy Summary Across Top 20 Outliers

| Adjudicated Failure / Response Category | Count | Proportion | Crops Affected | Empirical Visual Observations |
|---|---:|---:|---|---|
| **Peripheral Cell Clipping** | 9 | 45.0% | `pool_b_89_cand04`, `128_cand04`, `37_cand09`, `66_cand09`, `108_cand02`, `133_cand01`, `108_cand04`, `108_cand12`, `49_cand10` | Neighboring red blood cells or dense clusters graze or cross the 15% outer margin. Inpainting truncates adjacent cellular contours, altering effective silhouette aspect ratio. |
| **Aperture Border Removal** | 4 | 20.0% | `pool_b_96_cand07`, `147_cand01`, `96_cand10`, `92_cand11` | Genuine dark microscope aperture arc erased and replaced with plasma color canvas, causing prediction to shift toward sickled ($+0.28$ to $+0.96$). |
| **Threshold Boundary Cases Clarified** | 3 | 15.0% | `pool_b_37_cand06`, `pool_b_56_cand12`, `pool_b_37_cand03` | Originally borderline predictions ($P \approx 0.28 - 0.51$). Margin cleaning eliminated confusing peripheral noise/shadows, clarifying non-sickled morphology. |
| **Empty / Faint Plasma Background Crops** | 2 | 10.0% | `pool_b_116_cand11`, `pool_b_122_cand04` | Crop contains virtually no dark cell body (<1% dark pixels). Flat synthetic inpainting canvas induces artificial edge gradients on uniform plasma. |
| **Subthreshold Border and Crowding** | 1 | 5.0% | `pool_b_108_cand03` | Peripheral field contains subthreshold dark edge material and dense RBC crowding, both erased by canvas. |
| **Faint Crop Boundary Erasure** | 1 | 5.0% | `pool_a_399_box1` | Faint sickled crop with subtle boundary gradient; margin inpainting blended the anchoring edge contour. |

> [!IMPORTANT]
> **Factual Outlier Distribution**: Exactly **16 of the top 20 outliers (80.0%)** occurred on crops with **no aperture border (`border = 0`)**. Only 4 of the top 20 outliers possessed an aperture border. This empirical finding confirms that large individual prediction shifts are predominantly driven by general peripheral image modifications (especially adjacent cell truncation and synthetic blend boundaries) rather than isolated aperture border reliance.

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
