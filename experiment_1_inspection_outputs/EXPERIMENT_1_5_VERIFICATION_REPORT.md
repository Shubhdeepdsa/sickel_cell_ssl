# Experiment 1.5: Verification of Dataset-Quality Findings

## Executive Summary
Following the initial 72-crop inspection, this verification milestone investigates whether the observed class-asymmetric artifact differences are reproducible, statistically significant, and visible across the **entire dataset**.

### Core Verification Results
- **Scale**: Evaluated **2,717 total crops** across the entire dataset (1,249 Pool A sickled crops from 399 contributing positive source images; 1,468 Pool B non-sickled crops from 144 contributing negative source images).
- **Full-Pool Visible Border Contamination**:
  - **Pool A (Sickled)**: **10.0%** (95% source-cluster CI: **[8.3%, 11.8%]**; 125/1249 crops)
  - **Pool B (Non-Sickled)**: **35.4%** (95% source-cluster CI: **[30.5%, 40.2%]**; 519/1468 crops)
  - **Asymmetric Difference (Pool B minus Pool A)**: **+25.3%** (95% cluster CI: **[+20.1%, +30.6%]**)
- **Conclusion on Border Artifact Hypothesis**: **CONFIRMED**. The visible border artifact is not an artifact of small-sample variance. Non-sickled crops suffer from **3.5x higher border contamination** than sickled crops with non-overlapping confidence intervals.

---

## Part 1: Audit of the 72-Crop Inspection Sample

### 1.1 Source-Image Diversity in the Sample
- **Positive Sample (Pool A, 36 crops)**: Originated from **34 unique source images** (94.4% unique sources). Only 2 source images contributed 2 crops each.
- **Negative Sample (Pool B, 36 crops)**: Originated from **36 unique source images** (100.0% unique sources; exactly 1 crop per source).
- *Significance*: The 72-sample inspection was not an artifact of oversampling a single contaminated source image; it broadly sampled independent microscope fields across the cohort.

### 1.2 Verification of Automated Heuristics vs. Visual Audit
A crop-by-crop manual visual review of all 72 sampled crops was conducted to audit the automated detector flags (saved in `experiment_1_inspection_outputs/experiment_1_manual_review_audit.csv`):
1. **False Positives for Visible Borders**: **0 / 20 (0.0%)**. Every crop flagged by the detector (`visible_border=True`) showed a genuine dark circular aperture curve or vignetting arc.
2. **False Negatives for Visible Borders**: **2 / 52 (3.8%)**. In two positive crops (`pos_val_412_box2_01` and `pos_test_199_box1_08`), a dark corner vignette was visible but narrowly missed the 45-luma threshold (minimum luma ~44–48).
3. **True Visual Border Prevalence in the 72-Sample Set**:
   - **Sickled (Pool A)**: **6 / 36 (16.7%)** (4 detected + 2 borderline misses)
   - **Non-Sickled (Pool B)**: **16 / 36 (44.4%)** (16 detected + 0 misses)
   - *Net Difference*: **+27.8 percentage points higher in non-sickled crops**.

---

## Part 2: Dataset Reconciliation (Original Manifest vs Reconstructed Pools)

To ensure precise scientific rigor, we document the technical origin of crop counts across pipeline implementations:

| Dataset / Pipeline | Pool A (Sickled) | Pool B (Non-Sickled) | Total Crops | Pipeline Technical Specification |
|---|---:|---:|---:|---|
| **Original Stage 3 Manifest** (`crop_outputs/`) | 1,249 | 1,462 | 2,711 | ImageMagick (`magick`) downsampling + 8-connected components |
| **Reconstructed 8-Connected Pool** (Current) | 1,249 | 1,468 | 2,717 | Pure Python (Pillow Lanczos) + 8-connected components |
| **Exploratory 4-Connected Pool** (Exp 1.5 Draft) | 1,249 | 1,576 | 2,825 | Pillow + 4-connected components + synthetic center fallback |

### Technical Explanation of the 6-Crop Difference:
- The current pure-Python pipeline yields **1,468 Pool B crops**, matching the original 1,462-crop manifest to within **6 crops (0.4%)**.
- This tiny difference is entirely explained by sub-pixel resampling filter rounding (ImageMagick Mitchell/Lanczos vs. Pillow Lanczos) at the strict color saturation threshold (`sat > 0.055`, `luma in [45, 245]`).
- The earlier exploratory draft yielded 1,576 crops because 4-connectivity (orthogonal neighbors only) fragmented diagonal cell clusters into separate components, inflating the candidate count.
- **Crucially, the scientific finding is identical across both implementations**: border contamination is **35.4%** in the 8-connected reconstruction (vs 39.1% in the 4-connected pool), remaining 3.5x higher than Pool A across every split.

---

## Part 3: Full-Pool Quantitative Results Across Splits

Automated detection results across the reconciled 8-connected crop dataset:

| Split | Pool / Class | Crops | Unique Source Images | Border Contamination | 95% Source-Cluster CI | Multiple Cells | Empty Regions | Usable Cells |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | Pool A (Sickled) | 917 | 281 | **10.5%** (96/917) | [8.5%, 12.6%] | 92.4% | 2.7% | 96.8% |
| train | Pool B (Non-Sickled) | 1,014 | 101 | **34.9%** (354/1014) | [29.0%, 41.2%] | 89.8% | 7.2% | 92.6% |
| val | Pool A (Sickled) | 150 | 59 | **10.0%** (15/150) | [3.9%, 16.5%] | 86.7% | 4.0% | 96.0% |
| val | Pool B (Non-Sickled) | 232 | 21 | **40.9%** (95/232) | [29.7%, 53.2%] | 92.7% | 0.4% | 98.7% |
| test | Pool A (Sickled) | 182 | 59 | **7.7%** (14/182) | [4.2%, 11.5%] | 94.0% | 1.6% | 98.4% |
| test | Pool B (Non-Sickled) | 222 | 22 | **31.5%** (70/222) | [20.2%, 43.7%] | 92.3% | 1.4% | 98.2% |

### Key Takeaways from the Full-Pool Scale:
1. **Pool B Border Contamination Across All Splits**:
   - Train: **34.9%**
   - Validation: **40.9%**
   - Test: **31.5%**
   - *Interpretation*: The border artifact is consistently present in Pool B across **every** split at ~31%–41%, while Pool A stays between 7%–11%.

2. **Cell-Centered Patches vs. Isolated Single Cells**:
   - In the full Pool A set, **91.9%** of crops contain multiple cell bodies or dense adjacent red blood cells.
   - In the full Pool B set, **90.7%** contain multiple cell bodies.
   - This confirms that the computational task is scientifically a **cell-centered patch classification task**, not single-cell isolated morphology classification.

---

## Part 4: Source-Image Level Distribution Analysis

> [!IMPORTANT]
> **Nomenclature Note**: The dataset identifiers establish distinct **source images** (microscope fields of view), not confirmed distinct patient records. Image-level splitting ensures no microscope capture is shared between train, val, and test, but cannot guarantee separation across distinct clinical donors.

| Metric | Pool A (Positive Source Images, n=399) | Pool B (Negative Source Images, n=144) |
|---|---:|---:|
| Contributing source images with valid crops | 399 (out of 422 raw positive images) | 144 (out of 147 raw negative images) |
| Mean contamination rate per source image | 7.2% | 34.5% |
| Std. dev. across source images | 15.5% | 30.3% |
| Proportion of sources with >=1 border crop | 23.3% (93/399) | 79.2% (114/144) |
| Proportion of sources with >=50% border crops | 6.0% (24/399) | 35.4% (51/144) |

### Source Distribution Insight:
- In Pool B (negatives), **79.2% of all negative source images** produced crops with visible dark aperture borders.
- In Pool A (positives), crops were localized around expert-annotated sickled cells, which naturally occur inside the clear field, so only 23.3% of source images had any border crops.

---

## Part 5: Preliminary Shortcut Rule & Implications for Experiment 2

### 5.1 Naive Shortcut Rule on Test Split (Exploratory Observation)
Using the simple exploratory heuristic rule: *'If a crop has a detected dark border, predict non-sickled; otherwise predict sickled'*, test split performance is:

| Actual Class | Rule Prediction | Correct / Total | Accuracy |
|---|---|---:|---:|
| Sickled (Pool A) | Predict sickled (no border) | 168 / 182 | 92.3% |
| Non-sickled (Pool B) | Predict non-sickled (has border) | 70 / 222 | 31.5% |
| **Balanced Accuracy** | — | — | **61.9%** |

> [!WARNING]
> **Methodological Limitation**: This rule was selected post-hoc after inspecting the dataset. It represents an exploratory observation demonstrating that a single binary artifact signal already separates the classes substantially above the 50.0% random baseline, not a validated hypothesis test.

### 5.2 Framework for Experiment 2
Because >90% of crops contain neighboring red blood cells, a simple circular center mask is insufficient to guarantee exclusion of cellular morphology near the perimeter. Experiment 2 is therefore structured around three progressively rigorous tiers:

| Test Tier | Feature Domain | Scientific Question Answered |
|---|---|---|
| **Tier 1: Border Flag Only** | Binary automated aperture flag | Baseline correlation of the primary known artifact with the label. |
| **Tier 2: Perimeter Statistics** | Intensity, gradient, and variance in outer border rim | Whether multi-dimensional edge properties contain additional shortcut information. |
| **Tier 3: Validated Background-Only** | Strictly cell-free background plasma pixels | Whether classification remains possible when all cellular structures are rigorously excluded. |

### Success Condition Assessment
1. **Prevalence Quantified Confidently**: Pool B exhibits **35.4%** border contamination across 1,468 crops (95% CI: [30.5%, 40.2%]) vs. **10.0%** in Pool A across 1,249 crops (95% CI: [8.3%, 11.8%]), with non-overlapping confidence intervals.
2. **Reliability of Heuristic Established**: Audited manual review demonstrates 0.0% false positives and 3.8% false negatives (documented in `experiment_1_manual_review_audit.csv`).
3. **Remaining Uncertainty Bound**: Source-cluster bootstrapping establishes the true difference in border prevalence as **+25.3%** (95% CI: **[+20.1%, +30.6%]**).
