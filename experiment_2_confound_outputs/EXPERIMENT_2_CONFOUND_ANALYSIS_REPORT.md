# Experiment 2: Confound & Shortcut Feature Analysis

## Executive Summary
**Core Question**: *How accurately can we predict the class of a microscopy crop when we deliberately restrict the classifier to image characteristics that are not supposed to determine sickle-cell morphology?*

This experiment rigorously evaluates three progressively structured tiers of non-morphological image signals across independent source-image partitions (`train`, `val`, `test`), enforcing strict separation of feature normalization, validation-based tuning, and source-cluster bootstrap uncertainty estimation.

### Key Findings Summary Table

| Experiment / Model Tier | Balanced Accuracy (Test) | 95% Source-Cluster CI | Source-Level Bal. Acc. | Sickled Recall (TPR) | Non-Sickled Spec (TNR) | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| **Majority / Random Reference** | 50.0% | — | 50.0% | 50.0% | 50.0% | Uninformative baseline reference level |
| **Tier 1A: Border-Only Rule** | **61.9%** | [55.8%, 68.4%] | — | 92.3% | 31.5% | Simple heuristic rule (if border -> non-sickled) |
| **Tier 1B: Border-Only Logistic Reg.** | **61.9%** | [55.8%, 68.4%] | 73.3% | 92.3% | 31.5% | Train-fitted univariate logistic model (th=0.22) |
| **Tier 2: Perimeter Classifier** | **67.9%** | [60.9%, 75.1%] | **79.9%** | 73.6% | 62.2% | Regularized L2 model on 12 outer-ring edge features (C=1.0) |
| **Tier 3: Background-Only Classifier** | **64.1%** | [57.7%, 70.9%] | **70.0%** | 36.8% | 91.4% | Residual plasma color & luma after cell dilation (C=10.0) |

### Primary Scientific Conclusion
1. **Perimeter features achieve 67.9% balanced accuracy** on unseen source-separated test crops (**79.9% at source-image level**), substantially exceeding random chance by **+17.9% percentage points** without examining the central cell body.
2. **Validated background plasma features achieve 64.1% balanced accuracy** (**70.0% at source-image level**) even when all erythrocyte structures are segmented and dilated away.
3. **Conclusion**: Unintended macroscopic imaging artifacts (aperture border vignetting and background plasma staining/illumination differences) provide strong, learnable shortcuts. Any model trained on this dataset has access to predictive visual signals completely detached from sickle-cell morphology.

---

## Part 1: Dataset Specification & Four Methodological Rules

### 1.1 Dataset Separation (Rule 1)
- **Measured Dataset**: Reconciled pure-Python 8-connected candidate pool consisting of **2,717 total crops** (1,249 Pool A sickled crops from 399 positive source images; 1,468 Pool B non-sickled crops from 144 negative source images).
- **Historical Dataset Contrast**: Explicitly separated from the historical 1,462-crop Stage 3 Pool B manifest. The 6-crop variance (0.4%) is documented and attributed to subpixel filter rounding.

### 1.2 No Test Set Tuning (Rule 2)
- All feature scalers (`StandardScaler`) were fitted **exclusively on the Train split** ($N = 1,931$).
- Hyperparameter regularization strength $C$ and probability decision thresholds were tuned **strictly on the Validation split** ($N = 382$).
- The Test partition ($N = 404$) was evaluated once as a hypothesis test of shortcut availability.

### 1.3 Source-Cluster Aware Evaluation (Rule 3)
- Because multiple crops originate from the same microscope photograph, crops are not independent.
- Uncertainty bounds are computed via **2,000 source-cluster bootstrap iterations** (resampling unique source images with replacement).
- Performance is reported both at the **crop level** and at the **source-image level** (aggregating crop probabilities per source).

### 1.4 Preserved Audit Limitations (Rule 4)
- **Corrected Audit Error Rates**: From the Experiment 1.5 manual audit (72 crops), the detector false-negative rate is **9.1%** (2 missed out of 22 ground-truth border crops), while **3.8%** (2 out of 52 negative predictions) is the *False Omission Rate* ($FOR = FN / (TN + FN)$).
- Independent manual verification records are preserved in `experiment_1_inspection_outputs/experiment_1_manual_review_audit.csv`.

---

## Part 2: Tier 1 — Border Indicator Baseline

Tier 1 tests the predictive power of the primary known artifact: whether a crop intersects the dark circular microscope aperture or vignette rim.

### 2.1 Confusion Matrix & Performance (Test Partition, N = 404 crops)

| Actual Class | Predicted Sickled | Predicted Non-Sickled | Total Crops | Class Accuracy |
|---|---:|---:|---:|---:|
| **Sickled (Pool A)** | 168 (TN for border) | 14 (Border present) | 182 | 92.3% |
| **Non-Sickled (Pool B)** | 152 (No border) | 70 (Border present) | 222 | 31.5% |

- **Balanced Accuracy**: **61.9%** (95% source-cluster CI: **[55.8%, 68.4%]**)
- **Logistic Regression Model**: P(sickled | border) = sigmoid(0.22 + -1.52 * border) (optimal validation threshold = 0.22). Test Balanced Accuracy: **61.9%**.

---

## Part 3: Tier 2 — Perimeter Features Classifier

> [!IMPORTANT]
> **Nomenclature Note**: These are strictly designated **Perimeter Features**, NOT background-only features, because adjacent or clustered red blood cells can impinge upon the outer 15% crop boundary.

### 3.1 Perimeter Feature Set & Ablation
12 features extracted from the outer 15% margin (19 pixels from edge), normalized using training statistics:

| Feature Name | Description | Model Weight ($w$) | Univariate Bal. Acc. | Univariate Sickled Recall | Univariate Non-Sickled Spec |
|---|---|---:|---:|---:|---:|
| `t2_perim_luma_p10` | Perimeter feature | -0.990 | 56.4% | 78.0% | 34.7% |
| `t2_perim_radial_diff` | Perimeter feature | +0.966 | 65.5% | 94.5% | 36.5% |
| `t2_perim_corner_min` | Perimeter feature | +0.610 | 61.9% | 92.3% | 31.5% |
| `t2_perim_luma_min` | Perimeter feature | -0.529 | 62.3% | 91.8% | 32.9% |
| `t2_perim_g_mean` | Perimeter feature | +0.448 | 55.5% | 80.8% | 30.2% |
| `t2_perim_luma_mean` | Perimeter feature | +0.339 | 57.5% | 78.0% | 36.9% |
| `t2_perim_b_mean` | Perimeter feature | -0.326 | 55.7% | 80.8% | 30.6% |
| `t2_perim_r_mean` | Perimeter feature | +0.173 | 57.8% | 76.4% | 39.2% |
| `t2_perim_rb_ratio` | Perimeter feature | +0.128 | 62.0% | 30.8% | 93.2% |
| `t2_perim_luma_std` | Perimeter feature | -0.077 | 70.6% | 75.8% | 65.3% |
| `t2_perim_rg_ratio` | Perimeter feature | -0.055 | 52.4% | 23.6% | 81.1% |
| `t2_perim_grad_energy` | Perimeter feature | -0.042 | 63.5% | 65.4% | 61.7% |

### 3.2 Perimeter Classifier Test Results
- **Tuned Hyperparameters**: $C = 1.0$, Decision Threshold = 0.51.
- **Crop-Level Balanced Accuracy**: **67.9%** (95% source-cluster CI: **[60.9%, 75.1%]**).
- **Source-Image Level Balanced Accuracy**: **79.9%** (46/59 positive sources, 18/22 negative sources correctly classified).
- **Confusion Matrix**: TP=134, TN=138, FP=84, FN=48 (Sickled Recall = 73.6%, Non-Sickled Specificity = 62.2%).

---

## Part 4: Tier 3 — Validated Background-Only Experiment

Tier 3 evaluates whether classification remains possible when all cellular structures (central cell and peripheral RBC bodies) are segmented and dilated away.

### 4.1 Segmentation Method & Quality Audit
- **Segmentation Protocol**: Foreground cells segmented via field-filtered Otsu thresholding on illuminated pixels ($>50$), followed by **5-pixel morphological dilation (11x11 structuring element)** to strictly eliminate cell membrane borders, diffraction fringes, and halo artifacts.
- **Background Retained**: Across all 2,717 crops, the mean background plasma coverage is **37.2%** (median **36.0%**).
- **Mask Limitation Reporting**: In **208 crops (7.7%)**, dense RBC packing left <10% background; in **0 crops (0.0%)**, <0.6% background was retained, triggering proxy fallback.
- A detailed 36-crop manual mask audit is saved in `experiment_2_confound_outputs/experiment_2_mask_inspection_audit.csv`.

### 4.2 Background Feature Set & Ablation
8 features computed exclusively from segmented, cell-free plasma pixels:

| Feature Name | Description | Model Weight ($w$) | Univariate Bal. Acc. | Univariate Sickled Recall | Univariate Non-Sickled Spec |
|---|---|---:|---:|---:|---:|
| `t3_bg_r_mean` | Cell-free plasma feature | +8.014 | 50.0% | 100.0% | 0.0% |
| `t3_bg_g_mean` | Cell-free plasma feature | -6.305 | 48.8% | 86.8% | 10.8% |
| `t3_bg_luma_mean` | Cell-free plasma feature | -2.863 | 50.0% | 0.0% | 100.0% |
| `t3_bg_rg_ratio` | Cell-free plasma feature | -1.723 | 53.0% | 33.5% | 72.5% |
| `t3_bg_sat_mean` | Cell-free plasma feature | +1.144 | 61.6% | 36.8% | 86.5% |
| `t3_bg_b_mean` | Cell-free plasma feature | +0.912 | 54.1% | 8.2% | 100.0% |
| `t3_bg_rb_ratio` | Cell-free plasma feature | -0.598 | 56.0% | 16.5% | 95.5% |
| `t3_bg_luma_std` | Cell-free plasma feature | -0.431 | 50.0% | 100.0% | 0.0% |

### 4.3 Background Classifier Test Results
- **Tuned Hyperparameters**: $C = 10.0$, Decision Threshold = 0.55.
- **Crop-Level Balanced Accuracy**: **64.1%** (95% source-cluster CI: **[57.7%, 70.9%]**).
- **Source-Image Level Balanced Accuracy**: **70.0%** (29/59 positive sources, 20/22 negative sources).
- **Confusion Matrix**: TP=67, TN=203, FP=19, FN=115.

---

## Part 5: Scientific Interpretation & Transition to Experiment 3

### 5.1 Synthesis of Findings
1. **Information Leakage Proven**: Non-morphological image characteristics contain substantial class-predictive information.
   - A single border flag achieves **61.9%** balanced accuracy.
   - Perimeter features achieve **67.9%** balanced accuracy (**79.9%** source-level).
   - Even strictly non-cellular plasma features achieve **64.1%** balanced accuracy (**70.0%** source-level).
2. **The Mechanism of Shortcut Availability**:
   - **Geometric**: Asymmetric crop placement relative to the circular field of view leaves dark aperture borders overwhelmingly in non-sickled crops.
   - **Photometric**: Positive source images and negative source images exhibit systematic differences in illumination level and staining color balance (R/G and R/B ratios) in the clear background plasma.

### 5.2 Critical Scientific Distinction
- **What Experiment 2 has established**: Unintended shortcut signals are easily accessible in the dataset. Simple linear models achieve 62%–80%+ balanced accuracy without ever inspecting the central sickle-cell morphology.
- **What Experiment 2 has NOT established**: Whether the original deep ResNet models (which achieved 98–99% accuracy) relied predominantly on these shortcuts or genuinely learned fine-grained sickle-cell morphology.

### 5.3 Clear Roadmap to Experiment 3
Experiment 2 directly sets up **Experiment 3**: *Does removing or equalizing these non-morphological artifacts change the predictions of the original high-performing ResNet models?*
- Specifically, by masking borders, equalizing plasma color temperature, or evaluating models on artifact-free vs artifact-contaminated test subsets, we can definitively establish how much of the original 98–99% score was genuine vs artifact-driven.
