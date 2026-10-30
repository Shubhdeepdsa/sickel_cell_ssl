# Experiment 2: Confound & Shortcut Feature Analysis

## Executive Summary
**Core Question**: *How accurately can we predict the class of a microscopy crop when we deliberately restrict the classifier to image characteristics that are not supposed to determine sickle-cell morphology?*

This experiment rigorously evaluates three progressively structured tiers of non-morphological image signals across independent source-image partitions (`train`, `val`, `test`), enforcing strict separation of feature normalization, validation-based tuning, source-level threshold consistency, and source-cluster bootstrap uncertainty estimation.

### Key Findings Summary Table

| Experiment / Model Tier | Balanced Accuracy (Test Crops) | 95% Source-Cluster CI | Source-Level Bal. Acc. (th=0.50) | Source-Level Bal. Acc. (th=th*) | Sickled Recall (TPR) | Non-Sickled Spec (TNR) | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| **Majority / Random Reference** | 50.0% | — | 50.0% | 50.0% | 50.0% | 50.0% | Uninformative baseline reference level |
| **Tier 1A: Border-Only Rule** | **61.9%** | [55.7%, 68.5%] | — | — | 92.3% | 31.5% | Simple heuristic rule (if border -> non-sickled) |
| **Tier 1B: Border Logistic Reg.** | **61.9%** | [55.7%, 68.5%] | 73.3% | 50.0% | 92.3% | 31.5% | Univariate logistic model (th*=0.22) |
| **Tier 2: Perimeter Classifier** | **67.9%** | [60.7%, 75.1%] | **79.9%** | **79.6%** | 73.6% | 62.2% | Regularized L2 on 12 perimeter features (C=1.0, th*=0.51) |
| **Tier 3: Estimated Background** | **64.1%** | [57.8%, 70.3%] | **70.0%** | **67.2%** | 36.8% | 91.4% | Residual plasma color & luma after cell dilation (C=10.0, th*=0.55) |

### Primary Scientific Conclusion
1. **Perimeter-derived features achieve 67.9% balanced accuracy** on unseen source-separated test crops (**79.9% at source-image level under th=0.50**, and **79.6% under th*=0.51**), substantially exceeding random chance by **+17.9% percentage points** without examining the central cell body.
2. **Estimated background plasma features achieve 64.1% balanced accuracy** (**70.0% at source level under th=0.50**, and **67.2% under th*=0.55**) when cellular structures are masked and dilated.
3. **Conclusion**: Unintended macroscopic imaging artifacts (aperture border vignetting and background plasma staining/illumination differences) provide strong, learnable shortcut signals in this dataset.

---

## Part 1: Dataset Specification & Four Methodological Rules

### 1.1 Dataset Separation (Rule 1)
- **Measured Dataset**: Reconciled pure-Python 8-connected candidate pool consisting of **2,717 total crops** (1,249 Pool A sickled crops from 399 positive source images; 1,468 Pool B non-sickled crops from 144 negative source images).
- **Historical Dataset Contrast**: Explicitly separated from the historical 1,462-crop Stage 3 Pool B manifest. The 6-crop variance (0.4%) is documented and attributed to subpixel filter rounding in the legacy OpenCV pipeline.

### 1.2 No Test Set Tuning (Rule 2)
- All feature scalers (`StandardScaler`) were fitted **exclusively on the Train split** ($N = 1,931$).
- Hyperparameter regularization strength $C$ and probability decision thresholds were tuned **strictly on the Validation split** ($N = 382$).
- The Test partition ($N = 404$) was evaluated once as an exploratory assessment of shortcut signal availability.

### 1.3 Source-Cluster Aware Evaluation & Dual Threshold Reporting (Rule 3)
- Because multiple crops originate from the same microscope photograph, crops are not independent.
- Uncertainty bounds are computed via **2,000 source-cluster bootstrap iterations** (resampling unique source images with replacement).
- Performance is reported both at the **crop level** and at the **source-image level** (aggregating crop probabilities per source). To ensure full transparency and avoid threshold mismatch:
  - Source-level aggregation is reported both under standard balance ($th = 0.50$) and under the validation-tuned threshold ($th^*$).

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

- **Balanced Accuracy (Heuristic Rule)**: **61.9%** (95% source-cluster CI: **[55.7%, 68.5%]**)
- **Logistic Regression Model**: P(sickled | border) = sigmoid(0.22 + -1.52 * border) (optimal validation threshold $th^* = 0.22$).
  - Test Crop Balanced Accuracy: **61.9%**.
  - Source-Level Balanced Accuracy at $th = 0.50$: **73.3%** (Sickled TPR: 83.1%, Non-Sickled TNR: 63.6%).
  - Source-Level Balanced Accuracy at $th^* = 0.22$: **50.0%** (Sickled TPR: 100.0%, Non-Sickled TNR: 0.0%). Note: because $th^* = 0.22$ is aggressive, averaging non-border crops pushes nearly all sources above threshold.

---

## Part 3: Tier 2 — Perimeter-Derived Classification

> [!IMPORTANT]
> **Perimeter Feature Caveat & Cell Body Adjacency**:
> These features are explicitly designated **Perimeter-Derived Features**, not background-only features. In dense blood smears, adjacent or peripheral erythrocytes frequently impinge upon the outer 15% margin (19 pixels from edge). Furthermore, radial differential features compare this margin against an inner annulus that also intersects cellular cytoplasm and membrane edges. Consequently, Tier 2 captures both aperture border vignetting and peripheral cell-packing geometry, rather than purely non-cellular background.

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
- **Tuned Hyperparameters**: $C = 1.0$, Decision Threshold $th^* = 0.51$.
- **Crop-Level Balanced Accuracy**: **67.9%** (95% source-cluster CI: **[60.7%, 75.1%]**).
- **Source-Level Balanced Accuracy ($th = 0.50$)**: **79.9%** (46/59 positive sources, 18/22 negative sources).
- **Source-Level Balanced Accuracy ($th^* = 0.51$)**: **79.6%** (43/59 positive sources, 19/22 negative sources).
- **Crop Confusion Matrix**: TP=134, TN=138, FP=84, FN=48 (Sickled Recall = 73.6%, Non-Sickled Specificity = 62.2%).

---

## Part 4: Tier 3 — Estimated Background Features Experiment

Tier 3 evaluates whether classification remains possible when all identifiable cellular structures are segmented and dilated away.

### 4.1 Segmentation Method & Documented Visual Audit
- **Segmentation Protocol**: Foreground cells segmented via field-filtered Otsu thresholding on illuminated pixels ($>50$), followed by **5-pixel morphological dilation (11x11 structuring element)** to strictly eliminate cell membrane borders, diffraction fringes, and halo artifacts.
- **Background Retained**: Across all 2,717 crops, the mean background plasma coverage is **37.2%** (median **36.0%**).
- **Dense Field Limitation**: In **208 crops (7.7%)**, dense RBC packing left <10% background; in **0 crops (0.0%)**, <0.6% background was retained, triggering proxy fallback.

#### Visual Mask Inspection Audit (36 Randomly Sampled Crops)
To verify whether Otsu + 5px dilation truly excluded cell morphology, a systematic visual audit was conducted and logged in `experiment_2_confound_outputs/experiment_2_mask_inspection_audit.csv`:
- **Pristine (Clean Plasma)**: **23 / 36 (63.9%)** — Flawless plasma masking with zero erythrocyte remnants.
- **Acceptable (Minor Halo / Stained Background)**: **8 / 36 (22.2%)** — Adequate plasma masking with minor diffraction fringes from adjacent out-of-focus cells.
- **Compromised Masks**: **5 / 36 (13.9%)**, specifically:
  - **3 crops (8.3%)**: Faint erythrocyte central pallor or thin cytoplasmic membranes were missed by Otsu and leaked into the mask.
  - **2 crops (5.6%)**: Severe RBC clumping left <10% plasma, limiting background measurement reliability.
> [!NOTE]
> Because 13.9% of sampled masks had residual cell leakage or extreme clumping, these features are designated **Estimated Background Features** rather than 'pure background'.

### 4.2 Background Feature Set & Ablation
8 features computed exclusively from segmented, cell-free plasma pixels:

| Feature Name | Description | Model Weight ($w$) | Univariate Bal. Acc. | Univariate Sickled Recall | Univariate Non-Sickled Spec |
|---|---|---:|---:|---:|---:|
| `t3_bg_r_mean` | Estimated background plasma feature | +8.014 | 50.0% | 100.0% | 0.0% |
| `t3_bg_g_mean` | Estimated background plasma feature | -6.305 | 48.8% | 86.8% | 10.8% |
| `t3_bg_luma_mean` | Estimated background plasma feature | -2.863 | 50.0% | 0.0% | 100.0% |
| `t3_bg_rg_ratio` | Estimated background plasma feature | -1.723 | 53.0% | 33.5% | 72.5% |
| `t3_bg_sat_mean` | Estimated background plasma feature | +1.144 | 61.6% | 36.8% | 86.5% |
| `t3_bg_b_mean` | Estimated background plasma feature | +0.912 | 54.1% | 8.2% | 100.0% |
| `t3_bg_rb_ratio` | Estimated background plasma feature | -0.598 | 56.0% | 16.5% | 95.5% |
| `t3_bg_luma_std` | Estimated background plasma feature | -0.431 | 50.0% | 100.0% | 0.0% |

### 4.3 Estimated Background Classifier Test Results
- **Tuned Hyperparameters**: $C = 10.0$, Decision Threshold $th^* = 0.55$.
- **Crop-Level Balanced Accuracy**: **64.1%** (95% source-cluster CI: **[57.8%, 70.3%]**).
- **Source-Level Balanced Accuracy ($th = 0.50$)**: **70.0%** (29/59 positive sources, 20/22 negative sources).
- **Source-Level Balanced Accuracy ($th^* = 0.55$)**: **67.2%** (23/59 positive sources, 21/22 negative sources).
- **Crop Confusion Matrix**: TP=67, TN=203, FP=19, FN=115 (Sickled Recall = 36.8%, Non-Sickled Specificity = 91.4%).

---

## Part 5: Scientific Interpretation & Transition to Experiment 3

### 5.1 Synthesis of Findings: Shortcut Signal Availability & Confounding
1. **Predictive Shortcut Signals Exist in the Dataset**:
   - A single border flag achieves **61.9%** balanced accuracy.
   - Perimeter features achieve **67.9%** balanced accuracy (**79.9%** source-level).
   - Even estimated background plasma features achieve **64.1%** balanced accuracy (**70.0%** source-level).
2. **Mechanisms of Non-Morphological Predictability**:
   - **Geometric Asymmetry**: Asymmetric crop placement relative to the circular microscope aperture leaves dark vignette boundaries overwhelmingly in non-sickled crops (35.4% vs 10.0%).
   - **Photometric Differences**: Source images exhibit systematic differences in illumination intensity and staining color balance (R/G and R/B ratios) in the surrounding plasma.

### 5.2 Critical Scientific Distinction
- **Supported by Evidence**: Perimeter-derived and estimated background features contain statistically significant label-predictive signals in this reconstructed dataset. Simple linear models achieve 62%–80% balanced accuracy without analyzing the central erythrocyte.
- **Not Yet Established**: Whether the historical deep ResNet-18 models (which achieved 98–99% reported accuracy) actually relied on these shortcut signals, or whether their predictions were primarily driven by genuine cell morphology.

### 5.3 Clear Roadmap to Experiment 3
Experiment 2 directly motivates **Experiment 3**: *How do deep convolutional models behave when non-morphological shortcut features are systematically intervened upon or controlled?*
- **Intervention & Subgroup Strategy**:
  1. Stratify test evaluations on artifact-present vs artifact-absent subsets.
  2. Apply controlled perimeter neutralizations (masking the outer margin) alongside matched control interventions (e.g., masking corners of clean crops).
  3. Apply background plasma color and illumination equalization alongside matched control edits.
  4. Measure paired prediction probability shifts ($\Delta p$) at both the crop and source levels.
- **Realistic Scientific Scope**: Because neural networks exploit redundant visual cues and image interventions may introduce distribution shifts, Experiment 3 will not yield an exact percentage attribution of 'genuine vs shortcut', but will test whether the models are sensitive or robust to these non-morphological features.
