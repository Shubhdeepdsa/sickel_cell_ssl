# Experiment 3 Protocol: Controlled Intervention & Shortcut Reliance Analysis

**Status**: Frozen Pre-Registration Protocol  
**Date**: October 2026  
**Authors**: Antigravity Research Pair  
**Branch**: `experiment-2-confound-analysis` (pre-execution protocol freeze)

---

## 1. Executive Summary & Scientific Purpose

In **Experiment 1 & 1.5**, we established that dark microscope aperture borders occur with marked asymmetry in the sickle-cell dataset (35.4% in non-sickled Pool B crops vs. 10.0% in sickled Pool A crops). In **Experiment 2**, we demonstrated that linear classifiers restricted solely to perimeter statistics or estimated background plasma features achieve **67.9%** and **64.1%** balanced accuracy on test crops (**79.9%** and **70.0%** at the source-image level under default 0.50 threshold), confirming that strong non-morphological shortcut signals exist in the dataset.

However, linear shortcut *availability* in the data does not prove that deep convolutional neural networks (e.g., ResNet-18) *actually relied* on those shortcuts to achieve their reported 98–99% accuracy. Deep networks may have learned robust morphological invariants (e.g., sickle elongation, pointed tips, loss of central pallor), ignoring perimeter and background variations.

**Experiment 3** is designed to test causal reliance directly. It answers the central scientific question:
> **Core Question**: *Does intervening upon or stratifying by non-morphological image artifacts (aperture borders, perimeter pixels, and background plasma) systematically alter the predictions of convolutional neural networks trained on this dataset?*

To prevent confirmation bias, post-hoc metric hacking, and uncontrolled confounding, this protocol freezes all experimental stages, datasets, intervention operators, control baselines, and statistical test criteria **prior to running model inference**.

---

## 2. Stage 3A: Model Lineage, Provenance & Checkpoint Verification

### 2.1 Checkpoint Audit Findings
A comprehensive disk and git history audit was conducted across the workstation:
- **Search Scope**: All `.pt`, `.pth`, and `best_model.pt` binaries across local paths, Spotlight indexes, and git object history.
- **Result**: Zero historical checkpoint binaries were found. Historical execution logs (`interpretability_outputs/stage5_grayscale/run_manifest.json`) confirm that Stage 5 models were originally trained under `/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/stage5_grayscale_outputs`, a directory that does not exist on this environment. Furthermore, `.gitignore` explicitly excludes `stage5_outputs/` and `stage6_outputs/`.
- **Methodological Principle**: We will **never** silently retrain a new model and falsely present it as the historical original.

### 2.2 Dual Execution Paths
Experiment 3 accommodates two transparent pathways:

| Execution Path | Description | Prerequisites & Verification |
|---|---|---|
| **Option A: Historical Checkpoint Recovery** | Evaluation of historical checkpoint weights from the original study. | Checkpoint `.pt` file provided; SHA256 logged; test-set baseline accuracy must match historical report (98–99%). |
| **Option B: Transparent Seed-Locked Reproduction** | Transparent, clean retraining of a reference ResNet-18 using `train_stage5.py` under strictly locked random seeds (e.g., seed 42). | Explicitly designated as **"Stage 5 ResNet-18 Reproduction Surrogate"** in all figures, tables, and reports; full training manifest and convergence logs published. |

If Option A checkpoints become available, they will be evaluated under this exact protocol. Otherwise, Option B will serve as the reference target.

---

## 3. Stage 3B: Non-Interventional Subgroup Stratification

Before applying any synthetic image transformations, we perform a purely observational subgroup analysis on unmodified test crops ($N = 404$).

### 3.1 Subgroup Partitions
Test crops will be partitioned into four disjoint strata based on class label and the automated border detector:
1. **Stratum 1 (Sickled, Clean / No Border)**: Pool A test crops without aperture borders ($N \approx 168$).
2. **Stratum 2 (Sickled, Border Present)**: Pool A test crops with aperture borders ($N \approx 14$).
3. **Stratum 3 (Non-Sickled, Clean / No Border)**: Pool B test crops without aperture borders ($N \approx 152$).
4. **Stratum 4 (Non-Sickled, Border Present)**: Pool B test crops with aperture borders ($N \approx 70$).

### 3.2 Evaluation Metrics
For each stratum:
- Model Accuracy and Error Rate (FPR / FNR).
- Mean Predicted Probability of Sickled Class: $\bar{p} = \frac{1}{|S|} \sum_{i \in S} p(\text{sickled} \mid x_i)$.
- Prediction Entropy / Uncertainty: $\bar{H} = -\frac{1}{|S|} \sum_{i \in S} [p \log_2 p + (1-p) \log_2 (1-p)]$.

### 3.3 Subgroup Hypothesis Test
- **Confound Reliance Signature**: If the model relies on borders to predict non-sickled, then in Stratum 3 (clean non-sickled crops), the model will exhibit higher error rates and higher predicted sickled probability $\bar{p}$ compared to Stratum 4 (border-present non-sickled crops).
- **Morphological Invariance Signature**: If the model relies on cell morphology, error rates and mean probabilities in Stratum 3 and Stratum 4 will not show a statistically significant difference beyond sample size expectations.

---

## 4. Stage 3C: Controlled Perimeter & Border Interventions

Stage 3C tests the causal effect of removing dark aperture borders from crops.

### 4.1 Intervention Operator ($T_{\text{border}}$)
For any crop exhibiting an aperture border:
- **Intervention Mask ($M_{\text{perim}}$)**: The outer 15% perimeter ring (19 pixels from each edge) or the detected border component mask.
- **Inpainting / Replacement Strategy**: Inpaint the border region with the median color and luminance of the crop's segmented clear plasma background, applying a Gaussian edge blend ($\sigma = 1.5$) to prevent high-frequency artificial boundary step-discontinuities.
- **Output**: Cleaned crop $x^{\text{border-neutralized}}$.

### 4.2 Matched Control Operator ($T_{\text{control}}$)
Synthetic modifications inherently alter image statistics. To separate the causal effect of border removal from general distribution-shift degradation, we apply matched negative control edits to **clean, border-free crops**:
- **Control Intervention**: Apply the identical perimeter replacement operator (or matched 19-pixel corner masking) to clean crops from both Pool A and Pool B.
- **Output**: Control-perturbed crop $x^{\text{control}}$.

### 4.3 Causal Shift Metric
For each crop $i$, we compute the paired shift in predicted probability of the sickled class:
$$\Delta p_i = p(\text{sickled} \mid T(x_i)) - p(\text{sickled} \mid x_i)$$

- **Hypothesis Test**: 
  $$\Delta \bar{p}_{\text{border-present, non-sickled}} \gg \Delta \bar{p}_{\text{control, non-sickled}}$$
  A meaningful shortcut reliance is detected if and only if neutralizing the border shifts non-sickled predictions toward sickled significantly more than the control perturbation.

---

## 5. Stage 3D: Background Plasma Illumination & Staining Equalization

Experiment 2 established that clear plasma background features (mean R, G, B, and R/B ratios) achieve 64.1% balanced accuracy. Stage 3D tests whether models rely on background tint.

### 5.1 Photometric Equalization Operator ($T_{\text{plasma}}$)
1. Using the Otsu + 5px dilation background mask, calculate the mean background RGB vector for the source image: $\boldsymbol{\mu}_{\text{bg}} = (\bar{R}_{\text{bg}}, \bar{G}_{\text{bg}}, \bar{B}_{\text{bg}})$.
2. Compute the target global median background vector across all training images: $\boldsymbol{\mu}_{\text{target}}$.
3. Compute the multiplicative channel correction factor: $\mathbf{s} = \boldsymbol{\mu}_{\text{target}} / \boldsymbol{\mu}_{\text{bg}}$.
4. Apply the color correction globally to the crop: $x^{\text{equalized}} = \text{clip}(x \odot \mathbf{s}, 0, 255)$.

### 5.2 Photometric Control Operator ($T_{\text{photo-control}}$)
Apply a neutral luminance-matched shift (rescaling luminance without altering the R/G or R/B color ratio) to verify that changes are not simply due to subtle brightness scaling.

### 5.3 Causal Shift Metric
Track $\Delta p_i$ under plasma equalization versus neutral brightness control.

---

## 6. Stage 3E: Paired Shift & Source-Cluster Statistical Analysis

### 6.1 Two-Level Aggregation
1. **Crop-Level Paired Analysis**:
   - Fraction of categorical prediction flips: $F_{\text{flip}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}(T(x_i)) \neq \hat{y}(x_i))$.
   - Directional flip rate: Non-sickled $\to$ Sickled vs. Sickled $\to$ Non-sickled.
   - Paired Wilcoxon signed-rank test on probability shifts $\Delta p_i$.
2. **Source-Image Level Aggregation**:
   - Compute mean probability shift per source image $j$: $\Delta P_j = \frac{1}{|K_j|} \sum_{i \in K_j} \Delta p_i$.
   - Evaluate whether source-level diagnosis flips under standard ($th = 0.50$) and validation-tuned ($th^*$) thresholds.

### 6.2 Resampling & Uncertainty Bounds
- All confidence intervals for flip rates and metric shifts are calculated using **2,000 source-cluster bootstrap iterations**, resampling entire source image clusters with replacement.

---

## 7. Pre-Registered Hypotheses & Interpretation Matrix

To maintain scientific integrity, the outcome interpretation is pre-registered as follows:

| Outcome Pattern | Observation | Scientific Conclusion |
|---|---|---|
| **Pattern 1: Substantial Shortcut Reliance** | Neutralizing borders on non-sickled crops causes significant positive shift $\Delta \bar{p} > +0.20$ and high flip rate ($>20\%$), well exceeding matched control edits. Clean test accuracy drops significantly. | **Confound Reliance Confirmed**: The CNN heavily exploited macroscopic image shortcuts rather than relying purely on cell morphology. High historical performance was partially artifact-driven. |
| **Pattern 2: Moderate Confound Sensitivity** | Border neutralization and plasma equalization produce modest probability shifts ($+0.05 \le \Delta \bar{p} \le +0.15$), but categorical predictions rarely flip (<5%). | **Partial Reliance / Redundant Representations**: The model exhibits sensitivity to background/perimeter cues, but cellular morphological features are sufficiently strong to maintain correct classification. |
| **Pattern 3: Complete Morphological Invariance (Null Hypothesis)** | Probability shifts on intervened images are negligible ($|\Delta \bar{p}| < 0.05$) and do not differ from matched negative control perturbations. Clean subgroup accuracy is identical to contaminated subgroup accuracy. | **Shortcut Invariance Confirmed**: Despite linear confound availability in the dataset, deep convolutional representations successfully extracted cell-shape invariants and ignored macroscopic artifacts. High performance is genuinely morphologically grounded. |

---

## 8. Preserved Caveats & Redundancy Constraints

1. **Synthetic Perturbation Shift**: Inpainting or masking introduces artificial boundaries. Therefore, any shift observed in intervened crops must always be benchmarked against the matched negative control operator.
2. **Feature Redundancy in Deep Networks**: Because deep neural networks learn highly overcomplete feature representations, an unchanged prediction after border removal does not conclusively prove the feature was never used; it indicates the model possesses sufficient redundant morphological cues to maintain its decision.
3. **No Claim of Exact Percentage Split**: We pre-register that Experiment 3 will measure *sensitivity bounds and directional probability shifts*, rather than claiming an exact single number for "genuine vs artifact-driven accuracy".
