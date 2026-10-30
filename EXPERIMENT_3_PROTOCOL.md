# Experiment 3 Protocol: Controlled Intervention & Shortcut Reliance Analysis

**Protocol Type**: Frozen Exploratory Intervention Protocol  
**Reference Model**: ImageNet-Pretrained ResNet-18 Reference Model  
**Date**: October 2026  
**Authors**: Antigravity Research Pair  
**Branch**: `experiment-3-reference-model`

---

## 1. Executive Summary & Scientific Purpose

In **Experiment 1 & 1.5**, we established that dark microscope aperture borders occur with marked asymmetry in the sickle-cell dataset (35.4% in non-sickled Pool B crops vs. 10.0% in sickled Pool A crops). In **Experiment 2**, we demonstrated that linear classifiers restricted solely to perimeter statistics or estimated background plasma features achieve **67.9%** and **64.1%** balanced accuracy on test crops (**79.9%** and **70.0%** at the source-image level under default 0.50 threshold), confirming that strong non-morphological shortcut signals exist in the dataset.

However, linear shortcut *availability* in the data does not prove that deep convolutional neural networks (e.g., ResNet-18) *actually relied* on those shortcuts. Deep networks may have learned representations that are largely insensitive to perimeter and background variations under standard viewing conditions.

**Experiment 3** investigates the following scientific question:
> **Core Question**: *Does intervening upon or stratifying by non-morphological image artifacts (aperture borders, perimeter pixels, and background plasma) systematically alter the predictions of an ImageNet-pretrained ResNet-18 reference model trained on this dataset?*

### Critical Methodological Framing
1. **Newly Trained Reference Model**: Because historical training checkpoints from the original study were not recovered, we train a new reference model designated:
   $$\text{\textbf{Experiment 3 — ImageNet ResNet-18 Reference Model}}$$
   This model is trained specifically for this investigation. It is **not** claimed to be the historical model, nor is it assumed to reproduce the historical 98–99% performance unless that is empirically demonstrated.
2. **Prioritization of Initial Reference Architecture**: We focus initially on **ImageNet-pretrained ResNet-18 only**. Contrastive self-supervised pre-training (SimCLR) and training from scratch are deferred to keep computation tractable on Apple Silicon (M4, 16 GB RAM) and focus on the primary question.
3. **Exploratory Intervention Protocol**: Because earlier exploratory stages inspected the test partition, this protocol represents a **frozen exploratory intervention protocol**, not an independent confirmatory study on an unexamined distribution.
4. **Careful Causal Interpretation**:
   - An absence of prediction change after border or plasma intervention indicates **insensitivity to the tested interventions under the experimental conditions**, not definitive proof of "complete morphological invariance".
   - A large prediction shift following an inpainting intervention indicates **intervention sensitivity**, which requires verifying that cell morphology was preserved and that the intervention did not merely create an out-of-distribution image artifact.

---

## 2. Stage 1: Establish the Reference Dataset

### 2.1 Reconciled Manifests & Explicit Variance Tracking
- **Pool A (Sickled)**: 1,249 crops extracted from 399 positive patient source images.
- **Pool B (Non-Sickled)**: 1,468 pure-Python 8-connected candidate crops extracted from 144 negative source images.
- **Historical Dataset Contrast**: The historical Stage 3 Pool B manifest recorded 1,462 crops. The 6-crop variance (+0.4%) is documented as arising from legacy OpenCV morphology vs. exact pure-Python bounding box rounding. The precise crop files and manifests (`split_outputs/pool_a_crops_manifest.csv`, `split_outputs/pool_b_crops_manifest.csv`) are frozen before training.

### 2.2 Source-Split Integrity Verification
- Source images are strictly partitioned without patient/image overlap:
  - **Train**: 1,931 crops (Pool A: 917, Pool B: 1,014) across 382 candidate-bearing source images.
  - **Validation**: 382 crops (Pool A: 150, Pool B: 232) across 80 candidate-bearing source images.
  - **Test**: 404 crops (Pool A: 182, Pool B: 222) across 81 candidate-bearing source images.
- No source image appears in more than one split partition (0 leakage across splits verified).

---

## 3. Stage 2: Train One Reference Model (ImageNet ResNet-18)

### 3.1 Pre-Training System & Hardware Compatibility
- **Hardware Platform**: Apple Silicon M4, 16 GB Unified RAM.
- **Acceleration Backend**: PyTorch Apple Metal Performance Shaders (`mps`). Fallback: CPU.
- **Memory & Batch Size**: Batch size set to 32 or 64 to avoid unified memory pressure.
- **Pre-Flight Smoke Test**: Execute a minimal 1-epoch / 2-batch smoke test on training/validation data to verify tensor flow, gradient calculation, and loss computation without evaluating the test set.

### 3.2 Reference Training Protocol
- **Architecture**: Standard Torchvision ResNet-18 initialized with default ImageNet-1k weights (`ResNet18_Weights.DEFAULT`), with the final fully connected layer modified to a 2-class linear head (`fc = nn.Linear(512, 2)`).
- **Optimizer**: AdamW or SGD with momentum ($0.9$), weight decay $10^{-4}$.
- **Learning Rate**: $10^{-4}$ with cosine annealing or plateau scheduling.
- **Loss Function**: Cross-Entropy with inverse class weighting to balance Pool A and Pool B.
- **Data Augmentation (Train only)**: Random horizontal flip, vertical flip, slight rotation ($\pm 15^\circ$), and standard ImageNet normalization.
- **Model Selection Rule**: Model selection is governed **strictly by Validation Balanced Accuracy**. The test set is **never** evaluated during training or hyperparameter selection.
- **Artifact Preservation**:
  - Checkpoint binary: `best_model.pt`.
  - SHA-256 cryptographic hash of the saved checkpoint.
  - Complete software environment report (Python version, PyTorch version, MPS status).
  - Training and validation loss/metric curves logged per epoch.

### 3.3 Baseline Quality Gate
Before proceeding to artifact interventions, the baseline performance of the frozen reference model on the unmodified evaluation set must be inspected:
- If the model fails to achieve adequate baseline classification (e.g. balanced accuracy $<75\%$), artifact interventions will **not** be interpreted as reflecting high-performing classifiers; the training setup will first be diagnosed.

---

## 4. Stage 3: Freeze Model & Run Interventions

Once the model passes the quality gate, its weights are frozen permanently.

### 4.1 Subgroup Stratification on Unmodified Test Crops
Evaluate baseline predictions across four natural strata:
1. **Stratum 1 (Sickled, Clean / No Border)**: Pool A test crops without aperture borders ($N = 168$).
2. **Stratum 2 (Sickled, Border Present)**: Pool A test crops with aperture borders ($N = 14$).
3. **Stratum 3 (Non-Sickled, Clean / No Border)**: Pool B test crops without aperture borders ($N = 152$).
4. **Stratum 4 (Non-Sickled, Border Present)**: Pool B test crops with aperture borders ($N = 70$).

Compare error rates, class probabilities $\bar{p}$, and prediction entropy across strata.

### 4.2 Controlled Perimeter & Border Interventions ($T_{\text{border}}$ vs $T_{\text{control}}$)
- **Border Inpainting Operator ($T_{\text{border}}$)**: For crops with detected aperture borders, replace the outer 15% perimeter ring (19 pixels from edge) with the median color and luma of the local plasma background, blended with a Gaussian boundary kernel ($\sigma = 1.5$).
- **Matched Negative Control ($T_{\text{control}}$)**: Apply the identical inpainting operator to clean, border-free crops from both classes. This isolates the causal effect of border removal from the generic distribution shift caused by synthetic editing.
- **Metric**: Paired probability shift:
  $$\Delta p_i = p(\text{sickled} \mid T(x_i)) - p(\text{sickled} \mid x_i)$$
  Assess whether $\Delta \bar{p}_{\text{border-present, non-sickled}}$ exceeds $\Delta \bar{p}_{\text{control, non-sickled}}$.

### 4.3 Background Plasma Equalization Interventions ($T_{\text{plasma}}$ vs $T_{\text{photo-control}}$)
- **Photometric Equalization ($T_{\text{plasma}}$)**: Rebalance plasma background RGB vectors to the global training median background color and brightness.
- **Matched Luminance Control ($T_{\text{photo-control}}$)**: Apply a neutral brightness adjustment without altering color balance (R/G and R/B ratios).
- **Metric**: Track $\Delta p_i$ under color equalization versus neutral luminance scaling.

### 4.4 Statistical Uncertainty & Aggregation
- **Paired Shift Analysis**: Measure the distribution of $\Delta p_i$, categorical prediction flip rates ($F_{\text{flip}}$), and directional flips (Non-sickled $\to$ Sickled).
- **Source-Image Level Aggregation**: Average probabilities per source image and evaluate diagnosis shifts under both standard ($th = 0.50$) and validation-tuned ($th^*$) thresholds.
- **Uncertainty Bounds**: Compute 95% confidence intervals via **2,000 source-cluster bootstrap iterations** (resampling unique source images with replacement).

---

## 5. Pre-Registered Hypotheses & Interpretation Matrix

| Outcome Pattern | Empirical Observations | Scientific Conclusion |
|---|---|---|
| **Pattern 1: Substantial Shortcut Sensitivity** | Neutralizing borders on non-sickled crops causes significant positive shift $\Delta \bar{p} > +0.20$ and high flip rate ($>20\%$), well exceeding matched control edits. Clean test accuracy drops significantly. | **Shortcut Reliance Indicated**: The reference CNN is sensitive to macroscopic aperture borders, using them as a shortcut for non-sickled classification. High baseline performance was partially artifact-driven. |
| **Pattern 2: Moderate Confound Sensitivity** | Border neutralization and plasma equalization produce modest probability shifts ($+0.05 \le \Delta \bar{p} \le +0.15$), but categorical decisions rarely flip ($<5\%$). Shifts are comparable to or slightly above control edits. | **Partial Sensitivity / Redundant Representation**: The model exhibits modest sensitivity to perimeter/background cues, but cellular morphological features are sufficiently distinct to maintain classification decisions. |
| **Pattern 3: Intervention Insensitivity** | Probability shifts on intervened crops are negligible ($|\Delta \bar{p}| < 0.05$) and do not differ from matched negative control perturbations. Clean subgroup accuracy matches border-present accuracy. | **Intervention Insensitivity Observed**: Under the tested conditions, the reference model's predictions are insensitive to perimeter borders and background color shifts, indicating that its representations rely primarily on features internal to the cell body. |

---

## 6. Methodological Caveats & Protocol Constraints

1. **Synthetic Perturbation Distribution Shift**: Any image modification (even smooth inpainting) introduces an artificial perturbation. All intervention results must be benchmarked against matched control perturbations on clean images.
2. **Feature Redundancy in Neural Networks**: An unchanged classification output does not prove that non-morphological cues were ignored during training; it demonstrates that the model possesses sufficient redundant morphological features to preserve its output when the artifact is removed.
3. **No Direct Inference to Historical Models**: Results from this reference model describe the behavior of an ImageNet-pretrained ResNet-18 trained on this dataset version; they do not establish the exact internal representations of the historical models unless identical weights are evaluated.
