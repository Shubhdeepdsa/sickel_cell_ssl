# Shortcut Availability Versus Model Sensitivity: An Exploratory Audit of Smartphone Sickle-Cell Microscopy Classification

**Author**: Shubhdeep Das  
**Technical Assistance**: Antigravity AI Coding Assistant  
**Repository Branch**: `experiment-3-reference-model`  
**Date**: October 2026  
**Document Status**: Final Consolidated Technical Research Manuscript  

---

## Abstract

Automated classification of sickle-cell erythrocyte morphology from smartphone-based microscopy offers significant potential for low-cost, point-of-care screening. However, deep neural networks frequently achieve high diagnostic accuracy by exploiting unintended dataset artifacts rather than genuine pathological morphology. In this investigation, we conducted an end-to-end evaluation audit of an erythrocyte classification pipeline previously reported to achieve 98–99% classification accuracy.

In **Experiment 1**, systematic inspection revealed a striking macroscopic asymmetry: dark circular microscope aperture borders contaminated **35.4%** (519 / 1,468) of reconstructed non-sickled cell crops compared to only **10.0%** (125 / 1,249) of sickled crops. In **Experiment 2**, restricted-feature classifiers demonstrated that perimeter color, luminance, and edge statistics alone achieved **67.9% balanced accuracy** (73.6% sickled recall, 62.2% non-sickled specificity), establishing that peripheral imaging cues carried substantial label-predictive information.

To evaluate whether high-performing convolutional networks necessarily depend on these signals, in **Experiment 3** we trained an ImageNet-pretrained ResNet-18 reference model under a frozen exploratory protocol on 1,931 source-partitioned crops. The model achieved **99.5% balanced accuracy** (402/404 correct crops) on the test partition and **100.0%** across 81 source-image files. When subjected to controlled perimeter inpainting ($T_{\text{border}}$ vs $T_{\text{control}}$) and background plasma equalization ($T_{\text{plasma}}$), the model exhibited limited average differential sensitivity: the net border-removal contrast was $\Delta_{\text{contrast}} = -0.0026$ (95% source-cluster bootstrap CI: $[-0.0474, +0.0470]$), and plasma color equalization shifted predictions by only $+0.0011$.

In **Experiment 3.1**, forensic inspection of individual prediction changes revealed that 14.4% of non-sickled crops experienced large prediction swings ($|\Delta P| > 0.10$). An image-by-image visual review of the 20 largest prediction shifts revealed that **16 of the 20 extreme outliers (80.0%) possessed no aperture border**, with 45.0% (9/20) of failures adjudicated to peripheral neighboring erythrocytes being severed by the inpainting margin.

These findings separate three questions that are frequently conflated in clinical AI audits:
1. *Does the dataset contain an unintended shortcut?* **Yes**; class-associated border and perimeter differences are present.
2. *Can a classifier exploit that shortcut?* **Yes**; restricted perimeter features predict class labels well above chance.
3. *Did the high-performing reference model depend on that shortcut?* **The interventions did not establish substantial differential dependence on the tested aperture border on average**, although individual predictions exhibited meaningful vulnerability to general peripheral image modifications.

While these results decouple **shortcut availability** from **demonstrated model reliance**, they do not prove exclusive reliance on biological cell morphology, as other unmeasured acquisition characteristics and patient-level covariates remain untested.

---

## 1. Introduction

Sickle cell disease (SCD) is a life-threatening genetic hemoglobinopathy characterized by the polymerisation of abnormal hemoglobin (HbS) under hypoxic conditions, prompting erythrocytes to deform into rigid, crescent-shaped morphologies. Point-of-care screening using optical smartphone-microscope attachments could dramatically expand screening access in low-resource clinics. However, computer vision models applied to biomedical microscopy frequently suffer from *shortcut learning*—relying on non-biological visual features such as illumination gradients, staining artifacts, or field-of-view vignetting that correlate with clinical labels in a specific dataset.

Earlier stages of this research reported test accuracies exceeding 98–99% using convolutional backbones and self-supervised SimCLR representations. Before progressing to multi-center trials or clinical validation, we initiated a systematic evaluation audit to address a foundational methodological question:

> **Core Research Question**: *When a medical microscopy dataset contains measurable, label-predictive peripheral shortcuts, does a high-performing deep convolutional classifier necessarily depend on them to achieve high diagnostic accuracy?*

This manuscript consolidates the complete four-stage experimental audit designed to test this question.

---

## 2. Dataset Reconstruction & Artifact Characterization (Experiments 1 & 1.5)

### 2.1 Reconstructed Reference Pool & Partitioning
To ensure absolute reproducibility, we reconstructed the crop dataset from primary smartphone microscopy images:

- **Pool A (Sickled)**: 1,249 crops extracted from 399 positive smartphone microscopy source images using annotated bounding boxes.
- **Pool B (Non-Sickled)**: 1,468 candidate erythrocyte crops extracted from 144 candidate-bearing negative source images via connected-component luminance and saturation filtering.
- **Partitioning Integrity**:
  - **Training Split**: 1,931 crops (917 sickled, 1,014 non-sickled) from 382 unique source images.
  - **Validation Split**: 382 crops (150 sickled, 232 non-sickled) from 80 unique source images.
  - **Test Split**: 404 crops (182 sickled, 222 non-sickled) from 81 unique source images.

```
Manifest Provenance:
- Pool A: crop_outputs/pool_a_labeled_sickled_manifest.csv (SHA-256: 1e849ffd8350578e...)
- Pool B: crop_outputs/pool_b_labeled_non_sickled_manifest.csv (SHA-256: 61fab3f079a4d965...)
```

> [!IMPORTANT]
> **Source-Image Independence vs. Patient-Level Generalization**:
> No source-image identifiers were shared across training, validation, and test partitions (zero source-image leakage). However, **patient-level and acquisition-session independence could not be verified** because the required clinical, slide-level, and session metadata were unavailable in the public dataset. All source-level metrics reported herein reflect consistency across source-image files, not verified independent patient encounters.

### 2.2 Discovery of the Aperture Border Asymmetry
In Experiment 1 and 1.5, we applied an automated edge luminance detector to flag crops intersecting the dark circular field-of-view aperture stop of the smartphone adapter:

$$\text{Border Frequency (Sickled / Pool A)} = 10.0\% \quad (125 / 1,249)$$
$$\text{Border Frequency (Non-Sickled / Pool B)} = 35.4\% \quad (519 / 1,468)$$

The estimated difference of **25.4 percentage points** (95% source-cluster bootstrap CI: $[20.1\%, 30.7\%]$) was consistent across all partitions. Negative candidate generation disproportionately extracted crops from peripheral slide regions adjacent to the circular aperture boundary.

---

## 3. Shortcut Availability Analysis (Experiment 2)

Experiment 2 tested whether an algorithm could predict the clinical label using only peripheral image features, without examining central cellular morphology:

1. **Tier 1 (Binary Border Indicator)**: A single-feature decision stump using only the presence/absence of an aperture border achieved **61.9% balanced accuracy** on the test partition.
2. **Tier 2 (Perimeter Edge Statistics)**: A regularized logistic regression model fitted strictly on 12 features extracted from the outer 15% crop margin (19 pixels from edge: mean RGB, luminance, gradient magnitude, and chromaticity ratios) achieved **67.9% balanced accuracy** (73.6% sickled recall [134/182], 62.2% non-sickled specificity [138/222]).
3. **Tier 3 (Isolated Background Plasma)**: Classifiers trained on segmented extracellular background plasma achieved **64.1% balanced accuracy**.

> [!NOTE]
> **Perimeter Feature Content**:
> As documented in Experiment 2, the outer 15% margin captures both microscope aperture vignetting and peripheral erythrocyte packing geometry (adjacent cells grazing the margin). Thus, Tier 2 demonstrates that peripheral image information is highly predictive, but that information is not exclusively non-morphological.

---

## 4. Reference Model Training & Baseline Evaluation (Experiment 3 Stage 2)

To evaluate whether a deep network relies on these peripheral cues, we trained an ImageNet-pretrained ResNet-18 reference model under a **frozen exploratory protocol**:

### 4.1 Implementation & System Verification
- **Architecture**: Torchvision ResNet-18 initialized with ImageNet-1K weights (`ResNet18_Weights.DEFAULT`), replacing the final layer with a 2-class linear classifier.
- **Hardware Acceleration**: Apple Silicon M4 unified memory running PyTorch with Metal Performance Shaders (`mps`).
- **Optimization**: AdamW optimizer ($\text{lr} = 10^{-4}$, weight decay $= 10^{-4}$), inverse-frequency class loss weighting, Cosine Annealing learning rate schedule across 20 epochs.
- **Model Selection**: Checkpoint selection was strictly locked to peak validation balanced accuracy to prevent test-set snooping.

### 4.2 Baseline Test Evaluation
The reference model achieved peak validation performance at Epoch 13 (**99.8% validation balanced accuracy**). The weights were frozen into `best_model.pt` (SHA-256: `83cd725076f60899bea0dfe4b7bbaa729ff82a60293b4e799d3082a6ae0419bb`).

| Evaluation Level | Sample Size | Balanced Accuracy | Raw Accuracy | Sickled Recall (TPR) | Non-Sickled Spec (TNR) | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|
| **Crop Level (Standard $th=0.50$)** | 404 crops | **99.5%** | 99.5% | 99.5% (181/182) | 99.5% (221/222) | 0.9997 |
| **Source-Image Level (Cluster Mean)** | 81 source files | **100.0%** | 100.0% | 100.0% (59/59) | 100.0% (22/22) | 1.0000 |

With test performance exceeding the pre-registered baseline quality gate ($\ge 75.0\%$), the model was deemed sufficiently performant to begin controlled artifact interventions.

---

## 5. Controlled Artifact Interventions (Experiment 3 Stage 3)

### 5.1 Subgroup Stratification on Natural Crops
We stratified natural, unmodified test crops by class and border presence:

| Subgroup Stratum | Number of Crops | Reference Model Accuracy | Mean $P(\text{Sickled})$ | Std Dev |
|---|---:|---:|---:|---:|
| **Stratum 1: Sickled (Clean / No Border)** | 168 | **99.4%** | 0.994 | 0.075 |
| **Stratum 2: Sickled (Border Present)** | 14 | **100.0%** | 0.996 | 0.012 |
| **Stratum 3: Non-Sickled (Clean / No Border)** | 152 | **99.3%** | 0.013 | 0.063 |
| **Stratum 4: Non-Sickled (Border Present)** | 70 | **100.0%** | 0.003 | 0.023 |

**Key Observation**: On the 152 natural non-sickled crops with **zero detected aperture border**, the model achieved **99.3% accuracy** (151/152 correct) with a mean predicted sickled probability of only **0.013**. The aperture border is clearly not necessary for the model to classify non-sickled cells.

### 5.2 Controlled Perimeter Inpainting ($T_{\text{border}}$ vs $T_{\text{control}}$)
We replaced the outer 15% margin (19 pixels) with local background color using a Gaussian-smoothed edge blend, comparing border-bearing non-sickled crops ($T_{\text{border}}$) against an identical edit on border-free non-sickled crops ($T_{\text{control}}$):

- **Border Removal on Border Non-Sickled ($n=70$)**: Mean $\Delta P = \mathbf{+0.0447}$, Decision Flip Rate = **2.9%** (2 / 70).
- **Control Edit on Clean Non-Sickled ($n=152$)**: Mean $\Delta P = \mathbf{+0.0473}$, Decision Flip Rate = **3.9%** (6 / 152).

### 5.3 Background Plasma Equalization
We multiplicatively rescaled all RGB channels to match the empirical training-set median background RGB (`(216.13, 217.12, 209.12)`):

- **Plasma Equalization ($T_{\text{plasma-exact}}$)**: Mean $\Delta P = \mathbf{+0.0011}$ (Mean $|\Delta P| = 0.0048$, 95% CI: $[-0.001, +0.004]$).
- **Luminance Scaling ($T_{\text{photo-control}}$)**: Mean $\Delta P = \mathbf{+0.0010}$ (95% CI: $[-0.000, +0.003]$).

---

## 6. Forensic Failure Analysis & Adjudication (Experiment 3.1)

### 6.1 Direct Contrast Uncertainty Bounds
To test whether border removal produced an effect distinguishable from the control edit, we computed 2,000 source-cluster bootstrap resamples on the contrast:

$$\Delta_{\text{contrast}} = \mathbb{E}[\Delta P \mid \text{Border}] - \mathbb{E}[\Delta P \mid \text{Clean}] = -0.0026$$
$$95\% \text{ Source-Cluster Bootstrap CI} = \mathbf{[-0.0474, +0.0470]}$$

$$\Delta_{|\text{contrast}|} = \mathbb{E}[|\Delta P| \mid \text{Border}] - \mathbb{E}[|\Delta P| \mid \text{Clean}] = -0.0173$$
$$95\% \text{ Source-Cluster Bootstrap CI} = \mathbf{[-0.0640, +0.0339]}$$

> [!NOTE]
> **Interpretation as an Uncertainty Bound**:
> The estimated contrast is near zero and the 95% bootstrap interval spans zero, supporting the finding that the study did not detect a clear differential response between the two groups. However, because the interval admits modest differences of up to $\sim \pm 0.05$, this represents an **absence of detected difference rather than proven mathematical equivalence**. Furthermore, this comparison reflects an observational contrast between different crops across clusters rather than a matched intervention on identical cellular content.

### 6.2 Signed vs Absolute Shift Distributions
While the signed average shift was modest (+0.045), examining absolute probability shifts ($|\Delta P|$) revealed substantial individual sensitivity:
- **Median $|\Delta P|$**: 0.0000 (over 75% of crops had $|\Delta P| \le 0.0005$).
- **Crops with $|\Delta P| > 0.10$**: **14.4%** (32 / 222 non-sickled crops).
  - Border-Bearing: **14.3%** (10 / 70)
  - Border-Free: **14.5%** (22 / 152)

Large prediction shifts occurred at virtually identical frequencies regardless of whether the original image contained a border.

### 6.3 Adjudicated Visual Review of Top 20 Outliers
To determine what the intervention modified in individual cases, we conducted an image-by-image visual review of the 20 largest prediction shifts (recorded in `outlier_visual_audit_adjudication.csv`):

| Adjudicated Category | Count | Proportion | Crops Affected | Empirical Visual Observations |
|---|---:|---:|---|---|
| **Peripheral Cell Clipping** | 9 | **45.0%** | `pool_b_89_cand04`, `128_cand04`, `37_cand09`, `66_cand09`, `108_cand02`, `133_cand01`, `108_cand04`, `108_cand12`, `49_cand10` | Neighboring red blood cells graze the 15% outer margin. Inpainting severs adjacent cell bodies, creating asymmetric contours that mimic abnormal cell morphology. |
| **Aperture Border Removal** | 4 | **20.0%** | `pool_b_96_cand07`, `147_cand01`, `96_cand10`, `92_cand11` | Genuine dark aperture boundary erased and replaced with plasma canvas, shifting $P(\text{Sickled})$ upward ($+0.28$ to $+0.96$). |
| **Threshold Boundary Cases Clarified** | 3 | **15.0%** | `pool_b_37_cand06`, `pool_b_56_cand12`, `pool_b_37_cand03` | Borderline predictions ($P \approx 0.28 - 0.51$) resolved to confident non-sickled ($P \approx 0.002 - 0.28$) when peripheral noise and edge shadows were erased. |
| **Empty / Faint Plasma Background Crops** | 2 | **10.0%** | `pool_b_116_cand11`, `pool_b_122_cand04` | Crops contain virtually no dark cell body (<1% dark pixels); uniform inpainting canvas introduces artificial edge gradients on flat plasma. |
| **Subthreshold Border and Crowding** | 1 | **5.0%** | `pool_b_108_cand03` | Peripheral field contains subthreshold dark edge material and dense RBC crowding, both erased by canvas. |
| **Faint Crop Boundary Erasure** | 1 | **5.0%** | `pool_a_399_box1` | Faint sickled crop with subtle boundary gradient; margin inpainting blended the anchoring edge contour. |

> [!IMPORTANT]
> **Adjudicated Outlier Distribution**: Exactly **16 of the top 20 outliers (80.0%) were border-free crops (`border = 0`)**. This confirms that large prediction changes were predominantly driven by general peripheral image modifications (especially adjacent cell truncation and synthetic blend boundaries) rather than isolated aperture border reliance.

---

## 7. Discussion & Scientific Implications

### 7.1 Decoupling Shortcut Availability from Model Reliance
The primary scientific contribution of this study is the empirical separation of **shortcut availability** from **demonstrated model reliance**:
1. Experiments 1 and 2 proved that an aperture-border shortcut was strongly associated with class labels and sufficient for shallow models to predict labels well above chance (67.9% balanced accuracy).
2. Experiments 3 and 3.1 demonstrated that a high-performing ResNet-18 reference model showed no clear additional average sensitivity to border removal relative to control edits ($\Delta_{\text{contrast}} = -0.0026$), while maintaining 99.3% accuracy on border-free crops.

A predictive shortcut can exist in a dataset without being essential to every deep model trained on that dataset.

### 7.2 Methodological Boundaries & Limitations
We emphasize the strict methodological boundaries of our findings:
- **No Proof of Exclusive Morphology Learning**: Showing limited sensitivity to aperture borders and color shifts does not prove that the model relies exclusively on biological sickle-cell morphology. Other untested acquisition covariates, such as optical aberrations, slide preparation artifacts, or subtle contextual patterns, could remain influential.
- **Single Architecture Scope**: We evaluated one newly trained reference model (ImageNet-initialized ResNet-18). These experiments do not establish whether self-supervised models (e.g., SimCLR) or models trained strictly from scratch exhibit the same level of intervention sensitivity.
- **Source-Image Independence vs Clinical Generalization**: Partitioning by source-image prevents intra-image crop leakage, but does not guarantee generalization to independent clinics, staining protocols, or patient demographics.

---

## 8. Data Governance & Public Release Roadmap

- **Current Repository Status**: The code and manifests are maintained in a **private research repository** on branch `experiment-3-reference-model`.
- **Pre-Publication Image Audit**: In addition to untracking new outlier panels, an audit identified 137 historical microscopy-derived visualization files currently tracked in the Git tree and prior commit history.
- **Release Prerequisite**: Before transitioning the repository to public access, either:
  1. A formal redistribution-rights review must confirm permission to host derived crop visualizations under the original dataset license; or
  2. A Git history prune (via `git-filter-repo` or BFG) must remove historical image binaries from prior commits while preserving all commit logs, source scripts, manifests, and technical reports.

---

## 9. Conclusion

In smartphone sickle-cell microscopy classification, prominent peripheral artifacts can provide predictive shortcuts without necessarily governing the predictions of a high-performing convolutional network. By pairing confound identification with controlled image interventions, source-cluster uncertainty bounds, and image-level visual audits, researchers can distinguish shortcut availability from demonstrated model reliance, establishing realistic evidence boundaries prior to clinical deployment.
