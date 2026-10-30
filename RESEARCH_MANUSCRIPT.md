# When Dataset Shortcuts Do Not Translate into Model Reliance: An Evaluation Audit of Smartphone Sickle-Cell Microscopy Classification

**Authors / Research Team**: Antigravity Research Lab & Pair Programming Team  
**Repository Branch**: `experiment-3-reference-model`  
**Date**: October 2026  
**Status**: Completed Technical Research Manuscript & Evidence Audit  

---

## Abstract

Automated classification of sickle-cell erythrocyte morphology from smartphone-based microscopy holds promise for point-of-care screening in low-resource settings. However, deep neural networks often achieve high diagnostic accuracy by exploiting unintended dataset artifacts rather than genuine pathological morphology. In this investigation, we conducted an end-to-end evaluation audit of an erythrocyte classification pipeline previously reported to achieve 98–99% accuracy. 

In **Experiment 1**, systematic inspection revealed a striking visual asymmetry: dark circular microscope aperture borders contaminated **35.4%** of reconstructed non-sickled cell crops compared to only **10.0%** of sickled crops. In **Experiment 2**, restricted-feature classifiers demonstrated that perimeter color and edge statistics alone achieved **67.9% balanced accuracy**, proving that peripheral non-morphological shortcuts were available to predict class labels. 

To determine whether high-performing convolutional networks necessarily rely on these shortcuts, in **Experiment 3** we trained an ImageNet-pretrained ResNet-18 reference model on 1,931 rigorously source-partitioned crops (zero patient/source leakage). The model achieved **99.5% balanced accuracy** (402/404 correct) on the test set and **100.0%** across 81 source images. When subjected to controlled perimeter inpainting ($T_{\text{border}}$ vs $T_{\text{control}}$) and background plasma equalization ($T_{\text{plasma}}$), the model exhibited limited average differential sensitivity: the net border-removal contrast was $\Delta_{\text{contrast}} = -0.0026$ (95% source-cluster bootstrap CI: $[-0.0474, +0.0470]$), and plasma equalization shifted predictions by only $+0.0011$. 

In **Experiment 3.1**, forensic inspection of individual prediction changes revealed that 14.4% of non-sickled crops experienced large prediction swings ($|\Delta P| > 0.10$). Crucially, **16 of the top 20 extreme outliers (80.0%) possessed no aperture border**, with visual review adjudicating that 50% of failures arose from peripheral adjacent cells being clipped by the margin inpainting canvas. 

These findings decouple **shortcut availability** from **demonstrated model reliance**: a high-performing model can emerge without depending on a prominent dataset artifact. However, intervention insensitivity cannot be equated with proven biological morphology learning, as other unmeasured acquisition characteristics and contextual features remain unresolved.

---

## 1. Introduction

Sickle cell disease (SCD) is a hereditary hemoglobinopathy affecting millions worldwide, characterized by the polymerisation of abnormal hemoglobin (HbS) under deoxygenated conditions, inducing red blood cells (RBCs) to adopt rigid, crescent-like ("sickled") morphologies. Deep learning applied to smartphone-mounted microscopy offers a low-cost screening pathway in resource-constrained environments. However, computer vision models deployed on clinical microscopy are notoriously prone to *shortcut learning*—relying on spurious correlations such as illumination gradients, staining variations, or acquisition artifacts rather than intended biological morphology.

Previous internal iterations of this project achieved remarkable test accuracies exceeding 98–99% using convolutional backbones and self-supervised SimCLR representations. Before progressing to multi-center validation or clinical deployment, we initiated a pre-registered evaluation audit to answer a fundamental scientific question:

> **Core Research Question**: *When a medical-image dataset contains unintended, label-predictive imaging shortcuts, does a high-performing deep convolutional classifier necessarily depend on them?*

This manuscript consolidates the four-stage experimental roadmap designed to investigate this question.

---

## 2. Dataset Reconstruction & Artifact Characterization (Experiments 1 & 1.5)

### 2.1 Reconstructed Reference Pool
To ensure rigorous reproducibility, we reconstructed the labeled dataset from primary smartphone microscopy images, enforcing complete source-partition separation to prevent patient/slide-level data leakage:

- **Pool A (Sickled)**: 1,249 crops extracted from 399 positive smartphone microscopy source images using annotated bounding boxes.
- **Pool B (Non-Sickled)**: 1,468 candidate erythrocyte crops extracted from 144 candidate-bearing negative source images via connected-component luminance and saturation filtering.
- **Source Partitioning**: All crops were allocated into frozen partitions preserving source-image boundaries:
  - **Training Split**: 1,931 crops (917 sickled, 1,014 non-sickled) from 382 source images.
  - **Validation Split**: 382 crops (150 sickled, 232 non-sickled) from 80 source images.
  - **Test Split**: 404 crops (182 sickled, 222 non-sickled) from 81 source images.
  - **Split Leakage**: Exactly 0 source images overlap across splits.

```
Manifest Provenance:
- Pool A: crop_outputs/pool_a_labeled_sickled_manifest.csv (SHA-256: 1e849ffd8350578e...)
- Pool B: crop_outputs/pool_b_labeled_non_sickled_manifest.csv (SHA-256: 61fab3f079a4d965...)
```

### 2.2 Discovery of the Aperture Border Asymmetry
In Experiment 1 and 1.5, we applied an automated dark-border detector based on annular edge luminance gradients. Across the reconstructed dataset:

$$\text{Border Frequency (Sickled / Pool A)} = 10.0\% \quad (125 / 1,249)$$
$$\text{Border Frequency (Non-Sickled / Pool B)} = 35.4\% \quad (520 / 1,468)$$

The estimated difference of **25.4 percentage points** (95% source-cluster bootstrap CI: $[20.1\%, 30.7\%]$) was systematic and persistent across training, validation, and test partitions. Negative crops disproportionately sampled peripheral regions near the dark circular field-of-view stop of the smartphone microscope adapter.

---

## 3. Shortcut Availability Analysis (Experiment 2)

Having established a physical artifact disparity, Experiment 2 evaluated whether an algorithmic classifier could exploit these peripheral non-morphological cues to predict class labels without examining central cellular morphology:

1. **Tier 1 (Binary Border Indicator)**: A single-feature decision stump achieved **61.9% balanced accuracy** on the test set.
2. **Tier 2 (Perimeter Edge Statistics)**: Regularized logistic regression fitted strictly on the outer 15% crop margin (mean RGB, luminance, gradient magnitude, and chromaticity ratios) achieved **67.9% balanced accuracy** (71.4% sickled recall, 64.4% non-sickled specificity).
3. **Tier 3 (Isolated Background Plasma Features)**: Classifiers trained on segmented extracellular plasma regions achieved **64.1% balanced accuracy**.

**Finding**: Non-morphological peripheral cues contain measurable, statistically reliable diagnostic signal in this dataset. However, demonstrating that a shortcut is *available* does not prove that a deep neural network *relies* on it.

---

## 4. Reference Model Training & Baseline Quality Gate (Experiment 3 Stage 2)

To evaluate model reliance, we trained an ImageNet-pretrained ResNet-18 reference model under the frozen Stage 5 protocol:

### 4.1 Implementation & System Verification
- **Architecture**: Torchvision ResNet-18 initialized with ImageNet-1K weights (`ResNet18_Weights.DEFAULT`), replacing the final fully-connected layer with a 2-class linear projection.
- **Hardware Acceleration**: Apple Silicon M4 unified memory utilizing the PyTorch Metal Performance Shaders (`mps`) backend.
- **Optimization**: AdamW optimizer ($\text{lr} = 10^{-4}$, weight decay $= 10^{-4}$), inverse-frequency class loss weighting, Cosine Annealing learning rate schedule across 20 epochs.
- **Model Selection**: Checkpoint selection was strictly locked to peak validation balanced accuracy to eliminate test-set snooping.

### 4.2 Baseline Test Evaluation
The model achieved peak validation performance at Epoch 13 (**99.8% validation balanced accuracy**). The weights were frozen into `best_model.pt` (SHA-256: `83cd725076f60899bea0dfe4b7bbaa729ff82a60293b4e799d3082a6ae0419bb`).

| Evaluation Level | Sample Size | Balanced Accuracy | Raw Accuracy | Sickled Recall (TPR) | Non-Sickled Spec (TNR) | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|
| **Crop Level (Standard $th=0.50$)** | 404 crops | **99.5%** | 99.5% | 99.5% (181/182) | 99.5% (221/222) | 0.9997 |
| **Source Level (Cluster Mean)** | 81 images | **100.0%** | 100.0% | 100.0% (59/59) | 100.0% (22/22) | 1.0000 |

With test balanced accuracy far exceeding the pre-registered quality gate ($\ge 75.0\%$), the frozen reference model was eligible for causal intervention testing.

---

## 5. Controlled Artifact Interventions (Experiment 3 Stage 3)

We tested model sensitivity across three pre-registered intervention paradigms:

### 5.1 Subgroup Stratification on Natural Crops
We stratified natural, unmodified test crops by class and aperture border presence:

| Subgroup Stratum | Number of Crops | Reference Model Accuracy | Mean $P(\text{Sickled})$ | Std Dev |
|---|---:|---:|---:|---:|
| **Stratum 1: Sickled (Clean / No Border)** | 168 | **99.4%** | 0.994 | 0.075 |
| **Stratum 2: Sickled (Border Present)** | 14 | **100.0%** | 0.996 | 0.012 |
| **Stratum 3: Non-Sickled (Clean / No Border)** | 152 | **99.3%** | 0.013 | 0.063 |
| **Stratum 4: Non-Sickled (Border Present)** | 70 | **100.0%** | 0.003 | 0.023 |

**Key Observation**: The model does not require the dark aperture border to recognize non-sickled cells. On the 152 clean non-sickled crops with zero border artifact, the model achieved **99.3% accuracy** (151/152 correct) with a mean predicted sickled probability of only **0.013**.

### 5.2 Controlled Perimeter Inpainting ($T_{\text{border}}$ vs $T_{\text{control}}$)
To test causal dependence on the aperture border, we replaced the outer 15% margin (19 pixels) with local background color using a Gaussian-smoothed edge blend. We evaluated border-bearing non-sickled crops ($T_{\text{border}}$) against an identical edit on border-free non-sickled crops ($T_{\text{control}}$):

- **Border Removal on Border Non-Sickled ($n=70$)**: Mean $\Delta P = \mathbf{+0.0447}$, Decision Flip Rate = **2.9%** (2 / 70).
- **Control Edit on Clean Non-Sickled ($n=152$)**: Mean $\Delta P = \mathbf{+0.0473}$, Decision Flip Rate = **3.9%** (6 / 152).

### 5.3 Background Plasma Equalization
To test whether the model exploited global staining or plasma chromaticity differences, we multiplicatively rescaled all RGB channels to match the empirical training-set median background RGB (`(216.13, 217.12, 209.12)`):

- **Plasma Equalization ($T_{\text{plasma-exact}}$)**: Mean $\Delta P = \mathbf{+0.0011}$ (Mean $|\Delta P| = 0.0048$, 95% CI: $[-0.001, +0.004]$).
- **Luminance Scaling ($T_{\text{photo-control}}$)**: Mean $\Delta P = \mathbf{+0.0010}$ (95% CI: $[-0.000, +0.003]$).

---

## 6. Forensic Failure Analysis & Adjudication (Experiment 3.1)

### 6.1 Contrast Uncertainty Bounds
To rigorously evaluate whether border removal produced an effect distinguishable from the control edit, we computed 2,000 source-cluster bootstrap resamples on the direct difference:

$$\Delta_{\text{contrast}} = \mathbb{E}[\Delta P \mid \text{Border}] - \mathbb{E}[\Delta P \mid \text{Clean}] = -0.0026$$
$$95\% \text{ Source-Cluster Bootstrap CI} = \mathbf{[-0.0474, +0.0470]}$$

$$\Delta_{|\text{contrast}|} = \mathbb{E}[|\Delta P| \mid \text{Border}] - \mathbb{E}[|\Delta P| \mid \text{Clean}] = -0.0173$$
$$95\% \text{ Source-Cluster Bootstrap CI} = \mathbf{[-0.0640, +0.0339]}$$

**Interpretation**: The point estimate is near zero and the confidence interval spans zero, demonstrating that our investigation did not detect a differential border-specific effect beyond general perimeter editing. However, because the interval admits possible differences of up to $\pm 0.05$, this represents an **absence of detected difference rather than proven mathematical equivalence**.

### 6.2 Signed vs Absolute Shift Distributions
While the signed average shift was modest (+0.045), examining absolute probability shifts ($|\Delta P|$) revealed substantial heterogeneity:

```
Non-Sickled Response to Perimeter Replacement:
- Median |Delta P|: 0.0000 (75% of crops had |Delta P| <= 0.0005)
- Crops with |Delta P| > 0.10: 14.4% (32 / 222)
  * Border-Bearing: 14.3% (10 / 70)
  * Border-Free:    14.5% (22 / 152)
```

The proportion of extreme prediction shifts was virtually identical between border-bearing and border-free crops.

### 6.3 Adjudicated Visual Audit of the Top 20 Outliers
To uncover why individual predictions swung dramatically, we performed an image-by-image visual review of the 20 largest prediction shifts (logged in `outlier_visual_audit_adjudication.csv`):

```
Adjudicated Mechanism Taxonomy (Top 20 Outliers):
1. Peripheral Cellular Clipping: 10 / 20 (50.0%)
   - Neighboring erythrocytes or dense clusters graze or cross the 15% outer margin. 
   - Inpainting severed cellular contours, altering aspect ratios and inducing false sickle predictions.
2. Aperture Border Removal: 4 / 20 (20.0%)
   - Genuine dark aperture arcs erased and replaced with plasma canvas, shifting P toward sickled.
3. Empty / Faint Plasma Background Crops: 3 / 20 (15.0%)
   - Crops contained <1% dark cell body; flat synthetic canvas introduced artificial boundary gradients.
4. Threshold Boundary Cases Clarified: 3 / 20 (15.0%)
   - Ambiguous boundary cases (P ~ 0.28-0.51) resolved cleanly to non-sickled when edge clutter was erased.
```

**Critical Factual Distribution**: Exactly **16 of the top 20 outliers (80.0%) were border-free crops (`border = 0`)**. Only 4 outliers possessed an aperture border. This confirms that extreme prediction shifts were predominantly caused by the artificial perimeter removal of cellular material rather than reliance on the microscope aperture.

---

## 7. Discussion & Scientific Implications

### 7.1 Decoupling Shortcut Availability from Model Reliance
A central contribution of this research is the empirical decoupling of shortcut *availability* from model *reliance*:
1. Experiments 1 and 2 proved that a prominent imaging artifact (aperture border) was strongly correlated with class labels and contained sufficient information for shallow classifiers to achieve 67.9% balanced accuracy.
2. Experiments 3 and 3.1 demonstrated that an ImageNet-pretrained ResNet-18 did not rely on this artifact to achieve 99.5% test balanced accuracy. On border-free crops, the model maintained 99.3% accuracy, and removing borders produced no greater average shift than control edits on clean crops.

This illustrates that deep convolutional representations initialized with natural image features can ignore accessible linear or low-level shortcuts when robust morphological representations are available.

### 7.2 Methodological Boundaries & Negative Claims
We deliberately avoid overclaiming that the reference model has "learned pure sickle-cell biology":
- **Untested Covariates**: The model may rely on other subtle acquisition covariates, such as optical chromatic aberration, lens distortion profiles, or slide staining chemistry, that were not altered by our interventions.
- **Observational Limitations**: The border-versus-control comparison represents an observational contrast across different image crops, not an experiment on identical image content.
- **Source-Image Independence vs. Patient Generalization**: While source-image splitting ensures that crops from the same field of view never span splits, true patient-level generalization requires multi-center validation on distinct clinical cohorts.

---

## 8. Reproducibility & Research Governance

- **Repository**: `https://github.com/Shubhdeepdsa/sickel_cell_ssl`
- **Active Branch**: `experiment-3-reference-model`
- **Environment**: Python 3.12.14, PyTorch 2.14.1 (`mps` backend on Apple Silicon M4)
- **Deterministic Provenance**: Random seeds locked to `42`; SHA-256 logged for manifests and model weights.
- **Data Protection Compliance**: Raw private microscopy images and full-resolution patient slides are excluded from git tracking via `.gitignore`. All audit scripts are executable locally from the committed manifests.

---

## 9. Conclusion

In smartphone sickle-cell microscopy classification, unintended peripheral artifacts can provide predictive shortcuts without necessarily corrupting deep neural network predictions. By combining statistical confound analysis, controlled image interventions, source-cluster bootstrap uncertainty estimation, and qualitative visual failure audits, research teams can rigorously distinguish spurious shortcut reliance from robust performance prior to clinical translation.
