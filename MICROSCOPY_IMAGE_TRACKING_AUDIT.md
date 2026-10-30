# Git Repository Microscopy Image Tracking Audit

**Date**: 2026-10-30  
**Context**: Pre-publication compliance & dataset governance review for sickle-cell microscopy research codebase.

---

## 1. Executive Summary

A comprehensive scan of the repository's git index revealed **148 tracked image files** (`.png`, `.jpg`, `.jpeg`). 

All 148 images were committed in historical commits between **April 8 and April 16, 2026** (commits `22fe2b1`, `376a4bd`, `647b1cc`, and `804f640`). 

Recent experiments (Experiment 1 and 1.5) strictly enforced `.gitignore` exclusions, ensuring **zero new image files** were committed. However, `.gitignore` only applies to untracked files; files committed prior to the `.gitignore` update remain actively tracked by git.

---

## 2. Inventory and Categorization of Tracked Images

The 148 image files are categorized into four distinct risk tiers based on microscopy content:

### Category A: Pure Metric Charts & Data Visualizations (11 files) — **SAFE FOR RELEASE**
These files contain only synthetic matplotlib/seaborn plots, confusion matrices, ROC curves, and performance heatmaps. They contain **zero biomedical or microscopy pixels**:
- `fig02_stage4_confusion_matrix.png`
- `fig03_stage4_roc_curve.png`
- `fig05_stage5_baseline_metrics.png`
- `fig06_stage6_low_label_balanced_accuracy.png`
- `fig07_stage6_gain_over_scratch.png`
- `fig08_stage6_heatmap.png`
- `fig09_stage7_drop_from_clean.png`
- `fig10_stage7_perturbation_plot.png`
- `fig11_stage7_low_label_clean.png`
- `fig12_stage7_low_label_gain_over_scratch.png`
- `fig13_overall_robustness_snapshot.png`

---

### Category B: Contact Sheets & Crop Montages (2 files) — **REQUIRES REDISTRIBUTION REVIEW**
These files are composite image grids containing extracted microscopy crops:
- `fig01_pool_b_contact_sheet.jpg`: Montage of candidate non-sickled crops (Pool B spot-check).
- `fig04_stage4_error_sheet.jpg`: Montage of misclassified cell crops from Stage 4 baseline.

---

### Category C: Neural Interpretability Overlays (110 files) — **REQUIRES REDISTRIBUTION REVIEW**
These files contain individual cell microscopy crops overlaid with model explanation heatmaps (Grad-CAM, occlusion sensitivity, nearest neighbors, perturbation sweeps):
- **Root figures (6 files)**:
  - `fig14_xai_rgb_failure_gradcam.png`
  - `fig15_xai_rgb_failure_occlusion.png`
  - `fig16_xai_rgb_failure_neighbors.png`
  - `fig17_xai_rgb_failure_shortcut_sensitivity.png`
  - `fig18_xai_rgb_ambiguous_shortcut_sensitivity.png`
  - `fig19_xai_grayscale_ambiguous_shortcut_sensitivity.png`
- **Grayscale Interpretability Directory (52 files across 4 sample sources)**:
  - `interpretability_outputs/stage5_grayscale/samples/A__positive_20__0009/` (13 images)
  - `interpretability_outputs/stage5_grayscale/samples/A__positive_310__0002/` (13 images)
  - `interpretability_outputs/stage5_grayscale/samples/B__negative_100__0004/` (13 images)
  - `interpretability_outputs/stage5_grayscale/samples/B__negative_37__0012/` (13 images)
- **RGB Interpretability Directory (52 files across 4 sample sources)**:
  - `interpretability_outputs/stage5_rgb/samples/A__positive_20__0009/` (13 images)
  - `interpretability_outputs/stage5_rgb/samples/A__positive_310__0002/` (13 images)
  - `interpretability_outputs/stage5_rgb/samples/B__negative_100__0004/` (13 images)
  - `interpretability_outputs/stage5_rgb/samples/B__negative_37__0012/` (13 images)

---

### Category D: Preprocessing Sweeps & Step Visualizations (25 files) — **REQUIRES REDISTRIBUTION REVIEW**
These files display whole-field microscope captures and luminance thresholding sweeps:
- `background_band_detection_combined_lonly_band50/` (11 files): Microscope field band sweeps.
- `simple_preprocess_test_5p5n/` (11 files): Threshold sweeps across 5 positive and 5 negative fields.
- `new_flow_crop_preprocessing/step_visuals/` (3 files): Intermediate preprocessing steps (`negative_128_preprocessing_steps.png`, `negative_23_preprocessing_steps.png`, `positive_2_preprocessing_steps.png`).

---

## 3. Policy & Governance Assessment

1. **Private vs. Public Exposure**:
   - The repository is currently private. In a private research context, keeping these artifacts internally does not cause public exposure.
   - However, if the repository is made public (e.g. accompanying a paper publication or open-source release), committing microscopy-derived imagery must strictly comply with the dataset's licensing and institutional data-sharing agreements.
2. **Current `.gitignore` Health**:
   - The `.gitignore` file now excludes:
     ```gitignore
     datasets*/
     crop_outputs/
     experiment_1_inspection_outputs/*.png
     experiment_1_inspection_outputs/*.jpg
     ```
   - This prevents any new crops, contact sheets, or datasets from being staged.

---

## 4. Pre-Publication Remediation Protocol

When preparing to publish or open-source this repository, execute the following protocol:

### Step 1: Untrack Microscopy Images from Git Index
```bash
git rm --cached fig01_pool_b_contact_sheet.jpg fig04_stage4_error_sheet.jpg fig14_*.png fig15_*.png fig16_*.png fig17_*.png fig18_*.png fig19_*.png
git rm -r --cached interpretability_outputs/
git rm -r --cached background_band_detection_combined_lonly_band50/
git rm -r --cached simple_preprocess_test_5p5n/
git rm -r --cached new_flow_crop_preprocessing/
```

### Step 2: Purge Binary Blobs from Git Commit History
Because git history preserves past commits, run `git-filter-repo` (recommended) to remove binary image blobs from historical commits:
```bash
git-filter-repo --invert-paths --path-match 'interpretability_outputs/' --path-match 'background_band_detection_combined_lonly_band50/' --path-match 'simple_preprocess_test_5p5n/' --path-match 'new_flow_crop_preprocessing/' --path 'fig01_pool_b_contact_sheet.jpg' --path 'fig04_stage4_error_sheet.jpg'
```

### Step 3: Retain Only Synthetic Metric Plots
Keep Category A plots (`fig02`, `fig03`, `fig05`–`fig13`) in the repository as reproducible numerical reporting assets.

---

## 5. Experiment 3 & 3.1 Status Update (October 2026)

- **Audit Verification**: All newly created diagnostic artifacts from Experiment 3.1 (`experiment_3_reference_outputs/**/*.png`, including the 20 outlier comparison panels) were untracked from the Git index (`git rm --cached`) and `.gitignore` was updated accordingly.
- **Repository Visibility**: The repository remains **private**.
- **Action Required Prior to Public Release**: A formal decision must be made to either (a) verify open-access redistribution rights for the 137 historical microscopy-derived files under the source dataset license, or (b) execute the Step 1 & Step 2 cleanup protocol using `git-filter-repo` to ensure a completely clean public Git history.
