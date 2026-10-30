# Experiment 1: Dataset Quality & Crop Inspection Report

## Executive Summary
Before deploying downstream models, this experiment investigates the quality, cleanliness, and artifact presence in the automated crop dataset. A balanced stratified sample of **72 crops** was evaluated across all three dataset partitions (`train`, `val`, and `test`) and both classes (`positive_sickled` Pool A and `negative_non_sickled` Pool B).

> [!NOTE]
> The automated metrics below represent **automated quality indicators** produced by computer-vision heuristics (e.g. edge variance, connected components, perimeter luma) rather than manual expert-verified morphological diagnoses.

### Core Quality Summary
- **Positive (Sickled) Crops (Pool A - 36 inspected)**:
  - **Automated Usable Cell Indicator**: **36/36 (100.0%)** contain identifiable, focused foreground components.
  - **Multiple Cells / Cell-Centered Patch**: 33/36 (91.7%) contain adjacent or clustered red blood cells in addition to the target sickled cell.
  - **Visible Border Vignette**: 4/36 (11.1%) overlap with the outer microscope frame / dark vignette rim.
  - **Empty Regions**: 0/36 (0.0%) had insufficient cell contrast.
  - **Suspicious Artifacts**: 0/36 (0.0%) contained dark raster box traces or severe blur.

- **Negative (Non-Sickled) Crops (Pool B - 36 inspected)**:
  - **Automated Usable Cell Indicator**: **34/36 (94.4%)** contain valid red blood cell structures.
  - **Visible Border Vignette**: **16/36 (44.4%)** capture the dark circular microscope aperture or boundary.
  - **Multiple Cells / Cell-Centered Patch**: 27/36 (75.0%) capture dense fields of normal red blood cells.
  - **Empty Regions**: 2/36 (5.6%) contain bare plasma or very low-contrast areas.
  - **Suspicious Artifacts**: 1/36 (2.8%) show notable debris or focus degradation.

## Detailed Breakdown Across Splits

| Split | Class | Inspected | Usable Cell | Visible Border | Empty Region | Multiple Cells | Suspicious Artifacts |
|---|---|---:|---:|---:|---:|---:|---:|
| train | Positive (Pool A) | 12 | 12 (100%) | 1 (8%) | 0 (0%) | 12 (100%) | 0 (0%) |
| train | Negative (Pool B) | 12 | 10 (83%) | 6 (50%) | 2 (17%) | 8 (67%) | 1 (8%) |
| val | Positive (Pool A) | 12 | 12 (100%) | 0 (0%) | 0 (0%) | 11 (92%) | 0 (0%) |
| val | Negative (Pool B) | 12 | 12 (100%) | 2 (17%) | 0 (0%) | 10 (83%) | 0 (0%) |
| test | Positive (Pool A) | 12 | 12 (100%) | 3 (25%) | 0 (0%) | 10 (83%) | 0 (0%) |
| test | Negative (Pool B) | 12 | 12 (100%) | 8 (67%) | 0 (0%) | 9 (75%) | 0 (0%) |

## Visual Contact Sheet
All 72 sampled crops were compiled into an inspection contact sheet with visual categorization tags:
- **Green** (`single usable cell`): Clean, well-centered individual cell component.
- **Blue** (`multiple cells`): Multiple adjacent or clumped red blood cells.
- **Crimson** (`visible border`): Boundary of microscope circular aperture ring or sensor edge.
- **Goldenrod** (`empty region`): Background plasma with negligible cell content.
- **Violet** (`suspicious artifacts`): Raster box line remnants or stain debris.

*(Note: Raw patient microscopy imagery and generated contact sheet images are kept locally untracked in `.gitignore` to adhere strictly to biomedical data privacy and licensing policies.)*

## Scientific Interpretation

### 1. Empirical Confirmation of Pool B Border Artifacts
The automated negative crop extraction algorithm (`extract_candidate_boxes` in `crop_creation.py`) utilizes classical contour thresholding across whole negative source images. In this 72-sample inspection, **44.4% of sampled negative crops (16/36) capture the outer dark circular vignette of the microscope**, compared to **11.1% (4/36) in positive crops**. Because positive crops (Pool A) were localized by human experts around specific sickled cells away from the microscope periphery, this asymmetry creates a serious **shortcut learning hypothesis** where models can differentiate positive from negative images merely by recognizing circular aperture curvature or dark perimeter gradients.

### 2. Multi-Cell Density & Framing: Cell-Centered Patches
In thin blood smears, red blood cells naturally cluster along monolayer boundaries. Approximately **83.3% of crops (60/72) across both pools contain multiple neighboring cells**. This indicates that the dataset is best scientifically framed as **cell-centered patch classification** rather than isolated single-cell morphology classification. Downstream analyses must acknowledge that surround context and neighboring RBCs are present in the vast majority of inputs.

### 3. Split Distribution Assessment
Border artifacts were observed in negative crops across all three dataset partitions:
- **Train**: 6/12 (50.0%)
- **Validation**: 2/12 (16.7%)
- **Test**: 8/12 (66.7%)

However, with only 12 samples per group, this sample size is too small to establish whether artifact prevalence is genuinely consistent between partitions. These apparent differences could reflect either random sampling variance or true distribution shifts. This limitation motivated **Experiment 1.5**, where automated detection was scaled across all 2,825 crops in the complete dataset with source-clustered bootstrap uncertainty intervals.

## Concrete Action Items for Model Pipeline
1. **Enforce Mask-Aware Preprocessing**: The frozen field-mask pipeline (`FROZEN_PREPROCESSING_PIPELINE.md` / `prototype_border_preprocessing_pipeline.py`) must be applied to suppress and zero-fill the outer dark circular border before extracting candidate negative crops.
2. **Implement Pool B Border Distance Filtering**: When generating candidate negative crops, enforce a minimum distance requirement from the detected field perimeter to prevent border-leakage shortcuts.
3. **Prune Low-Contrast Crops**: Discard negative candidates where foreground coverage is below 8% to prevent empty plasma crops from diluting the non-sickled class.