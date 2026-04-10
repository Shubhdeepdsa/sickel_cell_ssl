# Sickle Cell SSL Project Handoff Summary

## Purpose of this document

This file is a handoff summary for another LLM or collaborator. It explains what was built, how the Uganda smartphone microscopy dataset was processed, what experiments were run, what the results were, what is complete versus partial, and what the next model should and should not claim.

## Executive summary

This project tests a focused question:

**Does in-domain self-supervised pretraining help binary sickled-vs-non-sickled cell crop classification on a small, noisy smartphone-microscopy dataset, especially when labeled data is limited?**

Current answer from the implemented pipeline:

- Yes, **SimCLR on in-domain Uganda crop candidates clearly improves over supervised training from scratch**, and that advantage grows as labels are reduced.
- However, **ImageNet fine-tuning is still the strongest overall baseline** in the current runs.
- The main unresolved scientific risk is **color/stain dependence and crop-pipeline artifacts**, especially because:
  - stain metadata is not currently available in the manifests,
  - positive boxes were recovered from raster overlays rather than original annotation coordinates,
  - non-sickled crops were generated heuristically from negative images and need more manual filtering.

The most honest short conclusion is:

**The SSL idea shows real signal in this pipeline, especially in the low-label regime, but the best current model is still ImageNet fine-tuning, and stain/color robustness plus label cleanliness remain the main limitations.**

## Project scope and framing

What this project is:

- A **cell-crop classification** pipeline.
- Built around the **Uganda smartphone sickle-cell microscopy dataset**.
- Designed to test SSL in a **low-resource, domain-matched smartphone setting**.
- Focused on **binary classification**:
  - `sickled`
  - `non_sickled`

What this project is not:

- Not whole-image diagnosis.
- Not cell counting.
- Not segmentation.
- Not diffusion-based augmentation work.
- Not multi-dataset pretraining yet.
- Not a clinically validated medical model.

## Dataset used

Primary dataset used in the actual pipeline:

- Uganda smartphone microscopy sickle-cell dataset under `datasets/`

Dataset folders in this workspace:

- `datasets/Positive/Labelled`
- `datasets/Positive/Unlabelled`
- `datasets/Negative/Clear`

Important note:

- The often-discussed auxiliary morphology dataset such as Chula-RBC-12 was **not used** in the current implemented experiments.
- This repo currently represents a **Uganda-only pipeline**.

## Repository status

Core scripts:

- `eda.py`
- `split_protocol.py`
- `crop_creation.py`
- `train_binary_classifier.py`
- `train_stage5.py`
- `train_stage6_low_label.py`
- `evaluate_stage7_robustness.py`
- `evaluate_stage7_low_label_robustness.py`

Key outputs already present:

- `eda_outputs/`
- `split_outputs/`
- `crop_outputs/`
- `stage4_outputs/`
- `stage5_outputs/`
- `stage6_outputs/`
- `stage7_outputs/`
- `stage7_low_label_outputs/`

Important status note:

- `stage6_outputs/` and `stage7_low_label_outputs/` are currently untracked in git.
- The experiment artifacts exist locally and should be treated as real outputs, but not yet as cleanly versioned final deliverables.

## Stage 1: Data audit

Implemented in `eda.py`.

### Goal

Before modeling, the project audited the raw Uganda dataset to answer:

- How many positive and negative source images exist?
- Are stain labels explicit?
- What is the image quality distribution?
- How are annotations stored?
- Can the positive annotations be converted into usable crop coordinates?

### Audit results

From `eda_outputs/stage1_data_audit.md`:

- Positive raw images for modeling: `422`
- Positive labelled overlay images: `422`
- Negative clear images: `147`
- Positive labelled/unlabelled filename pairs: `422`
- Total image files scanned: `991`
- Source images for downstream positive/negative splitting: `569`

Image size summary:

- Median width: `1000`
- Median height: `1009`
- Width range: `469` to `4160`
- Height range: `436` to `3440`

Image quality summary:

- `usable`: `565`
- `low_contrast`: `4`

Color-cast proxy summary:

- `balanced_mixed_cast`: `238`
- `blue_purple_cast`: `120`
- `green_cyan_cast`: `11`
- `pink_red_cast`: `200`

### Stain metadata status

This is important:

- **No explicit stain labels were encoded in the current folder names or filenames.**
- All current stain values in split manifests are effectively `unknown`.
- A stain template was prepared later so metadata can be added if recovered.

### Annotation storage reality

No separate XML/JSON/TXT coordinate annotations were found.

Instead:

- the positive annotations are **black rectangle overlays baked directly into the labelled JPGs** in `datasets/Positive/Labelled`,
- and the corresponding raw images are in `datasets/Positive/Unlabelled`.

This means the project had to **recover annotation boxes from raster overlays** rather than load a clean annotation file.

### How annotation boxes were recovered

`eda.py` uses an image-processing heuristic:

1. Resize the labelled image and its matching raw image to a fixed analysis size of `384 x 384`.
2. Try a paired-difference mask:
   - reference image brighter than labelled image by more than `20` luma units,
   - labelled max channel less than `135`,
   - labelled channel range less than `75`.
3. If paired differencing fails, fall back to single-image dark-rectangle detection:
   - R, G, B each below `95`,
   - channel range below `95`.
4. Remove edge-connected components.
5. Keep connected components only if they look box-like:
   - component size thresholding,
   - width/height filtering,
   - fill ratio filtering,
   - rectangularity score threshold.
6. Estimate interior foreground coverage and classify each recovered box as:
   - `likely_crop_candidate`
   - `loose_or_empty`
   - `overfilled`
   - or reject it earlier

### Annotation recovery results

Recovered from the raster overlays:

- Labelled images with at least one detected box: `421 / 422`
- Total detected boxes: `1408`
- Likely crop-candidate boxes: `993 / 1408` (`70.5%`)

Tightness buckets:

- `likely_crop_candidate`: `993`
- `loose_or_empty`: `159`
- `overfilled`: `256`

### Stage 1 caveat

These are **recovered raster-derived boxes**, not original annotation coordinates. They are good enough for a first-pass pipeline, but should not be described as perfect gold-standard boxes without manual verification.

### Stage 1 outputs

- `eda_outputs/image_audit.csv`
- `eda_outputs/annotation_boxes.csv`
- `eda_outputs/stage1_data_audit.md`

## Stage 2: Source-image split protocol

Implemented in `split_protocol.py`.

### Goal

Prevent leakage by splitting at the **source image level before crop generation**.

This is critical because splitting after crop creation would let train and test share image-specific style, background, and capture artifacts from the same original image.

### Split policy

- `70%` train
- `15%` val
- `15%` test
- deterministic seed: `42`

### Actual split counts

From `split_outputs/stage2_split_protocol.md`:

- Train: `399 / 569`
- Val: `85 / 569`
- Test: `85 / 569`

By source-image label:

- Train: `296 positive`, `103 negative`
- Val: `63 positive`, `22 negative`
- Test: `63 positive`, `22 negative`

### Stain stratification status

The script supports stratifying by `label + stain`, but because stain labels were unavailable:

- all stain values remained `unknown`,
- stain stratification did not add information in the current run.

### Source-key design

The pipeline uses a globally unique `source_key`, for example:

- `positive:1`
- `negative:1`

This matters because numeric filenames overlap between positive and negative folders.

### Stage 2 outputs

- `split_outputs/source_image_splits.csv`
- `split_outputs/split_label_summary.csv`
- `split_outputs/split_stain_summary.csv`
- `split_outputs/crop_annotation_splits.csv`
- `split_outputs/stain_metadata_template.csv`
- `split_outputs/stage2_split_protocol.md`

## Stage 3: Crop creation

Implemented in `crop_creation.py`.

### Goal

Create the three crop pools needed for supervised learning and SSL:

- **Pool A**: labeled sickled crops
- **Pool B**: labeled non-sickled crops
- **Pool C**: unlabeled candidate crops for SimCLR

Every crop inherits the Stage 2 source-image split. No random crop-level splitting is used.

### Pool A: labeled sickled crops

Source:

- recovered raster boxes from positive labelled overlays

Procedure:

- Scale each recovered overlay box back to the raw source image coordinates.
- Expand to a square crop using:
  - margin factor `1.35`
  - minimum side length equal to the output crop size
- Write final crops at `128 x 128`
- By default, boxes marked `loose_or_empty` are excluded

### Pool B: labeled non-sickled crops

Source:

- negative images only

Procedure in `crop_creation.py`:

1. Resize source image to `384 x 384` for analysis.
2. Build a candidate mask from likely cell foreground:
   - luma between `45` and `245`
   - saturation above `0.055`
   - max channel above `75`
3. Extract connected components.
4. Keep components only if they satisfy shape constraints:
   - not tiny,
   - not too large,
   - aspect ratio not extreme,
   - foreground coverage at least `0.10`
5. Convert each component to an original-resolution square crop:
   - square margin factor `2.2`
   - minimum crop size `64`
6. Sort by component area and keep at most `12` candidates per negative source image.

This pool is labeled `non_sickled`, but it is still a **heuristic crop pool**, not a manually verified cell-by-cell expert annotation set.

### Pool C: unlabeled candidate crops

Source:

- both positive and negative source images

Procedure:

- Same classical candidate extractor as Pool B
- Keep at most `8` candidate crops per source image
- Label is `unlabeled`

### Crop counts

From `crop_outputs/stage3_crop_creation.md`:

- Pool A labeled sickled: `1249`
  - train: `917`
  - val: `150`
  - test: `182`
- Pool B labeled non-sickled: `1462`
  - train: `1000`
  - val: `236`
  - test: `226`
- Pool C unlabeled candidates: `3971`
  - train: `2790`
  - val: `568`
  - test: `613`

### Stage 3 quality-control note

Pool B has a known weakness:

- the contact sheet shows that some crops likely include border/context artifacts, edge-of-field regions, or imperfect cell candidates.

This should be explicitly treated as a limitation of the current label construction process.

### Stage 3 outputs

- `crop_outputs/pool_a_labeled_sickled_manifest.csv`
- `crop_outputs/pool_b_labeled_non_sickled_manifest.csv`
- `crop_outputs/pool_c_unlabeled_candidates_manifest.csv`
- `crop_outputs/pool_b_spotcheck_sample.csv`
- `crop_outputs/pool_b_spotcheck_contact_sheet.jpg`
- `crop_outputs/stage3_crop_creation.md`

## Stage 4: First downstream baseline

Implemented in:

- `train_binary_classifier.py`
- `visualize_stage4.py`

### Goal

Build a fast sanity-check baseline before deep learning:

- task: binary crop classification
- classes: `sickled` vs `non_sickled`

### Data used

- Pool A sickled crops
- Pool B non-sickled crops

Counts:

- Train: `917 sickled`, `1000 non_sickled`
- Val: `150 sickled`, `236 non_sickled`
- Test: `182 sickled`, `226 non_sickled`

Leakage check:

- sources with multiple splits: `0`
- sources with conflicting labels: `0`

### Model

This was a classical baseline, not SSL:

- logistic regression
- features from:
  - resized RGB thumbnail features,
  - RGB histograms,
  - simple texture/brightness features
- regularization selected by validation balanced accuracy
- selected `C = 0.01`

### Stage 4 results

Validation:

- Accuracy: `0.8057`
- Balanced accuracy: `0.8083`
- Macro F1: `0.8000`
- ROC AUC: `0.8673`

Test:

- Accuracy: `0.8505`
- Balanced accuracy: `0.8485`
- Macro F1: `0.8486`
- ROC AUC: `0.8988`

Test confusion matrix:

| Actual | Predicted non_sickled | Predicted sickled |
|---|---:|---:|
| non_sickled | 196 | 30 |
| sickled | 31 | 151 |

Interpretation:

- This showed that the crop pipeline contains real signal.
- It also showed that a model can do much better than majority class guessing.
- But it did **not** prove morphology-only learning, because style and crop-construction artifacts could still contribute.

### Stage 4 outputs

- `stage4_outputs/binary_cell_classifier.joblib`
- `stage4_outputs/metrics.json`
- `stage4_outputs/test_predictions.csv`
- `stage4_outputs/validation_predictions.csv`
- `stage4_outputs/stage4_binary_classifier.md`
- `stage4_outputs/visual_diagnostics.md`
- visual plots and error sheets under `stage4_outputs/visuals/`

## Stage 5: Deep baselines and in-domain SimCLR

Implemented in `train_stage5.py`.

### Goal

Compare three stronger baselines:

1. `supervised_scratch`
2. `imagenet_finetune`
3. `simclr_finetune`

### Supervised classifier details

Backbone:

- ResNet-18

Training details:

- image size: `224`
- batch size: `32`
- epochs: `20`
- optimizer: `AdamW`
- learning rate: `1e-4`
- weight decay: `1e-4`
- scheduler: cosine annealing
- loss: class-weighted cross-entropy

Data augmentation for supervised training:

- resize to `224 x 224`
- random horizontal flip
- random vertical flip
- random rotation up to `180`
- modest color jitter

### SimCLR pretraining details

Backbone:

- ResNet-18 encoder from random initialization

Projection head:

- linear -> ReLU -> linear
- projection dimension: `128`

Training details:

- epochs: `50`
- batch size: `64`
- learning rate: `3e-4`
- weight decay: `1e-4`
- temperature: `0.2`
- scheduler: cosine annealing

SimCLR augmentations:

- random resized crop with scale `(0.55, 1.0)`
- random horizontal and vertical flips
- random rotation up to `180`
- stronger color jitter
- random grayscale (`p = 0.10`)
- Gaussian blur

### What data SimCLR saw

Default SSL split policy was `train` only.

The SimCLR unlabeled pool was built from:

- Pool C unlabeled train crops: `2790`
- plus labeled train crops from Pool A and Pool B with labels removed: `1917`

So the Stage 5 train-only SSL pretraining pool was effectively:

- **`4707` in-domain train crops**

This is intentional and scientifically reasonable:

- labeled data is scarce for supervised fine-tuning,
- but the same train split can still provide unlabeled in-domain crops for SSL.

### SimCLR pretraining behavior

From `stage5_outputs/simclr_pretrain/simclr_history.csv`:

- SimCLR loss decreased from about `4.08` at epoch 1
- to about `1.20` by epoch 50

That at least indicates the contrastive pretraining objective was optimizing successfully.

### Stage 5 full-label results

From `stage5_outputs/stage5_baseline_comparison.csv`:

| Setup | Best val bal acc | Test crop bal acc | Test crop ROC AUC | Test source bal acc |
|---|---:|---:|---:|---:|
| supervised_scratch | 0.9343 | 0.9311 | 0.9854 | 0.9518 |
| imagenet_finetune | 0.9945 | 0.9863 | 0.9997 | 1.0000 |
| simclr_finetune | 0.9785 | 0.9851 | 0.9987 | 1.0000 |

### Stage 5 interpretation

- Deep learning clearly outperformed the Stage 4 classical baseline.
- SimCLR gave a **large boost over training from scratch**.
- But ImageNet fine-tuning remained **marginally stronger than SimCLR** at full labels.

Most honest Stage 5 takeaway:

- **In-domain SSL works and is worthwhile**
- but **it is not yet beating ImageNet pretraining in the current pipeline**

### Stage 5 outputs

- `stage5_outputs/stage5_report.md`
- `stage5_outputs/stage5_baseline_comparison.csv`
- `stage5_outputs/simclr_pretrain/simclr_encoder.pt`
- `stage5_outputs/simclr_pretrain/simclr_history.csv`
- per-setup checkpoints, histories, metrics, and predictions under:
  - `stage5_outputs/supervised_scratch/`
  - `stage5_outputs/imagenet_finetune/`
  - `stage5_outputs/simclr_finetune/`

## Stage 6: Low-label protocol

Implemented in `train_stage6_low_label.py`.

### Goal

Test whether SSL becomes more useful when labeled data is scarce.

### Protocol

- Labels were reduced only inside the train split.
- Sampling was done at the **source-image level**, not at the crop level.
- Validation and test stayed fixed.
- Fractions were nested and class-stratified:
  - `100%`
  - `50%`
  - `20%`
  - `10%`

This means:

- the 10% source set is a subset of 20%
- 20% is a subset of 50%
- 50% is a subset of 100%

### Train subset sizes

From `stage6_outputs/stage6_report.md`:

| Fraction | Train sources | Train crops |
|---|---:|---:|
| 100% | 382 | 1917 |
| 50% | 190 | 900 |
| 20% | 76 | 367 |
| 10% | 38 | 196 |

### Stage 6 low-label results

From `stage6_outputs/stage6_low_label_comparison.csv`:

| Fraction | Scratch bal acc | SimCLR bal acc | ImageNet bal acc | SimCLR minus scratch |
|---|---:|---:|---:|---:|
| 100% | 0.9311 | 0.9873 | 0.9863 | +0.0562 |
| 50% | 0.8388 | 0.9658 | 0.9895 | +0.1270 |
| 20% | 0.7745 | 0.9310 | 0.9724 | +0.1565 |
| 10% | 0.6844 | 0.8791 | 0.9305 | +0.1948 |

ROC AUC trend:

- Scratch drops sharply as labels are reduced.
- SimCLR stays much stronger than scratch.
- ImageNet remains the strongest overall baseline.

Source-level balanced accuracy:

- Scratch:
  - `0.9518` at 100%
  - `0.8297` at 50%
  - `0.7704` at 20%
  - `0.7419` at 10%
- SimCLR:
  - `1.0000` at 100%
  - `0.9688` at 50%
  - `0.9688` at 20%
  - `0.8810` at 10%
- ImageNet:
  - `1.0000` at 100%
  - `1.0000` at 50%
  - `0.9688` at 20%
  - `0.9831` at 10%

### Stage 6 interpretation

This is one of the main scientific results in the repo.

What Stage 6 shows:

- **SimCLR is materially useful in the low-label regime.**
- The benefit over scratch increases as labels shrink.
- However, **ImageNet remains better than SimCLR at every tested label fraction**.

So the honest interpretation is:

- the SSL idea is **not weak**,
- but it is **not yet the best initialization strategy** in this pipeline.

### Stage 6 outputs

- `stage6_outputs/stage6_report.md`
- `stage6_outputs/stage6_low_label_comparison.csv`
- fraction-specific checkpoints and metrics under:
  - `stage6_outputs/fraction_100/`
  - `stage6_outputs/fraction_050/`
  - `stage6_outputs/fraction_020/`
  - `stage6_outputs/fraction_010/`

## Stage 7: Robustness evaluation on full-label models

Implemented in `evaluate_stage7_robustness.py`.

### Goal

Measure whether the learned representations are stable under mild smartphone-style perturbations.

### Perturbations used

- `clean`
- `blurred`
  - Gaussian blur radius `1.25`
- `lower_brightness`
  - brightness factor `0.75`
- `mild_compression`
  - JPEG quality `45`
- `mild_stain_color`
  - RGB scales `(1.08, 0.97, 1.06)`
  - color factor `1.08`
  - contrast factor `1.03`

### Stage 7 standard robustness results

From `stage7_outputs/stage7_report.md`, which evaluates the Stage 5 full-label checkpoints:

| Setup | Clean | Blur | Brightness | Compression | Stain/color | Biggest drop |
|---|---:|---:|---:|---:|---:|---|
| supervised_scratch | 0.9311 | 0.9261 | 0.8824 | 0.9090 | 0.5066 | stain/color |
| imagenet_finetune | 0.9863 | 0.9835 | 0.9808 | 0.9478 | 0.9635 | compression |
| simclr_finetune | 0.9851 | 0.9846 | 0.9785 | 0.9851 | 0.7212 | stain/color |

All values above are **test crop balanced accuracy**.

### Stage 7 interpretation

This is the other major scientific result:

- Scratch is highly vulnerable to stain/color shift.
- SimCLR is much better than scratch overall, but in this specific Stage 5 robustness run it still showed a large stain/color drop.
- ImageNet was strongest and most stable overall.

So the clean full-label performance story is good for SimCLR, but **color/stain robustness remains a real weakness**.

### Important nuance about checkpoint lineage

`stage7_outputs/` evaluates the Stage 5 full-label checkpoints from `stage5_outputs/`.

Those are **not necessarily identical** to the full-label checkpoints created later under Stage 6 `fraction_100`, even though they use the same dataset and nearly the same protocol. The two result families should be kept conceptually separate.

## Stage 7 low-label robustness

Implemented in `evaluate_stage7_low_label_robustness.py`.

### Goal

Measure whether low-label gains survive under perturbation.

### Current completion status

This stage is **only partially complete** in saved outputs:

- `fraction_100`: complete
- `fraction_050`: complete
- `fraction_020`: not present
- `fraction_010`: not present

Do not claim Stage 7 low-label robustness is complete for all fractions.

### Fraction 50 robustness results

These are especially useful because they directly test the low-label setting.

From `stage7_low_label_outputs/fraction_050/*/metrics.json`:

| Setup | Clean | Blur | Brightness | Compression | Stain/color |
|---|---:|---:|---:|---:|---:|
| supervised_scratch | 0.8388 | 0.8439 | 0.6198 | 0.8061 | 0.5000 |
| imagenet_finetune | 0.9895 | 0.9834 | 0.9868 | 0.9676 | 0.9099 |
| simclr_finetune | 0.9658 | 0.9536 | 0.9619 | 0.9636 | 0.7212 |

Interpretation at 50% labels:

- SimCLR retains a **large advantage over scratch** under blur, brightness, and compression.
- Scratch collapses badly on brightness and stain/color.
- SimCLR still suffers on stain/color, though much less than scratch.
- ImageNet remains best.

### Fraction 100 low-label robustness results

These use the Stage 6 `fraction_100` checkpoints, not the Stage 5 checkpoints.

From `stage7_low_label_outputs/fraction_100/*/metrics.json`:

| Setup | Clean | Blur | Brightness | Compression | Stain/color |
|---|---:|---:|---:|---:|---:|
| supervised_scratch | 0.9311 | 0.9261 | 0.8824 | 0.9090 | 0.5066 |
| imagenet_finetune | 0.9863 | 0.9835 | 0.9808 | 0.9478 | 0.9635 |
| simclr_finetune | 0.9873 | 0.9851 | 0.9812 | 0.9873 | 0.9226 |

Important note:

- The `fraction_100` SimCLR robustness numbers here are **better than the Stage 5 robustness run**, especially on stain/color.
- That is not necessarily an inconsistency; it reflects a different trained checkpoint lineage:
  - Stage 5 direct training output
  - versus Stage 6 retrained `fraction_100` output

If another LLM is summarizing the project, it should state which checkpoint family it is referring to.

## Main conclusions

### What is convincingly shown

1. **The Uganda-only cell-crop framing works well enough to support meaningful experiments.**
2. **Source-level splitting before crop creation was correctly enforced**, which avoids a major leakage failure mode.
3. **In-domain SimCLR helps substantially over supervised-from-scratch training.**
4. **That SimCLR benefit becomes larger as labeled data shrinks.**
5. **ImageNet fine-tuning is still the strongest baseline overall.**

### What is not yet solved

1. **Stain/color robustness is still a key weakness**, especially for scratch and some SimCLR checkpoints.
2. **Pool B is only heuristically labeled**, not manually cleaned at scale.
3. **Positive annotation boxes were recovered from raster overlays**, not original coordinate labels.
4. **Field vs Leishman stain metadata is still missing**, so stain-aware splitting and evaluation were not yet done.
5. **Low-label robustness is not complete for the 20% and 10% fractions.**

## What another LLM should not overclaim

Do not say:

- that stain-balanced evaluation was done using real stain labels,
- that the model is robust to all domain shifts,
- that this is a whole-image diagnostic model,
- that non-sickled crop labels are expert-verified at the cell level,
- that original bounding-box annotation files were used,
- that low-label robustness was completed for every fraction.

## Recommended next steps

If continuing this project, the next sensible actions are:

1. **Manually review and filter Pool B** to remove border/context artifacts and obvious bad negatives.
2. **Spot-check recovered Pool A boxes** and, if possible, improve raster-to-box recovery.
3. **Recover real stain metadata** if available and rerun split/evaluation with stain-aware stratification.
4. **Finish Stage 7 low-label robustness** for `20%` and `10%`.
5. **Re-run or consolidate full-label robustness** so there is a single canonical report for:
   - Stage 5 checkpoints
   - Stage 6 fraction_100 checkpoints
6. Only after the Uganda-only story is stable, consider:
   - auxiliary morphology pretraining,
   - external-domain testing,
   - counting or whole-image tasks

## Most useful files to inspect first

If another LLM needs to continue quickly, the highest-value files are:

- `PROJECT_HANDOFF_SUMMARY.md`
- `pipeline_summary_stages_1_to_5.md`
- `eda_outputs/stage1_data_audit.md`
- `split_outputs/stage2_split_protocol.md`
- `crop_outputs/stage3_crop_creation.md`
- `stage5_outputs/stage5_report.md`
- `stage6_outputs/stage6_report.md`
- `stage7_outputs/stage7_report.md`
- `stage6_outputs/stage6_low_label_comparison.csv`
- `stage5_outputs/stage5_baseline_comparison.csv`

## Paste-ready continuation brief

If you want to continue with another LLM, a good short handoff prompt is:

> We built a Uganda-only smartphone microscopy pipeline for sickled vs non-sickled cell crop classification. We audited 569 source images, recovered positive raster box annotations, split at source-image level, created Pool A sickled crops, Pool B heuristic non-sickled crops, and Pool C unlabeled candidate crops. Stage 5 showed SimCLR and ImageNet both strongly outperform scratch at full labels; Stage 6 showed SimCLR gains over scratch grow as labels shrink, but ImageNet remains best. Stage 7 showed stain/color robustness is still the main weakness, especially for scratch and some SimCLR checkpoints. Please read `PROJECT_HANDOFF_SUMMARY.md` first and help me decide the next rigorous step without overclaiming stain metadata or label cleanliness.

