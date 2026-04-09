# Sickle Cell Pipeline Summary: Stages 1-5

This document summarizes the pipeline implemented so far, the outputs generated, and the current experimental results through Stage 4. Stage 5 is implemented but has not been run yet.

## Stage 1 - Data Audit

Implemented in `eda.py`.

Purpose:
- Inspect the Uganda sickle-cell dataset before modeling.
- Count positive and negative images.
- Inspect stain metadata availability.
- Inspect image quality.
- Determine how annotations are stored.
- Estimate whether bounding boxes are usable for crop extraction.

Key results:
- Positive raw images for modeling: 422.
- Positive labelled overlay images: 422.
- Negative clear images: 147.
- Positive labelled/unlabelled filename pairs: 422.
- Total image files scanned: 991.
- Source images for downstream positive/negative splitting: 569.
- Median image dimensions: 1000 x 1009.
- Width range: 469 to 4160.
- Height range: 436 to 3440.
- Explicit stain labels: not encoded for all 569 source images.
- Heuristic color cast over source images:
  - balanced_mixed_cast: 238
  - blue_purple_cast: 120
  - green_cyan_cast: 11
  - pink_red_cast: 200
- Image quality over source images:
  - usable: 565
  - low_contrast: 4

Annotation finding:
- No separate XML/JSON/TXT annotation files were found.
- The positive annotations are black rectangle overlays baked into `datasets/Positive/Labelled/*.jpg`.
- Matching raw images exist in `datasets/Positive/Unlabelled/*.jpg`.
- Detected annotation boxes: 1408.
- Labelled images with at least one detected box: 421 / 422.
- Likely crop-candidate boxes: 993 / 1408 (70.5%).
- Tightness buckets:
  - likely_crop_candidate: 993
  - loose_or_empty: 159
  - overfilled: 256

Important caveat:
- The annotation boxes are raster overlays, not clean coordinate labels. We used image processing to recover likely boxes. This is usable for a first pass, but should be manually spot-checked before treating it as gold-standard annotation.

Generated outputs:
- `eda_outputs/image_audit.csv`
- `eda_outputs/annotation_boxes.csv`
- `eda_outputs/stage1_data_audit.md`

## Stage 2 - Split Protocol

Implemented in `split_protocol.py`.

Purpose:
- Split at the source-image level before crop generation.
- Avoid random crop-level splitting, which would leak style from the same source image into train/test.
- Preserve source splits for all later crop manifests.

Split target:
- 70% train
- 15% validation
- 15% test

Actual source-image split:
- train: 399 / 569
- val: 85 / 569
- test: 85 / 569

Label balance:
- train: 296 positive, 103 negative
- val: 63 positive, 22 negative
- test: 63 positive, 22 negative

Stain balance:
- All stain values are currently `unknown` because Field vs Leishman is not encoded in the directory names or filenames.
- A template was created so stain labels can be added later:
  - `split_outputs/stain_metadata_template.csv`

Important design choice:
- The split manifest uses a globally unique `source_key`, for example `positive:1` and `negative:1`, because numeric image names overlap across positive and negative folders.

Generated outputs:
- `split_outputs/source_image_splits.csv`
- `split_outputs/split_label_summary.csv`
- `split_outputs/split_stain_summary.csv`
- `split_outputs/crop_annotation_splits.csv`
- `split_outputs/stain_metadata_template.csv`
- `split_outputs/stage2_split_protocol.md`

## Stage 3 - Crop Creation

Implemented in `crop_creation.py`.

Purpose:
- Create three crop pools while preserving the Stage 2 source-image split:
  - Pool A: labeled sickled crops from positive-image bounding boxes.
  - Pool B: labeled non-sickled crops from negative images using connected components.
  - Pool C: unlabeled candidate crops from both positive and negative images for SSL/SimCLR.

Generated crop pools:
- Pool A labeled sickled: 1249 crops.
  - train: 917
  - val: 150
  - test: 182
- Pool B labeled non-sickled: 1462 crops.
  - train: 1000
  - val: 236
  - test: 226
- Pool C unlabeled candidates: 3971 crops.
  - train: 2790
  - val: 568
  - test: 613

Verification:
- All crop files existed for all manifest rows.
- Sample crop image size: 128 x 128.
- Source split leakage check passed for all three crop pools:
  - `sources_with_multiple_splits = 0`

Important caveat:
- Pool B is generated from connected components in negative images. The spot-check contact sheet shows some edge-of-field and border/context crops. Pool B is useful for a first baseline, but should be manually reviewed/filtered before claiming final medical-grade labels.

Generated outputs:
- `crop_outputs/pool_a_labeled_sickled_manifest.csv`
- `crop_outputs/pool_b_labeled_non_sickled_manifest.csv`
- `crop_outputs/pool_c_unlabeled_candidates_manifest.csv`
- `crop_outputs/pool_b_spotcheck_sample.csv`
- `crop_outputs/pool_b_spotcheck_contact_sheet.jpg`
- `crop_outputs/stage3_crop_creation.md`
- Crop image directories under:
  - `crop_outputs/pool_a_labeled_sickled/`
  - `crop_outputs/pool_b_labeled_non_sickled/`
  - `crop_outputs/pool_c_unlabeled_candidates/`

## Stage 4 - First Downstream Task: Binary Cell Classification

Implemented in `train_binary_classifier.py` and `visualize_stage4.py`.

Task:
- Binary crop classification only:
  - `sickled`
  - `non_sickled`
- Not cell counting.
- Not whole-image diagnosis.

Training data:
- Pool A labeled sickled crops.
- Pool B labeled non-sickled crops.
- Crop rows inherit source-image splits from Stage 2.

Model:
- First baseline model, not the final model.
- Logistic regression on:
  - resized RGB thumbnail features,
  - RGB histogram features,
  - simple texture/brightness features.
- Regularization selected by validation balanced accuracy.
- Selected `C`: 0.01.

Dataset counts:
- train:
  - sickled: 917
  - non_sickled: 1000
- val:
  - sickled: 150
  - non_sickled: 236
- test:
  - sickled: 182
  - non_sickled: 226

Source leakage check:
- n_sources: 543.
- sources_with_multiple_splits: 0.
- sources_with_multiple_labels: 0.

Validation model selection:

| C | Balanced Accuracy | Accuracy | Macro F1 | ROC AUC |
|---|---:|---:|---:|---:|
| 0.01 | 0.8083 | 0.8057 | 0.8000 | 0.8673 |
| 0.1 | 0.7780 | 0.7746 | 0.7688 | 0.8421 |
| 1.0 | 0.7495 | 0.7487 | 0.7418 | 0.8050 |
| 10.0 | 0.7335 | 0.7306 | 0.7243 | 0.7734 |

Validation metrics for selected model:
- accuracy: 0.8057
- balanced_accuracy: 0.8083
- macro_f1: 0.8000
- ROC AUC: 0.8673

Test metrics:
- accuracy: 0.8505
- balanced_accuracy: 0.8485
- macro_f1: 0.8486
- ROC AUC: 0.8988

Test confusion matrix:

| Actual | Predicted non_sickled | Predicted sickled |
|---|---:|---:|
| non_sickled | 196 | 30 |
| sickled | 31 | 151 |

Majority baseline:
- Val majority label: non_sickled.
  - accuracy: 0.6114
  - balanced_accuracy: 0.5000
- Test majority label: non_sickled.
  - accuracy: 0.5539
  - balanced_accuracy: 0.5000

Interpretation:
- The model is not simply learning to always predict `non_sickled`.
- On test, it predicted 227 non_sickled and 181 sickled, which is close to the actual test distribution of 226 non_sickled and 182 sickled.
- Balanced accuracy is well above the one-class baseline of 0.5000.
- However, high accuracy does not yet prove true sickle-cell morphology learning. The model may still be learning artifacts from the crop pipeline, source image style, stain/color differences, microscope borders, or Pool A vs Pool B extraction differences.
- The Pool B spot-check and Stage 4 error sheets should be reviewed carefully before treating the model as scientifically reliable.

Stage 4 visual diagnostics:
- `stage4_outputs/visuals/confusion_matrix_test_counts.png`
- `stage4_outputs/visuals/confusion_matrix_test_normalized.png`
- `stage4_outputs/visuals/confusion_matrix_val_counts.png`
- `stage4_outputs/visuals/confusion_matrix_val_normalized.png`
- `stage4_outputs/visuals/roc_curve.png`
- `stage4_outputs/visuals/precision_recall_curve.png`
- `stage4_outputs/visuals/probability_hist_test.png`
- `stage4_outputs/visuals/probability_hist_val.png`
- `stage4_outputs/visuals/calibration_curve.png`
- `stage4_outputs/visuals/metric_bars.png`
- `stage4_outputs/visuals/test_crops_per_source.png`
- `stage4_outputs/visuals/test_error_contact_sheet.jpg`
- `stage4_outputs/visuals/validation_error_contact_sheet.jpg`
- `stage4_outputs/visual_diagnostics.md`

What a legitimate model should show in the visuals:
- Confusion matrix: both diagonal cells should be high, with neither row nor column dominating.
- ROC curve: should sit clearly above the random diagonal, with similar validation and test behavior.
- Precision-recall curve: should sit above class-prevalence baseline and not collapse at useful recall.
- Probability histogram: `sickled` and `non_sickled` probability distributions should separate for morphological reasons, not just crop borders/background.
- Calibration: should roughly track the diagonal if probabilities are to be trusted.
- Error sheets: false positives and false negatives should look like hard biological cases, not obvious crop extraction artifacts.

Generated outputs:
- `stage4_outputs/binary_cell_classifier.joblib`
- `stage4_outputs/binary_dataset_manifest.csv`
- `stage4_outputs/validation_predictions.csv`
- `stage4_outputs/test_predictions.csv`
- `stage4_outputs/validation_confusion_matrix.csv`
- `stage4_outputs/test_confusion_matrix.csv`
- `stage4_outputs/validation_model_selection.csv`
- `stage4_outputs/metrics.json`
- `stage4_outputs/stage4_binary_classifier.md`
- `stage4_outputs/visual_diagnostics.md`
- `stage4_outputs/visuals/`

## Stage 5 - Pretraining And Baselines

Implemented in `train_stage5.py`.

Purpose:
- Move beyond the Stage 4 logistic-regression baseline and compare three stronger model setups:
  - supervised from scratch,
  - ImageNet pretrained fine-tuning,
  - SimCLR pretrained on in-domain crops, then fine-tuned.

Implemented setups:
- `supervised_scratch`
  - ResNet-18 trained from random initialization on Pool A vs Pool B.
- `imagenet_finetune`
  - ResNet-18 initialized from ImageNet weights, then fine-tuned on Pool A vs Pool B.
- `simclr_finetune`
  - ResNet-18 SimCLR-pretrained on in-domain crop images, then fine-tuned on Pool A vs Pool B.

Important leakage policy:
- By default, SimCLR pretraining uses only train-split crops:
  - `--ssl-splits train`
- This avoids using validation/test source-image style during pretraining.
- If transductive SSL is intentionally desired, the command can be run with:
  - `--ssl-splits train val test`
- For a strict paper-style comparison, keep the default train-only SSL setting.

Installed dependencies for Stage 5:
- torch
- torchvision
- tqdm

Checks completed:
- `python3 -m py_compile train_stage5.py` passed.
- Import check for `torch`, `torchvision`, `tqdm`, and `train_stage5` passed.
- `uv run python train_stage5.py --help` passed.
- Full Stage 5 training was not run yet, so there are no Stage 5 metrics yet.

Expected Stage 5 outputs after training:
- `stage5_outputs/supervised_scratch/best_model.pt`
- `stage5_outputs/supervised_scratch/history.csv`
- `stage5_outputs/supervised_scratch/metrics.json`
- `stage5_outputs/supervised_scratch/val_predictions.csv`
- `stage5_outputs/supervised_scratch/test_predictions.csv`
- `stage5_outputs/imagenet_finetune/...`
- `stage5_outputs/simclr_pretrain/simclr_encoder.pt`
- `stage5_outputs/simclr_pretrain/simclr_history.csv`
- `stage5_outputs/simclr_finetune/...`
- `stage5_outputs/stage5_baseline_comparison.csv`
- `stage5_outputs/stage5_report.md`

Run a quick smoke test:

```bash
uv run python train_stage5.py --setup supervised_scratch --supervised-epochs 1 --num-workers 0
```

Run all three setups:

```bash
uv run python train_stage5.py --setup all
```

Run all three setups with shorter debugging epochs:

```bash
uv run python train_stage5.py --setup all --supervised-epochs 3 --simclr-epochs 5 --num-workers 0
```

## Overall Interpretation So Far

The pipeline is now complete through a first binary downstream baseline and has code ready for serious pretraining comparisons.

The most important methodological progress is that crops are split by source image before crop generation. This reduces a major leakage risk where train and test crops share style from the same source image.

The Stage 4 result is promising but not conclusive:
- Test balanced accuracy is 0.8485.
- Test ROC AUC is 0.8988.
- The classifier is not just predicting the majority class.
- But the result may still be inflated by crop-generation artifacts or differences between Pool A and Pool B. This is especially important because Pool B is algorithmically generated from negative images and includes some imperfect candidates.

The next serious answer should come from Stage 5:
- If ImageNet and SimCLR improve over supervised scratch, that supports representation learning.
- If SimCLR improves while preserving strict train-only SSL pretraining, that is stronger evidence that in-domain unlabeled crops are useful.
- If all methods perform similarly or all show suspiciously high performance, the next step should be stricter crop cleaning and artifact checks rather than claiming a strong disease model.
