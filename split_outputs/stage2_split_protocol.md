# Stage 2 Split Protocol

## Protocol
- Split unit: source image, before crop generation.
- Target split: 70% train, 15% val, 15% test.
- Stratification key: label plus stain.
- Seed: recorded in `source_image_splits.csv`.

## Split counts
- train: 399 / 569
- val: 85 / 569
- test: 85 / 569

## Label balance
- split=test, label=negative, count=22
- split=test, label=positive, count=63
- split=train, label=negative, count=103
- split=train, label=positive, count=296
- split=val, label=negative, count=22
- split=val, label=positive, count=63

## Stain balance
- split=test, label=negative, stain=unknown, count=22
- split=test, label=positive, stain=unknown, count=63
- split=train, label=negative, stain=unknown, count=103
- split=train, label=positive, stain=unknown, count=296
- split=val, label=negative, stain=unknown, count=22
- split=val, label=positive, stain=unknown, count=63

## Stain metadata note
- stain_source counts: {'not_available': 569}
- unknown stain rows: 569

If Field vs Leishman labels are available later, provide a CSV with columns `source_key,stain` or `source_id,stain` and rerun this script with `--stain-metadata path/to/file.csv`. This script will then stratify by those labels instead of `unknown`.

## Generated files
- `source_image_splits.csv`: one row per source image. Crops must inherit this split.
- `split_label_summary.csv`: split counts by positive/negative.
- `split_stain_summary.csv`: split counts by label and stain.
- `crop_annotation_splits.csv`: annotation-box rows joined to the source split when `eda_outputs/annotation_boxes.csv` exists.
- `stain_metadata_template.csv`: optional template to fill if Field/Leishman labels become available.
