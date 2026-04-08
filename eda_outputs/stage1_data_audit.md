# Stage 1 Data Audit

## Dataset paths
- root: `/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/datasets`
- positive labelled overlays: `/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/datasets/Positive/Labelled`
- positive raw/unlabelled: `/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/datasets/Positive/Unlabelled`
- negative clear: `/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/datasets/Negative/Clear`
- ImageMagick pixel analysis: `available at /opt/homebrew/bin/magick`

## Image counts
- Positive raw images for modeling: 422
- Positive labelled overlay images: 422
- Negative clear images: 147
- Positive labelled/unlabelled filename pairs: 422
- Total image files scanned: 991

## Image dimensions
- Median width: 1000.0
- Median height: 1009.0
- Width range: 469.0 to 4160.0
- Height range: 436.0 to 3440.0

## Explicit stain distribution
- not_encoded: 569

No stain labels are encoded in the current directory or filename structure unless the value above says otherwise. The color cast distribution below is only a pixel-color proxy, not a biological stain label.

## Heuristic color cast distribution
- balanced_mixed_cast: 238
- blue_purple_cast: 120
- green_cyan_cast: 11
- pink_red_cast: 200

## Image quality buckets
- low_contrast: 4
- usable: 565

## Annotation storage
No separate annotation files were found. The annotations appear to be black rectangle overlays baked directly into `Positive/Labelled/*.jpg`, with matching raw images in `Positive/Unlabelled/*.jpg`.

## Detected annotation boxes
- Labelled images with at least one detected box: 421 / 422
- Total detected boxes: 1408
- Likely crop-candidate boxes: 993 / 1408 (70.5%)

Tightness buckets:
- likely_crop_candidate: 993
- loose_or_empty: 159
- overfilled: 256

Crop-readiness note: Candidate boxes were detected, but crop extraction should first convert raster overlays into coordinate files and verify a sample manually.

## Generated files
- `eda_outputs/image_audit.csv`: one row per scanned image with dimensions, quality metrics, and stain/color proxies.
- `eda_outputs/annotation_boxes.csv`: one row per detected raster box in the labelled positive images.
- `eda_outputs/stage1_data_audit.md`: this report.
