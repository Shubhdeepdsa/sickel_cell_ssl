# Frozen Preprocessing Baseline

This records the preprocessing pipeline we settled on after the border-artifact debugging.

## Why this was needed

The earlier crop pipeline was unreliable for full-image use because it depended on annotation-style crops and could leak border artifacts into the model. The final preprocessing step is meant to make the raw microscope images more consistent before any later sliding-window crop generation or SSL training.

## Final preprocessing steps

1. Resize each full image so the shorter side becomes `1024` pixels.
2. Estimate the background only from the outer frame of the resized image.
3. Use `L*` luminance distance from the frame background as the field-separation signal.
4. Clean the mask with closing, opening, largest-component selection, and convex hull.
5. Erode the final field mask slightly to remove the uncertain rim.
6. Zero-fill everything outside the final mask.

## Frozen settings

- resize target: `1024` on the shorter side
- background band half-width: `50`
- background offset: `15.0`
- morphology radius: `1.5%` of the shorter image dimension
- final erosion radius: `1%` of the shorter image dimension
- distance metric: luminance-only

## What the pipeline does

- preserves the usable circular microscope field
- suppresses the outer black / gradient border
- avoids the square-mask failure seen in earlier versions
- produces a cleaner full-image base for later crop generation

## How to run the validation sweep

The best compact visual validation is the combined 5 positive + 5 negative board:

```bash
uv run python sweep_background_band_detection.py \
  --output-dir simple_preprocess_test_5p5n \
  --band-half-widths 50 \
  --source-keys \
  positive:101 positive:11 positive:110 positive:112 positive:113 \
  negative:100 negative:107 negative:108 negative:116 negative:122
```

The resulting combined visual board is:

- `simple_preprocess_test_5p5n/combined_band_sweep.png`

The numeric summary is:

- `simple_preprocess_test_5p5n/sweep_summary.csv`

## Relevant folders to review

- `background_band_detection_combined_lonly_band50/`
- `isolate_positive_264_framebg_lonly/`
- `isolate_two_lonly_all_band50/`
- `simple_preprocess_test_5p5n/`

## Notes

- The temporary `datasets_masked_resized_1024/` copy was only a working artifact and is not part of the frozen baseline.
- The frozen preprocessing logic lives in `prototype_border_preprocessing_pipeline.py`.
- The validation-board generation lives in `sweep_background_band_detection.py`.
