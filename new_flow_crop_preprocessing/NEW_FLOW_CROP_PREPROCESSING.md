# New Flow Crop Preprocessing

This is the frozen preprocessing step for the new crop-based training flow.

## What it does

1. Resizes each full image so the shorter side becomes `1024` pixels.
2. Estimates background from the outer frame only.
3. Uses luminance-only distance to isolate the usable microscope field.
4. Cleans the field mask with morphology and a slight inward erosion.
5. Zero-fills everything outside the final mask.

## What it produces

- resized masked copies of the dataset
- a preprocessing manifest with image-level mask statistics
- a config file recording the frozen preprocessing settings

## Command to run

```bash
uv run python build_masked_resized_dataset.py
```

## Main output folder

- `datasets_masked_resized_1024/`

## Relevant validation board

- `simple_preprocess_test_5p5n/combined_band_sweep.png`

## Frozen settings

- resize target: `1024`
- background band half-width: `50`
- distance offset: `15.0`
- morphology radius: `1.5%`
- final erosion radius: `1%`
- distance metric: luminance-only
