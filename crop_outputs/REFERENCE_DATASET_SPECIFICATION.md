# Reference Dataset Specification (Experiment 3)

## 1. Overview & Lineage
This document records the exact crop dataset reconstructed for **Experiment 3: Reference Model Training & Shortcut Reliance Analysis**.

### Manifests & Checksums
- **Pool A Manifest**: `crop_outputs/pool_a_labeled_sickled_manifest.csv`
  - SHA-256: `1e849ffd8350578e192948e1d0907699491c228ba1de6a96ff234f00d7c81d27`
  - Total Crops: **1,249** (Sickled)
- **Pool B Manifest**: `crop_outputs/pool_b_labeled_non_sickled_manifest.csv`
  - SHA-256: `61fab3f079a4d9650f51f1646b364499275f3d73c255859b971f9c95bc6cf9eb`
  - Total Crops: **1,468** (Non-Sickled)
- **Combined Dataset**: **2,717 crops**

## 2. Partition Counts & Split Integrity

| Split Partition | Pool A (Sickled) | Pool B (Non-Sickled) | Total Crops | Unique Sources | Source Leakage Check |
|---|---:|---:|---:|---:|:---:|
| **Train** | 917 | 1,014 | 1,931 | 382 | **PASS (0)** |
| **Validation** | 150 | 232 | 382 | 80 | **PASS (0)** |
| **Test** | 182 | 222 | 404 | 81 | **PASS (0)** |
| **Total** | **1,249** | **1,468** | **2,717** | **543** | **PASS (0)** |

## 3. Explicit Comparison to Historical Stage 3 Manifest

| Attribute | Historical Stage 3 Pool | Experiment 3 Reference Pool | Variance / Delta | Methodological Explanation |
|---|---:|---:|---:|---|
| **Pool A (Sickled)** | 1,249 crops | 1,249 crops | **0 (0.0%)** | Exact 1:1 match across all bounding boxes. |
| **Pool B (Non-Sickled)** | 1,462 crops | 1,468 crops | **+6 crops (+0.4%)** | 6 additional candidates retained due to pure-Python connected component boundary handling vs OpenCV subpixel threshold rounding. |
| **Total Labeled Pool** | 2,711 crops | 2,717 crops | **+6 crops (+0.2%)** | Completely documented; historical models were evaluated on 1,462 crops. |

## 4. Preservation & Immutability
- All crop JPEGs are stored at `crop_outputs/pool_a_labeled_sickled/` and `crop_outputs/pool_b_labeled_non_sickled/`.
- Crop resolution: **128 x 128 pixels**, 3-channel RGB (sRGB profile).
- This reference dataset is frozen prior to reference model training.
