# Stage 5 Interpretability Report

- Model root: `/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/stage5_grayscale_outputs`
- Output directory: `/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale`
- Device: `mps`
- Setups: `supervised_scratch, imagenet_finetune, simclr_finetune`
- Samples analyzed: `4`

## Prediction Summary
### `A__positive_310__0002.jpg`
- Actual label: `sickled`
- Source: `positive:310`
- `supervised_scratch`: predicted `sickled` (P[sickled]=0.9697), actual `sickled`.
- `imagenet_finetune`: predicted `sickled` (P[sickled]=0.9999), actual `sickled`.
- `simclr_finetune`: predicted `sickled` (P[sickled]=0.9995), actual `sickled`.

### `B__negative_100__0004.jpg`
- Actual label: `non_sickled`
- Source: `negative:100`
- `supervised_scratch`: predicted `non_sickled` (P[sickled]=0.0000), actual `non_sickled`.
- `imagenet_finetune`: predicted `non_sickled` (P[sickled]=0.0000), actual `non_sickled`.
- `simclr_finetune`: predicted `non_sickled` (P[sickled]=0.0000), actual `non_sickled`.

### `B__negative_37__0012.jpg`
- Actual label: `non_sickled`
- Source: `negative:37`
- `supervised_scratch`: predicted `sickled` (P[sickled]=0.7413), actual `non_sickled`.
- `imagenet_finetune`: predicted `sickled` (P[sickled]=0.8314), actual `non_sickled`.
- `simclr_finetune`: predicted `sickled` (P[sickled]=0.8444), actual `non_sickled`.

### `A__positive_20__0009.jpg`
- Actual label: `sickled`
- Source: `positive:20`
- `supervised_scratch`: predicted `sickled` (P[sickled]=0.5588), actual `sickled`.
- `imagenet_finetune`: predicted `sickled` (P[sickled]=0.9252), actual `sickled`.
- `simclr_finetune`: predicted `non_sickled` (P[sickled]=0.1051), actual `sickled`.

## Visual Outputs

For each sample and setup, the script writes:
- `Grad-CAM`: which spatial regions most support the predicted class.
- `Occlusion`: where masking causes the largest drop in predicted-class probability.
- `Nearest neighbors`: closest training crops in each model's penultimate feature space.
- `Perturbation grid`: the same crop under grayscale/color/quality perturbations with updated probabilities.
- `Shortcut sensitivity`: sample-level probability curves across nuisance perturbations for all setups.

### Sample `A__positive_310__0002.jpg`

- Source: `positive:310`
- Actual label: `sickled`
- Crop path: `/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/crop_outputs/pool_a_labeled_sickled/test/A__positive_310__0002.jpg`
- `supervised_scratch` prediction: `sickled` with `P[sickled]=0.9697`.
- `supervised_scratch` Grad-CAM: [supervised_scratch_gradcam.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/A__positive_310__0002/supervised_scratch_gradcam.png)
- `supervised_scratch` occlusion: [supervised_scratch_occlusion.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/A__positive_310__0002/supervised_scratch_occlusion.png)
- `supervised_scratch` nearest neighbors: [supervised_scratch_neighbors.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/A__positive_310__0002/supervised_scratch_neighbors.png)
- `supervised_scratch` perturbation grid: [supervised_scratch_perturbations.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/A__positive_310__0002/supervised_scratch_perturbations.png)
- `imagenet_finetune` prediction: `sickled` with `P[sickled]=0.9999`.
- `imagenet_finetune` Grad-CAM: [imagenet_finetune_gradcam.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/A__positive_310__0002/imagenet_finetune_gradcam.png)
- `imagenet_finetune` occlusion: [imagenet_finetune_occlusion.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/A__positive_310__0002/imagenet_finetune_occlusion.png)
- `imagenet_finetune` nearest neighbors: [imagenet_finetune_neighbors.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/A__positive_310__0002/imagenet_finetune_neighbors.png)
- `imagenet_finetune` perturbation grid: [imagenet_finetune_perturbations.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/A__positive_310__0002/imagenet_finetune_perturbations.png)
- `simclr_finetune` prediction: `sickled` with `P[sickled]=0.9995`.
- `simclr_finetune` Grad-CAM: [simclr_finetune_gradcam.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/A__positive_310__0002/simclr_finetune_gradcam.png)
- `simclr_finetune` occlusion: [simclr_finetune_occlusion.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/A__positive_310__0002/simclr_finetune_occlusion.png)
- `simclr_finetune` nearest neighbors: [simclr_finetune_neighbors.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/A__positive_310__0002/simclr_finetune_neighbors.png)
- `simclr_finetune` perturbation grid: [simclr_finetune_perturbations.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/A__positive_310__0002/simclr_finetune_perturbations.png)
- Sample-level shortcut sensitivity: [shortcut_sensitivity.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/A__positive_310__0002/shortcut_sensitivity.png)

### Sample `B__negative_100__0004.jpg`

- Source: `negative:100`
- Actual label: `non_sickled`
- Crop path: `/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/crop_outputs/pool_b_labeled_non_sickled/test/B__negative_100__0004.jpg`
- `supervised_scratch` prediction: `non_sickled` with `P[sickled]=0.0000`.
- `supervised_scratch` Grad-CAM: [supervised_scratch_gradcam.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/B__negative_100__0004/supervised_scratch_gradcam.png)
- `supervised_scratch` occlusion: [supervised_scratch_occlusion.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/B__negative_100__0004/supervised_scratch_occlusion.png)
- `supervised_scratch` nearest neighbors: [supervised_scratch_neighbors.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/B__negative_100__0004/supervised_scratch_neighbors.png)
- `supervised_scratch` perturbation grid: [supervised_scratch_perturbations.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/B__negative_100__0004/supervised_scratch_perturbations.png)
- `imagenet_finetune` prediction: `non_sickled` with `P[sickled]=0.0000`.
- `imagenet_finetune` Grad-CAM: [imagenet_finetune_gradcam.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/B__negative_100__0004/imagenet_finetune_gradcam.png)
- `imagenet_finetune` occlusion: [imagenet_finetune_occlusion.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/B__negative_100__0004/imagenet_finetune_occlusion.png)
- `imagenet_finetune` nearest neighbors: [imagenet_finetune_neighbors.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/B__negative_100__0004/imagenet_finetune_neighbors.png)
- `imagenet_finetune` perturbation grid: [imagenet_finetune_perturbations.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/B__negative_100__0004/imagenet_finetune_perturbations.png)
- `simclr_finetune` prediction: `non_sickled` with `P[sickled]=0.0000`.
- `simclr_finetune` Grad-CAM: [simclr_finetune_gradcam.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/B__negative_100__0004/simclr_finetune_gradcam.png)
- `simclr_finetune` occlusion: [simclr_finetune_occlusion.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/B__negative_100__0004/simclr_finetune_occlusion.png)
- `simclr_finetune` nearest neighbors: [simclr_finetune_neighbors.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/B__negative_100__0004/simclr_finetune_neighbors.png)
- `simclr_finetune` perturbation grid: [simclr_finetune_perturbations.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/B__negative_100__0004/simclr_finetune_perturbations.png)
- Sample-level shortcut sensitivity: [shortcut_sensitivity.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/B__negative_100__0004/shortcut_sensitivity.png)

### Sample `B__negative_37__0012.jpg`

- Source: `negative:37`
- Actual label: `non_sickled`
- Crop path: `/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/crop_outputs/pool_b_labeled_non_sickled/test/B__negative_37__0012.jpg`
- `supervised_scratch` prediction: `sickled` with `P[sickled]=0.7413`.
- `supervised_scratch` Grad-CAM: [supervised_scratch_gradcam.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/B__negative_37__0012/supervised_scratch_gradcam.png)
- `supervised_scratch` occlusion: [supervised_scratch_occlusion.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/B__negative_37__0012/supervised_scratch_occlusion.png)
- `supervised_scratch` nearest neighbors: [supervised_scratch_neighbors.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/B__negative_37__0012/supervised_scratch_neighbors.png)
- `supervised_scratch` perturbation grid: [supervised_scratch_perturbations.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/B__negative_37__0012/supervised_scratch_perturbations.png)
- `imagenet_finetune` prediction: `sickled` with `P[sickled]=0.8314`.
- `imagenet_finetune` Grad-CAM: [imagenet_finetune_gradcam.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/B__negative_37__0012/imagenet_finetune_gradcam.png)
- `imagenet_finetune` occlusion: [imagenet_finetune_occlusion.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/B__negative_37__0012/imagenet_finetune_occlusion.png)
- `imagenet_finetune` nearest neighbors: [imagenet_finetune_neighbors.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/B__negative_37__0012/imagenet_finetune_neighbors.png)
- `imagenet_finetune` perturbation grid: [imagenet_finetune_perturbations.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/B__negative_37__0012/imagenet_finetune_perturbations.png)
- `simclr_finetune` prediction: `sickled` with `P[sickled]=0.8444`.
- `simclr_finetune` Grad-CAM: [simclr_finetune_gradcam.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/B__negative_37__0012/simclr_finetune_gradcam.png)
- `simclr_finetune` occlusion: [simclr_finetune_occlusion.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/B__negative_37__0012/simclr_finetune_occlusion.png)
- `simclr_finetune` nearest neighbors: [simclr_finetune_neighbors.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/B__negative_37__0012/simclr_finetune_neighbors.png)
- `simclr_finetune` perturbation grid: [simclr_finetune_perturbations.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/B__negative_37__0012/simclr_finetune_perturbations.png)
- Sample-level shortcut sensitivity: [shortcut_sensitivity.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/B__negative_37__0012/shortcut_sensitivity.png)

### Sample `A__positive_20__0009.jpg`

- Source: `positive:20`
- Actual label: `sickled`
- Crop path: `/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/crop_outputs/pool_a_labeled_sickled/test/A__positive_20__0009.jpg`
- `supervised_scratch` prediction: `sickled` with `P[sickled]=0.5588`.
- `supervised_scratch` Grad-CAM: [supervised_scratch_gradcam.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/A__positive_20__0009/supervised_scratch_gradcam.png)
- `supervised_scratch` occlusion: [supervised_scratch_occlusion.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/A__positive_20__0009/supervised_scratch_occlusion.png)
- `supervised_scratch` nearest neighbors: [supervised_scratch_neighbors.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/A__positive_20__0009/supervised_scratch_neighbors.png)
- `supervised_scratch` perturbation grid: [supervised_scratch_perturbations.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/A__positive_20__0009/supervised_scratch_perturbations.png)
- `imagenet_finetune` prediction: `sickled` with `P[sickled]=0.9252`.
- `imagenet_finetune` Grad-CAM: [imagenet_finetune_gradcam.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/A__positive_20__0009/imagenet_finetune_gradcam.png)
- `imagenet_finetune` occlusion: [imagenet_finetune_occlusion.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/A__positive_20__0009/imagenet_finetune_occlusion.png)
- `imagenet_finetune` nearest neighbors: [imagenet_finetune_neighbors.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/A__positive_20__0009/imagenet_finetune_neighbors.png)
- `imagenet_finetune` perturbation grid: [imagenet_finetune_perturbations.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/A__positive_20__0009/imagenet_finetune_perturbations.png)
- `simclr_finetune` prediction: `non_sickled` with `P[sickled]=0.1051`.
- `simclr_finetune` Grad-CAM: [simclr_finetune_gradcam.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/A__positive_20__0009/simclr_finetune_gradcam.png)
- `simclr_finetune` occlusion: [simclr_finetune_occlusion.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/A__positive_20__0009/simclr_finetune_occlusion.png)
- `simclr_finetune` nearest neighbors: [simclr_finetune_neighbors.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/A__positive_20__0009/simclr_finetune_neighbors.png)
- `simclr_finetune` perturbation grid: [simclr_finetune_perturbations.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/A__positive_20__0009/simclr_finetune_perturbations.png)
- Sample-level shortcut sensitivity: [shortcut_sensitivity.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/interpretability_outputs/stage5_grayscale/samples/A__positive_20__0009/shortcut_sensitivity.png)

