# Experiment 3 — ImageNet ResNet-18 Reference Model Report

## 1. Executive Summary & Provenance
This report documents the training, validation selection, and baseline evaluation of the primary reference model for **Experiment 3: Controlled Intervention & Shortcut Reliance Analysis**.

### Critical Lineage & Methodology Statement
- **Model Identity**: `Experiment 3 — ImageNet ResNet-18 Reference Model`.
- **Distinct from Historical Models**: This model is a newly trained reference architecture initialized from Torchvision ImageNet weights (`ResNet18_Weights.DEFAULT`). It is **not** the historical model, nor is it assumed to replicate historical 98–99% figures unless separately demonstrated.
- **Protocol Status**: Frozen exploratory reference model prior to Stage 3 artifact interventions.
- **Checkpoint SHA-256**: `83cd725076f60899bea0dfe4b7bbaa729ff82a60293b4e799d3082a6ae0419bb`
- **Best Validation Epoch**: Epoch **13** (Val Balanced Accuracy: **99.8%**)

### Baseline Quality Gate Assessment
- **Baseline Evaluation Gate Criterion**: Balanced accuracy $\ge 75.0\%$ on unchanged test crops.
- **Status**: **PASSED (>= 75.0%)** — The reference model achieved **99.5% balanced accuracy** on test crops (**100.0% at source level**).
- **Decision**: Eligible to proceed to Stage 3 artifact interventions.

---

## 2. Unchanged Baseline Test Performance

| Evaluation Level | Decision Threshold ($th$) | Balanced Accuracy | Accuracy | Sickled Recall (TPR) | Non-Sickled Spec (TNR) | ROC AUC |
|---|:---:|---:|---:|---:|---:|---:|
| **Crop Level (Standard)** | $th = 0.50$ | **99.5%** | 99.5% | 99.5% | 99.5% | 1.000 |
| **Crop Level (Val-Tuned)** | $th^* = 0.50$ | **99.5%** | 99.5% | 99.5% | 99.5% | 1.000 |
| **Source Level (Standard)** | $th = 0.50$ | **100.0%** | 100.0% | 100.0% | 100.0% | 1.000 |
| **Source Level (Val-Tuned)** | $th^* = 0.50$ | **100.0%** | 100.0% | 100.0% | 100.0% | 1.000 |

### Crop-Level Confusion Matrix (Standard $th = 0.50$)
- **True Positives (Sickled correctly identified)**: 181 / 182 (99.5%)
- **True Negatives (Non-Sickled correctly identified)**: 221 / 222 (99.5%)
- **False Positives (Non-Sickled predicted as Sickled)**: 1
- **False Negatives (Sickled predicted as Non-Sickled)**: 1

---

## 3. Training Dynamics & Convergence History

| Epoch | Train Loss | Val Bal. Acc. | Val Accuracy | Val TPR | Val TNR | Time (s) | Best Checkpoint |
|:---:|---:|---:|---:|---:|---:|---:|:---:|
| 01 | 0.1984 | 96.3% | 95.5% | 100.0% | 92.7% | 19.8s |  |
| 02 | 0.0722 | 97.5% | 97.1% | 99.3% | 95.7% | 18.0s |  |
| 03 | 0.0446 | 98.8% | 98.7% | 99.3% | 98.3% | 19.1s |  |
| 04 | 0.0323 | 98.5% | 98.7% | 97.3% | 99.6% | 18.7s |  |
| 05 | 0.0143 | 98.9% | 99.0% | 98.7% | 99.1% | 21.2s |  |
| 06 | 0.0220 | 97.8% | 98.2% | 96.0% | 99.6% | 22.9s |  |
| 07 | 0.0189 | 97.8% | 98.2% | 96.0% | 99.6% | 23.3s |  |
| 08 | 0.0119 | 98.7% | 98.7% | 98.7% | 98.7% | 23.8s |  |
| 09 | 0.0072 | 99.1% | 99.2% | 98.7% | 99.6% | 24.2s |  |
| 10 | 0.0079 | 98.2% | 97.9% | 99.3% | 97.0% | 24.3s |  |
| 11 | 0.0051 | 98.3% | 98.2% | 98.7% | 97.8% | 24.6s |  |
| 12 | 0.0056 | 98.9% | 98.7% | 100.0% | 97.8% | 24.7s |  |
| 13 | 0.0029 | 99.8% | 99.7% | 100.0% | 99.6% | 24.8s | Yes |
| 14 | 0.0033 | 99.8% | 99.7% | 100.0% | 99.6% | 23.4s |  |
| 15 | 0.0023 | 99.4% | 99.2% | 100.0% | 98.7% | 23.3s |  |
| 16 | 0.0037 | 99.5% | 99.5% | 99.3% | 99.6% | 25.2s |  |
| 17 | 0.0029 | 99.5% | 99.5% | 99.3% | 99.6% | 24.5s |  |
| 18 | 0.0024 | 99.5% | 99.5% | 99.3% | 99.6% | 23.5s |  |
| 19 | 0.0019 | 99.5% | 99.5% | 99.3% | 99.6% | 23.5s |  |
| 20 | 0.0025 | 99.5% | 99.5% | 99.3% | 99.6% | 23.2s |  |

---

## 4. Software & Hardware Environment
- **Operating Platform**: `macOS-27.2-arm64-arm-64bit` (`arm64`)
- **Python Version**: `3.12.14`
- **PyTorch Version**: `2.14.1`
- **Apple MPS Acceleration**: `True` (Device: `mps`)
- **Training Seeds**: All random seeds locked to `42`.

---

## 5. Next Steps: Stage 3 Artifact Interventions
With the baseline evaluation gate passed and the model weights frozen in `best_model.pt` (`83cd725076f60899...`), Stage 3 will execute:
1. **Subgroup Stratification**: Measure natural performance disparity across border-positive vs clean negative crops.
2. **Border Removal ($T_{\text{border}}$)** vs **Matched Negative Control ($T_{\text{control}}$)**.
3. **Plasma Equalization ($T_{\text{plasma}}$)** vs **Matched Luminance Control ($T_{\text{photo-control}}$)**.
4. **Paired Shift Analysis**: Track continuous change in predicted probability $\Delta p = p(\text{sickled} \mid x_{\text{intervened}}) - p(\text{sickled} \mid x_{\text{orig}})$ with 2,000 source-cluster bootstrap CIs.
