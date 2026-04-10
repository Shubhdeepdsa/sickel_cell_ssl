# Stage 5-7 Visual Summary

This report collects the main plots needed to understand what worked, where performance comes from, and how the three model families compare.

## Stage 5
- Best full-label test crop balanced accuracy: `0.9863` from `imagenet_finetune`.
- Main comparison plot: [stage5_baseline_metrics.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/visual_reports/stage5_stage6_stage7/stage5/stage5_baseline_metrics.png)
- Simple winner plot: [stage5_test_crop_balanced_accuracy.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/visual_reports/stage5_stage6_stage7/stage5/stage5_test_crop_balanced_accuracy.png)

## Stage 6
- At `10%` labels, best test crop balanced accuracy: `0.9305` from `imagenet_finetune`.
- SimCLR minus Scratch at `10%`: `0.1948` balanced accuracy.
- Main low-label curve: [stage6_test_crop_balanced_accuracy.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/visual_reports/stage5_stage6_stage7/stage6/stage6_test_crop_balanced_accuracy.png)
- Gain-over-scratch plot: [stage6_gain_over_scratch.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/visual_reports/stage5_stage6_stage7/stage6/stage6_gain_over_scratch.png)
- Heatmap: [stage6_accuracy_heatmap.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/visual_reports/stage5_stage6_stage7/stage6/stage6_accuracy_heatmap.png)

## Stage 7
- Biggest robustness failure on full-label models: `supervised_scratch` under `mild_stain_color` with drop `0.4245`.
- Perturbation comparison: [stage7_balanced_accuracy_by_perturbation.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/visual_reports/stage5_stage6_stage7/stage7/stage7_balanced_accuracy_by_perturbation.png)
- Drop-from-clean plot: [stage7_drop_from_clean.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/visual_reports/stage5_stage6_stage7/stage7/stage7_drop_from_clean.png)

## Stage 7 Low-Label
- At `10%` labels on clean test crops, SimCLR minus Scratch: `0.1948` balanced accuracy.
- Most important plot set lives in [stage7_low_label](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/visual_reports/stage5_stage6_stage7/stage7_low_label)
- Example clean-fraction comparison: [stage7_low_label_clean.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/visual_reports/stage5_stage6_stage7/stage7_low_label/stage7_low_label_clean.png)
- Example low-label SimCLR-vs-Scratch gain: [stage7_low_label_gain_simclr_vs_scratch_clean.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/visual_reports/stage5_stage6_stage7/stage7_low_label/stage7_low_label_gain_simclr_vs_scratch_clean.png)

## Overall
- Overall clean low-label view: [overall_low_label_clean_performance.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/visual_reports/stage5_stage6_stage7/summary/overall_low_label_clean_performance.png)
- Robustness snapshot: [overall_robustness_snapshot.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/visual_reports/stage5_stage6_stage7/summary/overall_robustness_snapshot.png)
