# Grayscale Truth Test

This report explains the RGB-vs-grayscale rerun in simple terms. The main question is:

> If we remove color and stain cues, do the models still work well, especially when labels are limited?

If the answer is yes, that supports the idea that the models are learning real cell shape and morphology, not only color/stain shortcuts.

## How To Read The Numbers

- `Balanced accuracy`: higher is better. `1.0` is perfect. Around `0.5` means the model is close to guessing on a balanced two-class task.
- `ROC AUC`: higher is better. `1.0` is perfect. Around `0.5` means no useful ranking ability.
- `Drop from clean`: lower is better. `0.0` means the perturbation did not hurt. Bigger positive numbers mean the perturbation damaged performance more.
- A small negative drop can happen. It means the perturbed version happened to score slightly better than clean on that run. Treat that as "no meaningful harm."

## Main Takeaway

- Full-label best RGB model: `imagenet_finetune` at `0.9863`.
- Full-label best grayscale model: `imagenet_finetune` at `0.9950`.
- At `10%` labels, SimCLR minus Scratch in RGB: `0.1948`.
- At `10%` labels, SimCLR minus Scratch in grayscale: `0.2769`.
- Scratch stain/color drop at full labels: RGB `0.4245` vs grayscale `0.0044`.
- SimCLR stain/color drop at full labels: RGB `0.2639` vs grayscale `0.0000`.

In simpler terms: removing color did **not** make the whole pipeline collapse. That is evidence that morphology is carrying real signal. The grayscale rerun also removes most of the color/stain fragility we saw in the RGB robustness test.

## Stage 5: Full-Label Comparison

[stage5_rgb_vs_grayscale_balanced_accuracy.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/visual_reports/grayscale_truth_test/stage5/stage5_rgb_vs_grayscale_balanced_accuracy.png)

What this plot shows:
- Each group is one model family: Scratch, ImageNet, SimCLR.
- Blue/orange bars compare RGB and grayscale on the same full-label task.
- Higher bars are better.

What to look for:
- If grayscale bars were much lower, color was doing too much of the work.
- If grayscale bars stay close, morphology is still strong.

What it means here:
- Scratch improved slightly from `0.9311` to `0.9403`.
- ImageNet improved slightly from `0.9863` to `0.9950`.
- SimCLR improved slightly from `0.9851` to `0.9928`.
- So the grayscale rerun does not show collapse. That is the opposite of what we would expect if stain/color were the only real signal.

[stage5_rgb_vs_grayscale_roc_auc.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/visual_reports/grayscale_truth_test/stage5/stage5_rgb_vs_grayscale_roc_auc.png)

This second plot is the same idea, but for ROC AUC. Again, higher is better.

## Stage 6: Low-Label Comparison

[stage6_rgb_vs_grayscale_low_label_curves.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/visual_reports/grayscale_truth_test/stage6/stage6_rgb_vs_grayscale_low_label_curves.png)

What this plot shows:
- Each panel is one model family.
- The x-axis is how many labeled training images were kept: `100%`, `50%`, `20%`, `10%`.
- The y-axis is test balanced accuracy. Higher is better.
- RGB and grayscale are overlaid directly.

What to look for:
- Lines that stay high as labels shrink are stronger.
- If the grayscale line stays close to RGB, the model still works without color.
- The most important region is `20%` and `10%`, because that is where SSL is supposed to help.

What it means here in simple terms:
- At `10%`, Scratch gets `0.6844` in RGB and `0.6419` in grayscale.
- At `10%`, SimCLR gets `0.8791` in RGB and `0.9188` in grayscale.
- At `10%`, ImageNet gets `0.9305` in RGB and `0.9631` in grayscale.
- So the low-label benefit of SimCLR is still there even after removing color.

[stage6_simclr_gain_over_scratch_rgb_vs_grayscale.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/visual_reports/grayscale_truth_test/stage6/stage6_simclr_gain_over_scratch_rgb_vs_grayscale.png)

How to read this plot:
- This is not raw accuracy. It is `SimCLR minus Scratch`.
- Higher is better for SimCLR here.
- Values above zero mean SimCLR is helping over training from scratch.

What it means here:
- At `10%`, SimCLR beats Scratch by `0.1948` in RGB and `0.2769` in grayscale.
- That is strong evidence that SSL still carries useful value when labels are limited, even after removing color.

[stage6_fraction_10_rgb_vs_grayscale.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/visual_reports/grayscale_truth_test/stage6/stage6_fraction_10_rgb_vs_grayscale.png)

This is the same low-label story, but only for `10%` labels. It is the simplest plot to show someone if you want one fast summary of the SSL question.

## Stage 7: Full-Label Robustness

[stage7_full_rgb_vs_grayscale_robustness.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/visual_reports/grayscale_truth_test/stage7/stage7_full_rgb_vs_grayscale_robustness.png)

What this plot shows:
- Left side: RGB robustness.
- Right side: grayscale robustness.
- Each group is a perturbation: clean, blur, lower brightness, JPEG compression, and mild stain/color shift.
- Higher bars are better.

What to look for:
- If the bar under a perturbation drops a lot from clean, the model is sensitive to that problem.
- The `mild_stain_color` column is especially important here because this whole experiment is about removing color dependence.

[stage7_stain_color_drop_rgb_vs_grayscale.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/visual_reports/grayscale_truth_test/stage7/stage7_stain_color_drop_rgb_vs_grayscale.png)

How to read this plot:
- This uses `drop from clean`, so lower is better.
- A tall bar means the model was hurt a lot by stain/color change.
- A near-zero bar means the model is basically insensitive to that perturbation.

What it means here:
- Scratch stain/color drop fell from `0.4245` in RGB to `0.0044` in grayscale.
- ImageNet stain/color drop fell from `0.0227` in RGB to `0.0000` in grayscale.
- SimCLR stain/color drop fell from `0.2639` in RGB to `0.0000` in grayscale.
- This is one of the strongest pieces of evidence in the whole study. The huge stain/color fragility in RGB mostly disappears in grayscale.

[stage7_brightness_drop_rgb_vs_grayscale.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/visual_reports/grayscale_truth_test/stage7/stage7_brightness_drop_rgb_vs_grayscale.png)

This plot is also `drop from clean`, so lower is better.

What it means:
- Brightness still matters, especially for Scratch.
- But the stain/color problem is much more clearly solved by the grayscale rerun.

## Stage 7 Low-Label Robustness

[stage7_low_label_clean_rgb_vs_grayscale.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/visual_reports/grayscale_truth_test/stage7_low_label/stage7_low_label_clean_rgb_vs_grayscale.png)

What this plot shows:
- This is the low-label version of Stage 7, on clean test crops.
- Higher lines are better.
- It tells you whether low-label gains survive into the grayscale branch.

[stage7_low_label_clean_simclr_gain_rgb_vs_grayscale.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/visual_reports/grayscale_truth_test/stage7_low_label/stage7_low_label_clean_simclr_gain_rgb_vs_grayscale.png)

How to read this plot:
- Again, this is `SimCLR minus Scratch`.
- Higher is better for SimCLR.
- Values above zero mean SSL is still helping in the low-label regime.

What it means here:
- At `10%`, RGB SimCLR gain is `0.1948`.
- At `10%`, grayscale SimCLR gain is `0.2769`.
- So SSL is still valuable after color removal.

[stage7_low_label_stain_drop_rgb_vs_grayscale.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/visual_reports/grayscale_truth_test/stage7_low_label/stage7_low_label_stain_drop_rgb_vs_grayscale.png)

How to read this plot:
- This is `drop from clean` under the stain/color perturbation.
- Lower is better.
- Solid lines are RGB; dashed lines are grayscale.

What it means:
- In RGB, stain/color shifts can badly hurt low-label models, especially Scratch and sometimes SimCLR.
- In grayscale, those stain/color drops become tiny or disappear, which is exactly what we would hope if color shortcuts were removed.

[stage7_low_label_brightness_drop_rgb_vs_grayscale.png](/Users/shubhdeepdas/Desktop/Paper_Research/Sickel_cell/visual_reports/grayscale_truth_test/stage7_low_label/stage7_low_label_brightness_drop_rgb_vs_grayscale.png)

This is also `drop from clean`, so lower is better.

What it means:
- Brightness remains a real capture-quality issue even in grayscale.
- That tells us the grayscale rerun removed color dependence, but did not magically solve every acquisition problem.

## Bottom Line

In plain language:

- The grayscale rerun does **not** collapse. That means color/stain is not the whole story.
- The low-label SimCLR advantage remains positive in grayscale. That supports the idea that SSL is learning something useful beyond color shortcuts.
- The biggest RGB fragility was stain/color perturbation. That fragility mostly disappears in grayscale.
- Brightness still hurts weaker models, especially Scratch, so capture quality still matters.
- ImageNet remains the strongest overall baseline on this dataset, but SimCLR still shows real value over Scratch when labels are scarce.

So the simplest scientific reading is:

> there is real morphology signal here, and SSL still helps in the low-label setting, but ImageNet pretraining is currently the strongest overall starting point.
