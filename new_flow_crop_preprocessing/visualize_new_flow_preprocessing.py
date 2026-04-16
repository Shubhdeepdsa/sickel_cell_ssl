from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from new_flow_preprocessing_shared import compute_preprocessing_stages
from prototype_border_preprocessing_pipeline import DEFAULT_BACKGROUND_BAND_HALF_WIDTH, DEFAULT_DISTANCE_OFFSET, DEFAULT_FINAL_MASK_EROSION_RATIO, DEFAULT_MORPH_RADIUS_RATIO, DEFAULT_NORMALIZE_SHORTER_DIM, open_rgb_image, read_csv, zero_outside_mask


def choose_test_image(project_root: Path, label: str) -> dict[str, str]:
    rows = read_csv(project_root / "split_outputs" / "source_image_splits.csv")
    candidates = [row for row in rows if row["split"] == "test" and row["label"] == label]
    if not candidates:
        raise ValueError(f"No test images found for label={label}")
    return sorted(candidates, key=lambda row: int(row["source_id"]))[0]


def contour_overlay(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    boundary = mask & ~ndimage.binary_erosion(mask)
    ys, xs = np.nonzero(boundary)
    ax.scatter(xs, ys, s=0.35, c=[np.array(color) / 255.0], alpha=0.35)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3].copy()
    plt.close(fig)
    return array


def make_board(
    project_root: Path,
    row: dict[str, str],
    output_dir: Path,
    background_band_half_width: int,
    distance_offset: float,
    morph_radius_ratio: float,
    final_mask_erosion_ratio: float,
) -> None:
    raw_path = project_root / row["raw_path"]
    image = open_rgb_image(raw_path)
    stages = compute_preprocessing_stages(
        image,
        background_band_half_width=background_band_half_width,
        distance_offset=distance_offset,
        morph_radius_ratio=morph_radius_ratio,
        final_mask_erosion_ratio=final_mask_erosion_ratio,
    )
    zero_filled = zero_outside_mask(stages.resized_image, stages.final_mask)

    panels = [
        (image, "Original"),
        (stages.resized_image, f"Resized to short side {DEFAULT_NORMALIZE_SHORTER_DIM}"),
        (Image.fromarray(contour_overlay(stages.resized_image, stages.background_band, (255, 128, 0))), "Background band"),
        (Image.fromarray((stages.initial_foreground.astype(np.uint8) * 255)), f"Initial foreground\nthr={stages.threshold:.1f}"),
        (Image.fromarray(contour_overlay(stages.resized_image, stages.field_mask, (0, 255, 0))), "Field mask"),
        (Image.fromarray(np.asarray(zero_filled)), "Final zero-filled image"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, (panel, title) in zip(axes.flat, panels, strict=True):
        ax.imshow(panel)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(
        f"{row['source_key']} ({row['label']})\n"
        f"band=±{background_band_half_width}, field={stages.field_mask.mean():.3f}, final={stages.final_mask.mean():.3f}, "
        f"resize={stages.resize_info.resized_width}x{stages.resize_info.resized_height}",
        fontsize=12,
    )
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{row['source_key'].replace(':', '_')}_preprocessing_steps.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the frozen preprocessing chain on one positive and one negative test image.")
    parser.add_argument("--output-dir", type=Path, default=Path("new_flow_crop_preprocessing/step_visuals"))
    parser.add_argument("--background-band-half-width", type=int, default=DEFAULT_BACKGROUND_BAND_HALF_WIDTH)
    parser.add_argument("--distance-offset", type=float, default=DEFAULT_DISTANCE_OFFSET)
    parser.add_argument("--morph-radius-ratio", type=float, default=DEFAULT_MORPH_RADIUS_RATIO)
    parser.add_argument("--final-mask-erosion-ratio", type=float, default=DEFAULT_FINAL_MASK_EROSION_RATIO)
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    output_dir = (project_root / args.output_dir).resolve()

    for label in ("positive", "negative"):
        row = choose_test_image(project_root, label)
        make_board(
            project_root=project_root,
            row=row,
            output_dir=output_dir,
            background_band_half_width=args.background_band_half_width,
            distance_offset=args.distance_offset,
            morph_radius_ratio=args.morph_radius_ratio,
            final_mask_erosion_ratio=args.final_mask_erosion_ratio,
        )

    print(f"Wrote step visualizations to {output_dir}")


if __name__ == "__main__":
    main()
