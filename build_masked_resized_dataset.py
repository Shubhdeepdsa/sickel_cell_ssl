from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from new_flow_preprocessing_shared import compute_preprocessing_stages
from prototype_border_preprocessing_pipeline import DEFAULT_BACKGROUND_BAND_HALF_WIDTH, DEFAULT_DISTANCE_OFFSET, DEFAULT_FINAL_MASK_EROSION_RATIO, DEFAULT_MORPH_RADIUS_RATIO, open_rgb_image, resize_image_and_boxes, zero_outside_mask


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resize_with_known_size(image_path: Path, width: int, height: int):
    image = open_rgb_image(image_path)
    if image.size == (width, height):
        return image
    return image.resize((width, height), Image.Resampling.BILINEAR)


def save_rgb(image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(output_path, quality=95)
    else:
        image.save(output_path)


def process_positive_pair(
    project_root: Path,
    unlabelled_path: Path,
    output_root: Path,
    background_band_half_width: int,
    distance_offset: float,
    morph_radius_ratio: float,
    final_mask_erosion_ratio: float,
) -> list[dict[str, object]]:
    labelled_path = project_root / "datasets" / "Positive" / "Labelled" / unlabelled_path.name
    raw_image = open_rgb_image(unlabelled_path)
    stages = compute_preprocessing_stages(
        raw_image,
        background_band_half_width=background_band_half_width,
        distance_offset=distance_offset,
        morph_radius_ratio=morph_radius_ratio,
        final_mask_erosion_ratio=final_mask_erosion_ratio,
    )

    resized_labelled = resize_with_known_size(labelled_path, stages.resize_info.resized_width, stages.resize_info.resized_height)
    masked_raw = zero_outside_mask(stages.resized_image, stages.final_mask)
    masked_labelled = zero_outside_mask(resized_labelled, stages.final_mask)

    raw_output = output_root / "Positive" / "Unlabelled" / unlabelled_path.name
    labelled_output = output_root / "Positive" / "Labelled" / labelled_path.name
    save_rgb(masked_raw, raw_output)
    save_rgb(masked_labelled, labelled_output)

    common = {
        "source_name": unlabelled_path.name,
        "background_band_half_width": background_band_half_width,
        "distance_offset": distance_offset,
        "morph_radius_ratio": morph_radius_ratio,
        "final_mask_erosion_ratio": final_mask_erosion_ratio,
        "erosion_radius_px": stages.erosion_radius,
        "image_type": stages.descriptor.image_type,
        "background_sigma": round(stages.descriptor.sigma_l, 6),
        "field_fraction": round(float(stages.field_mask.mean()), 6),
        "final_fraction": round(float(stages.final_mask.mean()), 6),
        "original_width": stages.resize_info.original_width,
        "original_height": stages.resize_info.original_height,
        "resized_width": stages.resize_info.resized_width,
        "resized_height": stages.resize_info.resized_height,
        "scale_x": round(stages.resize_info.scale_x, 6),
        "scale_y": round(stages.resize_info.scale_y, 6),
    }
    return [
        {
            **common,
            "dataset_split": "Positive/Unlabelled",
            "input_path": str(unlabelled_path.relative_to(project_root)),
            "output_path": str(raw_output.relative_to(project_root)),
        },
        {
            **common,
            "dataset_split": "Positive/Labelled",
            "input_path": str(labelled_path.relative_to(project_root)),
            "output_path": str(labelled_output.relative_to(project_root)),
        },
    ]


def process_negative_image(
    project_root: Path,
    negative_path: Path,
    output_root: Path,
    background_band_half_width: int,
    distance_offset: float,
    morph_radius_ratio: float,
    final_mask_erosion_ratio: float,
) -> dict[str, object]:
    image = open_rgb_image(negative_path)
    stages = compute_preprocessing_stages(
        image,
        background_band_half_width=background_band_half_width,
        distance_offset=distance_offset,
        morph_radius_ratio=morph_radius_ratio,
        final_mask_erosion_ratio=final_mask_erosion_ratio,
    )
    masked = zero_outside_mask(stages.resized_image, stages.final_mask)
    output_path = output_root / "Negative" / "Clear" / negative_path.name
    save_rgb(masked, output_path)
    return {
        "source_name": negative_path.name,
        "dataset_split": "Negative/Clear",
        "input_path": str(negative_path.relative_to(project_root)),
        "output_path": str(output_path.relative_to(project_root)),
        "background_band_half_width": background_band_half_width,
        "distance_offset": distance_offset,
        "morph_radius_ratio": morph_radius_ratio,
        "final_mask_erosion_ratio": final_mask_erosion_ratio,
        "erosion_radius_px": stages.erosion_radius,
        "image_type": stages.descriptor.image_type,
        "background_sigma": round(stages.descriptor.sigma_l, 6),
        "field_fraction": round(float(stages.field_mask.mean()), 6),
        "final_fraction": round(float(stages.final_mask.mean()), 6),
        "original_width": stages.resize_info.original_width,
        "original_height": stages.resize_info.original_height,
        "resized_width": stages.resize_info.resized_width,
        "resized_height": stages.resize_info.resized_height,
        "scale_x": round(stages.resize_info.scale_x, 6),
        "scale_y": round(stages.resize_info.scale_y, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a masked and resized copy of the microscopy dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets_masked_resized_1024"))
    parser.add_argument("--background-band-half-width", type=int, default=DEFAULT_BACKGROUND_BAND_HALF_WIDTH)
    parser.add_argument("--distance-offset", type=float, default=DEFAULT_DISTANCE_OFFSET)
    parser.add_argument("--morph-radius-ratio", type=float, default=DEFAULT_MORPH_RADIUS_RATIO)
    parser.add_argument("--final-mask-erosion-ratio", type=float, default=DEFAULT_FINAL_MASK_EROSION_RATIO)
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    output_root = (project_root / args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []

    positive_unlabelled_dir = project_root / "datasets" / "Positive" / "Unlabelled"
    negative_clear_dir = project_root / "datasets" / "Negative" / "Clear"

    for unlabelled_path in sorted(positive_unlabelled_dir.iterdir()):
        if not unlabelled_path.is_file():
            continue
        manifest_rows.extend(
            process_positive_pair(
                project_root=project_root,
                unlabelled_path=unlabelled_path,
                output_root=output_root,
                background_band_half_width=args.background_band_half_width,
                distance_offset=args.distance_offset,
                morph_radius_ratio=args.morph_radius_ratio,
                final_mask_erosion_ratio=args.final_mask_erosion_ratio,
            )
        )

    for negative_path in sorted(negative_clear_dir.iterdir()):
        if not negative_path.is_file():
            continue
        manifest_rows.append(
            process_negative_image(
                project_root=project_root,
                negative_path=negative_path,
                output_root=output_root,
                background_band_half_width=args.background_band_half_width,
                distance_offset=args.distance_offset,
                morph_radius_ratio=args.morph_radius_ratio,
                final_mask_erosion_ratio=args.final_mask_erosion_ratio,
            )
        )

    write_csv(output_root / "preprocessing_manifest.csv", manifest_rows)
    (output_root / "preprocessing_config.json").write_text(
        json.dumps(
            {
                "normalize_shorter_dim": DEFAULT_NORMALIZE_SHORTER_DIM,
                "background_band_half_width": args.background_band_half_width,
                "distance_offset": args.distance_offset,
                "morph_radius_ratio": args.morph_radius_ratio,
                "final_mask_erosion_ratio": args.final_mask_erosion_ratio,
                "notes": [
                    "Mask is estimated from the resized raw image.",
                    "Positive/Labelled uses the mask derived from the matching Positive/Unlabelled image.",
                    "Outside the final mask is zero-filled after resizing.",
                ],
            },
            indent=2,
        )
    )
    print(f"Wrote masked dataset copy to {output_root}")


if __name__ == "__main__":
    main()
