from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - runtime guard in candidate extraction
    np = None

from eda import IMAGE_EXTENSIONS, image_to_rgb, iter_files, jpeg_dimensions, load_dataset_config


POOL_A = "pool_a_labeled_sickled"
POOL_B = "pool_b_labeled_non_sickled"
POOL_C = "pool_c_unlabeled_candidates"


@dataclass(frozen=True)
class SourceRecord:
    split: str
    source_key: str
    source_id: str
    label: str
    stain: str
    raw_path: Path
    annotation_path: Path | None


@dataclass(frozen=True)
class CandidateBox:
    x: int
    y: int
    width: int
    height: int
    component_area: int
    component_width: int
    component_height: int
    foreground_coverage: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_source_records(path: Path, project_root: Path) -> dict[str, SourceRecord]:
    records: dict[str, SourceRecord] = {}
    for row in read_csv(path):
        annotation_path = project_root / row["annotation_path"] if row.get("annotation_path") else None
        records[row["source_key"]] = SourceRecord(
            split=row["split"],
            source_key=row["source_key"],
            source_id=row["source_id"],
            label=row["label"],
            stain=row["stain"],
            raw_path=project_root / row["raw_path"],
            annotation_path=annotation_path,
        )
    return records


def clamp_square_box(
    x: float,
    y: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
    margin: float,
    min_size: int,
) -> tuple[int, int, int, int]:
    center_x = x + width / 2
    center_y = y + height / 2
    side = max(width, height) * margin
    side = max(side, min_size)
    side = min(side, image_width, image_height)

    crop_x = int(round(center_x - side / 2))
    crop_y = int(round(center_y - side / 2))
    crop_side = int(round(side))

    crop_x = max(0, min(crop_x, image_width - crop_side))
    crop_y = max(0, min(crop_y, image_height - crop_side))
    return crop_x, crop_y, crop_side, crop_side


def crop_image(
    magick_path: str,
    image_path: Path,
    crop_box: tuple[int, int, int, int],
    output_path: Path,
    output_size: int,
) -> None:
    x, y, width, height = crop_box
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        magick_path,
        str(image_path),
        "-auto-orient",
        "-crop",
        f"{width}x{height}+{x}+{y}",
        "+repage",
        "-resize",
        f"{output_size}x{output_size}!",
        str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def connected_components(mask: np.ndarray, min_area: int, max_area: int) -> list[tuple[int, int, int, int, int]]:
    working = mask.copy()
    height, width = working.shape
    components: list[tuple[int, int, int, int, int]] = []
    ys, xs = np.nonzero(working)

    for seed_y, seed_x in zip(ys.tolist(), xs.tolist()):
        if not working[seed_y, seed_x]:
            continue

        stack = [(seed_y, seed_x)]
        working[seed_y, seed_x] = False
        points_x: list[int] = []
        points_y: list[int] = []

        while stack:
            y, x = stack.pop()
            points_y.append(y)
            points_x.append(x)
            for next_y in range(max(0, y - 1), min(height, y + 2)):
                for next_x in range(max(0, x - 1), min(width, x + 2)):
                    if working[next_y, next_x]:
                        working[next_y, next_x] = False
                        stack.append((next_y, next_x))

        area = len(points_x)
        if min_area <= area <= max_area:
            min_x, max_x = min(points_x), max(points_x)
            min_y, max_y = min(points_y), max(points_y)
            components.append((min_x, min_y, max_x - min_x + 1, max_y - min_y + 1, area))
    return components


def extract_candidate_boxes(
    image_path: Path,
    original_width: int,
    original_height: int,
    magick_path: str,
    analysis_size: int,
    max_candidates: int,
) -> list[CandidateBox]:
    if np is None:
        raise RuntimeError("NumPy is required for candidate crop extraction. Run with `uv run crop_creation.py`.")

    rgb = image_to_rgb(image_path, analysis_size, magick_path)
    if rgb is None:
        return []

    pixels = rgb.astype(float)
    luma = 0.2126 * pixels[:, :, 0] + 0.7152 * pixels[:, :, 1] + 0.0722 * pixels[:, :, 2]
    max_channel = pixels.max(axis=2)
    min_channel = pixels.min(axis=2)
    saturation = np.divide(
        max_channel - min_channel,
        max_channel,
        out=np.zeros_like(max_channel),
        where=max_channel > 0,
    )

    microscope_field = (luma > 45) & (luma < 245)
    candidate_mask = microscope_field & (saturation > 0.055) & (max_channel > 75)

    # Remove tiny salt noise and huge fused background blobs by component filtering.
    components = connected_components(
        candidate_mask,
        min_area=max(18, analysis_size // 20),
        max_area=int(analysis_size * analysis_size * 0.025),
    )

    candidates: list[CandidateBox] = []
    scale_x = original_width / analysis_size
    scale_y = original_height / analysis_size
    for min_x, min_y, width, height, area in components:
        if min(width, height) < 4:
            continue
        if max(width, height) > analysis_size * 0.16:
            continue
        aspect_ratio = max(width / height, height / width)
        if aspect_ratio > 4.0:
            continue

        coverage = area / (width * height)
        if coverage < 0.10:
            continue

        crop_x = min_x * scale_x
        crop_y = min_y * scale_y
        crop_w = width * scale_x
        crop_h = height * scale_y
        square_box = clamp_square_box(
            crop_x,
            crop_y,
            crop_w,
            crop_h,
            original_width,
            original_height,
            margin=2.2,
            min_size=64,
        )
        candidates.append(
            CandidateBox(
                x=square_box[0],
                y=square_box[1],
                width=square_box[2],
                height=square_box[3],
                component_area=area,
                component_width=width,
                component_height=height,
                foreground_coverage=round(coverage, 4),
            )
        )

    candidates.sort(key=lambda box: box.component_area, reverse=True)
    return candidates[:max_candidates]


def make_crop_id(pool: str, source: SourceRecord, index: int) -> str:
    return f"{pool}__{source.source_key.replace(':', '_')}__{index:04d}"


def pool_path(output_dir: Path, pool: str, split: str, crop_id: str) -> Path:
    return output_dir / pool / split / f"{crop_id}.jpg"


def create_pool_a(
    annotation_boxes_path: Path,
    sources: dict[str, SourceRecord],
    project_root: Path,
    output_dir: Path,
    magick_path: str,
    output_size: int,
    margin: float,
    include_loose_boxes: bool,
    write_images: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    source_dims: dict[Path, tuple[int, int]] = {}
    labelled_dims: dict[Path, tuple[int, int]] = {}
    per_source_index: Counter[str] = Counter()

    print("Creating Pool A from positive annotation boxes", flush=True)
    for box_row in read_csv(annotation_boxes_path):
        source_key = f"positive:{Path(box_row['file_name']).stem}"
        source = sources.get(source_key)
        if source is None or source.annotation_path is None:
            continue
        if box_row.get("tightness_bucket") == "loose_or_empty" and not include_loose_boxes:
            continue

        raw_dims = source_dims.setdefault(source.raw_path, jpeg_dimensions(source.raw_path))
        labelled_dims_value = labelled_dims.setdefault(source.annotation_path, jpeg_dimensions(source.annotation_path))
        raw_width, raw_height = raw_dims
        labelled_width, labelled_height = labelled_dims_value
        if not raw_width or not raw_height or not labelled_width or not labelled_height:
            continue

        scale_x = raw_width / labelled_width
        scale_y = raw_height / labelled_height
        scaled_x = float(box_row["x"]) * scale_x
        scaled_y = float(box_row["y"]) * scale_y
        scaled_w = float(box_row["width"]) * scale_x
        scaled_h = float(box_row["height"]) * scale_y
        crop_box = clamp_square_box(scaled_x, scaled_y, scaled_w, scaled_h, raw_width, raw_height, margin, output_size)

        per_source_index[source.source_key] += 1
        crop_id = make_crop_id("A", source, per_source_index[source.source_key])
        output_path = pool_path(output_dir, POOL_A, source.split, crop_id)
        if write_images:
            crop_image(magick_path, source.raw_path, crop_box, output_path, output_size)

        rows.append(
            {
                "pool": "A",
                "label": "sickled",
                "split": source.split,
                "source_key": source.source_key,
                "source_id": source.source_id,
                "stain": source.stain,
                "source_path": str(source.raw_path.relative_to(project_root)),
                "annotation_path": str(source.annotation_path.relative_to(project_root)),
                "crop_path": str(output_path.relative_to(project_root)),
                "crop_x": crop_box[0],
                "crop_y": crop_box[1],
                "crop_width": crop_box[2],
                "crop_height": crop_box[3],
                "source_box_x": round(scaled_x),
                "source_box_y": round(scaled_y),
                "source_box_width": round(scaled_w),
                "source_box_height": round(scaled_h),
                "box_index": box_row["box_index"],
                "tightness_bucket": box_row.get("tightness_bucket", ""),
                "detection_method": box_row.get("detection_method", ""),
            }
        )
        if len(rows) % 250 == 0:
            print(f"  Pool A crops prepared: {len(rows)}", flush=True)
    return rows


def create_candidate_pool(
    pool_label: str,
    output_pool: str,
    sources: list[SourceRecord],
    project_root: Path,
    output_dir: Path,
    magick_path: str,
    output_size: int,
    analysis_size: int,
    max_candidates_per_source: int,
    write_images: bool,
    candidate_label: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    print(f"Creating Pool {pool_label} from {len(sources)} source images", flush=True)
    for source_number, source in enumerate(sources, start=1):
        original_width, original_height = jpeg_dimensions(source.raw_path)
        if not original_width or not original_height:
            continue
        boxes = extract_candidate_boxes(
            source.raw_path,
            original_width,
            original_height,
            magick_path,
            analysis_size,
            max_candidates=max_candidates_per_source,
        )
        for index, box in enumerate(boxes, start=1):
            crop_id = make_crop_id(pool_label, source, index)
            output_path = pool_path(output_dir, output_pool, source.split, crop_id)
            crop_box = (box.x, box.y, box.width, box.height)
            if write_images:
                crop_image(magick_path, source.raw_path, crop_box, output_path, output_size)
            rows.append(
                {
                    "pool": pool_label,
                    "label": candidate_label,
                    "split": source.split,
                    "source_key": source.source_key,
                    "source_id": source.source_id,
                    "source_label": source.label,
                    "stain": source.stain,
                    "source_path": str(source.raw_path.relative_to(project_root)),
                    "crop_path": str(output_path.relative_to(project_root)),
                    "crop_x": box.x,
                    "crop_y": box.y,
                    "crop_width": box.width,
                    "crop_height": box.height,
                    "component_area": box.component_area,
                    "component_width": box.component_width,
                    "component_height": box.component_height,
                    "foreground_coverage": box.foreground_coverage,
                    "extraction_method": "saturation_connected_components",
                }
            )
        if source_number % 50 == 0:
            print(f"  Pool {pool_label} sources processed: {source_number}/{len(sources)}", flush=True)
    return rows


def write_spotcheck_files(
    output_dir: Path,
    pool_b_rows: list[dict[str, object]],
    project_root: Path,
    sample_size: int,
    seed: int,
    magick_path: str | None,
) -> None:
    rng = random.Random(seed)
    rows = pool_b_rows.copy()
    rng.shuffle(rows)
    sample = rows[:sample_size]
    write_csv(output_dir / "pool_b_spotcheck_sample.csv", sample)
    if not sample or magick_path is None:
        return

    montage_path = output_dir / "pool_b_spotcheck_contact_sheet.jpg"
    crop_paths = [project_root / str(row["crop_path"]) for row in sample if (project_root / str(row["crop_path"])).exists()]
    if not crop_paths:
        return
    command = [
        magick_path,
        "montage",
        *[str(path) for path in crop_paths],
        "-tile",
        "6x",
        "-geometry",
        "96x96+4+4",
        str(montage_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def write_report(output_dir: Path, manifests: dict[str, list[dict[str, object]]], write_images: bool) -> None:
    lines = [
        "# Stage 3 Crop Creation",
        "",
        "## Pools",
    ]
    for name, rows in manifests.items():
        split_counts = Counter(str(row["split"]) for row in rows)
        lines.append(f"- {name}: {len(rows)} crops; splits={dict(split_counts)}")
    lines.extend(
        [
            "",
            "## Protocol",
            "- Pool A: labeled sickled crops from positive source-image bounding boxes.",
            "- Pool B: labeled non-sickled crops from negative source images via saturation-based connected components.",
            "- Pool C: unlabeled candidate crops from positive and negative source images via the same classical candidate extractor.",
            "- Every crop inherits `split` from `split_outputs/source_image_splits.csv`; crop-level random splitting is not used.",
            f"- Crop image files written: {write_images}",
            "",
            "## Manual Review",
            "- Review `pool_b_spotcheck_sample.csv` and `pool_b_spotcheck_contact_sheet.jpg` before treating Pool B as fully clean.",
        ]
    )
    (output_dir / "stage3_crop_creation.md").write_text("\n".join(lines) + "\n")


def run_crop_creation(args: argparse.Namespace) -> None:
    magick_path = shutil.which("magick")
    if args.write_images and magick_path is None:
        raise RuntimeError("ImageMagick `magick` is required to write crop image files.")

    config = load_dataset_config(args.config)
    project_root = config.root.parent
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    sources = load_source_records(args.source_splits, project_root)
    source_list = list(sources.values())
    negative_sources = [source for source in source_list if source.label == "negative"]
    ssl_sources = source_list

    output_dir.mkdir(parents=True, exist_ok=True)
    pool_a_rows = create_pool_a(
        args.annotation_boxes,
        sources,
        project_root,
        output_dir,
        magick_path,
        args.crop_size,
        args.pool_a_margin,
        args.include_loose_boxes,
        args.write_images,
    )
    pool_b_rows = create_candidate_pool(
        "B",
        POOL_B,
        negative_sources,
        project_root,
        output_dir,
        magick_path,
        args.crop_size,
        args.analysis_size,
        args.pool_b_max_per_source,
        args.write_images,
        candidate_label="non_sickled",
    )
    pool_c_rows = create_candidate_pool(
        "C",
        POOL_C,
        ssl_sources,
        project_root,
        output_dir,
        magick_path,
        args.crop_size,
        args.analysis_size,
        args.pool_c_max_per_source,
        args.write_images,
        candidate_label="unlabeled",
    )

    write_csv(output_dir / "pool_a_labeled_sickled_manifest.csv", pool_a_rows)
    write_csv(output_dir / "pool_b_labeled_non_sickled_manifest.csv", pool_b_rows)
    write_csv(output_dir / "pool_c_unlabeled_candidates_manifest.csv", pool_c_rows)
    write_spotcheck_files(output_dir, pool_b_rows, project_root, args.spotcheck_size, args.seed, magick_path)
    write_report(
        output_dir,
        {
            "Pool A labeled sickled": pool_a_rows,
            "Pool B labeled non-sickled": pool_b_rows,
            "Pool C unlabeled candidates": pool_c_rows,
        },
        args.write_images,
    )

    print(f"Pool A crops: {len(pool_a_rows)}")
    print(f"Pool B crops: {len(pool_b_rows)}")
    print(f"Pool C crops: {len(pool_c_rows)}")
    print(f"Wrote {output_dir / 'stage3_crop_creation.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Stage 3 crop pools from source-image-level splits.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config.yaml")
    parser.add_argument(
        "--source-splits",
        type=Path,
        default=Path("split_outputs/source_image_splits.csv"),
        help="Source-image split manifest from Stage 2",
    )
    parser.add_argument(
        "--annotation-boxes",
        type=Path,
        default=Path("eda_outputs/annotation_boxes.csv"),
        help="Annotation boxes from Stage 1",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("crop_outputs"), help="Directory for crop pools")
    parser.add_argument("--seed", type=int, default=42, help="Seed for spot-check sampling")
    parser.add_argument("--crop-size", type=int, default=128, help="Output crop size in pixels")
    parser.add_argument("--analysis-size", type=int, default=384, help="Square analysis size for connected components")
    parser.add_argument("--pool-a-margin", type=float, default=1.35, help="Margin around positive annotation boxes")
    parser.add_argument("--pool-b-max-per-source", type=int, default=12, help="Max negative candidate crops per source image")
    parser.add_argument("--pool-c-max-per-source", type=int, default=8, help="Max unlabeled candidate crops per source image")
    parser.add_argument("--spotcheck-size", type=int, default=36, help="Number of Pool B crops to sample for manual review")
    parser.add_argument("--include-loose-boxes", action="store_true", help="Include Pool A boxes marked loose_or_empty")
    parser.add_argument("--manifest-only", dest="write_images", action="store_false", help="Write manifests only, no crop images")
    parser.set_defaults(write_images=True)
    run_crop_creation(parser.parse_args())


if __name__ == "__main__":
    main()
