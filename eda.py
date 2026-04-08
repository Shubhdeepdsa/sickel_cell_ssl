from __future__ import annotations

import argparse
import csv
import filecmp
import math
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - handled at runtime for partial audit mode
    np = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
ANNOTATION_EXTENSIONS = {".xml", ".json", ".txt", ".csv", ".yml", ".yaml"}
STAIN_TOKENS = {
    "giemsa",
    "leishman",
    "wright",
    "field",
    "eosin",
    "hematoxylin",
    "may",
    "grunwald",
    "romanowsky",
}


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    positive: Path
    negative: Path
    negative_clear: Path
    labelled: Path
    unlabelled: Path


def parse_simple_yaml_config(config_path: Path) -> dict[str, str]:
    """Parse the simple dataset mapping in config.yaml without a PyYAML dependency."""
    entries: dict[str, str] = {}
    in_dataset = False

    for raw_line in config_path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        if not raw_line.startswith(" ") and line.strip() == "dataset:":
            in_dataset = True
            continue

        if in_dataset and not raw_line.startswith(" "):
            break

        if not in_dataset:
            continue

        key, separator, value = line.strip().partition(":")
        if not separator:
            continue
        entries[key.strip()] = value.strip().strip("'\"")

    required = {"root", "positive", "negative", "negative_clear", "labelled", "unlabelled"}
    missing = sorted(required - entries.keys())
    if missing:
        raise ValueError(f"Missing dataset config keys: {', '.join(missing)}")
    return entries


def load_dataset_config(config_path: Path) -> DatasetConfig:
    raw = parse_simple_yaml_config(config_path)
    config_dir = config_path.parent.resolve()
    root = Path(raw["root"])
    if not root.is_absolute():
        root = config_dir / root
    root = root.resolve()

    def under_root(key: str) -> Path:
        value = Path(raw[key])
        return value.resolve() if value.is_absolute() else (root / value).resolve()

    return DatasetConfig(
        root=root,
        positive=under_root("positive"),
        negative=under_root("negative"),
        negative_clear=under_root("negative_clear"),
        labelled=under_root("labelled"),
        unlabelled=under_root("unlabelled"),
    )


def iter_files(path: Path, extensions: set[str]) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in extensions)


def jpeg_dimensions(path: Path) -> tuple[int | None, int | None]:
    # Reads JPEG SOF markers directly, avoiding Pillow for metadata-only work.
    try:
        with path.open("rb") as handle:
            if handle.read(2) != b"\xff\xd8":
                return None, None

            while True:
                marker_prefix = handle.read(1)
                if not marker_prefix:
                    return None, None
                if marker_prefix != b"\xff":
                    continue

                marker = handle.read(1)
                while marker == b"\xff":
                    marker = handle.read(1)
                if not marker:
                    return None, None

                marker_code = marker[0]
                if marker_code in {0xD8, 0xD9}:
                    continue

                length_bytes = handle.read(2)
                if len(length_bytes) != 2:
                    return None, None
                length = int.from_bytes(length_bytes, "big")
                if length < 2:
                    return None, None

                if marker_code in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                    data = handle.read(length - 2)
                    if len(data) < 5:
                        return None, None
                    height = int.from_bytes(data[1:3], "big")
                    width = int.from_bytes(data[3:5], "big")
                    return width, height

                handle.seek(length - 2, 1)
    except OSError:
        return None, None


def image_to_rgb(path: Path, size: int, magick_path: str | None) -> np.ndarray | None:
    if np is None or magick_path is None:
        return None

    command = [
        magick_path,
        str(path),
        "-auto-orient",
        "-resize",
        f"{size}x{size}!",
        "-colorspace",
        "sRGB",
        "-depth",
        "8",
        "RGB:-",
    ]
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError):
        return None

    expected = size * size * 3
    if len(result.stdout) != expected:
        return None
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape((size, size, 3))


def quality_bucket(metrics: dict[str, float | str]) -> str:
    if metrics.get("pixel_status") != "decoded":
        return "metadata_only"

    brightness = float(metrics["brightness_mean"])
    contrast = float(metrics["contrast_std"])
    sharpness = float(metrics["sharpness_proxy"])
    dark_fraction = float(metrics["dark_fraction"])
    bright_fraction = float(metrics["bright_fraction"])

    if dark_fraction > 0.42 or brightness < 55:
        return "underexposed_or_heavy_vignette"
    if bright_fraction > 0.18 or brightness > 230:
        return "overexposed"
    if contrast < 24:
        return "low_contrast"
    if sharpness < 3.5:
        return "soft_or_blurry"
    return "usable"


def infer_explicit_stain(path: Path) -> str:
    tokens = {part.lower().replace("-", "_") for part in path.parts}
    stem_tokens = set(path.stem.lower().replace("-", "_").split("_"))
    for token in sorted(STAIN_TOKENS):
        if token in tokens or token in stem_tokens:
            return token
    return "not_encoded"


def color_cast_bucket(rgb: np.ndarray) -> str:
    pixels = rgb.astype(float)
    max_channel = pixels.max(axis=2)
    min_channel = pixels.min(axis=2)
    saturation = np.divide(
        max_channel - min_channel,
        max_channel,
        out=np.zeros_like(max_channel),
        where=max_channel > 0,
    )
    luma = 0.2126 * pixels[:, :, 0] + 0.7152 * pixels[:, :, 1] + 0.0722 * pixels[:, :, 2]
    mask = (luma > 60) & (luma < 245) & (saturation > 0.035)
    if not mask.any():
        return "low_saturation_or_unclassified"

    red, green, blue = pixels[mask].mean(axis=0)
    if red > blue + 8 and red >= green - 5:
        return "pink_red_cast"
    if blue > red + 8 and blue >= green - 5:
        return "blue_purple_cast"
    if green > red + 8 and green > blue + 8:
        return "green_cyan_cast"
    return "balanced_mixed_cast"


def image_metrics(path: Path, magick_path: str | None) -> dict[str, float | str]:
    rgb = image_to_rgb(path, size=192, magick_path=magick_path)
    if rgb is None:
        return {
            "pixel_status": "not_decoded",
            "brightness_mean": math.nan,
            "contrast_std": math.nan,
            "sharpness_proxy": math.nan,
            "dark_fraction": math.nan,
            "bright_fraction": math.nan,
            "saturation_mean": math.nan,
            "color_cast": "not_available",
        }

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
    gradient_y = np.abs(np.diff(luma, axis=0)).mean()
    gradient_x = np.abs(np.diff(luma, axis=1)).mean()

    return {
        "pixel_status": "decoded",
        "brightness_mean": round(float(luma.mean()), 3),
        "contrast_std": round(float(luma.std()), 3),
        "sharpness_proxy": round(float((gradient_x + gradient_y) / 2), 3),
        "dark_fraction": round(float((luma < 45).mean()), 4),
        "bright_fraction": round(float((luma > 245).mean()), 4),
        "saturation_mean": round(float(saturation.mean()), 4),
        "color_cast": color_cast_bucket(rgb),
    }


def skipped_pixel_metrics(reason: str) -> dict[str, float | str]:
    return {
        "pixel_status": reason,
        "brightness_mean": math.nan,
        "contrast_std": math.nan,
        "sharpness_proxy": math.nan,
        "dark_fraction": math.nan,
        "bright_fraction": math.nan,
        "saturation_mean": math.nan,
        "color_cast": "not_evaluated",
    }


def remove_edge_connected(mask: np.ndarray) -> np.ndarray:
    cleaned = mask.copy()
    height, width = cleaned.shape
    stack: list[tuple[int, int]] = []

    for x in range(width):
        if cleaned[0, x]:
            stack.append((0, x))
        if cleaned[height - 1, x]:
            stack.append((height - 1, x))
    for y in range(height):
        if cleaned[y, 0]:
            stack.append((y, 0))
        if cleaned[y, width - 1]:
            stack.append((y, width - 1))

    while stack:
        y, x = stack.pop()
        if not cleaned[y, x]:
            continue
        cleaned[y, x] = False
        for next_y in range(max(0, y - 1), min(height, y + 2)):
            for next_x in range(max(0, x - 1), min(width, x + 2)):
                if cleaned[next_y, next_x]:
                    stack.append((next_y, next_x))
    return cleaned


def annotation_mask_from_pair(
    labelled_path: Path,
    reference_path: Path | None,
    analysis_size: int,
    magick_path: str | None,
) -> tuple[np.ndarray | None, str]:
    if np is None or reference_path is None or not reference_path.exists():
        return None, "single_image"

    try:
        if filecmp.cmp(labelled_path, reference_path, shallow=False):
            return None, "single_image_reference_identical"
    except OSError:
        return None, "single_image"

    labelled = image_to_rgb(labelled_path, size=analysis_size, magick_path=magick_path)
    reference = image_to_rgb(reference_path, size=analysis_size, magick_path=magick_path)
    if labelled is None or reference is None:
        return None, "single_image"

    labelled_pixels = labelled.astype(float)
    reference_pixels = reference.astype(float)
    labelled_luma = (
        0.2126 * labelled_pixels[:, :, 0]
        + 0.7152 * labelled_pixels[:, :, 1]
        + 0.0722 * labelled_pixels[:, :, 2]
    )
    reference_luma = (
        0.2126 * reference_pixels[:, :, 0]
        + 0.7152 * reference_pixels[:, :, 1]
        + 0.0722 * reference_pixels[:, :, 2]
    )
    labelled_channel_range = labelled_pixels.max(axis=2) - labelled_pixels.min(axis=2)

    mask = (
        (reference_luma - labelled_luma > 20)
        & (labelled_pixels.max(axis=2) < 135)
        & (labelled_channel_range < 75)
    )
    if int(mask.sum()) < 20:
        return None, "single_image_pair_diff_empty"
    return mask, "paired_difference"


def annotation_mask_from_single_image(
    path: Path,
    analysis_size: int,
    magick_path: str | None,
) -> np.ndarray | None:
    rgb = image_to_rgb(path, size=analysis_size, magick_path=magick_path)
    if rgb is None:
        return None

    pixels = rgb.astype(float)
    channel_range = pixels.max(axis=2) - pixels.min(axis=2)
    return (
        (pixels[:, :, 0] < 95)
        & (pixels[:, :, 1] < 95)
        & (pixels[:, :, 2] < 95)
        & (channel_range < 95)
    )


def component_box_score(component: np.ndarray) -> float:
    height, width = component.shape
    if height < 3 or width < 3:
        return 0.0

    thickness = max(1, min(height, width) // 8)
    top = component[:thickness, :].mean()
    bottom = component[-thickness:, :].mean()
    left = component[:, :thickness].mean()
    right = component[:, -thickness:].mean()
    if height > 2 * thickness and width > 2 * thickness:
        interior = component[thickness:-thickness, thickness:-thickness].mean()
    else:
        interior = component.mean()
    edge_score = min(top, bottom, left, right)
    return float(edge_score - interior)


def classify_box_tightness(width: int, height: int, area_ratio: float, foreground_coverage: float) -> str:
    if min(width, height) < 12:
        return "too_small"
    if area_ratio > 0.08:
        return "too_large"
    if foreground_coverage < 0.06:
        return "loose_or_empty"
    if foreground_coverage > 0.85:
        return "overfilled"
    return "likely_crop_candidate"


def detect_annotation_boxes(
    path: Path,
    original_width: int | None,
    original_height: int | None,
    magick_path: str | None,
    reference_path: Path | None = None,
) -> list[dict[str, object]]:
    if np is None or original_width is None or original_height is None:
        return []

    analysis_size = 384
    raw_mask, detection_method = annotation_mask_from_pair(path, reference_path, analysis_size, magick_path)
    if raw_mask is None:
        raw_mask = annotation_mask_from_single_image(path, analysis_size, magick_path)
    if raw_mask is None:
        return []
    rgb = image_to_rgb(path, size=analysis_size, magick_path=magick_path)
    if rgb is None:
        return []

    candidates = remove_edge_connected(raw_mask)
    boxes: list[dict[str, object]] = []

    ys, xs = np.nonzero(candidates)
    for seed_y, seed_x in zip(ys.tolist(), xs.tolist()):
        if not candidates[seed_y, seed_x]:
            continue

        stack = [(seed_y, seed_x)]
        points_y: list[int] = []
        points_x: list[int] = []
        candidates[seed_y, seed_x] = False

        while stack:
            y, x = stack.pop()
            points_y.append(y)
            points_x.append(x)
            for next_y in range(max(0, y - 1), min(analysis_size, y + 2)):
                for next_x in range(max(0, x - 1), min(analysis_size, x + 2)):
                    if candidates[next_y, next_x]:
                        candidates[next_y, next_x] = False
                        stack.append((next_y, next_x))

        if len(points_x) < 18:
            continue

        min_x, max_x = min(points_x), max(points_x)
        min_y, max_y = min(points_y), max(points_y)
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        if min(width, height) < 10 or max(width, height) > analysis_size * 0.35:
            continue

        fill_ratio = len(points_x) / (width * height)
        if not 0.025 <= fill_ratio <= 0.5:
            continue

        component = np.zeros((height, width), dtype=bool)
        component[np.array(points_y) - min_y, np.array(points_x) - min_x] = True
        rectangularity = component_box_score(component)
        if rectangularity < 0.02:
            continue

        inset = max(2, min(width, height) // 12)
        interior = rgb[min_y + inset : max_y - inset + 1, min_x + inset : max_x - inset + 1].astype(float)
        if interior.size == 0:
            foreground_coverage = 0.0
        else:
            interior_luma = 0.2126 * interior[:, :, 0] + 0.7152 * interior[:, :, 1] + 0.0722 * interior[:, :, 2]
            interior_max = interior.max(axis=2)
            interior_min = interior.min(axis=2)
            interior_saturation = np.divide(
                interior_max - interior_min,
                interior_max,
                out=np.zeros_like(interior_max),
                where=interior_max > 0,
            )
            foreground_coverage = float(((interior_saturation > 0.055) & (interior_luma > 50) & (interior_luma < 245)).mean())

        area_ratio = (width * height) / (analysis_size * analysis_size)
        boxes.append(
            {
                "x": round(min_x / analysis_size * original_width),
                "y": round(min_y / analysis_size * original_height),
                "width": round(width / analysis_size * original_width),
                "height": round(height / analysis_size * original_height),
                "analysis_x": min_x,
                "analysis_y": min_y,
                "analysis_width": width,
                "analysis_height": height,
                "fill_ratio": round(fill_ratio, 4),
                "rectangularity_score": round(rectangularity, 4),
                "area_ratio": round(area_ratio, 5),
                "foreground_coverage": round(foreground_coverage, 4),
                "tightness_bucket": classify_box_tightness(width, height, area_ratio, foreground_coverage),
                "detection_method": detection_method,
            }
        )

    return sorted(boxes, key=lambda box: (int(box["y"]), int(box["x"])))


def build_image_rows(config: DatasetConfig, magick_path: str | None) -> list[dict[str, object]]:
    groups = [
        ("positive_unlabelled", "positive", "raw_positive", config.unlabelled),
        ("positive_labelled", "positive", "raster_annotation_overlay", config.labelled),
        ("negative_clear", "negative", "clear", config.negative_clear),
    ]

    rows: list[dict[str, object]] = []
    for group, label, subset, directory in groups:
        files = iter_files(directory, IMAGE_EXTENSIONS)
        print(f"Scanning {group}: {len(files)} files")
        for path in files:
            width, height = jpeg_dimensions(path) if path.suffix.lower() in {".jpg", ".jpeg"} else (None, None)
            metrics = (
                skipped_pixel_metrics("annotation_overlay_not_evaluated")
                if group == "positive_labelled"
                else image_metrics(path, magick_path)
            )
            row = {
                "group": group,
                "label": label,
                "subset": subset,
                "relative_path": str(path.relative_to(config.root.parent)),
                "file_name": path.name,
                "stem": path.stem,
                "suffix": path.suffix.lower(),
                "file_size_bytes": path.stat().st_size,
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 5) if width and height else math.nan,
                "explicit_stain": infer_explicit_stain(path.relative_to(config.root)),
            }
            row.update(metrics)
            row["quality_bucket"] = (
                "annotation_overlay_not_quality_source" if group == "positive_labelled" else quality_bucket(row)
            )
            rows.append(row)
    return rows


def build_annotation_rows(config: DatasetConfig, magick_path: str | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    files = iter_files(config.labelled, IMAGE_EXTENSIONS)
    print(f"Detecting annotation boxes in {len(files)} labelled overlays")
    for image_index, path in enumerate(files, start=1):
        if image_index % 50 == 0:
            print(f"  processed {image_index}/{len(files)} labelled overlays")
        width, height = jpeg_dimensions(path)
        reference_path = config.unlabelled / path.name
        boxes = detect_annotation_boxes(path, width, height, magick_path, reference_path=reference_path)
        for index, box in enumerate(boxes, start=1):
            rows.append(
                {
                    "relative_path": str(path.relative_to(config.root.parent)),
                    "file_name": path.name,
                    "stem": path.stem,
                    "box_index": index,
                    **box,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_by(rows: list[dict[str, object]], key: str) -> Counter[str]:
    return Counter(str(row.get(key, "")) for row in rows)


def median(values: list[float]) -> float:
    clean = sorted(value for value in values if not math.isnan(value))
    if not clean:
        return math.nan
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2


def format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "- none"
    return "\n".join(f"- {key}: {value}" for key, value in sorted(counter.items()))


def write_markdown_report(
    path: Path,
    config: DatasetConfig,
    image_rows: list[dict[str, object]],
    annotation_rows: list[dict[str, object]],
    annotation_files: list[Path],
    magick_path: str | None,
) -> None:
    by_group = defaultdict(list)
    for row in image_rows:
        by_group[str(row["group"])].append(row)
    source_rows = [
        row for row in image_rows if row["group"] in {"positive_unlabelled", "negative_clear"}
    ]

    labelled_stems = {Path(str(row["relative_path"])).stem for row in by_group["positive_labelled"]}
    unlabelled_stems = {Path(str(row["relative_path"])).stem for row in by_group["positive_unlabelled"]}
    paired_positive = labelled_stems & unlabelled_stems
    box_rows_by_image = defaultdict(list)
    for row in annotation_rows:
        box_rows_by_image[str(row["file_name"])].append(row)

    width_values = [float(row["width"]) for row in image_rows if row["width"]]
    height_values = [float(row["height"]) for row in image_rows if row["height"]]
    tightness_counts = count_by(annotation_rows, "tightness_bucket")
    images_with_boxes = len(box_rows_by_image)
    labelled_count = len(by_group["positive_labelled"])
    likely_boxes = tightness_counts.get("likely_crop_candidate", 0)
    total_boxes = len(annotation_rows)
    likely_percent = (likely_boxes / total_boxes * 100) if total_boxes else 0

    if not annotation_files and labelled_count:
        annotation_storage = (
            "No separate annotation files were found. The annotations appear to be black rectangle overlays "
            "baked directly into `Positive/Labelled/*.jpg`, with matching raw images in `Positive/Unlabelled/*.jpg`."
        )
    elif annotation_files:
        annotation_storage = f"Found {len(annotation_files)} separate annotation-like files under the dataset root."
    else:
        annotation_storage = "No explicit annotation files or labelled image overlays were found."

    crop_readiness = (
        "Candidate boxes were detected, but crop extraction should first convert raster overlays into coordinate files "
        "and verify a sample manually."
    )
    if total_boxes and likely_percent >= 75:
        crop_readiness = (
            "Most detected boxes look usable as crop candidates, but they are still raster overlays rather than "
            "first-class annotation coordinates."
        )
    elif not total_boxes:
        crop_readiness = "No boxes were detected automatically; crop extraction is not ready from the current files."

    report = f"""# Stage 1 Data Audit

## Dataset paths
- root: `{config.root}`
- positive labelled overlays: `{config.labelled}`
- positive raw/unlabelled: `{config.unlabelled}`
- negative clear: `{config.negative_clear}`
- ImageMagick pixel analysis: `{"available at " + magick_path if magick_path else "not available; metadata-only audit"}`

## Image counts
- Positive raw images for modeling: {len(by_group["positive_unlabelled"])}
- Positive labelled overlay images: {labelled_count}
- Negative clear images: {len(by_group["negative_clear"])}
- Positive labelled/unlabelled filename pairs: {len(paired_positive)}
- Total image files scanned: {len(image_rows)}

## Image dimensions
- Median width: {median(width_values):.1f}
- Median height: {median(height_values):.1f}
- Width range: {min(width_values) if width_values else "n/a"} to {max(width_values) if width_values else "n/a"}
- Height range: {min(height_values) if height_values else "n/a"} to {max(height_values) if height_values else "n/a"}

## Explicit stain distribution
{format_counter(count_by(source_rows, "explicit_stain"))}

No stain labels are encoded in the current directory or filename structure unless the value above says otherwise. The color cast distribution below is only a pixel-color proxy, not a biological stain label.

## Heuristic color cast distribution
{format_counter(count_by(source_rows, "color_cast"))}

## Image quality buckets
{format_counter(count_by(source_rows, "quality_bucket"))}

## Annotation storage
{annotation_storage}

## Detected annotation boxes
- Labelled images with at least one detected box: {images_with_boxes} / {labelled_count}
- Total detected boxes: {total_boxes}
- Likely crop-candidate boxes: {likely_boxes} / {total_boxes} ({likely_percent:.1f}%)

Tightness buckets:
{format_counter(tightness_counts)}

Crop-readiness note: {crop_readiness}

## Generated files
- `eda_outputs/image_audit.csv`: one row per scanned image with dimensions, quality metrics, and stain/color proxies.
- `eda_outputs/annotation_boxes.csv`: one row per detected raster box in the labelled positive images.
- `eda_outputs/stage1_data_audit.md`: this report.
"""
    path.write_text(report)


def run_audit(config_path: Path, output_dir: Path) -> None:
    config = load_dataset_config(config_path)
    for key, value in config.__dict__.items():
        if not value.exists():
            raise FileNotFoundError(f"Configured dataset path for {key!r} does not exist: {value}")

    magick_path = shutil.which("magick")
    annotation_files = iter_files(config.root, ANNOTATION_EXTENSIONS)
    image_rows = build_image_rows(config, magick_path)
    annotation_rows = build_annotation_rows(config, magick_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "image_audit.csv", image_rows)
    write_csv(output_dir / "annotation_boxes.csv", annotation_rows)
    write_markdown_report(
        output_dir / "stage1_data_audit.md",
        config=config,
        image_rows=image_rows,
        annotation_rows=annotation_rows,
        annotation_files=annotation_files,
        magick_path=magick_path,
    )

    print(f"Scanned {len(image_rows)} images")
    print(f"Detected {len(annotation_rows)} annotation-box candidates")
    print(f"Wrote {output_dir / 'stage1_data_audit.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 data audit for the Uganda sickle-cell dataset.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("eda_outputs"), help="Directory for audit outputs")
    args = parser.parse_args()
    run_audit(args.config, args.output_dir)


if __name__ == "__main__":
    main()
