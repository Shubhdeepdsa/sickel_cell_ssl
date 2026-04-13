from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


DEFAULT_WINDOW_SIZES = (128, 160, 224)
DEFAULT_STRIDE_RATIO = 0.5
DEFAULT_POSITIVE_COVERAGE = 0.50
DEFAULT_IGNORE_COVERAGE = 0.10
DEFAULT_SSL_IMAGE_SIZE = 224


@dataclass(frozen=True)
class WindowRecord:
    image_role: str
    source_key: str
    raw_path: str
    window_id: str
    x: int
    y: int
    size: int
    valid_area_ratio: float
    touches_invalid_area: bool
    fully_valid: bool


@dataclass(frozen=True)
class AnnotationBox:
    box_index: str
    x: int
    y: int
    width: int
    height: int


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


def open_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def load_predictions_by_source(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in read_csv(path):
        grouped.setdefault(row["source_key"], []).append(row)
    return grouped


def default_image_selection(project_root: Path) -> tuple[Path, Path, str, str]:
    predictions = load_predictions_by_source(project_root / "full_image_sliding_window_outputs" / "image_level_predictions.csv")
    sampled_rows = read_csv(project_root / "full_image_sliding_window_outputs" / "sampled_source_images.csv")

    scored: list[tuple[int, str, str, str]] = []
    for row in sampled_rows:
        source_key = row["source_key"]
        source_predictions = predictions[source_key]
        positive_votes = sum(pred["predicted_label"] == "positive" for pred in source_predictions)
        scored.append((positive_votes, source_key, row["label"], row["raw_path"]))

    negative_candidates = sorted((item for item in scored if item[2] == "negative"), reverse=True)
    positive_candidates = sorted((item for item in scored if item[2] == "positive"))

    negative_path = project_root / negative_candidates[0][3]
    positive_path = project_root / positive_candidates[0][3]
    positive_annotation = project_root / "datasets" / "Positive" / "Labelled" / positive_path.name
    return positive_path, positive_annotation, positive_candidates[0][1], negative_candidates[0][1]


def largest_component(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    working = mask.copy()
    best_points: list[tuple[int, int]] = []
    ys, xs = np.nonzero(working)
    for seed_y, seed_x in zip(ys.tolist(), xs.tolist()):
        if not working[seed_y, seed_x]:
            continue
        stack = [(seed_y, seed_x)]
        working[seed_y, seed_x] = False
        points: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            points.append((y, x))
            for next_y in range(max(0, y - 1), min(height, y + 2)):
                for next_x in range(max(0, x - 1), min(width, x + 2)):
                    if working[next_y, next_x]:
                        working[next_y, next_x] = False
                        stack.append((next_y, next_x))
        if len(points) > len(best_points):
            best_points = points
    output = np.zeros_like(mask, dtype=bool)
    for y, x in best_points:
        output[y, x] = True
    return output


def estimate_circle_mask(image: Image.Image, analysis_size: int = 512) -> tuple[np.ndarray, dict[str, float]]:
    thumbnail = image.resize((analysis_size, analysis_size))
    rgb = np.asarray(thumbnail, dtype=np.float32)
    max_channel = rgb.max(axis=2)
    luma = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    initial = (max_channel > 14.0) & (luma > 14.0)
    component = largest_component(initial)
    ys, xs = np.nonzero(component)
    if len(xs) < 100:
        width, height = image.size
        full_mask = np.ones((height, width), dtype=bool)
        return full_mask, {"center_x": width / 2, "center_y": height / 2, "radius": min(width, height) / 2, "valid_fraction": 1.0}

    center_x_small = float(xs.mean())
    center_y_small = float(ys.mean())
    distances = np.sqrt((xs - center_x_small) ** 2 + (ys - center_y_small) ** 2)
    radius_small = float(np.quantile(distances, 0.995) * 1.02)

    width, height = image.size
    scale_x = width / analysis_size
    scale_y = height / analysis_size
    center_x = center_x_small * scale_x
    center_y = center_y_small * scale_y
    radius = radius_small * max(scale_x, scale_y)

    yy, xx = np.mgrid[0:height, 0:width]
    full_mask = ((xx - center_x) ** 2 + (yy - center_y) ** 2) <= radius**2
    return full_mask.astype(bool), {
        "center_x": float(center_x),
        "center_y": float(center_y),
        "radius": float(radius),
        "valid_fraction": float(full_mask.mean()),
    }


def save_mask_visuals(image: Image.Image, mask: np.ndarray, circle: dict[str, float], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    original_path = output_dir / "original.png"
    mask_path = output_dir / "mask.png"
    masked_path = output_dir / "masked.png"
    circle_path = output_dir / "circle_overlay.png"

    image.save(original_path)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_path)

    rgb = np.asarray(image).copy()
    rgb[~mask] = 0
    Image.fromarray(rgb, mode="RGB").save(masked_path)

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    cx = circle["center_x"]
    cy = circle["center_y"]
    radius = circle["radius"]
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline="lime", width=4)
    overlay.save(circle_path)


def clip_window_sizes(window_sizes: list[int], width: int, height: int) -> list[int]:
    shorter_side = min(width, height)
    clipped = [max(64, min(int(size), shorter_side)) for size in window_sizes]
    return sorted(set(clipped))


def positions(limit: int, size: int, stride: int) -> list[int]:
    if size >= limit:
        return [0]
    max_start = limit - size
    values = list(range(0, max_start + 1, stride))
    if values[-1] != max_start:
        values.append(max_start)
    return values


def build_windows(
    image_role: str,
    source_key: str,
    raw_path: str,
    width: int,
    height: int,
    mask: np.ndarray,
    window_sizes: list[int],
    stride_ratio: float,
) -> list[WindowRecord]:
    windows: list[WindowRecord] = []
    for size in clip_window_sizes(window_sizes, width, height):
        stride = max(1, int(round(size * stride_ratio)))
        for y in positions(height, size, stride):
            for x in positions(width, size, stride):
                region = mask[y : y + size, x : x + size]
                valid_ratio = float(region.mean())
                touches_invalid = valid_ratio < 1.0
                windows.append(
                    WindowRecord(
                        image_role=image_role,
                        source_key=source_key,
                        raw_path=raw_path,
                        window_id=f"{source_key.replace(':', '_')}__s{size}_x{x}_y{y}",
                        x=x,
                        y=y,
                        size=size,
                        valid_area_ratio=valid_ratio,
                        touches_invalid_area=touches_invalid,
                        fully_valid=not touches_invalid,
                    )
                )
    return windows


def render_window_overlay(image: Image.Image, windows: list[WindowRecord], output_path: Path, color: str, title: str) -> None:
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()
    for window in windows:
        rect = plt.Rectangle((window.x, window.y), window.size, window.size, fill=False, edgecolor=color, linewidth=0.5, alpha=0.4)
        ax.add_patch(rect)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def render_all_scale_overlays(image: Image.Image, windows: list[WindowRecord], output_dir: Path) -> None:
    by_size: dict[int, list[WindowRecord]] = {}
    for window in windows:
        by_size.setdefault(window.size, []).append(window)
    colors = {128: "cyan", 160: "orange", 224: "magenta"}
    for size, size_windows in sorted(by_size.items()):
        render_window_overlay(image, size_windows, output_dir / f"grid_scale_{size}.png", colors.get(size, "cyan"), f"Window Grid {size}px")
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()
    for size, size_windows in sorted(by_size.items()):
        for window in size_windows:
            rect = plt.Rectangle((window.x, window.y), window.size, window.size, fill=False, edgecolor=colors.get(size, "cyan"), linewidth=0.5, alpha=0.25)
            ax.add_patch(rect)
    plt.title("All Sliding-Window Scales")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "grid_all_scales.png", dpi=180, bbox_inches="tight")
    plt.close()


def crop_contact_sheet(
    image: Image.Image,
    mask: np.ndarray,
    windows: list[WindowRecord],
    output_path: Path,
    title: str,
    max_items: int = 9,
) -> None:
    selected = windows[:max_items]
    if not selected:
        return
    n_cols = 3
    n_rows = math.ceil(len(selected) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)
    for axis in axes.flat:
        axis.axis("off")
    for axis, window in zip(axes.flat, selected):
        crop = image.crop((window.x, window.y, window.x + window.size, window.y + window.size))
        crop_mask = mask[window.y : window.y + window.size, window.x : window.x + window.size]
        crop_rgb = np.asarray(crop).copy()
        crop_rgb[~crop_mask] = 0
        axis.imshow(crop_rgb)
        axis.set_title(f"{window.window_id}\nvalid={window.valid_area_ratio:.2f}", fontsize=8)
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_annotation_boxes(annotation_csv: Path, image_name: str) -> list[AnnotationBox]:
    boxes: list[AnnotationBox] = []
    for row in read_csv(annotation_csv):
        if row["file_name"] != image_name:
            continue
        if row.get("tightness_bucket") == "loose_or_empty":
            continue
        boxes.append(
            AnnotationBox(
                box_index=row["box_index"],
                x=int(row["x"]),
                y=int(row["y"]),
                width=int(row["width"]),
                height=int(row["height"]),
            )
        )
    return boxes


def scale_annotation_boxes(
    boxes: list[AnnotationBox],
    labelled_size: tuple[int, int],
    raw_size: tuple[int, int],
) -> list[AnnotationBox]:
    labelled_width, labelled_height = labelled_size
    raw_width, raw_height = raw_size
    scale_x = raw_width / labelled_width
    scale_y = raw_height / labelled_height
    scaled: list[AnnotationBox] = []
    for box in boxes:
        scaled.append(
            AnnotationBox(
                box_index=box.box_index,
                x=int(round(box.x * scale_x)),
                y=int(round(box.y * scale_y)),
                width=int(round(box.width * scale_x)),
                height=int(round(box.height * scale_y)),
            )
        )
    return scaled


def annotation_coverage(window: WindowRecord, box: AnnotationBox) -> float:
    x1 = max(window.x, box.x)
    y1 = max(window.y, box.y)
    x2 = min(window.x + window.size, box.x + box.width)
    y2 = min(window.y + window.size, box.y + box.height)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area = max(1, box.width * box.height)
    return inter / area


def annotation_center_fraction(window: WindowRecord, box: AnnotationBox) -> tuple[float, float]:
    center_x = box.x + box.width / 2
    center_y = box.y + box.height / 2
    frac_x = (center_x - window.x) / max(1, window.size)
    frac_y = (center_y - window.y) / max(1, window.size)
    return frac_x, frac_y


def label_positive_windows(
    windows: list[WindowRecord],
    boxes: list[AnnotationBox],
    positive_threshold: float,
    ignore_threshold: float,
) -> list[dict[str, object]]:
    boxes_by_id = {box.box_index: box for box in boxes}
    labeled: list[dict[str, object]] = []
    for window in windows:
        coverages = [(box.box_index, annotation_coverage(window, box)) for box in boxes]
        touched = [(box_id, score) for box_id, score in coverages if score > 0.0]
        if touched:
            best_box_id, best_coverage = max(touched, key=lambda item: item[1])
            best_box = boxes_by_id[best_box_id]
            center_x_frac, center_y_frac = annotation_center_fraction(window, best_box)
            border_margin = min(center_x_frac, center_y_frac, 1.0 - center_x_frac, 1.0 - center_y_frac)
        else:
            best_box_id, best_coverage = "", 0.0
            center_x_frac, center_y_frac = 0.0, 0.0
            border_margin = 0.0
        if best_coverage >= positive_threshold:
            label = "positive"
        elif best_coverage >= ignore_threshold:
            label = "ignore"
        else:
            label = "negative"
        labeled.append(
            {
                "image_role": window.image_role,
                "source_key": window.source_key,
                "raw_path": window.raw_path,
                "window_id": window.window_id,
                "x": window.x,
                "y": window.y,
                "size": window.size,
                "valid_area_ratio": round(window.valid_area_ratio, 6),
                "touches_invalid_area": window.touches_invalid_area,
                "fully_valid": window.fully_valid,
                "window_label": label,
                "max_annotation_coverage": round(best_coverage, 6),
                "touched_annotation_count": len(touched),
                "best_annotation_box_index": best_box_id,
                "best_annotation_center_x_frac": round(center_x_frac, 6),
                "best_annotation_center_y_frac": round(center_y_frac, 6),
                "best_annotation_border_margin": round(border_margin, 6),
            }
        )
    return labeled


def render_positive_label_overlay(
    image: Image.Image,
    boxes: list[AnnotationBox],
    labeled_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    colors = {"positive": "#00ff00", "ignore": "#ffd84d", "negative": "#b0b0b0"}
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()
    for row in labeled_rows:
        rect = plt.Rectangle(
            (int(row["x"]), int(row["y"])),
            int(row["size"]),
            int(row["size"]),
            fill=False,
            edgecolor=colors[str(row["window_label"])],
            linewidth=0.5,
            alpha=0.35,
        )
        ax.add_patch(rect)
    for box in boxes:
        rect = plt.Rectangle((box.x, box.y), box.width, box.height, fill=False, edgecolor="red", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(box.x, max(0, box.y - 3), f"ann {box.box_index}", color="red", fontsize=7)
    plt.title("Positive Image Window Labels + Annotation Boxes")
    plt.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def labeled_contact_sheet(
    image: Image.Image,
    mask: np.ndarray,
    boxes: list[AnnotationBox],
    labeled_rows: list[dict[str, object]],
    label: str,
    output_path: Path,
    max_items: int = 9,
) -> None:
    boxes_by_id = {box.box_index: box for box in boxes}
    selected = diverse_labeled_rows([row for row in labeled_rows if row["window_label"] == label], max_items=max_items)
    if not selected:
        return
    n_cols = 3
    n_rows = math.ceil(len(selected) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)
    for axis in axes.flat:
        axis.axis("off")
    for axis, row in zip(axes.flat, selected):
        x, y, size = int(row["x"]), int(row["y"]), int(row["size"])
        crop = image.crop((x, y, x + size, y + size))
        crop_mask = mask[y : y + size, x : x + size]
        crop_rgb = np.asarray(crop).copy()
        crop_rgb[~crop_mask] = 0
        axis.imshow(crop_rgb)
        best_box_id = str(row.get("best_annotation_box_index", ""))
        if best_box_id and best_box_id in boxes_by_id:
            box = boxes_by_id[best_box_id]
            rect = plt.Rectangle((box.x - x, box.y - y), box.width, box.height, fill=False, edgecolor="red", linewidth=1.5)
            axis.add_patch(rect)
            center_x = box.x - x + box.width / 2
            center_y = box.y - y + box.height / 2
            axis.plot(center_x, center_y, marker="o", markersize=3, color="red")
        axis.set_title(
            (
                f"{label}\ncover={float(row['max_annotation_coverage']):.2f}"
                f" center=({float(row['best_annotation_center_x_frac']):.2f},"
                f"{float(row['best_annotation_center_y_frac']):.2f})"
            ),
            fontsize=8,
        )
        axis.axis("off")
    fig.suptitle(f"{label.title()} Window Examples")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def diverse_labeled_rows(rows: list[dict[str, object]], max_items: int) -> list[dict[str, object]]:
    if not rows:
        return []
    def margin_bucket(row: dict[str, object]) -> str:
        margin = float(row.get("best_annotation_border_margin", 0.0))
        if margin < 0.15:
            return "edge"
        if margin < 0.30:
            return "mid"
        return "center"

    ordered = sorted(
        rows,
        key=lambda row: (
            margin_bucket(row),
            -float(row["max_annotation_coverage"]),
            str(row.get("best_annotation_box_index", "")),
            float(row["valid_area_ratio"]),
        ),
    )
    selected: list[dict[str, object]] = []
    used_pairs: set[tuple[str, str]] = set()
    for bucket in ("center", "mid", "edge"):
        bucket_rows = [row for row in ordered if margin_bucket(row) == bucket]
        for row in bucket_rows:
            pair = (bucket, str(row.get("best_annotation_box_index", "")))
            if pair in used_pairs:
                continue
            selected.append(row)
            used_pairs.add(pair)
            if len(selected) >= max_items:
                return selected
    for row in sorted(rows, key=lambda row: (-float(row["max_annotation_coverage"]), -float(row.get("best_annotation_border_margin", 0.0)))):
        x = int(row["x"])
        y = int(row["y"])
        size = int(row["size"])
        center = (x + size / 2, y + size / 2)
        if all(
            abs(center[0] - (int(existing["x"]) + int(existing["size"]) / 2)) > size * 0.5
            or abs(center[1] - (int(existing["y"]) + int(existing["size"]) / 2)) > size * 0.5
            for existing in selected
        ):
            selected.append(row)
        if len(selected) >= max_items:
            break
    return selected[:max_items]


def apply_spatial_ssl_transform(image: Image.Image, mask_image: Image.Image, rng: random.Random, output_size: int) -> tuple[Image.Image, Image.Image]:
    width, height = image.size
    crop_scale = rng.uniform(0.55, 1.0)
    aspect_ratio = rng.uniform(0.8, 1.25)
    crop_w = min(width, max(32, int(round(width * crop_scale * math.sqrt(aspect_ratio)))))
    crop_h = min(height, max(32, int(round(height * crop_scale / math.sqrt(aspect_ratio)))))
    left = 0 if width == crop_w else rng.randint(0, width - crop_w)
    top = 0 if height == crop_h else rng.randint(0, height - crop_h)
    image = image.crop((left, top, left + crop_w, top + crop_h))
    mask_image = mask_image.crop((left, top, left + crop_w, top + crop_h))

    if rng.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask_image = mask_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        mask_image = mask_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    angle = rng.uniform(-180, 180)
    image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
    mask_image = mask_image.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)

    image = image.resize((output_size, output_size), Image.Resampling.BILINEAR)
    mask_image = mask_image.resize((output_size, output_size), Image.Resampling.NEAREST)
    return image, mask_image


def apply_photometric_ssl_transform(image: Image.Image, rng: random.Random) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.65, 1.35))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.65, 1.35))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.75, 1.25))
    if rng.random() < 0.10:
        image = image.convert("L").convert("RGB")
    if rng.random() < 0.80:
        sigma = rng.uniform(0.1, 2.0)
        image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
    return image


def masked_crop_array(image: Image.Image, mask_image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image).copy()
    mask = np.asarray(mask_image) > 127
    rgb[~mask] = 0
    return rgb


def mild_compression(image: Image.Image) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=45, optimize=True)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB")


def mild_stain_color(image: Image.Image) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    array[..., 0] *= 1.08
    array[..., 1] *= 0.97
    array[..., 2] *= 1.06
    array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array, mode="RGB")
    image = ImageEnhance.Color(image).enhance(1.08)
    image = ImageEnhance.Contrast(image).enhance(1.03)
    return image


def choose_ssl_samples(
    positive_rows: list[dict[str, object]],
    negative_windows: list[WindowRecord],
) -> list[tuple[str, int, int, int, str]]:
    samples: list[tuple[str, int, int, int, str]] = []
    for label, count in (("positive", 2), ("ignore", 2), ("negative", 2)):
        label_rows = [row for row in positive_rows if row["window_label"] == label][:count]
        for row in label_rows:
            samples.append(("positive", int(row["x"]), int(row["y"]), int(row["size"]), label))
    border_windows = [window for window in negative_windows if window.touches_invalid_area][:2]
    central_windows = [window for window in negative_windows if not window.touches_invalid_area][:2]
    for window in border_windows + central_windows:
        samples.append(("negative", window.x, window.y, window.size, "border" if window.touches_invalid_area else "central"))
    return samples


def ssl_pair_sheet(
    image: Image.Image,
    mask: np.ndarray,
    samples: list[tuple[str, int, int, int, str]],
    output_path: Path,
    image_role: str,
) -> None:
    if not samples:
        return
    rows = len(samples)
    cols = 11
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.4 * rows))
    axes = np.atleast_2d(axes)
    column_titles = [
        "orig",
        "mask",
        "masked",
        "view A",
        "mask A",
        "view B",
        "mask B",
        "gray",
        "blur",
        "dark",
        "compression",
    ]
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=8)
    for row_index, (_, x, y, size, tag) in enumerate(samples):
        crop = image.crop((x, y, x + size, y + size))
        crop_mask_np = mask[y : y + size, x : x + size]
        mask_image = Image.fromarray((crop_mask_np.astype(np.uint8) * 255), mode="L")
        masked = masked_crop_array(crop, mask_image)

        rng_a = random.Random(f"{image_role}-{row_index}-a")
        rng_b = random.Random(f"{image_role}-{row_index}-b")
        view_a, mask_a = apply_spatial_ssl_transform(crop, mask_image, rng_a, DEFAULT_SSL_IMAGE_SIZE)
        view_b, mask_b = apply_spatial_ssl_transform(crop, mask_image, rng_b, DEFAULT_SSL_IMAGE_SIZE)
        view_a = apply_photometric_ssl_transform(view_a, rng_a)
        view_b = apply_photometric_ssl_transform(view_b, rng_b)
        gray = crop.resize((DEFAULT_SSL_IMAGE_SIZE, DEFAULT_SSL_IMAGE_SIZE)).convert("L").convert("RGB")
        blur = crop.resize((DEFAULT_SSL_IMAGE_SIZE, DEFAULT_SSL_IMAGE_SIZE)).filter(ImageFilter.GaussianBlur(radius=1.25))
        dark = ImageEnhance.Brightness(crop.resize((DEFAULT_SSL_IMAGE_SIZE, DEFAULT_SSL_IMAGE_SIZE))).enhance(0.75)
        compressed = mild_compression(crop.resize((DEFAULT_SSL_IMAGE_SIZE, DEFAULT_SSL_IMAGE_SIZE)))

        items = [
            np.asarray(crop.resize((DEFAULT_SSL_IMAGE_SIZE, DEFAULT_SSL_IMAGE_SIZE))),
            np.asarray(mask_image.resize((DEFAULT_SSL_IMAGE_SIZE, DEFAULT_SSL_IMAGE_SIZE), Image.Resampling.NEAREST), dtype=np.uint8),
            masked,
            np.asarray(view_a),
            np.asarray(mask_a, dtype=np.uint8),
            np.asarray(view_b),
            np.asarray(mask_b, dtype=np.uint8),
            np.asarray(gray),
            np.asarray(blur),
            np.asarray(dark),
            np.asarray(compressed),
        ]
        for col, item in enumerate(items):
            axis = axes[row_index, col]
            if item.ndim == 2:
                axis.imshow(item, cmap="gray", vmin=0, vmax=255)
            else:
                axis.imshow(item)
            if col == 0:
                axis.set_ylabel(f"{tag}\n({x},{y},{size})", fontsize=8)
            axis.axis("off")
    fig.suptitle(f"{image_role.title()} SSL Pair Prototype")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_report(
    output_path: Path,
    positive_source_key: str,
    negative_source_key: str,
    positive_circle: dict[str, float],
    negative_circle: dict[str, float],
    positive_windows: list[WindowRecord],
    negative_windows: list[WindowRecord],
    positive_labels: list[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    positive_counts = {label: sum(row["window_label"] == label for row in positive_labels) for label in ("positive", "ignore", "negative")}
    lines = [
        "# Phase A Mask-Aware Sliding-Window Prototype",
        "",
        "## Images",
        f"- positive source: `{positive_source_key}`",
        f"- negative source: `{negative_source_key}`",
        "",
        "## Mask Stats",
        f"- positive valid fraction: `{positive_circle['valid_fraction']:.4f}`",
        f"- negative valid fraction: `{negative_circle['valid_fraction']:.4f}`",
        f"- positive circle: center=(`{positive_circle['center_x']:.1f}`, `{positive_circle['center_y']:.1f}`), radius=`{positive_circle['radius']:.1f}`",
        f"- negative circle: center=(`{negative_circle['center_x']:.1f}`, `{negative_circle['center_y']:.1f}`), radius=`{negative_circle['radius']:.1f}`",
        "",
        "## Window Grid",
        f"- window sizes: `{', '.join(str(size) for size in args.window_sizes)}`",
        f"- stride ratio: `{args.stride_ratio}`",
        f"- positive windows: `{len(positive_windows)}`",
        f"- negative windows: `{len(negative_windows)}`",
        f"- positive border-touching windows: `{sum(window.touches_invalid_area for window in positive_windows)}`",
        f"- negative border-touching windows: `{sum(window.touches_invalid_area for window in negative_windows)}`",
        "",
        "## Positive Image Labels",
        f"- positive windows: `{positive_counts['positive']}`",
        f"- ignore windows: `{positive_counts['ignore']}`",
        f"- negative windows: `{positive_counts['negative']}`",
        f"- positive coverage threshold: `{args.positive_coverage}`",
        f"- ignore lower bound: `{args.ignore_coverage}`",
        "",
        "## Interpretation",
        "- Border-touching windows are retained so useful edge content is not thrown away.",
        "- The validity mask marks outside-field pixels instead of turning them into an implicit label signal.",
        "- Positive windows are defined by annotation coverage, not by annotation-centered crop placement.",
        "- SSL pair sheets should be inspected to confirm image/mask spatial alignment after augmentation.",
        "",
        "## Output Files",
        "- `window_metadata.csv`",
        "- `positive_window_labels.csv`",
        "- per-image mask and grid overlays under `positive/` and `negative/`",
        "- SSL visualization sheets under `ssl/`",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototype a mask-aware sliding-window dataset on one positive and one negative full image.")
    parser.add_argument("--positive-raw", type=Path, default=None)
    parser.add_argument("--positive-annotation", type=Path, default=None)
    parser.add_argument("--negative-raw", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("phase_a_masked_dataset_prototype"))
    parser.add_argument("--annotation-csv", type=Path, default=Path("eda_outputs/annotation_boxes.csv"))
    parser.add_argument("--window-sizes", nargs="+", type=int, default=list(DEFAULT_WINDOW_SIZES))
    parser.add_argument("--stride-ratio", type=float, default=DEFAULT_STRIDE_RATIO)
    parser.add_argument("--positive-coverage", type=float, default=DEFAULT_POSITIVE_COVERAGE)
    parser.add_argument("--ignore-coverage", type=float, default=DEFAULT_IGNORE_COVERAGE)
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.positive_raw is None or args.negative_raw is None:
        default_positive_raw, default_positive_annotation, default_positive_key, default_negative_key = default_image_selection(project_root)
        positive_raw = args.positive_raw.resolve() if args.positive_raw else default_positive_raw
        positive_annotation = args.positive_annotation.resolve() if args.positive_annotation else default_positive_annotation
        negative_raw = args.negative_raw.resolve() if args.negative_raw else (project_root / next(
            row["raw_path"]
            for row in read_csv(project_root / "full_image_sliding_window_outputs" / "sampled_source_images.csv")
            if row["source_key"] == default_negative_key
        ))
        positive_source_key = default_positive_key
        negative_source_key = default_negative_key
    else:
        positive_raw = args.positive_raw.resolve()
        positive_annotation = args.positive_annotation.resolve() if args.positive_annotation else project_root / "datasets" / "Positive" / "Labelled" / positive_raw.name
        negative_raw = args.negative_raw.resolve()
        positive_source_key = f"positive:{positive_raw.stem}"
        negative_source_key = f"negative:{negative_raw.stem}"

    positive_image = open_rgb_image(positive_raw)
    negative_image = open_rgb_image(negative_raw)

    positive_mask, positive_circle = estimate_circle_mask(positive_image)
    negative_mask, negative_circle = estimate_circle_mask(negative_image)

    save_mask_visuals(positive_image, positive_mask, positive_circle, output_dir / "positive")
    save_mask_visuals(negative_image, negative_mask, negative_circle, output_dir / "negative")

    positive_windows = build_windows(
        "positive",
        positive_source_key,
        str(positive_raw.relative_to(project_root)),
        positive_image.width,
        positive_image.height,
        positive_mask,
        list(args.window_sizes),
        args.stride_ratio,
    )
    negative_windows = build_windows(
        "negative",
        negative_source_key,
        str(negative_raw.relative_to(project_root)),
        negative_image.width,
        negative_image.height,
        negative_mask,
        list(args.window_sizes),
        args.stride_ratio,
    )

    render_all_scale_overlays(positive_image, positive_windows, output_dir / "positive")
    render_all_scale_overlays(negative_image, negative_windows, output_dir / "negative")

    positive_central = [window for window in positive_windows if not window.touches_invalid_area]
    positive_border = [window for window in positive_windows if window.touches_invalid_area]
    negative_central = [window for window in negative_windows if not window.touches_invalid_area]
    negative_border = [window for window in negative_windows if window.touches_invalid_area]

    crop_contact_sheet(positive_image, positive_mask, positive_central, output_dir / "positive" / "central_windows.png", "Positive Central Windows")
    crop_contact_sheet(positive_image, positive_mask, positive_border, output_dir / "positive" / "border_windows.png", "Positive Border Windows")
    crop_contact_sheet(negative_image, negative_mask, negative_central, output_dir / "negative" / "central_windows.png", "Negative Central Windows")
    crop_contact_sheet(negative_image, negative_mask, negative_border, output_dir / "negative" / "border_windows.png", "Negative Border Windows")

    boxes = load_annotation_boxes(args.annotation_csv.resolve(), positive_annotation.name)
    with Image.open(positive_annotation) as labelled_image:
        labelled_size = ImageOps.exif_transpose(labelled_image).size
    boxes = scale_annotation_boxes(boxes, labelled_size=labelled_size, raw_size=positive_image.size)
    positive_labels = label_positive_windows(
        positive_windows,
        boxes,
        positive_threshold=args.positive_coverage,
        ignore_threshold=args.ignore_coverage,
    )
    render_positive_label_overlay(positive_image, boxes, positive_labels, output_dir / "positive" / "label_overlay.png")
    labeled_contact_sheet(positive_image, positive_mask, boxes, positive_labels, "positive", output_dir / "positive" / "positive_examples.png")
    labeled_contact_sheet(positive_image, positive_mask, boxes, positive_labels, "ignore", output_dir / "positive" / "ignore_examples.png")
    labeled_contact_sheet(positive_image, positive_mask, boxes, positive_labels, "negative", output_dir / "positive" / "negative_examples.png")

    window_rows = [
        {
            "image_role": window.image_role,
            "source_key": window.source_key,
            "raw_path": window.raw_path,
            "window_id": window.window_id,
            "x": window.x,
            "y": window.y,
            "size": window.size,
            "valid_area_ratio": round(window.valid_area_ratio, 6),
            "touches_invalid_area": window.touches_invalid_area,
            "fully_valid": window.fully_valid,
        }
        for window in positive_windows + negative_windows
    ]
    write_csv(output_dir / "window_metadata.csv", window_rows)
    write_csv(output_dir / "positive_window_labels.csv", positive_labels)

    ssl_samples = choose_ssl_samples(positive_labels, negative_windows)
    positive_ssl_samples = [sample for sample in ssl_samples if sample[0] == "positive"]
    negative_ssl_samples = [sample for sample in ssl_samples if sample[0] == "negative"]
    ssl_pair_sheet(positive_image, positive_mask, positive_ssl_samples, output_dir / "ssl" / "ssl_pairs_positive.png", "positive")
    ssl_pair_sheet(negative_image, negative_mask, negative_ssl_samples, output_dir / "ssl" / "ssl_pairs_negative.png", "negative")

    run_manifest = {
        "positive_raw": str(positive_raw),
        "positive_annotation": str(positive_annotation),
        "negative_raw": str(negative_raw),
        "positive_source_key": positive_source_key,
        "negative_source_key": negative_source_key,
        "window_sizes": list(args.window_sizes),
        "stride_ratio": args.stride_ratio,
        "positive_coverage": args.positive_coverage,
        "ignore_coverage": args.ignore_coverage,
        "positive_circle": positive_circle,
        "negative_circle": negative_circle,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2))
    build_report(
        output_dir / "phase_a_report.md",
        positive_source_key=positive_source_key,
        negative_source_key=negative_source_key,
        positive_circle=positive_circle,
        negative_circle=negative_circle,
        positive_windows=positive_windows,
        negative_windows=negative_windows,
        positive_labels=positive_labels,
        args=args,
    )
    print(f"Wrote {output_dir / 'phase_a_report.md'}")


if __name__ == "__main__":
    main()
