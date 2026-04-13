from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from scipy import ndimage, optimize, spatial


DEFAULT_WINDOW_SIZES = (128, 160, 224)
DEFAULT_STRIDE_RATIO = 0.5
DEFAULT_DISTANCE_OFFSET = 15.0
DEFAULT_MORPH_RADIUS_RATIO = 0.015
DEFAULT_SHRINK_RATIO = 0.12
DEFAULT_COVERAGE_THRESHOLD = 0.70
DEFAULT_NORMALIZE_SHORTER_DIM = 1024
DEFAULT_FINAL_MASK_EROSION_RATIO = 0.01
DEFAULT_BACKGROUND_BAND_HALF_WIDTH = 50


@dataclass(frozen=True)
class AnnotationBox:
    box_index: str
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class WindowRecord:
    image_role: str
    source_key: str
    raw_path: str
    validity_source: str
    window_id: str
    x: int
    y: int
    size: int
    coverage: float
    accepted: bool
    touches_border: bool


@dataclass(frozen=True)
class BackgroundDescriptor:
    mean_l: float
    mean_a: float
    mean_b: float
    sigma_l: float
    peak_bin: int
    image_type: str


@dataclass(frozen=True)
class EllipseParameters:
    center_x: float
    center_y: float
    semi_a: float
    semi_b: float
    angle_rad: float


@dataclass(frozen=True)
class ResizeInfo:
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    scale_x: float
    scale_y: float


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


def pil_lab_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("LAB"), dtype=np.float32)


def load_predictions_by_source(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in read_csv(path):
        grouped.setdefault(row["source_key"], []).append(row)
    return grouped


def default_image_selection(project_root: Path) -> tuple[Path, Path, Path, str, str]:
    predictions = load_predictions_by_source(project_root / "full_image_sliding_window_outputs" / "image_level_predictions.csv")
    sampled_rows = read_csv(project_root / "full_image_sliding_window_outputs" / "sampled_source_images.csv")
    scored: list[tuple[int, str, str, str]] = []
    for row in sampled_rows:
        source_predictions = predictions[row["source_key"]]
        positive_votes = sum(pred["predicted_label"] == "positive" for pred in source_predictions)
        scored.append((positive_votes, row["source_key"], row["label"], row["raw_path"]))

    negative_candidates = sorted((item for item in scored if item[2] == "negative"), reverse=True)
    positive_candidates = sorted((item for item in scored if item[2] == "positive"))
    positive_raw = project_root / positive_candidates[0][3]
    negative_raw = project_root / negative_candidates[0][3]
    positive_annotation = project_root / "datasets" / "Positive" / "Labelled" / positive_raw.name
    return positive_raw, positive_annotation, negative_raw, positive_candidates[0][1], negative_candidates[0][1]


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


def scale_annotation_boxes(boxes: list[AnnotationBox], labelled_size: tuple[int, int], raw_size: tuple[int, int]) -> list[AnnotationBox]:
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


def resize_image_and_boxes(
    image: Image.Image,
    boxes: list[AnnotationBox] | None,
    normalize_shorter_dim: int,
) -> tuple[Image.Image, list[AnnotationBox] | None, ResizeInfo]:
    original_width, original_height = image.size
    shorter_dim = min(image.size)
    if shorter_dim == normalize_shorter_dim:
        info = ResizeInfo(
            original_width=original_width,
            original_height=original_height,
            resized_width=original_width,
            resized_height=original_height,
            scale_x=1.0,
            scale_y=1.0,
        )
        return image, boxes, info
    scale = normalize_shorter_dim / shorter_dim
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))
    resized_image = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    resized_boxes: list[AnnotationBox] | None = None
    if boxes is not None:
        resized_boxes = []
        for box in boxes:
            resized_boxes.append(
                AnnotationBox(
                    box_index=box.box_index,
                    x=int(round(box.x * scale)),
                    y=int(round(box.y * scale)),
                    width=int(round(box.width * scale)),
                    height=int(round(box.height * scale)),
                )
            )
    info = ResizeInfo(
        original_width=original_width,
        original_height=original_height,
        resized_width=resized_width,
        resized_height=resized_height,
        scale_x=scale,
        scale_y=scale,
    )
    return resized_image, resized_boxes, info


def disk_structure(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, count = ndimage.label(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, count + 1))
    best_label = int(np.argmax(sizes) + 1)
    return labeled == best_label


def convex_hull_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return mask.copy()
    points = np.stack([xs, ys], axis=1)
    try:
        hull = spatial.ConvexHull(points)
    except Exception:
        return mask.copy()
    polygon = [tuple(points[index]) for index in hull.vertices]
    width = mask.shape[1]
    height = mask.shape[0]
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    draw.polygon(polygon, fill=255)
    return np.asarray(canvas, dtype=np.uint8) > 0


def erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    structure = disk_structure(radius)
    eroded = ndimage.binary_erosion(mask, structure=structure)
    if eroded.sum() == 0:
        return mask.copy()
    return eroded


def estimate_background_descriptor(
    lab: np.ndarray,
    band_half_width: int = 10,
    frame_ratio: float = 0.06,
) -> tuple[BackgroundDescriptor, np.ndarray, np.ndarray]:
    l_channel = lab[:, :, 0]
    height, width = l_channel.shape
    frame_width = max(1, int(round(min(height, width) * frame_ratio)))
    frame_mask = np.zeros_like(l_channel, dtype=bool)
    frame_mask[:frame_width, :] = True
    frame_mask[-frame_width:, :] = True
    frame_mask[:, :frame_width] = True
    frame_mask[:, -frame_width:] = True

    frame_l = l_channel[frame_mask]
    histogram = np.bincount(frame_l.astype(np.uint8).ravel(), minlength=256)
    peak_bin = int(np.argmax(histogram[:81]))
    lower = max(0, peak_bin - int(band_half_width))
    upper = min(255, peak_bin + int(band_half_width))
    background_band = frame_mask & (l_channel >= lower) & (l_channel <= upper)
    if background_band.sum() < 64:
        background_band = frame_mask & (l_channel <= max(10, peak_bin + 5))
    if background_band.sum() < 64:
        background_band = l_channel <= max(10, peak_bin + 5)
    l_values = l_channel[background_band]
    a_values = lab[:, :, 1][background_band]
    b_values = lab[:, :, 2][background_band]
    sigma_l = float(np.std(l_values))
    descriptor = BackgroundDescriptor(
        mean_l=float(np.mean(l_values)),
        mean_a=float(np.mean(a_values)),
        mean_b=float(np.mean(b_values)),
        sigma_l=sigma_l,
        peak_bin=peak_bin,
        image_type="gray_halo" if sigma_l > 12.0 else "black_background",
    )
    return descriptor, histogram, background_band


def field_mask_from_background(
    image: Image.Image,
    descriptor: BackgroundDescriptor,
    morph_radius_ratio: float,
    distance_offset: float,
    use_convex_hull: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    shorter_dim = min(image.size)
    bilateral = image.filter(ImageFilter.GaussianBlur(radius=max(1.0, shorter_dim * 0.003)))
    lab = pil_lab_array(bilateral)
    distance_map = np.abs(lab[:, :, 0] - descriptor.mean_l)
    threshold = descriptor.sigma_l * 2.0 + distance_offset
    initial_foreground = distance_map > threshold
    radius = max(1, int(round(shorter_dim * morph_radius_ratio)))
    structure = disk_structure(radius)
    closed = ndimage.binary_closing(initial_foreground, structure=structure)
    opened = ndimage.binary_opening(closed, structure=structure)
    component = largest_component(opened)
    field_mask = convex_hull_mask(component) if use_convex_hull else component
    return distance_map, initial_foreground, opened, field_mask, float(threshold)


def ellipse_residuals(params: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    center_x, center_y, log_a, log_b, angle = params
    semi_a = math.exp(log_a)
    semi_b = math.exp(log_b)
    cos_t = math.cos(angle)
    sin_t = math.sin(angle)
    dx = xs - center_x
    dy = ys - center_y
    x_rot = dx * cos_t + dy * sin_t
    y_rot = -dx * sin_t + dy * cos_t
    return (x_rot / semi_a) ** 2 + (y_rot / semi_b) ** 2 - 1.0


def fit_ellipse_to_mask(mask: np.ndarray) -> EllipseParameters:
    boundary = mask & ~ndimage.binary_erosion(mask)
    ys, xs = np.nonzero(boundary)
    if len(xs) < 20:
        ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        height, width = mask.shape
        return EllipseParameters(width / 2, height / 2, width / 2, height / 2, 0.0)

    center_x0 = float(xs.mean())
    center_y0 = float(ys.mean())
    coords = np.stack([xs - center_x0, ys - center_y0], axis=1)
    covariance = np.cov(coords.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]
    theta0 = float(math.atan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    cos_t = math.cos(theta0)
    sin_t = math.sin(theta0)
    x_rot = (xs - center_x0) * cos_t + (ys - center_y0) * sin_t
    y_rot = -(xs - center_x0) * sin_t + (ys - center_y0) * cos_t
    semi_a0 = max(8.0, float(np.quantile(np.abs(x_rot), 0.98)))
    semi_b0 = max(8.0, float(np.quantile(np.abs(y_rot), 0.98)))
    if semi_b0 > semi_a0:
        semi_a0, semi_b0 = semi_b0, semi_a0
        theta0 += math.pi / 2.0

    initial = np.array([center_x0, center_y0, math.log(semi_a0), math.log(semi_b0), theta0], dtype=np.float64)
    result = optimize.least_squares(
        ellipse_residuals,
        initial,
        args=(xs.astype(np.float64), ys.astype(np.float64)),
        max_nfev=300,
    )
    center_x, center_y, log_a, log_b, angle = result.x
    semi_a = math.exp(log_a)
    semi_b = math.exp(log_b)
    if semi_b > semi_a:
        semi_a, semi_b = semi_b, semi_a
        angle += math.pi / 2.0
    return EllipseParameters(float(center_x), float(center_y), float(semi_a), float(semi_b), float(angle))


def shrink_ellipse(ellipse: EllipseParameters, shrink_ratio: float) -> EllipseParameters:
    return EllipseParameters(
        center_x=ellipse.center_x,
        center_y=ellipse.center_y,
        semi_a=max(4.0, ellipse.semi_a * (1.0 - shrink_ratio)),
        semi_b=max(4.0, ellipse.semi_b * (1.0 - shrink_ratio)),
        angle_rad=ellipse.angle_rad,
    )


def ellipse_mask(shape: tuple[int, int], ellipse: EllipseParameters) -> np.ndarray:
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    dx = xx - ellipse.center_x
    dy = yy - ellipse.center_y
    cos_t = math.cos(ellipse.angle_rad)
    sin_t = math.sin(ellipse.angle_rad)
    x_rot = dx * cos_t + dy * sin_t
    y_rot = -dx * sin_t + dy * cos_t
    return (x_rot / ellipse.semi_a) ** 2 + (y_rot / ellipse.semi_b) ** 2 <= 1.0


def zero_outside_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgb = np.asarray(image).copy()
    rgb[~mask] = 0
    return Image.fromarray(rgb, mode="RGB")


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
    mask: np.ndarray,
    validity_source: str,
    window_sizes: list[int],
    stride_ratio: float,
    coverage_threshold: float,
) -> list[WindowRecord]:
    height, width = mask.shape
    windows: list[WindowRecord] = []
    for size in clip_window_sizes(window_sizes, width, height):
        stride = max(1, int(round(size * stride_ratio)))
        for y in positions(height, size, stride):
            for x in positions(width, size, stride):
                region = mask[y : y + size, x : x + size]
                coverage = float(region.mean())
                windows.append(
                    WindowRecord(
                        image_role=image_role,
                        source_key=source_key,
                        raw_path=raw_path,
                        validity_source=validity_source,
                        window_id=f"{source_key.replace(':', '_')}__s{size}_x{x}_y{y}",
                        x=x,
                        y=y,
                        size=size,
                        coverage=coverage,
                        accepted=coverage >= coverage_threshold,
                        touches_border=coverage < 1.0,
                    )
                )
    return windows


def render_histogram(output_path: Path, histogram: np.ndarray, descriptor: BackgroundDescriptor) -> None:
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(256), histogram, color="#7f7f7f", width=1.0)
    plt.axvline(descriptor.peak_bin, color="red", linestyle="--", linewidth=1.5, label="peak")
    plt.axvspan(max(0, descriptor.peak_bin - 10), min(255, descriptor.peak_bin + 10), color="orange", alpha=0.25, label="background band")
    plt.xlim(0, 120)
    plt.xlabel("L* bin")
    plt.ylabel("count")
    plt.title(f"Stage 1 Histogram ({descriptor.image_type})")
    plt.legend(loc="upper right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def render_binary_mask(mask: np.ndarray, output_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap="gray", vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def render_distance_map(distance_map: np.ndarray, threshold: float, output_path: Path) -> None:
    plt.figure(figsize=(8, 8))
    plt.imshow(distance_map, cmap="magma")
    plt.colorbar(fraction=0.046, pad=0.04, label="LAB distance")
    plt.title(f"Stage 2 Distance Map (threshold={threshold:.2f})")
    plt.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def render_background_band(image: Image.Image, background_band: np.ndarray, output_path: Path) -> None:
    rgb = np.asarray(image).copy()
    overlay = rgb.copy()
    overlay[background_band] = np.array([255, 128, 0], dtype=np.uint8)
    blended = (0.65 * rgb + 0.35 * overlay).astype(np.uint8)
    Image.fromarray(blended, mode="RGB").save(output_path)


def render_side_by_side(images: list[tuple[str, Image.Image]], output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, len(images), figsize=(6 * len(images), 6))
    axes = np.atleast_1d(axes)
    for axis, (label, image) in zip(axes, images):
        axis.imshow(image)
        axis.set_title(label)
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def ellipse_points(ellipse: EllipseParameters, num_points: int = 360) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, num_points)
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    cos_a = math.cos(ellipse.angle_rad)
    sin_a = math.sin(ellipse.angle_rad)
    xs = ellipse.center_x + ellipse.semi_a * cos_t * cos_a - ellipse.semi_b * sin_t * sin_a
    ys = ellipse.center_y + ellipse.semi_a * cos_t * sin_a + ellipse.semi_b * sin_t * cos_a
    return np.stack([xs, ys], axis=1)


def render_ellipse_overlay(
    image: Image.Image,
    field_mask: np.ndarray,
    ellipse: EllipseParameters,
    shrunk: EllipseParameters,
    output_path: Path,
    annotation_boxes: list[AnnotationBox] | None = None,
) -> None:
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()
    boundary = field_mask & ~ndimage.binary_erosion(field_mask)
    ys, xs = np.nonzero(boundary)
    ax.scatter(xs, ys, s=0.5, c="yellow", alpha=0.4, label="mask boundary")
    pts = ellipse_points(ellipse)
    ax.plot(pts[:, 0], pts[:, 1], color="cyan", linewidth=2, label="fitted ellipse")
    pts_shrunk = ellipse_points(shrunk)
    ax.plot(pts_shrunk[:, 0], pts_shrunk[:, 1], color="lime", linewidth=2, label="shrunk ellipse")
    if annotation_boxes:
        for box in annotation_boxes:
            rect = plt.Rectangle((box.x, box.y), box.width, box.height, fill=False, edgecolor="red", linewidth=1.2)
            ax.add_patch(rect)
    ax.legend(loc="lower right")
    plt.title("Stage 3 Ellipse Fit")
    plt.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def render_final_mask_overlay(
    image: Image.Image,
    field_mask: np.ndarray,
    final_mask: np.ndarray,
    ellipse: EllipseParameters,
    output_path: Path,
    annotation_boxes: list[AnnotationBox] | None = None,
) -> None:
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()
    field_boundary = field_mask & ~ndimage.binary_erosion(field_mask)
    final_boundary = final_mask & ~ndimage.binary_erosion(final_mask)
    ys, xs = np.nonzero(field_boundary)
    ax.scatter(xs, ys, s=0.4, c="yellow", alpha=0.35, label="cleaned field mask")
    ys_final, xs_final = np.nonzero(final_boundary)
    ax.scatter(xs_final, ys_final, s=0.4, c="lime", alpha=0.35, label="final eroded mask")
    pts = ellipse_points(ellipse)
    ax.plot(pts[:, 0], pts[:, 1], color="cyan", linewidth=1.5, label="ellipse reference")
    if annotation_boxes:
        for box in annotation_boxes:
            rect = plt.Rectangle((box.x, box.y), box.width, box.height, fill=False, edgecolor="red", linewidth=1.0)
            ax.add_patch(rect)
    ax.legend(loc="lower right")
    plt.title("Stage 3 Final Mask Overlay")
    plt.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def render_window_overlay(image: Image.Image, windows: list[WindowRecord], output_path: Path, title: str) -> None:
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()
    for window in windows:
        color = "lime" if window.accepted else "red"
        rect = plt.Rectangle((window.x, window.y), window.size, window.size, fill=False, edgecolor=color, linewidth=0.5, alpha=0.18)
        ax.add_patch(rect)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def render_coverage_histogram(windows: list[WindowRecord], coverage_threshold: float, output_path: Path) -> None:
    values = [window.coverage for window in windows]
    plt.figure(figsize=(8, 4))
    plt.hist(values, bins=30, color="#5f7fff", edgecolor="white")
    plt.axvline(coverage_threshold, color="red", linestyle="--", linewidth=1.5, label="accept threshold")
    plt.xlabel("crop coverage")
    plt.ylabel("count")
    plt.title("Stage 5/6 Coverage Distribution")
    plt.legend(loc="upper left")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def crop_contact_sheet(
    image: Image.Image,
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
        axis.imshow(crop)
        axis.set_title(f"cov={window.coverage:.2f}\nsize={window.size}", fontsize=8)
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def coverage_window_rows(windows: list[WindowRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for window in windows:
        rows.append(
            {
                "image_role": window.image_role,
                "source_key": window.source_key,
                "raw_path": window.raw_path,
                "validity_source": window.validity_source,
                "window_id": window.window_id,
                "x": window.x,
                "y": window.y,
                "size": window.size,
                "coverage": round(window.coverage, 6),
                "accepted": window.accepted,
                "touches_border": window.touches_border,
            }
        )
    return rows


def annotation_ellipse_coverage(box: AnnotationBox, mask: np.ndarray) -> float:
    x1 = max(0, box.x)
    y1 = max(0, box.y)
    x2 = min(mask.shape[1], box.x + box.width)
    y2 = min(mask.shape[0], box.y + box.height)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = mask[y1:y2, x1:x2]
    return float(region.mean())


def build_report(
    output_path: Path,
    positive_source_key: str,
    negative_source_key: str,
    args: argparse.Namespace,
    summaries: list[dict[str, object]],
) -> None:
    lines = [
        "# Border Preprocessing Prototype",
        "",
        "This prototype implements the simplified preprocessing pipeline: background estimation, field mask, conservative ellipse fit, zero-fill, crop coverage computation, and crop filtering.",
        "",
        f"- shrink ratio: `{args.shrink_ratio:.3f}`",
        f"- coverage threshold: `{args.coverage_threshold:.2f}`",
        f"- distance offset: `{args.distance_offset:.1f}`",
        f"- background band half-width: `{args.background_band_half_width}`",
        f"- morph radius ratio: `{args.morph_radius_ratio:.4f}`",
        f"- window sizes: `{', '.join(str(size) for size in args.window_sizes)}`",
        f"- stride ratio: `{args.stride_ratio:.2f}`",
        f"- normalize shorter dim: `{args.normalize_shorter_dim}`",
        f"- final mask erosion ratio: `{args.final_mask_erosion_ratio:.4f}`",
        f"- use convex hull: `{not args.no_convex_hull}`",
        "",
        f"- positive source: `{positive_source_key}`",
        f"- negative source: `{negative_source_key}`",
        "",
        "## Per-image summary",
        "",
    ]
    for summary in summaries:
        lines.extend(
            [
                f"### {summary['image_role'].title()}",
                f"- image type: `{summary['image_type']}`",
                f"- resized image: `{summary['resized_width']} x {summary['resized_height']}` (from `{summary['original_width']} x {summary['original_height']}`)",
                f"- background sigma: `{summary['sigma_l']:.2f}`",
                f"- field-mask area fraction: `{summary['field_fraction']:.4f}`",
                f"- valid ellipse area fraction: `{summary['ellipse_fraction']:.4f}`",
                f"- final eroded-mask area fraction: `{summary['final_mask_fraction']:.4f}`",
                f"- ellipse coverage windows: total=`{summary['ellipse_total_windows']}`, accepted=`{summary['ellipse_accepted_windows']}`, rejected=`{summary['ellipse_rejected_windows']}`, border-accepted=`{summary['ellipse_border_accepted']}`",
                f"- field-mask coverage windows: total=`{summary['mask_total_windows']}`, accepted=`{summary['mask_accepted_windows']}`, rejected=`{summary['mask_rejected_windows']}`, border-accepted=`{summary['mask_border_accepted']}`",
                f"- final eroded-mask windows: total=`{summary['final_total_windows']}`, accepted=`{summary['final_accepted_windows']}`, rejected=`{summary['final_rejected_windows']}`, border-accepted=`{summary['final_border_accepted']}`",
                "",
            ]
        )
        if "annotation_min_coverage" in summary:
            lines.extend(
                [
                    f"- annotation coverage in shrunk ellipse: min=`{summary['annotation_min_coverage']:.3f}`, mean=`{summary['annotation_mean_coverage']:.3f}`",
                    f"- annotation coverage in cleaned field mask: min=`{summary['annotation_mask_min_coverage']:.3f}`, mean=`{summary['annotation_mask_mean_coverage']:.3f}`",
                    f"- annotation coverage in final eroded mask: min=`{summary['annotation_final_min_coverage']:.3f}`, mean=`{summary['annotation_final_mean_coverage']:.3f}`",
                    "",
                ]
            )
    lines.extend(
        [
            "## What to inspect",
            "",
            "- `step1_histogram.png`: whether background peak selection is sensible",
            "- `step2_distance_map.png` and `step2_field_mask.png`: whether the field mask covers the microscope field and excludes outer background",
            "- `step3_final_mask_overlay.png`: whether the final eroded mask follows the cleaned mask boundary instead of the ellipse",
            "- `step4_zero_filled.png`: whether the main exported cropped image now follows the final eroded field mask",
            "- `step5_window_overlay_ellipse.png` vs `step5_window_overlay_field_mask.png`: compare the two validity-region strategies",
            "- `step6_*_final_field_mask.png`: inspect the default final crop-filtering behavior",
            "- `normalized_side_by_side.png`: whether positive and negative inputs are at a comparable working scale before masking",
            "",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def process_image(
    image: Image.Image,
    image_role: str,
    source_key: str,
    raw_path: str,
    output_dir: Path,
    window_sizes: list[int],
    stride_ratio: float,
    distance_offset: float,
    background_band_half_width: int,
    morph_radius_ratio: float,
    shrink_ratio: float,
    coverage_threshold: float,
    final_mask_erosion_ratio: float,
    use_convex_hull: bool,
    resize_info: ResizeInfo,
    annotation_boxes: list[AnnotationBox] | None = None,
) -> tuple[list[WindowRecord], dict[str, object]]:
    lab = pil_lab_array(image)
    descriptor, histogram, background_band = estimate_background_descriptor(lab, band_half_width=background_band_half_width)
    distance_map, initial_foreground, morph_mask, field_mask, threshold = field_mask_from_background(
        image,
        descriptor=descriptor,
        morph_radius_ratio=morph_radius_ratio,
        distance_offset=distance_offset,
        use_convex_hull=use_convex_hull,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    image.save(output_dir / "original.png")
    (output_dir / "step0_resize_info.json").write_text(
        json.dumps(
            {
                "original_width": resize_info.original_width,
                "original_height": resize_info.original_height,
                "resized_width": resize_info.resized_width,
                "resized_height": resize_info.resized_height,
                "scale_x": resize_info.scale_x,
                "scale_y": resize_info.scale_y,
            },
            indent=2,
        )
    )
    render_histogram(output_dir / "step1_histogram.png", histogram, descriptor)
    render_background_band(image, background_band, output_dir / "step1_background_band.png")
    render_distance_map(distance_map, threshold, output_dir / "step2_distance_map.png")
    render_binary_mask(initial_foreground.astype(np.uint8), output_dir / "step2_initial_foreground.png", "Stage 2 Initial Foreground")
    render_binary_mask(morph_mask.astype(np.uint8), output_dir / "step2_morph_mask.png", "Stage 2 Morphology Result")
    render_binary_mask(field_mask.astype(np.uint8), output_dir / "step2_field_mask.png", "Stage 2 Largest Component Field Mask")
    zero_outside_mask(image, field_mask).save(output_dir / "step2_field_mask_zero_filled.png")

    fitted = fit_ellipse_to_mask(field_mask)
    shrunk = shrink_ellipse(fitted, shrink_ratio)
    valid_mask = ellipse_mask(field_mask.shape, shrunk)
    erosion_radius = max(1, int(round(min(image.size) * final_mask_erosion_ratio)))
    final_mask = erode_mask(field_mask, erosion_radius)
    render_ellipse_overlay(image, field_mask, fitted, shrunk, output_dir / "step3_ellipse_overlay.png", annotation_boxes=annotation_boxes)
    render_final_mask_overlay(image, field_mask, final_mask, fitted, output_dir / "step3_final_mask_overlay.png", annotation_boxes=annotation_boxes)

    ellipse_zero_filled = zero_outside_mask(image, valid_mask)
    zero_filled = zero_outside_mask(image, final_mask)
    zero_filled.save(output_dir / "step4_zero_filled.png")
    render_binary_mask(final_mask.astype(np.uint8), output_dir / "step4_final_mask.png", "Stage 4 Final Eroded Field Mask")

    ellipse_windows = build_windows(
        image_role=image_role,
        source_key=source_key,
        raw_path=raw_path,
        mask=valid_mask,
        validity_source="ellipse",
        window_sizes=window_sizes,
        stride_ratio=stride_ratio,
        coverage_threshold=coverage_threshold,
    )
    mask_windows = build_windows(
        image_role=image_role,
        source_key=source_key,
        raw_path=raw_path,
        mask=field_mask,
        validity_source="field_mask",
        window_sizes=window_sizes,
        stride_ratio=stride_ratio,
        coverage_threshold=coverage_threshold,
    )
    final_windows = build_windows(
        image_role=image_role,
        source_key=source_key,
        raw_path=raw_path,
        mask=final_mask,
        validity_source="final_field_mask",
        window_sizes=window_sizes,
        stride_ratio=stride_ratio,
        coverage_threshold=coverage_threshold,
    )
    render_window_overlay(ellipse_zero_filled, ellipse_windows, output_dir / "step5_window_overlay_ellipse.png", "Ellipse Coverage: Accepted (green) vs Rejected (red) Windows")
    render_window_overlay(zero_outside_mask(image, field_mask), mask_windows, output_dir / "step5_window_overlay_field_mask.png", "Field-Mask Coverage: Accepted (green) vs Rejected (red) Windows")
    render_window_overlay(zero_filled, final_windows, output_dir / "step5_window_overlay.png", "Final Eroded-Field-Mask Coverage: Accepted (green) vs Rejected (red) Windows")
    render_coverage_histogram(ellipse_windows, coverage_threshold, output_dir / "step5_coverage_histogram_ellipse.png")
    render_coverage_histogram(mask_windows, coverage_threshold, output_dir / "step5_coverage_histogram_field_mask.png")
    render_coverage_histogram(final_windows, coverage_threshold, output_dir / "step5_coverage_histogram.png")

    def emit_threshold_sheets(windows: list[WindowRecord], source_name: str, image_for_crops: Image.Image) -> None:
        accepted_near = sorted(
            [window for window in windows if window.accepted and coverage_threshold <= window.coverage < min(1.0, coverage_threshold + 0.10)],
            key=lambda window: window.coverage,
        )
        rejected_near = sorted(
            [window for window in windows if (coverage_threshold - 0.15) <= window.coverage < coverage_threshold],
            key=lambda window: window.coverage,
            reverse=True,
        )
        border_kept = sorted(
            [window for window in windows if window.accepted and window.touches_border],
            key=lambda window: window.coverage,
        )
        crop_contact_sheet(image_for_crops, accepted_near, output_dir / f"step6_accepted_near_threshold_{source_name}.png", f"{source_name} Accepted Crops Near Threshold")
        crop_contact_sheet(image_for_crops, rejected_near, output_dir / f"step6_rejected_near_threshold_{source_name}.png", f"{source_name} Rejected Crops Near Threshold")
        crop_contact_sheet(image_for_crops, border_kept, output_dir / f"step6_border_kept_examples_{source_name}.png", f"{source_name} Accepted Border-Touching Crops")

    emit_threshold_sheets(ellipse_windows, "ellipse", ellipse_zero_filled)
    emit_threshold_sheets(mask_windows, "field_mask", zero_outside_mask(image, field_mask))
    emit_threshold_sheets(final_windows, "final_field_mask", zero_filled)

    summary: dict[str, object] = {
        "image_role": image_role,
        "image_type": descriptor.image_type,
        "original_width": resize_info.original_width,
        "original_height": resize_info.original_height,
        "resized_width": resize_info.resized_width,
        "resized_height": resize_info.resized_height,
        "sigma_l": descriptor.sigma_l,
        "field_fraction": float(field_mask.mean()),
        "ellipse_fraction": float(valid_mask.mean()),
        "final_mask_fraction": float(final_mask.mean()),
        "ellipse_total_windows": len(ellipse_windows),
        "ellipse_accepted_windows": sum(window.accepted for window in ellipse_windows),
        "ellipse_rejected_windows": sum(not window.accepted for window in ellipse_windows),
        "ellipse_border_accepted": sum(window.accepted and window.touches_border for window in ellipse_windows),
        "mask_total_windows": len(mask_windows),
        "mask_accepted_windows": sum(window.accepted for window in mask_windows),
        "mask_rejected_windows": sum(not window.accepted for window in mask_windows),
        "mask_border_accepted": sum(window.accepted and window.touches_border for window in mask_windows),
        "final_total_windows": len(final_windows),
        "final_accepted_windows": sum(window.accepted for window in final_windows),
        "final_rejected_windows": sum(not window.accepted for window in final_windows),
        "final_border_accepted": sum(window.accepted and window.touches_border for window in final_windows),
        "threshold": threshold,
        "final_mask_erosion_ratio": final_mask_erosion_ratio,
        "final_mask_erosion_radius": erosion_radius,
        "use_convex_hull": use_convex_hull,
        "fitted_ellipse": {
            "center_x": fitted.center_x,
            "center_y": fitted.center_y,
            "semi_a": fitted.semi_a,
            "semi_b": fitted.semi_b,
            "angle_rad": fitted.angle_rad,
        },
        "shrunk_ellipse": {
            "center_x": shrunk.center_x,
            "center_y": shrunk.center_y,
            "semi_a": shrunk.semi_a,
            "semi_b": shrunk.semi_b,
            "angle_rad": shrunk.angle_rad,
        },
    }
    if annotation_boxes:
        coverages = [annotation_ellipse_coverage(box, valid_mask) for box in annotation_boxes]
        summary["annotation_min_coverage"] = float(min(coverages)) if coverages else 0.0
        summary["annotation_mean_coverage"] = float(np.mean(coverages)) if coverages else 0.0
        mask_coverages = [annotation_ellipse_coverage(box, field_mask) for box in annotation_boxes]
        summary["annotation_mask_min_coverage"] = float(min(mask_coverages)) if mask_coverages else 0.0
        summary["annotation_mask_mean_coverage"] = float(np.mean(mask_coverages)) if mask_coverages else 0.0
        final_coverages = [annotation_ellipse_coverage(box, final_mask) for box in annotation_boxes]
        summary["annotation_final_min_coverage"] = float(min(final_coverages)) if final_coverages else 0.0
        summary["annotation_final_mean_coverage"] = float(np.mean(final_coverages)) if final_coverages else 0.0
    return ellipse_windows + mask_windows + final_windows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototype the simplified border-preprocessing pipeline on one positive and one negative full image.")
    parser.add_argument("--positive-raw", type=Path, default=None)
    parser.add_argument("--positive-annotation", type=Path, default=None)
    parser.add_argument("--negative-raw", type=Path, default=None)
    parser.add_argument("--annotation-csv", type=Path, default=Path("eda_outputs/annotation_boxes.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("phase_a_border_preprocessing_prototype"))
    parser.add_argument("--window-sizes", nargs="+", type=int, default=list(DEFAULT_WINDOW_SIZES))
    parser.add_argument("--stride-ratio", type=float, default=DEFAULT_STRIDE_RATIO)
    parser.add_argument("--distance-offset", type=float, default=DEFAULT_DISTANCE_OFFSET)
    parser.add_argument("--background-band-half-width", type=int, default=DEFAULT_BACKGROUND_BAND_HALF_WIDTH)
    parser.add_argument("--morph-radius-ratio", type=float, default=DEFAULT_MORPH_RADIUS_RATIO)
    parser.add_argument("--shrink-ratio", type=float, default=DEFAULT_SHRINK_RATIO)
    parser.add_argument("--coverage-threshold", type=float, default=DEFAULT_COVERAGE_THRESHOLD)
    parser.add_argument("--normalize-shorter-dim", type=int, default=DEFAULT_NORMALIZE_SHORTER_DIM)
    parser.add_argument("--final-mask-erosion-ratio", type=float, default=DEFAULT_FINAL_MASK_EROSION_RATIO)
    parser.add_argument("--no-convex-hull", action="store_true")
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.positive_raw is None or args.negative_raw is None:
        default_positive_raw, default_positive_annotation, default_negative_raw, positive_source_key, negative_source_key = default_image_selection(project_root)
        positive_raw = args.positive_raw.resolve() if args.positive_raw else default_positive_raw
        positive_annotation = args.positive_annotation.resolve() if args.positive_annotation else default_positive_annotation
        negative_raw = args.negative_raw.resolve() if args.negative_raw else default_negative_raw
    else:
        positive_raw = args.positive_raw.resolve()
        positive_annotation = args.positive_annotation.resolve() if args.positive_annotation else project_root / "datasets" / "Positive" / "Labelled" / positive_raw.name
        negative_raw = args.negative_raw.resolve()
        positive_source_key = f"positive:{positive_raw.stem}"
        negative_source_key = f"negative:{negative_raw.stem}"

    positive_image = open_rgb_image(positive_raw)
    negative_image = open_rgb_image(negative_raw)
    with Image.open(positive_annotation) as labelled_image:
        labelled_size = ImageOps.exif_transpose(labelled_image).size
    positive_boxes = scale_annotation_boxes(
        load_annotation_boxes(args.annotation_csv.resolve(), positive_annotation.name),
        labelled_size=labelled_size,
        raw_size=positive_image.size,
    )
    positive_image, positive_boxes, positive_resize_info = resize_image_and_boxes(
        positive_image,
        positive_boxes,
        normalize_shorter_dim=args.normalize_shorter_dim,
    )
    negative_image, _, negative_resize_info = resize_image_and_boxes(
        negative_image,
        None,
        normalize_shorter_dim=args.normalize_shorter_dim,
    )
    render_side_by_side(
        [
            (f"positive resized\n{positive_image.width}x{positive_image.height}", positive_image),
            (f"negative resized\n{negative_image.width}x{negative_image.height}", negative_image),
        ],
        output_dir / "normalized_side_by_side.png",
        "Step 0 Normalized Inputs",
    )

    positive_windows, positive_summary = process_image(
        image=positive_image,
        image_role="positive",
        source_key=positive_source_key,
        raw_path=str(positive_raw.relative_to(project_root)),
        output_dir=output_dir / "positive",
        window_sizes=list(args.window_sizes),
        stride_ratio=args.stride_ratio,
        distance_offset=args.distance_offset,
        background_band_half_width=args.background_band_half_width,
        morph_radius_ratio=args.morph_radius_ratio,
        shrink_ratio=args.shrink_ratio,
        coverage_threshold=args.coverage_threshold,
        final_mask_erosion_ratio=args.final_mask_erosion_ratio,
        use_convex_hull=not args.no_convex_hull,
        resize_info=positive_resize_info,
        annotation_boxes=positive_boxes,
    )
    negative_windows, negative_summary = process_image(
        image=negative_image,
        image_role="negative",
        source_key=negative_source_key,
        raw_path=str(negative_raw.relative_to(project_root)),
        output_dir=output_dir / "negative",
        window_sizes=list(args.window_sizes),
        stride_ratio=args.stride_ratio,
        distance_offset=args.distance_offset,
        background_band_half_width=args.background_band_half_width,
        morph_radius_ratio=args.morph_radius_ratio,
        shrink_ratio=args.shrink_ratio,
        coverage_threshold=args.coverage_threshold,
        final_mask_erosion_ratio=args.final_mask_erosion_ratio,
        use_convex_hull=not args.no_convex_hull,
        resize_info=negative_resize_info,
    )

    write_csv(output_dir / "window_metadata.csv", coverage_window_rows(positive_windows + negative_windows))
    run_manifest = {
        "positive_raw": str(positive_raw),
        "positive_annotation": str(positive_annotation),
        "negative_raw": str(negative_raw),
        "positive_source_key": positive_source_key,
        "negative_source_key": negative_source_key,
        "window_sizes": list(args.window_sizes),
        "stride_ratio": args.stride_ratio,
        "distance_offset": args.distance_offset,
        "background_band_half_width": args.background_band_half_width,
        "morph_radius_ratio": args.morph_radius_ratio,
        "shrink_ratio": args.shrink_ratio,
        "coverage_threshold": args.coverage_threshold,
        "normalize_shorter_dim": args.normalize_shorter_dim,
        "final_mask_erosion_ratio": args.final_mask_erosion_ratio,
        "use_convex_hull": not args.no_convex_hull,
        "positive_summary": positive_summary,
        "negative_summary": negative_summary,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2))
    build_report(
        output_dir / "phase_a_report.md",
        positive_source_key=positive_source_key,
        negative_source_key=negative_source_key,
        args=args,
        summaries=[positive_summary, negative_summary],
    )
    print(f"Wrote {output_dir / 'phase_a_report.md'}")


if __name__ == "__main__":
    main()
