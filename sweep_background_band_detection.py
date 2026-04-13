from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage

from prototype_border_preprocessing_pipeline import (
    DEFAULT_DISTANCE_OFFSET,
    DEFAULT_FINAL_MASK_EROSION_RATIO,
    DEFAULT_MORPH_RADIUS_RATIO,
    DEFAULT_NORMALIZE_SHORTER_DIM,
    EllipseParameters,
    erode_mask,
    estimate_background_descriptor,
    field_mask_from_background,
    fit_ellipse_to_mask,
    open_rgb_image,
    read_csv,
    render_side_by_side,
    resize_image_and_boxes,
)


def choose_images(project_root: Path, n_positive: int, n_negative: int, source_keys: list[str] | None = None) -> list[dict[str, str]]:
    rows = read_csv(project_root / "full_image_sliding_window_outputs" / "sampled_source_images.csv")
    if source_keys:
        wanted = set(source_keys)
        return [row for row in rows if row["source_key"] in wanted]
    positives = sorted((row for row in rows if row["label"] == "positive"), key=lambda row: int(row["source_id"]))[:n_positive]
    negatives = sorted((row for row in rows if row["label"] == "negative"), key=lambda row: int(row["source_id"]))[:n_negative]
    return positives + negatives


def ellipse_points(ellipse: EllipseParameters, num_points: int = 360) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, num_points)
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    cos_a = np.cos(ellipse.angle_rad)
    sin_a = np.sin(ellipse.angle_rad)
    xs = ellipse.center_x + ellipse.semi_a * cos_t * cos_a - ellipse.semi_b * sin_t * sin_a
    ys = ellipse.center_y + ellipse.semi_a * cos_t * sin_a + ellipse.semi_b * sin_t * cos_a
    return np.stack([xs, ys], axis=1)


def overlay_background_band(image: Image.Image, band: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image).astype(np.uint8)
    overlay = rgb.copy()
    overlay[band] = np.array([255, 128, 0], dtype=np.uint8)
    return (0.65 * rgb + 0.35 * overlay).astype(np.uint8)


def draw_detection_overlay(image: Image.Image, field_mask: np.ndarray, final_mask: np.ndarray, ellipse: EllipseParameters) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    field_boundary = field_mask & ~ndimage.binary_erosion(field_mask)
    final_boundary = final_mask & ~ndimage.binary_erosion(final_mask)
    ys, xs = np.nonzero(field_boundary)
    ax.scatter(xs, ys, s=0.4, c="yellow", alpha=0.35)
    ys2, xs2 = np.nonzero(final_boundary)
    ax.scatter(xs2, ys2, s=0.4, c="lime", alpha=0.35)
    pts = ellipse_points(ellipse)
    ax.plot(pts[:, 0], pts[:, 1], color="cyan", linewidth=1.2)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3].copy()
    plt.close(fig)
    return array


def save_board(
    image: Image.Image,
    values: list[int],
    distance_offset: float,
    morph_radius_ratio: float,
    final_mask_erosion_ratio: float,
    use_convex_hull: bool,
    output_path: Path,
    label: str,
    source_key: str,
) -> list[dict[str, object]]:
    image_resized, _, resize_info = resize_image_and_boxes(image, None, DEFAULT_NORMALIZE_SHORTER_DIM)
    records: list[dict[str, object]] = []
    cols = len(values)
    fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 8))
    if cols == 1:
        axes = np.asarray(axes).reshape(2, 1)
    else:
        axes = np.asarray(axes)
    for axis in axes.flat:
        axis.axis("off")

    for col, band_half_width in enumerate(values):
        lab = np.asarray(image_resized.convert("LAB"), dtype=np.float32)
        descriptor, _, background_band = estimate_background_descriptor(lab, band_half_width=band_half_width)
        _, _, _, field_mask, threshold = field_mask_from_background(
            image_resized,
            descriptor=descriptor,
            morph_radius_ratio=morph_radius_ratio,
            distance_offset=distance_offset,
            use_convex_hull=use_convex_hull,
        )
        ellipse = fit_ellipse_to_mask(field_mask)
        erosion_radius = max(1, int(round(min(image_resized.size) * final_mask_erosion_ratio)))
        final_mask = erode_mask(field_mask, erosion_radius)

        axes[0, col].imshow(overlay_background_band(image_resized, background_band))
        axes[0, col].set_title(f"band=±{band_half_width}\nseed sigma={descriptor.sigma_l:.1f}", fontsize=9)
        axes[0, col].axis("off")

        axes[1, col].imshow(draw_detection_overlay(image_resized, field_mask, final_mask, ellipse))
        axes[1, col].set_title(
            f"field={field_mask.mean():.3f}\nfinal={final_mask.mean():.3f}\nthr={threshold:.1f}",
            fontsize=9,
        )
        axes[1, col].axis("off")

        records.append(
            {
                "source_key": source_key,
                "label": label,
                "band_half_width": band_half_width,
                "resized_width": resize_info.resized_width,
                "resized_height": resize_info.resized_height,
                "background_sigma": round(descriptor.sigma_l, 4),
                "image_type": descriptor.image_type,
                "field_fraction": round(float(field_mask.mean()), 6),
                "final_fraction": round(float(final_mask.mean()), 6),
                "threshold": round(float(threshold), 4),
                "use_convex_hull": use_convex_hull,
            }
        )

    fig.suptitle(f"{source_key} ({label})\nTop: Stage 1 background seed, Bottom: final detection", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return records


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_combined_frame(rows: list[dict[str, str]], output_dir: Path, output_path: Path) -> None:
    positives = [row for row in rows if row["label"] == "positive"]
    negatives = [row for row in rows if row["label"] == "negative"]
    ordered = positives + negatives
    if not ordered:
        return

    board_lookup: dict[str, Image.Image] = {}
    for row in ordered:
        board_path = output_dir / f"{row['source_key'].replace(':', '_')}__band_sweep.png"
        with Image.open(board_path) as image:
            board_lookup[row["source_key"]] = image.convert("RGB")

    board_width = max(image.width for image in board_lookup.values())
    board_height = max(image.height for image in board_lookup.values())
    cols = max(len(positives), len(negatives), 1)
    rows_count = 2 if negatives else 1
    header_height = 72
    gutter = 16
    canvas_width = cols * board_width + (cols + 1) * gutter
    canvas_height = rows_count * board_height + (rows_count + 1) * gutter + header_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gutter, 16), "Combined border-detection sweep", fill="black")
    draw.text((gutter, 40), "Top row: positives, bottom row: negatives", fill="black")

    def paste_group(group: list[dict[str, str]], row_index: int) -> None:
        y = header_height + gutter + row_index * (board_height + gutter)
        for col_index, row in enumerate(group):
            x = gutter + col_index * (board_width + gutter)
            image = board_lookup[row["source_key"]]
            offset_x = x + (board_width - image.width) // 2
            offset_y = y + (board_height - image.height) // 2
            canvas.paste(image, (offset_x, offset_y))

    paste_group(positives, 0)
    if negatives:
        paste_group(negatives, 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep the Stage 1 background-band half-width over 5 positive and 5 negative images.")
    parser.add_argument("--output-dir", type=Path, default=Path("background_band_detection_sweep"))
    parser.add_argument("--band-half-widths", nargs="+", type=int, default=[34, 40, 46, 50])
    parser.add_argument("--n-positive", type=int, default=5)
    parser.add_argument("--n-negative", type=int, default=5)
    parser.add_argument("--distance-offset", type=float, default=DEFAULT_DISTANCE_OFFSET)
    parser.add_argument("--morph-radius-ratio", type=float, default=DEFAULT_MORPH_RADIUS_RATIO)
    parser.add_argument("--final-mask-erosion-ratio", type=float, default=DEFAULT_FINAL_MASK_EROSION_RATIO)
    parser.add_argument("--no-convex-hull", action="store_true")
    parser.add_argument("--source-keys", nargs="*", default=None)
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = choose_images(project_root, args.n_positive, args.n_negative, source_keys=args.source_keys)
    summary_rows: list[dict[str, object]] = []
    for row in rows:
        image = open_rgb_image(project_root / row["raw_path"])
        board_path = output_dir / f"{row['source_key'].replace(':', '_')}__band_sweep.png"
        summary_rows.extend(
            save_board(
                image=image,
                values=list(args.band_half_widths),
                distance_offset=args.distance_offset,
                morph_radius_ratio=args.morph_radius_ratio,
                final_mask_erosion_ratio=args.final_mask_erosion_ratio,
                use_convex_hull=not args.no_convex_hull,
                output_path=board_path,
                label=row["label"],
                source_key=row["source_key"],
            )
        )

    build_combined_frame(rows, output_dir, output_dir / "combined_band_sweep.png")
    write_csv(output_dir / "sweep_summary.csv", summary_rows)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "band_half_widths": list(args.band_half_widths),
                "n_positive": args.n_positive,
                "n_negative": args.n_negative,
                "distance_offset": args.distance_offset,
                "morph_radius_ratio": args.morph_radius_ratio,
                "final_mask_erosion_ratio": args.final_mask_erosion_ratio,
                "use_convex_hull": not args.no_convex_hull,
                "source_keys": args.source_keys,
            },
            indent=2,
        )
    )
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
