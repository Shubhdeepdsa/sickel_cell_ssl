from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from prototype_border_preprocessing_pipeline import (
    DEFAULT_NORMALIZE_SHORTER_DIM,
    BackgroundDescriptor,
    ResizeInfo,
    erode_mask,
    estimate_background_descriptor,
    field_mask_from_background,
    pil_lab_array,
    resize_image_and_boxes,
)


@dataclass(frozen=True)
class PreprocessingStages:
    descriptor: BackgroundDescriptor
    resize_info: ResizeInfo
    resized_image: Image.Image
    background_band: np.ndarray
    initial_foreground: np.ndarray
    morph_mask: np.ndarray
    field_mask: np.ndarray
    final_mask: np.ndarray
    threshold: float
    erosion_radius: int


def compute_preprocessing_stages(
    image: Image.Image,
    background_band_half_width: int,
    distance_offset: float,
    morph_radius_ratio: float,
    final_mask_erosion_ratio: float,
) -> PreprocessingStages:
    resized_image, _, resize_info = resize_image_and_boxes(image, None, DEFAULT_NORMALIZE_SHORTER_DIM)
    lab = pil_lab_array(resized_image)
    descriptor, _, background_band = estimate_background_descriptor(lab, band_half_width=background_band_half_width)
    distance_map, initial_foreground, morph_mask, field_mask, threshold = field_mask_from_background(
        resized_image,
        descriptor=descriptor,
        morph_radius_ratio=morph_radius_ratio,
        distance_offset=distance_offset,
        use_convex_hull=True,
    )
    erosion_radius = max(1, int(round(min(resized_image.size) * final_mask_erosion_ratio)))
    final_mask = erode_mask(field_mask, erosion_radius)
    return PreprocessingStages(
        descriptor=descriptor,
        resize_info=resize_info,
        resized_image=resized_image,
        background_band=background_band,
        initial_foreground=initial_foreground,
        morph_mask=morph_mask,
        field_mask=field_mask,
        final_mask=final_mask,
        threshold=threshold,
        erosion_radius=erosion_radius,
    )
