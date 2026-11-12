"""Display helpers for segmentation plots."""

from __future__ import annotations

import numpy as np

from lfm.all_models.plotting import DisplayPolicy, blend_overlay, prepare_display


def prepare_image_for_display(
    img: np.ndarray,
    *,
    method: str = "minmax",
    clip_percentile: float = 2.0,
    std_clip: float = 3.0,
) -> tuple[np.ndarray, str]:
    """Convert HWC multispectral image to displayable grayscale/RGB."""
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    num_channels = img.shape[2]

    if num_channels == 1:
        note = "Grayscale"
    elif num_channels == 2:
        note = "2ch (band 0 grayscale)"
    elif num_channels == 3:
        note = "RGB (BGR->RGB)"
    elif num_channels >= 5:
        note = f"{num_channels}ch (RGB from bands 3,1,0)"
    else:
        note = f"{num_channels}ch (first 3)"

    return prepare_display(
        img,
        policy=DisplayPolicy(
            bands=tuple(
                [0]
                if num_channels <= 2
                else [2, 1, 0]
                if num_channels == 3
                else [3, 1, 0]
                if num_channels >= 5
                else [0, 1, 2]
            ),
            stretch="global_minmax" if method == "minmax" else method,
            clip_percentile=clip_percentile,
            std_clip=std_clip,
            epsilon=1e-8,
            validate_finite=False,
        ),
    ), note


def create_overlay_image(img_vis: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    color = (1.0, 1.0, 0.0)
    return create_colored_overlay_image(img_vis, pred_mask, color=color)


def create_colored_overlay_image(
    img_vis: np.ndarray,
    pred_mask: np.ndarray,
    *,
    color: tuple[float, float, float],
) -> np.ndarray:

    return blend_overlay(img_vis, color, foreground=pred_mask == 1, alpha=0.5)
