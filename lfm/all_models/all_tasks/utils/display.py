"""Display helpers for segmentation plots."""

from __future__ import annotations

import numpy as np


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
        display_img = img
        note = "Grayscale"
        squeeze_grayscale = True
    elif num_channels == 2:
        display_img = img[:, :, [0]]
        note = "2ch (band 0 grayscale)"
        squeeze_grayscale = True
    elif num_channels == 3:
        display_img = img[:, :, [2, 1, 0]]
        note = "RGB (BGR->RGB)"
        squeeze_grayscale = False
    elif num_channels >= 5:
        display_img = img[:, :, [3, 1, 0]]
        note = f"{num_channels}ch (RGB from bands 3,1,0)"
        squeeze_grayscale = False
    else:
        display_img = img[:, :, :3]
        note = f"{num_channels}ch (first 3)"
        squeeze_grayscale = False

    if method == "percentile":
        img_normalized = np.zeros_like(display_img, dtype=np.float32)
        for c in range(display_img.shape[2]):
            band = display_img[:, :, c]
            p_low, p_high = np.percentile(
                band, [clip_percentile, 100 - clip_percentile]
            )
            band = np.clip(band, p_low, p_high)
            img_normalized[:, :, c] = (band - p_low) / (p_high - p_low + 1e-8)
    elif method == "std_clip":
        img_clipped = np.clip(display_img, -std_clip, std_clip)
        img_normalized = (img_clipped + std_clip) / (2 * std_clip)
    elif method == "minmax":
        img_normalized = (
            display_img - display_img.min()
        ) / (display_img.max() - display_img.min() + 1e-8)
    elif method is None:
        img_normalized = display_img
    else:
        raise ValueError(f"Unknown display method: {method}")
    img_normalized = np.clip(img_normalized, 0, 1)

    if squeeze_grayscale:
        return img_normalized[:, :, 0], note
    return img_normalized, note


def create_overlay_image(img_vis: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    color = (1.0, 1.0, 0.0)
    return create_colored_overlay_image(img_vis, pred_mask, color=color)


def create_colored_overlay_image(
    img_vis: np.ndarray,
    pred_mask: np.ndarray,
    *,
    color: tuple[float, float, float],
) -> np.ndarray:

    if img_vis.ndim == 2:
        img_rgb = np.stack([img_vis] * 3, axis=2)
    else:
        img_rgb = img_vis

    overlay = img_rgb.copy()
    mask_bool = pred_mask == 1
    overlay[mask_bool, 0] = color[0]
    overlay[mask_bool, 1] = color[1]
    overlay[mask_bool, 2] = color[2]
    alpha = np.where(mask_bool[:, :, None], 0.5, 0.0)
    return np.clip(overlay * alpha + img_rgb * (1 - alpha), 0, 1)
