"""Tensor conversion helpers for lunar segmentation data."""

from __future__ import annotations

import numpy as np
import torch


def image_to_chw_float(arr: np.ndarray) -> torch.Tensor:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3:
        # rasterio returns CHW; tifffile commonly returns HWC.
        if (
            arr.shape[0] not in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
            and arr.shape[-1] <= 32
        ):
            arr = np.moveaxis(arr, -1, 0)
    else:
        raise ValueError(f"Expected 2D or 3D image array, got shape {arr.shape}")
    return torch.as_tensor(arr, dtype=torch.float32)


def mask_to_hw_long(arr: np.ndarray) -> torch.Tensor:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[0] if arr.shape[0] <= arr.shape[-1] else arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D label array, got shape {arr.shape}")
    return torch.as_tensor(arr, dtype=torch.long)


def shift_mask(mask: torch.Tensor, shift_xy: tuple[int, int] | None) -> torch.Tensor:
    """Translate a HW mask by integer pixels, filling exposed pixels with 0."""
    if shift_xy is None:
        return mask
    shift_x, shift_y = int(shift_xy[0]), int(shift_xy[1])
    if shift_x == 0 and shift_y == 0:
        return mask
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask for shift, got shape {tuple(mask.shape)}")

    height, width = mask.shape
    shifted = torch.zeros_like(mask)

    src_x0 = max(0, -shift_x)
    src_x1 = min(width, width - shift_x)
    dst_x0 = max(0, shift_x)
    dst_x1 = min(width, width + shift_x)

    src_y0 = max(0, -shift_y)
    src_y1 = min(height, height - shift_y)
    dst_y0 = max(0, shift_y)
    dst_y1 = min(height, height + shift_y)

    if src_x0 >= src_x1 or src_y0 >= src_y1:
        return shifted

    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return shifted
