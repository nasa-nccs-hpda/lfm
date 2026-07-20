"""Shared helpers for the TerraTorch data adapters."""

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


def load_nc_band(path: str | Path, band_name: str = "band_data") -> np.ndarray:
    """Load a single band from a NetCDF4/HDF5 ``.nc`` file as a 2-D array.

    Args:
        path: Path to the ``.nc`` file.
        band_name: HDF5 dataset key to read (e.g. ``"band_data"`` for the
            NAC/DTM tiles, ``"data"`` for the IMP segmentation tiles).

    Returns:
        A float32 array of shape ``(H, W)``.  If the stored band is
        ``(1, H, W)`` (single-band-first convention) the leading singleton
        is squeezed away.
    """
    with h5py.File(path, "r") as f:
        arr = np.asarray(f[band_name], dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


class D4Transform:
    """Random symmetry from the D4 group (8 elements).

    Applies a uniformly-random combination of {0°, 90°, 180°, 270°} rotation
    and an optional horizontal flip to both the image and its dense targets,
    keeping pixel-to-pixel alignment.  Meant to be used as the ``transforms``
    argument of a dataset that returns per-sample dicts.

    Recognised keys:

    * ``"image"`` — ``(C, H, W)`` float tensor.
    * ``mask_key`` (default ``"mask"``) — ``(H, W)`` or ``(1, H, W)`` int
      tensor.  Missing keys are ignored.

    Only sensible for square inputs (H == W) because rotations by 90° / 270°
    swap the last two dimensions.

    Args:
        mask_key: Sample-dict key that holds the segmentation mask (must
            match the ``mask_output_tag`` used by the dataset).
        p: Probability of applying a *non-identity* transform.  ``1.0``
            means always sample one of the 8 group elements uniformly;
            ``0.0`` disables the transform entirely.
    """

    _NUM_ROTATIONS: int = 4  # 0°, 90°, 180°, 270°

    def __init__(self, mask_key: str = "mask", p: float = 1.0) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.mask_key = mask_key
        self.p = p

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if self.p <= 0.0 or torch.rand(()).item() >= self.p:
            return sample
        k = int(torch.randint(self._NUM_ROTATIONS, (1,)).item())
        flip = bool(torch.randint(2, (1,)).item())

        image = sample.get("image")
        if isinstance(image, torch.Tensor):
            if k:
                image = torch.rot90(image, k=k, dims=(-2, -1))
            if flip:
                image = torch.flip(image, dims=(-1,))
            sample["image"] = image

        mask = sample.get(self.mask_key)
        if isinstance(mask, torch.Tensor):
            if k:
                mask = torch.rot90(mask, k=k, dims=(-2, -1))
            if flip:
                mask = torch.flip(mask, dims=(-1,))
            sample[self.mask_key] = mask

        return sample
