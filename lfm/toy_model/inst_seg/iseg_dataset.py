"""Toy instance segmentation metadata helpers used by Lightning wrappers."""

from __future__ import annotations

import os
import warnings
from glob import glob
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio

from lfm.all_models.all_tasks.data.image_io import read_image_file

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

warnings.filterwarnings("ignore", message=".*HF Hub.*")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")


def get_input_metadata(
    base_dir: str,
    band_filter: Optional[list[int]] = None,
) -> list[str]:
    """Extract band metadata and return DINO RGB weight assignments per band."""
    image_dir = f"{base_dir}/chips"
    all_image_paths: list[str] = []
    for pattern in ("*.tif", "*.tiff", "*.npy", "*.npz", "*.nc"):
        all_image_paths.extend(glob(os.path.join(image_dir, pattern)))
    all_image_paths = sorted(all_image_paths)
    if not all_image_paths:
        raise FileNotFoundError(
            f"No image chips found in {image_dir} for .tif/.tiff/.npy/.npz/.nc"
        )
    image_path = all_image_paths[0]

    wavelengths = {0: 415, 1: 566, 2: 604, 3: 643, 4: 689}

    def _get_weight_assignment(description: str, band_idx: int) -> str:
        desc_lower = description.lower()
        if band_idx in wavelengths:
            wl = wavelengths[band_idx]
            if wl < 500:
                return "blue"
            if wl < 580:
                return "green"
            if wl < 620:
                return "0.7*red+0.3*green"
            return "red"
        if "uv" in desc_lower:
            return "blue"
        return "red"

    if Path(image_path).suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(image_path) as src:
            num_bands = src.count
            descriptions = [
                src.descriptions[i] or f"Band {i + 1}" for i in range(num_bands)
            ]
    else:
        image = np.asarray(read_image_file(Path(image_path)))
        if image.ndim == 2:
            num_bands = 1
        elif image.ndim == 3:
            num_bands = (
                image.shape[0] if image.shape[0] <= image.shape[-1] else image.shape[-1]
            )
        else:
            raise ValueError(f"Expected 2D or 3D image array, got {image.shape}")
        descriptions = [f"Band {i + 1}" for i in range(num_bands)]

    if band_filter is None:
        band_filter = list(range(num_bands))

    return [_get_weight_assignment(descriptions[idx], idx) for idx in band_filter]
