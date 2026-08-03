"""Toy semantic segmentation dataset helpers used by Lightning wrappers."""

from __future__ import annotations

import os
import warnings
from glob import glob
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import torch

from lfm.all_models.all_tasks.data import (
    FinetuneStatsNormalization,
    NoDataPolicy,
    NoNormalization,
    read_image_file,
)
from lfm.all_models.sem_seg import SemanticSegmentationDataset

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

warnings.filterwarnings("ignore", message=".*HF Hub.*")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")


class LunarCraterDataset(SemanticSegmentationDataset):
    """Toy semantic dataset preserving the historical tuple output contract."""

    def __init__(
        self,
        base_dir: str,
        mean: np.ndarray,
        std: np.ndarray,
        target_size: tuple[int, int] = (304, 304),
        max_samples: int | None = None,
        band_filter: list[int] | None = None,
        normalize_inputs: bool = True,
        scale_inputs: bool = True,
        spatial_transform: str = "resize",
        split_name: str | None = None,
        image_file_type: str = ".tif",
        label_file_type: str = ".npy",
        image_glob: str | None = None,
        label_glob: str | None = None,
        image_suffix: str | None = None,
        label_suffix: str | None = None,
        label_npz_key: str = "mask",
        binarize_label: bool = False,
        ignore_nodata_in_loss: bool = False,
        nodata_ignore_index: int = -1,
    ) -> None:
        self.mean = mean.astype(np.float32) if mean is not None else None
        self.std = std.astype(np.float32) if std is not None else None
        self.normalize_inputs = normalize_inputs
        self.image_file_type = image_file_type
        self.label_file_type = label_file_type
        self.ignore_nodata_in_loss = ignore_nodata_in_loss
        self.nodata_ignore_index = int(nodata_ignore_index)
        normalization = (
            FinetuneStatsNormalization(self.mean, self.std)
            if normalize_inputs and self.mean is not None and self.std is not None
            else NoNormalization()
        )
        super().__init__(
            base_dir,
            target_size=target_size,
            max_samples=max_samples,
            band_filter=band_filter,
            spatial_transform=spatial_transform,
            split_name=split_name,
            image_glob=image_glob or f"*chip*{image_file_type}",
            label_glob=label_glob or f"*label*{label_file_type}",
            image_suffix=image_suffix,
            label_suffix=label_suffix,
            label_npz_key=label_npz_key,
            binarize_label=binarize_label,
            scale_inputs=scale_inputs,
            normalization=normalization,
            nodata_policy=NoDataPolicy(
                ignore_in_loss=ignore_nodata_in_loss,
                ignore_index=nodata_ignore_index,
                image_fill_value=0.0,
            ),
        )

    def format_output(
        self,
        sample: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        """Return the tuple shape expected by the Toy semantic Lightning module."""
        return (
            sample["image"],
            sample["mask"],
            sample["image_path"],
            sample["label_path"],
        )


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
