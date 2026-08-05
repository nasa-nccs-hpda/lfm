"""Shared dataset base for lunar semantic and instance segmentation data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from lfm.all_models.all_tasks.data.image_io import (
    PairRecord,
    find_pair_records,
    image_to_hwc_float,
    read_image_file,
    read_image_file_with_nodata_mask,
    read_label_file,
)
from lfm.all_models.all_tasks.data.nodata import NoDataPolicy
from lfm.all_models.all_tasks.data.normalization import (
    NoNormalization,
    NormalizationStrategy,
)

LabelBinarizationMode = Literal["auto", "always", "never"]


def normalize_label_binarization_mode(
    value: bool | LabelBinarizationMode,
) -> LabelBinarizationMode:
    """Normalize legacy boolean binarization flags to an explicit mode."""
    if value is True:
        return "always"
    if value is False:
        return "never"
    if value not in {"auto", "always", "never"}:
        raise ValueError(
            "label binarization must be one of 'auto', 'always', or 'never', "
            f"got {value!r}."
        )
    return value


class LunarSegmentationDataset(Dataset):
    """Shared image-side dataset for lunar segmentation tasks.

    Subclasses can override ``format_target`` or ``format_output`` to produce
    task-specific targets while reusing file matching, image IO, band filtering,
    nodata handling, normalization, crop/resize, and tensor conversion.
    """

    def __init__(
        self,
        base_dir: str | Path,
        *,
        target_size: tuple[int, int] = (256, 256),
        max_samples: int | None = None,
        band_filter: list[int] | None = None,
        spatial_transform: str = "crop",
        split_name: str | None = None,
        image_glob: str = "*chip*.tif",
        label_glob: str = "*label.*",
        image_suffix: str | None = None,
        label_suffix: str | None = None,
        require_all_labels: bool = False,
        label_npz_key: str = "mask",
        binarize_label: bool | LabelBinarizationMode = "auto",
        scale_inputs: bool = True,
        normalization: NormalizationStrategy | None = None,
        nodata_policy: NoDataPolicy | None = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.image_dir = self.base_dir / "chips"
        self.label_dir = self.base_dir / "labels"
        self.target_size = tuple(target_size)
        self.spatial_transform = spatial_transform
        if self.spatial_transform not in {"resize", "crop"}:
            raise ValueError(
                f"spatial_transform must be 'resize' or 'crop', got {spatial_transform}"
            )
        self.band_filter = band_filter
        self.split_name = split_name or self.base_dir.name
        self.log_prefix = f"[{self.split_name}] "
        self.label_npz_key = label_npz_key
        self.label_binarization = normalize_label_binarization_mode(binarize_label)
        self.scale_inputs = scale_inputs
        self.normalization = normalization or NoNormalization()
        self.nodata_policy = nodata_policy or NoDataPolicy()

        records = find_pair_records(
            self.image_dir,
            self.label_dir,
            image_glob=image_glob,
            label_glob=label_glob,
            image_suffix=image_suffix,
            label_suffix=label_suffix,
            require_all_labels=require_all_labels,
        )
        if max_samples is not None and max_samples < len(records):
            records = records[:max_samples]
            print(f"{self.log_prefix}Limited to {max_samples} samples")
        self.records: list[PairRecord] = records
        self.valid_image_paths = [str(record.image_path) for record in self.records]
        self.valid_label_paths = [str(record.label_path) for record in self.records]

        example = read_image_file(self.records[0].image_path)
        example_hwc = image_to_hwc_float(example)
        self.num_channels = int(example_hwc.shape[2])
        if band_filter is None:
            self.band_filter = list(range(self.num_channels))
            print(
                f"{self.log_prefix}No band filter, using all {self.num_channels} bands."
            )
        else:
            if max(band_filter, default=-1) >= self.num_channels:
                raise ValueError(
                    f"Band filter {band_filter} is incompatible with "
                    f"{self.num_channels} input channel(s)."
                )
            self.band_filter = list(band_filter)
            print(f"{self.log_prefix}Filtered inputs to channels: {self.band_filter}")

        print(f"{self.log_prefix}Found {len(self.records)} matched image-label pairs")
        print(
            f"{self.log_prefix}Dataset configured for {len(self.band_filter)} channel(s)"
        )

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def min_max_scale_bands(
        image: np.ndarray,
        nodata_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        scaled = np.zeros_like(image, dtype=np.float32)
        for band_idx in range(image.shape[2]):
            band = image[:, :, band_idx]
            valid = np.isfinite(band)
            if nodata_mask is not None:
                valid = valid & ~nodata_mask
            if not np.any(valid):
                continue
            valid_band = band[valid]
            band_min, band_max = valid_band.min(), valid_band.max()
            if band_max > band_min:
                scaled[:, :, band_idx] = (band - band_min) / (band_max - band_min)
            else:
                scaled[:, :, band_idx] = band
            if nodata_mask is not None and np.any(nodata_mask):
                scaled[:, :, band_idx][nodata_mask] = 0.0
        return scaled

    @staticmethod
    def center_crop_arrays(
        image: np.ndarray,
        label: np.ndarray,
        target_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        target_h, target_w = target_size
        image_h, image_w = image.shape[:2]
        label_h, label_w = label.shape[:2]
        if (image_h, image_w) != (label_h, label_w):
            raise ValueError(
                f"Image and label spatial shapes differ: "
                f"image={(image_h, image_w)}, label={(label_h, label_w)}"
            )
        if image_h < target_h or image_w < target_w:
            raise ValueError(
                f"Cannot crop image of shape {(image_h, image_w)} to {target_size}"
            )
        top = (image_h - target_h) // 2
        left = (image_w - target_w) // 2
        return (
            image[top : top + target_h, left : left + target_w, :],
            label[top : top + target_h, left : left + target_w],
        )

    def load_label(self, label_path: Path) -> np.ndarray:
        label = read_label_file(label_path, npz_key=self.label_npz_key).astype(np.int64)
        if self._should_binarize_label(label_path):
            label = (label > 0).astype(np.int64)
        return label

    def _should_binarize_label(self, label_path: Path) -> bool:
        if self.label_binarization == "always":
            return True
        if self.label_binarization == "never":
            return False
        return label_path.suffix.lower() != ".npy"

    def prepare_sample(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        image, nodata_mask = read_image_file_with_nodata_mask(record.image_path)
        image = image_to_hwc_float(image)
        label = self.load_label(record.label_path)

        if image.shape[2] != self.num_channels:
            raise ValueError(
                f"Channel mismatch: expected {self.num_channels}, got {image.shape[2]} "
                f"for {record.image_path}"
            )
        image = image[:, :, self.band_filter].astype(np.float32)

        scale_nodata_mask = nodata_mask if self.nodata_policy.ignore_in_loss else None
        if self.scale_inputs:
            image = self.min_max_scale_bands(image, scale_nodata_mask)
        else:
            image = self.nodata_policy.apply_to_image(image, nodata_mask)

        label = self.nodata_policy.apply_to_label(label, nodata_mask)
        image = self.normalization.apply(image, band_filter=self.band_filter)

        if self.spatial_transform == "crop":
            image, label = self.center_crop_arrays(image, label, self.target_size)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        label_tensor = torch.from_numpy(label)

        if self.spatial_transform == "resize":
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=self.target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            label_tensor = (
                F.interpolate(
                    label_tensor.unsqueeze(0).unsqueeze(0).float(),
                    size=self.target_size,
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
                .long()
            )

        return {
            "image": image_tensor,
            "mask": label_tensor,
            "image_path": str(record.image_path),
            "label_path": str(record.label_path),
            "filename": record.image_path.name,
        }

    def format_output(self, sample: dict[str, Any]) -> Any:
        return sample

    def __getitem__(self, idx: int) -> Any:
        return self.format_output(self.prepare_sample(idx))
