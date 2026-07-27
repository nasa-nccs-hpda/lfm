"""Shared instance segmentation dataset boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from lfm.all_models.all_tasks.data import LunarSegmentationDataset
from lfm.all_models.all_tasks.data.image_crop_resize import center_crop
from lfm.all_models.all_tasks.data.image_io import (
    read_image_file,
    read_label_file_with_metadata,
)
from lfm.all_models.all_tasks.data.normalization import (
    NormalizationStrategy,
    build_normalization_strategy,
)
from lfm.all_models.all_tasks.data.tensor_utils import (
    image_to_chw_float,
    mask_to_hw_long,
    shift_mask,
)
from lfm.all_models.inst_seg.instance_data_utils import mask_to_binary_instance_targets


def minmax_scale_per_band(image: torch.Tensor) -> torch.Tensor:
    flat = image.flatten(start_dim=1)
    band_min = flat.min(dim=1).values.view(-1, 1, 1)
    band_max = flat.max(dim=1).values.view(-1, 1, 1)
    denom = torch.clamp(band_max - band_min, min=1e-8)
    return (image - band_min) / denom


class InstanceSegmentationDataset(LunarSegmentationDataset):
    """Dataset for split lunar instance labels in Mask2Former format."""

    def __init__(
        self,
        split_root: str | Path,
        *,
        target_size: int | tuple[int, int] = 256,
        image_glob: str = "*.tif",
        label_glob: str = "*_label.npz",
        image_suffix: str = "_input_wac_static_chip",
        label_suffix: str = "_label",
        band_filter: list[int] | None = None,
        normalize_inputs: bool = False,
        means: list[float] | None = None,
        stds: list[float] | None = None,
        normalization: NormalizationStrategy | None = None,
        scale_inputs: bool = True,
        mask_shift: tuple[int, int] | None = None,
        max_samples: int | None = None,
        split_name: str | None = None,
    ) -> None:
        target_shape = (
            (int(target_size), int(target_size))
            if isinstance(target_size, int)
            else (int(target_size[0]), int(target_size[1]))
        )
        normalization_strategy = normalization or build_normalization_strategy(
            normalize_inputs=normalize_inputs
            and means is not None
            and stds is not None,
            means=means,
            stds=stds,
        )
        super().__init__(
            split_root,
            target_size=target_shape,
            max_samples=max_samples,
            band_filter=band_filter,
            spatial_transform="crop",
            split_name=split_name,
            image_glob=image_glob,
            label_glob=label_glob,
            image_suffix=image_suffix,
            label_suffix=label_suffix,
            require_all_labels=True,
            label_npz_key="mask",
            binarize_label=False,
            scale_inputs=scale_inputs,
            normalization=normalization_strategy,
        )
        self.normalize_inputs = normalize_inputs
        self.means = means
        self.stds = stds
        self.mask_shift = mask_shift

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = image_to_chw_float(read_image_file(record.image_path))
        if self.band_filter is not None:
            image = image[self.band_filter]

        label = read_label_file_with_metadata(record.label_path)
        label_mask = label["mask"] if isinstance(label, dict) else label
        mask = mask_to_hw_long(label_mask)
        mask = shift_mask(mask, self.mask_shift)
        original_size = tuple(mask.shape[-2:])

        if self.scale_inputs:
            image = minmax_scale_per_band(image)
        image = self.normalization.apply_tensor(image)

        image, mask, _ = center_crop(
            image,
            mask,
            self.target_size,
            sample_name=record.image_path.name,
        )
        mask_labels, class_labels = mask_to_binary_instance_targets(mask)
        return {
            "pixel_values": image.float(),
            "mask_labels": mask_labels,
            "class_labels": class_labels,
            "instance_mask": mask.long(),
            "filename": record.image_path.name,
            "original_size": original_size,
        }
