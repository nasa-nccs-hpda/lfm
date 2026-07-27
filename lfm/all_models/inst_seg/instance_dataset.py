"""Shared instance segmentation dataset boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from lfm.all_models.all_tasks.data import LunarSegmentationDataset, NoDataPolicy
from lfm.all_models.all_tasks.data.image_crop_resize import center_crop
from lfm.all_models.all_tasks.data.image_io import (
    read_image_file_with_nodata_mask,
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


def minmax_scale_per_band(
    image: torch.Tensor,
    nodata_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    flat = image.flatten(start_dim=1)
    if nodata_mask is not None and torch.any(nodata_mask):
        valid = ~nodata_mask.flatten()
        if not torch.any(valid):
            return torch.zeros_like(image)
        flat_valid = flat[:, valid]
        band_min = flat_valid.min(dim=1).values.view(-1, 1, 1)
        band_max = flat_valid.max(dim=1).values.view(-1, 1, 1)
    else:
        band_min = flat.min(dim=1).values.view(-1, 1, 1)
        band_max = flat.max(dim=1).values.view(-1, 1, 1)
    denom = torch.clamp(band_max - band_min, min=1e-8)
    scaled = (image - band_min) / denom
    if nodata_mask is not None and torch.any(nodata_mask):
        scaled = scaled.clone()
        scaled[:, nodata_mask] = 0.0
    return scaled


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
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        ignore_nodata_in_loss: bool = False,
        nodata_ignore_index: int = -1,
        nodata_policy: NoDataPolicy | None = None,
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
        nodata_strategy = nodata_policy or NoDataPolicy(
            ignore_in_loss=ignore_nodata_in_loss,
            ignore_index=nodata_ignore_index,
            image_fill_value=(
                float(no_data_replace) if no_data_replace is not None else 0.0
            ),
            label_fill_value=no_label_replace,
            fill_image_nodata=no_data_replace is not None,
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
            nodata_policy=nodata_strategy,
        )
        self.normalize_inputs = normalize_inputs
        self.means = means
        self.stds = stds
        self.mask_shift = mask_shift
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.ignore_nodata_in_loss = ignore_nodata_in_loss
        self.nodata_ignore_index = int(nodata_ignore_index)
        self.nodata_policy = nodata_strategy

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_array, nodata_mask_array = read_image_file_with_nodata_mask(
            record.image_path
        )
        image = image_to_chw_float(image_array)
        nodata_mask = torch.as_tensor(nodata_mask_array, dtype=torch.bool)
        if self.band_filter is not None:
            image = image[self.band_filter]

        label = read_label_file_with_metadata(record.label_path)
        label_mask = label["mask"] if isinstance(label, dict) else label
        mask = mask_to_hw_long(label_mask)
        image = self.nodata_policy.apply_to_image_tensor(image, nodata_mask)
        mask = self.nodata_policy.apply_to_mask_tensor(
            mask,
            nodata_mask,
            ignore_nodata=False,
        )
        mask = shift_mask(mask, self.mask_shift)
        mask = self.nodata_policy.apply_to_mask_tensor(
            mask,
            nodata_mask,
            fill_label=False,
        )
        original_size = tuple(mask.shape[-2:])

        if self.scale_inputs:
            scale_nodata_mask = (
                nodata_mask if self.nodata_policy.ignore_in_loss else None
            )
            image = minmax_scale_per_band(image, scale_nodata_mask)
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
