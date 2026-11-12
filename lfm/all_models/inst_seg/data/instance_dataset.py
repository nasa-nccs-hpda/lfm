"""Shared instance segmentation dataset boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from lfm.all_models.all_tasks.data import (
    LunarSegmentationDataset,
    NoDataPolicy,
    build_nodata_policy,
)
from lfm.all_models.all_tasks.data.image_crop_resize import center_crop
from lfm.all_models.all_tasks.data.image_io import (
    PairRecord,
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
from lfm.all_models.inst_seg.data.eligibility import SkippedInstanceChip
from lfm.all_models.inst_seg.data.instance_data_utils import (
    boxes_to_tensor,
    instance_mask_to_object_detection_targets,
    mask_to_binary_instance_targets,
)


def minmax_scale_per_band(
    image: torch.Tensor,
    nodata_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    scaled = torch.zeros_like(image)
    for band_idx in range(image.shape[0]):
        band = image[band_idx]
        valid = torch.isfinite(band)
        if nodata_mask is not None:
            if nodata_mask.ndim == 2:
                band_nodata_mask = nodata_mask
            elif nodata_mask.ndim == 3:
                band_nodata_mask = nodata_mask[band_idx]
            else:
                raise ValueError(
                    f"Expected 2D or 3D nodata mask, got {tuple(nodata_mask.shape)}"
                )
            valid = valid & ~band_nodata_mask
        if not torch.any(valid):
            continue
        valid_band = band[valid]
        band_min = valid_band.min()
        band_max = valid_band.max()
        if band_max > band_min:
            scaled[band_idx][valid] = (band[valid] - band_min) / torch.clamp(
                band_max - band_min,
                min=1e-8,
            )
        else:
            # Preserve the existing constant-valid-band behavior, excluding NoData.
            scaled[band_idx][valid] = band[valid]
    return scaled


class LunarInstanceMaskDataset(LunarSegmentationDataset):
    """Dataset for split lunar instance-id masks with shared preprocessing."""

    def __init__(
        self,
        split_root: str | Path,
        *,
        target_size: int | tuple[int, int] = 256,
        image_glob: str = "*chip*.tif",
        label_glob: str = "*label*.npz",
        image_suffix: str | None = None,
        label_suffix: str | None = None,
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
        excluded_nodata_values: list[float] | tuple[float, ...] | None = None,
        image_nodata_policy: str = "union",
        nodata_policy: NoDataPolicy | None = None,
        max_samples: int | None = None,
        split_name: str | None = None,
        records: list[PairRecord] | None = None,
        include_raw_image: bool = False,
        skip_invalid_chips: bool = False,
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
        nodata_strategy = build_nodata_policy(
            no_data_replace=no_data_replace,
            no_label_replace=no_label_replace,
            ignore_nodata_in_loss=ignore_nodata_in_loss,
            nodata_ignore_index=nodata_ignore_index,
            excluded_nodata_values=excluded_nodata_values,
            image_nodata_policy=image_nodata_policy,
            nodata_policy=nodata_policy,
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
            binarize_label="never",
            scale_inputs=scale_inputs,
            normalization=normalization_strategy,
            nodata_policy=nodata_strategy,
            records=records,
        )
        self.include_raw_image = include_raw_image
        self.skip_invalid_chips = skip_invalid_chips
        if skip_invalid_chips and not nodata_strategy.ignore_in_loss:
            raise ValueError(
                "Skipping invalid chips requires ignore_nodata_in_loss=True."
            )
        self.normalize_inputs = normalize_inputs
        self.means = means
        self.stds = stds
        self.mask_shift = mask_shift
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.ignore_nodata_in_loss = ignore_nodata_in_loss
        self.nodata_ignore_index = int(nodata_ignore_index)
        self.excluded_nodata_values = tuple(
            float(value) for value in excluded_nodata_values or ()
        )
        self.nodata_policy = nodata_strategy

    def _load_instance_sample(
        self,
        index: int,
    ) -> tuple[dict[str, Any], torch.Tensor | None, int | None] | SkippedInstanceChip:
        record = self.records[index]
        image_array, nodata_mask_array = read_image_file_with_nodata_mask(
            record.image_path,
            excluded_values=self.nodata_policy.excluded_values,
            per_band=True,
        )
        image = image_to_chw_float(image_array)
        nodata_mask = torch.as_tensor(nodata_mask_array, dtype=torch.bool)
        if self.band_filter is not None:
            image = image[self.band_filter]
            if nodata_mask.ndim == 3:
                nodata_mask = nodata_mask[self.band_filter]
        label_nodata_mask = (
            nodata_mask.any(dim=0) if nodata_mask.ndim == 3 else nodata_mask
        )

        label = read_label_file_with_metadata(record.label_path)
        label_mask = label["mask"] if isinstance(label, dict) else label
        mask = mask_to_hw_long(label_mask)
        if tuple(image.shape[-2:]) != tuple(mask.shape):
            raise ValueError(
                f"Image/label geometry differs for {record.image_path.name}."
            )
        crater_boxes = None
        num_craters = None
        if isinstance(label, dict):
            crater_boxes = boxes_to_tensor(label.get("bboxes"))
            raw_num_craters = label.get("num_craters")
            if raw_num_craters is not None:
                num_craters = int(torch.as_tensor(raw_num_craters).item())

        raw_image = image.clone() if self.include_raw_image else None
        mask = self.nodata_policy.apply_to_mask_tensor(
            mask,
            label_nodata_mask,
            ignore_nodata=False,
        )
        mask = shift_mask(mask, self.mask_shift)
        if crater_boxes is not None and self.mask_shift is not None:
            shift_x, shift_y = int(self.mask_shift[0]), int(self.mask_shift[1])
            if shift_x != 0 or shift_y != 0:
                crater_boxes = crater_boxes.clone()
                crater_boxes[:, 0] += float(shift_x)
                crater_boxes[:, 1] += float(shift_y)
        mask = self.nodata_policy.apply_to_mask_tensor(
            mask,
            label_nodata_mask,
            fill_label=False,
        )
        original_size = tuple(mask.shape[-2:])
        image_valid_mask, _, _ = center_crop(
            ~nodata_mask, mask, self.target_size, sample_name=record.image_path.name
        )
        _, cropped_mask, _ = center_crop(
            image, mask, self.target_size, sample_name=record.image_path.name
        )
        valid_mask = image_valid_mask.all(dim=0) & (
            cropped_mask != self.nodata_policy.ignore_index
        )
        if self.skip_invalid_chips and not torch.any(valid_mask):
            return SkippedInstanceChip(
                chip_id=record.chip_id
                or f"{self.base_dir.name}/{record.image_path.name}",
                image_path=str(record.image_path),
                label_path=str(record.label_path),
            )
        if raw_image is not None:
            # Keep the canonical crop before either model's input transformation.
            raw_image, _, _ = center_crop(
                raw_image, mask, self.target_size, sample_name=record.image_path.name
            )

        if self.scale_inputs:
            # Compute min/max from raw valid values, before filling missing pixels.
            image = minmax_scale_per_band(image, nodata_mask)
        else:
            fill_mask = (
                label_nodata_mask
                if self.nodata_policy.image_nodata_policy == "union"
                else nodata_mask
            )
            image = self.nodata_policy.apply_to_image_tensor(image, fill_mask)
        image = self.normalization.apply_tensor(image)
        if self.nodata_policy.ignore_in_loss:
            fill_mask = (
                label_nodata_mask
                if self.nodata_policy.image_nodata_policy == "union"
                else nodata_mask
            )
            image = image.masked_fill(fill_mask, 0.0)

        image, mask, crater_boxes = center_crop(
            image,
            mask,
            self.target_size,
            boxes=crater_boxes,
            sample_name=record.image_path.name,
        )
        sample = {
            "image": image.float(),
            "mask": mask.long(),
            "filename": record.image_path.name,
            "original_size": original_size,
            "chip_id": record.chip_id
            or f"{self.base_dir.name}/{record.image_path.name}",
        }
        if self.nodata_policy.ignore_in_loss:
            sample["valid_mask"] = valid_mask
            sample["image_valid_mask"] = image_valid_mask
            # Stored boxes/counts can include portions outside the common footprint.
            crater_boxes, _, _ = instance_mask_to_object_detection_targets(mask)
            num_craters = int(crater_boxes.shape[0])
        if raw_image is not None:
            sample["raw_image"] = raw_image
        if crater_boxes is not None:
            sample["crater_boxes"] = crater_boxes
            num_craters = int(crater_boxes.shape[0])
        return sample, crater_boxes, num_craters

    def __getitem__(self, index: int) -> dict[str, Any] | SkippedInstanceChip:
        loaded = self._load_instance_sample(index)
        if isinstance(loaded, SkippedInstanceChip):
            return loaded
        sample, _, num_craters = loaded
        if num_craters is not None:
            sample["num_craters"] = torch.tensor(num_craters, dtype=torch.long)
        return sample


class InstanceSegmentationDataset(LunarInstanceMaskDataset):
    """Dataset for split lunar instance labels in Mask2Former format."""

    def __getitem__(self, index: int) -> dict[str, Any] | SkippedInstanceChip:
        loaded = self._load_instance_sample(index)
        if isinstance(loaded, SkippedInstanceChip):
            return loaded
        sample, _, _ = loaded
        mask_labels, class_labels = mask_to_binary_instance_targets(sample["mask"])
        item = {
            "pixel_values": sample["image"].float(),
            "mask_labels": mask_labels,
            "class_labels": class_labels,
            "instance_mask": sample["mask"].long(),
            "filename": sample["filename"],
            "original_size": sample["original_size"],
            "chip_id": sample["chip_id"],
        }
        if "raw_image" in sample:
            item["raw_image"] = sample["raw_image"]
        for name in ("valid_mask", "image_valid_mask"):
            if name in sample:
                item[name] = sample[name]
        return item


class ObjectDetectionInstanceSegmentationDataset(LunarInstanceMaskDataset):
    """Dataset emitting object-detection targets from instance-id masks."""

    def __init__(
        self,
        *args,
        target_box_format: str = "xyxy",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if target_box_format not in {"xyxy", "cxcywh"}:
            raise ValueError(f"Unsupported target_box_format: {target_box_format}")
        self.target_box_format = target_box_format

    def __getitem__(self, index: int) -> dict[str, Any] | SkippedInstanceChip:
        loaded = self._load_instance_sample(index)
        if isinstance(loaded, SkippedInstanceChip):
            return loaded
        sample, _, _ = loaded
        boxes, labels, masks = instance_mask_to_object_detection_targets(
            sample["mask"],
            box_format=self.target_box_format,
        )
        if self.nodata_policy.ignore_in_loss:
            sample["valid_mask"] = sample["mask"] != int(
                self.nodata_policy.ignore_index
            )
        sample.pop("original_size", None)
        sample["boxes"] = boxes
        sample["labels"] = labels
        sample["masks"] = masks
        sample["num_craters"] = torch.tensor(labels.shape[0], dtype=torch.long)
        return sample
