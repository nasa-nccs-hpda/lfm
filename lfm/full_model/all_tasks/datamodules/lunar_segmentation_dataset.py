"""Base dataset for paired Lunar chip/label data."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset

from lfm.all_models.all_tasks.data.image_crop_resize import center_crop
from lfm.all_models.all_tasks.data.image_io import (
    find_pair_records,
    read_image_file_with_nodata_mask,
    read_label_file_with_metadata,
)
from lfm.all_models.all_tasks.data.normalization import (
    NormalizationStrategy,
    build_normalization_strategy,
)
from lfm.all_models.all_tasks.data.nodata import NoDataPolicy
from lfm.all_models.all_tasks.data.tensor_utils import (
    image_to_chw_float,
    mask_to_hw_long,
    shift_mask,
)


class LunarSegmentationDataset(Dataset):
    """Base paired chip/mask dataset with shared image preprocessing.

    Image normalization is optional and is applied as per-band z-score:
    ``(image - means) / stds``. For fine-tuning, pass statistics computed from
    the fine-tuning training split, after the same crop/no-data preprocessing.
    """

    def __init__(
        self,
        chips_dir: str | Path,
        labels_dir: str | Path,
        *,
        image_glob: str = "*.tif",
        label_glob: str = "*_label.*",
        image_suffix: str = "_input_wac_static_chip",
        label_suffix: str = "_label",
        crop_size: int | tuple[int, int] | None = 256,
        means: list[float] | None = None,
        stds: list[float] | None = None,
        normalization: NormalizationStrategy | None = None,
        binarize_mask: bool = True,
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        ignore_nodata_in_loss: bool = False,
        nodata_ignore_index: int = -1,
        nodata_policy: NoDataPolicy | None = None,
        mask_shift: tuple[int, int] | None = None,
        transform: Callable[[dict], dict] | None = None,
        split_name: str | None = None,
    ) -> None:
        self.split_name = split_name or Path(chips_dir).parent.name
        self.crop_size = crop_size
        self.means = means
        self.stds = stds
        self.normalization = normalization or build_normalization_strategy(
            normalize_inputs=means is not None and stds is not None,
            means=means,
            stds=stds,
        )
        self.binarize_mask = binarize_mask
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.ignore_nodata_in_loss = ignore_nodata_in_loss
        self.nodata_ignore_index = int(nodata_ignore_index)
        self.nodata_policy = nodata_policy or NoDataPolicy(
            ignore_in_loss=ignore_nodata_in_loss,
            ignore_index=nodata_ignore_index,
            image_fill_value=(
                float(no_data_replace) if no_data_replace is not None else 0.0
            ),
            label_fill_value=no_label_replace,
            fill_image_nodata=no_data_replace is not None,
        )
        self.mask_shift = mask_shift
        self.transform = transform
        self.records = find_pair_records(
            chips_dir=chips_dir,
            labels_dir=labels_dir,
            image_glob=image_glob,
            label_glob=label_glob,
            image_suffix=image_suffix,
            label_suffix=label_suffix,
            require_all_labels=True,
        )
        print(
            f"[{self.split_name}] Found {len(self.records)} matched image-label pairs "
            f"in {Path(chips_dir).parent}"
        )

    def __len__(self) -> int:
        return len(self.records)

    def _load_common(self, index: int) -> tuple[dict, object]:
        record = self.records[index]
        image_array, nodata_mask_array = read_image_file_with_nodata_mask(
            record.image_path
        )
        image = image_to_chw_float(image_array)
        nodata_mask = torch.as_tensor(nodata_mask_array, dtype=torch.bool)
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
        if self.binarize_mask:
            mask = (mask > 0).long()
        mask = self.nodata_policy.apply_to_mask_tensor(
            mask,
            nodata_mask,
            fill_label=False,
        )

        sample = {
            "image": image,
            "mask": mask,
            "filename": record.image_path.name,
        }
        return sample, label

    def _finalize_sample(
        self,
        sample: dict,
        *,
        boxes: torch.Tensor | None = None,
    ) -> tuple[dict, torch.Tensor | None]:
        if boxes is not None and self.mask_shift is not None and boxes.numel() > 0:
            shift_x, shift_y = int(self.mask_shift[0]), int(self.mask_shift[1])
            if shift_x != 0 or shift_y != 0:
                boxes = boxes.clone()
                boxes[:, 0] += float(shift_x)
                boxes[:, 1] += float(shift_y)

        if self.crop_size is not None:
            image, mask, boxes = center_crop(
                sample["image"],
                sample["mask"],
                self.crop_size,
                boxes=boxes,
                sample_name=sample["filename"],
            )
            sample["image"] = image
            sample["mask"] = mask

        sample["image"] = self.normalization.apply_tensor(sample["image"])
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, boxes
