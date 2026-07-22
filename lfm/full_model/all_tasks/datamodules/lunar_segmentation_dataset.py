"""Base dataset for paired Lunar chip/label data."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset

from .datamodule_utils import (
    center_crop,
    find_pair_records,
    image_to_chw_float,
    mask_to_hw_long,
    normalize_image,
    read_label_file,
    read_tif,
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
        binarize_mask: bool = True,
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        mask_shift: tuple[int, int] | None = None,
        transform: Callable[[dict], dict] | None = None,
        split_name: str | None = None,
    ) -> None:
        self.split_name = split_name or Path(chips_dir).parent.name
        self.crop_size = crop_size
        self.means = means
        self.stds = stds
        self.binarize_mask = binarize_mask
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.mask_shift = mask_shift
        self.transform = transform
        self.records = find_pair_records(
            chips_dir=chips_dir,
            labels_dir=labels_dir,
            image_glob=image_glob,
            label_glob=label_glob,
            image_suffix=image_suffix,
            label_suffix=label_suffix,
        )
        print(
            f"[{self.split_name}] Found {len(self.records)} matched image-label pairs "
            f"in {Path(chips_dir).parent}"
        )

    def __len__(self) -> int:
        return len(self.records)

    def _load_common(self, index: int) -> tuple[dict, object]:
        record = self.records[index]
        image = image_to_chw_float(read_tif(record.image_path))
        label = read_label_file(record.label_path)
        label_mask = label["mask"] if isinstance(label, dict) else label
        mask = mask_to_hw_long(label_mask)

        if self.no_data_replace is not None:
            image = torch.nan_to_num(image, nan=float(self.no_data_replace))
        if self.no_label_replace is not None:
            mask = torch.nan_to_num(
                mask.float(), nan=float(self.no_label_replace)
            ).long()
        mask = shift_mask(mask, self.mask_shift)
        if self.binarize_mask:
            mask = (mask > 0).long()

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

        sample["image"] = normalize_image(sample["image"], self.means, self.stds)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, boxes
