"""Split datamodule for toy Mask2Former instance segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from lfm.full_model.datamodules.datamodule_utils import (
    center_crop,
    find_pair_records,
    image_to_chw_float,
    mask_to_hw_long,
    normalize_image,
    read_label_file,
    read_tif,
    shift_mask,
)
from lfm.toy_model.inst_seg.iseg_dataset import get_input_metadata


def _minmax_scale_per_band(image: torch.Tensor) -> torch.Tensor:
    flat = image.flatten(start_dim=1)
    band_min = flat.min(dim=1).values.view(-1, 1, 1)
    band_max = flat.max(dim=1).values.view(-1, 1, 1)
    denom = torch.clamp(band_max - band_min, min=1e-8)
    return (image - band_min) / denom


def _mask_to_binary_instance_targets(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask_labels: list[torch.Tensor] = []
    class_labels: list[torch.Tensor] = []
    for instance_id in torch.unique(mask).tolist():
        if int(instance_id) == 0:
            continue
        instance_mask = (mask == int(instance_id)).float()
        if instance_mask.any():
            mask_labels.append(instance_mask)
            class_labels.append(torch.tensor(1, dtype=torch.long))

    if not mask_labels:
        return (
            torch.zeros((0, *mask.shape[-2:]), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.long),
        )

    return torch.stack(mask_labels).float(), torch.stack(class_labels).long()


class ToyInstanceSegSplitDataset(Dataset):
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
        mask_shift: tuple[int, int] | None = None,
        max_samples: int | None = None,
        split_name: str | None = None,
    ) -> None:
        split_root = Path(split_root)
        self.split_name = split_name or split_root.name
        self.target_size = target_size
        self.band_filter = band_filter
        self.normalize_inputs = normalize_inputs
        self.means = means
        self.stds = stds
        self.mask_shift = mask_shift
        self.records = find_pair_records(
            chips_dir=split_root / "chips",
            labels_dir=split_root / "labels",
            image_glob=image_glob,
            label_glob=label_glob,
            image_suffix=image_suffix,
            label_suffix=label_suffix,
        )
        if max_samples is not None:
            self.records = self.records[:max_samples]
            print(f"[{self.split_name}] Limited to {len(self.records)} samples")
        print(f"[{self.split_name}] Found {len(self.records)} matched image-label pairs in {split_root}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = image_to_chw_float(read_tif(record.image_path))
        if self.band_filter is not None:
            image = image[self.band_filter]

        label = read_label_file(record.label_path)
        label_mask = label["mask"] if isinstance(label, dict) else label
        mask = mask_to_hw_long(label_mask)
        mask = shift_mask(mask, self.mask_shift)
        original_size = tuple(mask.shape[-2:])

        image = _minmax_scale_per_band(image)
        if self.normalize_inputs:
            image = normalize_image(image, self.means, self.stds)

        image, mask, _ = center_crop(
            image,
            mask,
            self.target_size,
            sample_name=record.image_path.name,
        )
        mask_labels, class_labels = _mask_to_binary_instance_targets(mask)
        return {
            "pixel_values": image.float(),
            "mask_labels": mask_labels,
            "class_labels": class_labels,
            "instance_mask": mask.long(),
            "filename": record.image_path.name,
            "original_size": original_size,
        }


def collate_toy_instance_segmentation(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "mask_labels": [item["mask_labels"] for item in batch],
        "class_labels": [item["class_labels"] for item in batch],
        "instance_mask": torch.stack([item["instance_mask"] for item in batch]),
        "filename": [item["filename"] for item in batch],
        "original_size": [item["original_size"] for item in batch],
    }


class ToyInstanceSegSplitDataModule(LightningDataModule):
    """Lightning datamodule for split toy instance segmentation data."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        batch_size: int = 2,
        num_workers: int = 10,
        target_size: int | tuple[int, int] = 256,
        band_filter: list[int] | None = None,
        normalize_inputs: bool = False,
        mask_shift: tuple[int, int] | None = None,
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
        max_test_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size
        self.band_filter = band_filter
        self.normalize_inputs = normalize_inputs
        self.mask_shift = mask_shift
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples
        self.means: list[float] | None = None
        self.stds: list[float] | None = None
        self.weight_assignments: list[str] | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.weight_assignments is None:
            self.weight_assignments = get_input_metadata(
                str(self.data_root / "train"),
                self.band_filter,
            )
        if self.normalize_inputs and (self.means is None or self.stds is None):
            self.means, self.stds = self._calculate_train_stats()

        if stage in (None, "fit"):
            self.train_dataset = self._make_dataset("train", self.max_train_samples)
            self.val_dataset = self._make_dataset("val", self.max_val_samples)
        if stage in (None, "test", "predict"):
            self.test_dataset = self._make_dataset("test", self.max_test_samples)

    def _make_dataset(self, split: str, max_samples: int | None) -> ToyInstanceSegSplitDataset:
        return ToyInstanceSegSplitDataset(
            self.data_root / split,
            target_size=self.target_size,
            band_filter=self.band_filter,
            normalize_inputs=self.normalize_inputs,
            means=self.means,
            stds=self.stds,
            mask_shift=self.mask_shift,
            max_samples=max_samples,
            split_name=split,
        )

    def _calculate_train_stats(self) -> tuple[list[float], list[float]]:
        stats_dataset = ToyInstanceSegSplitDataset(
            self.data_root / "train",
            target_size=self.target_size,
            band_filter=self.band_filter,
            normalize_inputs=False,
            mask_shift=self.mask_shift,
            max_samples=self.max_train_samples,
            split_name="train-stats",
        )
        n_pixels = 0
        sum_x = None
        sum_x2 = None
        for item in stats_dataset:
            x = item["pixel_values"]
            pixels = x.shape[-2] * x.shape[-1]
            x_sum = x.sum(dim=(1, 2))
            x2_sum = (x * x).sum(dim=(1, 2))
            sum_x = x_sum if sum_x is None else sum_x + x_sum
            sum_x2 = x2_sum if sum_x2 is None else sum_x2 + x2_sum
            n_pixels += pixels
        if sum_x is None or sum_x2 is None or n_pixels == 0:
            raise RuntimeError("No train pixels available for toy instance statistics.")
        means = (sum_x / n_pixels).tolist()
        stds = torch.sqrt(torch.clamp(sum_x2 / n_pixels - (sum_x / n_pixels) ** 2, min=1e-12)).tolist()
        print("[train] Toy instance z-score mean:", means)
        print("[train] Toy instance z-score std:", stds)
        return means, stds

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_toy_instance_segmentation,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_toy_instance_segmentation,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_toy_instance_segmentation,
        )
