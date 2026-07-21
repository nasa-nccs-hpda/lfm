"""Split datamodule for Toy DINO Mask R-CNN instance segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from lfm.full_model.datamodules.datamodule_utils import (
    collate_object_detection_instance_segmentation,
    instance_mask_to_object_detection_targets,
)
from lfm.toy_model.inst_seg.iseg_dataset import get_input_metadata
from lfm.toy_model.inst_seg.lightning_wrappers.toy_instance_seg_datamodule import (
    ToyInstanceSegSplitDataset,
)


class ToyDinoMaskRCNNSplitDataset(ToyInstanceSegSplitDataset):
    """Toy split dataset emitting TorchVision Mask R-CNN targets."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        boxes, labels, masks = instance_mask_to_object_detection_targets(
            item["instance_mask"],
            box_format="xyxy",
        )
        return {
            "image": item["pixel_values"].float(),
            "mask": item["instance_mask"].long(),
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "filename": item["filename"],
            "num_craters": torch.tensor(labels.shape[0], dtype=torch.long),
        }


class ToyDinoMaskRCNNSplitDataModule(LightningDataModule):
    """Lightning datamodule for Toy DINO Mask R-CNN experiments."""

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

    def _make_dataset(self, split: str, max_samples: int | None) -> ToyDinoMaskRCNNSplitDataset:
        return ToyDinoMaskRCNNSplitDataset(
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
        stats_dataset = ToyDinoMaskRCNNSplitDataset(
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
            x = item["image"]
            pixels = x.shape[-2] * x.shape[-1]
            x_sum = x.sum(dim=(1, 2))
            x2_sum = (x * x).sum(dim=(1, 2))
            sum_x = x_sum if sum_x is None else sum_x + x_sum
            sum_x2 = x2_sum if sum_x2 is None else sum_x2 + x2_sum
            n_pixels += pixels
        if sum_x is None or sum_x2 is None or n_pixels == 0:
            raise RuntimeError("No train pixels available for Toy DINO Mask R-CNN statistics.")
        means = (sum_x / n_pixels).tolist()
        stds = torch.sqrt(torch.clamp(sum_x2 / n_pixels - (sum_x / n_pixels) ** 2, min=1e-12)).tolist()
        print("[train] Toy DINO Mask R-CNN z-score mean:", means)
        print("[train] Toy DINO Mask R-CNN z-score std:", stds)
        return means, stds

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_object_detection_instance_segmentation,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_object_detection_instance_segmentation,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_object_detection_instance_segmentation,
        )
