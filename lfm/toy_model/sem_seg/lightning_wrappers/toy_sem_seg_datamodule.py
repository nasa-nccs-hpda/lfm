"""Split-folder datamodule for the toy semantic segmentation baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from lfm.toy_model.sem_seg.sseg_dataset import (
    LunarCraterDataset,
    get_input_metadata,
)


class ToySemSegSplitDataModule(LightningDataModule):
    """Use the old toy semantic-seg dataset with explicit train/val/test splits.

    This wrapper intentionally preserves the old model data behavior for the
    comparison baseline: .npy labels and per-sample band min-max scaling inside
    ``LunarCraterDataset``. If ``normalize_inputs=True``, per-band z-score
    statistics are computed from the training split after the same crop/min-max
    preprocessing used by the model.
    """

    def __init__(
        self,
        data_root: str | Path,
        *,
        batch_size: int = 16,
        num_workers: int = 8,
        target_size: tuple[int, int] = (256, 256),
        spatial_transform: str = "crop",
        band_filter: list[int] | None = None,
        normalize_inputs: bool = False,
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
        max_test_samples: int | None = None,
        output_dir: str | Path | None = None,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size
        self.spatial_transform = spatial_transform
        self.band_filter = band_filter
        self.normalize_inputs = normalize_inputs
        self.max_samples = {
            "train": max_train_samples,
            "val": max_val_samples,
            "test": max_test_samples,
        }
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.pin_memory = pin_memory

        self.weight_assignments: list[str] | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None) -> None:
        self._validate_split_dirs()
        if self.weight_assignments is None:
            self.weight_assignments = get_input_metadata(
                str(self.data_root / "train"),
                self.band_filter,
            )
        if self.normalize_inputs and (self.mean is None or self.std is None):
            self._calculate_train_stats()

        if stage in (None, "fit"):
            self.train_dataset = self._make_dataset("train")
            self.val_dataset = self._make_dataset("val")
            self._write_file_lists(["train", "val"])
            self._write_sanity_report()

        if stage in (None, "validate"):
            self.val_dataset = self._make_dataset("val")
            self._write_file_lists(["val"])

        if stage in (None, "test", "predict"):
            self.test_dataset = self._make_dataset("test")
            self._write_file_lists(["test"])

    def _validate_split_dirs(self) -> None:
        required = []
        for split in ["train", "val", "test"]:
            required.extend(
                [
                    self.data_root / split / "chips",
                    self.data_root / split / "labels",
                ]
            )
        missing = [path for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing split data directories:\n" + "\n".join(str(path) for path in missing)
            )

    def _make_dataset(self, split: str) -> LunarCraterDataset:
        return LunarCraterDataset(
            base_dir=str(self.data_root / split),
            mean=self.mean,
            std=self.std,
            target_size=self.target_size,
            spatial_transform=self.spatial_transform,
            max_samples=self.max_samples[split],
            band_filter=self.band_filter,
            normalize_inputs=self.normalize_inputs,
            split_name=split,
        )

    def _calculate_train_stats(self) -> None:
        stats_dataset = LunarCraterDataset(
            base_dir=str(self.data_root / "train"),
            mean=None,
            std=None,
            target_size=self.target_size,
            spatial_transform=self.spatial_transform,
            max_samples=self.max_samples["train"],
            band_filter=self.band_filter,
            normalize_inputs=False,
            split_name="train-stats",
        )
        pixel_sum = None
        pixel_sq_sum = None
        pixel_count = 0

        for index in range(len(stats_dataset)):
            image, _, _, _ = stats_dataset[index]
            image = image.to(torch.float64)
            if pixel_sum is None:
                pixel_sum = torch.zeros(image.shape[0], dtype=torch.float64)
                pixel_sq_sum = torch.zeros(image.shape[0], dtype=torch.float64)
            pixel_sum += image.sum(dim=(1, 2))
            pixel_sq_sum += (image**2).sum(dim=(1, 2))
            pixel_count += image.shape[1] * image.shape[2]

        if pixel_sum is None or pixel_sq_sum is None or pixel_count == 0:
            raise ValueError("Could not compute toy train statistics from an empty dataset.")

        mean = pixel_sum / pixel_count
        variance = (pixel_sq_sum / pixel_count) - (mean**2)
        std = torch.sqrt(torch.clamp(variance, min=1e-12))
        self.mean = mean.numpy().astype(np.float32)
        self.std = std.numpy().astype(np.float32)

        print("[train] Toy z-score mean:", self.mean.tolist())
        print("[train] Toy z-score std:", self.std.tolist())

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup("validate")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            self.setup("test")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def _write_file_lists(self, splits: list[str]) -> None:
        if self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for split in splits:
            dataset = getattr(self, f"{split}_dataset", None)
            if dataset is None:
                continue
            path = self.output_dir / f"{split}_files.txt"
            with path.open("w", encoding="utf-8") as f:
                for image_path, label_path in zip(
                    dataset.valid_image_paths,
                    dataset.valid_label_paths,
                ):
                    f.write(f"{image_path}\t{label_path}\n")

    def _write_sanity_report(self) -> None:
        if self.output_dir is None or self.train_dataset is None or self.val_dataset is None:
            return

        report = self.get_sanity_summary()
        path = self.output_dir / "data_sanity_summary.txt"
        with path.open("w", encoding="utf-8") as f:
            for key, value in report.items():
                f.write(f"{key}: {value}\n")

    def get_sanity_summary(self) -> dict[str, Any]:
        if self.train_dataset is None:
            self.setup("fit")
        image, mask, image_path, label_path = self.train_dataset[0]
        foreground_fraction = float((mask > 0).float().mean().item())
        return {
            "data_root": str(self.data_root),
            "target_size": self.target_size,
            "spatial_transform": self.spatial_transform,
            "band_filter": self.band_filter,
            "normalize_inputs": self.normalize_inputs,
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
            "weight_assignments": self.weight_assignments,
            "train_samples": len(self.train_dataset),
            "val_samples": len(self.val_dataset) if self.val_dataset is not None else None,
            "test_samples": len(self.test_dataset) if self.test_dataset is not None else None,
            "sample_image_shape": tuple(image.shape),
            "sample_mask_shape": tuple(mask.shape),
            "sample_mask_values": torch.unique(mask).tolist(),
            "sample_foreground_fraction": foreground_fraction,
            "sample_image_path": image_path,
            "sample_label_path": label_path,
        }
