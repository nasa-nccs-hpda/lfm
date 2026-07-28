"""Shared datamodule layout helpers for split lunar datasets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class SplitFolderDataLayout:
    """Resolve standard ``split/chips`` and ``split/labels`` data directories."""

    data_root: Path
    chips_subdir: str = "chips"
    labels_subdir: str = "labels"

    def split_chips_dir(self, split: str) -> Path:
        return self.data_root / split / self.chips_subdir

    def split_labels_dir(self, split: str) -> Path:
        return self.data_root / split / self.labels_subdir

    def flat_chips_dir(self) -> Path:
        return self.data_root / self.chips_subdir

    def flat_labels_dir(self) -> Path:
        return self.data_root / self.labels_subdir

    def split_dirs(self, split: str) -> tuple[Path, Path]:
        return self.split_chips_dir(split), self.split_labels_dir(split)

    def flat_dirs(self) -> tuple[Path, Path]:
        return self.flat_chips_dir(), self.flat_labels_dir()

    def has_split(self, split: str) -> bool:
        chips_dir, labels_dir = self.split_dirs(split)
        return chips_dir.exists() and labels_dir.exists()

    def missing_split_dirs(self, splits: list[str]) -> list[Path]:
        missing: list[Path] = []
        for split in splits:
            chips_dir, labels_dir = self.split_dirs(split)
            if not chips_dir.exists():
                missing.append(chips_dir)
            if not labels_dir.exists():
                missing.append(labels_dir)
        return missing

    def require_split_dirs(self, splits: list[str]) -> None:
        missing = self.missing_split_dirs(splits)
        if missing:
            raise FileNotFoundError(
                "Missing split data directories:\n"
                + "\n".join(str(path) for path in missing)
            )


InputMetadataFn = Callable[[str, list[int] | None], list[str]]


class SplitSegmentationDataModule(LightningDataModule):
    """Shared LightningDataModule lifecycle for pre-split segmentation data.

    Subclasses provide task-specific dataset construction and, when needed,
    train-stat normalization hooks. This base owns the standard
    ``train/val/test/{chips,labels}`` lifecycle and dataloader construction.
    """

    collate_fn = None
    stats_image_key: str | None = None
    stats_log_label = "segmentation"

    def __init__(
        self,
        data_root: str | Path,
        *,
        batch_size: int,
        num_workers: int,
        chips_subdir: str = "chips",
        labels_subdir: str = "labels",
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
        max_test_samples: int | None = None,
        band_filter: list[int] | None = None,
        pin_memory: bool = True,
        input_metadata_fn: InputMetadataFn | None = None,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.chips_subdir = chips_subdir
        self.labels_subdir = labels_subdir
        self.data_layout = SplitFolderDataLayout(
            self.data_root,
            chips_subdir=chips_subdir,
            labels_subdir=labels_subdir,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.band_filter = band_filter
        self.pin_memory = pin_memory
        self.input_metadata_fn = input_metadata_fn
        self.weight_assignments: list[str] | None = None
        self.max_samples_by_split = {
            "train": max_train_samples,
            "val": max_val_samples,
            "test": max_test_samples,
        }
        self.max_samples = self.max_samples_by_split
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        self._validate_split_dirs()
        self._setup_input_metadata()
        if self._needs_train_stats():
            self._calculate_train_stats()

        if stage in (None, "fit"):
            self.train_dataset = self._make_dataset(
                "train",
                self.max_samples_by_split["train"],
            )
            self.val_dataset = self._make_dataset(
                "val",
                self.max_samples_by_split["val"],
            )
            self._after_fit_setup()

        if stage in (None, "validate"):
            self.val_dataset = self._make_dataset(
                "val",
                self.max_samples_by_split["val"],
            )
            self._after_validate_setup()

        if stage in (None, "test", "predict"):
            self.test_dataset = self._make_dataset(
                "test",
                self.max_samples_by_split["test"],
            )
            self._after_test_setup()

    def _validate_split_dirs(self) -> None:
        self.data_layout.require_split_dirs(["train", "val", "test"])

    def _setup_input_metadata(self) -> None:
        if self.weight_assignments is None and self.input_metadata_fn is not None:
            self.weight_assignments = self.input_metadata_fn(
                str(self.data_root / "train"),
                self.band_filter,
            )

    def _needs_train_stats(self) -> bool:
        return False

    def _make_dataset(self, split: str, max_samples: int | None) -> Dataset:
        raise NotImplementedError

    def _make_stats_dataset(self) -> Dataset:
        return self._make_dataset("train", self.max_samples_by_split["train"])

    def _stats_image_from_item(self, item: Any) -> torch.Tensor:
        if isinstance(item, dict):
            if self.stats_image_key is None:
                raise ValueError(
                    f"{self.__class__.__name__}.stats_image_key must be set for "
                    "dict dataset items."
                )
            return item[self.stats_image_key]
        return item[0]

    def _set_train_stats(self, means: list[float], stds: list[float]) -> None:
        self.means = means
        self.stds = stds

    def _calculate_train_stats(self) -> tuple[list[float], list[float]]:
        stats_dataset = self._make_stats_dataset()
        n_pixels = 0
        sum_x = None
        sum_x2 = None

        for item in stats_dataset:
            image = self._stats_image_from_item(item).to(torch.float64)
            pixels = image.shape[-2] * image.shape[-1]
            image_sum = image.sum(dim=(1, 2))
            image_sq_sum = (image**2).sum(dim=(1, 2))
            sum_x = image_sum if sum_x is None else sum_x + image_sum
            sum_x2 = image_sq_sum if sum_x2 is None else sum_x2 + image_sq_sum
            n_pixels += pixels

        if sum_x is None or sum_x2 is None or n_pixels == 0:
            raise RuntimeError(
                f"No train pixels available for {self.stats_log_label} statistics."
            )

        means_tensor = sum_x / n_pixels
        stds_tensor = torch.sqrt(
            torch.clamp(sum_x2 / n_pixels - means_tensor**2, min=1e-12)
        )
        means = means_tensor.tolist()
        stds = stds_tensor.tolist()
        self._set_train_stats(means, stds)
        print(f"[train] {self.stats_log_label} z-score mean:", means)
        print(f"[train] {self.stats_log_label} z-score std:", stds)
        return means, stds

    def _after_fit_setup(self) -> None:
        return None

    def _after_validate_setup(self) -> None:
        return None

    def _after_test_setup(self) -> None:
        return None

    def _dataloader(self, dataset: Dataset, *, shuffle: bool) -> DataLoader:
        kwargs: dict[str, Any] = {}
        if self.collate_fn is not None:
            kwargs["collate_fn"] = self.collate_fn
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            **kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup("validate")
        return self._dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            self.setup("test")
        return self._dataloader(self.test_dataset, shuffle=False)
