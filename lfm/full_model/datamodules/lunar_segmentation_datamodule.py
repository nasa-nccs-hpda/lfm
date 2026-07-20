"""Base Lightning datamodule for Lunar fine-tuning datasets."""

from __future__ import annotations

from pathlib import Path

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from .datamodule_utils import collate_semantic_segmentation
from .lunar_segmentation_dataset import LunarSegmentationDataset


class LunarSegmentationDatamodule(LightningDataModule):
    """Base datamodule for split or flat paired Lunar segmentation datasets.

    ``means`` and ``stds`` are datamodule-side normalization statistics for
    the emitted image tensors. The TerraMind task does not apply them.
    """

    dataset_cls = LunarSegmentationDataset
    collate_fn = staticmethod(collate_semantic_segmentation)

    def __init__(
        self,
        data_root: str | Path = "data",
        *,
        chips_subdir: str = "chips",
        labels_subdir: str = "labels",
        batch_size: int = 4,
        num_workers: int = 0,
        crop_size: int | tuple[int, int] | None = 256,
        means: list[float] | None = None,
        stds: list[float] | None = None,
        binarize_mask: bool = True,
        image_glob: str = "*.tif",
        label_glob: str = "*_label.*",
        image_suffix: str = "_input_wac_static_chip",
        label_suffix: str = "_label",
        val_fraction: float = 0.15,
        test_fraction: float = 0.0,
        split_seed: int = 42,
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.chips_subdir = chips_subdir
        self.labels_subdir = labels_subdir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = crop_size
        self.means = means
        self.stds = stds
        self.binarize_mask = binarize_mask
        self.image_glob = image_glob
        self.label_glob = label_glob
        self.image_suffix = image_suffix
        self.label_suffix = label_suffix
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.split_seed = split_seed
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _make_dataset(
        self,
        chips_dir: Path,
        labels_dir: Path,
        *,
        split_name: str | None = None,
    ) -> Dataset:
        return self.dataset_cls(
            chips_dir=chips_dir,
            labels_dir=labels_dir,
            image_glob=self.image_glob,
            label_glob=self.label_glob,
            image_suffix=self.image_suffix,
            label_suffix=self.label_suffix,
            crop_size=self.crop_size,
            means=self.means,
            stds=self.stds,
            binarize_mask=self.binarize_mask,
            no_data_replace=self.no_data_replace,
            no_label_replace=self.no_label_replace,
            split_name=split_name,
            **self._dataset_kwargs(),
        )

    def _dataset_kwargs(self) -> dict:
        return {}

    def _dataset_for_split(self, split: str) -> Dataset:
        return self._make_dataset(
            chips_dir=self.data_root / split / self.chips_subdir,
            labels_dir=self.data_root / split / self.labels_subdir,
            split_name=split,
        )

    def _flat_dataset(self) -> Dataset:
        return self._make_dataset(
            chips_dir=self.data_root / self.chips_subdir,
            labels_dir=self.data_root / self.labels_subdir,
            split_name="full",
        )

    def setup(self, stage: str | None = None) -> None:
        split_layout = (self.data_root / "train" / self.chips_subdir).exists()

        if split_layout:
            if stage in (None, "fit"):
                self.train_dataset = self._dataset_for_split("train")
                self.val_dataset = self._dataset_for_split("val")
            if stage in (None, "test"):
                test_chips = self.data_root / "test" / self.chips_subdir
                if test_chips.exists():
                    self.test_dataset = self._dataset_for_split("test")
            return

        full = self._flat_dataset()
        n_total = len(full)
        n_test = int(round(n_total * self.test_fraction))
        n_val = int(round(n_total * self.val_fraction))
        n_train = n_total - n_val - n_test
        if n_train <= 0:
            raise ValueError(
                f"Split fractions leave no training samples: total={n_total}, "
                f"val_fraction={self.val_fraction}, test_fraction={self.test_fraction}"
            )

        generator = torch.Generator().manual_seed(self.split_seed)
        splits = random_split(full, [n_train, n_val, n_test], generator=generator)
        self.train_dataset = splits[0]
        self.val_dataset = splits[1] if n_val else splits[0]
        self.test_dataset = splits[2] if n_test else None

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup("fit")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            self.setup("test")
        if self.test_dataset is None:
            raise RuntimeError("No test dataset configured.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def plot(self, sample, stage: str | None = None):
        return None
