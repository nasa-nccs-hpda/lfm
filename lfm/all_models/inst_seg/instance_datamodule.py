"""Shared instance segmentation datamodule boundary."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from lfm.all_models.all_tasks.data.normalization import (
    NormalizationStrategy,
    build_normalization_strategy,
)
from lfm.all_models.all_tasks.data.nodata import (
    NoDataPolicy,
    build_nodata_policy,
)
from lfm.all_models.all_tasks.data.base_datamodule import SplitFolderDataLayout
from lfm.all_models.inst_seg.instance_data_utils import (
    collate_mask2former_instance_segmentation,
)
from lfm.all_models.inst_seg.instance_dataset import InstanceSegmentationDataset

InputMetadataFn = Callable[[str, list[int] | None], list[str]]
InstanceCollateFn = Callable[[list[dict[str, Any]]], dict[str, Any]]


class InstanceSegmentationDataModule(LightningDataModule):
    """Shared split datamodule for lunar instance segmentation datasets."""

    dataset_cls: type[InstanceSegmentationDataset] = InstanceSegmentationDataset
    collate_fn: InstanceCollateFn = staticmethod(
        collate_mask2former_instance_segmentation
    )
    stats_image_key = "pixel_values"
    stats_log_label = "instance segmentation"

    def __init__(
        self,
        data_root: str | Path,
        *,
        batch_size: int = 2,
        num_workers: int = 10,
        target_size: int | tuple[int, int] = 256,
        image_glob: str = "*.tif",
        label_glob: str = "*_label.npz",
        chips_subdir: str = "chips",
        labels_subdir: str = "labels",
        image_suffix: str = "_input_wac_static_chip",
        label_suffix: str = "_label",
        band_filter: list[int] | None = None,
        normalize_inputs: bool = False,
        mask_shift: tuple[int, int] | None = None,
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
        max_test_samples: int | None = None,
        means: list[float] | None = None,
        stds: list[float] | None = None,
        normalization: NormalizationStrategy | None = None,
        scale_inputs: bool = True,
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        ignore_nodata_in_loss: bool = False,
        nodata_ignore_index: int = -1,
        nodata_policy: NoDataPolicy | None = None,
        input_metadata_fn: InputMetadataFn | None = None,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.data_layout = SplitFolderDataLayout(
            self.data_root,
            chips_subdir=chips_subdir,
            labels_subdir=labels_subdir,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size
        self.image_glob = image_glob
        self.label_glob = label_glob
        self.chips_subdir = chips_subdir
        self.labels_subdir = labels_subdir
        self.image_suffix = image_suffix
        self.label_suffix = label_suffix
        self.band_filter = band_filter
        self.normalize_inputs = normalize_inputs
        self.mask_shift = mask_shift
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples
        self.means = means
        self.stds = stds
        self.normalization = normalization or build_normalization_strategy(
            normalize_inputs=normalize_inputs
            and means is not None
            and stds is not None,
            means=means,
            stds=stds,
        )
        self.scale_inputs = scale_inputs
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.ignore_nodata_in_loss = ignore_nodata_in_loss
        self.nodata_ignore_index = int(nodata_ignore_index)
        self.nodata_policy = build_nodata_policy(
            no_data_replace=no_data_replace,
            no_label_replace=no_label_replace,
            ignore_nodata_in_loss=ignore_nodata_in_loss,
            nodata_ignore_index=nodata_ignore_index,
            nodata_policy=nodata_policy,
        )
        self.input_metadata_fn = input_metadata_fn
        self.weight_assignments: list[str] | None = None

    def setup(self, stage: str | None = None) -> None:
        self._validate_split_dirs()
        if self.weight_assignments is None and self.input_metadata_fn is not None:
            self.weight_assignments = self.input_metadata_fn(
                str(self.data_root / "train"),
                self.band_filter,
            )
        if self.normalize_inputs and (self.means is None or self.stds is None):
            self.means, self.stds = self._calculate_train_stats()
            self.normalization = build_normalization_strategy(
                normalize_inputs=True,
                means=self.means,
                stds=self.stds,
            )

        if stage in (None, "fit"):
            self.train_dataset = self._make_dataset("train", self.max_train_samples)
            self.val_dataset = self._make_dataset("val", self.max_val_samples)
        if stage in (None, "test", "predict"):
            self.test_dataset = self._make_dataset("test", self.max_test_samples)

    def _validate_split_dirs(self) -> None:
        self.data_layout.require_split_dirs(["train", "val", "test"])
        if self.chips_subdir != "chips" or self.labels_subdir != "labels":
            raise ValueError(
                "Toy instance input metadata currently expects split folders named "
                "'chips' and 'labels'."
            )

    def _make_dataset(
        self,
        split: str,
        max_samples: int | None,
    ) -> InstanceSegmentationDataset:
        return self.dataset_cls(
            self.data_root / split,
            target_size=self.target_size,
            image_glob=self.image_glob,
            label_glob=self.label_glob,
            image_suffix=self.image_suffix,
            label_suffix=self.label_suffix,
            band_filter=self.band_filter,
            normalize_inputs=self.normalize_inputs,
            means=self.means,
            stds=self.stds,
            normalization=self.normalization,
            scale_inputs=self.scale_inputs,
            mask_shift=self.mask_shift,
            no_data_replace=self.no_data_replace,
            no_label_replace=self.no_label_replace,
            ignore_nodata_in_loss=self.ignore_nodata_in_loss,
            nodata_ignore_index=self.nodata_ignore_index,
            nodata_policy=self.nodata_policy,
            max_samples=max_samples,
            split_name=split,
        )

    def _calculate_train_stats(self) -> tuple[list[float], list[float]]:
        stats_dataset = self.dataset_cls(
            self.data_root / "train",
            target_size=self.target_size,
            image_glob=self.image_glob,
            label_glob=self.label_glob,
            image_suffix=self.image_suffix,
            label_suffix=self.label_suffix,
            band_filter=self.band_filter,
            normalize_inputs=False,
            normalization=None,
            scale_inputs=self.scale_inputs,
            mask_shift=self.mask_shift,
            no_data_replace=self.no_data_replace,
            no_label_replace=self.no_label_replace,
            ignore_nodata_in_loss=self.ignore_nodata_in_loss,
            nodata_ignore_index=self.nodata_ignore_index,
            nodata_policy=self.nodata_policy,
            max_samples=self.max_train_samples,
            split_name="train-stats",
        )
        n_pixels = 0
        sum_x = None
        sum_x2 = None
        for item in stats_dataset:
            x = item[self.stats_image_key]
            pixels = x.shape[-2] * x.shape[-1]
            x_sum = x.sum(dim=(1, 2))
            x2_sum = (x * x).sum(dim=(1, 2))
            sum_x = x_sum if sum_x is None else sum_x + x_sum
            sum_x2 = x2_sum if sum_x2 is None else sum_x2 + x2_sum
            n_pixels += pixels
        if sum_x is None or sum_x2 is None or n_pixels == 0:
            raise RuntimeError(
                f"No train pixels available for {self.stats_log_label} statistics."
            )
        means = (sum_x / n_pixels).tolist()
        stds = torch.sqrt(
            torch.clamp(sum_x2 / n_pixels - (sum_x / n_pixels) ** 2, min=1e-12)
        ).tolist()
        print(f"[train] {self.stats_log_label} z-score mean:", means)
        print(f"[train] {self.stats_log_label} z-score std:", stds)
        return means, stds

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
