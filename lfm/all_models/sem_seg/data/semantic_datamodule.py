"""Shared semantic segmentation datamodule boundary."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lfm.all_models.all_tasks.data import LunarSegmentationDataModule
from lfm.all_models.sem_seg.data.semantic_dataset import SemanticSegmentationDataset

InputMetadataFn = Callable[[str, list[int] | None], list[str]]


class SemanticSegmentationDataModule(LunarSegmentationDataModule):
    """Shared split datamodule for lunar semantic segmentation datasets."""

    dataset_cls: type[SemanticSegmentationDataset] = SemanticSegmentationDataset
    input_metadata_fn: InputMetadataFn | None = None
    stats_log_label = "semantic segmentation"

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
        image_file_type: str = ".tif",
        label_file_type: str = ".npy",
        image_suffix: str = "_input_wac_static_chip",
        label_suffix: str = "_label",
        label_npz_key: str = "mask",
        binarize_label: bool = False,
        means: list[float] | np.ndarray | None = None,
        stds: list[float] | np.ndarray | None = None,
        scale_inputs: bool = True,
        ignore_nodata_in_loss: bool = False,
        nodata_ignore_index: int = -1,
    ) -> None:
        super().__init__(
            data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            max_test_samples=max_test_samples,
            band_filter=band_filter,
            pin_memory=pin_memory,
            input_metadata_fn=self.input_metadata_fn,
        )
        self.target_size = target_size
        self.spatial_transform = spatial_transform
        self.normalize_inputs = normalize_inputs
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.image_file_type = image_file_type
        self.label_file_type = label_file_type
        self.image_suffix = image_suffix
        self.label_suffix = label_suffix
        self.label_npz_key = label_npz_key
        self.binarize_label = binarize_label
        self.scale_inputs = scale_inputs
        self.ignore_nodata_in_loss = ignore_nodata_in_loss
        self.nodata_ignore_index = int(nodata_ignore_index)

        self.mean: np.ndarray | None = (
            np.asarray(means, dtype=np.float32) if means is not None else None
        )
        self.std: np.ndarray | None = (
            np.asarray(stds, dtype=np.float32) if stds is not None else None
        )

    def _needs_train_stats(self) -> bool:
        return self.normalize_inputs and (self.mean is None or self.std is None)

    def _make_dataset(
        self,
        split: str,
        max_samples: int | None,
    ) -> SemanticSegmentationDataset:
        return self.dataset_cls(
            base_dir=str(self.data_root / split),
            mean=self.mean,
            std=self.std,
            target_size=self.target_size,
            spatial_transform=self.spatial_transform,
            max_samples=max_samples,
            band_filter=self.band_filter,
            normalize_inputs=self.normalize_inputs,
            split_name=split,
            image_suffix=self.image_suffix,
            label_suffix=self.label_suffix,
            scale_inputs=self.scale_inputs,
            ignore_nodata_in_loss=self.ignore_nodata_in_loss,
            nodata_ignore_index=self.nodata_ignore_index,
            **self._dataset_kwargs(),
        )

    def _dataset_kwargs(self) -> dict[str, object]:
        return {
            "image_file_type": self.image_file_type,
            "label_file_type": self.label_file_type,
            "label_npz_key": self.label_npz_key,
            "binarize_label": self.binarize_label,
        }

    def _make_stats_dataset(self) -> SemanticSegmentationDataset:
        return self.dataset_cls(
            base_dir=str(self.data_root / "train"),
            mean=None,
            std=None,
            target_size=self.target_size,
            spatial_transform=self.spatial_transform,
            max_samples=self.max_samples_by_split["train"],
            band_filter=self.band_filter,
            normalize_inputs=False,
            split_name="train-stats",
            image_suffix=self.image_suffix,
            label_suffix=self.label_suffix,
            scale_inputs=self.scale_inputs,
            ignore_nodata_in_loss=self.ignore_nodata_in_loss,
            nodata_ignore_index=self.nodata_ignore_index,
            **self._dataset_kwargs(),
        )

    def _set_train_stats(self, means: list[float], stds: list[float]) -> None:
        self.mean = np.asarray(means, dtype=np.float32)
        self.std = np.asarray(stds, dtype=np.float32)

    def _after_fit_setup(self) -> None:
        self._write_file_lists(["train", "val"])
        self._write_sanity_report()

    def _after_validate_setup(self) -> None:
        self._write_file_lists(["val"])

    def _after_test_setup(self) -> None:
        self._write_file_lists(["test"])

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
        if (
            self.output_dir is None
            or self.train_dataset is None
            or self.val_dataset is None
        ):
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
            "val_samples": (
                len(self.val_dataset) if self.val_dataset is not None else None
            ),
            "test_samples": (
                len(self.test_dataset) if self.test_dataset is not None else None
            ),
            "sample_image_shape": tuple(image.shape),
            "sample_mask_shape": tuple(mask.shape),
            "sample_mask_values": torch.unique(mask).tolist(),
            "sample_foreground_fraction": foreground_fraction,
            "sample_image_path": image_path,
            "sample_label_path": label_path,
            "ignore_nodata_in_loss": self.ignore_nodata_in_loss,
            "nodata_ignore_index": self.nodata_ignore_index,
        }
