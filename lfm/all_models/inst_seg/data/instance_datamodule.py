"""Shared instance segmentation datamodule boundary."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from lfm.all_models.all_tasks.data.base_datamodule import (
    LunarSegmentationDataModule,
)
from lfm.all_models.all_tasks.data.normalization import (
    NormalizationStrategy,
    build_normalization_strategy,
)
from lfm.all_models.all_tasks.data.nodata import (
    NoDataPolicy,
    build_nodata_policy,
)
from lfm.all_models.all_tasks.data.collate import (
    collate_instance_segmentation,
    collate_mask2former_instance_segmentation,
    collate_object_detection_instance_segmentation,
)
from lfm.all_models.inst_seg.data.instance_dataset import (
    InstanceSegmentationDataset,
    LunarInstanceMaskDataset,
    ObjectDetectionInstanceSegmentationDataset,
)

InputMetadataFn = Callable[[str, list[int] | None], list[str]]
InstanceCollateFn = Callable[[list[dict[str, Any]]], dict[str, Any]]


class InstanceSegmentationDataModule(LunarSegmentationDataModule):
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
        image_glob: str = "*chip*.tif",
        label_glob: str = "*label*.npz",
        chips_subdir: str = "chips",
        labels_subdir: str = "labels",
        image_suffix: str | None = None,
        label_suffix: str | None = None,
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
        excluded_nodata_values: list[float] | tuple[float, ...] | None = None,
        nodata_policy: NoDataPolicy | None = None,
        input_metadata_fn: InputMetadataFn | None = None,
        pin_memory: bool = True,
    ) -> None:
        super().__init__(
            data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            chips_subdir=chips_subdir,
            labels_subdir=labels_subdir,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            max_test_samples=max_test_samples,
            band_filter=band_filter,
            pin_memory=pin_memory,
            input_metadata_fn=input_metadata_fn,
        )
        self.target_size = target_size
        self.image_glob = image_glob
        self.label_glob = label_glob
        self.image_suffix = image_suffix
        self.label_suffix = label_suffix
        self.normalize_inputs = normalize_inputs
        self.mask_shift = mask_shift
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
        self.excluded_nodata_values = tuple(
            float(value) for value in excluded_nodata_values or ()
        )
        self.nodata_policy = build_nodata_policy(
            no_data_replace=no_data_replace,
            no_label_replace=no_label_replace,
            ignore_nodata_in_loss=ignore_nodata_in_loss,
            nodata_ignore_index=nodata_ignore_index,
            excluded_nodata_values=self.excluded_nodata_values,
            nodata_policy=nodata_policy,
        )

    def _needs_train_stats(self) -> bool:
        return self.normalize_inputs and (self.means is None or self.stds is None)

    def _validate_split_dirs(self) -> None:
        super()._validate_split_dirs()
        if self.chips_subdir != "chips" or self.labels_subdir != "labels":
            raise ValueError(
                "Instance datamodules currently expect split folders named "
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
            excluded_nodata_values=self.excluded_nodata_values,
            nodata_policy=self.nodata_policy,
            max_samples=max_samples,
            split_name=split,
            **self._dataset_kwargs(),
        )

    def _dataset_kwargs(self) -> dict[str, object]:
        return {}

    def _make_stats_dataset(self) -> InstanceSegmentationDataset:
        return self.dataset_cls(
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
            excluded_nodata_values=self.excluded_nodata_values,
            nodata_policy=self.nodata_policy,
            max_samples=self.max_samples_by_split["train"],
            split_name="train-stats",
            **self._dataset_kwargs(),
        )

    def _set_train_stats(self, means: list[float], stds: list[float]) -> None:
        self.means = means
        self.stds = stds
        self.normalization = build_normalization_strategy(
            normalize_inputs=True,
            means=self.means,
            stds=self.stds,
        )


class InstanceMaskSegmentationDataModule(InstanceSegmentationDataModule):
    """Shared datamodule emitting image/mask instance-label samples."""

    dataset_cls: type[LunarInstanceMaskDataset] = LunarInstanceMaskDataset
    collate_fn = staticmethod(collate_instance_segmentation)
    stats_image_key = "image"


class ObjectDetectionInstanceSegmentationDataModule(InstanceSegmentationDataModule):
    """Shared datamodule emitting object-detection instance targets."""

    dataset_cls: type[ObjectDetectionInstanceSegmentationDataset] = (
        ObjectDetectionInstanceSegmentationDataset
    )
    collate_fn = staticmethod(collate_object_detection_instance_segmentation)
    stats_image_key = "image"

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

    def _dataset_kwargs(self) -> dict[str, object]:
        return {"target_box_format": self.target_box_format}
