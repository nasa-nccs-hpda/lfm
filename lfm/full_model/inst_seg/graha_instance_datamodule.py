"""Graha instance segmentation datamodules on shared instance bases."""

from __future__ import annotations

from pathlib import Path

from lfm.all_models.inst_seg.data.instance_datamodule import (
    InstanceMaskSegmentationDataModule,
    ObjectDetectionInstanceSegmentationDataModule,
)


class GrahaInstanceSegmentationDataModule(InstanceMaskSegmentationDataModule):
    """Graha instance-mask datamodule preserving TerraTorch image contracts."""

    stats_log_label = "Graha instance segmentation"

    def __init__(
        self,
        data_root: str | Path,
        *,
        batch_size: int = 4,
        num_workers: int = 0,
        crop_size: int | tuple[int, int] = 256,
        image_glob: str = "*chip*.tif",
        label_glob: str = "*label*.*",
        chips_subdir: str = "chips",
        labels_subdir: str = "labels",
        image_suffix: str | None = None,
        label_suffix: str | None = None,
        band_filter: list[int] | None = None,
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
        max_test_samples: int | None = None,
        means: list[float] | None = None,
        stds: list[float] | None = None,
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        mask_shift: tuple[int, int] | None = None,
        ignore_nodata_in_loss: bool = False,
        nodata_ignore_index: int = -1,
        pin_memory: bool = True,
    ) -> None:
        super().__init__(
            data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            target_size=crop_size,
            image_glob=image_glob,
            label_glob=label_glob,
            chips_subdir=chips_subdir,
            labels_subdir=labels_subdir,
            image_suffix=image_suffix,
            label_suffix=label_suffix,
            band_filter=band_filter,
            normalize_inputs=means is not None and stds is not None,
            mask_shift=mask_shift,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            max_test_samples=max_test_samples,
            means=means,
            stds=stds,
            scale_inputs=False,
            no_data_replace=no_data_replace,
            no_label_replace=no_label_replace,
            ignore_nodata_in_loss=ignore_nodata_in_loss,
            nodata_ignore_index=nodata_ignore_index,
            pin_memory=pin_memory,
        )
        self.crop_size = crop_size


class GrahaObjectDetectionInstanceDataModule(
    ObjectDetectionInstanceSegmentationDataModule
):
    """Graha datamodule emitting TerraTorch ObjectDetectionTask targets."""

    stats_log_label = "Graha object detection instance segmentation"

    def __init__(
        self,
        data_root: str | Path,
        *,
        batch_size: int = 4,
        num_workers: int = 0,
        crop_size: int | tuple[int, int] = 256,
        image_glob: str = "*chip*.tif",
        label_glob: str = "*label*.*",
        chips_subdir: str = "chips",
        labels_subdir: str = "labels",
        image_suffix: str | None = None,
        label_suffix: str | None = None,
        band_filter: list[int] | None = None,
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
        max_test_samples: int | None = None,
        means: list[float] | None = None,
        stds: list[float] | None = None,
        target_box_format: str = "xyxy",
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        mask_shift: tuple[int, int] | None = None,
        ignore_nodata_in_loss: bool = False,
        nodata_ignore_index: int = -1,
        pin_memory: bool = True,
    ) -> None:
        super().__init__(
            data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            target_size=crop_size,
            image_glob=image_glob,
            label_glob=label_glob,
            chips_subdir=chips_subdir,
            labels_subdir=labels_subdir,
            image_suffix=image_suffix,
            label_suffix=label_suffix,
            band_filter=band_filter,
            normalize_inputs=means is not None and stds is not None,
            mask_shift=mask_shift,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            max_test_samples=max_test_samples,
            means=means,
            stds=stds,
            scale_inputs=False,
            no_data_replace=no_data_replace,
            no_label_replace=no_label_replace,
            ignore_nodata_in_loss=ignore_nodata_in_loss,
            nodata_ignore_index=nodata_ignore_index,
            target_box_format=target_box_format,
            pin_memory=pin_memory,
        )
        self.crop_size = crop_size
