"""Semantic segmentation datasets and datamodules for Lunar fine-tuning."""

from __future__ import annotations

import torch

from lfm.all_models.all_tasks.data.collate import (
    collate_semantic_segmentation,
)
from lfm.all_models.all_tasks.data.nodata import build_nodata_policy
from lfm.all_models.all_tasks.data.normalization import build_normalization_strategy
from lfm.all_models.sem_seg import (
    SemanticSegmentationDataModule,
    SemanticSegmentationDataset,
)


def _target_shape(target_size: int | tuple[int, int]) -> tuple[int, int]:
    return (
        (int(target_size), int(target_size))
        if isinstance(target_size, int)
        else (int(target_size[0]), int(target_size[1]))
    )


class LunarSemanticMaskSegmentationDataset(SemanticSegmentationDataset):
    """Paired image/semantic-mask dataset."""

    def __init__(
        self,
        base_dir: str,
        *,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        target_size: int | tuple[int, int] = 256,
        max_samples: int | None = None,
        band_filter: list[int] | None = None,
        image_glob: str = "*.tif",
        label_glob: str = "*_label.*",
        image_suffix: str = "_input_wac_static_chip",
        label_suffix: str = "_label",
        binarize_mask: bool = True,
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        ignore_nodata_in_loss: bool = False,
        nodata_ignore_index: int = -1,
        split_name: str | None = None,
    ) -> None:
        super().__init__(
            base_dir,
            target_size=_target_shape(target_size),
            max_samples=max_samples,
            band_filter=band_filter,
            spatial_transform="crop",
            split_name=split_name,
            image_glob=image_glob,
            label_glob=label_glob,
            image_suffix=image_suffix,
            label_suffix=label_suffix,
            require_all_labels=True,
            label_npz_key="mask",
            binarize_label=binarize_mask,
            scale_inputs=False,
            normalization=build_normalization_strategy(
                normalize_inputs=mean is not None and std is not None,
                means=mean,
                stds=std,
            ),
            nodata_policy=build_nodata_policy(
                no_data_replace=no_data_replace,
                no_label_replace=no_label_replace,
                ignore_nodata_in_loss=ignore_nodata_in_loss,
                nodata_ignore_index=nodata_ignore_index,
            ),
        )

    def format_output(self, sample: dict) -> dict[str, torch.Tensor | str]:
        return {
            "image": sample["image"],
            "mask": sample["mask"],
            "filename": sample["filename"],
        }


class LunarSemanticMaskSegmentationDatamodule(SemanticSegmentationDataModule):
    dataset_cls = LunarSemanticMaskSegmentationDataset
    collate_fn = staticmethod(collate_semantic_segmentation)
    stats_image_key = "image"
    stats_log_label = "Graha semantic"

    def __init__(
        self,
        data_root: str,
        *,
        chips_subdir: str = "chips",
        labels_subdir: str = "labels",
        batch_size: int = 4,
        num_workers: int = 0,
        crop_size: int | tuple[int, int] | None = 256,
        means: list[float] | None = None,
        stds: list[float] | None = None,
        image_glob: str = "*.tif",
        label_glob: str = "*_label.*",
        image_suffix: str = "_input_wac_static_chip",
        label_suffix: str = "_label",
        band_filter: list[int] | None = None,
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
        max_test_samples: int | None = None,
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        ignore_nodata_in_loss: bool = False,
        nodata_ignore_index: int = -1,
        pin_memory: bool = True,
    ) -> None:
        if crop_size is None:
            raise ValueError("Graha semantic datamodules require crop_size.")
        target_shape = _target_shape(crop_size)
        super().__init__(
            data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            target_size=target_shape,
            band_filter=band_filter,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            max_test_samples=max_test_samples,
            pin_memory=pin_memory,
            image_suffix=image_suffix,
            label_suffix=label_suffix,
            means=means,
            stds=stds,
            scale_inputs=False,
            ignore_nodata_in_loss=ignore_nodata_in_loss,
            nodata_ignore_index=nodata_ignore_index,
        )
        self.chips_subdir = chips_subdir
        self.labels_subdir = labels_subdir
        self.data_layout = self.data_layout.__class__(
            self.data_root,
            chips_subdir=chips_subdir,
            labels_subdir=labels_subdir,
        )
        self.crop_size = crop_size
        self.image_glob = image_glob
        self.label_glob = label_glob
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace

    def _make_dataset(
        self,
        split: str,
        max_samples: int | None,
    ) -> LunarSemanticMaskSegmentationDataset:
        return self.dataset_cls(
            str(self.data_root / split),
            mean=self.mean.tolist() if self.mean is not None else None,
            std=self.std.tolist() if self.std is not None else None,
            target_size=self.crop_size,
            max_samples=max_samples,
            band_filter=self.band_filter,
            image_glob=self.image_glob,
            label_glob=self.label_glob,
            image_suffix=self.image_suffix,
            label_suffix=self.label_suffix,
            no_data_replace=self.no_data_replace,
            no_label_replace=self.no_label_replace,
            ignore_nodata_in_loss=self.ignore_nodata_in_loss,
            nodata_ignore_index=self.nodata_ignore_index,
            split_name=split,
        )

    def _make_stats_dataset(self) -> LunarSemanticMaskSegmentationDataset:
        return self.dataset_cls(
            str(self.data_root / "train"),
            mean=None,
            std=None,
            target_size=self.crop_size,
            max_samples=self.max_samples_by_split["train"],
            band_filter=self.band_filter,
            image_glob=self.image_glob,
            label_glob=self.label_glob,
            image_suffix=self.image_suffix,
            label_suffix=self.label_suffix,
            no_data_replace=self.no_data_replace,
            no_label_replace=self.no_label_replace,
            ignore_nodata_in_loss=self.ignore_nodata_in_loss,
            nodata_ignore_index=self.nodata_ignore_index,
            split_name="train-stats",
        )

    def get_sanity_summary(self) -> dict:
        if self.train_dataset is None:
            self.setup("fit")
        sample = self.train_dataset[0]
        foreground_fraction = float((sample["mask"] > 0).float().mean().item())
        return {
            "data_root": str(self.data_root),
            "crop_size": self.crop_size,
            "band_filter": self.band_filter,
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
            "train_samples": len(self.train_dataset),
            "val_samples": (
                len(self.val_dataset) if self.val_dataset is not None else None
            ),
            "test_samples": (
                len(self.test_dataset) if self.test_dataset is not None else None
            ),
            "sample_image_shape": tuple(sample["image"].shape),
            "sample_mask_shape": tuple(sample["mask"].shape),
            "sample_mask_values": torch.unique(sample["mask"]).tolist(),
            "sample_foreground_fraction": foreground_fraction,
            "sample_filename": sample["filename"],
            "ignore_nodata_in_loss": self.ignore_nodata_in_loss,
            "nodata_ignore_index": self.nodata_ignore_index,
        }
