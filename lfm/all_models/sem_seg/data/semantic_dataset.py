"""Shared semantic segmentation dataset boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from lfm.all_models.all_tasks.data import (
    LunarSegmentationDataset,
    NoDataPolicy,
    build_nodata_policy,
)
from lfm.all_models.all_tasks.data.normalization import (
    NormalizationStrategy,
    build_normalization_strategy,
)


class SemanticSegmentationDataset(LunarSegmentationDataset):
    """Dataset specialization for semantic masks.

    The all-task base class owns image-side loading and preprocessing. This
    subclass names the semantic target contract so Toy and Graha semantic
    datasets can share the same task-specific boundary before the datamodule
    hierarchy is fully unified.
    """

    def __init__(
        self,
        base_dir: str | Path,
        *,
        mean: list[float] | np.ndarray | None = None,
        std: list[float] | np.ndarray | None = None,
        target_size: tuple[int, int] = (256, 256),
        max_samples: int | None = None,
        band_filter: list[int] | None = None,
        spatial_transform: str = "crop",
        split_name: str | None = None,
        image_glob: str = "*chip*.tif",
        label_glob: str = "*label.*",
        image_suffix: str | None = None,
        label_suffix: str | None = None,
        require_all_labels: bool = False,
        label_npz_key: str = "mask",
        binarize_label: bool | str = "auto",
        normalize_inputs: bool = False,
        normalization: NormalizationStrategy | None = None,
        scale_inputs: bool = True,
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        ignore_nodata_in_loss: bool = False,
        nodata_ignore_index: int = -1,
        excluded_nodata_values: list[float] | tuple[float, ...] | None = None,
        image_nodata_policy: str = "union",
        nodata_policy: NoDataPolicy | None = None,
    ) -> None:
        normalization_strategy = normalization or build_normalization_strategy(
            normalize_inputs=normalize_inputs and mean is not None and std is not None,
            means=mean,
            stds=std,
        )
        nodata_strategy = build_nodata_policy(
            no_data_replace=no_data_replace,
            no_label_replace=no_label_replace,
            ignore_nodata_in_loss=ignore_nodata_in_loss,
            nodata_ignore_index=nodata_ignore_index,
            excluded_nodata_values=excluded_nodata_values,
            image_nodata_policy=image_nodata_policy,
            nodata_policy=nodata_policy,
        )
        super().__init__(
            base_dir,
            target_size=target_size,
            max_samples=max_samples,
            band_filter=band_filter,
            spatial_transform=spatial_transform,
            split_name=split_name,
            image_glob=image_glob,
            label_glob=label_glob,
            image_suffix=image_suffix,
            label_suffix=label_suffix,
            require_all_labels=require_all_labels,
            label_npz_key=label_npz_key,
            binarize_label=binarize_label,
            scale_inputs=scale_inputs,
            normalization=normalization_strategy,
            nodata_policy=nodata_strategy,
        )

    def format_output(self, sample: dict[str, Any]) -> dict[str, Any]:
        return {
            "image": sample["image"],
            "mask": sample["mask"],
            "filename": sample["filename"],
            "image_path": sample["image_path"],
            "label_path": sample["label_path"],
        }
