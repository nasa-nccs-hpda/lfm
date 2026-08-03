"""Shared semantic segmentation dataset boundary."""

from __future__ import annotations

from typing import Any

from lfm.all_models.all_tasks.data import LunarSegmentationDataset


class SemanticSegmentationDataset(LunarSegmentationDataset):
    """Dataset specialization for semantic masks.

    The all-task base class owns image-side loading and preprocessing. This
    subclass names the semantic target contract so Toy and Graha semantic
    datasets can share the same task-specific boundary before the datamodule
    hierarchy is fully unified.
    """

    def format_output(self, sample: dict[str, Any]) -> dict[str, Any]:
        return {
            "image": sample["image"],
            "mask": sample["mask"],
            "filename": sample["filename"],
            "image_path": sample["image_path"],
            "label_path": sample["label_path"],
        }
