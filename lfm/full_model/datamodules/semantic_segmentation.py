"""Semantic segmentation datasets and datamodules for Lunar fine-tuning."""

from __future__ import annotations

import torch

from .datamodule import LunarSegmentationDatamodule
from .datamodule_utils import collate_semantic_segmentation
from .lunar_segmentation_dataset import LunarSegmentationDataset


class LunarSemanticSegmentationDataset(LunarSegmentationDataset):
    """Paired image/semantic-mask dataset."""

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample, _ = self._load_common(index)
        sample, _ = self._finalize_sample(sample)
        return sample


class LunarSemanticSegmentationDatamodule(LunarSegmentationDatamodule):
    dataset_cls = LunarSemanticSegmentationDataset
    collate_fn = staticmethod(collate_semantic_segmentation)
