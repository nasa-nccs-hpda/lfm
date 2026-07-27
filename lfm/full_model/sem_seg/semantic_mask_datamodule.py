"""Semantic segmentation datasets and datamodules for Lunar fine-tuning."""

from __future__ import annotations

import torch

from lfm.full_model.all_tasks.datamodules import (
    LunarSegmentationDatamodule,
    LunarSegmentationDataset,
)
from lfm.all_models.all_tasks.data.collate import (
    collate_semantic_segmentation,
)


class LunarSemanticMaskSegmentationDataset(LunarSegmentationDataset):
    """Paired image/semantic-mask dataset."""

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample, _ = self._load_common(index)
        sample, _ = self._finalize_sample(sample)
        return sample


class LunarSemanticMaskSegmentationDatamodule(LunarSegmentationDatamodule):
    dataset_cls = LunarSemanticMaskSegmentationDataset
    collate_fn = staticmethod(collate_semantic_segmentation)
