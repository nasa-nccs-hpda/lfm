"""Shared semantic segmentation data classes."""

from lfm.all_models.sem_seg.data.semantic_datamodule import (
    SemanticSegmentationDataModule,
)
from lfm.all_models.sem_seg.data.semantic_dataset import SemanticSegmentationDataset

__all__ = ["SemanticSegmentationDataModule", "SemanticSegmentationDataset"]
