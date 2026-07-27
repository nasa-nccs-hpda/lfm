"""Semantic segmentation datamodule backed by instance-label archives."""

from __future__ import annotations

import numpy as np
import torch

from lfm.full_model.all_tasks.datamodules import (
    LunarSegmentationDatamodule,
    LunarSegmentationDataset,
)
from lfm.all_models.all_tasks.data.collate import (
    collate_instance_segmentation,
)
from lfm.all_models.inst_seg.instance_data_utils import boxes_to_tensor


class LunarSemanticFromInstanceDataset(LunarSegmentationDataset):
    """Use instance ``.npz`` labels as semantic masks plus crater boxes."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.pop("label_glob", None)
        kwargs.pop("binarize_mask", None)
        super().__init__(
            *args,
            label_glob="*_label.npz",
            binarize_mask=True,
            **kwargs,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample, label = self._load_common(index)
        crater_boxes = None
        num_craters = None

        if isinstance(label, dict):
            crater_boxes = boxes_to_tensor(label.get("bboxes"))
            raw_num_craters = label.get("num_craters")
            if raw_num_craters is not None:
                num_craters = int(np.asarray(raw_num_craters).item())

        sample, crater_boxes = self._finalize_sample(sample, boxes=crater_boxes)
        if crater_boxes is not None:
            sample["crater_boxes"] = crater_boxes
            num_craters = int(crater_boxes.shape[0])
        if num_craters is not None:
            sample["num_craters"] = torch.tensor(num_craters, dtype=torch.long)
        return sample


class LunarSemanticFromInstanceDatamodule(LunarSegmentationDatamodule):
    """Semantic datamodule that preserves instance boxes for shape loss."""

    dataset_cls = LunarSemanticFromInstanceDataset
    collate_fn = staticmethod(collate_instance_segmentation)
