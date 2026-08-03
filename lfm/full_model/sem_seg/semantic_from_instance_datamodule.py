"""Semantic segmentation datamodule backed by instance-label archives."""

from __future__ import annotations

import numpy as np
import torch

from lfm.all_models.all_tasks.data.collate import collate_instance_segmentation
from lfm.all_models.all_tasks.data.image_crop_resize import (
    crop_boxes_xywh_to_xyxy,
)
from lfm.all_models.all_tasks.data.image_io import read_label_file_with_metadata
from lfm.all_models.inst_seg.data.instance_data_utils import boxes_to_tensor
from lfm.full_model.sem_seg.semantic_mask_datamodule import (
    LunarSemanticMaskSegmentationDatamodule,
    LunarSemanticMaskSegmentationDataset,
)


class LunarSemanticFromInstanceDataset(LunarSemanticMaskSegmentationDataset):
    """Use instance ``.npz`` labels as semantic masks plus crater boxes."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.pop("label_glob", None)
        kwargs.pop("binarize_mask", None)
        super().__init__(
            *args,
            label_glob="*label*.npz",
            binarize_mask=True,
            **kwargs,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.prepare_sample(index)
        record = self.records[index]
        label = read_label_file_with_metadata(record.label_path)
        crater_boxes = None
        num_craters = None

        if isinstance(label, dict):
            crater_boxes = boxes_to_tensor(label.get("bboxes"))
            raw_num_craters = label.get("num_craters")
            if raw_num_craters is not None:
                num_craters = int(np.asarray(raw_num_craters).item())

        sample = self.format_output(sample)
        if crater_boxes is not None and self.spatial_transform == "crop":
            original_size = tuple(label["mask"].shape[-2:])
            crater_boxes = _center_crop_boxes(
                crater_boxes,
                original_size=original_size,
                target_size=self.target_size,
            )
        if crater_boxes is not None:
            sample["crater_boxes"] = crater_boxes
            num_craters = int(crater_boxes.shape[0])
        if num_craters is not None:
            sample["num_craters"] = torch.tensor(num_craters, dtype=torch.long)
        return sample


def _center_crop_boxes(
    boxes: torch.Tensor,
    *,
    original_size: tuple[int, int],
    target_size: int | tuple[int, int],
) -> torch.Tensor:
    height, width = int(original_size[0]), int(original_size[1])
    crop_h, crop_w = (
        (int(target_size), int(target_size))
        if isinstance(target_size, int)
        else (int(target_size[0]), int(target_size[1]))
    )
    top = max((height - crop_h) // 2, 0)
    left = max((width - crop_w) // 2, 0)
    return crop_boxes_xywh_to_xyxy(
        boxes,
        left=left,
        top=top,
        crop_w=crop_w,
        crop_h=crop_h,
    )


class LunarSemanticFromInstanceDatamodule(LunarSemanticMaskSegmentationDatamodule):
    """Semantic datamodule that preserves instance boxes for shape loss."""

    dataset_cls = LunarSemanticFromInstanceDataset
    collate_fn = staticmethod(collate_instance_segmentation)
