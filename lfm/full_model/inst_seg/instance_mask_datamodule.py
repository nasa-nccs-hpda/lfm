"""Instance segmentation datasets and datamodules for Lunar fine-tuning."""

from __future__ import annotations

import numpy as np
import torch

from lfm.full_model.all_tasks.datamodules import (
    LunarSegmentationDatamodule,
    LunarSegmentationDataset,
)
from lfm.full_model.all_tasks.datamodules.datamodule_utils import (
    boxes_to_tensor,
    collate_instance_segmentation,
    collate_object_detection_instance_segmentation,
    instance_mask_to_object_detection_targets,
)


class LunarInstanceMaskSegmentationDataset(LunarSegmentationDataset):
    """Paired image/instance-label dataset with optional crater boxes."""

    def __init__(self, *args, binarize_mask: bool = False, **kwargs) -> None:
        super().__init__(*args, binarize_mask=binarize_mask, **kwargs)

    def _load_instance_sample(
        self,
        index: int,
    ) -> tuple[dict[str, torch.Tensor | str], torch.Tensor | None, int | None]:
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
        return sample, crater_boxes, num_craters

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample, _, num_craters = self._load_instance_sample(index)
        if num_craters is not None:
            sample["num_craters"] = torch.tensor(num_craters, dtype=torch.long)
        return sample


class LunarObjectDetectionInstanceMaskDataset(LunarInstanceMaskSegmentationDataset):
    """Instance-label dataset emitting TerraTorch ObjectDetectionTask targets."""

    def __init__(
        self,
        *args,
        binarize_mask: bool = False,
        target_box_format: str = "xyxy",
        **kwargs,
    ) -> None:
        if binarize_mask:
            raise ValueError(
                "Object-detection instance targets require instance-id masks."
            )
        super().__init__(*args, binarize_mask=False, **kwargs)
        if target_box_format not in {"xyxy", "cxcywh"}:
            raise ValueError(f"Unsupported target_box_format: {target_box_format}")
        self.target_box_format = target_box_format

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample, _, _ = self._load_instance_sample(index)
        boxes, labels, masks = instance_mask_to_object_detection_targets(
            sample["mask"],
            box_format=self.target_box_format,
        )
        sample["boxes"] = boxes
        sample["labels"] = labels
        sample["masks"] = masks
        sample["num_craters"] = torch.tensor(labels.shape[0], dtype=torch.long)
        return sample


class LunarInstanceMaskSegmentationDatamodule(LunarSegmentationDatamodule):
    dataset_cls = LunarInstanceMaskSegmentationDataset
    collate_fn = staticmethod(collate_instance_segmentation)

    def __init__(self, *args, binarize_mask: bool = False, **kwargs) -> None:
        super().__init__(*args, binarize_mask=binarize_mask, **kwargs)


class LunarObjectDetectionInstanceMaskDatamodule(LunarSegmentationDatamodule):
    """Instance datamodule emitting TerraTorch ObjectDetectionTask targets."""

    dataset_cls = LunarObjectDetectionInstanceMaskDataset
    collate_fn = staticmethod(collate_object_detection_instance_segmentation)

    def __init__(self, *args, target_box_format: str = "xyxy", **kwargs) -> None:
        super().__init__(*args, binarize_mask=False, **kwargs)
        if target_box_format not in {"xyxy", "cxcywh"}:
            raise ValueError(f"Unsupported target_box_format: {target_box_format}")
        self.target_box_format = target_box_format

    def _dataset_kwargs(self) -> dict:
        return {"target_box_format": self.target_box_format}
