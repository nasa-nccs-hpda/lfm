"""Toy semantic datamodule using instance-label archives."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from lfm.all_models.all_tasks.data.image_crop_resize import (
    crop_boxes_xywh_to_xyxy,
)
from lfm.all_models.inst_seg.instance_data_utils import boxes_to_tensor
from lfm.toy_model.sem_seg.lightning_wrappers.toy_sem_seg_datamodule import (
    LunarSemanticSegmentationSplitDataModule,
)
from lfm.toy_model.sem_seg.sseg_dataset import LunarCraterDataset


class ToySemSegFromInstanceDataset(LunarCraterDataset):
    """Convert instance ``.npz`` labels to semantic masks and keep boxes."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            label_file_type=".npz",
            label_npz_key="mask",
            binarize_label=True,
            **kwargs,
        )

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, str, str, torch.Tensor]:
        image, label, img_path, label_path = super().__getitem__(idx)
        with np.load(label_path) as archive:
            boxes = boxes_to_tensor(archive["bboxes"])
            original_size = tuple(archive["mask"].shape[-2:])
        if self.spatial_transform == "crop" and boxes.numel() > 0:
            boxes = _center_crop_boxes(
                boxes,
                original_size=original_size,
                target_size=self.target_size,
            )
        return image, label, img_path, label_path, boxes


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

    boxes = boxes.clone()
    return crop_boxes_xywh_to_xyxy(
        boxes,
        left=left,
        top=top,
        crop_w=crop_w,
        crop_h=crop_h,
    )


def collate_toy_semantic_from_instance(
    batch: list[tuple[torch.Tensor, torch.Tensor, str, str, torch.Tensor]],
) -> tuple[Any, ...]:
    images, labels, image_paths, label_paths, crater_boxes = zip(*batch)
    return (
        torch.stack(list(images)),
        torch.stack(list(labels)),
        list(image_paths),
        list(label_paths),
        list(crater_boxes),
    )


class ToySemSegFromInstanceDataModule(LunarSemanticSegmentationSplitDataModule):
    """Toy semantic datamodule for instance-label data roots."""

    dataset_cls = ToySemSegFromInstanceDataset
    collate_fn = staticmethod(collate_toy_semantic_from_instance)
    stats_log_label = "Toy semantic-from-instance"

    def _dataset_kwargs(self) -> dict[str, object]:
        return {"image_file_type": self.image_file_type}

    def get_sanity_summary(self) -> dict[str, Any]:
        if self.train_dataset is None:
            self.setup("fit")
        image, mask, image_path, label_path, crater_boxes = self.train_dataset[0]
        foreground_fraction = float((mask > 0).float().mean().item())
        return {
            "data_root": str(self.data_root),
            "target_size": self.target_size,
            "spatial_transform": self.spatial_transform,
            "semantic_label_source": "instance",
            "band_filter": self.band_filter,
            "normalize_inputs": self.normalize_inputs,
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
            "weight_assignments": self.weight_assignments,
            "train_samples": len(self.train_dataset),
            "val_samples": (
                len(self.val_dataset) if self.val_dataset is not None else None
            ),
            "test_samples": (
                len(self.test_dataset) if self.test_dataset is not None else None
            ),
            "sample_image_shape": tuple(image.shape),
            "sample_mask_shape": tuple(mask.shape),
            "sample_mask_values": torch.unique(mask).tolist(),
            "sample_foreground_fraction": foreground_fraction,
            "sample_crater_boxes_shape": tuple(crater_boxes.shape),
            "sample_image_path": image_path,
            "sample_label_path": label_path,
            "ignore_nodata_in_loss": self.ignore_nodata_in_loss,
            "nodata_ignore_index": self.nodata_ignore_index,
        }
