"""Split datamodule for Toy Mask R-CNN instance segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from lfm.all_models.all_tasks.data.collate import (
    collate_object_detection_instance_segmentation,
)
from lfm.all_models.inst_seg import InstanceSegmentationDataset
from lfm.all_models.inst_seg.data.eligibility import SkippedInstanceChip
from lfm.all_models.inst_seg.data.instance_data_utils import (
    instance_mask_to_object_detection_targets,
)
from lfm.all_models.inst_seg.data.instance_datamodule import (
    InstanceSegmentationDataModule,
)
from lfm.toy_model.inst_seg.iseg_dataset import get_input_metadata


class ToyDinoMaskRCNNSplitDataset(InstanceSegmentationDataset):
    """Toy split dataset emitting TorchVision Mask R-CNN targets."""

    def __getitem__(self, index: int) -> dict[str, Any] | SkippedInstanceChip:
        item = super().__getitem__(index)
        if isinstance(item, SkippedInstanceChip):
            return item
        boxes, labels, masks = instance_mask_to_object_detection_targets(
            item["instance_mask"],
            box_format="xyxy",
        )
        sample = {
            "image": item["pixel_values"].float(),
            "mask": item["instance_mask"].long(),
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "filename": item["filename"],
            "chip_id": item["chip_id"],
            "num_craters": torch.tensor(labels.shape[0], dtype=torch.long),
        }
        if "raw_image" in item:
            sample["raw_image"] = item["raw_image"]
        for name in ("valid_mask", "image_valid_mask"):
            if name in item:
                sample[name] = item[name]
        return sample


class ToyDinoMaskRCNNSplitDataModule(InstanceSegmentationDataModule):
    """Lightning datamodule for Toy Mask R-CNN experiments."""

    dataset_cls = ToyDinoMaskRCNNSplitDataset
    collate_fn = staticmethod(collate_object_detection_instance_segmentation)
    stats_image_key = "image"
    stats_log_label = "Toy Mask R-CNN"

    def __init__(
        self,
        data_root: str | Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            data_root,
            input_metadata_fn=get_input_metadata,
            **kwargs,
        )

    def _setup_input_metadata(self) -> None:
        if self.split_records is None:
            super()._setup_input_metadata()
        elif self.weight_assignments is None:
            self.weight_assignments = get_input_metadata(
                str(self.data_root / "train"),
                self.band_filter,
                image_path=self.split_records["train"][0].image_path,
            )
