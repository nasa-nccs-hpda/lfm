"""Split datamodule for Toy Mask2Former instance segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lfm.all_models.inst_seg.instance_datamodule import InstanceSegmentationDataModule
from lfm.toy_model.inst_seg.iseg_dataset import get_input_metadata


class ToyInstanceSegSplitDataModule(InstanceSegmentationDataModule):
    """Lightning datamodule for split toy instance segmentation data."""

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
