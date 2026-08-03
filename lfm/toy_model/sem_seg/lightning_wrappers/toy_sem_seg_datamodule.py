"""Split-folder datamodule for lunar semantic segmentation workflows."""

from __future__ import annotations

from lfm.all_models.sem_seg import SemanticSegmentationDataModule
from lfm.toy_model.sem_seg.sseg_dataset import (
    LunarCraterDataset,
    get_input_metadata,
)


class LunarSemanticSegmentationSplitDataModule(SemanticSegmentationDataModule):
    """Use the old toy semantic-seg dataset with explicit train/val/test splits.

    This wrapper intentionally preserves the legacy semantic data behavior used
    by the comparison workflow: .npy labels and per-sample band min-max scaling
    inside ``LunarCraterDataset``. If ``normalize_inputs=True``, per-band
    z-score statistics are computed from the training split after the same
    crop/min-max preprocessing used by the model.
    """

    dataset_cls = LunarCraterDataset
    input_metadata_fn = staticmethod(get_input_metadata)
    stats_log_label = "Toy semantic"
