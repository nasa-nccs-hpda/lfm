"""Focused tests for tiled Graha instance datacube inference."""

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from lfm.all_models.all_tasks.graha_inference import GrahaInstanceModel
from lfm.all_models.inst_seg.data_cube_inference import _class_aware_nms


def test_class_aware_nms_suppresses_same_class_tile_duplicate():
    boxes = np.asarray(
        [
            [10.0, 10.0, 30.0, 30.0],
            [11.0, 11.0, 31.0, 31.0],
            [11.0, 11.0, 31.0, 31.0],
        ],
        dtype=np.float32,
    )
    scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
    labels = np.asarray([1, 1, 2], dtype=np.int64)

    kept = _class_aware_nms(boxes, scores, labels, iou_threshold=0.5)

    np.testing.assert_array_equal(kept, [0, 2])


class _FakeInstanceTask(nn.Module):
    def predict_step(self, batch, batch_idx):
        height, width = batch["image"].shape[-2:]
        prediction = {
            "boxes": torch.tensor([[1.0, 1.0, 3.0, 3.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([1]),
            "masks": torch.ones(1, 1, height, width),
        }
        return SimpleNamespace(output=[prediction])


def test_graha_instance_model_normalizes_predict_step_output():
    model = GrahaInstanceModel(_FakeInstanceTask())
    predictions = model(torch.zeros(1, 3, 4, 4))

    assert len(predictions) == 1
    assert tuple(predictions[0]["masks"].shape) == (1, 1, 4, 4)
