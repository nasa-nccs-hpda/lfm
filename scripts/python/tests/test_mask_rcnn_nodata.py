"""Focused tests for NoData-aware Mask R-CNN target and loss handling."""

import torch
from torch.nn import functional as F

from lfm.all_models.all_tasks.data.collate import (
    collate_object_detection_instance_segmentation,
)
from lfm.all_models.inst_seg.mask_rcnn_nodata import (
    MASK_NODATA_VALUE,
    encode_nodata_in_instance_masks,
    validity_weighted_mask_loss,
)
from lfm.full_model.inst_seg.instance_graha_components import (
    make_downstream_object_detection_task_class,
)


def test_encode_nodata_marks_every_instance_without_changing_valid_pixels():
    masks = torch.tensor(
        [
            [[0, 1], [0, 0]],
            [[0, 0], [1, 0]],
        ],
        dtype=torch.uint8,
    )
    valid_mask = torch.tensor([[True, False], [True, True]])

    encoded = encode_nodata_in_instance_masks(masks, valid_mask)

    assert torch.equal(encoded[:, valid_mask], masks[:, valid_mask])
    assert torch.all(encoded[:, ~valid_mask] == MASK_NODATA_VALUE)


def test_validity_weighted_mask_loss_has_zero_gradient_on_invalid_pixel():
    logits = torch.zeros((1, 2, 2, 2), requires_grad=True)
    labels = torch.tensor([1])
    targets = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    validity = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]])

    loss = validity_weighted_mask_loss(logits, labels, targets, validity)
    loss.backward()

    expected = F.binary_cross_entropy_with_logits(
        torch.zeros(3),
        torch.tensor([1.0, 1.0, 0.0]),
    )
    assert torch.allclose(loss.detach(), expected)
    assert logits.grad is not None
    assert logits.grad[0, 1, 0, 1] == 0
    assert torch.count_nonzero(logits.grad[0, 0]) == 0


def test_validity_weighted_mask_loss_handles_fully_invalid_roi():
    logits = torch.randn((1, 2, 2, 2), requires_grad=True)
    loss = validity_weighted_mask_loss(
        logits,
        torch.tensor([1]),
        torch.zeros(1, 2, 2),
        torch.zeros(1, 2, 2),
    )

    loss.backward()

    assert loss.detach() == 0
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 0


def test_object_detection_collate_preserves_valid_mask():
    item = {
        "image": torch.zeros(1, 2, 2),
        "boxes": torch.zeros(0, 4),
        "labels": torch.zeros(0, dtype=torch.long),
        "masks": torch.zeros(0, 2, 2, dtype=torch.uint8),
        "mask": torch.zeros(2, 2, dtype=torch.long),
        "valid_mask": torch.tensor([[True, False], [True, True]]),
        "filename": "sample.tif",
    }

    batch = collate_object_detection_instance_segmentation([item])

    assert batch["valid_mask"].dtype == torch.bool
    assert tuple(batch["valid_mask"].shape) == (1, 2, 2)


class _FakeObjectDetectionTask(torch.nn.Module):
    masks_field = "masks"
    boxes_field = "boxes"
    labels_field = "labels"

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return {}


def test_graha_task_encodes_nodata_only_for_training_targets():
    task_cls = make_downstream_object_detection_task_class(_FakeObjectDetectionTask)
    task = task_cls(ignore_nodata_in_loss=True)
    batch = {
        "image": torch.zeros(1, 1, 2, 2),
        "boxes": [torch.tensor([[0.0, 0.0, 2.0, 2.0]])],
        "labels": [torch.tensor([1])],
        "masks": [torch.ones(1, 2, 2, dtype=torch.uint8)],
        "valid_mask": torch.tensor([[[True, False], [True, True]]]),
    }

    task.train()
    train_target = task.reformat_batch(batch, batch_size=1)[0]
    assert train_target["masks"][0, 0, 1] == MASK_NODATA_VALUE

    task.eval()
    eval_target = task.reformat_batch(batch, batch_size=1)[0]
    assert eval_target["masks"][0, 0, 1] == 1
