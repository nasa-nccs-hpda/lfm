"""NoData-aware helpers for torchvision Mask R-CNN training.

Torchvision's public Mask R-CNN target contract has no pixelwise ignore mask.
During training we therefore encode invalid pixels in each ground-truth instance
mask with a reserved uint8 value, then temporarily replace torchvision's mask
loss with an equivalent implementation that gives those pixels zero weight.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import torch
from torch.nn import functional as F


MASK_NODATA_VALUE = 255


def encode_nodata_in_instance_masks(
    masks: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Mark invalid spatial pixels in every binary instance mask.

    The value 255 survives torchvision's nearest-neighbor target resizing and
    cannot collide with the normal binary target values 0 and 1.
    """
    if masks.ndim != 3:
        raise ValueError(f"Expected masks with shape (N,H,W), got {tuple(masks.shape)}")
    if valid_mask.ndim != 2:
        raise ValueError(
            f"Expected valid_mask with shape (H,W), got {tuple(valid_mask.shape)}"
        )
    if tuple(masks.shape[-2:]) != tuple(valid_mask.shape):
        raise ValueError(
            "Instance-mask and validity-mask spatial shapes differ: "
            f"masks={tuple(masks.shape)}, valid_mask={tuple(valid_mask.shape)}"
        )
    if masks.shape[0] == 0 or torch.all(valid_mask):
        return masks.to(torch.uint8)

    encoded = masks.to(torch.uint8).clone()
    encoded[:, ~valid_mask.to(device=encoded.device, dtype=torch.bool)] = (
        MASK_NODATA_VALUE
    )
    return encoded


def validity_weighted_mask_loss(
    mask_logits: torch.Tensor,
    labels: torch.Tensor,
    projected_targets: torch.Tensor,
    projected_validity: torch.Tensor,
) -> torch.Tensor:
    """Compute Mask R-CNN BCE using only the projected valid-pixel fraction."""
    if projected_targets.numel() == 0:
        return mask_logits.sum() * 0

    row_indices = torch.arange(labels.shape[0], device=labels.device)
    selected_logits = mask_logits[row_indices, labels]
    weights = projected_validity.to(selected_logits.dtype).clamp_(0.0, 1.0)
    weight_sum = weights.sum()
    if not bool(weight_sum > 0):
        return selected_logits.sum() * 0

    # ROIAlign averages over each output bin. Dividing the foreground mass by
    # the valid mass preserves the target foreground fraction after invalid
    # samples have been removed from that bin.
    targets = projected_targets.to(selected_logits.dtype)
    targets = (targets / weights.clamp_min(torch.finfo(weights.dtype).eps)).clamp_(
        0.0, 1.0
    )
    pixel_loss = F.binary_cross_entropy_with_logits(
        selected_logits,
        targets,
        reduction="none",
    )
    return (pixel_loss * weights).sum() / weight_sum


def nodata_aware_maskrcnn_loss(
    mask_logits: torch.Tensor,
    proposals: Sequence[torch.Tensor],
    gt_masks: Sequence[torch.Tensor],
    gt_labels: Sequence[torch.Tensor],
    mask_matched_idxs: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Drop encoded NoData pixels from torchvision's Mask R-CNN mask loss."""
    from torchvision.models.detection.roi_heads import project_masks_on_boxes

    discretization_size = mask_logits.shape[-1]
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
    projected_targets = []
    projected_validity = []

    for masks, boxes, matched_idxs in zip(gt_masks, proposals, mask_matched_idxs):
        valid = masks != MASK_NODATA_VALUE
        clean_masks = torch.where(valid, masks, torch.zeros_like(masks))
        projected_targets.append(
            project_masks_on_boxes(
                clean_masks,
                boxes,
                matched_idxs,
                discretization_size,
            )
        )
        projected_validity.append(
            project_masks_on_boxes(
                valid.to(masks.dtype),
                boxes,
                matched_idxs,
                discretization_size,
            )
        )

    labels_tensor = torch.cat(labels, dim=0)
    targets_tensor = torch.cat(projected_targets, dim=0)
    validity_tensor = torch.cat(projected_validity, dim=0)
    return validity_weighted_mask_loss(
        mask_logits,
        labels_tensor,
        targets_tensor,
        validity_tensor,
    )


@contextmanager
def use_nodata_aware_maskrcnn_loss(enabled: bool) -> Iterator[None]:
    """Use the NoData-aware loss for one synchronous model forward."""
    if not enabled:
        yield
        return

    from torchvision.models.detection import roi_heads

    previous = roi_heads.maskrcnn_loss
    roi_heads.maskrcnn_loss = nodata_aware_maskrcnn_loss
    try:
        yield
    finally:
        roi_heads.maskrcnn_loss = previous
