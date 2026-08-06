"""Instance segmentation data and target formatting helpers."""

from __future__ import annotations

import numpy as np
import torch


def boxes_to_tensor(arr: np.ndarray | None) -> torch.Tensor:
    if arr is None:
        return torch.zeros((0, 4), dtype=torch.float32)
    arr = np.asarray(arr)
    if arr.size == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    arr = arr.reshape(-1, arr.shape[-1])
    if arr.shape[-1] < 4:
        raise ValueError(
            f"Expected bbox array with at least 4 columns, got {arr.shape}"
        )
    return torch.as_tensor(arr[:, :4], dtype=torch.float32)


def mask_to_binary_instance_targets(
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask_labels: list[torch.Tensor] = []
    class_labels: list[torch.Tensor] = []
    for instance_id in torch.unique(mask).tolist():
        if int(instance_id) <= 0:
            continue
        instance_mask = (mask == int(instance_id)).float()
        if instance_mask.any():
            mask_labels.append(instance_mask)
            class_labels.append(torch.tensor(1, dtype=torch.long))

    if not mask_labels:
        return (
            torch.zeros((0, *mask.shape[-2:]), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.long),
        )

    return torch.stack(mask_labels).float(), torch.stack(class_labels).long()


def instance_mask_to_object_detection_targets(
    mask: torch.Tensor,
    *,
    box_format: str = "xyxy",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert an instance-id mask into boxes, labels, and binary masks."""
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D instance mask, got shape {tuple(mask.shape)}")
    if box_format not in {"xyxy", "cxcywh"}:
        raise ValueError(f"Unsupported box_format: {box_format}")

    height, width = mask.shape[-2], mask.shape[-1]
    instance_ids = torch.unique(mask)
    instance_ids = instance_ids[instance_ids > 0]

    boxes: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []

    for instance_id in instance_ids.tolist():
        instance_mask = mask == int(instance_id)
        ys, xs = torch.where(instance_mask)
        if xs.numel() == 0 or ys.numel() == 0:
            continue

        x1 = xs.min().to(torch.float32)
        y1 = ys.min().to(torch.float32)
        x2 = xs.max().to(torch.float32) + 1.0
        y2 = ys.max().to(torch.float32) + 1.0
        if x2 <= x1 or y2 <= y1:
            continue

        if box_format == "xyxy":
            box = torch.stack([x1, y1, x2, y2])
        else:
            cx = ((x1 + x2) * 0.5) / float(width)
            cy = ((y1 + y2) * 0.5) / float(height)
            box_w = (x2 - x1) / float(width)
            box_h = (y2 - y1) / float(height)
            box = torch.stack([cx, cy, box_w, box_h])

        boxes.append(box)
        labels.append(torch.tensor(1, dtype=torch.long))
        masks.append(instance_mask.to(torch.uint8))

    if not boxes:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.long),
            torch.zeros((0, height, width), dtype=torch.uint8),
        )

    return (
        torch.stack(boxes).to(torch.float32),
        torch.stack(labels).to(torch.long),
        torch.stack(masks).to(torch.uint8),
    )
