"""Image, mask, and box crop helpers for lunar segmentation data."""

from __future__ import annotations

import torch


def crop_boxes_xywh_to_xyxy(
    boxes: torch.Tensor,
    *,
    left: int,
    top: int,
    crop_w: int,
    crop_h: int,
    min_size: float = 1.0,
) -> torch.Tensor:
    """Crop COCO ``xywh`` boxes and return clipped ``xyxy`` boxes."""
    if boxes.numel() == 0:
        return boxes.reshape(0, 4).to(dtype=torch.float32)

    boxes = boxes.to(dtype=torch.float32)
    x1 = boxes[:, 0] - float(left)
    y1 = boxes[:, 1] - float(top)
    x2 = boxes[:, 0] + boxes[:, 2] - float(left)
    y2 = boxes[:, 1] + boxes[:, 3] - float(top)

    cropped = torch.stack(
        [
            x1.clamp(0, crop_w),
            y1.clamp(0, crop_h),
            x2.clamp(0, crop_w),
            y2.clamp(0, crop_h),
        ],
        dim=1,
    )
    keep = (cropped[:, 2] - cropped[:, 0] >= float(min_size)) & (
        cropped[:, 3] - cropped[:, 1] >= float(min_size)
    )
    return cropped[keep]


def crop_boxes_xyxy(
    boxes: torch.Tensor,
    *,
    left: int,
    top: int,
    crop_w: int,
    crop_h: int,
    min_size: float = 1.0,
) -> torch.Tensor:
    """Crop ``xyxy`` boxes and return clipped ``xyxy`` boxes."""
    if boxes.numel() == 0:
        return boxes.reshape(0, 4).to(dtype=torch.float32)

    cropped = boxes.clone().to(dtype=torch.float32)
    cropped[:, [0, 2]] -= float(left)
    cropped[:, [1, 3]] -= float(top)
    cropped[:, [0, 2]] = cropped[:, [0, 2]].clamp(0, crop_w)
    cropped[:, [1, 3]] = cropped[:, [1, 3]].clamp(0, crop_h)
    keep = (cropped[:, 2] - cropped[:, 0] >= float(min_size)) & (
        cropped[:, 3] - cropped[:, 1] >= float(min_size)
    )
    return cropped[keep]


def center_crop(
    image: torch.Tensor,
    mask: torch.Tensor,
    crop_size: int | tuple[int, int],
    *,
    boxes: torch.Tensor | None = None,
    boxes_format: str = "xywh",
    output_boxes_format: str = "xyxy",
    min_box_size: float = 1.0,
    sample_name: str = "",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    old_h, old_w = mask.shape[-2], mask.shape[-1]
    crop_h, crop_w = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
    if crop_h > old_h or crop_w > old_w:
        raise ValueError(
            f"crop_size {(crop_h, crop_w)} exceeds sample size {(old_h, old_w)} "
            f"for {sample_name}"
        )
    top = (old_h - crop_h) // 2
    left = (old_w - crop_w) // 2
    image = image[:, top : top + crop_h, left : left + crop_w]
    mask = mask[top : top + crop_h, left : left + crop_w]

    if boxes is not None:
        if output_boxes_format != "xyxy":
            raise ValueError(f"Unsupported output_boxes_format: {output_boxes_format}")
        if boxes_format == "xywh":
            boxes = crop_boxes_xywh_to_xyxy(
                boxes,
                left=left,
                top=top,
                crop_w=crop_w,
                crop_h=crop_h,
                min_size=min_box_size,
            )
        elif boxes_format == "xyxy":
            boxes = crop_boxes_xyxy(
                boxes,
                left=left,
                top=top,
                crop_w=crop_w,
                crop_h=crop_h,
                min_size=min_box_size,
            )
        else:
            raise ValueError(f"Unsupported boxes_format: {boxes_format}")

    return image, mask, boxes
