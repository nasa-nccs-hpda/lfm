"""Toy semantic Lightning module with Graha-style compactness loss."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from lfm.toy_model.sem_seg.lightning_wrappers.toy_sem_seg_lightning import (
    ToySemSegLightningModule,
    binary_segmentation_stats,
)


def crater_shape_loss(
    prob: Tensor,
    boxes_per_image: list[Tensor],
    *,
    pad_frac: float = 0.3,
    min_side: int = 4,
    eps: float = 1e-6,
) -> Tensor:
    """Graha-style compactness loss over crater-box ROIs."""
    if prob.dim() != 3:
        raise ValueError(f"prob must be (B, H, W), got {tuple(prob.shape)}")
    batch_size, height, width = prob.shape
    if len(boxes_per_image) != batch_size:
        raise ValueError(
            f"boxes_per_image has length {len(boxes_per_image)}, "
            f"but batch dim is {batch_size}"
        )

    total = prob.new_tensor(0.0)
    count = 0
    for batch_index, boxes in enumerate(boxes_per_image):
        if boxes is None or boxes.numel() == 0:
            continue
        for row in boxes.to(prob.device).tolist():
            x1, y1, x2, y2 = row[:4]
            crater_w = max(x2 - x1, 0.0)
            crater_h = max(y2 - y1, 0.0)
            if crater_w < 1.0 or crater_h < 1.0:
                continue
            roi_x1 = max(0, int(x1 - pad_frac * crater_w))
            roi_y1 = max(0, int(y1 - pad_frac * crater_h))
            roi_x2 = min(width, int(x2 + pad_frac * crater_w))
            roi_y2 = min(height, int(y2 + pad_frac * crater_h))
            if (roi_x2 - roi_x1) < min_side or (roi_y2 - roi_y1) < min_side:
                continue

            patch = prob[batch_index, roi_y1:roi_y2, roi_x1:roi_x2]
            area = patch.sum() + eps
            pooled = patch.unsqueeze(0).unsqueeze(0)
            pooled = torch.nn.functional.avg_pool2d(
                pooled,
                kernel_size=2,
                stride=1,
                padding=0,
            )
            pooled = torch.nn.functional.pad(pooled, (0, 1, 0, 1))
            pooled = pooled.squeeze(0).squeeze(0)
            dx = pooled[:-1, 1:] - pooled[:-1, :-1]
            dy = pooled[1:, :-1] - pooled[:-1, :-1]
            perimeter = torch.sqrt(dx * dx + dy * dy + eps).sum()
            compactness = 4.0 * math.pi * area / (perimeter * perimeter + eps)
            total = total + (1.0 - compactness).clamp(min=0.0)
            count += 1

    if count == 0:
        return total
    return total / count


class ToySemSegShapeLightningModule(ToySemSegLightningModule):
    """Toy semantic module with optional crater compactness regularization."""

    def __init__(
        self,
        *args: Any,
        shape_loss_weight: float = 0.05,
        shape_loss_pad_frac: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.shape_loss_weight = float(shape_loss_weight)
        self.shape_loss_pad_frac = float(shape_loss_pad_frac)

    def _loss_with_shape(
        self,
        logits: Tensor,
        labels: Tensor,
        crater_boxes: list[Tensor] | None,
        *,
        stage: str,
        batch_size: int,
    ) -> Tensor:
        base_loss = self.criterion(logits, labels)
        if self.shape_loss_weight == 0.0 or not crater_boxes:
            return base_loss

        prob = torch.softmax(logits, dim=1)[:, 1]
        shape_loss = crater_shape_loss(
            prob,
            crater_boxes,
            pad_frac=self.shape_loss_pad_frac,
        )
        self.log(
            f"{stage}/shape_loss",
            shape_loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=(stage == "train"),
            batch_size=batch_size,
        )
        return base_loss + self.shape_loss_weight * shape_loss

    @staticmethod
    def _unpack_batch(
        batch: tuple[Any, ...],
    ) -> tuple[Tensor, Tensor, list[Tensor] | None]:
        images, labels, *rest = batch
        crater_boxes = rest[2] if len(rest) >= 3 else None
        return images, labels, crater_boxes

    def training_step(self, batch: tuple[Any, ...], batch_idx: int) -> Tensor:
        images, labels, crater_boxes = self._unpack_batch(batch)
        batch_size = images.shape[0]
        logits = self(images)
        loss = self._loss_with_shape(
            logits,
            labels,
            crater_boxes,
            stage="train",
            batch_size=batch_size,
        )
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch: tuple[Any, ...], batch_idx: int) -> Tensor:
        images, labels, crater_boxes = self._unpack_batch(batch)
        batch_size = images.shape[0]
        logits = self(images)
        loss = self._loss_with_shape(
            logits,
            labels,
            crater_boxes,
            stage="val",
            batch_size=batch_size,
        )
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        for name, value in binary_segmentation_stats(logits, labels).items():
            self.log(
                f"val/{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=(name == "f1"),
                batch_size=batch_size,
            )
        return loss

    def test_step(self, batch: tuple[Any, ...], batch_idx: int) -> Tensor:
        images, labels, crater_boxes = self._unpack_batch(batch)
        batch_size = images.shape[0]
        logits = self(images)
        loss = self._loss_with_shape(
            logits,
            labels,
            crater_boxes,
            stage="test",
            batch_size=batch_size,
        )
        self.log("test/loss", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        for name, value in binary_segmentation_stats(logits, labels).items():
            self.log(
                f"test/{name}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        return loss
