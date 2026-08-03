"""Toy TerraTorch object detection task for DINO Mask R-CNN experiments."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

try:
    from terratorch.tasks import ObjectDetectionTask
except ImportError as exc:  # pragma: no cover - depends on HPC environment.
    ObjectDetectionTask = None
    _TERRATORCH_IMPORT_ERROR = exc
else:
    _TERRATORCH_IMPORT_ERROR = None

_BASE_OBJECT_DETECTION_TASK = (
    ObjectDetectionTask if ObjectDetectionTask is not None else torch.nn.Module
)


def require_object_detection_task_class():
    """Return TerraTorch's ObjectDetectionTask or raise the import error."""
    if ObjectDetectionTask is None:
        raise ImportError(
            "TerraTorch is required for toy_architecture='dino-terratorch-mask-rcnn'."
        ) from _TERRATORCH_IMPORT_ERROR
    return ObjectDetectionTask


class ToyDinoTerraTorchObjectDetectionTask(_BASE_OBJECT_DETECTION_TASK):
    """TerraTorch ObjectDetectionTask adapted to the shared Toy instance batch."""

    def __init__(
        self,
        *args: Any,
        learning_rate: float = 5.0e-5,
        weight_decay: float = 1.0e-3,
        max_epochs: int = 100,
        max_grad_norm: float | None = 1.0,
        **kwargs: Any,
    ) -> None:
        if ObjectDetectionTask is None:
            raise ImportError(
                "TerraTorch is required for "
                "toy_architecture='dino-terratorch-mask-rcnn'."
            ) from _TERRATORCH_IMPORT_ERROR
        kwargs.setdefault("optimizer", None)
        kwargs.setdefault("optimizer_hparams", None)
        kwargs.setdefault("scheduler", None)
        kwargs.setdefault("scheduler_hparams", None)
        super().__init__(*args, **kwargs)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.max_epochs_for_scheduler = int(max_epochs)
        self.max_grad_norm = max_grad_norm

    def reformat_batch(self, batch: dict[str, Any], batch_size: int):
        targets = []
        for i in range(batch_size):
            masks = batch["masks"][i]
            if masks.ndim == 2:
                masks = masks[None]
            elif masks.ndim != 3:
                raise ValueError(
                    f"Expected masks to have shape (N,H,W), got {tuple(masks.shape)}"
                )
            targets.append(
                {
                    "boxes": batch["boxes"][i].to(torch.float32),
                    "labels": batch["labels"][i].to(torch.int64),
                    "masks": masks.to(torch.uint8),
                }
            )
        return targets

    def _loss_step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        images = batch["image"]
        targets = self.reformat_batch(batch, batch_size=images.shape[0])
        was_training = self.model.training
        self.model.train()
        output = self(images, targets)
        if not was_training:
            self.model.eval()
        loss_dict = output if isinstance(output, dict) else output.output
        loss = sum(loss_dict.values())
        self.log(
            f"{stage}/loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=images.shape[0],
        )
        if stage == "val":
            self.log(
                "val_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=images.shape[0],
            )
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._loss_step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._loss_step(batch, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._loss_step(batch, "test")

    def predict_step(
        self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ):
        images = batch["image"]
        was_training = self.model.training
        self.model.eval()
        output = self(images)
        self.model.train(was_training)
        return output if isinstance(output, list) else output.output

    def configure_optimizers(self):
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        if self.max_epochs_for_scheduler <= 1:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max(1, self.max_epochs_for_scheduler),
                eta_min=1e-7,
            )
        else:
            warmup_epochs = max(
                1, min(10, math.ceil(0.1 * self.max_epochs_for_scheduler))
            )
            warmup_epochs = min(warmup_epochs, self.max_epochs_for_scheduler - 1)
            scheduler = SequentialLR(
                optimizer,
                schedulers=[
                    LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
                    CosineAnnealingLR(
                        optimizer,
                        T_max=self.max_epochs_for_scheduler - warmup_epochs,
                        eta_min=1e-7,
                    ),
                ],
                milestones=[warmup_epochs],
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        if self.max_grad_norm is None:
            return
        self.clip_gradients(
            optimizer,
            gradient_clip_val=self.max_grad_norm,
            gradient_clip_algorithm="norm",
        )
