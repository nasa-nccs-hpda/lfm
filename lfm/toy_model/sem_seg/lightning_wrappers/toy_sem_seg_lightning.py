"""Lightning wrapper for the toy DINO semantic segmentation model."""

from __future__ import annotations

import math
from typing import Any

import torch
from lightning.pytorch import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from lfm.toy_model.sem_seg.sseg_utils import get_loss_function


def binary_segmentation_stats(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute simple foreground metrics for binary segmentation logits."""
    if logits.shape[1] == 2:
        probs = torch.softmax(logits, dim=1)[:, 1]
    else:
        probs = torch.sigmoid(logits[:, 0])
    preds = probs > 0.5
    target_fg = targets > 0

    tp = (preds & target_fg).sum().float()
    fp = (preds & ~target_fg).sum().float()
    fn = (~preds & target_fg).sum().float()
    tn = (~preds & ~target_fg).sum().float()
    eps = torch.tensor(1e-8, device=logits.device)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    pred_fg_fraction = preds.float().mean()
    gt_fg_fraction = target_fg.float().mean()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "pred_fg_fraction": pred_fg_fraction,
        "gt_fg_fraction": gt_fg_fraction,
    }


class ToySemSegLightningModule(LightningModule):
    """Lightning training wrapper for the old toy semantic segmentation model."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        loss_type: str = "focal_dice",
        learning_rate: float = 5e-5,
        weight_decay: float = 1e-3,
        max_epochs: int = 100,
        max_grad_norm: float = 1.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = get_loss_function(loss_type)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs_for_scheduler = max_epochs
        self.max_grad_norm = max_grad_norm
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[Any, ...], batch_idx: int) -> torch.Tensor:
        images, labels, *_ = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[Any, ...], batch_idx: int) -> torch.Tensor:
        images, labels, *_ = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for name, value in binary_segmentation_stats(logits, labels).items():
            self.log(f"val/{name}", value, on_step=False, on_epoch=True, prog_bar=(name == "f1"))
        return loss

    def test_step(self, batch: tuple[Any, ...], batch_idx: int) -> torch.Tensor:
        images, labels, *_ = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        for name, value in binary_segmentation_stats(logits, labels).items():
            self.log(f"test/{name}", value, on_step=False, on_epoch=True)
        return loss

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
            warmup_epochs = max(1, min(10, math.ceil(0.1 * self.max_epochs_for_scheduler)))
            warmup_epochs = min(warmup_epochs, self.max_epochs_for_scheduler - 1)
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=warmup_epochs,
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.max_epochs_for_scheduler - warmup_epochs,
                eta_min=1e-7,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
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
        self.clip_gradients(
            optimizer,
            gradient_clip_val=self.max_grad_norm,
            gradient_clip_algorithm="norm",
        )
