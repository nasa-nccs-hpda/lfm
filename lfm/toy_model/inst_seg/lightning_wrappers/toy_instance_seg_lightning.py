"""Lightning wrapper for toy DINO Mask2Former instance segmentation."""

from __future__ import annotations

import math
from typing import Any

import torch
from lightning.pytorch import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class ToyInstanceSegLightningModule(LightningModule):
    """Lightning wrapper around the toy DINO Mask2Former instance model."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        learning_rate: float = 5.0e-5,
        weight_decay: float = 1.0e-3,
        max_epochs: int = 100,
        max_grad_norm: float | None = 1.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs_for_scheduler = max_epochs
        self.max_grad_norm = max_grad_norm
        self.model.train()
        self.save_hyperparameters(ignore=["model"])

    def on_fit_start(self) -> None:
        self.model.train()

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        mask_labels: list[torch.Tensor] | None = None,
        class_labels: list[torch.Tensor] | None = None,
    ) -> Any:
        return self.model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )

    def _shared_loss_step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        pixel_values = batch["pixel_values"]
        batch_size = pixel_values.shape[0]
        mask_labels = [labels.to(self.device) for labels in batch["mask_labels"]]
        class_labels = [labels.to(self.device) for labels in batch["class_labels"]]
        outputs = self(
            pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )
        loss = outputs.loss
        self.log(
            f"{stage}/loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        if stage == "val":
            self.log(
                "val_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_loss_step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_loss_step(batch, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_loss_step(batch, "test")

    def predict_step(
        self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ):
        return self.model(pixel_values=batch["pixel_values"])

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
        if self.max_grad_norm is None:
            return
        self.clip_gradients(
            optimizer,
            gradient_clip_val=self.max_grad_norm,
            gradient_clip_algorithm="norm",
        )
