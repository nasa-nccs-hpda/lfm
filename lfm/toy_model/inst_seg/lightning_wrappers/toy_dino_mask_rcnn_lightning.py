"""Lightning wrapper for Toy Mask R-CNN instance segmentation."""

from __future__ import annotations

import math
from typing import Any

import torch
from lightning.pytorch import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from lfm.all_models.inst_seg.mask_ap import MaskAPValidationMixin, mask_ap_inference
from lfm.all_models.inst_seg.mask_rcnn_nodata import (
    encode_nodata_in_instance_masks,
    use_nodata_aware_maskrcnn_loss,
)


class ToyDinoMaskRCNNLightningModule(MaskAPValidationMixin, LightningModule):
    """Lightning wrapper around a TorchVision Mask R-CNN model."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        learning_rate: float = 5.0e-5,
        weight_decay: float = 1.0e-3,
        max_epochs: int = 100,
        max_grad_norm: float | None = 1.0,
        ignore_nodata_in_loss: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs_for_scheduler = max_epochs
        self.max_grad_norm = max_grad_norm
        self.ignore_nodata_in_loss = ignore_nodata_in_loss
        self.save_hyperparameters(ignore=["model"])

    def _images_and_targets(self, batch: dict[str, Any]):
        images = [image.to(self.device) for image in batch["image"]]
        targets = []
        if self.ignore_nodata_in_loss and "valid_mask" not in batch:
            raise KeyError(
                "ignore_nodata_in_loss=True requires 'valid_mask' in the batch."
            )
        for index, (boxes, labels, masks) in enumerate(
            zip(batch["boxes"], batch["labels"], batch["masks"])
        ):
            masks = masks.to(self.device, dtype=torch.uint8)
            if self.ignore_nodata_in_loss:
                masks = encode_nodata_in_instance_masks(
                    masks, batch["valid_mask"][index]
                )
            targets.append(
                {
                    "boxes": boxes.to(self.device, dtype=torch.float32),
                    "labels": labels.to(self.device, dtype=torch.int64),
                    "masks": masks,
                }
            )
        return images, targets

    def forward(self, images, targets=None):
        if isinstance(images, torch.Tensor):
            images = [image for image in images]
        with use_nodata_aware_maskrcnn_loss(
            self.ignore_nodata_in_loss and self.model.training and targets is not None
        ):
            return self.model(images, targets)

    def _loss_step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        images, targets = self._images_and_targets(batch)
        batch_size = len(images)
        was_training = self.model.training
        self.model.train()
        try:
            losses = self(images, targets)
        finally:
            self.model.train(was_training)
        loss = sum(value for value in losses.values())
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
        return self._loss_step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss = self._loss_step(batch, "val")
        with mask_ap_inference(self.model):
            predictions = self([image.to(self.device) for image in batch["image"]])
        self._validation_mask_ap.update(predictions, batch)
        return loss

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._loss_step(batch, "test")

    def predict_step(
        self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ):
        images = [image.to(self.device) for image in batch["image"]]
        was_training = self.model.training
        self.model.eval()
        predictions = self.model(images)
        self.model.train(was_training)
        return predictions

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
