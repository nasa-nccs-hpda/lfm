"""Reusable Toy true instance segmentation fine-tuning workflow."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from transformers import AutoImageProcessor

from lfm.all_models.all_tasks.data.normalization import (
    load_terramind_pretraining_stats,
)
from lfm.full_model.all_tasks.utils import (
    plot_instance_cache_predictions,
    save_graha_instance_prediction_cache,
    save_toy_instance_prediction_cache,
)
from lfm.toy_model.inst_seg.dino_mask_rcnn_model import create_dino_mask_rcnn_model
from lfm.toy_model.inst_seg.iseg_model import (
    create_mask2former_dinov3_model,
    load_dinov3_encoder,
)
from lfm.toy_model.inst_seg.lightning_wrappers import (
    ToyDinoMaskRCNNLightningModule,
    ToyDinoMaskRCNNSplitDataModule,
    ToyInstanceSegLightningModule,
    ToyInstanceSegSplitDataModule,
)


class FitProgressLogger(Callback):
    """Flush simple progress messages for non-interactive sbatch logs."""

    def __init__(self, model_name: str, log_every_n_batches: int = 5) -> None:
        self.model_name = model_name
        self.log_every_n_batches = max(1, log_every_n_batches)
        self._epoch_started_at: float | None = None

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._epoch_started_at = time.perf_counter()
        print(
            f"[{self.model_name}] train epoch {trainer.current_epoch + 1}/"
            f"{trainer.max_epochs} started",
            flush=True,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if batch_idx == 0 or (batch_idx + 1) % self.log_every_n_batches == 0:
            total = trainer.num_training_batches
            print(
                f"[{self.model_name}] epoch {trainer.current_epoch + 1} "
                f"train batch {batch_idx + 1}/{total}",
                flush=True,
            )

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        elapsed = (
            time.perf_counter() - self._epoch_started_at
            if self._epoch_started_at is not None
            else 0.0
        )
        print(
            f"[{self.model_name}] train epoch {trainer.current_epoch + 1} "
            f"finished in {elapsed:.1f}s",
            flush=True,
        )

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        print(f"[{self.model_name}] validation started", flush=True)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        print(f"[{self.model_name}] validation finished", flush=True)


class ToyInstancePlotCallback(Callback):
    """Save Toy instance validation plots at epoch end."""

    def __init__(
        self,
        output_dir: Path,
        image_processor,
        *,
        n_samples: int,
        every_n_epochs: int,
        score_threshold: float,
    ) -> None:
        self.output_dir = output_dir
        self.image_processor = image_processor
        self.n_samples = n_samples
        self.every_n_epochs = every_n_epochs
        self.score_threshold = score_threshold

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        if self.every_n_epochs <= 0 or (epoch + 1) % self.every_n_epochs != 0:
            return
        if self.image_processor is None:
            cache_dir = save_graha_instance_prediction_cache(
                task=pl_module,
                datamodule=trainer.datamodule,
                output_dir=self.output_dir,
                model_name="toy",
                split="val",
                n_samples=self.n_samples,
                score_threshold=self.score_threshold,
                setup_datamodule=False,
            )
        else:
            cache_dir = save_toy_instance_prediction_cache(
                task=pl_module,
                datamodule=trainer.datamodule,
                output_dir=self.output_dir,
                image_processor=self.image_processor,
                model_name="toy",
                split="val",
                n_samples=self.n_samples,
                score_threshold=self.score_threshold,
                setup_datamodule=False,
            )
        plot_instance_cache_predictions(
            cache_dir,
            self.output_dir / "plots" / "single_model" / "toy_model",
            model_name="toy",
            n_samples=self.n_samples,
            filename=f"validation_epoch_{epoch + 1:03d}.png",
        )


def load_lightning_checkpoint_state(
    module: torch.nn.Module, checkpoint_path: Path, model_name: str = "Toy"
) -> None:
    print(
        f"Loading {model_name} Lightning checkpoint weights from {checkpoint_path}",
        flush=True,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    module.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
    print(f"Loaded {model_name} Lightning checkpoint weights.", flush=True)


def _pretraining_stats(config: Any, modality_info: Path | None):
    if not config.toy_normalize_inputs:
        return None, None
    if config.normalization_source != "pretrain":
        if config.normalization_source == "finetune":
            return None, None
        raise ValueError(
            f"Unsupported normalization_source: {config.normalization_source}"
        )
    if modality_info is None:
        raise ValueError("modality_info is required for pretrain normalization.")
    return load_terramind_pretraining_stats(
        modality_info,
        normalization_modality=config.normalization_modality,
        band_filter=config.band_filter,
    )


def create_datamodule(
    config: Any,
    *,
    normalization_modality_info: Path | None = None,
):
    means, stds = _pretraining_stats(config, normalization_modality_info)
    datamodule_cls = (
        ToyDinoMaskRCNNSplitDataModule
        if config.toy_architecture == "dino-mask-rcnn"
        else ToyInstanceSegSplitDataModule
    )
    datamodule = datamodule_cls(
        data_root=config.data_root,
        batch_size=config.toy_batch_size,
        num_workers=config.toy_num_workers,
        target_size=config.target_size,
        image_glob=config.image_glob,
        label_glob=config.label_glob,
        image_suffix=config.image_suffix,
        label_suffix=config.label_suffix,
        band_filter=config.band_filter,
        normalize_inputs=config.toy_normalize_inputs,
        means=means,
        stds=stds,
        scale_inputs=config.normalization_source != "pretrain",
        mask_shift=config.mask_shift,
        no_data_replace=getattr(config, "no_data_replace", None),
        no_label_replace=getattr(config, "no_label_replace", None),
        ignore_nodata_in_loss=getattr(config, "ignore_nodata_in_loss", False),
        nodata_ignore_index=getattr(config, "nodata_ignore_index", -1),
        max_train_samples=config.max_train_samples,
        max_val_samples=config.max_val_samples,
        max_test_samples=config.max_test_samples,
    )
    datamodule.setup("fit")
    if datamodule.weight_assignments is None:
        raise RuntimeError("Toy datamodule did not infer weight assignments.")
    return datamodule


def create_task(config: Any, weight_assignments: list[str]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.dino_checkpoint is not None:
        encoder = load_dinov3_encoder(
            weights_local_checkpoint=str(config.dino_checkpoint),
            device=device,
        )
    else:
        encoder = load_dinov3_encoder(device=device)
    if config.toy_architecture == "dino-mask-rcnn":
        model = create_dino_mask_rcnn_model(
            encoder=encoder,
            num_bands=len(weight_assignments),
            target_size=config.target_size,
            weight_assignments=weight_assignments,
            freeze_backbone=config.toy_freeze_backbone,
            anchor_sizes=config.graha_anchor_sizes,
            anchor_aspect_ratios=config.graha_anchor_aspect_ratios,
        ).to(device)
        return ToyDinoMaskRCNNLightningModule(
            model=model,
            learning_rate=config.toy_learning_rate,
            weight_decay=config.toy_weight_decay,
            max_epochs=config.max_epochs,
            max_grad_norm=config.toy_gradient_clip_val,
        )
    model = create_mask2former_dinov3_model(
        encoder=encoder,
        freeze_backbone=config.toy_freeze_backbone,
        num_bands=len(weight_assignments),
        device=str(device),
        weight_assignments=weight_assignments,
    )
    return ToyInstanceSegLightningModule(
        model=model,
        learning_rate=config.toy_learning_rate,
        weight_decay=config.toy_weight_decay,
        max_epochs=config.max_epochs,
        max_grad_norm=config.toy_gradient_clip_val,
    )


def create_image_processor(config: Any):
    if config.toy_architecture == "dino-mask-rcnn":
        return None
    return AutoImageProcessor.from_pretrained(
        "facebook/mask2former-swin-large-coco-instance",
        do_resize=True,
        size={"height": config.target_size, "width": config.target_size},
        do_normalize=False,
        do_reduce_labels=False,
    )


def create_trainer(
    config: Any,
    output_dir: Path,
    image_processor,
    *,
    epoch_test_suite_callback_cls: type[Callback] | None = None,
) -> Trainer:
    callbacks: list[Callback] = [
        FitProgressLogger(
            "Toy",
            log_every_n_batches=config.progress_log_every_n_batches,
        ),
        ToyInstancePlotCallback(
            output_dir,
            image_processor,
            n_samples=config.plot_n_samples,
            every_n_epochs=config.plot_every_n_epochs,
            score_threshold=config.prediction_score_threshold,
        ),
        ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints" / "toy_model"),
            monitor="val_loss",
            mode="min",
            filename="model-epoch-{epoch:02d}-val-loss={val_loss:.3f}",
            auto_insert_metric_name=False,
            save_top_k=-1,
            save_last=False,
            save_weights_only=True,
            every_n_epochs=1,
        ),
    ]
    if config.run_epoch_test_suite:
        if epoch_test_suite_callback_cls is None:
            raise ValueError("epoch_test_suite_callback_cls is required.")
        callbacks.append(
            epoch_test_suite_callback_cls(
                output_dir=output_dir,
                model_name="toy_model",
                split=config.epoch_test_split,
                n_samples=config.epoch_test_n_samples,
                every_n_epochs=config.epoch_test_every_n_epochs,
                score_threshold=config.prediction_score_threshold,
                image_processor=image_processor,
            )
        )
    return Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="32",
        max_epochs=config.max_epochs,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        log_every_n_steps=5,
        logger=False,
        callbacks=callbacks,
    )
