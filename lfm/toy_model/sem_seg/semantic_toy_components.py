"""Reusable Toy semantic segmentation fine-tuning workflow."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from lfm.all_models.all_tasks.data.normalization import (
    load_terramind_pretraining_stats,
)
from lfm.all_models.all_tasks.utils import (
    ValidationPlotCallback,
)
from lfm.toy_model.sem_seg.lightning_wrappers.toy_sem_seg_datamodule import (
    LunarSemanticSegmentationSplitDataModule,
)
from lfm.toy_model.sem_seg.lightning_wrappers.toy_sem_seg_from_instance_datamodule import (
    ToySemSegFromInstanceDataModule,
)
from lfm.toy_model.sem_seg.lightning_wrappers.toy_sem_seg_lightning import (
    ToySemSegLightningModule,
)
from lfm.toy_model.sem_seg.lightning_wrappers.toy_sem_seg_shape_lightning import (
    ToySemSegShapeLightningModule,
)
from lfm.toy_model.sem_seg.sseg_model import DINOSegmentation, load_dinov3_encoder


class FitProgressLogger(Callback):
    """Flush simple progress messages for non-interactive sbatch logs."""

    def __init__(self, model_name: str, log_every_n_batches: int = 25) -> None:
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
            print(
                f"[{self.model_name}] epoch {trainer.current_epoch + 1} "
                f"train batch {batch_idx + 1}/{trainer.num_training_batches}",
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
    if not config.normalize_inputs:
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
    output_dir: Path,
    *,
    normalization_modality_info: Path | None = None,
) -> LunarSemanticSegmentationSplitDataModule:
    datamodule_cls = (
        ToySemSegFromInstanceDataModule
        if config.semantic_label_source == "instance"
        else LunarSemanticSegmentationSplitDataModule
    )
    means, stds = _pretraining_stats(config, normalization_modality_info)
    datamodule = datamodule_cls(
        data_root=config.data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        target_size=config.target_size,
        spatial_transform=config.spatial_transform,
        image_file_type=config.image_file_type,
        image_suffix=config.image_suffix,
        label_suffix=config.label_suffix,
        band_filter=config.band_filter,
        normalize_inputs=config.normalize_inputs,
        means=means,
        stds=stds,
        scale_inputs=config.normalization_source != "pretrain",
        ignore_nodata_in_loss=getattr(config, "ignore_nodata_in_loss", False),
        nodata_ignore_index=getattr(config, "nodata_ignore_index", -1),
        excluded_nodata_values=getattr(config, "excluded_nodata_values", None),
        max_train_samples=config.max_train_samples,
        max_val_samples=config.max_val_samples,
        max_test_samples=config.max_test_samples,
        output_dir=output_dir,
    )
    datamodule.setup("fit")
    print("Data sanity summary:")
    for key, value in datamodule.get_sanity_summary().items():
        print(f"  {key}: {value}")
    return datamodule


def create_model(config: Any, weight_assignments: list[str]) -> DINOSegmentation:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.dino_checkpoint is not None:
        encoder = load_dinov3_encoder(
            weights_local_checkpoint=str(config.dino_checkpoint),
            device=device,
        )
    else:
        encoder = load_dinov3_encoder(device=device)

    return DINOSegmentation(
        encoder=encoder,
        num_classes=2,
        img_size=config.target_size,
        freeze_encoder=config.freeze_encoder,
        weight_assignments=weight_assignments,
    )


def create_lightning_module(
    config: Any,
    model: DINOSegmentation,
) -> ToySemSegLightningModule:
    module_cls = (
        ToySemSegShapeLightningModule
        if config.use_toy_shape_loss
        else ToySemSegLightningModule
    )
    kwargs = {}
    if config.use_toy_shape_loss:
        kwargs = {
            "shape_loss_weight": config.toy_shape_loss_weight,
            "shape_loss_pad_frac": config.toy_shape_loss_pad_frac,
        }
    return module_cls(
        model=model,
        loss_type=config.toy_loss_type,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_epochs=config.max_epochs,
        max_grad_norm=config.toy_gradient_clip_val,
        ignore_index=(
            getattr(config, "nodata_ignore_index", -1)
            if getattr(config, "ignore_nodata_in_loss", False)
            else None
        ),
        **kwargs,
    )


def create_trainer(
    config: Any,
    output_dir: Path,
    *,
    plots_subdir: str | Path = "plots",
    epoch_test_suite_callback_cls: type[Callback] | None = None,
) -> Trainer:
    print("Creating Lightning trainer...", flush=True)
    callbacks: list[Callback] = [
        FitProgressLogger(
            "Toy",
            log_every_n_batches=config.progress_log_every_n_batches,
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
        ValidationPlotCallback(
            output_dir=output_dir,
            n_samples=config.plot_n_samples,
            every_n_epochs=config.plot_every_n_epochs,
            plots_subdir=plots_subdir,
            display_method="minmax",
            dpi=150,
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
                ignore_index=(
                    getattr(config, "nodata_ignore_index", -1)
                    if getattr(config, "ignore_nodata_in_loss", False)
                    else None
                ),
            )
        )
    return Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="32",
        max_epochs=config.max_epochs,
        check_val_every_n_epoch=1,
        log_every_n_steps=5,
        logger=False,
        callbacks=callbacks,
    )
