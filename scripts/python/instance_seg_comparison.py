"""Run a true instance segmentation comparison between Toy DINO and Graha.

Both models use the same split instance dataset rooted at
``train/val/test/{chips,labels}``. Toy uses DINOv3 + Mask2Former; Graha uses
Lunar-FM + TerraTorch Mask R-CNN.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from argparse import Namespace
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from transformers import AutoImageProcessor

import instance_seg_finetuning as graha_workflow
from lfm.full_model.utils import (
    create_timestamped_output_dir,
    plot_instance_cache_comparison,
    plot_instance_cache_predictions,
    save_graha_instance_prediction_cache,
    save_toy_instance_prediction_cache,
)
from lfm.full_model.utils.utils import ensure_data_symlink
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


class GrahaInstancePlotCallback(Callback):
    """Save Graha instance validation plots at epoch end."""

    def __init__(
        self,
        output_dir: Path,
        *,
        n_samples: int,
        every_n_epochs: int,
        score_threshold: float,
    ) -> None:
        self.output_dir = output_dir
        self.n_samples = n_samples
        self.every_n_epochs = every_n_epochs
        self.score_threshold = score_threshold

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        if self.every_n_epochs <= 0 or (epoch + 1) % self.every_n_epochs != 0:
            return
        cache_dir = save_graha_instance_prediction_cache(
            task=pl_module,
            datamodule=trainer.datamodule,
            output_dir=self.output_dir,
            model_name="graha",
            split="val",
            n_samples=self.n_samples,
            score_threshold=self.score_threshold,
            setup_datamodule=False,
        )
        plot_instance_cache_predictions(
            cache_dir,
            self.output_dir / "plots" / "single_model" / "full_model",
            model_name="graha",
            n_samples=self.n_samples,
            filename=f"validation_epoch_{epoch + 1:03d}.png",
        )


@dataclass(frozen=True)
class InstanceComparisonConfig:
    notebook_dir: Path
    lfm_root: Path
    data_root: Path
    base_output_dir: Path
    dino_checkpoint: Path | None
    dino_lightning_checkpoint: Path | None
    graha_pretrain_dir: Path | None
    graha_lightning_checkpoint: Path | None
    toy_architecture: str
    target_size: int
    band_filter: list[int]
    max_train_samples: int | None
    max_val_samples: int | None
    max_test_samples: int | None
    toy_batch_size: int
    toy_num_workers: int
    graha_stats_batch_size: int
    graha_batch_size: int
    graha_num_workers: int
    max_epochs: int
    toy_learning_rate: float
    toy_weight_decay: float
    toy_freeze_backbone: bool
    toy_normalize_inputs: bool
    toy_gradient_clip_val: float | None
    graha_backbone_lr: float
    graha_head_lr: float
    graha_layer_decay: float
    graha_weight_decay: float
    graha_warmup_steps: int
    graha_anchor_sizes: list[list[int]]
    graha_anchor_aspect_ratios: list[float]
    graha_score_threshold: float
    plot_every_n_epochs: int
    plot_n_samples: int
    progress_log_every_n_batches: int
    prediction_split: str
    prediction_n_samples: int
    prediction_score_threshold: float
    mask_shift: tuple[int, int]
    skip_toy_fit: bool
    skip_graha_fit: bool
    seed: int


def build_config(args: argparse.Namespace) -> InstanceComparisonConfig:
    script_dir = Path(__file__).resolve().parent
    lfm_root = script_dir.parents[1]
    notebook_dir = lfm_root / "notebooks" / "full_model"
    scripts_output_dir = lfm_root / "scripts" / "outputs"
    return InstanceComparisonConfig(
        notebook_dir=notebook_dir,
        lfm_root=lfm_root,
        data_root=Path(args.data_root).resolve() if args.data_root else notebook_dir / "data",
        base_output_dir=(
            Path(args.base_output_dir).resolve()
            if args.base_output_dir
            else scripts_output_dir / "instance_seg_comparison"
        ),
        dino_checkpoint=Path(args.dino_checkpoint).resolve() if args.dino_checkpoint else None,
        dino_lightning_checkpoint=(
            Path(args.dino_lightning_checkpoint).resolve()
            if args.dino_lightning_checkpoint
            else None
        ),
        graha_pretrain_dir=Path(args.graha_pretrain_dir).resolve()
        if args.graha_pretrain_dir
        else None,
        graha_lightning_checkpoint=(
            Path(args.graha_lightning_checkpoint).resolve()
            if args.graha_lightning_checkpoint
            else None
        ),
        toy_architecture=args.toy_architecture,
        target_size=args.target_size,
        band_filter=args.band_filter,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        toy_batch_size=args.toy_batch_size,
        toy_num_workers=args.toy_num_workers,
        graha_stats_batch_size=args.graha_stats_batch_size,
        graha_batch_size=args.graha_batch_size,
        graha_num_workers=args.graha_num_workers,
        max_epochs=args.max_epochs,
        toy_learning_rate=args.toy_learning_rate,
        toy_weight_decay=args.toy_weight_decay,
        toy_freeze_backbone=args.toy_freeze_backbone,
        toy_normalize_inputs=args.toy_normalize_inputs,
        toy_gradient_clip_val=None
        if args.disable_toy_gradient_clipping
        else args.toy_gradient_clip_val,
        graha_backbone_lr=args.graha_backbone_lr,
        graha_head_lr=args.graha_head_lr,
        graha_layer_decay=args.graha_layer_decay,
        graha_weight_decay=args.graha_weight_decay,
        graha_warmup_steps=args.graha_warmup_steps,
        graha_anchor_sizes=args.graha_anchor_sizes,
        graha_anchor_aspect_ratios=args.graha_anchor_aspect_ratios,
        graha_score_threshold=args.graha_score_threshold,
        plot_every_n_epochs=args.plot_every_n_epochs,
        plot_n_samples=args.plot_n_samples,
        progress_log_every_n_batches=args.progress_log_every_n_batches,
        prediction_split=args.prediction_split,
        prediction_n_samples=args.prediction_n_samples,
        prediction_score_threshold=args.prediction_score_threshold,
        mask_shift=tuple(args.mask_shift),
        skip_toy_fit=args.no_fit or args.skip_toy_fit,
        skip_graha_fit=args.no_fit or args.skip_graha_fit,
        seed=args.seed,
    )


def validate_paths(config: InstanceComparisonConfig) -> None:
    required = []
    for split in ("train", "val", "test"):
        required.extend(
            [
                config.data_root / split / "chips",
                config.data_root / split / "labels",
            ]
        )
    for path in [
        config.dino_checkpoint,
        config.dino_lightning_checkpoint,
        config.graha_pretrain_dir,
        config.graha_lightning_checkpoint,
    ]:
        if path is not None:
            required.append(path)
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required paths:\n" + "\n".join(str(path) for path in missing)
        )


def save_config(config: InstanceComparisonConfig, output_dir: Path) -> None:
    def encode(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return list(value)
        return value

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump({key: encode(value) for key, value in asdict(config).items()}, f, indent=2)


def load_lightning_checkpoint_state(module: torch.nn.Module, checkpoint_path: Path, model_name: str) -> None:
    print(f"Loading {model_name} Lightning checkpoint weights from {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    module.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
    print(f"Loaded {model_name} Lightning checkpoint weights.", flush=True)


def create_toy_datamodule(config: InstanceComparisonConfig):
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
        band_filter=config.band_filter,
        normalize_inputs=config.toy_normalize_inputs,
        mask_shift=config.mask_shift,
        max_train_samples=config.max_train_samples,
        max_val_samples=config.max_val_samples,
        max_test_samples=config.max_test_samples,
    )
    datamodule.setup("fit")
    if datamodule.weight_assignments is None:
        raise RuntimeError("Toy datamodule did not infer weight assignments.")
    return datamodule


def create_toy_task(config: InstanceComparisonConfig, weight_assignments: list[str]):
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


def create_toy_image_processor(config: InstanceComparisonConfig):
    if config.toy_architecture == "dino-mask-rcnn":
        return None
    return AutoImageProcessor.from_pretrained(
        "facebook/mask2former-swin-large-coco-instance",
        do_resize=True,
        size={"height": config.target_size, "width": config.target_size},
        do_normalize=False,
        do_reduce_labels=False,
    )


def create_toy_trainer(
    config: InstanceComparisonConfig,
    output_dir: Path,
    image_processor,
) -> Trainer:
    return Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="32",
        max_epochs=config.max_epochs,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        log_every_n_steps=5,
        logger=False,
        callbacks=[
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
                save_last=True,
                every_n_epochs=1,
            ),
        ],
    )


def run_toy(config: InstanceComparisonConfig, output_dir: Path) -> Path | None:
    title = (
        "Toy DINO Mask R-CNN"
        if config.toy_architecture == "dino-mask-rcnn"
        else "Toy DINO Mask2Former"
    )
    print(f"\n=== {title} instance segmentation ===", flush=True)
    started = time.perf_counter()
    seed_everything(config.seed)
    datamodule = create_toy_datamodule(config)
    task = create_toy_task(config, datamodule.weight_assignments or [])
    image_processor = create_toy_image_processor(config)
    trainer = create_toy_trainer(config, output_dir, image_processor)
    prediction_cache = None

    if config.skip_toy_fit:
        print("Skipping Toy trainer.fit().", flush=True)
        if config.dino_lightning_checkpoint is not None:
            load_lightning_checkpoint_state(task, config.dino_lightning_checkpoint, "Toy")
    else:
        ckpt_path = (
            str(config.dino_lightning_checkpoint)
            if config.dino_lightning_checkpoint is not None
            else None
        )
        if ckpt_path is not None:
            print(f"Resuming Toy trainer.fit() from {ckpt_path}", flush=True)
        print("Starting Toy trainer.fit()...", flush=True)
        trainer.fit(task, datamodule=datamodule, ckpt_path=ckpt_path)
        print("Finished Toy trainer.fit().", flush=True)
        toy_checkpoints = sorted((output_dir / "checkpoints" / "toy_model").glob("*.ckpt"))
        print(f"[Toy] saved {len(toy_checkpoints)} checkpoint file(s).", flush=True)

    if image_processor is None:
        prediction_cache = save_graha_instance_prediction_cache(
            task=task,
            datamodule=datamodule,
            output_dir=output_dir,
            model_name="toy",
            split=config.prediction_split,
            n_samples=config.prediction_n_samples,
            score_threshold=config.prediction_score_threshold,
        )
    else:
        prediction_cache = save_toy_instance_prediction_cache(
            task=task,
            datamodule=datamodule,
            output_dir=output_dir,
            image_processor=image_processor,
            model_name="toy",
            split=config.prediction_split,
            n_samples=config.prediction_n_samples,
            score_threshold=config.prediction_score_threshold,
        )
    plot_path = plot_instance_cache_predictions(
        prediction_cache,
        output_dir / "plots" / "single_model" / "toy_model",
        model_name="toy",
        n_samples=config.prediction_n_samples,
        filename=f"{config.prediction_split}_instance_predictions.png",
    )
    print(f"[Toy] saved validation prediction plot: {plot_path}", flush=True)

    elapsed = time.perf_counter() - started
    print(f"Toy elapsed seconds: {elapsed:.3f}", flush=True)
    del trainer, task, datamodule
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prediction_cache


def build_graha_config(config: InstanceComparisonConfig, output_dir: Path):
    args = Namespace(
        simlink_dest=None,
        data_root=str(config.data_root),
        base_output_dir=str(output_dir),
        pretrain_dir=str(config.graha_pretrain_dir) if config.graha_pretrain_dir else None,
        lightning_checkpoint=str(config.graha_lightning_checkpoint)
        if config.graha_lightning_checkpoint
        else None,
        crop_size=config.target_size,
        stats_batch_size=config.graha_stats_batch_size,
        batch_size=config.graha_batch_size,
        num_workers=config.graha_num_workers,
        max_epochs=config.max_epochs,
        backbone_lr=config.graha_backbone_lr,
        head_lr=config.graha_head_lr,
        layer_decay=config.graha_layer_decay,
        weight_decay=config.graha_weight_decay,
        warmup_steps=config.graha_warmup_steps,
        anchor_sizes=config.graha_anchor_sizes,
        anchor_aspect_ratios=config.graha_anchor_aspect_ratios,
        score_threshold=config.graha_score_threshold,
        plot_predictions=True,
        prediction_split=config.prediction_split,
        prediction_n_samples=config.prediction_n_samples,
        prediction_score_threshold=config.prediction_score_threshold,
        progress_log_every_n_batches=config.progress_log_every_n_batches,
        mask_shift=config.mask_shift,
        seed=config.seed,
        no_fit=config.skip_graha_fit,
        loss_smoke_only=False,
    )
    return graha_workflow.build_config(args)


def run_graha(config: InstanceComparisonConfig, output_dir: Path) -> Path | None:
    print("\n=== Graha/Lunar-FM Mask R-CNN instance segmentation ===", flush=True)
    started = time.perf_counter()
    graha_workflow.configure_proj_environment()
    graha_config = build_graha_config(config, output_dir)
    graha_workflow.configure_python_paths(graha_config)
    graha_workflow.print_config(graha_config)
    graha_workflow.validate_required_paths(graha_config)

    deps = graha_workflow.import_project_dependencies()
    datamodule_cls = deps["LunarObjectDetectionInstanceSegmentationDatamodule"]
    task_cls = graha_workflow.make_notebook_object_detection_task_class(
        deps["LunarObjectDetectionTask"]
    )

    seed_everything(graha_config.seed)
    means, stds = graha_workflow.calculate_train_stats(graha_config, datamodule_cls)
    datamodule = graha_workflow.create_datamodule(graha_config, datamodule_cls, means, stds)
    sample_batch = graha_workflow.inspect_batch(datamodule)
    task = graha_workflow.create_task(graha_config, task_cls, sample_batch)
    graha_workflow.run_loss_smoke(task, sample_batch)
    trainer = graha_workflow.create_trainer(graha_config, output_dir)
    trainer.callbacks.append(
        GrahaInstancePlotCallback(
            output_dir,
            n_samples=config.plot_n_samples,
            every_n_epochs=config.plot_every_n_epochs,
            score_threshold=config.prediction_score_threshold,
        )
    )
    prediction_cache = None

    if config.skip_graha_fit:
        print("Skipping Graha trainer.fit().", flush=True)
        if config.graha_lightning_checkpoint is not None:
            graha_workflow.load_lightning_checkpoint_state(
                task,
                config.graha_lightning_checkpoint,
            )
    else:
        ckpt_path = (
            str(config.graha_lightning_checkpoint)
            if config.graha_lightning_checkpoint is not None
            else None
        )
        if ckpt_path is not None:
            print(f"Resuming Graha trainer.fit() from {ckpt_path}", flush=True)
        print("Starting Graha trainer.fit()...", flush=True)
        trainer.fit(task, datamodule=datamodule, ckpt_path=ckpt_path)
        print("Finished Graha trainer.fit().", flush=True)
        graha_checkpoints = sorted((output_dir / "checkpoints" / "full_model").glob("*.ckpt"))
        print(f"[Graha] saved {len(graha_checkpoints)} checkpoint file(s).", flush=True)

    if graha_config.plot_predictions:
        prediction_cache = save_graha_instance_prediction_cache(
            task=task,
            datamodule=datamodule,
            output_dir=output_dir,
            model_name="graha",
            split=config.prediction_split,
            n_samples=config.prediction_n_samples,
            score_threshold=config.prediction_score_threshold,
        )
        plot_path = plot_instance_cache_predictions(
            prediction_cache,
            output_dir / "plots" / "single_model" / "full_model",
            model_name="graha",
            n_samples=config.prediction_n_samples,
            filename=f"{config.prediction_split}_instance_predictions.png",
        )
        print(f"[Graha] saved validation prediction plot: {plot_path}", flush=True)

    elapsed = time.perf_counter() - started
    print(f"Graha elapsed seconds: {elapsed:.3f}", flush=True)
    del trainer, task, datamodule, sample_batch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prediction_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simlink-dest", "--symlink-dest", dest="simlink_dest", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--base-output-dir", type=str, default=None)
    parser.add_argument("--dino-checkpoint", type=str, default=None)
    parser.add_argument("--dino-lightning-checkpoint", type=str, default=None)
    parser.add_argument("--graha-pretrain-dir", type=str, default=None)
    parser.add_argument("--graha-lightning-checkpoint", type=str, default=None)
    parser.add_argument(
        "--toy-architecture",
        choices=["mask2former", "dino-mask-rcnn"],
        default="mask2former",
        help="Toy instance head to train. Use dino-mask-rcnn for a tighter Mask R-CNN comparison.",
    )
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--band-filter", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--toy-batch-size", type=int, default=2)
    parser.add_argument("--toy-num-workers", type=int, default=10)
    parser.add_argument("--graha-stats-batch-size", type=int, default=16)
    parser.add_argument("--graha-batch-size", type=int, default=2)
    parser.add_argument("--graha-num-workers", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--toy-learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--toy-weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--toy-freeze-backbone", action="store_true")
    parser.add_argument("--toy-normalize-inputs", action="store_true")
    parser.add_argument("--toy-gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--disable-toy-gradient-clipping", action="store_true")
    parser.add_argument("--graha-backbone-lr", type=float, default=5.0e-5)
    parser.add_argument("--graha-head-lr", type=float, default=2.0e-4)
    parser.add_argument("--graha-layer-decay", type=float, default=0.75)
    parser.add_argument("--graha-weight-decay", type=float, default=0.05)
    parser.add_argument("--graha-warmup-steps", type=int, default=500)
    parser.add_argument(
        "--graha-anchor-sizes",
        type=lambda value: [[int(x)] for x in value.split(",")],
        default=[[8], [16], [32], [64]],
    )
    parser.add_argument(
        "--graha-anchor-aspect-ratios",
        type=lambda value: [float(x) for x in value.split(",")],
        default=[0.5, 1.0, 2.0],
    )
    parser.add_argument("--graha-score-threshold", type=float, default=0.5)
    parser.add_argument("--plot-every-n-epochs", type=int, default=1)
    parser.add_argument("--plot-n-samples", type=int, default=5)
    parser.add_argument(
        "--progress-log-every-n-batches",
        type=int,
        default=25,
        help="Flush train-batch progress every N batches in sbatch logs.",
    )
    parser.add_argument("--prediction-split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--prediction-n-samples", type=int, default=5)
    parser.add_argument("--prediction-score-threshold", type=float, default=0.5)
    parser.add_argument("--mask-shift", type=int, nargs=2, default=(0, 0))
    parser.add_argument("--skip-toy-fit", action="store_true")
    parser.add_argument("--skip-graha-fit", action="store_true")
    parser.add_argument("--no-fit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    notebook_dir = Path(__file__).resolve().parents[1] / "notebooks" / "full_model"
    ensure_data_symlink(args.simlink_dest, notebook_dir / "data")
    config = build_config(args)
    validate_paths(config)
    output_dir = create_timestamped_output_dir(config.base_output_dir)
    (output_dir / "checkpoints" / "toy_model").mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints" / "full_model").mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir)

    toy_prediction_cache = run_toy(config, output_dir)
    graha_prediction_cache = run_graha(config, output_dir)

    if toy_prediction_cache is not None and graha_prediction_cache is not None:
        plot_instance_cache_comparison(
            {
                "toy": toy_prediction_cache,
                "graha": graha_prediction_cache,
            },
            output_dir / "plots" / "comparison",
            n_samples=config.prediction_n_samples,
        )

    elapsed = time.perf_counter() - started
    with (output_dir / "timing_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"seconds": round(elapsed, 3)}, f, indent=2)
    print(f"Comparison elapsed seconds: {elapsed:.3f}", flush=True)


if __name__ == "__main__":
    main()
