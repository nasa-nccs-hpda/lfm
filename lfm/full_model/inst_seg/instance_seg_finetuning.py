"""Direct Graha/Lunar-FM true instance segmentation fine-tuning workflow.

This is the reusable implementation behind ``instance_seg_finetuning.py`` and
``instance_seg_finetuning.ipynb``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from lfm.full_model.all_tasks.utils import create_timestamped_output_dir, plot_instance_predictions
from lfm.full_model.all_tasks.utils import load_terramind_wac_pretraining_stats
from lfm.full_model.all_tasks.utils.utils import ensure_data_symlink


class FitProgressLogger(Callback):
    """Flush simple progress messages for non-interactive sbatch logs."""

    def __init__(self, model_name: str = "Graha", log_every_n_batches: int = 5) -> None:
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
        if not trainer.sanity_checking:
            print(f"[{self.model_name}] validation started", flush=True)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not trainer.sanity_checking:
            print(f"[{self.model_name}] validation finished", flush=True)


@dataclass(frozen=True)
class InstanceFineTuningConfig:
    package_dir: Path
    notebook_dir: Path
    lfm_root: Path
    graha_root: Path
    pretrain_dir: Path
    backbone_weights: Path
    backbone_cfg: Path
    modality_info: Path
    data_root: Path
    base_output_dir: Path
    lightning_checkpoint: Path | None
    normalized_wac_data_range: list[float]
    graha_wac_mode: str
    graha_vis_uv_merge_method: str
    normalization_source: str
    crop_size: int
    stats_batch_size: int
    batch_size: int
    num_workers: int
    max_epochs: int
    backbone_lr: float
    head_lr: float
    layer_decay: float
    weight_decay: float
    warmup_steps: int
    anchor_sizes: list[list[int]]
    anchor_aspect_ratios: list[float]
    score_threshold: float
    plot_predictions: bool
    prediction_split: str
    prediction_n_samples: int
    prediction_score_threshold: float
    progress_log_every_n_batches: int
    mask_shift: tuple[int, int]
    seed: int


def configure_proj_environment() -> None:
    """Point PROJ/GDAL at the active conda environment before rasterio imports."""
    conda_prefix = Path(sys.executable).parents[1]
    for candidate in [
        conda_prefix / "share" / "proj",
        conda_prefix / "Library" / "share" / "proj",
    ]:
        if (candidate / "proj.db").exists():
            proj_dir = candidate
            break
    else:
        raise FileNotFoundError(f"No proj.db found under {conda_prefix}")

    os.environ["PROJ_LIB"] = str(proj_dir)
    os.environ["PROJ_DATA"] = str(proj_dir)
    os.environ["GDAL_DATA"] = str(conda_prefix / "share" / "gdal")
    print("PROJ_DATA =", os.environ["PROJ_DATA"])
    print("GDAL_DATA =", os.environ["GDAL_DATA"])


def build_config(args: argparse.Namespace) -> InstanceFineTuningConfig:
    package_dir = Path(__file__).resolve().parent
    full_model_dir = package_dir.parent
    lfm_root = package_dir.parents[2]
    notebook_dir = lfm_root / "notebooks" / "full_model"
    graha_root = full_model_dir / "graha-lunar-fm"
    scripts_output_dir = lfm_root / "scripts" / "outputs"

    pretrain_dir = Path(
        "/explore/nobackup/projects/lfm/gabby/Lunar-FM/experiments/"
        "lunarfm_base_dual_full_nas_no_nans_256_256_lr1e-4_wd0.05"
    ).resolve()
    if args.pretrain_dir is not None:
        pretrain_dir = Path(args.pretrain_dir).resolve()

    data_root = Path(args.data_root).resolve() if args.data_root else notebook_dir / "data"
    base_output_dir = (
        Path(args.base_output_dir).resolve()
        if args.base_output_dir
        else scripts_output_dir / "instance_seg_finetuning"
    )
    lightning_checkpoint = (
        Path(args.lightning_checkpoint).resolve() if args.lightning_checkpoint else None
    )

    return InstanceFineTuningConfig(
        package_dir=package_dir,
        notebook_dir=notebook_dir,
        lfm_root=lfm_root,
        graha_root=graha_root,
        pretrain_dir=pretrain_dir,
        backbone_weights=pretrain_dir / "checkpoints/checkpoint_weights_final.pt",
        backbone_cfg=pretrain_dir / "full_config.yaml",
        modality_info=pretrain_dir / "modality_info.yaml",
        data_root=data_root,
        base_output_dir=base_output_dir,
        lightning_checkpoint=lightning_checkpoint,
        normalized_wac_data_range=[-1.0, 1.0],
        graha_wac_mode=args.graha_wac_mode,
        graha_vis_uv_merge_method=args.graha_vis_uv_merge_method,
        normalization_source=getattr(args, "normalization_source", "pretrain"),
        crop_size=args.crop_size,
        stats_batch_size=args.stats_batch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        layer_decay=args.layer_decay,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        anchor_sizes=args.anchor_sizes,
        anchor_aspect_ratios=args.anchor_aspect_ratios,
        score_threshold=args.score_threshold,
        plot_predictions=args.plot_predictions,
        prediction_split=args.prediction_split,
        prediction_n_samples=args.prediction_n_samples,
        prediction_score_threshold=args.prediction_score_threshold,
        progress_log_every_n_batches=args.progress_log_every_n_batches,
        mask_shift=tuple(args.mask_shift),
        seed=args.seed,
    )


def configure_python_paths(config: InstanceFineTuningConfig) -> None:
    for path in [config.graha_root, config.lfm_root]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def print_config(config: InstanceFineTuningConfig) -> None:
    print("Package directory:", config.package_dir)
    print("Notebook directory:", config.notebook_dir)
    print("Graha/Lunar-FM code root:", config.graha_root)
    print("Data root:", config.data_root)
    print("Backbone weights:", config.backbone_weights)
    print("Backbone config:", config.backbone_cfg)
    print("Modality info:", config.modality_info)
    print("Base output directory:", config.base_output_dir)
    print("Lightning checkpoint:", config.lightning_checkpoint)
    print("Normalized WAC modality data_range:", config.normalized_wac_data_range)


def validate_required_paths(config: InstanceFineTuningConfig) -> None:
    required_paths = [
        config.graha_root,
        config.backbone_weights,
        config.backbone_cfg,
        config.modality_info,
        config.data_root,
        config.data_root / "train" / "chips",
        config.data_root / "train" / "labels",
        config.data_root / "val" / "chips",
        config.data_root / "val" / "labels",
    ]
    if config.lightning_checkpoint is not None:
        required_paths.append(config.lightning_checkpoint)
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required paths:\n" + "\n".join(str(path) for path in missing)
        )


def import_project_dependencies() -> dict[str, Any]:
    import terratorch_integration  # noqa: F401
    from terratorch_integration.lunar_object_detection_task import LunarObjectDetectionTask

    from lfm.full_model.inst_seg.instance_mask_datamodule import (
        LunarObjectDetectionInstanceMaskDatamodule,
    )

    return {
        "LunarObjectDetectionTask": LunarObjectDetectionTask,
        "LunarObjectDetectionInstanceMaskDatamodule": (
            LunarObjectDetectionInstanceMaskDatamodule
        ),
    }


def make_notebook_object_detection_task_class(lunar_object_detection_task_cls):
    """Create a task subclass that handles zero-instance Mask R-CNN targets."""

    class NotebookLunarObjectDetectionTask(lunar_object_detection_task_cls):
        def reformat_batch(self, batch: Any, batch_size: int):
            y = []
            has_masks = (
                "masks" in batch
                or "mask" in batch
                or self.masks_field in batch
            )
            for i in range(batch_size):
                target = {
                    "boxes": batch[self.boxes_field][i],
                    "labels": batch[self.labels_field][i],
                }
                if has_masks:
                    masks = batch[self.masks_field][i]
                    if masks.ndim == 2:
                        masks = masks[None]
                    elif masks.ndim != 3:
                        raise ValueError(
                            f"Expected masks to have shape (N,H,W), got {tuple(masks.shape)}"
                        )
                    target["masks"] = masks.to(torch.uint8)
                y.append(target)
            return y

    return NotebookLunarObjectDetectionTask


def common_datamodule_args(config: InstanceFineTuningConfig) -> dict[str, Any]:
    return {
        "data_root": config.data_root,
        "crop_size": config.crop_size,
        "image_glob": "*.tif",
        "label_glob": "*_label.*",
        "image_suffix": "_input_wac_static_chip",
        "label_suffix": "_label",
        "target_box_format": "xyxy",
        "no_data_replace": 0.0,
        "no_label_replace": None,
        "mask_shift": config.mask_shift,
    }


def calculate_train_stats(
    config: InstanceFineTuningConfig,
    datamodule_cls,
) -> tuple[list[float], list[float]]:
    stats_datamodule = datamodule_cls(
        **common_datamodule_args(config),
        batch_size=config.stats_batch_size,
        num_workers=config.num_workers,
        means=None,
        stds=None,
    )
    stats_datamodule.setup("fit")

    n_pixels = 0
    sum_x = None
    sum_x2 = None

    for batch in stats_datamodule.train_dataloader():
        x = batch["image"]
        b, _, h, w = x.shape
        batch_pixels = b * h * w
        x_sum = x.sum(dim=(0, 2, 3))
        x2_sum = (x * x).sum(dim=(0, 2, 3))
        sum_x = x_sum if sum_x is None else sum_x + x_sum
        sum_x2 = x2_sum if sum_x2 is None else sum_x2 + x2_sum
        n_pixels += batch_pixels

    if sum_x is None or sum_x2 is None or n_pixels == 0:
        raise RuntimeError("No training pixels were available for statistics.")

    means_tensor = sum_x / n_pixels
    stds_tensor = torch.sqrt(torch.clamp(sum_x2 / n_pixels - means_tensor**2, min=1e-12))
    means = means_tensor.tolist()
    stds = stds_tensor.tolist()
    print("per-band means:", means)
    print("per-band stds:", stds)
    return means, stds


def get_normalization_stats(
    config: InstanceFineTuningConfig,
    datamodule_cls,
) -> tuple[list[float], list[float]]:
    if config.normalization_source == "pretrain":
        return load_terramind_wac_pretraining_stats(config.modality_info)
    if config.normalization_source == "finetune":
        return calculate_train_stats(config, datamodule_cls)
    raise ValueError(f"Unsupported normalization_source: {config.normalization_source}")


def create_datamodule(
    config: InstanceFineTuningConfig,
    datamodule_cls,
    means: list[float],
    stds: list[float],
):
    return datamodule_cls(
        **common_datamodule_args(config),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        means=means,
        stds=stds,
    )


def inspect_batch(datamodule) -> dict[str, Any]:
    datamodule.setup("fit")
    sample_batch = next(iter(datamodule.train_dataloader()))
    print("batch keys:", sample_batch.keys())
    print("image:", tuple(sample_batch["image"].shape), sample_batch["image"].dtype)
    print("batch image per-band mean:", sample_batch["image"].mean(dim=(0, 2, 3)).tolist())
    print("batch image per-band std:", sample_batch["image"].std(dim=(0, 2, 3)).tolist())
    print("boxes per image:", [tuple(x.shape) for x in sample_batch["boxes"]])
    print("labels per image:", [tuple(x.shape) for x in sample_batch["labels"]])
    print("masks per image:", [tuple(x.shape) for x in sample_batch["masks"]])
    print("first filename:", sample_batch["filename"][0])
    return sample_batch


def create_task(config: InstanceFineTuningConfig, task_cls, sample_batch: dict[str, Any]):
    wac_num_channels = int(sample_batch["image"].shape[1])
    modality_args = _graha_modality_args(config, wac_num_channels)
    print("WAC channels registered for model:", wac_num_channels)
    print("Graha WAC mode:", config.graha_wac_mode)
    print("Backbone modalities:", modality_args["backbone_modalities"])
    print("Backbone merge method:", modality_args["backbone_merge_method"])

    return task_cls(
        model_factory="ObjectDetectionModelFactory",
        model_args={
            "framework": "mask-rcnn",
            "backbone": "lunarmind_v1_base",
            "backbone_checkpoint_path": str(config.backbone_weights),
            "backbone_cfg": str(config.backbone_cfg),
            "backbone_modality_info_path": str(config.modality_info),
            **modality_args,
            "num_classes": 2,
            "in_channels": wac_num_channels,
            "framework_min_size": config.crop_size,
            "framework_max_size": config.crop_size,
            "backbone_patch_size": 8,
            "backbone_remove_register_tokens": False,
            "necks": [
                {"name": "SelectIndices", "indices": [2, 5, 8, 11]},
                {"name": "ReshapeTokensToImage", "remove_cls_token": False, "h": 32},
                {"name": "LearnedInterpolateToPyramidal"},
                {"name": "FeaturePyramidNetworkNeck"},
            ],
        },
        freeze_backbone=False,
        freeze_decoder=False,
        class_names=["Background", "Crater"],
        backbone_lr=config.backbone_lr,
        head_lr=config.head_lr,
        layer_decay=config.layer_decay,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        anchor_sizes=config.anchor_sizes,
        anchor_aspect_ratios=config.anchor_aspect_ratios,
        score_threshold=config.score_threshold,
    )


def _graha_modality_args(config: InstanceFineTuningConfig, wac_num_channels: int) -> dict[str, Any]:
    if config.graha_wac_mode == "new-wac":
        return {
            "backbone_modalities": ["wac"],
            "backbone_new_modalities": {
                "wac": {
                    "type": "image",
                    "num_channels": wac_num_channels,
                    "data_range": config.normalized_wac_data_range,
                },
            },
            "backbone_merge_method": None,
        }
    if config.graha_wac_mode == "vis-uv":
        if wac_num_channels != 7:
            raise ValueError(
                f"graha_wac_mode='vis-uv' expects 7 channels (5 vis + 2 uv), got {wac_num_channels}"
            )
        return {
            "backbone_modalities": ["vis", "uv"],
            "backbone_new_modalities": None,
            "backbone_merge_method": config.graha_vis_uv_merge_method,
        }
    raise ValueError(f"Unsupported graha_wac_mode: {config.graha_wac_mode}")


def run_loss_smoke(task, sample_batch: dict[str, Any]) -> None:
    task.train()
    x = sample_batch["image"]
    targets = task.reformat_batch(sample_batch, batch_size=x.shape[0])
    loss_dict = task(x, targets)
    if not isinstance(loss_dict, dict):
        loss_dict = loss_dict.output
    loss = sum(loss_dict.values())
    print("loss terms:", {key: float(value.detach().cpu()) for key, value in loss_dict.items()})
    print("single-step train loss:", float(loss.detach().cpu()))


def create_trainer(config: InstanceFineTuningConfig, output_dir: Path) -> Trainer:
    return Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="32",
        max_epochs=config.max_epochs,
        check_val_every_n_epoch=1,
        log_every_n_steps=5,
        logger=False,
        callbacks=[
            FitProgressLogger(
                "Graha",
                log_every_n_batches=config.progress_log_every_n_batches,
            ),
            ModelCheckpoint(
                dirpath=str(output_dir / "checkpoints" / "full_model"),
                monitor="val_segm_map",
                mode="max",
                filename="model-epoch-{epoch:02d}-val-segm-map={val_segm_map:.3f}",
                auto_insert_metric_name=False,
                save_top_k=-1,
                save_last=False,
                save_weights_only=True,
                every_n_epochs=1,
            ),
        ],
    )


def load_lightning_checkpoint_state(task: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint_path = Path(checkpoint_path).resolve()
    print(f"Loading Lightning checkpoint weights from {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    task.load_state_dict(state_dict, strict=True)
    print("Loaded Lightning checkpoint weights.", flush=True)


def save_config(config: InstanceFineTuningConfig, output_dir: Path) -> None:
    def encode(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return list(value)
        return value

    payload = {key: encode(value) for key, value in asdict(config).items()}
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_timing(started_at: float, output_dir: Path) -> None:
    elapsed = time.perf_counter() - started_at
    payload = {"seconds": round(elapsed, 3)}
    with (output_dir / "timing_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Elapsed time: {elapsed:.3f}s")


def save_instance_prediction_plots(
    task,
    datamodule,
    config: InstanceFineTuningConfig,
    output_dir: Path,
) -> Path:
    return plot_instance_predictions(
        task=task,
        datamodule=datamodule,
        output_dir=output_dir,
        split=config.prediction_split,
        n_samples=config.prediction_n_samples,
        filename=f"{config.prediction_split}_instance_predictions.png",
        plots_subdir=Path("plots") / "instance_predictions",
        score_threshold=config.prediction_score_threshold,
        display_method="minmax",
        dpi=200,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simlink-dest",
        "--symlink-dest",
        dest="simlink_dest",
        type=str,
        default=None,
        help="Optional source directory for notebooks/full_model/data.",
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--base-output-dir", type=str, default=None)
    parser.add_argument("--pretrain-dir", type=str, default=None)
    parser.add_argument("--lightning-checkpoint", type=str, default=None)
    parser.add_argument("--graha-wac-mode", choices=["new-wac", "vis-uv"], default="new-wac")
    parser.add_argument("--graha-vis-uv-merge-method", choices=["mean", "max"], default="mean")
    parser.add_argument(
        "--normalization-source",
        choices=["pretrain", "finetune"],
        default="pretrain",
        help="Use TerraMind pretraining stats or finetuning train-split stats for input z-score.",
    )
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--stats-batch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--backbone-lr", type=float, default=5.0e-5)
    parser.add_argument("--head-lr", type=float, default=2.0e-4)
    parser.add_argument("--layer-decay", type=float, default=0.75)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument(
        "--anchor-sizes",
        type=lambda value: [[int(x)] for x in value.split(",")],
        default=[[8], [16], [32], [64]],
        help="Comma-separated one-size-per-FPN-level anchors, e.g. 8,16,32,64.",
    )
    parser.add_argument(
        "--anchor-aspect-ratios",
        type=lambda value: [float(x) for x in value.split(",")],
        default=[0.5, 1.0, 2.0],
    )
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument(
        "--plot-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save true-instance prediction plots after setup/training.",
    )
    parser.add_argument("--prediction-split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--prediction-n-samples", type=int, default=5)
    parser.add_argument("--prediction-score-threshold", type=float, default=0.5)
    parser.add_argument(
        "--progress-log-every-n-batches",
        type=int,
        default=25,
        help="Flush train-batch progress every N batches in sbatch logs.",
    )
    parser.add_argument(
        "--mask-shift",
        type=int,
        nargs=2,
        metavar=("X_PIXELS", "Y_PIXELS"),
        default=(0, 0),
        help=(
            "Integer label-mask shift applied before crop. Positive X moves labels right; "
            "positive Y moves labels down."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fit", action="store_true")
    parser.add_argument("--loss-smoke-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    started_at = time.perf_counter()
    args = parse_args()
    configure_proj_environment()
    notebook_dir = Path(__file__).resolve().parents[1] / "notebooks" / "full_model"
    ensure_data_symlink(args.simlink_dest, notebook_dir / "data")
    config = build_config(args)
    configure_python_paths(config)
    print_config(config)
    validate_required_paths(config)

    deps = import_project_dependencies()
    datamodule_cls = deps["LunarObjectDetectionInstanceMaskDatamodule"]
    task_cls = make_notebook_object_detection_task_class(deps["LunarObjectDetectionTask"])

    output_dir = create_timestamped_output_dir(config.base_output_dir)
    (output_dir / "checkpoints" / "full_model").mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir)
    seed_everything(config.seed)

    means, stds = get_normalization_stats(config, datamodule_cls)
    datamodule = create_datamodule(config, datamodule_cls, means, stds)
    sample_batch = inspect_batch(datamodule)
    task = create_task(config, task_cls, sample_batch)
    run_loss_smoke(task, sample_batch)

    if args.loss_smoke_only or args.no_fit:
        print("Skipping trainer.fit().")
        if config.lightning_checkpoint is not None:
            load_lightning_checkpoint_state(task, config.lightning_checkpoint)
        if config.plot_predictions:
            save_instance_prediction_plots(task, datamodule, config, output_dir)
        save_timing(started_at, output_dir)
        return

    trainer = create_trainer(config, output_dir)
    ckpt_path = (
        str(config.lightning_checkpoint) if config.lightning_checkpoint is not None else None
    )
    if ckpt_path is not None:
        print(f"Resuming trainer.fit() from {ckpt_path}", flush=True)
    trainer.fit(task, datamodule=datamodule, ckpt_path=ckpt_path)
    if config.plot_predictions:
        save_instance_prediction_plots(task, datamodule, config, output_dir)
    save_timing(started_at, output_dir)


if __name__ == "__main__":
    main()
