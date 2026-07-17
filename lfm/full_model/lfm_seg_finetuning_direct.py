"""Direct Graha/Lunar-FM semantic segmentation fine-tuning script.

This is a cleaned Python version of ``lfm_seg_finetuning_direct.ipynb``.
It builds the TerraMind/TerraTorch stack directly from Python classes,
without using the TerraTorch CLI.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint


@dataclass(frozen=True)
class FineTuningConfig:
    package_dir: Path
    notebook_dir: Path
    lfm_root: Path
    repo_root: Path
    pretrain_dir: Path
    backbone_weights: Path
    backbone_cfg: Path
    modality_info: Path
    data_root: Path
    base_output_dir: Path
    lightning_checkpoint: Path | None
    normalized_wac_data_range: list[float]
    crop_size: int
    stats_batch_size: int
    batch_size: int
    num_workers: int
    max_epochs: int
    cache_predictions: bool
    prediction_split: str
    prediction_n_samples: int
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


def build_config(args: argparse.Namespace) -> FineTuningConfig:
    """Create the script configuration from defaults plus CLI overrides."""
    package_dir = Path(__file__).resolve().parent
    lfm_root = package_dir.parents[1]
    notebook_dir = lfm_root / "notebooks" / "full_model"
    repo_root = package_dir / "graha-lunar-fm"

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
        else notebook_dir / "outputs" / "graha_finetuning"
    )
    lightning_checkpoint = (
        Path(args.lightning_checkpoint).resolve() if args.lightning_checkpoint else None
    )

    return FineTuningConfig(
        package_dir=package_dir,
        notebook_dir=notebook_dir,
        lfm_root=lfm_root,
        repo_root=repo_root,
        pretrain_dir=pretrain_dir,
        backbone_weights=pretrain_dir / "checkpoints/checkpoint_weights_final.pt",
        backbone_cfg=pretrain_dir / "full_config.yaml",
        modality_info=pretrain_dir / "modality_info.yaml",
        data_root=data_root,
        base_output_dir=base_output_dir,
        lightning_checkpoint=lightning_checkpoint,
        normalized_wac_data_range=[-1.0, 1.0],
        crop_size=args.crop_size,
        stats_batch_size=args.stats_batch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        cache_predictions=args.cache_predictions,
        prediction_split=args.prediction_split,
        prediction_n_samples=args.prediction_n_samples,
        seed=args.seed,
    )


def configure_python_paths(config: FineTuningConfig) -> None:
    """Make local notebook helpers and Graha/Lunar-FM code importable."""
    for path in [config.repo_root, config.lfm_root]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def print_config(config: FineTuningConfig) -> None:
    print("Package directory:", config.package_dir)
    print("Notebook directory:", config.notebook_dir)
    print("Graha/Lunar-FM code root:", config.repo_root)
    print("Data root:", config.data_root)
    print("Backbone weights:", config.backbone_weights)
    print("Backbone config:", config.backbone_cfg)
    print("Modality info:", config.modality_info)
    print("Base output directory:", config.base_output_dir)
    print("Lightning checkpoint:", config.lightning_checkpoint)
    print("Normalized WAC modality data_range:", config.normalized_wac_data_range)


def validate_required_paths(config: FineTuningConfig) -> None:
    required_paths = [
        config.repo_root,
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
    """Import local helpers after sys.path has been configured."""
    import terratorch_integration  # noqa: F401
    from lfm.full_model.datamodules import LunarSemanticSegmentationDatamodule
    from lfm.full_model.utils import ValidationPlotCallback
    from terratorch_integration.lunar_segmentation_task import LunarShapeSegmentationTask
    from lfm.full_model.utils import create_timestamped_output_dir
    from lfm.full_model.utils import save_prediction_cache

    return {
        "LunarSemanticSegmentationDatamodule": LunarSemanticSegmentationDatamodule,
        "LunarShapeSegmentationTask": LunarShapeSegmentationTask,
        "ValidationPlotCallback": ValidationPlotCallback,
        "create_timestamped_output_dir": create_timestamped_output_dir,
        "save_prediction_cache": save_prediction_cache,
    }


def make_notebook_task_class(lunar_shape_segmentation_task_cls):
    """Create the task subclass that strips datamodule-only metadata."""

    class NotebookLunarShapeSegmentationTask(lunar_shape_segmentation_task_cls):
        """Drop datamodule metadata that should not be forwarded to the model."""

        def _drop_extra_batch_metadata(self, batch):
            if isinstance(batch, dict):
                batch.pop("num_craters", None)
            return batch

        def training_step(self, batch, *args, **kwargs):
            return super().training_step(self._drop_extra_batch_metadata(batch), *args, **kwargs)

        def validation_step(self, batch, *args, **kwargs):
            return super().validation_step(self._drop_extra_batch_metadata(batch), *args, **kwargs)

        def test_step(self, batch, *args, **kwargs):
            return super().test_step(self._drop_extra_batch_metadata(batch), *args, **kwargs)

        def predict_step(self, batch, *args, **kwargs):
            return super().predict_step(self._drop_extra_batch_metadata(batch), *args, **kwargs)

    return NotebookLunarShapeSegmentationTask


def create_output_dirs(
    config: FineTuningConfig,
    create_timestamped_output_dir,
    *,
    use_timestamp: bool = True,
) -> Path:
    output_dir = (
        create_timestamped_output_dir(config.base_output_dir)
        if use_timestamp
        else Path(config.base_output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    print("Output directory:", output_dir)
    print("Plots directory:", plots_dir)
    return output_dir


def common_datamodule_args(config: FineTuningConfig) -> dict[str, Any]:
    return {
        "data_root": config.data_root,
        "crop_size": config.crop_size,
        "image_glob": "*.tif",
        "label_glob": "*_label.*",
        "image_suffix": "_input_wac_static_chip",
        "label_suffix": "_label",
        "no_data_replace": 0.0,
        "no_label_replace": None,
    }


def calculate_train_stats(config: FineTuningConfig, datamodule_cls) -> tuple[list[float], list[float]]:
    """Compute per-band mean/std from cropped, unnormalized training batches."""
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


def create_datamodule(
    config: FineTuningConfig,
    datamodule_cls,
    means: list[float],
    stds: list[float],
):
    """Create the normalized fine-tuning datamodule."""
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
    print(
        "batch image per-band mean:",
        sample_batch["image"].mean(dim=(0, 2, 3)).tolist(),
    )
    print(
        "batch image per-band std:",
        sample_batch["image"].std(dim=(0, 2, 3)).tolist(),
    )
    print("mask:", tuple(sample_batch["mask"].shape), sample_batch["mask"].dtype)
    print("mask values:", torch.unique(sample_batch["mask"]).tolist())
    if "crater_boxes" in sample_batch:
        print("crater boxes per image:", [tuple(x.shape) for x in sample_batch["crater_boxes"]])
    return sample_batch


def create_task(config: FineTuningConfig, task_cls, sample_batch: dict[str, Any]):
    wac_modality = "wac"
    wac_num_channels = int(sample_batch["image"].shape[1])
    print("WAC channels registered for model:", wac_num_channels)

    return task_cls(
        backbone_lr=5.0e-5,
        head_lr=2.0e-4,
        layer_decay=0.75,
        weight_decay=0.05,
        warmup_steps=500,
        shape_loss_weight=0.05,
        shape_loss_pad_frac=0.3,
        model_factory="EncoderDecoderFactory",
        model_args={
            "backbone": "lunarmind_v1_base",
            "backbone_checkpoint_path": str(config.backbone_weights),
            "backbone_cfg": str(config.backbone_cfg),
            "backbone_modality_info_path": str(config.modality_info),
            "backbone_modalities": [wac_modality],
            "backbone_new_modalities": {
                wac_modality: {
                    "type": "image",
                    "num_channels": wac_num_channels,
                    "data_range": config.normalized_wac_data_range,
                },
            },
            "backbone_patch_size": 8,
            "backbone_remove_register_tokens": False,
            "backbone_merge_method": None,
            "necks": [
                {"name": "SelectIndices", "indices": [2, 5, 8, 11]},
                {"name": "ReshapeTokensToImage", "remove_cls_token": False, "h": 32},
                {"name": "LearnedInterpolateToPyramidal"},
            ],
            "decoder": "UNetDecoder",
            "decoder_channels": [512, 256, 128, 64],
            "head_channel_list": [256],
            "num_classes": 2,
            "head_dropout": 0.1,
        },
        loss="dice",
        class_names=["Background", "Crater"],
        freeze_backbone=False,
        freeze_decoder=False,
        plot_on_val=0,
    )


def inspect_backbone(task) -> None:
    backbone = task.model.encoder
    print(type(backbone))
    print("modalities:", backbone.modalities)
    print(f"backbone params: {backbone.get_num_params():,}")


def load_lightning_checkpoint_state(task: torch.nn.Module, checkpoint_path: Path, model_name: str) -> None:
    """Load Lightning checkpoint weights into an already-built task."""
    checkpoint_path = Path(checkpoint_path).resolve()
    print(f"Loading {model_name} Lightning checkpoint weights from {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    task.load_state_dict(state_dict, strict=True)
    print(f"Loaded {model_name} Lightning checkpoint weights.", flush=True)


def create_trainer(
    config: FineTuningConfig,
    output_dir: Path,
    validation_plot_callback_cls,
    *,
    plot_output_dir: Path | None = None,
    plots_subdir: str | Path = "plots",
    checkpoint_subdir: str | Path = Path("checkpoints") / "full_model",
) -> Trainer:
    plot_output_dir = output_dir if plot_output_dir is None else plot_output_dir
    return Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="32",
        max_epochs=config.max_epochs,
        check_val_every_n_epoch=1,
        log_every_n_steps=5,
        logger=False,
        callbacks=[
            validation_plot_callback_cls(
                output_dir=plot_output_dir,
                n_samples=5,
                every_n_epochs=1,
                plots_subdir=plots_subdir,
                dpi=150,
            ),
            ModelCheckpoint(
                dirpath=str(output_dir / checkpoint_subdir),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--base-output-dir", type=str, default=None)
    parser.add_argument("--pretrain-dir", type=str, default=None)
    parser.add_argument(
        "--lightning-checkpoint",
        type=str,
        default=None,
        help="Optional Lightning .ckpt. Resumes fit, or loads weights when --no-fit is set.",
    )
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--stats-batch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--cache-predictions", action="store_true")
    parser.add_argument("--prediction-split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--prediction-n-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fit", action="store_true", help="Build everything but skip trainer.fit().")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_proj_environment()
    config = build_config(args)
    configure_python_paths(config)
    print_config(config)
    validate_required_paths(config)

    deps = import_project_dependencies()
    datamodule_cls = deps["LunarSemanticSegmentationDatamodule"]
    task_cls = make_notebook_task_class(deps["LunarShapeSegmentationTask"])

    output_dir = create_output_dirs(config, deps["create_timestamped_output_dir"])
    seed_everything(config.seed)

    means, stds = calculate_train_stats(config, datamodule_cls)
    datamodule = create_datamodule(config, datamodule_cls, means, stds)
    sample_batch = inspect_batch(datamodule)
    task = create_task(config, task_cls, sample_batch)
    inspect_backbone(task)

    trainer = create_trainer(config, output_dir, deps["ValidationPlotCallback"])
    if args.no_fit:
        print("Skipping trainer.fit() because --no-fit was set.")
        if config.lightning_checkpoint is not None:
            load_lightning_checkpoint_state(task, config.lightning_checkpoint, "Graha")
            if config.cache_predictions:
                deps["save_prediction_cache"](
                    task=task,
                    datamodule=datamodule,
                    output_dir=output_dir,
                    model_name="graha",
                    split=config.prediction_split,
                    n_samples=config.prediction_n_samples,
                )
        return
    ckpt_path = (
        str(config.lightning_checkpoint) if config.lightning_checkpoint is not None else None
    )
    if ckpt_path is not None:
        print(f"Resuming trainer.fit() from {ckpt_path}", flush=True)
    trainer.fit(task, datamodule=datamodule, ckpt_path=ckpt_path)
    if config.cache_predictions:
        deps["save_prediction_cache"](
            task=task,
            datamodule=datamodule,
            output_dir=output_dir,
            model_name="graha",
            split=config.prediction_split,
            n_samples=config.prediction_n_samples,
        )


if __name__ == "__main__":
    main()
