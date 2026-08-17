"""GFFT/Fourier-VQ MultiMAE semantic segmentation workflow components."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from lfm.all_models.all_tasks import config_defaults as defaults
from lfm.full_model.all_tasks import (
    load_gfft_config,
    resolve_gfft_normalization_stats,
)
from lfm.full_model.sem_seg import semantic_graha_components as graha


def _resolve_gfft_backbone_checkpoint(config: Any) -> Path:
    if getattr(config, "gfft_backbone_checkpoint", None) is not None:
        return Path(config.gfft_backbone_checkpoint).resolve()

    config_path = getattr(config, "gfft_config_path", None)
    if config_path is not None:
        gfft_config = load_gfft_config(config_path)
        if gfft_config.backbone_checkpoint_path is not None:
            return Path(str(gfft_config.backbone_checkpoint_path)).expanduser()

    raise ValueError(
        "GFFT semantic workflow requires a backbone checkpoint. Pass "
        "--gfft-backbone-checkpoint, or pass --gfft-config-path pointing to a "
        "YAML with model.init_args.model_args.backbone_checkpoint_path."
    )


def build_finetuning_config(config: Any, output_dir: Path):
    """Build an internal fine-tuning config with GFFT backbone weights."""
    gfft_checkpoint = _resolve_gfft_backbone_checkpoint(config)
    graha_config = graha.build_comparison_config(config, output_dir)
    return replace(
        graha_config,
        backbone_weights=gfft_checkpoint,
    )


def build_config(args: Any):
    return graha.build_config(args)


def configure_proj_environment() -> None:
    graha.configure_proj_environment()


def configure_python_paths(config: Any) -> None:
    graha.configure_python_paths(config)


def validate_required_paths(config: Any) -> None:
    required_paths = [
        config.repo_root,
        config.backbone_weights,
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
            "Missing required GFFT paths:\n" + "\n".join(str(path) for path in missing)
        )


def print_config(config: Any) -> None:
    print("Package directory:", config.package_dir)
    print("Notebook directory:", config.notebook_dir)
    print("Graha/Lunar-FM code root:", config.repo_root)
    print("Data root:", config.data_root)
    print("GFFT backbone weights:", config.backbone_weights)
    print("GFFT config YAML:", config.gfft_config_path)
    print("Normalization source:", config.normalization_source)
    print("Normalization modality:", config.normalization_modality)
    print("Band filter:", config.band_filter)
    print("Base output directory:", config.base_output_dir)
    print("Lightning checkpoint:", config.lightning_checkpoint)


def import_project_dependencies() -> dict[str, Any]:
    return graha.import_project_dependencies()


def make_downstream_shape_segmentation_task_class(lunar_shape_segmentation_task_cls):
    return graha.make_downstream_shape_segmentation_task_class(
        lunar_shape_segmentation_task_cls
    )


def get_normalization_stats(
    config: Any,
    datamodule_cls,
) -> tuple[list[float], list[float]]:
    if config.normalization_source == "pretrain":
        config_path = getattr(config, "gfft_config_path", None)
        if config_path is None:
            raise ValueError(
                "GFFT pretrain normalization requires --gfft-config-path. "
                "Pass a GFFT YAML with data.init_args normalization stats, "
                "or use --normalization-source finetune."
            )
        means, stds = resolve_gfft_normalization_stats(
            load_gfft_config(config_path),
            normalization_modality=config.normalization_modality,
            band_filter=config.band_filter,
        )
        print("GFFT normalization means:", means)
        print("GFFT normalization stds:", stds)
        return means, stds
    if config.normalization_source == "finetune":
        return graha.calculate_train_stats(config, datamodule_cls)
    raise ValueError(f"Unsupported normalization_source: {config.normalization_source}")


def create_datamodule(
    config: Any,
    datamodule_cls,
    means: list[float],
    stds: list[float],
):
    return graha.create_datamodule(config, datamodule_cls, means, stds)


def inspect_batch(datamodule) -> dict[str, Any]:
    return graha.inspect_batch(datamodule)


def _gfft_modality_args(config: Any, num_channels: int) -> dict[str, Any]:
    image_size = int(config.crop_size)
    patch_size = 8

    backend_modalities = defaults.normalize_graha_backend_modalities(
        getattr(config, "graha_backend_modalities", None)
    )
    if backend_modalities is not None:
        expected_channels = defaults.expected_graha_backend_num_channels(
            backend_modalities
        )
        if expected_channels is not None and num_channels != expected_channels:
            raise ValueError(
                "graha_backend_modalities="
                f"{backend_modalities!r} expects {expected_channels} input "
                f"channel(s), got {num_channels}."
            )
        if "wac" not in backend_modalities:
            return {
                "backbone_modalities": backend_modalities,
                "backbone_merge_method": "concat",
            }

    if config.graha_input_modality_mode == "vis-uv":
        if num_channels != 7:
            raise ValueError(
                "GFFT vis-uv mode expects 7 channels (5 vis + 2 uv), "
                f"got {num_channels}."
            )
        return {
            "backbone_modalities": ["vis", "uv"],
            "backbone_merge_method": "concat",
        }

    if num_channels == 5:
        return {
            "backbone_modalities": ["vis"],
            "backbone_merge_method": "concat",
        }

    modality_name = "nac" if num_channels == 1 else "wac"
    return {
        "backbone_modalities": [modality_name],
        "backbone_new_modalities": {
            modality_name: {
                "num_channels": num_channels,
                "patch_size": patch_size,
                "image_size": image_size,
            },
        },
        "backbone_merge_method": "concat",
    }


def create_task(config: Any, task_cls, sample_batch: dict[str, Any]):
    num_channels = int(sample_batch["image"].shape[1])
    modality_args = _gfft_modality_args(config, num_channels)
    grid_size = int(config.crop_size) // 8

    print("GFFT channels registered for model:", num_channels)
    print(
        "GFFT backend modalities override:",
        getattr(config, "graha_backend_modalities", None),
    )
    print("GFFT backbone modalities:", modality_args["backbone_modalities"])
    print("GFFT backbone merge method:", modality_args["backbone_merge_method"])

    return task_cls(
        backbone_lr=5.0e-5,
        head_lr=5.0e-5,
        layer_decay=0.65,
        weight_decay=0.05,
        head_weight_decay=1.0e-3,
        betas=(0.9, 0.999),
        warmup_steps=500,
        eta_min=1.0e-6,
        shape_loss_weight=config.shape_loss_weight,
        shape_loss_pad_frac=config.shape_loss_pad_frac,
        model_factory="EncoderDecoderFactory",
        model_args={
            "backbone": "lunar_fvqmultimae",
            "backbone_checkpoint_path": str(config.backbone_weights),
            **modality_args,
            "backbone_patch_size": 8,
            "backbone_image_size": config.crop_size,
            "backbone_drop_path_rate": 0.1,
            "necks": [
                {"name": "SelectIndices", "indices": [2, 5, 8, 11]},
                {
                    "name": "ReshapeTokensToImage",
                    "remove_cls_token": False,
                    "h": grid_size,
                },
                {"name": "LearnedInterpolateToPyramidal"},
            ],
            "decoder": "UNetDecoder",
            "decoder_channels": [512, 256, 128, 64],
            "head_channel_list": [256],
            "num_classes": 2,
            "head_dropout": 0.1,
        },
        loss="dice",
        ignore_index=(
            config.nodata_ignore_index if config.ignore_nodata_in_loss else None
        ),
        class_names=["Background", "Crater"],
        freeze_backbone=False,
        freeze_decoder=False,
        plot_on_val=0,
    )


def inspect_backbone(task) -> None:
    graha.inspect_backbone(task)


def create_trainer(config: Any, output_dir: Path, *args: Any, **kwargs: Any):
    return graha.create_trainer(config, output_dir, *args, **kwargs)


def load_lightning_checkpoint_state(*args: Any, **kwargs: Any) -> None:
    graha.load_lightning_checkpoint_state(*args, **kwargs)
