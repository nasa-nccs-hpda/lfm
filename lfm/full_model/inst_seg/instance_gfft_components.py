"""GFFT/Fourier-VQ MultiMAE instance segmentation workflow components."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from lfm.full_model.inst_seg import instance_graha_components as graha


def build_comparison_config(config: Any, output_dir: Path):
    """Build a Graha-shaped config with GFFT backbone weights."""
    gfft_checkpoint = getattr(config, "gfft_backbone_checkpoint", None)
    if gfft_checkpoint is None:
        raise ValueError(
            "GFFT instance workflow requires config.gfft_backbone_checkpoint. "
            "Pass --gfft-backbone-checkpoint or build_config(..., "
            "gfft_backbone_checkpoint=...)."
        )
    graha_config = graha.build_comparison_config(config, output_dir)
    return replace(
        graha_config,
        backbone_weights=Path(gfft_checkpoint).resolve(),
    )


def build_config(args: Any):
    return graha.build_config(args)


def configure_proj_environment() -> None:
    graha.configure_proj_environment()


def configure_python_paths(config: Any) -> None:
    graha.configure_python_paths(config)


def validate_required_paths(config: Any) -> None:
    graha.validate_required_paths(config)


def import_project_dependencies() -> dict[str, Any]:
    return graha.import_project_dependencies()


def make_downstream_object_detection_task_class(lunar_object_detection_task_cls):
    return graha.make_downstream_object_detection_task_class(
        lunar_object_detection_task_cls
    )


def get_normalization_stats(
    config: Any,
    datamodule_cls,
) -> tuple[list[float], list[float]]:
    return graha.get_normalization_stats(config, datamodule_cls)


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
    print("GFFT backbone modalities:", modality_args["backbone_modalities"])
    print("GFFT backbone merge method:", modality_args["backbone_merge_method"])

    return task_cls(
        model_factory="ObjectDetectionModelFactory",
        model_args={
            "framework": "mask-rcnn",
            "backbone": "lunar_fvqmultimae",
            "backbone_checkpoint_path": str(config.backbone_weights),
            **modality_args,
            "num_classes": 2,
            "in_channels": num_channels,
            "framework_min_size": config.crop_size,
            "framework_max_size": config.crop_size,
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
                {"name": "MultilayerSimpleFeaturePyramid", "out_channels": 256},
                {"name": "FeaturePyramidNetworkNeck"},
            ],
        },
        freeze_backbone=False,
        freeze_decoder=False,
        class_names=["Background", "Crater"],
        backbone_lr=config.backbone_lr,
        head_lr=config.head_lr,
        layer_decay=0.65,
        weight_decay=config.weight_decay,
        head_weight_decay=1.0e-3,
        betas=(0.9, 0.999),
        warmup_steps=config.warmup_steps,
        eta_min=1.0e-6,
        anchor_sizes=config.anchor_sizes,
        anchor_aspect_ratios=config.anchor_aspect_ratios,
        score_threshold=config.score_threshold,
    )


def create_trainer(config: Any, output_dir: Path):
    return graha.create_trainer(config, output_dir)


def load_lightning_checkpoint_state(*args: Any, **kwargs: Any) -> None:
    graha.load_lightning_checkpoint_state(*args, **kwargs)
