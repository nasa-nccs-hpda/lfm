"""Instance segmentation experiment configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lfm.all_models.all_tasks import config_defaults as defaults
from lfm.all_models.all_tasks.cli_args import create_instance_experiment_parser
from lfm.all_models.all_tasks.config_validation import validate_experiment_config
from lfm.all_models.all_tasks.data_dictionary import resolve_data_dictionary


@dataclass(frozen=True)
class InstanceSegmentationExperimentConfig:
    notebook_dir: Path
    lfm_root: Path
    data_root: Path
    base_output_dir: Path
    dino_checkpoint: Path | None
    toy_lightning_checkpoint: Path | None
    graha_pretrain_dir: Path | None
    graha_lightning_checkpoint: Path | None
    gfft_config_path: Path | None
    gfft_backbone_checkpoint: Path | None
    dataset_modality: str
    graha_input_modality_mode: str
    graha_vis_uv_merge_method: str
    graha_freeze_backbone: bool
    normalization_source: str
    normalization_modality: str
    image_glob: str
    label_glob: str
    image_suffix: str
    label_suffix: str
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
    ignore_nodata_in_loss: bool
    nodata_ignore_index: int
    excluded_nodata_values: list[float] | None
    skip_toy_fit: bool
    skip_graha_fit: bool
    run_epoch_test_suite: bool
    epoch_test_split: str
    epoch_test_n_samples: int
    epoch_test_every_n_epochs: int
    seed: int


def _resolve_toy_lightning_checkpoint(args: argparse.Namespace) -> Path | None:
    return (
        Path(args.toy_lightning_checkpoint).resolve()
        if args.toy_lightning_checkpoint
        else None
    )


def build_config_from_args(
    args: argparse.Namespace,
) -> InstanceSegmentationExperimentConfig:
    lfm_root = Path(__file__).resolve().parents[3]
    notebook_dir = lfm_root / "notebooks" / "full_model"
    scripts_output_dir = lfm_root / "scripts" / "outputs"
    dataset_modality = getattr(
        args,
        "dataset_modality",
        defaults.DEFAULT_DATASET_MODALITY,
    )
    config = InstanceSegmentationExperimentConfig(
        notebook_dir=notebook_dir,
        lfm_root=lfm_root,
        data_root=(
            Path(args.data_root).resolve() if args.data_root else notebook_dir / "data"
        ),
        base_output_dir=(
            Path(args.base_output_dir).resolve()
            if args.base_output_dir
            else scripts_output_dir / "instance_seg_comparison"
        ),
        dino_checkpoint=(
            Path(args.dino_checkpoint).resolve() if args.dino_checkpoint else None
        ),
        toy_lightning_checkpoint=_resolve_toy_lightning_checkpoint(args),
        graha_pretrain_dir=(
            Path(args.graha_pretrain_dir).resolve() if args.graha_pretrain_dir else None
        ),
        graha_lightning_checkpoint=(
            Path(args.graha_lightning_checkpoint).resolve()
            if args.graha_lightning_checkpoint
            else None
        ),
        gfft_config_path=(
            Path(args.gfft_config_path).resolve()
            if getattr(args, "gfft_config_path", None)
            else None
        ),
        gfft_backbone_checkpoint=(
            Path(args.gfft_backbone_checkpoint).resolve()
            if getattr(args, "gfft_backbone_checkpoint", None)
            else None
        ),
        dataset_modality=dataset_modality,
        graha_input_modality_mode=defaults.resolve_graha_input_modality_mode(
            dataset_modality=dataset_modality,
            graha_input_modality_mode=getattr(
                args,
                "graha_input_modality_mode",
                None,
            ),
        ),
        graha_vis_uv_merge_method=args.graha_vis_uv_merge_method,
        graha_freeze_backbone=getattr(
            args,
            "graha_freeze_backbone",
            defaults.DEFAULT_GRAHA_FREEZE_BACKBONE,
        ),
        normalization_source=getattr(
            args,
            "normalization_source",
            defaults.DEFAULT_NORMALIZATION_SOURCE,
        ),
        normalization_modality=defaults.resolve_normalization_modality(
            dataset_modality=dataset_modality,
            normalization_modality=getattr(args, "normalization_modality", None),
        ),
        image_glob=args.image_glob,
        label_glob=args.label_glob,
        image_suffix=args.image_suffix,
        label_suffix=args.label_suffix,
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
        toy_gradient_clip_val=(
            None if args.disable_toy_gradient_clipping else args.toy_gradient_clip_val
        ),
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
        ignore_nodata_in_loss=getattr(
            args,
            "ignore_nodata_in_loss",
            defaults.DEFAULT_IGNORE_NODATA_IN_LOSS,
        ),
        nodata_ignore_index=getattr(
            args,
            "nodata_ignore_index",
            defaults.DEFAULT_NODATA_IGNORE_INDEX,
        ),
        excluded_nodata_values=getattr(args, "excluded_nodata_values", None),
        skip_toy_fit=args.no_fit or args.skip_toy_fit,
        skip_graha_fit=args.no_fit or args.skip_graha_fit,
        run_epoch_test_suite=args.run_epoch_test_suite,
        epoch_test_split=args.epoch_test_split,
        epoch_test_n_samples=args.epoch_test_n_samples,
        epoch_test_every_n_epochs=args.epoch_test_every_n_epochs,
        seed=args.seed,
    )
    validate_experiment_config(config, task="instance")
    return config


def build_config(
    *,
    data_dict: dict[str, Any] | None = None,
    data_root: str | Path | None = None,
    base_output_dir: str | Path | None = None,
    dino_checkpoint: str | Path | None = None,
    toy_lightning_checkpoint: str | Path | None = None,
    graha_pretrain_dir: str | Path | None = None,
    graha_lightning_checkpoint: str | Path | None = None,
    gfft_config_path: str | Path | None = None,
    gfft_backbone_checkpoint: str | Path | None = None,
    **overrides: Any,
) -> InstanceSegmentationExperimentConfig:
    """Build an instance segmentation experiment config from explicit values."""
    args = create_instance_experiment_parser().parse_args([])
    data_dict_overrides = resolve_data_dictionary(data_dict)
    path_values = {
        "data_root": data_root,
        "base_output_dir": base_output_dir,
        "dino_checkpoint": dino_checkpoint,
        "toy_lightning_checkpoint": toy_lightning_checkpoint,
        "graha_pretrain_dir": graha_pretrain_dir,
        "graha_lightning_checkpoint": graha_lightning_checkpoint,
        "gfft_config_path": gfft_config_path,
        "gfft_backbone_checkpoint": gfft_backbone_checkpoint,
    }
    for name, value in path_values.items():
        if value is not None:
            setattr(args, name, str(value))

    for name, value in data_dict_overrides.items():
        if not hasattr(args, name):
            if name == "semantic_label_source":
                continue
            raise TypeError(f"Unknown instance data dictionary option: {name}")
        setattr(args, name, value)

    for name, value in overrides.items():
        if not hasattr(args, name):
            raise TypeError(f"Unknown instance experiment config option: {name}")
        setattr(args, name, value)

    return build_config_from_args(args)
