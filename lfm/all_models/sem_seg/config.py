"""Semantic segmentation experiment configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lfm.all_models.all_tasks import config_defaults as defaults
from lfm.all_models.all_tasks.cli_args import create_semantic_experiment_parser
from lfm.all_models.all_tasks.config_validation import validate_experiment_config
from lfm.all_models.all_tasks.data_dictionary import resolve_data_dictionary


@dataclass(frozen=True)
class SemanticSegmentationExperimentConfig:
    repo_root: Path
    notebook_dir: Path
    data_root: Path
    base_output_dir: Path
    dino_checkpoint: Path | None
    toy_lightning_checkpoint: Path | None
    band_filter: list[int]
    target_size: tuple[int, int]
    spatial_transform: str
    semantic_label_source: str
    image_glob: str
    label_glob: str
    image_suffix: str
    label_suffix: str
    image_file_type: str
    dataset_modality: str
    max_train_samples: int | None
    max_val_samples: int | None
    max_test_samples: int | None
    ignore_nodata_in_loss: bool
    nodata_ignore_index: int
    excluded_nodata_values: list[float] | None
    batch_size: int
    num_workers: int
    max_epochs: int
    learning_rate: float
    weight_decay: float
    toy_loss_type: str
    use_toy_shape_loss: bool
    toy_shape_loss_weight: float
    toy_shape_loss_pad_frac: float
    freeze_encoder: bool
    normalize_inputs: bool
    normalization_source: str
    normalization_modality: str
    toy_gradient_clip_val: float | None
    plot_every_n_epochs: int
    plot_n_samples: int
    cache_predictions: bool
    prediction_split: str
    prediction_n_samples: int
    graha_base_output_dir: Path
    graha_pretrain_dir: Path | None
    graha_lightning_checkpoint: Path | None
    gfft_config_path: Path | None
    gfft_backbone_checkpoint: Path | None
    graha_input_modality_mode: str
    graha_vis_uv_merge_method: str
    graha_freeze_backbone: bool
    graha_shape_loss_weight: float
    graha_shape_loss_pad_frac: float
    graha_backbone_lr: float
    graha_head_lr: float
    graha_layer_decay: float
    graha_weight_decay: float
    graha_warmup_steps: int
    graha_stats_batch_size: int
    graha_batch_size: int
    graha_num_workers: int
    progress_log_every_n_batches: int
    skip_toy_fit: bool
    skip_graha_fit: bool
    run_epoch_test_suite: bool
    epoch_test_split: str
    epoch_test_n_samples: int
    epoch_test_every_n_epochs: int
    seed: int


def _file_type_from_glob(pattern: str) -> str:
    stripped = pattern.replace("*", "")
    if stripped.startswith(".") and stripped.count(".") == 1:
        return stripped
    suffix = Path(stripped).suffix
    return suffix or ".tif"


def build_config_from_args(
    args: argparse.Namespace,
) -> SemanticSegmentationExperimentConfig:
    repo_root = Path(__file__).resolve().parents[3]
    notebook_dir = repo_root / "notebooks" / "full_model"
    scripts_output_dir = repo_root / "scripts" / "outputs"
    data_root = (
        Path(args.data_root).resolve() if args.data_root else notebook_dir / "data"
    )
    base_output_dir = (
        Path(args.base_output_dir).resolve()
        if args.base_output_dir
        else scripts_output_dir / "semantic_seg_comparison"
    )
    dino_checkpoint = (
        Path(args.dino_checkpoint).resolve() if args.dino_checkpoint else None
    )
    toy_lightning_checkpoint_arg = args.toy_lightning_checkpoint
    toy_lightning_checkpoint = (
        Path(toy_lightning_checkpoint_arg).resolve()
        if toy_lightning_checkpoint_arg
        else None
    )
    graha_base_output_dir = (
        Path(args.graha_base_output_dir).resolve()
        if args.graha_base_output_dir
        else scripts_output_dir / "graha_finetuning"
    )
    graha_pretrain_dir = (
        Path(args.graha_pretrain_dir).resolve() if args.graha_pretrain_dir else None
    )
    graha_lightning_checkpoint = (
        Path(args.graha_lightning_checkpoint).resolve()
        if args.graha_lightning_checkpoint
        else None
    )
    gfft_config_path = (
        Path(args.gfft_config_path).resolve()
        if getattr(args, "gfft_config_path", None)
        else None
    )
    gfft_backbone_checkpoint = (
        Path(args.gfft_backbone_checkpoint).resolve()
        if getattr(args, "gfft_backbone_checkpoint", None)
        else None
    )
    dataset_modality = getattr(
        args,
        "dataset_modality",
        defaults.DEFAULT_DATASET_MODALITY,
    )
    band_filter = (
        list(args.band_filter)
        if args.band_filter is not None
        else defaults.default_band_filter_for_dataset(dataset_modality)
    )
    label_file_type = _file_type_from_glob(args.label_glob)
    semantic_label_source = defaults.resolve_semantic_label_source(
        semantic_label_source=getattr(
            args,
            "semantic_label_source",
            defaults.DEFAULT_SEMANTIC_LABEL_SOURCE,
        ),
        label_glob=args.label_glob,
        label_file_type=label_file_type,
        data_root=data_root,
    )

    config = SemanticSegmentationExperimentConfig(
        repo_root=repo_root,
        notebook_dir=notebook_dir,
        data_root=data_root,
        base_output_dir=base_output_dir,
        dino_checkpoint=dino_checkpoint,
        toy_lightning_checkpoint=toy_lightning_checkpoint,
        band_filter=band_filter,
        target_size=(args.target_size, args.target_size),
        spatial_transform="crop",
        semantic_label_source=semantic_label_source,
        image_glob=args.image_glob,
        label_glob=args.label_glob,
        image_suffix=args.image_suffix,
        label_suffix=args.label_suffix,
        image_file_type=_file_type_from_glob(args.image_glob),
        dataset_modality=dataset_modality,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
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
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        toy_loss_type=args.toy_loss_type,
        use_toy_shape_loss=args.use_toy_shape_loss,
        toy_shape_loss_weight=args.toy_shape_loss_weight,
        toy_shape_loss_pad_frac=args.toy_shape_loss_pad_frac,
        freeze_encoder=args.freeze_encoder,
        normalize_inputs=args.normalize_inputs,
        normalization_source=getattr(
            args,
            "normalization_source",
            defaults.DEFAULT_NORMALIZATION_SOURCE,
        ),
        normalization_modality=defaults.resolve_normalization_modality(
            dataset_modality=dataset_modality,
            normalization_modality=getattr(args, "normalization_modality", None),
        ),
        toy_gradient_clip_val=(
            None if args.disable_toy_gradient_clipping else args.toy_gradient_clip_val
        ),
        plot_every_n_epochs=args.plot_every_n_epochs,
        plot_n_samples=args.plot_n_samples,
        cache_predictions=args.cache_predictions,
        prediction_split=args.prediction_split,
        prediction_n_samples=args.prediction_n_samples,
        graha_base_output_dir=graha_base_output_dir,
        graha_pretrain_dir=graha_pretrain_dir,
        graha_lightning_checkpoint=graha_lightning_checkpoint,
        gfft_config_path=gfft_config_path,
        gfft_backbone_checkpoint=gfft_backbone_checkpoint,
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
        graha_shape_loss_weight=getattr(
            args,
            "graha_shape_loss_weight",
            defaults.DEFAULT_GRAHA_SHAPE_LOSS_WEIGHT,
        ),
        graha_shape_loss_pad_frac=getattr(
            args,
            "graha_shape_loss_pad_frac",
            defaults.DEFAULT_GRAHA_SHAPE_LOSS_PAD_FRAC,
        ),
        graha_backbone_lr=getattr(
            args,
            "graha_backbone_lr",
            defaults.DEFAULT_GRAHA_BACKBONE_LR,
        ),
        graha_head_lr=getattr(
            args,
            "graha_head_lr",
            defaults.DEFAULT_GRAHA_HEAD_LR,
        ),
        graha_layer_decay=getattr(
            args,
            "graha_layer_decay",
            defaults.DEFAULT_GRAHA_LAYER_DECAY,
        ),
        graha_weight_decay=getattr(
            args,
            "graha_weight_decay",
            defaults.DEFAULT_GRAHA_WEIGHT_DECAY,
        ),
        graha_warmup_steps=getattr(
            args,
            "graha_warmup_steps",
            defaults.DEFAULT_GRAHA_WARMUP_STEPS,
        ),
        graha_stats_batch_size=args.graha_stats_batch_size,
        graha_batch_size=args.graha_batch_size,
        graha_num_workers=args.graha_num_workers,
        progress_log_every_n_batches=getattr(
            args,
            "progress_log_every_n_batches",
            defaults.DEFAULT_PROGRESS_LOG_EVERY_N_BATCHES,
        ),
        skip_toy_fit=args.no_fit or args.skip_toy_fit,
        skip_graha_fit=args.no_fit or args.skip_graha_fit,
        run_epoch_test_suite=args.run_epoch_test_suite,
        epoch_test_split=args.epoch_test_split,
        epoch_test_n_samples=args.epoch_test_n_samples,
        epoch_test_every_n_epochs=args.epoch_test_every_n_epochs,
        seed=args.seed,
    )
    validate_experiment_config(config, task="semantic")
    return config


def build_config(
    *,
    data_dict: dict[str, Any] | None = None,
    data_root: str | Path | None = None,
    base_output_dir: str | Path | None = None,
    dino_checkpoint: str | Path | None = None,
    toy_lightning_checkpoint: str | Path | None = None,
    graha_base_output_dir: str | Path | None = None,
    graha_pretrain_dir: str | Path | None = None,
    graha_lightning_checkpoint: str | Path | None = None,
    gfft_config_path: str | Path | None = None,
    gfft_backbone_checkpoint: str | Path | None = None,
    **overrides: Any,
) -> SemanticSegmentationExperimentConfig:
    """Build a semantic segmentation experiment config from explicit values."""
    args = create_semantic_experiment_parser().parse_args([])
    data_dict_overrides = resolve_data_dictionary(data_dict)
    path_values = {
        "data_root": data_root,
        "base_output_dir": base_output_dir,
        "dino_checkpoint": dino_checkpoint,
        "toy_lightning_checkpoint": toy_lightning_checkpoint,
        "graha_base_output_dir": graha_base_output_dir,
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
            raise TypeError(f"Unknown semantic data dictionary option: {name}")
        setattr(args, name, value)

    for name, value in overrides.items():
        if not hasattr(args, name):
            raise TypeError(f"Unknown semantic experiment config option: {name}")
        setattr(args, name, value)

    return build_config_from_args(args)
