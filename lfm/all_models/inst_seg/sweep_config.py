"""Instance segmentation checkpoint sweep configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lfm.all_models.all_tasks import config_defaults as defaults
from lfm.all_models.inst_seg.config import (
    InstanceSegmentationExperimentConfig,
    build_config,
)


@dataclass(frozen=True)
class InstanceCheckpointSweepConfig:
    experiment_config: InstanceSegmentationExperimentConfig
    output_root: Path
    toy_checkpoint_dir: Path | None
    graha_checkpoint_dir: Path | None
    models: list[str]
    max_samples: int | None
    max_checkpoints: int | None
    verbose: bool

    def __getattr__(self, name: str) -> Any:
        return getattr(self.experiment_config, name)


def _validate_models(models: list[str]) -> list[str]:
    normalized = [model.lower() for model in models]
    unknown = sorted(set(normalized) - set(defaults.MODEL_CHOICES))
    if unknown:
        raise ValueError(f"Unknown model name(s): {unknown}")
    return normalized


def build_checkpoint_sweep_config_from_args(
    args: argparse.Namespace,
) -> InstanceCheckpointSweepConfig:
    lfm_root = Path(__file__).resolve().parents[3]
    scripts_output_dir = lfm_root / "scripts" / "outputs"
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else scripts_output_dir / "instance_checkpoint_sweep"
    )
    experiment_config = build_config(
        data_root=args.data_root,
        base_output_dir=output_root / "_setup",
        dino_checkpoint=args.dino_checkpoint,
        graha_pretrain_dir=args.graha_pretrain_dir,
        dataset_modality=args.dataset_modality,
        graha_input_modality_mode=args.graha_input_modality_mode,
        graha_backend_modalities=getattr(args, "graha_backend_modalities", None),
        graha_vis_uv_merge_method=args.graha_vis_uv_merge_method,
        normalization_source=getattr(
            args,
            "normalization_source",
            defaults.DEFAULT_NORMALIZATION_SOURCE,
        ),
        normalization_modality=args.normalization_modality,
        target_size=args.target_size,
        band_filter=args.band_filter,
        image_glob=args.image_glob,
        label_glob=args.label_glob,
        image_suffix=args.image_suffix,
        label_suffix=args.label_suffix,
        max_train_samples=None,
        max_val_samples=None,
        max_test_samples=args.max_samples,
        toy_batch_size=args.toy_batch_size,
        toy_num_workers=args.toy_num_workers,
        graha_stats_batch_size=args.graha_stats_batch_size,
        graha_batch_size=args.graha_batch_size,
        graha_num_workers=args.graha_num_workers,
        max_epochs=1,
        toy_learning_rate=defaults.DEFAULT_INSTANCE_LEARNING_RATE,
        toy_weight_decay=defaults.DEFAULT_INSTANCE_WEIGHT_DECAY,
        toy_freeze_backbone=False,
        toy_normalize_inputs=args.toy_normalize_inputs,
        toy_architecture=args.toy_architecture,
        toy_gradient_clip_val=defaults.DEFAULT_TOY_GRADIENT_CLIP_VAL,
        disable_toy_gradient_clipping=True,
        graha_backbone_lr=args.graha_backbone_lr,
        graha_head_lr=args.graha_head_lr,
        graha_layer_decay=args.graha_layer_decay,
        graha_weight_decay=args.graha_weight_decay,
        graha_warmup_steps=args.graha_warmup_steps,
        graha_anchor_sizes=args.graha_anchor_sizes,
        graha_anchor_aspect_ratios=args.graha_anchor_aspect_ratios,
        graha_score_threshold=args.graha_score_threshold,
        plot_every_n_epochs=0,
        plot_n_samples=defaults.DEFAULT_PLOT_N_SAMPLES,
        progress_log_every_n_batches=10**9,
        prediction_split=args.prediction_split,
        prediction_n_samples=(
            args.max_samples if args.max_samples is not None else 10**9
        ),
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
        image_nodata_policy=getattr(
            args,
            "image_nodata_policy",
            defaults.DEFAULT_IMAGE_NODATA_POLICY,
        ),
        skip_toy_fit=True,
        skip_graha_fit=True,
        no_fit=True,
        run_epoch_test_suite=False,
        epoch_test_split=args.prediction_split,
        epoch_test_n_samples=(
            args.max_samples if args.max_samples is not None else 10**9
        ),
        epoch_test_every_n_epochs=defaults.DEFAULT_EPOCH_TEST_EVERY_N_EPOCHS,
        seed=args.seed,
    )
    return InstanceCheckpointSweepConfig(
        experiment_config=experiment_config,
        output_root=output_root,
        toy_checkpoint_dir=(
            Path(args.toy_checkpoint_dir).resolve() if args.toy_checkpoint_dir else None
        ),
        graha_checkpoint_dir=(
            Path(args.graha_checkpoint_dir).resolve()
            if args.graha_checkpoint_dir
            else None
        ),
        models=_validate_models(args.models),
        max_samples=args.max_samples,
        max_checkpoints=args.max_checkpoints,
        verbose=args.verbose,
    )
