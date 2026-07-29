"""Semantic segmentation checkpoint sweep configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lfm.all_models.all_tasks import config_defaults as defaults
from lfm.all_models.sem_seg.config import (
    SemanticSegmentationExperimentConfig,
    build_config,
)


@dataclass(frozen=True)
class SemanticCheckpointSweepConfig:
    experiment_config: SemanticSegmentationExperimentConfig
    output_root: Path
    toy_checkpoint_dir: Path | None
    graha_checkpoint_dir: Path | None
    models: list[str]
    max_checkpoints: int | None
    verbose: bool
    preload_test_batches: bool

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
) -> SemanticCheckpointSweepConfig:
    lfm_root = Path(__file__).resolve().parents[3]
    scripts_output_dir = lfm_root / "scripts" / "outputs"
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else scripts_output_dir / "semantic_checkpoint_sweep"
    )
    experiment_config = build_config(
        data_root=Path(args.data_root).resolve() if args.data_root else None,
        base_output_dir=output_root / "_toy_setup",
        dino_checkpoint=args.dino_checkpoint,
        graha_base_output_dir=output_root / "_graha_setup",
        graha_pretrain_dir=args.graha_pretrain_dir,
        band_filter=args.band_filter,
        target_size=args.target_size,
        semantic_label_source=getattr(
            args,
            "semantic_label_source",
            defaults.DEFAULT_SEMANTIC_LABEL_SOURCE,
        ),
        image_glob=args.image_glob,
        label_glob=args.label_glob,
        image_suffix=args.image_suffix,
        label_suffix=args.label_suffix,
        max_train_samples=None,
        max_val_samples=None,
        max_test_samples=args.max_test_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=1,
        normalize_inputs=args.normalize_inputs,
        normalization_source=getattr(
            args,
            "normalization_source",
            defaults.DEFAULT_NORMALIZATION_SOURCE,
        ),
        normalization_modality=getattr(
            args,
            "normalization_modality",
            defaults.DEFAULT_NORMALIZATION_MODALITY,
        ),
        prediction_split=defaults.DEFAULT_SWEEP_SPLIT,
        prediction_n_samples=defaults.DEFAULT_SEMANTIC_PREDICTION_N_SAMPLES,
        graha_input_modality_mode=args.graha_input_modality_mode,
        graha_vis_uv_merge_method=args.graha_vis_uv_merge_method,
        graha_stats_batch_size=args.graha_stats_batch_size,
        graha_batch_size=args.graha_batch_size,
        graha_num_workers=args.graha_num_workers,
        seed=args.seed,
        no_fit=True,
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
    )
    return SemanticCheckpointSweepConfig(
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
        max_checkpoints=args.max_checkpoints,
        verbose=getattr(args, "verbose", False),
        preload_test_batches=getattr(args, "preload_test_batches", True),
    )
