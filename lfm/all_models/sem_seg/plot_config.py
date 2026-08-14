"""Semantic segmentation checkpoint comparison plot configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from lfm.all_models.all_tasks import CheckpointRecord, discover_checkpoints
from lfm.all_models.all_tasks import config_defaults as defaults
from lfm.all_models.sem_seg.config import (
    SemanticSegmentationExperimentConfig,
    build_config,
)


@dataclass(frozen=True)
class SemanticModelPlotSpec:
    key: str
    display_name: str
    model_family: str
    checkpoint_path: Path


@dataclass(frozen=True)
class SemanticCheckpointComparisonPlotConfig:
    experiment_config: SemanticSegmentationExperimentConfig
    output_dir: Path
    model_specs: list[SemanticModelPlotSpec]
    n_samples: int


def _final_checkpoint_from_dir(checkpoint_dir: Path) -> CheckpointRecord:
    try:
        checkpoints = discover_checkpoints(checkpoint_dir)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not resolve a final checkpoint from {checkpoint_dir}. "
            "If this came from a three-model run, inspect the corresponding "
            "model job log and confirm that training wrote at least one .ckpt file."
        ) from exc
    return checkpoints[-1]


def _resolve_checkpoint(
    *,
    checkpoint_path: Path | None,
    checkpoint_dir: Path | None,
    label: str,
) -> Path | None:
    if checkpoint_path is not None and checkpoint_dir is not None:
        raise ValueError(
            f"Pass either --{label}-checkpoint or --{label}-checkpoint-dir, not both."
        )
    if checkpoint_path is not None:
        checkpoint_path = checkpoint_path.resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"{label} checkpoint does not exist: {checkpoint_path}"
            )
        return checkpoint_path
    if checkpoint_dir is None:
        return None
    record = _final_checkpoint_from_dir(checkpoint_dir)
    print(f"[{label}] selected final checkpoint: {record.path}", flush=True)
    return record.path


def _apply_run_root_defaults(args: argparse.Namespace) -> None:
    if args.run_root is None:
        return
    run_root = args.run_root.resolve()
    if args.toy_checkpoint_dir is None and args.toy_checkpoint is None:
        args.toy_checkpoint_dir = run_root / "checkpoints" / "toy_model"
    if args.graha_checkpoint_dir is None and args.graha_checkpoint is None:
        args.graha_checkpoint_dir = run_root / "checkpoints" / "full_model"
    if args.gfft_checkpoint_dir is None and args.gfft_checkpoint is None:
        args.gfft_checkpoint_dir = run_root / "checkpoints" / "gfft_model"


def _build_model_specs(args: argparse.Namespace) -> list[SemanticModelPlotSpec]:
    _apply_run_root_defaults(args)
    specs = []
    toy = _resolve_checkpoint(
        checkpoint_path=args.toy_checkpoint,
        checkpoint_dir=args.toy_checkpoint_dir,
        label="toy",
    )
    if toy is not None:
        specs.append(
            SemanticModelPlotSpec(
                key="toy",
                display_name="Toy",
                model_family="toy",
                checkpoint_path=toy,
            )
        )

    graha = _resolve_checkpoint(
        checkpoint_path=args.graha_checkpoint,
        checkpoint_dir=args.graha_checkpoint_dir,
        label="graha",
    )
    if graha is not None:
        specs.append(
            SemanticModelPlotSpec(
                key="graha",
                display_name="Graha",
                model_family="graha",
                checkpoint_path=graha,
            )
        )

    gfft = _resolve_checkpoint(
        checkpoint_path=args.gfft_checkpoint,
        checkpoint_dir=args.gfft_checkpoint_dir,
        label="gfft",
    )
    if gfft is not None:
        specs.append(
            SemanticModelPlotSpec(
                key="gfft",
                display_name="GFFT",
                model_family="gfft",
                checkpoint_path=gfft,
            )
        )

    if len(specs) < 2:
        raise ValueError("At least two model checkpoints are required for comparison.")
    return specs


def build_checkpoint_comparison_plot_config_from_args(
    args: argparse.Namespace,
) -> SemanticCheckpointComparisonPlotConfig:
    output_dir = args.output_dir.resolve()
    experiment_config = build_config(
        data_root=args.data_root.resolve(),
        base_output_dir=output_dir / "_setup",
        dino_checkpoint=args.dino_checkpoint,
        gfft_config_path=args.gfft_config_path,
        gfft_backbone_checkpoint=args.gfft_backbone_checkpoint,
        graha_base_output_dir=output_dir / "_graha_setup",
        graha_pretrain_dir=args.graha_pretrain_dir,
        dataset_modality=args.dataset_modality,
        band_filter=args.band_filter,
        target_size=args.target_size,
        semantic_label_source=args.semantic_label_source,
        image_glob=args.image_glob,
        label_glob=args.label_glob,
        image_suffix=args.image_suffix,
        label_suffix=args.label_suffix,
        max_train_samples=(
            args.max_samples if args.prediction_split == "train" else None
        ),
        max_val_samples=args.max_samples if args.prediction_split == "val" else None,
        max_test_samples=args.max_samples if args.prediction_split == "test" else None,
        ignore_nodata_in_loss=args.ignore_nodata_in_loss,
        nodata_ignore_index=args.nodata_ignore_index,
        excluded_nodata_values=args.excluded_nodata_values,
        image_nodata_policy=args.image_nodata_policy,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=1,
        learning_rate=defaults.DEFAULT_SEMANTIC_LEARNING_RATE,
        weight_decay=defaults.DEFAULT_SEMANTIC_WEIGHT_DECAY,
        normalize_inputs=args.normalize_inputs,
        normalization_source=args.normalization_source,
        normalization_modality=args.normalization_modality,
        cache_predictions=True,
        prediction_split=args.prediction_split,
        prediction_n_samples=args.n_samples,
        graha_input_modality_mode=args.graha_input_modality_mode,
        graha_vis_uv_merge_method=args.graha_vis_uv_merge_method,
        graha_shape_loss_weight=args.graha_shape_loss_weight,
        graha_shape_loss_pad_frac=args.graha_shape_loss_pad_frac,
        graha_stats_batch_size=args.graha_stats_batch_size,
        graha_batch_size=args.graha_batch_size,
        graha_num_workers=args.graha_num_workers,
        progress_log_every_n_batches=10**9,
        skip_toy_fit=True,
        skip_graha_fit=True,
        no_fit=True,
        run_epoch_test_suite=False,
        seed=args.seed,
    )
    return SemanticCheckpointComparisonPlotConfig(
        experiment_config=experiment_config,
        output_dir=output_dir,
        model_specs=_build_model_specs(args),
        n_samples=args.n_samples,
    )
