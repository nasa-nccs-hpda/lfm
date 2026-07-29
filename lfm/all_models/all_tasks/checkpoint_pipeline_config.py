"""Config and command builders for train-then-checkpoint-sweep pipelines."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class CheckpointPipelineConfig:
    """Resolved settings for a train-then-checkpoint-sweep pipeline."""

    task: str
    base_output_dir: Path
    existing_training_output_dir: Path | None
    sweep_output_root: Path | None
    skip_sweep: bool
    models: tuple[str, ...]
    training_command: tuple[str, ...]
    sweep_command_args: tuple[str, ...]
    sweep_script: str


@dataclass(frozen=True)
class CheckpointPipelineResult:
    """Runtime summary for a train-then-checkpoint-sweep pipeline."""

    task: str
    training_output_dir: str
    sweep_output_dir: str | None
    training_seconds: float | None
    sweep_seconds: float | None


def append_flag(command: list[str], flag: str, value: Any) -> None:
    """Append a CLI flag/value pair, preserving argparse list and boolean behavior."""
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    if isinstance(value, (list, tuple)):
        if not value:
            return
        command.append(flag)
        command.extend(str(item) for item in value)
        return
    command.extend([flag, str(value)])


def checkpoint_dir(training_output_dir: Path, model: str) -> Path:
    subdir = "toy_model" if model == "toy" else "full_model"
    checkpoint_directory = training_output_dir / "checkpoints" / subdir
    if not checkpoint_directory.exists():
        raise FileNotFoundError(
            f"Expected {model} checkpoint directory does not exist: "
            f"{checkpoint_directory}"
        )
    return checkpoint_directory


def build_checkpoint_pipeline_config_from_args(
    args: argparse.Namespace,
    *,
    repo_root: Path | None = None,
    python_executable: str | None = None,
) -> CheckpointPipelineConfig:
    """Convert parsed CLI args into a reusable pipeline config object."""
    resolved_repo_root = repo_root or Path.cwd()
    executable = python_executable or sys.executable
    default_base = (
        resolved_repo_root
        / "scripts"
        / "outputs"
        / f"{args.task}_train_then_checkpoint_sweep"
    )
    base_output_dir = (
        Path(args.base_output_dir).resolve() if args.base_output_dir else default_base
    )
    existing_training_output_dir = (
        Path(args.existing_training_output_dir).resolve()
        if args.existing_training_output_dir
        else None
    )
    sweep_output_root = (
        Path(args.sweep_output_root).resolve() if args.sweep_output_root else None
    )

    if args.task == "semantic":
        training_command = _semantic_comparison_command(
            args,
            base_output_dir,
            python_executable=executable,
        )
        sweep_script = "scripts/python/semantic_seg/semantic_checkpoint_sweep.py"
        sweep_command_args = _semantic_sweep_command_args(args)
    else:
        training_command = _instance_comparison_command(
            args,
            base_output_dir,
            python_executable=executable,
        )
        sweep_script = "scripts/python/instance_seg/instance_checkpoint_sweep.py"
        sweep_command_args = _instance_sweep_command_args(args)

    return CheckpointPipelineConfig(
        task=args.task,
        base_output_dir=base_output_dir,
        existing_training_output_dir=existing_training_output_dir,
        sweep_output_root=sweep_output_root,
        skip_sweep=args.skip_sweep,
        models=tuple(args.models),
        training_command=tuple(training_command),
        sweep_command_args=tuple(sweep_command_args),
        sweep_script=sweep_script,
    )


def build_sweep_command(
    config: CheckpointPipelineConfig,
    training_output_dir: Path,
    sweep_output_dir: Path,
    *,
    python_executable: str | None = None,
) -> list[str]:
    """Build the checkpoint sweep command after the training output is known."""
    executable = python_executable or sys.executable
    command = [
        executable,
        "-u",
        config.sweep_script,
        "--output-root",
        str(sweep_output_dir),
    ]
    if "toy" in config.models:
        append_flag(
            command,
            "--toy-checkpoint-dir",
            checkpoint_dir(training_output_dir, "toy"),
        )
    if "graha" in config.models:
        append_flag(
            command,
            "--graha-checkpoint-dir",
            checkpoint_dir(training_output_dir, "graha"),
        )
    command.extend(config.sweep_command_args)
    return command


def _common_comparison_args(args: argparse.Namespace) -> list[str]:
    command: list[str] = []
    append_flag(command, "--simlink-dest", args.simlink_dest)
    append_flag(command, "--data-root", args.data_root)
    append_flag(command, "--dataset-modality", args.dataset_modality)
    append_flag(command, "--dino-checkpoint", args.dino_checkpoint)
    append_flag(command, "--graha-pretrain-dir", args.graha_pretrain_dir)
    append_flag(command, "--graha-input-modality-mode", args.graha_input_modality_mode)
    append_flag(command, "--graha-vis-uv-merge-method", args.graha_vis_uv_merge_method)
    append_flag(command, "--normalization-source", args.normalization_source)
    append_flag(command, "--normalization-modality", args.normalization_modality)
    append_flag(command, "--target-size", args.target_size)
    append_flag(command, "--band-filter", args.band_filter)
    append_flag(command, "--max-train-samples", args.max_train_samples)
    append_flag(command, "--max-val-samples", args.max_val_samples)
    append_flag(command, "--max-test-samples", args.max_test_samples)
    append_flag(command, "--max-epochs", args.max_epochs)
    append_flag(command, "--plot-every-n-epochs", args.plot_every_n_epochs)
    append_flag(command, "--plot-n-samples", args.plot_n_samples)
    append_flag(command, "--prediction-split", args.prediction_split)
    append_flag(command, "--prediction-n-samples", args.prediction_n_samples)
    append_flag(command, "--run-epoch-test-suite", args.run_epoch_test_suite)
    append_flag(command, "--epoch-test-split", args.epoch_test_split)
    append_flag(command, "--epoch-test-n-samples", args.epoch_test_n_samples)
    append_flag(command, "--epoch-test-every-n-epochs", args.epoch_test_every_n_epochs)
    append_flag(command, "--seed", args.seed)
    return command


def _semantic_comparison_command(
    args: argparse.Namespace,
    base_output_dir: Path,
    *,
    python_executable: str,
) -> list[str]:
    command = [
        python_executable,
        "-u",
        "scripts/python/semantic_seg/semantic_seg_comparison.py",
        "--base-output-dir",
        str(base_output_dir),
    ]
    command.extend(_common_comparison_args(args))
    append_flag(command, "--semantic-label-source", args.semantic_label_source)
    append_flag(command, "--image-glob", args.image_glob)
    append_flag(command, "--label-glob", args.label_glob)
    append_flag(command, "--image-suffix", args.image_suffix)
    append_flag(command, "--label-suffix", args.label_suffix)
    append_flag(command, "--ignore-nodata-in-loss", args.ignore_nodata_in_loss)
    append_flag(command, "--nodata-ignore-index", args.nodata_ignore_index)
    append_flag(command, "--batch-size", args.batch_size)
    append_flag(command, "--num-workers", args.num_workers)
    append_flag(command, "--learning-rate", args.learning_rate)
    append_flag(command, "--weight-decay", args.weight_decay)
    append_flag(command, "--toy-loss-type", args.toy_loss_type)
    append_flag(command, "--use-toy-shape-loss", args.use_toy_shape_loss)
    append_flag(command, "--toy-shape-loss-weight", args.toy_shape_loss_weight)
    append_flag(command, "--toy-shape-loss-pad-frac", args.toy_shape_loss_pad_frac)
    append_flag(command, "--graha-shape-loss-weight", args.graha_shape_loss_weight)
    append_flag(command, "--graha-shape-loss-pad-frac", args.graha_shape_loss_pad_frac)
    append_flag(command, "--normalize-inputs", args.normalize_inputs)
    append_flag(
        command, "--disable-toy-gradient-clipping", args.disable_toy_gradient_clipping
    )
    append_flag(command, "--graha-batch-size", args.graha_batch_size)
    append_flag(command, "--graha-num-workers", args.graha_num_workers)
    append_flag(command, "--graha-stats-batch-size", args.graha_stats_batch_size)
    append_flag(
        command, "--progress-log-every-n-batches", args.progress_log_every_n_batches
    )
    command.extend(args.comparison_extra_arg)
    return command


def _instance_comparison_command(
    args: argparse.Namespace,
    base_output_dir: Path,
    *,
    python_executable: str,
) -> list[str]:
    command = [
        python_executable,
        "-u",
        "scripts/python/instance_seg/instance_seg_comparison.py",
        "--base-output-dir",
        str(base_output_dir),
    ]
    command.extend(_common_comparison_args(args))
    append_flag(command, "--image-glob", args.image_glob)
    append_flag(command, "--label-glob", args.label_glob)
    append_flag(command, "--image-suffix", args.image_suffix)
    append_flag(command, "--label-suffix", args.label_suffix)
    append_flag(command, "--toy-batch-size", args.toy_batch_size)
    append_flag(command, "--toy-num-workers", args.toy_num_workers)
    append_flag(command, "--toy-learning-rate", args.toy_learning_rate)
    append_flag(command, "--toy-weight-decay", args.toy_weight_decay)
    append_flag(command, "--toy-normalize-inputs", args.toy_normalize_inputs)
    append_flag(command, "--toy-architecture", args.toy_architecture)
    append_flag(
        command, "--disable-toy-gradient-clipping", args.disable_toy_gradient_clipping
    )
    append_flag(command, "--graha-batch-size", args.graha_batch_size)
    append_flag(command, "--graha-num-workers", args.graha_num_workers)
    append_flag(command, "--graha-stats-batch-size", args.graha_stats_batch_size)
    append_flag(command, "--graha-backbone-lr", args.graha_backbone_lr)
    append_flag(command, "--graha-head-lr", args.graha_head_lr)
    append_flag(command, "--graha-layer-decay", args.graha_layer_decay)
    append_flag(command, "--graha-weight-decay", args.graha_weight_decay)
    append_flag(command, "--graha-warmup-steps", args.graha_warmup_steps)
    append_flag(command, "--graha-anchor-sizes", args.graha_anchor_sizes)
    append_flag(
        command, "--graha-anchor-aspect-ratios", args.graha_anchor_aspect_ratios
    )
    append_flag(command, "--graha-score-threshold", args.graha_score_threshold)
    append_flag(
        command, "--prediction-score-threshold", args.prediction_score_threshold
    )
    append_flag(
        command, "--progress-log-every-n-batches", args.progress_log_every_n_batches
    )
    append_flag(command, "--mask-shift", args.mask_shift)
    command.extend(args.comparison_extra_arg)
    return command


def _semantic_sweep_command_args(args: argparse.Namespace) -> list[str]:
    command: list[str] = []
    append_flag(command, "--simlink-dest", args.simlink_dest)
    append_flag(command, "--data-root", args.data_root)
    append_flag(command, "--models", args.models)
    append_flag(command, "--dataset-modality", args.dataset_modality)
    append_flag(command, "--band-filter", args.band_filter)
    append_flag(command, "--target-size", args.target_size)
    append_flag(command, "--semantic-label-source", args.semantic_label_source)
    append_flag(command, "--image-glob", args.image_glob)
    append_flag(command, "--label-glob", args.label_glob)
    append_flag(command, "--image-suffix", args.image_suffix)
    append_flag(command, "--label-suffix", args.label_suffix)
    append_flag(command, "--batch-size", args.batch_size)
    append_flag(command, "--num-workers", args.num_workers)
    append_flag(command, "--normalize-inputs", args.normalize_inputs)
    append_flag(command, "--normalization-source", args.normalization_source)
    append_flag(command, "--normalization-modality", args.normalization_modality)
    append_flag(command, "--max-test-samples", args.sweep_max_samples)
    append_flag(command, "--ignore-nodata-in-loss", args.ignore_nodata_in_loss)
    append_flag(command, "--nodata-ignore-index", args.nodata_ignore_index)
    append_flag(command, "--dino-checkpoint", args.dino_checkpoint)
    append_flag(command, "--graha-pretrain-dir", args.graha_pretrain_dir)
    append_flag(command, "--graha-input-modality-mode", args.graha_input_modality_mode)
    append_flag(command, "--graha-vis-uv-merge-method", args.graha_vis_uv_merge_method)
    append_flag(command, "--graha-stats-batch-size", args.graha_stats_batch_size)
    append_flag(command, "--graha-batch-size", args.graha_batch_size)
    append_flag(command, "--graha-num-workers", args.graha_num_workers)
    append_flag(command, "--max-checkpoints", args.max_checkpoints)
    append_flag(command, "--seed", args.seed)
    append_flag(command, "--verbose", args.verbose)
    command.extend(args.sweep_extra_arg)
    return command


def _instance_sweep_command_args(args: argparse.Namespace) -> list[str]:
    command: list[str] = []
    append_flag(command, "--simlink-dest", args.simlink_dest)
    append_flag(command, "--data-root", args.data_root)
    append_flag(command, "--models", args.models)
    append_flag(command, "--dataset-modality", args.dataset_modality)
    append_flag(command, "--target-size", args.target_size)
    append_flag(command, "--band-filter", args.band_filter)
    append_flag(command, "--image-glob", args.image_glob)
    append_flag(command, "--label-glob", args.label_glob)
    append_flag(command, "--image-suffix", args.image_suffix)
    append_flag(command, "--label-suffix", args.label_suffix)
    append_flag(command, "--max-samples", args.sweep_max_samples)
    append_flag(command, "--toy-batch-size", args.toy_batch_size)
    append_flag(command, "--toy-num-workers", args.toy_num_workers)
    append_flag(command, "--toy-normalize-inputs", args.toy_normalize_inputs)
    append_flag(command, "--normalization-source", args.normalization_source)
    append_flag(command, "--normalization-modality", args.normalization_modality)
    append_flag(command, "--toy-architecture", args.toy_architecture)
    append_flag(command, "--dino-checkpoint", args.dino_checkpoint)
    append_flag(command, "--graha-pretrain-dir", args.graha_pretrain_dir)
    append_flag(command, "--graha-input-modality-mode", args.graha_input_modality_mode)
    append_flag(command, "--graha-vis-uv-merge-method", args.graha_vis_uv_merge_method)
    append_flag(command, "--graha-stats-batch-size", args.graha_stats_batch_size)
    append_flag(command, "--graha-batch-size", args.graha_batch_size)
    append_flag(command, "--graha-num-workers", args.graha_num_workers)
    append_flag(command, "--graha-backbone-lr", args.graha_backbone_lr)
    append_flag(command, "--graha-head-lr", args.graha_head_lr)
    append_flag(command, "--graha-layer-decay", args.graha_layer_decay)
    append_flag(command, "--graha-weight-decay", args.graha_weight_decay)
    append_flag(command, "--graha-warmup-steps", args.graha_warmup_steps)
    append_flag(command, "--graha-anchor-sizes", args.graha_anchor_sizes)
    append_flag(
        command, "--graha-anchor-aspect-ratios", args.graha_anchor_aspect_ratios
    )
    append_flag(command, "--graha-score-threshold", args.graha_score_threshold)
    append_flag(command, "--prediction-split", args.sweep_split)
    append_flag(
        command, "--prediction-score-threshold", args.prediction_score_threshold
    )
    append_flag(command, "--mask-shift", args.mask_shift)
    append_flag(command, "--max-checkpoints", args.max_checkpoints)
    append_flag(command, "--seed", args.seed)
    append_flag(command, "--verbose", args.verbose)
    command.extend(args.sweep_extra_arg)
    return command


def command_to_display(command: Sequence[str]) -> str:
    return " ".join(str(part) for part in command)
