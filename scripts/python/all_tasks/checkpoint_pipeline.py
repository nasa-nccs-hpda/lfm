"""Run comparison training, then sweep the checkpoints from that run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class PipelineResult:
    task: str
    training_output_dir: str
    sweep_output_dir: str | None
    training_seconds: float | None
    sweep_seconds: float | None


def _append_flag(command: list[str], flag: str, value) -> None:
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


def _run_command(command: Sequence[str]) -> float:
    print("\nRunning command:", flush=True)
    print(" ".join(str(part) for part in command), flush=True)
    started_at = time.perf_counter()
    subprocess.run(command, check=True)
    elapsed = time.perf_counter() - started_at
    print(f"Command finished in {elapsed:.1f}s", flush=True)
    return elapsed


def _snapshot_children(path: Path) -> set[Path]:
    if not path.exists():
        return set()
    return {child.resolve() for child in path.iterdir() if child.is_dir()}


def _latest_child(path: Path, *, exclude: set[Path]) -> Path:
    candidates = [
        child
        for child in path.iterdir()
        if child.is_dir() and child.resolve() not in exclude
    ]
    if not candidates:
        candidates = [child for child in path.iterdir() if child.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No output directory was created under {path}")
    return max(candidates, key=lambda child: child.stat().st_mtime).resolve()


def _checkpoint_dir(training_output_dir: Path, model: str) -> Path:
    subdir = "toy_model" if model == "toy" else "full_model"
    checkpoint_dir = training_output_dir / "checkpoints" / subdir
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Expected {model} checkpoint directory does not exist: {checkpoint_dir}"
        )
    return checkpoint_dir


def _common_comparison_args(args: argparse.Namespace) -> list[str]:
    command: list[str] = []
    _append_flag(command, "--simlink-dest", args.simlink_dest)
    _append_flag(command, "--data-root", args.data_root)
    _append_flag(command, "--dino-checkpoint", args.dino_checkpoint)
    _append_flag(command, "--graha-pretrain-dir", args.graha_pretrain_dir)
    _append_flag(command, "--graha-input-modality-mode", args.graha_input_modality_mode)
    _append_flag(command, "--graha-vis-uv-merge-method", args.graha_vis_uv_merge_method)
    _append_flag(command, "--normalization-source", args.normalization_source)
    _append_flag(command, "--normalization-modality", args.normalization_modality)
    _append_flag(command, "--target-size", args.target_size)
    _append_flag(command, "--band-filter", args.band_filter)
    _append_flag(command, "--max-train-samples", args.max_train_samples)
    _append_flag(command, "--max-val-samples", args.max_val_samples)
    _append_flag(command, "--max-test-samples", args.max_test_samples)
    _append_flag(command, "--max-epochs", args.max_epochs)
    _append_flag(command, "--plot-every-n-epochs", args.plot_every_n_epochs)
    _append_flag(command, "--plot-n-samples", args.plot_n_samples)
    _append_flag(command, "--prediction-split", args.prediction_split)
    _append_flag(command, "--prediction-n-samples", args.prediction_n_samples)
    _append_flag(command, "--run-epoch-test-suite", args.run_epoch_test_suite)
    _append_flag(command, "--epoch-test-split", args.epoch_test_split)
    _append_flag(command, "--epoch-test-n-samples", args.epoch_test_n_samples)
    _append_flag(command, "--epoch-test-every-n-epochs", args.epoch_test_every_n_epochs)
    _append_flag(command, "--seed", args.seed)
    return command


def _semantic_comparison_command(
    args: argparse.Namespace, base_output_dir: Path
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "scripts/python/semantic_seg/semantic_seg_comparison.py",
        "--base-output-dir",
        str(base_output_dir),
    ]
    command.extend(_common_comparison_args(args))
    _append_flag(command, "--spatial-transform", args.spatial_transform)
    _append_flag(command, "--semantic-label-source", args.semantic_label_source)
    _append_flag(command, "--image-glob", args.image_glob)
    _append_flag(command, "--label-glob", args.label_glob)
    _append_flag(command, "--image-suffix", args.image_suffix)
    _append_flag(command, "--label-suffix", args.label_suffix)
    _append_flag(command, "--ignore-nodata-in-loss", args.ignore_nodata_in_loss)
    _append_flag(command, "--nodata-ignore-index", args.nodata_ignore_index)
    _append_flag(command, "--batch-size", args.batch_size)
    _append_flag(command, "--num-workers", args.num_workers)
    _append_flag(command, "--learning-rate", args.learning_rate)
    _append_flag(command, "--weight-decay", args.weight_decay)
    _append_flag(command, "--loss-type", args.loss_type)
    _append_flag(command, "--use-toy-shape-loss", args.use_toy_shape_loss)
    _append_flag(command, "--toy-shape-loss-weight", args.toy_shape_loss_weight)
    _append_flag(command, "--toy-shape-loss-pad-frac", args.toy_shape_loss_pad_frac)
    _append_flag(command, "--graha-shape-loss-weight", args.graha_shape_loss_weight)
    _append_flag(command, "--graha-shape-loss-pad-frac", args.graha_shape_loss_pad_frac)
    _append_flag(command, "--normalize-inputs", args.normalize_inputs)
    _append_flag(
        command, "--disable-toy-gradient-clipping", args.disable_toy_gradient_clipping
    )
    _append_flag(command, "--graha-batch-size", args.graha_batch_size)
    _append_flag(command, "--graha-num-workers", args.graha_num_workers)
    _append_flag(command, "--graha-stats-batch-size", args.graha_stats_batch_size)
    _append_flag(
        command, "--progress-log-every-n-batches", args.progress_log_every_n_batches
    )
    command.extend(args.comparison_extra_arg)
    return command


def _instance_comparison_command(
    args: argparse.Namespace, base_output_dir: Path
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "scripts/python/instance_seg/instance_seg_comparison.py",
        "--base-output-dir",
        str(base_output_dir),
    ]
    command.extend(_common_comparison_args(args))
    _append_flag(command, "--image-glob", args.image_glob)
    _append_flag(command, "--label-glob", args.label_glob)
    _append_flag(command, "--image-suffix", args.image_suffix)
    _append_flag(command, "--label-suffix", args.label_suffix)
    _append_flag(command, "--toy-batch-size", args.toy_batch_size)
    _append_flag(command, "--toy-num-workers", args.toy_num_workers)
    _append_flag(command, "--toy-learning-rate", args.toy_learning_rate)
    _append_flag(command, "--toy-weight-decay", args.toy_weight_decay)
    _append_flag(command, "--toy-normalize-inputs", args.toy_normalize_inputs)
    _append_flag(command, "--toy-architecture", args.toy_architecture)
    _append_flag(
        command, "--disable-toy-gradient-clipping", args.disable_toy_gradient_clipping
    )
    _append_flag(command, "--graha-batch-size", args.graha_batch_size)
    _append_flag(command, "--graha-num-workers", args.graha_num_workers)
    _append_flag(command, "--graha-stats-batch-size", args.graha_stats_batch_size)
    _append_flag(command, "--graha-backbone-lr", args.graha_backbone_lr)
    _append_flag(command, "--graha-head-lr", args.graha_head_lr)
    _append_flag(command, "--graha-layer-decay", args.graha_layer_decay)
    _append_flag(command, "--graha-weight-decay", args.graha_weight_decay)
    _append_flag(command, "--graha-warmup-steps", args.graha_warmup_steps)
    _append_flag(command, "--graha-anchor-sizes", args.graha_anchor_sizes)
    _append_flag(
        command, "--graha-anchor-aspect-ratios", args.graha_anchor_aspect_ratios
    )
    _append_flag(command, "--graha-score-threshold", args.graha_score_threshold)
    _append_flag(
        command, "--prediction-score-threshold", args.prediction_score_threshold
    )
    _append_flag(
        command, "--progress-log-every-n-batches", args.progress_log_every_n_batches
    )
    _append_flag(command, "--mask-shift", args.mask_shift)
    command.extend(args.comparison_extra_arg)
    return command


def _semantic_sweep_command(
    args: argparse.Namespace, training_output_dir: Path, sweep_output_dir: Path
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "scripts/python/semantic_seg/semantic_checkpoint_sweep.py",
        "--output-root",
        str(sweep_output_dir),
    ]
    if "toy" in args.models:
        _append_flag(
            command, "--toy-checkpoint-dir", _checkpoint_dir(training_output_dir, "toy")
        )
    if "graha" in args.models:
        _append_flag(
            command,
            "--graha-checkpoint-dir",
            _checkpoint_dir(training_output_dir, "graha"),
        )
    _append_flag(command, "--simlink-dest", args.simlink_dest)
    _append_flag(command, "--data-root", args.data_root)
    _append_flag(command, "--models", args.models)
    _append_flag(command, "--band-filter", args.band_filter)
    _append_flag(command, "--target-size", args.target_size)
    _append_flag(command, "--spatial-transform", args.spatial_transform)
    _append_flag(command, "--semantic-label-source", args.semantic_label_source)
    _append_flag(command, "--image-glob", args.image_glob)
    _append_flag(command, "--label-glob", args.label_glob)
    _append_flag(command, "--image-suffix", args.image_suffix)
    _append_flag(command, "--label-suffix", args.label_suffix)
    _append_flag(command, "--batch-size", args.batch_size)
    _append_flag(command, "--num-workers", args.num_workers)
    _append_flag(command, "--normalize-inputs", args.normalize_inputs)
    _append_flag(command, "--normalization-source", args.normalization_source)
    _append_flag(command, "--normalization-modality", args.normalization_modality)
    _append_flag(command, "--max-test-samples", args.sweep_max_samples)
    _append_flag(command, "--ignore-nodata-in-loss", args.ignore_nodata_in_loss)
    _append_flag(command, "--nodata-ignore-index", args.nodata_ignore_index)
    _append_flag(command, "--dino-checkpoint", args.dino_checkpoint)
    _append_flag(command, "--graha-pretrain-dir", args.graha_pretrain_dir)
    _append_flag(command, "--graha-input-modality-mode", args.graha_input_modality_mode)
    _append_flag(command, "--graha-vis-uv-merge-method", args.graha_vis_uv_merge_method)
    _append_flag(command, "--graha-stats-batch-size", args.graha_stats_batch_size)
    _append_flag(command, "--graha-batch-size", args.graha_batch_size)
    _append_flag(command, "--graha-num-workers", args.graha_num_workers)
    _append_flag(command, "--max-checkpoints", args.max_checkpoints)
    _append_flag(command, "--seed", args.seed)
    _append_flag(command, "--verbose", args.verbose)
    command.extend(args.sweep_extra_arg)
    return command


def _instance_sweep_command(
    args: argparse.Namespace, training_output_dir: Path, sweep_output_dir: Path
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "scripts/python/instance_seg/instance_checkpoint_sweep.py",
        "--output-root",
        str(sweep_output_dir),
    ]
    if "toy" in args.models:
        _append_flag(
            command, "--toy-checkpoint-dir", _checkpoint_dir(training_output_dir, "toy")
        )
    if "graha" in args.models:
        _append_flag(
            command,
            "--graha-checkpoint-dir",
            _checkpoint_dir(training_output_dir, "graha"),
        )
    _append_flag(command, "--simlink-dest", args.simlink_dest)
    _append_flag(command, "--data-root", args.data_root)
    _append_flag(command, "--models", args.models)
    _append_flag(command, "--target-size", args.target_size)
    _append_flag(command, "--band-filter", args.band_filter)
    _append_flag(command, "--image-glob", args.image_glob)
    _append_flag(command, "--label-glob", args.label_glob)
    _append_flag(command, "--image-suffix", args.image_suffix)
    _append_flag(command, "--label-suffix", args.label_suffix)
    _append_flag(command, "--max-samples", args.sweep_max_samples)
    _append_flag(command, "--toy-batch-size", args.toy_batch_size)
    _append_flag(command, "--toy-num-workers", args.toy_num_workers)
    _append_flag(command, "--toy-normalize-inputs", args.toy_normalize_inputs)
    _append_flag(command, "--normalization-source", args.normalization_source)
    _append_flag(command, "--normalization-modality", args.normalization_modality)
    _append_flag(command, "--toy-architecture", args.toy_architecture)
    _append_flag(command, "--dino-checkpoint", args.dino_checkpoint)
    _append_flag(command, "--graha-pretrain-dir", args.graha_pretrain_dir)
    _append_flag(command, "--graha-input-modality-mode", args.graha_input_modality_mode)
    _append_flag(command, "--graha-vis-uv-merge-method", args.graha_vis_uv_merge_method)
    _append_flag(command, "--graha-stats-batch-size", args.graha_stats_batch_size)
    _append_flag(command, "--graha-batch-size", args.graha_batch_size)
    _append_flag(command, "--graha-num-workers", args.graha_num_workers)
    _append_flag(command, "--graha-backbone-lr", args.graha_backbone_lr)
    _append_flag(command, "--graha-head-lr", args.graha_head_lr)
    _append_flag(command, "--graha-layer-decay", args.graha_layer_decay)
    _append_flag(command, "--graha-weight-decay", args.graha_weight_decay)
    _append_flag(command, "--graha-warmup-steps", args.graha_warmup_steps)
    _append_flag(command, "--graha-anchor-sizes", args.graha_anchor_sizes)
    _append_flag(
        command, "--graha-anchor-aspect-ratios", args.graha_anchor_aspect_ratios
    )
    _append_flag(command, "--graha-score-threshold", args.graha_score_threshold)
    _append_flag(command, "--prediction-split", args.sweep_split)
    _append_flag(
        command, "--prediction-score-threshold", args.prediction_score_threshold
    )
    _append_flag(command, "--mask-shift", args.mask_shift)
    _append_flag(command, "--max-checkpoints", args.max_checkpoints)
    _append_flag(command, "--seed", args.seed)
    _append_flag(command, "--verbose", args.verbose)
    command.extend(args.sweep_extra_arg)
    return command


def run_pipeline(args: argparse.Namespace) -> PipelineResult:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    default_base = (
        repo_root / "scripts" / "outputs" / f"{args.task}_train_then_checkpoint_sweep"
    )
    base_output_dir = (
        Path(args.base_output_dir).resolve() if args.base_output_dir else default_base
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)

    training_seconds: float | None = None
    if args.existing_training_output_dir:
        training_output_dir = Path(args.existing_training_output_dir).resolve()
        print(f"Using existing training output dir: {training_output_dir}", flush=True)
    else:
        before = _snapshot_children(base_output_dir)
        command = (
            _semantic_comparison_command(args, base_output_dir)
            if args.task == "semantic"
            else _instance_comparison_command(args, base_output_dir)
        )
        training_seconds = _run_command(command)
        training_output_dir = _latest_child(base_output_dir, exclude=before)
        print(f"Detected training output dir: {training_output_dir}", flush=True)

    if args.skip_sweep:
        result = PipelineResult(
            task=args.task,
            training_output_dir=str(training_output_dir),
            sweep_output_dir=None,
            training_seconds=training_seconds,
            sweep_seconds=None,
        )
    else:
        sweep_output_dir = (
            Path(args.sweep_output_root).resolve()
            if args.sweep_output_root
            else training_output_dir / "checkpoint_sweep"
        )
        command = (
            _semantic_sweep_command(args, training_output_dir, sweep_output_dir)
            if args.task == "semantic"
            else _instance_sweep_command(args, training_output_dir, sweep_output_dir)
        )
        sweep_seconds = _run_command(command)
        result = PipelineResult(
            task=args.task,
            training_output_dir=str(training_output_dir),
            sweep_output_dir=str(sweep_output_dir),
            training_seconds=training_seconds,
            sweep_seconds=sweep_seconds,
        )

    summary_path = training_output_dir / "train_then_checkpoint_sweep_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"Saved pipeline summary to {summary_path}", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["semantic", "instance"], required=True)
    parser.add_argument(
        "--simlink-dest", "--symlink-dest", dest="simlink_dest", type=str, default=None
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--base-output-dir", type=str, default=None)
    parser.add_argument("--existing-training-output-dir", type=str, default=None)
    parser.add_argument("--sweep-output-root", type=str, default=None)
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument(
        "--models", nargs="+", default=["toy", "graha"], choices=["toy", "graha"]
    )
    parser.add_argument("--dino-checkpoint", type=str, default=None)
    parser.add_argument("--graha-pretrain-dir", type=str, default=None)
    parser.add_argument(
        "--graha-input-modality-mode", choices=["new-wac", "vis-uv"], default="new-wac"
    )
    parser.add_argument(
        "--graha-vis-uv-merge-method", choices=["mean", "max"], default="mean"
    )
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument(
        "--band-filter", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6]
    )
    parser.add_argument(
        "--semantic-label-source",
        choices=["semantic", "instance"],
        default="semantic",
        help="Semantic task label format: .npy semantic masks or .npz instance labels converted to semantic masks.",
    )
    parser.add_argument(
        "--image-glob",
        default="*.tif",
        help="Semantic chip filename glob inside each split/chips directory.",
    )
    parser.add_argument(
        "--label-glob",
        default="*_label.*",
        help="Semantic label filename glob inside each split/labels directory.",
    )
    parser.add_argument(
        "--image-suffix",
        default="_input_wac_static_chip",
        help="Semantic suffix stripped from chip stems before matching labels.",
    )
    parser.add_argument(
        "--label-suffix",
        default="_label",
        help="Semantic suffix stripped from label stems before matching chips.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument(
        "--ignore-nodata-in-loss",
        action="store_true",
        help="Ignore TIFF nodata pixels in semantic segmentation loss and metrics.",
    )
    parser.add_argument(
        "--nodata-ignore-index",
        type=int,
        default=-1,
        help="Target label value used for ignored nodata pixels.",
    )
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--plot-every-n-epochs", type=int, default=1)
    parser.add_argument("--plot-n-samples", type=int, default=5)
    parser.add_argument(
        "--prediction-split", choices=["train", "val", "test"], default="val"
    )
    parser.add_argument("--prediction-n-samples", type=int, default=5)
    parser.add_argument("--run-epoch-test-suite", action="store_true")
    parser.add_argument(
        "--epoch-test-split", choices=["train", "val", "test"], default="test"
    )
    parser.add_argument("--epoch-test-n-samples", type=int, default=100)
    parser.add_argument("--epoch-test-every-n-epochs", type=int, default=1)
    parser.add_argument(
        "--sweep-split", choices=["train", "val", "test"], default="test"
    )
    parser.add_argument("--sweep-max-samples", type=int, default=None)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument(
        "--spatial-transform", choices=["crop", "resize"], default="crop"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--loss-type", type=str, default="focal_dice")
    parser.add_argument("--use-toy-shape-loss", action="store_true")
    parser.add_argument("--toy-shape-loss-weight", type=float, default=0.05)
    parser.add_argument("--toy-shape-loss-pad-frac", type=float, default=0.3)
    parser.add_argument("--graha-shape-loss-weight", type=float, default=0.05)
    parser.add_argument("--graha-shape-loss-pad-frac", type=float, default=0.3)
    parser.add_argument("--normalize-inputs", action="store_true")
    parser.add_argument(
        "--normalization-source",
        choices=["pretrain", "finetune"],
        default="pretrain",
        help="When normalizing inputs, use TerraMind pretraining stats or finetuning train-split stats.",
    )
    parser.add_argument(
        "--normalization-modality",
        choices=["vis_uv", "nac"],
        default="vis_uv",
        help="Which modality family to use when --normalization-source=pretrain.",
    )

    parser.add_argument("--toy-batch-size", type=int, default=2)
    parser.add_argument("--toy-num-workers", type=int, default=10)
    parser.add_argument("--toy-learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--toy-weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--toy-normalize-inputs", action="store_true")
    parser.add_argument(
        "--toy-architecture",
        choices=["mask2former", "dino-mask-rcnn"],
        default="mask2former",
    )
    parser.add_argument("--disable-toy-gradient-clipping", action="store_true")

    parser.add_argument("--graha-stats-batch-size", type=int, default=16)
    parser.add_argument("--graha-batch-size", type=int, default=2)
    parser.add_argument("--graha-num-workers", type=int, default=10)
    parser.add_argument("--graha-backbone-lr", type=float, default=5.0e-5)
    parser.add_argument("--graha-head-lr", type=float, default=2.0e-4)
    parser.add_argument("--graha-layer-decay", type=float, default=0.75)
    parser.add_argument("--graha-weight-decay", type=float, default=0.05)
    parser.add_argument("--graha-warmup-steps", type=int, default=500)
    parser.add_argument("--graha-anchor-sizes", type=str, default="8,16,32,64")
    parser.add_argument("--graha-anchor-aspect-ratios", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--graha-score-threshold", type=float, default=0.5)
    parser.add_argument("--prediction-score-threshold", type=float, default=0.5)
    parser.add_argument("--progress-log-every-n-batches", type=int, default=20)
    parser.add_argument("--mask-shift", type=int, nargs=2, default=(0, 0))

    parser.add_argument(
        "--comparison-extra-arg",
        action="append",
        default=[],
        help="Extra raw argument token for the comparison command. Repeat for multiple tokens.",
    )
    parser.add_argument(
        "--sweep-extra-arg",
        action="append",
        default=[],
        help="Extra raw argument token for the checkpoint sweep command. Repeat for multiple tokens.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
