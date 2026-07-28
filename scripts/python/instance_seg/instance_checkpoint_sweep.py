"""Run true instance-segmentation checkpoint sweeps over a split.

For each checkpoint, this script saves one folder per sample containing:

- ``{sample_key}_input.npy``
- ``{sample_key}_label.npy``
- ``{sample_key}_pred.npy``
- ``{sample_key}_pred_classes.npy``
- ``{sample_key}_pred_logits.npy``
- ``{sample_key}_gt_boxes.npy``
- ``{sample_key}_pred_boxes.npy``
- ``{sample_key}_pred_scores.npy``
- ``metrics.npy``
- ``metrics.txt``

Each checkpoint directory also receives aggregate ``metrics.npy`` and
``metrics.txt`` files. The same functions are intended for use from the
companion notebook and from sbatch.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import contextlib
import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import seed_everything
from tqdm.auto import tqdm

LFM_ROOT = Path(__file__).resolve().parents[3]
if str(LFM_ROOT) not in sys.path:
    sys.path.insert(0, str(LFM_ROOT))

import instance_seg_comparison as comparison_workflow
from lfm.all_models.all_tasks import (
    CheckpointRecord,
    CheckpointSweepExperiment,
    discover_checkpoints,
    load_lightning_checkpoint_state,
    write_checkpoint_metrics_summary,
)
from lfm.all_models.inst_seg.instance_test_suite import (
    INSTANCE_TEST_SUITE_METRICS as METRIC_NAMES,
    write_instance_test_suite_outputs,
)
from lfm.full_model.inst_seg.instance_model_adapter import GrahaInstanceModelAdapter
from lfm.all_models.all_tasks.utils.utils import ensure_data_symlink
from lfm.all_models.all_tasks.utils import (
    save_graha_instance_prediction_cache,
    save_toy_instance_prediction_cache,
)

GRAHA_ADAPTER = GrahaInstanceModelAdapter()


@dataclass(frozen=True)
class InstanceSweepConfig:
    notebook_dir: Path
    data_root: Path
    output_root: Path
    toy_checkpoint_dir: Path | None
    graha_checkpoint_dir: Path | None
    models: list[str]
    target_size: int
    band_filter: list[int]
    image_glob: str
    label_glob: str
    image_suffix: str
    label_suffix: str
    max_samples: int | None
    toy_batch_size: int
    toy_num_workers: int
    toy_normalize_inputs: bool
    normalization_source: str
    normalization_modality: str
    toy_architecture: str
    dino_checkpoint: Path | None
    graha_pretrain_dir: Path | None
    graha_input_modality_mode: str
    graha_vis_uv_merge_method: str
    graha_stats_batch_size: int
    graha_batch_size: int
    graha_num_workers: int
    graha_backbone_lr: float
    graha_head_lr: float
    graha_layer_decay: float
    graha_weight_decay: float
    graha_warmup_steps: int
    graha_anchor_sizes: list[list[int]]
    graha_anchor_aspect_ratios: list[float]
    graha_score_threshold: float
    prediction_split: str
    prediction_score_threshold: float
    mask_shift: tuple[int, int]
    ignore_nodata_in_loss: bool
    nodata_ignore_index: int
    max_checkpoints: int | None
    seed: int
    verbose: bool


def build_config(args: argparse.Namespace) -> InstanceSweepConfig:
    script_dir = Path(__file__).resolve().parent
    lfm_root = script_dir.parents[2]
    notebook_dir = lfm_root / "notebooks" / "full_model"
    scripts_output_dir = lfm_root / "scripts" / "outputs"
    models = [model.lower() for model in args.models]
    unknown = sorted(set(models) - {"toy", "graha"})
    if unknown:
        raise ValueError(f"Unknown model name(s): {unknown}")

    return InstanceSweepConfig(
        notebook_dir=notebook_dir,
        data_root=(
            Path(args.data_root).resolve() if args.data_root else notebook_dir / "data"
        ),
        output_root=(
            Path(args.output_root).resolve()
            if args.output_root
            else scripts_output_dir / "instance_checkpoint_sweep"
        ),
        toy_checkpoint_dir=(
            Path(args.toy_checkpoint_dir).resolve() if args.toy_checkpoint_dir else None
        ),
        graha_checkpoint_dir=(
            Path(args.graha_checkpoint_dir).resolve()
            if args.graha_checkpoint_dir
            else None
        ),
        models=models,
        target_size=args.target_size,
        band_filter=args.band_filter,
        image_glob=args.image_glob,
        label_glob=args.label_glob,
        image_suffix=args.image_suffix,
        label_suffix=args.label_suffix,
        max_samples=args.max_samples,
        toy_batch_size=args.toy_batch_size,
        toy_num_workers=args.toy_num_workers,
        toy_normalize_inputs=args.toy_normalize_inputs,
        normalization_source=getattr(args, "normalization_source", "pretrain"),
        normalization_modality=getattr(args, "normalization_modality", "vis_uv"),
        toy_architecture=args.toy_architecture,
        dino_checkpoint=(
            Path(args.dino_checkpoint).resolve() if args.dino_checkpoint else None
        ),
        graha_pretrain_dir=(
            Path(args.graha_pretrain_dir).resolve() if args.graha_pretrain_dir else None
        ),
        graha_input_modality_mode=args.graha_input_modality_mode,
        graha_vis_uv_merge_method=args.graha_vis_uv_merge_method,
        graha_stats_batch_size=args.graha_stats_batch_size,
        graha_batch_size=args.graha_batch_size,
        graha_num_workers=args.graha_num_workers,
        graha_backbone_lr=args.graha_backbone_lr,
        graha_head_lr=args.graha_head_lr,
        graha_layer_decay=args.graha_layer_decay,
        graha_weight_decay=args.graha_weight_decay,
        graha_warmup_steps=args.graha_warmup_steps,
        graha_anchor_sizes=args.graha_anchor_sizes,
        graha_anchor_aspect_ratios=args.graha_anchor_aspect_ratios,
        graha_score_threshold=args.graha_score_threshold,
        prediction_split=args.prediction_split,
        prediction_score_threshold=args.prediction_score_threshold,
        mask_shift=tuple(args.mask_shift),
        ignore_nodata_in_loss=getattr(args, "ignore_nodata_in_loss", False),
        nodata_ignore_index=getattr(args, "nodata_ignore_index", -1),
        max_checkpoints=args.max_checkpoints,
        seed=args.seed,
        verbose=args.verbose,
    )


@contextlib.contextmanager
def _quiet(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


def _prediction_count(config: InstanceSweepConfig) -> int:
    return config.max_samples if config.max_samples is not None else 10**9


def _make_comparison_args(config: InstanceSweepConfig) -> argparse.Namespace:
    return argparse.Namespace(
        data_root=str(config.data_root),
        base_output_dir=str(config.output_root / "_setup"),
        dino_checkpoint=str(config.dino_checkpoint) if config.dino_checkpoint else None,
        toy_lightning_checkpoint=None,
        graha_pretrain_dir=(
            str(config.graha_pretrain_dir) if config.graha_pretrain_dir else None
        ),
        graha_lightning_checkpoint=None,
        graha_input_modality_mode=config.graha_input_modality_mode,
        graha_vis_uv_merge_method=config.graha_vis_uv_merge_method,
        normalization_source=config.normalization_source,
        normalization_modality=config.normalization_modality,
        target_size=config.target_size,
        band_filter=config.band_filter,
        image_glob=config.image_glob,
        label_glob=config.label_glob,
        image_suffix=config.image_suffix,
        label_suffix=config.label_suffix,
        max_train_samples=None,
        max_val_samples=None,
        max_test_samples=config.max_samples,
        toy_batch_size=config.toy_batch_size,
        toy_num_workers=config.toy_num_workers,
        graha_stats_batch_size=config.graha_stats_batch_size,
        graha_batch_size=config.graha_batch_size,
        graha_num_workers=config.graha_num_workers,
        max_epochs=1,
        toy_learning_rate=5.0e-5,
        toy_weight_decay=1.0e-3,
        toy_freeze_backbone=False,
        toy_normalize_inputs=config.toy_normalize_inputs,
        toy_architecture=config.toy_architecture,
        toy_gradient_clip_val=1.0,
        disable_toy_gradient_clipping=True,
        graha_backbone_lr=config.graha_backbone_lr,
        graha_head_lr=config.graha_head_lr,
        graha_layer_decay=config.graha_layer_decay,
        graha_weight_decay=config.graha_weight_decay,
        graha_warmup_steps=config.graha_warmup_steps,
        graha_anchor_sizes=config.graha_anchor_sizes,
        graha_anchor_aspect_ratios=config.graha_anchor_aspect_ratios,
        graha_score_threshold=config.graha_score_threshold,
        plot_every_n_epochs=0,
        plot_n_samples=5,
        progress_log_every_n_batches=10**9,
        prediction_split=config.prediction_split,
        prediction_n_samples=_prediction_count(config),
        prediction_score_threshold=config.prediction_score_threshold,
        mask_shift=config.mask_shift,
        ignore_nodata_in_loss=config.ignore_nodata_in_loss,
        nodata_ignore_index=config.nodata_ignore_index,
        skip_toy_fit=True,
        skip_graha_fit=True,
        no_fit=True,
        run_epoch_test_suite=False,
        epoch_test_split=config.prediction_split,
        epoch_test_n_samples=_prediction_count(config),
        epoch_test_every_n_epochs=1,
        seed=config.seed,
    )


def _setup_toy(config: InstanceSweepConfig):
    from lfm.toy_model.inst_seg.instance_model_adapter import ToyInstanceModelAdapter

    toy_adapter = ToyInstanceModelAdapter()
    comparison_config = comparison_workflow.build_config(_make_comparison_args(config))
    with _quiet(not config.verbose):
        datamodule = toy_adapter.create_datamodule(
            comparison_config,
            normalization_modality_info=(
                comparison_workflow.get_toy_normalization_modality_info(
                    comparison_config
                )
            ),
        )
        datamodule.setup(config.prediction_split)
        task = toy_adapter.create_model_or_task(comparison_config, datamodule)
        image_processor = toy_adapter.create_image_processor(comparison_config)
    task.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return comparison_config, datamodule, task, image_processor


def _setup_graha(config: InstanceSweepConfig):
    comparison_config = comparison_workflow.build_config(_make_comparison_args(config))
    graha_config = GRAHA_ADAPTER.build_comparison_config(
        comparison_config,
        config.output_root / "_graha_setup",
    )
    with _quiet(not config.verbose):
        GRAHA_ADAPTER.configure_environment()
        GRAHA_ADAPTER.configure_python_paths(graha_config)
        GRAHA_ADAPTER.validate_required_paths(graha_config)
        deps = GRAHA_ADAPTER.import_project_dependencies()
        datamodule_cls = deps["LunarObjectDetectionInstanceMaskDatamodule"]
        task_cls = GRAHA_ADAPTER.make_task_class(deps["LunarObjectDetectionTask"])
        means, stds = GRAHA_ADAPTER.get_normalization_stats(
            graha_config,
            datamodule_cls,
        )
        datamodule = GRAHA_ADAPTER.create_datamodule(
            graha_config,
            datamodule_cls,
            means,
            stds,
        )
        datamodule.setup(config.prediction_split)
        task = GRAHA_ADAPTER.create_model_or_task(graha_config, datamodule, task_cls)
    task.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return datamodule, task


def run_toy_sweep(
    config: InstanceSweepConfig, checkpoints: list[CheckpointRecord] | None = None
) -> list[dict[str, Any]]:
    if checkpoints is None:
        if config.toy_checkpoint_dir is None:
            raise ValueError("Toy sweep requested but toy_checkpoint_dir is not set.")
        checkpoints = discover_checkpoints(
            config.toy_checkpoint_dir, max_checkpoints=config.max_checkpoints
        )
        print(f"[Toy] Found {len(checkpoints)} checkpoint(s).")
    _, datamodule, task, image_processor = _setup_toy(config)
    model_output_dir = config.output_root / "toy_model"
    rows = []
    for checkpoint in tqdm(checkpoints, desc="Toy checkpoints", dynamic_ncols=True):
        load_lightning_checkpoint_state(task, checkpoint.path)
        if image_processor is None:
            cache_dir = save_graha_instance_prediction_cache(
                task=task,
                datamodule=datamodule,
                output_dir=model_output_dir / checkpoint.name,
                model_name="toy",
                split=config.prediction_split,
                n_samples=_prediction_count(config),
                score_threshold=0.0,
            )
        else:
            cache_dir = save_toy_instance_prediction_cache(
                task=task,
                datamodule=datamodule,
                output_dir=model_output_dir / checkpoint.name,
                image_processor=image_processor,
                model_name="toy",
                split=config.prediction_split,
                n_samples=_prediction_count(config),
                score_threshold=0.0,
            )
        metrics = write_instance_test_suite_outputs(
            cache_dir=cache_dir,
            checkpoint_output_dir=model_output_dir / checkpoint.name,
            checkpoint=checkpoint,
            model_name="Toy",
            score_threshold=config.prediction_score_threshold,
        )
        rows.append(
            {
                "checkpoint_name": checkpoint.name,
                "epoch": checkpoint.epoch,
                "checkpoint_path": checkpoint.path,
                **metrics,
            }
        )
    write_checkpoint_metrics_summary(
        model_output_dir,
        rows,
        metric_names=METRIC_NAMES,
    )
    del task, datamodule
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def run_graha_sweep(
    config: InstanceSweepConfig, checkpoints: list[CheckpointRecord] | None = None
) -> list[dict[str, Any]]:
    if checkpoints is None:
        if config.graha_checkpoint_dir is None:
            raise ValueError(
                "Graha sweep requested but graha_checkpoint_dir is not set."
            )
        checkpoints = discover_checkpoints(
            config.graha_checkpoint_dir, max_checkpoints=config.max_checkpoints
        )
        print(f"[Graha] Found {len(checkpoints)} checkpoint(s).")
    datamodule, task = _setup_graha(config)
    model_output_dir = config.output_root / "graha_model"
    rows = []
    for checkpoint in tqdm(checkpoints, desc="Graha checkpoints", dynamic_ncols=True):
        load_lightning_checkpoint_state(task, checkpoint.path)
        cache_dir = save_graha_instance_prediction_cache(
            task=task,
            datamodule=datamodule,
            output_dir=model_output_dir / checkpoint.name,
            model_name="graha",
            split=config.prediction_split,
            n_samples=_prediction_count(config),
            score_threshold=0.0,
        )
        metrics = write_instance_test_suite_outputs(
            cache_dir=cache_dir,
            checkpoint_output_dir=model_output_dir / checkpoint.name,
            checkpoint=checkpoint,
            model_name="Graha",
            score_threshold=config.prediction_score_threshold,
        )
        rows.append(
            {
                "checkpoint_name": checkpoint.name,
                "epoch": checkpoint.epoch,
                "checkpoint_path": checkpoint.path,
                **metrics,
            }
        )
    write_checkpoint_metrics_summary(
        model_output_dir,
        rows,
        metric_names=METRIC_NAMES,
    )
    del task, datamodule
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def run_sweep(config: InstanceSweepConfig) -> dict[str, list[dict[str, Any]]]:
    def run_model_sweep(
        model: str, checkpoints: list[CheckpointRecord]
    ) -> list[dict[str, Any]]:
        if model == "toy":
            return run_toy_sweep(config, checkpoints)
        if model == "graha":
            return run_graha_sweep(config, checkpoints)
        raise ValueError(f"Unknown model name: {model}")

    return CheckpointSweepExperiment(
        output_root=config.output_root,
        models=config.models,
        checkpoint_dirs={
            "toy": config.toy_checkpoint_dir,
            "graha": config.graha_checkpoint_dir,
        },
        run_model_sweep=run_model_sweep,
        max_checkpoints=config.max_checkpoints,
        seed=config.seed,
        seed_fn=seed_everything,
    ).run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simlink-dest", "--symlink-dest", dest="simlink_dest", type=str, default=None
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--toy-checkpoint-dir", type=str, default=None)
    parser.add_argument("--graha-checkpoint-dir", type=str, default=None)
    parser.add_argument(
        "--models", nargs="+", default=["toy", "graha"], choices=["toy", "graha"]
    )
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument(
        "--band-filter", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6]
    )
    parser.add_argument(
        "--image-glob",
        default="*.tif",
        help="Chip filename glob inside each split/chips directory.",
    )
    parser.add_argument(
        "--label-glob",
        default="*_label.*",
        help="Label filename glob inside each split/labels directory.",
    )
    parser.add_argument(
        "--image-suffix",
        default="_input_wac_static_chip",
        help="Suffix stripped from chip stems before matching labels.",
    )
    parser.add_argument(
        "--label-suffix",
        default="_label",
        help="Suffix stripped from label stems before matching chips.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--toy-batch-size",
        "--batch-size",
        dest="toy_batch_size",
        type=int,
        default=2,
        help="Toy instance batch size. --batch-size is accepted for parity with semantic scripts.",
    )
    parser.add_argument("--toy-num-workers", type=int, default=4)
    parser.add_argument(
        "--toy-normalize-inputs",
        "--normalize-inputs",
        dest="toy_normalize_inputs",
        action="store_true",
        help="Enable Toy instance z-score normalization. --normalize-inputs is accepted for parity with semantic scripts.",
    )
    parser.add_argument(
        "--normalization-source",
        choices=["pretrain", "finetune"],
        default="pretrain",
    )
    parser.add_argument(
        "--normalization-modality",
        choices=["vis_uv", "nac"],
        default="vis_uv",
    )
    parser.add_argument(
        "--toy-architecture",
        choices=["mask2former", "dino-mask-rcnn"],
        default="mask2former",
    )
    parser.add_argument("--dino-checkpoint", type=str, default=None)
    parser.add_argument("--graha-pretrain-dir", type=str, default=None)
    parser.add_argument(
        "--graha-input-modality-mode", choices=["new-wac", "vis-uv"], default="new-wac"
    )
    parser.add_argument(
        "--graha-vis-uv-merge-method", choices=["mean", "max"], default="mean"
    )
    parser.add_argument("--graha-stats-batch-size", type=int, default=16)
    parser.add_argument("--graha-batch-size", type=int, default=2)
    parser.add_argument("--graha-num-workers", type=int, default=4)
    parser.add_argument("--graha-backbone-lr", type=float, default=5.0e-5)
    parser.add_argument("--graha-head-lr", type=float, default=2.0e-4)
    parser.add_argument("--graha-layer-decay", type=float, default=0.75)
    parser.add_argument("--graha-weight-decay", type=float, default=0.05)
    parser.add_argument("--graha-warmup-steps", type=int, default=500)
    parser.add_argument(
        "--graha-anchor-sizes",
        type=lambda value: [[int(x)] for x in value.split(",")],
        default=[[8], [16], [32], [64]],
    )
    parser.add_argument(
        "--graha-anchor-aspect-ratios",
        type=lambda value: [float(x) for x in value.split(",")],
        default=[0.5, 1.0, 2.0],
    )
    parser.add_argument("--graha-score-threshold", type=float, default=0.5)
    parser.add_argument(
        "--prediction-split", choices=["train", "val", "test"], default="test"
    )
    parser.add_argument("--prediction-score-threshold", type=float, default=0.5)
    parser.add_argument("--mask-shift", type=int, nargs=2, default=(0, 0))
    parser.add_argument(
        "--ignore-nodata-in-loss",
        action="store_true",
        help="Thread TIFF nodata pixels through instance target preprocessing.",
    )
    parser.add_argument(
        "--nodata-ignore-index",
        type=int,
        default=-1,
        help="Target label value used for ignored nodata pixels.",
    )
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notebook_dir = Path(__file__).resolve().parents[2] / "notebooks" / "full_model"
    ensure_data_symlink(args.simlink_dest, notebook_dir / "data")
    config = build_config(args)
    print("Output root:", config.output_root)
    print("Data root:", config.data_root)
    run_sweep(config)


if __name__ == "__main__":
    main()
