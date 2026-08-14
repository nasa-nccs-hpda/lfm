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
from lfm.all_models.all_tasks.cli_args import parse_instance_checkpoint_sweep_args
from lfm.all_models.inst_seg.testing.instance_test_suite import (
    INSTANCE_TEST_SUITE_METRICS as METRIC_NAMES,
    write_instance_test_suite_outputs,
)
from lfm.all_models.inst_seg.config import build_config_from_args
from lfm.all_models.inst_seg.sweep_config import (
    InstanceCheckpointSweepConfig,
    build_checkpoint_sweep_config_from_args,
)
from lfm.full_model.inst_seg.instance_model_adapter import GrahaInstanceModelAdapter
from lfm.all_models.all_tasks.utils.utils import ensure_data_symlink
from lfm.all_models.all_tasks.utils import (
    save_graha_instance_prediction_cache,
    save_toy_instance_prediction_cache,
)

GRAHA_ADAPTER = GrahaInstanceModelAdapter()


@contextlib.contextmanager
def _quiet(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


def _prediction_count(config: InstanceCheckpointSweepConfig) -> int:
    return config.max_samples if config.max_samples is not None else 10**9


def _make_comparison_args(config: InstanceCheckpointSweepConfig) -> argparse.Namespace:
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
        excluded_nodata_values=config.excluded_nodata_values,
        skip_toy_fit=True,
        skip_graha_fit=True,
        no_fit=True,
        run_epoch_test_suite=False,
        epoch_test_split=config.prediction_split,
        epoch_test_n_samples=_prediction_count(config),
        epoch_test_every_n_epochs=1,
        seed=config.seed,
    )


def _setup_toy(config: InstanceCheckpointSweepConfig):
    from lfm.toy_model.inst_seg.instance_model_adapter import ToyInstanceModelAdapter

    toy_adapter = ToyInstanceModelAdapter()
    comparison_config = build_config_from_args(_make_comparison_args(config))
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


def _setup_graha(config: InstanceCheckpointSweepConfig):
    comparison_config = build_config_from_args(_make_comparison_args(config))
    graha_config = GRAHA_ADAPTER.build_comparison_config(
        comparison_config,
        config.output_root / "_graha_setup",
    )
    with _quiet(not config.verbose):
        GRAHA_ADAPTER.configure_environment()
        GRAHA_ADAPTER.configure_python_paths(graha_config)
        GRAHA_ADAPTER.validate_required_paths(graha_config)
        deps = GRAHA_ADAPTER.import_project_dependencies()
        datamodule_cls = deps["GrahaObjectDetectionInstanceDataModule"]
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
    config: InstanceCheckpointSweepConfig,
    checkpoints: list[CheckpointRecord] | None = None,
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
    for checkpoint in tqdm(
        checkpoints,
        desc="Toy checkpoints",
        dynamic_ncols=True,
        file=sys.stdout,
    ):
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
    config: InstanceCheckpointSweepConfig,
    checkpoints: list[CheckpointRecord] | None = None,
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
    for checkpoint in tqdm(
        checkpoints,
        desc="Graha checkpoints",
        dynamic_ncols=True,
        file=sys.stdout,
    ):
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


def run_sweep(
    config: InstanceCheckpointSweepConfig,
) -> dict[str, list[dict[str, Any]]]:
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
    return parse_instance_checkpoint_sweep_args(description=__doc__)


def main() -> None:
    args = parse_args()
    notebook_dir = Path(__file__).resolve().parents[2] / "notebooks" / "full_model"
    ensure_data_symlink(args.simlink_dest, notebook_dir / "data")
    config = build_checkpoint_sweep_config_from_args(args)
    print("Output root:", config.output_root)
    print("Data root:", config.data_root)
    run_sweep(config)


if __name__ == "__main__":
    main()
