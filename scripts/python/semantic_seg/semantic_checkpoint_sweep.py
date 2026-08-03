"""Run semantic-segmentation checkpoint sweeps over the test split.

For each checkpoint, this script saves one folder per test sample containing:

- ``{sample_key}_input.npy``
- ``{sample_key}_label.npy``
- ``{sample_key}_pred.npy``
- ``{sample_key}_class_pred.npy``
- ``{sample_key}_logits.npy``
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
from types import SimpleNamespace
from typing import Any

import torch
from lightning.pytorch import seed_everything
from tqdm.auto import tqdm
from torch.utils.data import Subset

LFM_ROOT = Path(__file__).resolve().parents[3]
if str(LFM_ROOT) not in sys.path:
    sys.path.insert(0, str(LFM_ROOT))

from lfm.all_models.all_tasks import (
    CheckpointRecord,
    CheckpointSweepExperiment,
    discover_checkpoints,
    write_checkpoint_metrics_summary,
)
from lfm.all_models.all_tasks.cli_args import parse_semantic_checkpoint_sweep_args
from lfm.all_models.sem_seg.testing.semantic_test_suite import (
    SEMANTIC_CHECKPOINT_METRICS as METRIC_NAMES,
    run_semantic_checkpoint,
)
from lfm.all_models.sem_seg.config import build_config_from_args as build_toy_config
from lfm.all_models.sem_seg.sweep_config import (
    SemanticCheckpointSweepConfig,
    build_checkpoint_sweep_config_from_args,
)
from lfm.all_models.all_tasks.utils.utils import ensure_data_symlink
from lfm.full_model.sem_seg.semantic_model_adapter import GrahaSemanticModelAdapter
from lfm.toy_model.sem_seg.semantic_model_adapter import ToySemanticModelAdapter
from semantic_seg_comparison import (
    get_toy_normalization_modality_info,
)

TOY_ADAPTER = ToySemanticModelAdapter()
GRAHA_ADAPTER = GrahaSemanticModelAdapter()


@contextlib.contextmanager
def _quiet(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


def _limit_dataset(
    dataset, max_samples: int | None, *, model_name: str, split_name: str
):
    if max_samples is None:
        return dataset
    if max_samples < 0:
        raise ValueError(f"max_samples must be non-negative, got {max_samples}")
    limited_count = min(max_samples, len(dataset))
    print(
        f"[{model_name} {split_name}] Limited to {limited_count} of {len(dataset)} samples.",
        flush=True,
    )
    return Subset(dataset, range(limited_count))


def _cache_batch_on_cpu(batch: Any) -> Any:
    if isinstance(batch, dict):
        return {key: _cache_batch_on_cpu(value) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_cache_batch_on_cpu(value) for value in batch)
    if isinstance(batch, list):
        return [_cache_batch_on_cpu(value) for value in batch]
    if torch.is_tensor(batch):
        return batch.detach().cpu()
    return batch


def preload_test_batches(dataloader, *, model_name: str) -> list[Any]:
    """Load processed test batches into CPU memory once before checkpoint sweep."""
    cached_batches = []
    iterator = iter(dataloader)
    try:
        for batch in tqdm(
            iterator,
            total=len(dataloader) if hasattr(dataloader, "__len__") else None,
            desc=f"{model_name} preload test batches",
            dynamic_ncols=True,
            file=sys.stdout,
        ):
            cached_batches.append(_cache_batch_on_cpu(batch))
    finally:
        shutdown_workers = getattr(iterator, "_shutdown_workers", None)
        if shutdown_workers is not None:
            shutdown_workers()
        del iterator
        gc.collect()
    print(f"[{model_name}] Preloaded {len(cached_batches)} test batch(es).", flush=True)
    return cached_batches


def _make_toy_args(config: SemanticCheckpointSweepConfig) -> argparse.Namespace:
    return SimpleNamespace(
        data_root=str(config.data_root),
        base_output_dir=str(config.output_root / "_toy_setup"),
        dino_checkpoint=str(config.dino_checkpoint) if config.dino_checkpoint else None,
        toy_lightning_checkpoint=None,
        band_filter=config.band_filter,
        target_size=config.target_size,
        spatial_transform="crop",
        semantic_label_source=config.semantic_label_source,
        image_glob=config.image_glob,
        label_glob=config.label_glob,
        image_suffix=config.image_suffix,
        label_suffix=config.label_suffix,
        max_train_samples=None,
        max_val_samples=None,
        max_test_samples=config.max_test_samples,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        max_epochs=1,
        learning_rate=5e-5,
        weight_decay=0.05,
        toy_loss_type="dice",
        use_toy_shape_loss=False,
        toy_shape_loss_weight=0.05,
        toy_shape_loss_pad_frac=0.3,
        freeze_encoder=False,
        normalize_inputs=config.normalize_inputs,
        normalization_source=config.normalization_source,
        normalization_modality=config.normalization_modality,
        toy_gradient_clip_val=1.0,
        disable_toy_gradient_clipping=True,
        plot_every_n_epochs=1,
        plot_n_samples=5,
        cache_predictions=False,
        prediction_split="test",
        prediction_n_samples=20,
        graha_base_output_dir=None,
        graha_pretrain_dir=(
            str(config.graha_pretrain_dir) if config.graha_pretrain_dir else None
        ),
        graha_input_modality_mode=config.graha_input_modality_mode,
        graha_vis_uv_merge_method=config.graha_vis_uv_merge_method,
        graha_lightning_checkpoint=None,
        graha_stats_batch_size=config.graha_stats_batch_size,
        graha_batch_size=config.graha_batch_size,
        graha_num_workers=config.graha_num_workers,
        seed=config.seed,
        no_fit=False,
        skip_toy_fit=False,
        skip_graha_fit=False,
        run_epoch_test_suite=False,
        epoch_test_split="test",
        epoch_test_n_samples=(
            config.max_test_samples if config.max_test_samples is not None else 10**9
        ),
        epoch_test_every_n_epochs=1,
        ignore_nodata_in_loss=config.ignore_nodata_in_loss,
        nodata_ignore_index=config.nodata_ignore_index,
    )


def run_toy_sweep(
    config: SemanticCheckpointSweepConfig,
    checkpoints: list[CheckpointRecord] | None = None,
) -> list[dict[str, Any]]:
    if checkpoints is None:
        if config.toy_checkpoint_dir is None:
            raise ValueError("Toy sweep requested but toy_checkpoint_dir is not set.")
        checkpoints = discover_checkpoints(
            config.toy_checkpoint_dir, max_checkpoints=config.max_checkpoints
        )
        print(f"[Toy] Found {len(checkpoints)} checkpoint(s).")
    toy_config = build_toy_config(_make_toy_args(config))
    setup_dir = config.output_root / "_toy_setup"
    with _quiet(not config.verbose):
        datamodule = TOY_ADAPTER.create_datamodule(
            toy_config,
            setup_dir,
            normalization_modality_info=get_toy_normalization_modality_info(toy_config),
        )
        datamodule.setup("test")
        task = TOY_ADAPTER.create_model_or_task(toy_config, datamodule)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task.to(device)

    dataloader = datamodule.test_dataloader()
    test_batches = (
        preload_test_batches(dataloader, model_name="Toy")
        if config.preload_test_batches
        else dataloader
    )
    model_output_dir = config.output_root / "toy_model"
    rows = []
    checkpoint_bar = tqdm(
        checkpoints,
        desc="Toy checkpoints",
        dynamic_ncols=True,
        file=sys.stdout,
    )
    for checkpoint in checkpoint_bar:
        checkpoint_bar.set_postfix(checkpoint=checkpoint.name)
        metrics = run_semantic_checkpoint(
            task=task,
            test_batches=test_batches,
            checkpoint=checkpoint,
            output_dir=model_output_dir,
            model_name="Toy",
            ignore_index=(
                config.nodata_ignore_index if config.ignore_nodata_in_loss else None
            ),
        )
        rows.append(
            {
                "checkpoint_name": checkpoint.name,
                "epoch": checkpoint.epoch,
                "checkpoint_path": checkpoint.path,
                **metrics,
            }
        )
        checkpoint_bar.set_postfix(
            checkpoint=checkpoint.name,
            f1=f"{metrics['foreground_f1']:.4f}",
            iou=f"{metrics['iou']:.4f}",
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


def _make_graha_args(config: SemanticCheckpointSweepConfig) -> argparse.Namespace:
    return SimpleNamespace(
        data_root=str(config.data_root),
        base_output_dir=str(config.output_root / "_graha_setup"),
        pretrain_dir=(
            str(config.graha_pretrain_dir) if config.graha_pretrain_dir else None
        ),
        lightning_checkpoint=None,
        graha_input_modality_mode=config.graha_input_modality_mode,
        graha_vis_uv_merge_method=config.graha_vis_uv_merge_method,
        normalization_source=config.normalization_source,
        normalization_modality=config.normalization_modality,
        band_filter=config.band_filter,
        semantic_label_source=config.semantic_label_source,
        image_glob=config.image_glob,
        label_glob=config.label_glob,
        image_suffix=config.image_suffix,
        label_suffix=config.label_suffix,
        shape_loss_weight=0.0,
        shape_loss_pad_frac=0.3,
        crop_size=config.target_size,
        stats_batch_size=config.graha_stats_batch_size,
        batch_size=config.graha_batch_size,
        num_workers=config.graha_num_workers,
        max_epochs=1,
        cache_predictions=False,
        prediction_split="test",
        prediction_n_samples=20,
        progress_log_every_n_batches=25,
        seed=config.seed,
        no_fit=True,
        ignore_nodata_in_loss=config.ignore_nodata_in_loss,
        nodata_ignore_index=config.nodata_ignore_index,
    )


def run_graha_sweep(
    config: SemanticCheckpointSweepConfig,
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
    with _quiet(not config.verbose):
        GRAHA_ADAPTER.configure_environment()
        graha_config = GRAHA_ADAPTER.build_config(_make_graha_args(config))
        GRAHA_ADAPTER.configure_python_paths(graha_config)
        GRAHA_ADAPTER.validate_required_paths(graha_config)

        deps = GRAHA_ADAPTER.import_project_dependencies()
        datamodule_cls = deps[
            (
                "LunarSemanticFromInstanceDatamodule"
                if config.semantic_label_source == "instance"
                else "LunarSemanticMaskSegmentationDatamodule"
            )
        ]
        task_cls = GRAHA_ADAPTER.make_task_class(deps["LunarShapeSegmentationTask"])
        means, stds = GRAHA_ADAPTER.get_normalization_stats(
            graha_config,
            datamodule_cls,
        )
        datamodule = GRAHA_ADAPTER.create_datamodule(
            graha_config, datamodule_cls, means, stds
        )
        datamodule.setup("test")
        datamodule.test_dataset = _limit_dataset(
            datamodule.test_dataset,
            config.max_test_samples,
            model_name="Graha",
            split_name="test",
        )

        sample_batch = GRAHA_ADAPTER.inspect_batch(datamodule)
        task = GRAHA_ADAPTER.create_task(graha_config, task_cls, sample_batch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task.to(device)

    dataloader = datamodule.test_dataloader()
    test_batches = (
        preload_test_batches(dataloader, model_name="Graha")
        if config.preload_test_batches
        else dataloader
    )
    model_output_dir = config.output_root / "graha_model"
    rows = []
    checkpoint_bar = tqdm(
        checkpoints,
        desc="Graha checkpoints",
        dynamic_ncols=True,
        file=sys.stdout,
    )
    for checkpoint in checkpoint_bar:
        checkpoint_bar.set_postfix(checkpoint=checkpoint.name)
        metrics = run_semantic_checkpoint(
            task=task,
            test_batches=test_batches,
            checkpoint=checkpoint,
            output_dir=model_output_dir,
            model_name="Graha",
            ignore_index=(
                config.nodata_ignore_index if config.ignore_nodata_in_loss else None
            ),
        )
        rows.append(
            {
                "checkpoint_name": checkpoint.name,
                "epoch": checkpoint.epoch,
                "checkpoint_path": checkpoint.path,
                **metrics,
            }
        )
        checkpoint_bar.set_postfix(
            checkpoint=checkpoint.name,
            f1=f"{metrics['foreground_f1']:.4f}",
            iou=f"{metrics['iou']:.4f}",
        )

    write_checkpoint_metrics_summary(
        model_output_dir,
        rows,
        metric_names=METRIC_NAMES,
    )
    del task, datamodule, sample_batch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def run_sweep(
    config: SemanticCheckpointSweepConfig,
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
    return parse_semantic_checkpoint_sweep_args(description=__doc__)


def main() -> None:
    args = parse_args()
    notebook_dir = Path(__file__).resolve().parents[2] / "notebooks" / "full_model"
    ensure_data_symlink(args.simlink_dest, notebook_dir / "data")
    config = build_checkpoint_sweep_config_from_args(args)
    print(
        "REMINDER: after rerunning training, confirm checkpoint directory structure before large sweeps."
    )
    print("Output root:", config.output_root)
    print("Data root:", config.data_root)
    run_sweep(config)


if __name__ == "__main__":
    main()
