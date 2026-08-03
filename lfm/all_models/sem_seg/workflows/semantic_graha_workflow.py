"""Graha semantic segmentation workflow orchestration."""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import Callback

from lfm.all_models.all_tasks.utils import save_prediction_cache
from lfm.full_model.sem_seg import semantic_graha_components


def _record_timing(
    record_timing: Callable[..., None] | None,
    timing_rows: list[dict[str, Any]] | None,
    *,
    stage: str,
    started_at: float,
) -> None:
    if record_timing is None or timing_rows is None:
        return
    record_timing(timing_rows, model="Graha", stage=stage, started_at=started_at)


def run_graha_workflow(
    config: Any,
    *,
    no_fit: bool,
    comparison_output_dir: Path,
    epoch_test_suite_callback_cls: type[Callback] | None = None,
    timing_rows: list[dict[str, Any]] | None = None,
    record_timing: Callable[..., None] | None = None,
) -> tuple[Path, Path | None]:
    """Run the Graha/Lunar-FM semantic segmentation path for comparison."""
    graha_total_started_at = time.perf_counter()
    semantic_graha_components.configure_proj_environment()
    graha_config = semantic_graha_components.build_comparison_config(
        config,
        comparison_output_dir,
    )
    semantic_graha_components.configure_python_paths(graha_config)
    semantic_graha_components.print_config(graha_config)
    semantic_graha_components.validate_required_paths(graha_config)

    deps = semantic_graha_components.import_project_dependencies()
    datamodule_cls = deps[
        (
            "LunarSemanticFromInstanceDatamodule"
            if config.semantic_label_source == "instance"
            else "LunarSemanticMaskSegmentationDatamodule"
        )
    ]
    task_cls = semantic_graha_components.make_downstream_shape_segmentation_task_class(
        deps["LunarShapeSegmentationTask"]
    )

    output_dir = semantic_graha_components.create_output_dirs(
        graha_config,
        deps["create_timestamped_output_dir"],
        use_timestamp=False,
    )
    seed_everything(graha_config.seed)
    stats_started_at = time.perf_counter()
    means, stds = semantic_graha_components.get_normalization_stats(
        graha_config,
        datamodule_cls,
    )
    _record_timing(
        record_timing,
        timing_rows,
        stage="stats",
        started_at=stats_started_at,
    )
    datamodule = semantic_graha_components.create_datamodule(
        graha_config,
        datamodule_cls,
        means,
        stds,
    )
    sample_batch = semantic_graha_components.inspect_batch(datamodule)
    task = semantic_graha_components.create_task(graha_config, task_cls, sample_batch)
    semantic_graha_components.inspect_backbone(task)
    trainer = semantic_graha_components.create_trainer(
        graha_config,
        output_dir,
        deps["ValidationPlotCallback"],
        plot_output_dir=comparison_output_dir,
        plots_subdir=Path("plots") / "single_model" / "full_model",
        checkpoint_subdir=Path("checkpoints") / "full_model",
    )
    if config.run_epoch_test_suite:
        if epoch_test_suite_callback_cls is None:
            raise ValueError("epoch_test_suite_callback_cls is required.")
        trainer.callbacks.append(
            epoch_test_suite_callback_cls(
                output_dir=comparison_output_dir,
                model_name="full_model",
                split=config.epoch_test_split,
                n_samples=config.epoch_test_n_samples,
                every_n_epochs=config.epoch_test_every_n_epochs,
                ignore_index=(
                    config.nodata_ignore_index if config.ignore_nodata_in_loss else None
                ),
            )
        )

    if no_fit:
        print("Skipping Graha trainer.fit() because --no-fit was set.")
        if config.graha_lightning_checkpoint is not None:
            semantic_graha_components.load_lightning_checkpoint_state(
                task,
                config.graha_lightning_checkpoint,
                "Graha",
            )
    else:
        fit_started_at = time.perf_counter()
        ckpt_path = (
            str(config.graha_lightning_checkpoint)
            if config.graha_lightning_checkpoint is not None
            else None
        )
        if ckpt_path is not None:
            print(f"Resuming Graha trainer.fit() from {ckpt_path}", flush=True)
        trainer.fit(task, datamodule=datamodule, ckpt_path=ckpt_path)
        _record_timing(
            record_timing,
            timing_rows,
            stage="fit",
            started_at=fit_started_at,
        )

    prediction_cache = None
    if config.cache_predictions:
        cache_started_at = time.perf_counter()
        prediction_cache = save_prediction_cache(
            task=task,
            datamodule=datamodule,
            output_dir=output_dir,
            model_name="graha",
            split=config.prediction_split,
            n_samples=config.prediction_n_samples,
        )
        _record_timing(
            record_timing,
            timing_rows,
            stage="prediction_cache",
            started_at=cache_started_at,
        )

    del trainer, task, datamodule, sample_batch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Released Graha model objects and cleared CUDA cache.", flush=True)
    _record_timing(
        record_timing,
        timing_rows,
        stage="total",
        started_at=graha_total_started_at,
    )
    return output_dir, prediction_cache
