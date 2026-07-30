"""GFFT semantic segmentation workflow orchestration."""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import Callback

from lfm.full_model.sem_seg import semantic_gfft_components


def _record_timing(
    record_timing: Callable[..., None] | None,
    timing_rows: list[dict[str, Any]] | None,
    *,
    stage: str,
    started_at: float,
) -> None:
    if record_timing is None or timing_rows is None:
        return
    record_timing(timing_rows, model="GFFT", stage=stage, started_at=started_at)


def run_gfft_workflow(
    config: Any,
    *,
    no_fit: bool,
    output_dir: Path,
    epoch_test_suite_callback_cls: type[Callback] | None = None,
    timing_rows: list[dict[str, Any]] | None = None,
    record_timing: Callable[..., None] | None = None,
) -> Path:
    """Run a single GFFT/Fourier-VQ MultiMAE semantic workflow."""
    print("\n=== GFFT/Fourier-VQ MultiMAE semantic segmentation ===", flush=True)
    total_started_at = time.perf_counter()
    semantic_gfft_components.configure_proj_environment()
    gfft_config = semantic_gfft_components.build_comparison_config(config, output_dir)
    semantic_gfft_components.configure_python_paths(gfft_config)
    semantic_gfft_components.print_config(gfft_config)
    semantic_gfft_components.validate_required_paths(gfft_config)

    deps = semantic_gfft_components.import_project_dependencies()
    datamodule_cls = deps[
        (
            "LunarSemanticFromInstanceDatamodule"
            if config.semantic_label_source == "instance"
            else "LunarSemanticMaskSegmentationDatamodule"
        )
    ]
    task_cls = semantic_gfft_components.make_downstream_shape_segmentation_task_class(
        deps["LunarShapeSegmentationTask"]
    )

    seed_everything(gfft_config.seed)
    stats_started_at = time.perf_counter()
    means, stds = semantic_gfft_components.get_normalization_stats(
        gfft_config,
        datamodule_cls,
    )
    _record_timing(
        record_timing,
        timing_rows,
        stage="stats",
        started_at=stats_started_at,
    )
    datamodule = semantic_gfft_components.create_datamodule(
        gfft_config,
        datamodule_cls,
        means,
        stds,
    )
    sample_batch = semantic_gfft_components.inspect_batch(datamodule)
    task = semantic_gfft_components.create_task(gfft_config, task_cls, sample_batch)
    semantic_gfft_components.inspect_backbone(task)
    trainer = semantic_gfft_components.create_trainer(
        gfft_config,
        output_dir,
        deps["ValidationPlotCallback"],
        plot_output_dir=output_dir,
        plots_subdir=Path("plots") / "single_model" / "gfft_model",
        checkpoint_subdir=Path("checkpoints") / "gfft_model",
    )

    if config.run_epoch_test_suite:
        if epoch_test_suite_callback_cls is None:
            raise ValueError("epoch_test_suite_callback_cls is required.")
        trainer.callbacks.append(
            epoch_test_suite_callback_cls(
                output_dir=output_dir,
                model_name="gfft",
                split=config.epoch_test_split,
                n_samples=config.epoch_test_n_samples,
                every_n_epochs=config.epoch_test_every_n_epochs,
                ignore_index=(
                    config.nodata_ignore_index if config.ignore_nodata_in_loss else None
                ),
            )
        )

    if no_fit:
        print("Skipping GFFT trainer.fit() because --no-fit/--skip-graha-fit was set.")
        if config.graha_lightning_checkpoint is not None:
            semantic_gfft_components.load_lightning_checkpoint_state(
                task,
                config.graha_lightning_checkpoint,
                "GFFT",
            )
    else:
        fit_started_at = time.perf_counter()
        ckpt_path = (
            str(config.graha_lightning_checkpoint)
            if config.graha_lightning_checkpoint is not None
            else None
        )
        if ckpt_path is not None:
            print(f"Resuming GFFT trainer.fit() from {ckpt_path}", flush=True)
        trainer.fit(task, datamodule=datamodule, ckpt_path=ckpt_path)
        _record_timing(
            record_timing,
            timing_rows,
            stage="fit",
            started_at=fit_started_at,
        )

    del trainer, task, datamodule, sample_batch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Released GFFT model objects and cleared CUDA cache.", flush=True)
    _record_timing(
        record_timing,
        timing_rows,
        stage="total",
        started_at=total_started_at,
    )
    return output_dir
