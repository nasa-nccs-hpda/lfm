"""Toy semantic segmentation workflow orchestration."""

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
from lfm.toy_model.sem_seg import semantic_toy_components


def _record_timing(
    record_timing: Callable[..., None] | None,
    timing_rows: list[dict[str, Any]] | None,
    *,
    stage: str,
    started_at: float,
) -> None:
    if record_timing is None or timing_rows is None:
        return
    record_timing(timing_rows, model="Toy", stage=stage, started_at=started_at)


def run_toy_workflow(
    config: Any,
    *,
    output_dir: Path,
    normalization_modality_info: Path | None = None,
    epoch_test_suite_callback_cls: type[Callback] | None = None,
    timing_rows: list[dict[str, Any]] | None = None,
    record_timing: Callable[..., None] | None = None,
) -> Path | None:
    """Run the Toy semantic segmentation path."""
    toy_total_started_at = time.perf_counter()
    seed_everything(config.seed)

    toy_datamodule = semantic_toy_components.create_datamodule(
        config,
        output_dir,
        normalization_modality_info=normalization_modality_info,
    )
    if toy_datamodule.weight_assignments is None:
        raise RuntimeError("Toy DataModule did not create weight assignments.")

    toy_model = semantic_toy_components.create_model(
        config,
        toy_datamodule.weight_assignments,
    )
    toy_task = semantic_toy_components.create_lightning_module(config, toy_model)
    toy_trainer = semantic_toy_components.create_trainer(
        config,
        output_dir,
        plots_subdir=Path("plots") / "single_model" / "toy_model",
        epoch_test_suite_callback_cls=epoch_test_suite_callback_cls,
    )
    print("Toy Lightning trainer created.", flush=True)

    if config.skip_toy_fit:
        print("Skipping Toy trainer.fit().")
        if config.toy_lightning_checkpoint is not None:
            semantic_toy_components.load_lightning_checkpoint_state(
                toy_task,
                config.toy_lightning_checkpoint,
                "Toy",
            )
    else:
        print("Starting Toy trainer.fit()...", flush=True)
        fit_started_at = time.perf_counter()
        toy_ckpt_path = (
            str(config.toy_lightning_checkpoint)
            if config.toy_lightning_checkpoint is not None
            else None
        )
        if toy_ckpt_path is not None:
            print(f"Resuming Toy trainer.fit() from {toy_ckpt_path}", flush=True)
        toy_trainer.fit(
            toy_task,
            datamodule=toy_datamodule,
            ckpt_path=toy_ckpt_path,
        )
        _record_timing(
            record_timing,
            timing_rows,
            stage="fit",
            started_at=fit_started_at,
        )
        print("Toy trainer.fit() complete.", flush=True)

    toy_prediction_cache = None
    if config.cache_predictions:
        cache_started_at = time.perf_counter()
        toy_prediction_cache = save_prediction_cache(
            task=toy_task,
            datamodule=toy_datamodule,
            output_dir=output_dir,
            model_name="toy",
            split=config.prediction_split,
            n_samples=config.prediction_n_samples,
        )
        _record_timing(
            record_timing,
            timing_rows,
            stage="prediction_cache",
            started_at=cache_started_at,
        )

    if not config.skip_toy_fit:
        print("Starting Toy trainer.test() on final weights...", flush=True)
        test_started_at = time.perf_counter()
        toy_trainer.test(toy_task, datamodule=toy_datamodule, ckpt_path=None)
        _record_timing(
            record_timing,
            timing_rows,
            stage="test_final",
            started_at=test_started_at,
        )
        print("Toy trainer.test() complete.", flush=True)

    del toy_trainer, toy_task, toy_model, toy_datamodule
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Released Toy model objects and cleared CUDA cache.", flush=True)
    _record_timing(
        record_timing,
        timing_rows,
        stage="total",
        started_at=toy_total_started_at,
    )
    return toy_prediction_cache
