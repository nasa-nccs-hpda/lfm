"""Graha instance segmentation workflow orchestration."""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import Callback

from lfm.full_model.all_tasks.utils import (
    plot_instance_cache_predictions,
    save_graha_instance_prediction_cache,
)
from lfm.full_model.inst_seg import instance_graha_components


def run_graha_workflow(
    config: Any,
    output_dir: Path,
    *,
    validation_plot_callback_cls: type[Callback] | None = None,
    epoch_test_suite_callback_cls: type[Callback] | None = None,
) -> Path | None:
    print("\n=== Graha/Lunar-FM Mask R-CNN instance segmentation ===", flush=True)
    started = time.perf_counter()
    instance_graha_components.configure_proj_environment()
    graha_config = instance_graha_components.build_comparison_config(config, output_dir)
    instance_graha_components.configure_python_paths(graha_config)
    instance_graha_components.print_config(graha_config)
    instance_graha_components.validate_required_paths(graha_config)

    deps = instance_graha_components.import_project_dependencies()
    datamodule_cls = deps["LunarObjectDetectionInstanceMaskDatamodule"]
    task_cls = instance_graha_components.make_downstream_object_detection_task_class(
        deps["LunarObjectDetectionTask"]
    )

    seed_everything(graha_config.seed)
    means, stds = instance_graha_components.get_normalization_stats(
        graha_config,
        datamodule_cls,
    )
    graha_datamodule = instance_graha_components.create_datamodule(
        graha_config,
        datamodule_cls,
        means,
        stds,
    )
    graha_sample_batch = instance_graha_components.inspect_batch(graha_datamodule)
    graha_task = instance_graha_components.create_task(
        graha_config,
        task_cls,
        graha_sample_batch,
    )
    instance_graha_components.run_loss_smoke(graha_task, graha_sample_batch)
    graha_trainer = instance_graha_components.create_trainer(
        graha_config,
        output_dir,
    )
    if validation_plot_callback_cls is not None:
        graha_trainer.callbacks.append(
            validation_plot_callback_cls(
                output_dir,
                n_samples=config.plot_n_samples,
                every_n_epochs=config.plot_every_n_epochs,
                score_threshold=config.prediction_score_threshold,
            )
        )
    if config.run_epoch_test_suite:
        if epoch_test_suite_callback_cls is None:
            raise ValueError("epoch_test_suite_callback_cls is required.")
        graha_trainer.callbacks.append(
            epoch_test_suite_callback_cls(
                output_dir=output_dir,
                model_name="full_model",
                split=config.epoch_test_split,
                n_samples=config.epoch_test_n_samples,
                every_n_epochs=config.epoch_test_every_n_epochs,
                score_threshold=config.prediction_score_threshold,
                image_processor=None,
            )
        )
    graha_prediction_cache = None

    if config.skip_graha_fit:
        print("Skipping Graha trainer.fit().", flush=True)
        if config.graha_lightning_checkpoint is not None:
            instance_graha_components.load_lightning_checkpoint_state(
                graha_task,
                config.graha_lightning_checkpoint,
            )
    else:
        graha_ckpt_path = (
            str(config.graha_lightning_checkpoint)
            if config.graha_lightning_checkpoint is not None
            else None
        )
        if graha_ckpt_path is not None:
            print(f"Resuming Graha trainer.fit() from {graha_ckpt_path}", flush=True)
        print("Starting Graha trainer.fit()...", flush=True)
        graha_trainer.fit(
            graha_task,
            datamodule=graha_datamodule,
            ckpt_path=graha_ckpt_path,
        )
        print("Finished Graha trainer.fit().", flush=True)
        graha_checkpoints = sorted(
            (output_dir / "checkpoints" / "full_model").glob("*.ckpt")
        )
        print(f"[Graha] saved {len(graha_checkpoints)} checkpoint file(s).", flush=True)

    if graha_config.plot_predictions:
        graha_prediction_cache = save_graha_instance_prediction_cache(
            task=graha_task,
            datamodule=graha_datamodule,
            output_dir=output_dir,
            model_name="graha",
            split=config.prediction_split,
            n_samples=config.prediction_n_samples,
            score_threshold=config.prediction_score_threshold,
        )
        plot_path = plot_instance_cache_predictions(
            graha_prediction_cache,
            output_dir / "plots" / "single_model" / "full_model",
            model_name="graha",
            n_samples=config.prediction_n_samples,
            filename=f"{config.prediction_split}_instance_predictions.png",
        )
        print(f"[Graha] saved validation prediction plot: {plot_path}", flush=True)

    elapsed = time.perf_counter() - started
    print(f"Graha elapsed seconds: {elapsed:.3f}", flush=True)
    del graha_trainer, graha_task, graha_datamodule, graha_sample_batch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return graha_prediction_cache
