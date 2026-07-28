"""Toy instance segmentation workflow orchestration."""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import Callback

from lfm.all_models.all_tasks.utils import (
    plot_instance_cache_predictions,
    save_graha_instance_prediction_cache,
    save_toy_instance_prediction_cache,
)
from lfm.toy_model.inst_seg import instance_toy_components


def run_toy_workflow(
    config: Any,
    output_dir: Path,
    *,
    normalization_modality_info: Path | None = None,
    epoch_test_suite_callback_cls: type[Callback] | None = None,
) -> Path | None:
    title = (
        "Toy Mask R-CNN (DINOv3 backbone)"
        if config.toy_architecture == "dino-mask-rcnn"
        else "Toy Mask2Former (DINOv3 backbone)"
    )
    print(f"\n=== {title} instance segmentation ===", flush=True)
    started = time.perf_counter()
    seed_everything(config.seed)
    toy_datamodule = instance_toy_components.create_datamodule(
        config,
        normalization_modality_info=normalization_modality_info,
    )
    toy_task = instance_toy_components.create_task(
        config,
        toy_datamodule.weight_assignments or [],
    )
    toy_image_processor = instance_toy_components.create_image_processor(config)
    toy_trainer = instance_toy_components.create_trainer(
        config,
        output_dir,
        toy_image_processor,
        epoch_test_suite_callback_cls=epoch_test_suite_callback_cls,
    )
    toy_prediction_cache = None

    if config.skip_toy_fit:
        print("Skipping Toy trainer.fit().", flush=True)
        if config.toy_lightning_checkpoint is not None:
            instance_toy_components.load_lightning_checkpoint_state(
                toy_task,
                config.toy_lightning_checkpoint,
                "Toy",
            )
    else:
        toy_ckpt_path = (
            str(config.toy_lightning_checkpoint)
            if config.toy_lightning_checkpoint is not None
            else None
        )
        if toy_ckpt_path is not None:
            print(f"Resuming Toy trainer.fit() from {toy_ckpt_path}", flush=True)
        print("Starting Toy trainer.fit()...", flush=True)
        toy_trainer.fit(
            toy_task,
            datamodule=toy_datamodule,
            ckpt_path=toy_ckpt_path,
        )
        print("Finished Toy trainer.fit().", flush=True)
        toy_checkpoints = sorted(
            (output_dir / "checkpoints" / "toy_model").glob("*.ckpt")
        )
        print(f"[Toy] saved {len(toy_checkpoints)} checkpoint file(s).", flush=True)

    if toy_image_processor is None:
        toy_prediction_cache = save_graha_instance_prediction_cache(
            task=toy_task,
            datamodule=toy_datamodule,
            output_dir=output_dir,
            model_name="toy",
            split=config.prediction_split,
            n_samples=config.prediction_n_samples,
            score_threshold=config.prediction_score_threshold,
        )
    else:
        toy_prediction_cache = save_toy_instance_prediction_cache(
            task=toy_task,
            datamodule=toy_datamodule,
            output_dir=output_dir,
            image_processor=toy_image_processor,
            model_name="toy",
            split=config.prediction_split,
            n_samples=config.prediction_n_samples,
            score_threshold=config.prediction_score_threshold,
        )
    plot_path = plot_instance_cache_predictions(
        toy_prediction_cache,
        output_dir / "plots" / "single_model" / "toy_model",
        model_name="toy",
        n_samples=config.prediction_n_samples,
        filename=f"{config.prediction_split}_instance_predictions.png",
    )
    print(f"[Toy] saved validation prediction plot: {plot_path}", flush=True)

    elapsed = time.perf_counter() - started
    print(f"Toy elapsed seconds: {elapsed:.3f}", flush=True)
    del toy_trainer, toy_task, toy_datamodule
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return toy_prediction_cache
