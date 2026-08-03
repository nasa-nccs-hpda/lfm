"""GFFT instance segmentation workflow orchestration."""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import Callback

from lfm.full_model.inst_seg import instance_gfft_components


def run_gfft_workflow(
    config: Any,
    output_dir: Path,
    *,
    epoch_test_suite_callback_cls: type[Callback] | None = None,
) -> None:
    """Run a single GFFT/Fourier-VQ MultiMAE Mask R-CNN instance workflow."""
    print(
        "\n=== GFFT/Fourier-VQ MultiMAE Mask R-CNN instance segmentation ===",
        flush=True,
    )
    started = time.perf_counter()
    instance_gfft_components.configure_proj_environment()
    gfft_config = instance_gfft_components.build_finetuning_config(config, output_dir)
    instance_gfft_components.configure_python_paths(gfft_config)
    instance_gfft_components.print_config(gfft_config)
    instance_gfft_components.validate_required_paths(gfft_config)

    deps = instance_gfft_components.import_project_dependencies()
    datamodule_cls = deps["GrahaObjectDetectionInstanceDataModule"]
    task_cls = instance_gfft_components.make_downstream_object_detection_task_class(
        deps["LunarObjectDetectionTask"]
    )

    seed_everything(gfft_config.seed)
    means, stds = instance_gfft_components.get_normalization_stats(
        gfft_config,
        datamodule_cls,
    )
    datamodule = instance_gfft_components.create_datamodule(
        gfft_config,
        datamodule_cls,
        means,
        stds,
    )
    sample_batch = instance_gfft_components.inspect_batch(datamodule)
    task = instance_gfft_components.create_task(gfft_config, task_cls, sample_batch)
    instance_gfft_components.run_loss_smoke(task, sample_batch)
    trainer = instance_gfft_components.create_trainer(gfft_config, output_dir)

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
                score_threshold=config.prediction_score_threshold,
                image_processor=None,
            )
        )

    if config.skip_graha_fit:
        print("Skipping GFFT trainer.fit() because --no-fit/--skip-graha-fit was set.")
        if config.graha_lightning_checkpoint is not None:
            instance_gfft_components.load_lightning_checkpoint_state(
                task,
                config.graha_lightning_checkpoint,
            )
    else:
        ckpt_path = (
            str(config.graha_lightning_checkpoint)
            if config.graha_lightning_checkpoint is not None
            else None
        )
        if ckpt_path is not None:
            print(f"Resuming GFFT trainer.fit() from {ckpt_path}", flush=True)
        print("Starting GFFT trainer.fit()...", flush=True)
        trainer.fit(task, datamodule=datamodule, ckpt_path=ckpt_path)
        print("Finished GFFT trainer.fit().", flush=True)
        checkpoints = sorted((output_dir / "checkpoints" / "gfft_model").glob("*.ckpt"))
        print(f"[GFFT] saved {len(checkpoints)} checkpoint file(s).", flush=True)

    elapsed = time.perf_counter() - started
    print(f"GFFT elapsed seconds: {elapsed:.3f}", flush=True)
    del trainer, task, datamodule, sample_batch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
