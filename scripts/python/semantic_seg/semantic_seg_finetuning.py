"""Run Graha/Lunar-FM semantic segmentation fine-tuning."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lightning.pytorch import seed_everything

LFM_ROOT = Path(__file__).resolve().parents[3]
if str(LFM_ROOT) not in sys.path:
    sys.path.insert(0, str(LFM_ROOT))

from lfm.full_model.sem_seg import semantic_graha_components as components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--base-output-dir", type=str, default=None)
    parser.add_argument("--pretrain-dir", type=str, default=None)
    parser.add_argument(
        "--lightning-checkpoint",
        type=str,
        default=None,
        help="Optional Lightning .ckpt. Resumes fit, or loads weights when --no-fit is set.",
    )
    parser.add_argument(
        "--graha-input-modality-mode", choices=["new-wac", "vis-uv"], default="new-wac"
    )
    parser.add_argument(
        "--graha-vis-uv-merge-method", choices=["mean", "max"], default="mean"
    )
    parser.add_argument(
        "--normalization-source",
        choices=["pretrain", "finetune"],
        default="pretrain",
        help="Use TerraMind pretraining stats or finetuning train-split stats.",
    )
    parser.add_argument(
        "--normalization-modality",
        choices=["vis_uv", "nac"],
        default="vis_uv",
    )
    parser.add_argument("--band-filter", type=int, nargs="+", default=None)
    parser.add_argument(
        "--semantic-label-source",
        choices=["semantic", "instance"],
        default="semantic",
    )
    parser.add_argument("--image-glob", default="*.tif")
    parser.add_argument("--label-glob", default="*_label.*")
    parser.add_argument("--image-suffix", default="_input_wac_static_chip")
    parser.add_argument("--label-suffix", default="_label")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--ignore-nodata-in-loss", action="store_true")
    parser.add_argument("--nodata-ignore-index", type=int, default=-1)
    parser.add_argument("--shape-loss-weight", type=float, default=0.05)
    parser.add_argument("--shape-loss-pad-frac", type=float, default=0.3)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--stats-batch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--cache-predictions", action="store_true")
    parser.add_argument(
        "--prediction-split", choices=["train", "val", "test"], default="val"
    )
    parser.add_argument("--prediction-n-samples", type=int, default=20)
    parser.add_argument("--progress-log-every-n-batches", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    components.configure_proj_environment()
    config = components.build_config(args)
    components.configure_python_paths(config)
    components.print_config(config)
    components.validate_required_paths(config)

    deps = components.import_project_dependencies()
    datamodule_cls = deps[
        (
            "LunarSemanticFromInstanceDatamodule"
            if config.semantic_label_source == "instance"
            else "LunarSemanticMaskSegmentationDatamodule"
        )
    ]
    task_cls = components.make_downstream_shape_segmentation_task_class(
        deps["LunarShapeSegmentationTask"]
    )

    output_dir = components.create_output_dirs(
        config,
        deps["create_timestamped_output_dir"],
    )
    seed_everything(config.seed)

    means, stds = components.get_normalization_stats(config, datamodule_cls)
    datamodule = components.create_datamodule(config, datamodule_cls, means, stds)
    sample_batch = components.inspect_batch(datamodule)
    task = components.create_task(config, task_cls, sample_batch)
    components.inspect_backbone(task)

    trainer = components.create_trainer(
        config,
        output_dir,
        deps["ValidationPlotCallback"],
    )
    if args.no_fit:
        print("Skipping trainer.fit() because --no-fit was set.")
        if config.lightning_checkpoint is not None:
            components.load_lightning_checkpoint_state(
                task,
                config.lightning_checkpoint,
                "Graha",
            )
            if config.cache_predictions:
                deps["save_prediction_cache"](
                    task=task,
                    datamodule=datamodule,
                    output_dir=output_dir,
                    model_name="graha",
                    split=config.prediction_split,
                    n_samples=config.prediction_n_samples,
                )
        return

    ckpt_path = (
        str(config.lightning_checkpoint)
        if config.lightning_checkpoint is not None
        else None
    )
    if ckpt_path is not None:
        print(f"Resuming trainer.fit() from {ckpt_path}", flush=True)
    trainer.fit(task, datamodule=datamodule, ckpt_path=ckpt_path)
    if config.cache_predictions:
        deps["save_prediction_cache"](
            task=task,
            datamodule=datamodule,
            output_dir=output_dir,
            model_name="graha",
            split=config.prediction_split,
            n_samples=config.prediction_n_samples,
        )


if __name__ == "__main__":
    main()
