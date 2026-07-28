"""Run Graha/Lunar-FM instance segmentation fine-tuning."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from lightning.pytorch import seed_everything

LFM_ROOT = Path(__file__).resolve().parents[3]
if str(LFM_ROOT) not in sys.path:
    sys.path.insert(0, str(LFM_ROOT))

from lfm.all_models.all_tasks.utils import create_timestamped_output_dir
from lfm.all_models.all_tasks.utils.utils import ensure_data_symlink
from lfm.full_model.inst_seg import instance_graha_components as components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simlink-dest",
        "--symlink-dest",
        dest="simlink_dest",
        type=str,
        default=None,
        help="Optional source directory for notebooks/full_model/data.",
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--base-output-dir", type=str, default=None)
    parser.add_argument("--pretrain-dir", type=str, default=None)
    parser.add_argument("--lightning-checkpoint", type=str, default=None)
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
    parser.add_argument("--image-glob", default="*.tif")
    parser.add_argument("--label-glob", default="*_label.*")
    parser.add_argument("--image-suffix", default="_input_wac_static_chip")
    parser.add_argument("--label-suffix", default="_label")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--stats-batch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--backbone-lr", type=float, default=5.0e-5)
    parser.add_argument("--head-lr", type=float, default=2.0e-4)
    parser.add_argument("--layer-decay", type=float, default=0.75)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument(
        "--anchor-sizes",
        type=lambda value: [[int(x)] for x in value.split(",")],
        default=[[8], [16], [32], [64]],
    )
    parser.add_argument(
        "--anchor-aspect-ratios",
        type=lambda value: [float(x) for x in value.split(",")],
        default=[0.5, 1.0, 2.0],
    )
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument(
        "--plot-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--prediction-split", choices=["train", "val", "test"], default="val"
    )
    parser.add_argument("--prediction-n-samples", type=int, default=5)
    parser.add_argument("--prediction-score-threshold", type=float, default=0.5)
    parser.add_argument("--progress-log-every-n-batches", type=int, default=25)
    parser.add_argument("--mask-shift", type=int, nargs=2, default=(0, 0))
    parser.add_argument("--ignore-nodata-in-loss", action="store_true")
    parser.add_argument("--nodata-ignore-index", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fit", action="store_true")
    parser.add_argument("--loss-smoke-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    started_at = time.perf_counter()
    args = parse_args()
    components.configure_proj_environment()
    notebook_dir = LFM_ROOT / "notebooks" / "full_model"
    ensure_data_symlink(args.simlink_dest, notebook_dir / "data")
    config = components.build_config(args)
    components.configure_python_paths(config)
    components.print_config(config)
    components.validate_required_paths(config)

    deps = components.import_project_dependencies()
    datamodule_cls = deps["LunarObjectDetectionInstanceMaskDatamodule"]
    task_cls = components.make_downstream_object_detection_task_class(
        deps["LunarObjectDetectionTask"]
    )

    output_dir = create_timestamped_output_dir(config.base_output_dir)
    (output_dir / "checkpoints" / "full_model").mkdir(parents=True, exist_ok=True)
    components.save_config(config, output_dir)
    seed_everything(config.seed)

    means, stds = components.get_normalization_stats(config, datamodule_cls)
    datamodule = components.create_datamodule(config, datamodule_cls, means, stds)
    sample_batch = components.inspect_batch(datamodule)
    task = components.create_task(config, task_cls, sample_batch)
    components.run_loss_smoke(task, sample_batch)

    if args.loss_smoke_only or args.no_fit:
        print("Skipping trainer.fit().")
        if config.lightning_checkpoint is not None:
            components.load_lightning_checkpoint_state(
                task, config.lightning_checkpoint
            )
        if config.plot_predictions:
            components.save_instance_prediction_plots(
                task,
                datamodule,
                config,
                output_dir,
            )
        components.save_timing(started_at, output_dir)
        return

    trainer = components.create_trainer(config, output_dir)
    ckpt_path = (
        str(config.lightning_checkpoint)
        if config.lightning_checkpoint is not None
        else None
    )
    if ckpt_path is not None:
        print(f"Resuming trainer.fit() from {ckpt_path}", flush=True)
    trainer.fit(task, datamodule=datamodule, ckpt_path=ckpt_path)
    if config.plot_predictions:
        components.save_instance_prediction_plots(task, datamodule, config, output_dir)
    components.save_timing(started_at, output_dir)


if __name__ == "__main__":
    main()
