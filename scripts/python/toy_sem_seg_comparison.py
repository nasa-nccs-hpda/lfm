"""Train the toy DINO semantic segmentation model on split full-model data."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from argparse import Namespace
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from lfm.full_model import lfm_seg_finetuning_direct as graha_workflow
from lfm.toy_model.sem_seg.lightning_wrappers.toy_sem_seg_datamodule import (
    ToySemSegSplitDataModule,
)
from lfm.toy_model.sem_seg.lightning_wrappers.toy_sem_seg_lightning import (
    ToySemSegLightningModule,
)
from lfm.full_model.utils import (
    ValidationPlotCallback,
    create_timestamped_output_dir,
    evaluate_prediction_caches,
    plot_prediction_cache_comparison,
    save_prediction_cache,
)
from lfm.full_model.utils.utils import ensure_data_symlink
from lfm.toy_model.sem_seg.sseg_model import DINOSegmentation, load_dinov3_encoder


@dataclass(frozen=True)
class ToyComparisonConfig:
    repo_root: Path
    notebook_dir: Path
    data_root: Path
    base_output_dir: Path
    dino_checkpoint: Path | None
    dino_lightning_checkpoint: Path | None
    band_filter: list[int]
    target_size: tuple[int, int]
    spatial_transform: str
    max_train_samples: int | None
    max_val_samples: int | None
    max_test_samples: int | None
    batch_size: int
    num_workers: int
    max_epochs: int
    learning_rate: float
    weight_decay: float
    loss_type: str
    freeze_encoder: bool
    normalize_inputs: bool
    toy_gradient_clip_val: float | None
    plot_every_n_epochs: int
    plot_n_samples: int
    cache_predictions: bool
    prediction_split: str
    prediction_n_samples: int
    graha_base_output_dir: Path
    graha_pretrain_dir: Path | None
    graha_lightning_checkpoint: Path | None
    graha_stats_batch_size: int
    graha_batch_size: int
    graha_num_workers: int
    skip_dino_fit: bool
    skip_graha_fit: bool
    seed: int


def build_config(args: argparse.Namespace) -> ToyComparisonConfig:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    notebook_dir = repo_root / "notebooks" / "full_model"
    scripts_output_dir = repo_root / "scripts" / "outputs"
    data_root = Path(args.data_root).resolve() if args.data_root else notebook_dir / "data"
    base_output_dir = (
        Path(args.base_output_dir).resolve()
        if args.base_output_dir
        else scripts_output_dir / "toy_sem_seg_comparison"
    )
    dino_checkpoint = Path(args.dino_checkpoint).resolve() if args.dino_checkpoint else None
    dino_lightning_checkpoint = (
        Path(args.dino_lightning_checkpoint).resolve()
        if args.dino_lightning_checkpoint
        else None
    )
    graha_base_output_dir = (
        Path(args.graha_base_output_dir).resolve()
        if args.graha_base_output_dir
        else scripts_output_dir / "graha_finetuning"
    )
    graha_pretrain_dir = (
        Path(args.graha_pretrain_dir).resolve() if args.graha_pretrain_dir else None
    )
    graha_lightning_checkpoint = (
        Path(args.graha_lightning_checkpoint).resolve()
        if args.graha_lightning_checkpoint
        else None
    )

    return ToyComparisonConfig(
        repo_root=repo_root,
        notebook_dir=notebook_dir,
        data_root=data_root,
        base_output_dir=base_output_dir,
        dino_checkpoint=dino_checkpoint,
        dino_lightning_checkpoint=dino_lightning_checkpoint,
        band_filter=args.band_filter,
        target_size=(args.target_size, args.target_size),
        spatial_transform=args.spatial_transform,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss_type=args.loss_type,
        freeze_encoder=args.freeze_encoder,
        normalize_inputs=args.normalize_inputs,
        toy_gradient_clip_val=None
        if args.disable_toy_gradient_clipping
        else args.toy_gradient_clip_val,
        plot_every_n_epochs=args.plot_every_n_epochs,
        plot_n_samples=args.plot_n_samples,
        cache_predictions=args.cache_predictions,
        prediction_split=args.prediction_split,
        prediction_n_samples=args.prediction_n_samples,
        graha_base_output_dir=graha_base_output_dir,
        graha_pretrain_dir=graha_pretrain_dir,
        graha_lightning_checkpoint=graha_lightning_checkpoint,
        graha_stats_batch_size=args.graha_stats_batch_size,
        graha_batch_size=args.graha_batch_size,
        graha_num_workers=args.graha_num_workers,
        skip_dino_fit=args.no_fit or args.skip_dino_fit,
        skip_graha_fit=args.no_fit or args.skip_graha_fit,
        seed=args.seed,
    )


def validate_data_paths(config: ToyComparisonConfig) -> None:
    required = []
    for split in ["train", "val", "test"]:
        required.extend(
            [
                config.data_root / split / "chips",
                config.data_root / split / "labels",
            ]
        )
    for checkpoint_path in [
        config.dino_checkpoint,
        config.dino_lightning_checkpoint,
        config.graha_pretrain_dir,
        config.graha_lightning_checkpoint,
    ]:
        if checkpoint_path is not None:
            required.append(checkpoint_path)
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required split data paths:\n" + "\n".join(str(path) for path in missing)
        )


def save_config(config: ToyComparisonConfig, output_dir: Path) -> None:
    def encode(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return list(value)
        return value

    payload = {key: encode(value) for key, value in asdict(config).items()}
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_lightning_checkpoint_state(task: torch.nn.Module, checkpoint_path: Path, model_name: str) -> None:
    """Load Lightning checkpoint weights into an already-built task."""
    checkpoint_path = Path(checkpoint_path).resolve()
    print(f"Loading {model_name} Lightning checkpoint weights from {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    task.load_state_dict(state_dict, strict=True)
    print(f"Loaded {model_name} Lightning checkpoint weights.", flush=True)


def _format_seconds(seconds: float) -> str:
    whole_seconds = int(round(seconds))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def save_timing_summary(timing_rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not timing_rows:
        return

    json_path = output_dir / "timing_summary.json"
    csv_path = output_dir / "timing_summary.csv"
    fieldnames = ["model", "stage", "seconds", "elapsed_hms"]

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(timing_rows, f, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(timing_rows)

    print(f"Saved timing summary to {csv_path}")


def record_timing(
    timing_rows: list[dict[str, Any]],
    *,
    model: str,
    stage: str,
    started_at: float,
) -> None:
    elapsed = time.perf_counter() - started_at
    timing_rows.append(
        {
            "model": model,
            "stage": stage,
            "seconds": round(elapsed, 3),
            "elapsed_hms": _format_seconds(elapsed),
        }
    )
    print(f"[timing] {model} {stage}: {_format_seconds(elapsed)} ({elapsed:.3f}s)", flush=True)


def create_datamodule(config: ToyComparisonConfig, output_dir: Path) -> ToySemSegSplitDataModule:
    datamodule = ToySemSegSplitDataModule(
        data_root=config.data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        target_size=config.target_size,
        spatial_transform=config.spatial_transform,
        band_filter=config.band_filter,
        normalize_inputs=config.normalize_inputs,
        max_train_samples=config.max_train_samples,
        max_val_samples=config.max_val_samples,
        max_test_samples=config.max_test_samples,
        output_dir=output_dir,
    )
    datamodule.setup("fit")
    print("Data sanity summary:")
    for key, value in datamodule.get_sanity_summary().items():
        print(f"  {key}: {value}")
    return datamodule


def create_model(config: ToyComparisonConfig, weight_assignments: list[str]) -> DINOSegmentation:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.dino_checkpoint is not None:
        encoder = load_dinov3_encoder(
            weights_local_checkpoint=str(config.dino_checkpoint),
            device=device,
        )
    else:
        encoder = load_dinov3_encoder(device=device)

    return DINOSegmentation(
        encoder=encoder,
        num_classes=2,
        img_size=config.target_size,
        freeze_encoder=config.freeze_encoder,
        weight_assignments=weight_assignments,
    )


def create_lightning_module(
    config: ToyComparisonConfig,
    model: DINOSegmentation,
) -> ToySemSegLightningModule:
    return ToySemSegLightningModule(
        model=model,
        loss_type=config.loss_type,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_epochs=config.max_epochs,
        max_grad_norm=config.toy_gradient_clip_val,
    )


def create_trainer(
    config: ToyComparisonConfig,
    output_dir: Path,
    *,
    plots_subdir: str | Path = "plots",
) -> Trainer:
    print("Creating Lightning trainer...", flush=True)
    return Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="32",
        max_epochs=config.max_epochs,
        check_val_every_n_epoch=1,
        log_every_n_steps=5,
        logger=False,
        callbacks=[
            ModelCheckpoint(
                dirpath=str(output_dir / "checkpoints" / "toy_model"),
                monitor="val_loss",
                mode="min",
                filename="model-epoch-{epoch:02d}-val-loss={val_loss:.3f}",
                auto_insert_metric_name=False,
                save_top_k=-1,
                save_last=True,
                every_n_epochs=1,
            ),
            ValidationPlotCallback(
                output_dir=output_dir,
                n_samples=config.plot_n_samples,
                every_n_epochs=config.plot_every_n_epochs,
                plots_subdir=plots_subdir,
                display_method="minmax",
                dpi=150,
            ),
        ],
    )


def run_graha_workflow(
    config: ToyComparisonConfig,
    *,
    no_fit: bool,
    comparison_output_dir: Path,
    timing_rows: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path | None]:
    """Run the Graha/Lunar-FM path with the same split data and cache settings."""
    graha_total_started_at = time.perf_counter()
    graha_workflow.configure_proj_environment()
    graha_args = Namespace(
        data_root=str(config.data_root),
        base_output_dir=str(comparison_output_dir),
        pretrain_dir=str(config.graha_pretrain_dir) if config.graha_pretrain_dir else None,
        lightning_checkpoint=str(config.graha_lightning_checkpoint)
        if config.graha_lightning_checkpoint
        else None,
        crop_size=config.target_size[0],
        stats_batch_size=config.graha_stats_batch_size,
        batch_size=config.graha_batch_size,
        num_workers=config.graha_num_workers,
        max_epochs=config.max_epochs,
        cache_predictions=config.cache_predictions,
        prediction_split=config.prediction_split,
        prediction_n_samples=config.prediction_n_samples,
        seed=config.seed,
        no_fit=no_fit,
    )
    graha_config = graha_workflow.build_config(graha_args)
    graha_workflow.configure_python_paths(graha_config)
    graha_workflow.print_config(graha_config)
    graha_workflow.validate_required_paths(graha_config)

    deps = graha_workflow.import_project_dependencies()
    datamodule_cls = deps["LunarSemanticSegmentationDatamodule"]
    task_cls = graha_workflow.make_notebook_task_class(deps["LunarShapeSegmentationTask"])

    output_dir = graha_workflow.create_output_dirs(
        graha_config,
        deps["create_timestamped_output_dir"],
        use_timestamp=False,
    )
    seed_everything(graha_config.seed)
    stats_started_at = time.perf_counter()
    means, stds = graha_workflow.calculate_train_stats(graha_config, datamodule_cls)
    if timing_rows is not None:
        record_timing(
            timing_rows,
            model="Graha",
            stage="stats",
            started_at=stats_started_at,
        )
    datamodule = graha_workflow.create_datamodule(
        graha_config,
        datamodule_cls,
        means,
        stds,
    )
    sample_batch = graha_workflow.inspect_batch(datamodule)
    task = graha_workflow.create_task(graha_config, task_cls, sample_batch)
    graha_workflow.inspect_backbone(task)
    trainer = graha_workflow.create_trainer(
        graha_config,
        output_dir,
        deps["ValidationPlotCallback"],
        plot_output_dir=comparison_output_dir,
        plots_subdir=Path("plots") / "single_model" / "full_model",
        checkpoint_subdir=Path("checkpoints") / "full_model",
    )

    if no_fit:
        print("Skipping Graha trainer.fit() because --no-fit was set.")
        if config.graha_lightning_checkpoint is not None:
            load_lightning_checkpoint_state(
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
        if timing_rows is not None:
            record_timing(
                timing_rows,
                model="Graha",
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
        if timing_rows is not None:
            record_timing(
                timing_rows,
                model="Graha",
                stage="prediction_cache",
                started_at=cache_started_at,
            )

    del trainer, task, datamodule, sample_batch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Released Graha model objects and cleared CUDA cache.", flush=True)
    if timing_rows is not None:
        record_timing(
            timing_rows,
            model="Graha",
            stage="total",
            started_at=graha_total_started_at,
        )
    return output_dir, prediction_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simlink-dest",
        "--symlink-dest",
        dest="simlink_dest",
        type=str,
        default=None,
        help=(
            "Optional source directory for notebooks/full_model/data. If ./data is already a "
            "symlink, it must point to this same directory."
        ),
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--base-output-dir", type=str, default=None)
    parser.add_argument("--dino-checkpoint", type=str, default=None)
    parser.add_argument(
        "--dino-lightning-checkpoint",
        type=str,
        default=None,
        help="Optional DINO Lightning .ckpt. Resumes fit, or loads weights when DINO fit is skipped.",
    )
    parser.add_argument("--band-filter", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--spatial-transform", choices=["resize", "crop"], default="crop")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--loss-type", type=str, default="focal_dice")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument(
        "--normalize-inputs",
        action="store_true",
        help="Enable toy DINO z-score normalization using train-split stats.",
    )
    parser.add_argument("--toy-gradient-clip-val", type=float, default=1.0)
    parser.add_argument(
        "--disable-toy-gradient-clipping",
        action="store_true",
        help="Disable toy DINO gradient clipping to match Graha's current trainer path.",
    )
    parser.add_argument("--plot-every-n-epochs", type=int, default=1)
    parser.add_argument("--plot-n-samples", type=int, default=5)
    parser.add_argument("--cache-predictions", action="store_true")
    parser.add_argument("--prediction-split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--prediction-n-samples", type=int, default=20)
    parser.add_argument("--graha-base-output-dir", type=str, default=None)
    parser.add_argument("--graha-pretrain-dir", type=str, default=None)
    parser.add_argument(
        "--graha-lightning-checkpoint",
        type=str,
        default=None,
        help="Optional Graha Lightning .ckpt. Resumes fit, or loads weights when Graha fit is skipped.",
    )
    parser.add_argument("--graha-stats-batch-size", type=int, default=16)
    parser.add_argument("--graha-batch-size", type=int, default=16)
    parser.add_argument("--graha-num-workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fit", action="store_true", help="Build data/model/trainer but skip fit.")
    parser.add_argument("--skip-dino-fit", action="store_true", help="Skip only DINO fitting.")
    parser.add_argument("--skip-graha-fit", action="store_true", help="Skip only Graha fitting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notebook_dir = Path(__file__).resolve().parents[1] / "notebooks" / "full_model"
    ensure_data_symlink(args.simlink_dest, notebook_dir / "data")
    config = build_config(args)
    validate_data_paths(config)
    output_dir = create_timestamped_output_dir(config.base_output_dir)
    save_config(config, output_dir)
    timing_rows: list[dict[str, Any]] = []

    seed_everything(config.seed)
    dino_total_started_at = time.perf_counter()
    datamodule = create_datamodule(config, output_dir)
    if datamodule.weight_assignments is None:
        raise RuntimeError("DataModule did not create weight assignments.")

    model = create_model(config, datamodule.weight_assignments)
    task = create_lightning_module(config, model)
    trainer = create_trainer(
        config,
        output_dir,
        plots_subdir=Path("plots") / "single_model" / "toy_model",
    )
    print("Lightning trainer created.", flush=True)

    if config.skip_dino_fit:
        print("Skipping DINO trainer.fit().")
        if config.dino_lightning_checkpoint is not None:
            load_lightning_checkpoint_state(task, config.dino_lightning_checkpoint, "DINO")
    else:
        print("Starting DINO trainer.fit()...", flush=True)
        fit_started_at = time.perf_counter()
        ckpt_path = (
            str(config.dino_lightning_checkpoint)
            if config.dino_lightning_checkpoint is not None
            else None
        )
        if ckpt_path is not None:
            print(f"Resuming DINO trainer.fit() from {ckpt_path}", flush=True)
        trainer.fit(task, datamodule=datamodule, ckpt_path=ckpt_path)
        record_timing(
            timing_rows,
            model="DINO",
            stage="fit",
            started_at=fit_started_at,
        )
        print("DINO trainer.fit() complete.", flush=True)

    toy_prediction_cache = None
    if config.cache_predictions:
        cache_started_at = time.perf_counter()
        toy_prediction_cache = save_prediction_cache(
            task=task,
            datamodule=datamodule,
            output_dir=output_dir,
            model_name="toy",
            split=config.prediction_split,
            n_samples=config.prediction_n_samples,
        )
        record_timing(
            timing_rows,
            model="DINO",
            stage="prediction_cache",
            started_at=cache_started_at,
        )

    if not config.skip_dino_fit:
        print("Starting DINO trainer.test() on final weights...", flush=True)
        test_started_at = time.perf_counter()
        trainer.test(task, datamodule=datamodule, ckpt_path=None)
        record_timing(
            timing_rows,
            model="DINO",
            stage="test_final",
            started_at=test_started_at,
        )
        print("DINO trainer.test() complete.", flush=True)

    del trainer, task, model, datamodule
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Released DINO model objects and cleared CUDA cache.", flush=True)
    record_timing(
        timing_rows,
        model="DINO",
        stage="total",
        started_at=dino_total_started_at,
    )

    _, graha_prediction_cache = run_graha_workflow(
        config,
        no_fit=config.skip_graha_fit,
        comparison_output_dir=output_dir,
        timing_rows=timing_rows,
    )

    if config.cache_predictions and toy_prediction_cache and graha_prediction_cache:
        comparison_started_at = time.perf_counter()
        comparison_caches = {
            "toy": toy_prediction_cache,
            "graha": graha_prediction_cache,
        }
        plot_prediction_cache_comparison(
            comparison_caches,
            output_dir / "plots" / "comparison",
            n_samples=min(5, config.prediction_n_samples),
        )
        _, metric_summary = evaluate_prediction_caches(
            comparison_caches,
            output_dir / "comparison_metrics",
        )
        print("Comparison metric summary:")
        for row in metric_summary:
            print("  " + json.dumps(row, sort_keys=True))
        record_timing(
            timing_rows,
            model="Comparison",
            stage="plots_and_metrics",
            started_at=comparison_started_at,
        )

    save_timing_summary(timing_rows, output_dir)


if __name__ == "__main__":
    main()
