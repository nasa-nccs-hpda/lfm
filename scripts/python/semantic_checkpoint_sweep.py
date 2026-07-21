"""Run semantic-segmentation checkpoint sweeps over the test split.

For each checkpoint, this script saves one folder per test sample containing:

- ``{sample_key}_input.npy``
- ``{sample_key}_label.npy``
- ``{sample_key}_pred.npy``
- ``metrics.npy``
- ``metrics.txt``

Each checkpoint directory also receives aggregate ``metrics.npy`` and
``metrics.txt`` files. The same functions are intended for use from the
companion notebook and from sbatch.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from lightning.pytorch import seed_everything
from tqdm.auto import tqdm
from torch.utils.data import Subset

from lfm.full_model import lfm_seg_finetuning_direct as graha_workflow
from lfm.full_model.utils.utils import ensure_data_symlink
from toy_sem_seg_comparison import (
    build_config as build_toy_config,
    create_datamodule as create_toy_datamodule,
    create_lightning_module as create_toy_lightning_module,
    create_model as create_toy_model,
)


METRIC_NAMES = [
    "pixel_accuracy",
    "foreground_precision",
    "foreground_recall",
    "foreground_f1",
    "iou",
    "predicted_foreground_fraction",
    "ground_truth_foreground_fraction",
]


@dataclass(frozen=True)
class CheckpointRecord:
    path: Path
    epoch: int | None
    name: str


@dataclass(frozen=True)
class SweepConfig:
    notebook_dir: Path
    data_root: Path
    output_root: Path
    toy_checkpoint_dir: Path | None
    graha_checkpoint_dir: Path | None
    models: list[str]
    band_filter: list[int]
    target_size: int
    spatial_transform: str
    batch_size: int
    num_workers: int
    normalize_inputs: bool
    max_test_samples: int | None
    dino_checkpoint: Path | None
    graha_pretrain_dir: Path | None
    graha_wac_mode: str
    graha_vis_uv_merge_method: str
    graha_stats_batch_size: int
    graha_batch_size: int
    graha_num_workers: int
    max_checkpoints: int | None
    seed: int
    verbose: bool
    preload_test_batches: bool


def build_config(args: argparse.Namespace) -> SweepConfig:
    script_dir = Path(__file__).resolve().parent
    notebook_dir = script_dir.parents[1] / "notebooks" / "full_model"
    scripts_output_dir = script_dir.parents[1] / "scripts" / "outputs"
    data_root = Path(args.data_root).resolve() if args.data_root else notebook_dir / "data"
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else scripts_output_dir / "semantic_checkpoint_sweep"
    )
    toy_checkpoint_dir = (
        Path(args.toy_checkpoint_dir).resolve() if args.toy_checkpoint_dir else None
    )
    graha_checkpoint_dir = (
        Path(args.graha_checkpoint_dir).resolve() if args.graha_checkpoint_dir else None
    )
    dino_checkpoint = Path(args.dino_checkpoint).resolve() if args.dino_checkpoint else None
    graha_pretrain_dir = (
        Path(args.graha_pretrain_dir).resolve() if args.graha_pretrain_dir else None
    )

    models = [model.lower() for model in args.models]
    unknown = sorted(set(models) - {"toy", "graha"})
    if unknown:
        raise ValueError(f"Unknown model name(s): {unknown}")

    return SweepConfig(
        notebook_dir=notebook_dir,
        data_root=data_root,
        output_root=output_root,
        toy_checkpoint_dir=toy_checkpoint_dir,
        graha_checkpoint_dir=graha_checkpoint_dir,
        models=models,
        band_filter=args.band_filter,
        target_size=args.target_size,
        spatial_transform=args.spatial_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize_inputs=args.normalize_inputs,
        max_test_samples=args.max_test_samples,
        dino_checkpoint=dino_checkpoint,
        graha_pretrain_dir=graha_pretrain_dir,
        graha_wac_mode=args.graha_wac_mode,
        graha_vis_uv_merge_method=args.graha_vis_uv_merge_method,
        graha_stats_batch_size=args.graha_stats_batch_size,
        graha_batch_size=args.graha_batch_size,
        graha_num_workers=args.graha_num_workers,
        max_checkpoints=args.max_checkpoints,
        seed=args.seed,
        verbose=getattr(args, "verbose", False),
        preload_test_batches=getattr(args, "preload_test_batches", True),
    )


@contextlib.contextmanager
def _quiet(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


def _load_lightning_checkpoint_state(task: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint_path = Path(checkpoint_path).resolve()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*You are using `torch.load` with `weights_only=False`.*",
            category=FutureWarning,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    task.load_state_dict(state_dict, strict=True)


def _limit_dataset(dataset, max_samples: int | None, *, model_name: str, split_name: str):
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


def discover_checkpoints(checkpoint_dir: Path, *, max_checkpoints: int | None = None) -> list[CheckpointRecord]:
    checkpoint_dir = Path(checkpoint_dir).resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    paths = sorted(path for path in checkpoint_dir.rglob("*.ckpt") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No .ckpt files found under {checkpoint_dir}")

    records = [
        CheckpointRecord(
            path=path,
            epoch=_parse_epoch(path),
            name=_checkpoint_output_name(path, _parse_epoch(path)),
        )
        for path in paths
    ]
    records.sort(key=lambda item: (item.epoch is None, item.epoch if item.epoch is not None else 10**9, str(item.path)))

    unique_records = []
    used_names: dict[str, int] = {}
    for record in records:
        count = used_names.get(record.name, 0)
        used_names[record.name] = count + 1
        if count:
            record = CheckpointRecord(
                path=record.path,
                epoch=record.epoch,
                name=f"{record.name}_{count + 1}",
            )
        unique_records.append(record)

    if max_checkpoints is not None:
        unique_records = unique_records[:max_checkpoints]
    return unique_records


def _parse_epoch(path: Path) -> int | None:
    text = str(path)
    for pattern in [r"epoch[=_-](\d+)", r"model-(\d+)-", r"epoch_(\d+)"]:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def _checkpoint_output_name(path: Path, epoch: int | None) -> str:
    if epoch is not None:
        return f"epoch_{epoch:03d}"
    stem = path.stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")
    return stem or "checkpoint"


def _sample_key_from_path(path: str | Path | None, fallback_index: int) -> str:
    if path is None:
        return f"sample_{fallback_index:04d}"
    stem = Path(str(path)).stem
    return stem.split("_input", 1)[0]


def _extract_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor, list[str | None]]:
    if isinstance(batch, dict):
        images = batch["image"]
        labels = batch["mask"]
        filenames = batch.get("filename")
        if filenames is None:
            image_paths = [None] * images.shape[0]
        elif isinstance(filenames, (str, Path)):
            image_paths = [str(filenames)]
        else:
            image_paths = [str(item) for item in filenames]
        return images, labels, image_paths
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        images = batch[0]
        labels = batch[1]
        image_paths = batch[2] if len(batch) > 2 else [None] * images.shape[0]
        return images, labels, [str(item) if item is not None else None for item in image_paths]
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def _move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, dict):
        return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    if isinstance(batch, (tuple, list)):
        return tuple(value.to(device) if torch.is_tensor(value) else value for value in batch)
    return batch.to(device) if torch.is_tensor(batch) else batch


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


def _logits_from_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if hasattr(output, "output"):
        return output.output
    raise TypeError(f"Unsupported model output type: {type(output)}")


def _hard_predictions(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] > 1:
        return logits.argmax(dim=1).long()
    return (torch.sigmoid(logits[:, 0]) > 0.5).long()


def _confusion_counts(pred: np.ndarray, label: np.ndarray) -> dict[str, float]:
    pred_bool = pred.astype(bool).reshape(-1)
    label_bool = label.astype(bool).reshape(-1)
    return {
        "tp": float(np.sum(pred_bool & label_bool)),
        "fp": float(np.sum(pred_bool & ~label_bool)),
        "fn": float(np.sum(~pred_bool & label_bool)),
        "tn": float(np.sum(~pred_bool & ~label_bool)),
        "n": float(pred_bool.size),
        "pred_fg": float(np.sum(pred_bool)),
        "label_fg": float(np.sum(label_bool)),
    }


def _metrics_from_counts(counts: dict[str, float]) -> dict[str, float]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    n = counts["n"]
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    return {
        "pixel_accuracy": float(accuracy),
        "foreground_precision": float(precision),
        "foreground_recall": float(recall),
        "foreground_f1": float(f1),
        "iou": float(iou),
        "predicted_foreground_fraction": float(counts["pred_fg"] / (n + eps)),
        "ground_truth_foreground_fraction": float(counts["label_fg"] / (n + eps)),
    }


def _empty_counts() -> dict[str, float]:
    return {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0, "n": 0.0, "pred_fg": 0.0, "label_fg": 0.0}


def _add_counts(total: dict[str, float], part: dict[str, float]) -> None:
    for key, value in part.items():
        total[key] += value


def _metrics_to_array(metrics: dict[str, float]) -> np.ndarray:
    dtype = [(name, "f8") for name in METRIC_NAMES]
    row = np.zeros((), dtype=dtype)
    for name in METRIC_NAMES:
        row[name] = float(metrics[name])
    return row


def _write_metrics(output_dir: Path, metrics: dict[str, float], *, header: str | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "metrics.npy", _metrics_to_array(metrics))
    with (output_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        if header:
            f.write(header.rstrip() + "\n")
        for name in METRIC_NAMES:
            f.write(f"{name}: {metrics[name]:.8f}\n")


def _run_checkpoint(
    *,
    task: torch.nn.Module,
    test_batches,
    checkpoint: CheckpointRecord,
    output_dir: Path,
    model_name: str,
) -> dict[str, float]:
    _load_lightning_checkpoint_state(task, checkpoint.path)
    device = next(task.parameters()).device
    was_training = task.training
    task.eval()

    checkpoint_output_dir = output_dir / checkpoint.name
    checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
    counts_total = _empty_counts()
    sample_index = 0

    batch_bar = tqdm(
        test_batches,
        desc=f"{model_name} {checkpoint.name} batches",
        leave=False,
        dynamic_ncols=True,
    )

    with torch.no_grad():
        for batch in batch_bar:
            batch = _move_batch_to_device(batch, device)
            images, labels, image_paths = _extract_batch(batch)
            logits = _logits_from_output(task(images))
            preds = _hard_predictions(logits)

            images_np = images.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            preds_np = preds.detach().cpu().numpy()

            for i in range(images_np.shape[0]):
                sample_key = _sample_key_from_path(
                    image_paths[i] if i < len(image_paths) else None,
                    sample_index,
                )
                sample_dir = checkpoint_output_dir / sample_key
                sample_dir.mkdir(parents=True, exist_ok=True)

                np.save(sample_dir / f"{sample_key}_input.npy", images_np[i])
                np.save(sample_dir / f"{sample_key}_label.npy", labels_np[i])
                np.save(sample_dir / f"{sample_key}_pred.npy", preds_np[i])

                sample_counts = _confusion_counts(preds_np[i], labels_np[i])
                _add_counts(counts_total, sample_counts)
                sample_metrics = _metrics_from_counts(sample_counts)
                _write_metrics(
                    sample_dir,
                    sample_metrics,
                    header=(
                        f"model: {model_name}\n"
                        f"checkpoint: {checkpoint.path}\n"
                        f"sample_key: {sample_key}"
                    ),
                )
                sample_index += 1
            batch_bar.set_postfix(samples=sample_index)

    aggregate_metrics = _metrics_from_counts(counts_total)
    _write_metrics(
        checkpoint_output_dir,
        aggregate_metrics,
        header=(
            f"model: {model_name}\n"
            f"checkpoint: {checkpoint.path}\n"
            f"epoch: {checkpoint.epoch}\n"
            f"samples: {sample_index}"
        ),
    )
    task.train(was_training)
    print(
        f"[{model_name}] {checkpoint.name}: "
        f"F1={aggregate_metrics['foreground_f1']:.4f}, "
        f"IoU={aggregate_metrics['iou']:.4f}, samples={sample_index}",
        flush=True,
    )
    return aggregate_metrics


def _write_model_summary(model_output_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    model_output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = model_output_dir / "checkpoint_metrics_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        header = ["checkpoint_name", "epoch", "checkpoint_path", *METRIC_NAMES]
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write(
                "\t".join(
                    [
                        str(row["checkpoint_name"]),
                        "" if row["epoch"] is None else str(row["epoch"]),
                        str(row["checkpoint_path"]),
                        *[f"{row[name]:.8f}" for name in METRIC_NAMES],
                    ]
                )
                + "\n"
            )

    dtype = [
        ("checkpoint_name", "U128"),
        ("epoch", "i8"),
        ("checkpoint_path", "U1024"),
        *[(name, "f8") for name in METRIC_NAMES],
    ]
    arr = np.zeros(len(rows), dtype=dtype)
    for i, row in enumerate(rows):
        arr[i]["checkpoint_name"] = row["checkpoint_name"]
        arr[i]["epoch"] = -1 if row["epoch"] is None else int(row["epoch"])
        arr[i]["checkpoint_path"] = str(row["checkpoint_path"])
        for name in METRIC_NAMES:
            arr[i][name] = row[name]
    np.save(model_output_dir / "checkpoint_metrics_summary.npy", arr)
    print(f"Saved model summary to {summary_path}", flush=True)


def _make_toy_args(config: SweepConfig) -> argparse.Namespace:
    return SimpleNamespace(
        data_root=str(config.data_root),
        base_output_dir=str(config.output_root / "_toy_setup"),
        dino_checkpoint=str(config.dino_checkpoint) if config.dino_checkpoint else None,
        dino_lightning_checkpoint=None,
        band_filter=config.band_filter,
        target_size=config.target_size,
        spatial_transform=config.spatial_transform,
        max_train_samples=None,
        max_val_samples=None,
        max_test_samples=config.max_test_samples,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        max_epochs=1,
        learning_rate=5e-5,
        weight_decay=0.05,
        loss_type="dice",
        freeze_encoder=False,
        normalize_inputs=config.normalize_inputs,
        toy_gradient_clip_val=1.0,
        disable_toy_gradient_clipping=True,
        plot_every_n_epochs=1,
        plot_n_samples=5,
        cache_predictions=False,
        prediction_split="test",
        prediction_n_samples=20,
        graha_base_output_dir=None,
        graha_pretrain_dir=str(config.graha_pretrain_dir) if config.graha_pretrain_dir else None,
        graha_lightning_checkpoint=None,
        graha_stats_batch_size=config.graha_stats_batch_size,
        graha_batch_size=config.graha_batch_size,
        graha_num_workers=config.graha_num_workers,
        seed=config.seed,
        no_fit=False,
        skip_dino_fit=False,
        skip_graha_fit=False,
    )


def run_toy_sweep(config: SweepConfig) -> list[dict[str, Any]]:
    if config.toy_checkpoint_dir is None:
        raise ValueError("Toy sweep requested but toy_checkpoint_dir is not set.")

    checkpoints = discover_checkpoints(config.toy_checkpoint_dir, max_checkpoints=config.max_checkpoints)
    print(f"[Toy] Found {len(checkpoints)} checkpoint(s).")

    toy_config = build_toy_config(_make_toy_args(config))
    setup_dir = config.output_root / "_toy_setup"
    with _quiet(not config.verbose):
        datamodule = create_toy_datamodule(toy_config, setup_dir)
        datamodule.setup("test")
        if datamodule.weight_assignments is None:
            raise RuntimeError("Toy datamodule did not create weight assignments.")

        model = create_toy_model(toy_config, datamodule.weight_assignments)
        task = create_toy_lightning_module(toy_config, model)
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
    )
    for checkpoint in checkpoint_bar:
        checkpoint_bar.set_postfix(checkpoint=checkpoint.name)
        metrics = _run_checkpoint(
            task=task,
            test_batches=test_batches,
            checkpoint=checkpoint,
            output_dir=model_output_dir,
            model_name="Toy",
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

    _write_model_summary(model_output_dir, rows)
    del task, model, datamodule
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def _make_graha_args(config: SweepConfig) -> argparse.Namespace:
    return SimpleNamespace(
        data_root=str(config.data_root),
        base_output_dir=str(config.output_root / "_graha_setup"),
        pretrain_dir=str(config.graha_pretrain_dir) if config.graha_pretrain_dir else None,
        lightning_checkpoint=None,
        graha_wac_mode=config.graha_wac_mode,
        graha_vis_uv_merge_method=config.graha_vis_uv_merge_method,
        crop_size=config.target_size,
        stats_batch_size=config.graha_stats_batch_size,
        batch_size=config.graha_batch_size,
        num_workers=config.graha_num_workers,
        max_epochs=1,
        cache_predictions=False,
        prediction_split="test",
        prediction_n_samples=20,
        seed=config.seed,
        no_fit=True,
    )


def run_graha_sweep(config: SweepConfig) -> list[dict[str, Any]]:
    if config.graha_checkpoint_dir is None:
        raise ValueError("Graha sweep requested but graha_checkpoint_dir is not set.")

    checkpoints = discover_checkpoints(config.graha_checkpoint_dir, max_checkpoints=config.max_checkpoints)
    print(f"[Graha] Found {len(checkpoints)} checkpoint(s).")

    with _quiet(not config.verbose):
        graha_workflow.configure_proj_environment()
        graha_config = graha_workflow.build_config(_make_graha_args(config))
        graha_workflow.configure_python_paths(graha_config)
        graha_workflow.validate_required_paths(graha_config)

        deps = graha_workflow.import_project_dependencies()
        datamodule_cls = deps["LunarSemanticSegmentationDatamodule"]
        task_cls = graha_workflow.make_notebook_task_class(deps["LunarShapeSegmentationTask"])
        means, stds = graha_workflow.calculate_train_stats(graha_config, datamodule_cls)
        datamodule = graha_workflow.create_datamodule(graha_config, datamodule_cls, means, stds)
        datamodule.setup("test")
        datamodule.test_dataset = _limit_dataset(
            datamodule.test_dataset,
            config.max_test_samples,
            model_name="Graha",
            split_name="test",
        )

        sample_batch = graha_workflow.inspect_batch(datamodule)
        task = graha_workflow.create_task(graha_config, task_cls, sample_batch)
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
    )
    for checkpoint in checkpoint_bar:
        checkpoint_bar.set_postfix(checkpoint=checkpoint.name)
        metrics = _run_checkpoint(
            task=task,
            test_batches=test_batches,
            checkpoint=checkpoint,
            output_dir=model_output_dir,
            model_name="Graha",
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

    _write_model_summary(model_output_dir, rows)
    del task, datamodule, sample_batch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def run_sweep(config: SweepConfig) -> dict[str, list[dict[str, Any]]]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)

    results: dict[str, list[dict[str, Any]]] = {}
    if "toy" in config.models:
        results["toy"] = run_toy_sweep(config)
    if "graha" in config.models:
        results["graha"] = run_graha_sweep(config)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simlink-dest", "--symlink-dest", dest="simlink_dest", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--toy-checkpoint-dir", type=str, default=None)
    parser.add_argument("--graha-checkpoint-dir", type=str, default=None)
    parser.add_argument("--models", nargs="+", default=["toy", "graha"], choices=["toy", "graha"])
    parser.add_argument("--band-filter", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--spatial-transform", choices=["crop", "resize"], default="crop")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--normalize-inputs", action="store_true")
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--dino-checkpoint", type=str, default=None)
    parser.add_argument("--graha-pretrain-dir", type=str, default=None)
    parser.add_argument("--graha-wac-mode", choices=["new-wac", "vis-uv"], default="new-wac")
    parser.add_argument("--graha-vis-uv-merge-method", choices=["mean", "max"], default="mean")
    parser.add_argument("--graha-stats-batch-size", type=int, default=16)
    parser.add_argument("--graha-batch-size", type=int, default=16)
    parser.add_argument("--graha-num-workers", type=int, default=10)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true", help="Show model/datamodule setup output.")
    parser.add_argument(
        "--no-preload-test-batches",
        dest="preload_test_batches",
        action="store_false",
        help="Disable one-time test dataloader preload and iterate the dataloader for every checkpoint.",
    )
    parser.set_defaults(preload_test_batches=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notebook_dir = Path(__file__).resolve().parents[1] / "notebooks" / "full_model"
    ensure_data_symlink(args.simlink_dest, notebook_dir / "data")
    config = build_config(args)
    print("REMINDER: after rerunning training, confirm checkpoint directory structure before large sweeps.")
    print("Output root:", config.output_root)
    print("Data root:", config.data_root)
    run_sweep(config)


if __name__ == "__main__":
    main()
