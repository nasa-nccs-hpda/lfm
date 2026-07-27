"""Run semantic-segmentation checkpoint sweeps over the test split.

For each checkpoint, this script saves one folder per test sample containing:

- ``{sample_key}_input.npy``
- ``{sample_key}_label.npy``
- ``{sample_key}_pred.npy``
- ``{sample_key}_class_pred.npy``
- ``{sample_key}_logits.npy``
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
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from lightning.pytorch import seed_everything
from tqdm.auto import tqdm
from torch.utils.data import Subset

from lfm.all_models.all_tasks import (
    CheckpointRecord,
    CheckpointSweepExperiment,
    discover_checkpoints,
    load_lightning_checkpoint_state,
    write_checkpoint_metrics_summary,
)
from lfm.full_model.all_tasks.utils.utils import ensure_data_symlink
from lfm.full_model.sem_seg.semantic_model_adapter import GrahaSemanticModelAdapter
from lfm.toy_model.sem_seg.semantic_model_adapter import ToySemanticModelAdapter
from semantic_seg_comparison import (
    build_config as build_toy_config,
    get_toy_normalization_modality_info,
)

TOY_ADAPTER = ToySemanticModelAdapter()
GRAHA_ADAPTER = GrahaSemanticModelAdapter()

METRIC_NAMES = [
    "pixel_accuracy",
    "foreground_precision",
    "foreground_recall",
    "foreground_f1",
    "iou",
    "average_precision",
    "background_average_precision",
    "mean_average_precision",
    "predicted_foreground_fraction",
    "ground_truth_foreground_fraction",
]


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
    semantic_label_source: str
    image_glob: str
    label_glob: str
    image_suffix: str
    label_suffix: str
    batch_size: int
    num_workers: int
    normalize_inputs: bool
    normalization_source: str
    normalization_modality: str
    max_test_samples: int | None
    ignore_nodata_in_loss: bool
    nodata_ignore_index: int
    dino_checkpoint: Path | None
    graha_pretrain_dir: Path | None
    graha_input_modality_mode: str
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
    notebook_dir = script_dir.parents[2] / "notebooks" / "full_model"
    scripts_output_dir = script_dir.parents[2] / "scripts" / "outputs"
    data_root = (
        Path(args.data_root).resolve() if args.data_root else notebook_dir / "data"
    )
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
    dino_checkpoint = (
        Path(args.dino_checkpoint).resolve() if args.dino_checkpoint else None
    )
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
        spatial_transform="crop",
        semantic_label_source=getattr(args, "semantic_label_source", "semantic"),
        image_glob=args.image_glob,
        label_glob=args.label_glob,
        image_suffix=args.image_suffix,
        label_suffix=args.label_suffix,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize_inputs=args.normalize_inputs,
        normalization_source=getattr(args, "normalization_source", "pretrain"),
        normalization_modality=getattr(args, "normalization_modality", "vis_uv"),
        max_test_samples=args.max_test_samples,
        ignore_nodata_in_loss=getattr(args, "ignore_nodata_in_loss", False),
        nodata_ignore_index=getattr(args, "nodata_ignore_index", -1),
        dino_checkpoint=dino_checkpoint,
        graha_pretrain_dir=graha_pretrain_dir,
        graha_input_modality_mode=args.graha_input_modality_mode,
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


def _limit_dataset(
    dataset, max_samples: int | None, *, model_name: str, split_name: str
):
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
        return (
            images,
            labels,
            [str(item) if item is not None else None for item in image_paths],
        )
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def _move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, dict):
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
    if isinstance(batch, (tuple, list)):
        return tuple(
            value.to(device) if torch.is_tensor(value) else value for value in batch
        )
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


def _class_probabilities(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] > 1:
        return torch.softmax(logits, dim=1)
    foreground = torch.sigmoid(logits[:, 0])
    return torch.stack([1.0 - foreground, foreground], dim=1)


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    if labels.size == 0 or not np.any(labels):
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    tp = np.cumsum(labels, dtype=np.float64)
    fp = np.cumsum(~labels, dtype=np.float64)
    recall = tp / max(float(labels.sum()), 1.0)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    changed = np.where(recall[1:] != recall[:-1])[0]
    return float(
        np.sum((recall[changed + 1] - recall[changed]) * precision[changed + 1])
    )


def _ap_metrics_from_scores(
    foreground_scores: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    labels_bool = np.asarray(labels).astype(bool)
    foreground_ap = _average_precision(foreground_scores, labels_bool)
    background_ap = _average_precision(1.0 - foreground_scores, ~labels_bool)
    return {
        "average_precision": foreground_ap,
        "background_average_precision": background_ap,
        "mean_average_precision": float((foreground_ap + background_ap) / 2.0),
    }


def _valid_arrays(
    pred: np.ndarray,
    label: np.ndarray,
    *,
    ignore_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pred_flat = pred.reshape(-1)
    label_flat = label.reshape(-1)
    if ignore_index is not None:
        valid = label_flat != int(ignore_index)
        pred_flat = pred_flat[valid]
        label_flat = label_flat[valid]
    return pred_flat, label_flat


def _confusion_counts(
    pred: np.ndarray,
    label: np.ndarray,
    *,
    ignore_index: int | None = None,
) -> dict[str, float]:
    pred_flat, label_flat = _valid_arrays(
        pred,
        label,
        ignore_index=ignore_index,
    )
    pred_bool = pred_flat.astype(bool)
    label_bool = label_flat.astype(bool)
    return {
        "tp": float(np.sum(pred_bool & label_bool)),
        "fp": float(np.sum(pred_bool & ~label_bool)),
        "fn": float(np.sum(~pred_bool & label_bool)),
        "tn": float(np.sum(~pred_bool & ~label_bool)),
        "n": float(pred_bool.size),
        "pred_fg": float(np.sum(pred_bool)),
        "label_fg": float(np.sum(label_bool)),
    }


def _metrics_from_counts(
    counts: dict[str, float],
    ap_metrics: dict[str, float] | None = None,
) -> dict[str, float]:
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
    metrics = {
        "pixel_accuracy": float(accuracy),
        "foreground_precision": float(precision),
        "foreground_recall": float(recall),
        "foreground_f1": float(f1),
        "iou": float(iou),
        "average_precision": 0.0,
        "background_average_precision": 0.0,
        "mean_average_precision": 0.0,
        "predicted_foreground_fraction": float(counts["pred_fg"] / (n + eps)),
        "ground_truth_foreground_fraction": float(counts["label_fg"] / (n + eps)),
    }
    if ap_metrics:
        metrics.update(ap_metrics)
    return metrics


def _empty_counts() -> dict[str, float]:
    return {
        "tp": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "tn": 0.0,
        "n": 0.0,
        "pred_fg": 0.0,
        "label_fg": 0.0,
    }


def _add_counts(total: dict[str, float], part: dict[str, float]) -> None:
    for key, value in part.items():
        total[key] += value


def _metrics_to_array(metrics: dict[str, float]) -> np.ndarray:
    dtype = [(name, "f8") for name in METRIC_NAMES]
    row = np.zeros((), dtype=dtype)
    for name in METRIC_NAMES:
        row[name] = float(metrics[name])
    return row


def _write_metrics(
    output_dir: Path, metrics: dict[str, float], *, header: str | None = None
) -> None:
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
    ignore_index: int | None = None,
) -> dict[str, float]:
    load_lightning_checkpoint_state(task, checkpoint.path)
    device = next(task.parameters()).device
    was_training = task.training
    task.eval()

    checkpoint_output_dir = output_dir / checkpoint.name
    checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
    counts_total = _empty_counts()
    all_foreground_scores = []
    all_labels = []
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
            probs = _class_probabilities(logits)
            preds = _hard_predictions(logits)

            images_np = images.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            logits_np = logits.detach().cpu().numpy()
            foreground_scores_np = probs[:, 1].detach().cpu().numpy()
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
                np.save(sample_dir / f"{sample_key}_class_pred.npy", preds_np[i])
                np.save(sample_dir / f"{sample_key}_logits.npy", logits_np[i])

                sample_counts = _confusion_counts(
                    preds_np[i],
                    labels_np[i],
                    ignore_index=ignore_index,
                )
                _add_counts(counts_total, sample_counts)
                valid_scores, valid_labels = _valid_arrays(
                    foreground_scores_np[i],
                    labels_np[i],
                    ignore_index=ignore_index,
                )
                sample_ap_metrics = _ap_metrics_from_scores(
                    valid_scores,
                    valid_labels,
                )
                all_foreground_scores.append(valid_scores.reshape(-1))
                all_labels.append(valid_labels.reshape(-1))
                sample_metrics = _metrics_from_counts(sample_counts, sample_ap_metrics)
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

    aggregate_ap_metrics = (
        _ap_metrics_from_scores(
            np.concatenate(all_foreground_scores), np.concatenate(all_labels)
        )
        if all_foreground_scores
        else None
    )
    aggregate_metrics = _metrics_from_counts(counts_total, aggregate_ap_metrics)
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
        f"IoU={aggregate_metrics['iou']:.4f}, "
        f"AP={aggregate_metrics['average_precision']:.4f}, "
        f"mAP={aggregate_metrics['mean_average_precision']:.4f}, samples={sample_index}",
        flush=True,
    )
    return aggregate_metrics


def _make_toy_args(config: SweepConfig) -> argparse.Namespace:
    return SimpleNamespace(
        data_root=str(config.data_root),
        base_output_dir=str(config.output_root / "_toy_setup"),
        dino_checkpoint=str(config.dino_checkpoint) if config.dino_checkpoint else None,
        toy_lightning_checkpoint=None,
        band_filter=config.band_filter,
        target_size=config.target_size,
        spatial_transform="crop",
        semantic_label_source=config.semantic_label_source,
        image_glob=config.image_glob,
        label_glob=config.label_glob,
        image_suffix=config.image_suffix,
        label_suffix=config.label_suffix,
        max_train_samples=None,
        max_val_samples=None,
        max_test_samples=config.max_test_samples,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        max_epochs=1,
        learning_rate=5e-5,
        weight_decay=0.05,
        toy_loss_type="dice",
        use_toy_shape_loss=False,
        toy_shape_loss_weight=0.05,
        toy_shape_loss_pad_frac=0.3,
        freeze_encoder=False,
        normalize_inputs=config.normalize_inputs,
        normalization_source=config.normalization_source,
        normalization_modality=config.normalization_modality,
        toy_gradient_clip_val=1.0,
        disable_toy_gradient_clipping=True,
        plot_every_n_epochs=1,
        plot_n_samples=5,
        cache_predictions=False,
        prediction_split="test",
        prediction_n_samples=20,
        graha_base_output_dir=None,
        graha_pretrain_dir=(
            str(config.graha_pretrain_dir) if config.graha_pretrain_dir else None
        ),
        graha_input_modality_mode=config.graha_input_modality_mode,
        graha_vis_uv_merge_method=config.graha_vis_uv_merge_method,
        graha_lightning_checkpoint=None,
        graha_stats_batch_size=config.graha_stats_batch_size,
        graha_batch_size=config.graha_batch_size,
        graha_num_workers=config.graha_num_workers,
        seed=config.seed,
        no_fit=False,
        skip_toy_fit=False,
        skip_graha_fit=False,
        run_epoch_test_suite=False,
        epoch_test_split="test",
        epoch_test_n_samples=(
            config.max_test_samples if config.max_test_samples is not None else 10**9
        ),
        epoch_test_every_n_epochs=1,
        ignore_nodata_in_loss=config.ignore_nodata_in_loss,
        nodata_ignore_index=config.nodata_ignore_index,
    )


def run_toy_sweep(
    config: SweepConfig, checkpoints: list[CheckpointRecord] | None = None
) -> list[dict[str, Any]]:
    if checkpoints is None:
        if config.toy_checkpoint_dir is None:
            raise ValueError("Toy sweep requested but toy_checkpoint_dir is not set.")
        checkpoints = discover_checkpoints(
            config.toy_checkpoint_dir, max_checkpoints=config.max_checkpoints
        )
        print(f"[Toy] Found {len(checkpoints)} checkpoint(s).")
    toy_config = build_toy_config(_make_toy_args(config))
    setup_dir = config.output_root / "_toy_setup"
    with _quiet(not config.verbose):
        datamodule = TOY_ADAPTER.create_datamodule(
            toy_config,
            setup_dir,
            normalization_modality_info=get_toy_normalization_modality_info(toy_config),
        )
        datamodule.setup("test")
        task = TOY_ADAPTER.create_model_or_task(toy_config, datamodule)
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
            ignore_index=(
                config.nodata_ignore_index if config.ignore_nodata_in_loss else None
            ),
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

    write_checkpoint_metrics_summary(
        model_output_dir,
        rows,
        metric_names=METRIC_NAMES,
    )
    del task, datamodule
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def _make_graha_args(config: SweepConfig) -> argparse.Namespace:
    return SimpleNamespace(
        data_root=str(config.data_root),
        base_output_dir=str(config.output_root / "_graha_setup"),
        pretrain_dir=(
            str(config.graha_pretrain_dir) if config.graha_pretrain_dir else None
        ),
        lightning_checkpoint=None,
        graha_input_modality_mode=config.graha_input_modality_mode,
        graha_vis_uv_merge_method=config.graha_vis_uv_merge_method,
        normalization_source=config.normalization_source,
        normalization_modality=config.normalization_modality,
        band_filter=config.band_filter,
        semantic_label_source=config.semantic_label_source,
        image_glob=config.image_glob,
        label_glob=config.label_glob,
        image_suffix=config.image_suffix,
        label_suffix=config.label_suffix,
        shape_loss_weight=0.0,
        shape_loss_pad_frac=0.3,
        crop_size=config.target_size,
        stats_batch_size=config.graha_stats_batch_size,
        batch_size=config.graha_batch_size,
        num_workers=config.graha_num_workers,
        max_epochs=1,
        cache_predictions=False,
        prediction_split="test",
        prediction_n_samples=20,
        progress_log_every_n_batches=25,
        seed=config.seed,
        no_fit=True,
        ignore_nodata_in_loss=config.ignore_nodata_in_loss,
        nodata_ignore_index=config.nodata_ignore_index,
    )


def run_graha_sweep(
    config: SweepConfig, checkpoints: list[CheckpointRecord] | None = None
) -> list[dict[str, Any]]:
    if checkpoints is None:
        if config.graha_checkpoint_dir is None:
            raise ValueError(
                "Graha sweep requested but graha_checkpoint_dir is not set."
            )
        checkpoints = discover_checkpoints(
            config.graha_checkpoint_dir, max_checkpoints=config.max_checkpoints
        )
        print(f"[Graha] Found {len(checkpoints)} checkpoint(s).")
    with _quiet(not config.verbose):
        GRAHA_ADAPTER.configure_environment()
        graha_config = GRAHA_ADAPTER.build_config(_make_graha_args(config))
        GRAHA_ADAPTER.configure_python_paths(graha_config)
        GRAHA_ADAPTER.validate_required_paths(graha_config)

        deps = GRAHA_ADAPTER.import_project_dependencies()
        datamodule_cls = deps[
            (
                "LunarSemanticFromInstanceDatamodule"
                if config.semantic_label_source == "instance"
                else "LunarSemanticMaskSegmentationDatamodule"
            )
        ]
        task_cls = GRAHA_ADAPTER.make_task_class(deps["LunarShapeSegmentationTask"])
        means, stds = GRAHA_ADAPTER.get_normalization_stats(
            graha_config,
            datamodule_cls,
        )
        datamodule = GRAHA_ADAPTER.create_datamodule(
            graha_config, datamodule_cls, means, stds
        )
        datamodule.setup("test")
        datamodule.test_dataset = _limit_dataset(
            datamodule.test_dataset,
            config.max_test_samples,
            model_name="Graha",
            split_name="test",
        )

        sample_batch = GRAHA_ADAPTER.inspect_batch(datamodule)
        task = GRAHA_ADAPTER.create_task(graha_config, task_cls, sample_batch)
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
            ignore_index=(
                config.nodata_ignore_index if config.ignore_nodata_in_loss else None
            ),
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

    write_checkpoint_metrics_summary(
        model_output_dir,
        rows,
        metric_names=METRIC_NAMES,
    )
    del task, datamodule, sample_batch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def run_sweep(config: SweepConfig) -> dict[str, list[dict[str, Any]]]:
    def run_model_sweep(
        model: str, checkpoints: list[CheckpointRecord]
    ) -> list[dict[str, Any]]:
        if model == "toy":
            return run_toy_sweep(config, checkpoints)
        if model == "graha":
            return run_graha_sweep(config, checkpoints)
        raise ValueError(f"Unknown model name: {model}")

    return CheckpointSweepExperiment(
        output_root=config.output_root,
        models=config.models,
        checkpoint_dirs={
            "toy": config.toy_checkpoint_dir,
            "graha": config.graha_checkpoint_dir,
        },
        run_model_sweep=run_model_sweep,
        max_checkpoints=config.max_checkpoints,
        seed=config.seed,
        seed_fn=seed_everything,
    ).run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simlink-dest", "--symlink-dest", dest="simlink_dest", type=str, default=None
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--toy-checkpoint-dir", type=str, default=None)
    parser.add_argument("--graha-checkpoint-dir", type=str, default=None)
    parser.add_argument(
        "--models", nargs="+", default=["toy", "graha"], choices=["toy", "graha"]
    )
    parser.add_argument(
        "--band-filter", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6]
    )
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument(
        "--semantic-label-source",
        choices=["semantic", "instance"],
        default="semantic",
    )
    parser.add_argument(
        "--image-glob",
        default="*.tif",
        help="Chip filename glob inside each split/chips directory.",
    )
    parser.add_argument(
        "--label-glob",
        default="*_label.*",
        help="Label filename glob inside each split/labels directory.",
    )
    parser.add_argument(
        "--image-suffix",
        default="_input_wac_static_chip",
        help="Suffix stripped from chip stems before matching labels.",
    )
    parser.add_argument(
        "--label-suffix",
        default="_label",
        help="Suffix stripped from label stems before matching chips.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--normalize-inputs", action="store_true")
    parser.add_argument(
        "--normalization-source",
        choices=["pretrain", "finetune"],
        default="pretrain",
    )
    parser.add_argument(
        "--normalization-modality",
        choices=["vis_uv", "nac"],
        default="vis_uv",
    )
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument(
        "--ignore-nodata-in-loss",
        action="store_true",
        help="Ignore TIFF nodata pixels in semantic segmentation metrics.",
    )
    parser.add_argument(
        "--nodata-ignore-index",
        type=int,
        default=-1,
        help="Target label value used for ignored nodata pixels.",
    )
    parser.add_argument("--dino-checkpoint", type=str, default=None)
    parser.add_argument("--graha-pretrain-dir", type=str, default=None)
    parser.add_argument(
        "--graha-input-modality-mode", choices=["new-wac", "vis-uv"], default="new-wac"
    )
    parser.add_argument(
        "--graha-vis-uv-merge-method", choices=["mean", "max"], default="mean"
    )
    parser.add_argument("--graha-stats-batch-size", type=int, default=16)
    parser.add_argument("--graha-batch-size", type=int, default=16)
    parser.add_argument("--graha-num-workers", type=int, default=10)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--verbose", action="store_true", help="Show model/datamodule setup output."
    )
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
    notebook_dir = Path(__file__).resolve().parents[2] / "notebooks" / "full_model"
    ensure_data_symlink(args.simlink_dest, notebook_dir / "data")
    config = build_config(args)
    print(
        "REMINDER: after rerunning training, confirm checkpoint directory structure before large sweeps."
    )
    print("Output root:", config.output_root)
    print("Data root:", config.data_root)
    run_sweep(config)


if __name__ == "__main__":
    main()
