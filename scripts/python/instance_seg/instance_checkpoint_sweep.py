"""Run true instance-segmentation checkpoint sweeps over a split.

For each checkpoint, this script saves one folder per sample containing:

- ``{sample_key}_input.npy``
- ``{sample_key}_label.npy``
- ``{sample_key}_pred.npy``
- ``{sample_key}_pred_classes.npy``
- ``{sample_key}_pred_logits.npy``
- ``{sample_key}_gt_boxes.npy``
- ``{sample_key}_pred_boxes.npy``
- ``{sample_key}_pred_scores.npy``
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
import json
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning.pytorch import seed_everything
from tqdm.auto import tqdm

import instance_seg_comparison as comparison_workflow
from lfm.full_model.inst_seg import instance_seg_finetuning as graha_workflow
from lfm.full_model.all_tasks.utils.plot_utils import (
    _instance_metrics,
    _load_instance_prediction_cache,
)
from lfm.full_model.all_tasks.utils.utils import ensure_data_symlink
from lfm.full_model.all_tasks.utils import (
    save_graha_instance_prediction_cache,
    save_toy_instance_prediction_cache,
)


METRIC_NAMES = [
    "semantic_f1",
    "instance_precision",
    "instance_recall",
    "instance_f1",
    "instance_mean_iou",
    "average_precision",
    "average_precision_50",
    "mean_average_precision",
    "num_pred",
    "num_gt",
]

AP_IOU_THRESHOLDS = np.round(np.arange(0.5, 1.0, 0.05), 2)


@dataclass(frozen=True)
class CheckpointRecord:
    path: Path
    epoch: int | None
    name: str


@dataclass(frozen=True)
class InstanceSweepConfig:
    notebook_dir: Path
    data_root: Path
    output_root: Path
    toy_checkpoint_dir: Path | None
    graha_checkpoint_dir: Path | None
    models: list[str]
    target_size: int
    band_filter: list[int]
    max_samples: int | None
    toy_batch_size: int
    toy_num_workers: int
    toy_normalize_inputs: bool
    toy_architecture: str
    dino_checkpoint: Path | None
    graha_pretrain_dir: Path | None
    graha_wac_mode: str
    graha_vis_uv_merge_method: str
    graha_stats_batch_size: int
    graha_batch_size: int
    graha_num_workers: int
    graha_backbone_lr: float
    graha_head_lr: float
    graha_layer_decay: float
    graha_weight_decay: float
    graha_warmup_steps: int
    graha_anchor_sizes: list[list[int]]
    graha_anchor_aspect_ratios: list[float]
    graha_score_threshold: float
    prediction_split: str
    prediction_score_threshold: float
    mask_shift: tuple[int, int]
    max_checkpoints: int | None
    seed: int
    verbose: bool


def build_config(args: argparse.Namespace) -> InstanceSweepConfig:
    script_dir = Path(__file__).resolve().parent
    lfm_root = script_dir.parents[2]
    notebook_dir = lfm_root / "notebooks" / "full_model"
    scripts_output_dir = lfm_root / "scripts" / "outputs"
    models = [model.lower() for model in args.models]
    unknown = sorted(set(models) - {"toy", "graha"})
    if unknown:
        raise ValueError(f"Unknown model name(s): {unknown}")

    return InstanceSweepConfig(
        notebook_dir=notebook_dir,
        data_root=Path(args.data_root).resolve() if args.data_root else notebook_dir / "data",
        output_root=(
            Path(args.output_root).resolve()
            if args.output_root
            else scripts_output_dir / "instance_checkpoint_sweep"
        ),
        toy_checkpoint_dir=(
            Path(args.toy_checkpoint_dir).resolve() if args.toy_checkpoint_dir else None
        ),
        graha_checkpoint_dir=(
            Path(args.graha_checkpoint_dir).resolve() if args.graha_checkpoint_dir else None
        ),
        models=models,
        target_size=args.target_size,
        band_filter=args.band_filter,
        max_samples=args.max_samples,
        toy_batch_size=args.toy_batch_size,
        toy_num_workers=args.toy_num_workers,
        toy_normalize_inputs=args.toy_normalize_inputs,
        toy_architecture=args.toy_architecture,
        dino_checkpoint=Path(args.dino_checkpoint).resolve() if args.dino_checkpoint else None,
        graha_pretrain_dir=Path(args.graha_pretrain_dir).resolve()
        if args.graha_pretrain_dir
        else None,
        graha_wac_mode=args.graha_wac_mode,
        graha_vis_uv_merge_method=args.graha_vis_uv_merge_method,
        graha_stats_batch_size=args.graha_stats_batch_size,
        graha_batch_size=args.graha_batch_size,
        graha_num_workers=args.graha_num_workers,
        graha_backbone_lr=args.graha_backbone_lr,
        graha_head_lr=args.graha_head_lr,
        graha_layer_decay=args.graha_layer_decay,
        graha_weight_decay=args.graha_weight_decay,
        graha_warmup_steps=args.graha_warmup_steps,
        graha_anchor_sizes=args.graha_anchor_sizes,
        graha_anchor_aspect_ratios=args.graha_anchor_aspect_ratios,
        graha_score_threshold=args.graha_score_threshold,
        prediction_split=args.prediction_split,
        prediction_score_threshold=args.prediction_score_threshold,
        mask_shift=tuple(args.mask_shift),
        max_checkpoints=args.max_checkpoints,
        seed=args.seed,
        verbose=args.verbose,
    )


@contextlib.contextmanager
def _quiet(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


def discover_checkpoints(
    checkpoint_dir: Path,
    *,
    max_checkpoints: int | None = None,
) -> list[CheckpointRecord]:
    checkpoint_dir = Path(checkpoint_dir).resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    paths = sorted(path for path in checkpoint_dir.rglob("*.ckpt") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No .ckpt files found under {checkpoint_dir}")
    records = [
        CheckpointRecord(path=path, epoch=_parse_epoch(path), name=_checkpoint_output_name(path, _parse_epoch(path)))
        for path in paths
    ]
    records.sort(key=lambda item: (item.epoch is None, item.epoch if item.epoch is not None else 10**9, str(item.path)))
    if max_checkpoints is not None:
        records = records[:max_checkpoints]
    return records


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
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_")
    return stem or "checkpoint"


def _load_lightning_checkpoint_state(module: torch.nn.Module, checkpoint_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*You are using `torch.load` with `weights_only=False`.*",
            category=FutureWarning,
        )
        checkpoint = torch.load(Path(checkpoint_path).resolve(), map_location="cpu", weights_only=False)
    module.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)


def _semantic_f1(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = (pred_mask > 0).reshape(-1)
    gt = (gt_mask > 0).reshape(-1)
    tp = float(np.sum(pred & gt))
    fp = float(np.sum(pred & ~gt))
    fn = float(np.sum(~pred & gt))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return float(2 * precision * recall / (precision + recall + 1e-8))


def _score_logits(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    eps = np.finfo(np.float32).eps
    clipped = np.clip(scores, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32)


def _normalize_prediction_scores(sample: dict[str, Any]) -> np.ndarray:
    pred_mask = np.asarray(sample["pred_mask"])
    num_instances = len([value for value in np.unique(pred_mask) if int(value) != 0])
    scores = np.asarray(sample["pred_scores"], dtype=np.float32).reshape(-1)
    if scores.shape[0] < num_instances:
        scores = np.pad(scores, (0, num_instances - scores.shape[0]), constant_values=1.0)
    return scores


def _threshold_instance_sample(sample: dict[str, Any], score_threshold: float) -> dict[str, Any]:
    scores = _normalize_prediction_scores(sample)
    keep = scores >= float(score_threshold)
    pred_mask = np.asarray(sample["pred_mask"])
    thresholded_mask = np.zeros_like(pred_mask, dtype=np.int32)
    kept_scores = scores[keep]
    kept_index = 1
    for pred_index, should_keep in enumerate(keep, start=1):
        if should_keep:
            thresholded_mask[pred_mask == pred_index] = kept_index
            kept_index += 1

    pred_boxes = np.asarray(sample["pred_boxes"], dtype=np.float32)
    if pred_boxes.shape[0] < scores.shape[0]:
        padding = np.zeros((scores.shape[0] - pred_boxes.shape[0], 4), dtype=np.float32)
        pred_boxes = np.concatenate([pred_boxes, padding], axis=0)

    result = dict(sample)
    result["pred_mask"] = thresholded_mask
    result["pred_boxes"] = pred_boxes[keep].astype(np.float32)
    result["pred_scores"] = kept_scores.astype(np.float32)
    return result


def _instance_binary_masks(instance_mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(instance_mask)
    return [(mask == value) for value in np.unique(mask) if int(value) != 0]


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    return intersection / union if union > 0 else 0.0


def _average_precision_from_detections(
    detections: list[tuple[float, bool]],
    num_gt: int,
) -> float:
    if num_gt <= 0:
        return 0.0
    if not detections:
        return 0.0
    detections = sorted(detections, key=lambda item: item[0], reverse=True)
    true_positive = np.asarray([item[1] for item in detections], dtype=bool)
    tp = np.cumsum(true_positive, dtype=np.float64)
    fp = np.cumsum(~true_positive, dtype=np.float64)
    recall = tp / float(num_gt)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    changed = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changed + 1] - recall[changed]) * precision[changed + 1]))


def _instance_ap_at_threshold(samples: list[dict[str, Any]], iou_threshold: float) -> float:
    detections: list[tuple[float, bool]] = []
    total_gt = 0
    for sample in samples:
        gt_masks = _instance_binary_masks(sample["gt_mask"])
        pred_masks = _instance_binary_masks(sample["pred_mask"])
        pred_scores = _normalize_prediction_scores(sample)
        total_gt += len(gt_masks)
        matched_gt: set[int] = set()
        order = np.argsort(-pred_scores, kind="mergesort") if pred_scores.size else np.asarray([], dtype=np.int64)
        for pred_index in order:
            if pred_index >= len(pred_masks):
                detections.append((float(pred_scores[pred_index]), False))
                continue
            best_iou = 0.0
            best_gt_index = None
            for gt_index, gt_mask in enumerate(gt_masks):
                if gt_index in matched_gt:
                    continue
                iou = _mask_iou(pred_masks[pred_index], gt_mask)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = gt_index
            is_tp = best_gt_index is not None and best_iou >= iou_threshold
            if is_tp:
                matched_gt.add(int(best_gt_index))
            detections.append((float(pred_scores[pred_index]), bool(is_tp)))
    return _average_precision_from_detections(detections, total_gt)


def _instance_ap_metrics(samples: list[dict[str, Any]]) -> dict[str, float]:
    if not samples:
        return {
            "average_precision": 0.0,
            "average_precision_50": 0.0,
            "mean_average_precision": 0.0,
        }
    ap_by_threshold = {
        float(threshold): _instance_ap_at_threshold(samples, float(threshold))
        for threshold in AP_IOU_THRESHOLDS
    }
    return {
        "average_precision": float(ap_by_threshold[0.5]),
        "average_precision_50": float(ap_by_threshold[0.5]),
        "mean_average_precision": float(np.mean(list(ap_by_threshold.values()))),
    }


def _sample_metrics(sample: dict[str, Any]) -> dict[str, float]:
    inst = _instance_metrics(sample["pred_mask"], sample["gt_mask"])
    ap_metrics = _instance_ap_metrics([sample])
    return {
        "semantic_f1": _semantic_f1(sample["pred_mask"], sample["gt_mask"]),
        "instance_precision": float(inst["precision"]),
        "instance_recall": float(inst["recall"]),
        "instance_f1": float(inst["f1"]),
        "instance_mean_iou": float(inst["mean_iou"]),
        **ap_metrics,
        "num_pred": float(inst["num_pred"]),
        "num_gt": float(inst["num_gt"]),
    }


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


def _aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {name: 0.0 for name in METRIC_NAMES}
    return {name: float(np.mean([row[name] for row in rows])) for name in METRIC_NAMES}


def _write_checkpoint_outputs(
    *,
    cache_dir: Path,
    checkpoint_output_dir: Path,
    checkpoint: CheckpointRecord,
    model_name: str,
    score_threshold: float,
) -> dict[str, float]:
    samples = _load_instance_prediction_cache(cache_dir)
    sample_metrics = []
    raw_sample_values = []
    for sample_key, sample in tqdm(
        sorted(samples.items()),
        desc=f"{model_name} {checkpoint.name} samples",
        leave=False,
        dynamic_ncols=True,
    ):
        display_key = str(sample_key).split("_input", 1)[0]
        thresholded_sample = _threshold_instance_sample(sample, score_threshold)
        sample_dir = checkpoint_output_dir / display_key
        sample_dir.mkdir(parents=True, exist_ok=True)

        np.save(sample_dir / f"{display_key}_input.npy", thresholded_sample["image"])
        np.save(sample_dir / f"{display_key}_label.npy", thresholded_sample["gt_mask"])
        np.save(sample_dir / f"{display_key}_pred.npy", thresholded_sample["pred_mask"])
        np.save(
            sample_dir / f"{display_key}_pred_classes.npy",
            np.ones((thresholded_sample["pred_scores"].shape[0],), dtype=np.int64),
        )
        np.save(sample_dir / f"{display_key}_pred_logits.npy", _score_logits(thresholded_sample["pred_scores"]))
        np.save(sample_dir / f"{display_key}_gt_boxes.npy", thresholded_sample["gt_boxes"])
        np.save(sample_dir / f"{display_key}_pred_boxes.npy", thresholded_sample["pred_boxes"])
        np.save(sample_dir / f"{display_key}_pred_scores.npy", thresholded_sample["pred_scores"])

        metrics = _sample_metrics(thresholded_sample)
        metrics.update(_instance_ap_metrics([sample]))
        sample_metrics.append(metrics)
        raw_sample_values.append(sample)
        _write_metrics(
            sample_dir,
            metrics,
            header=(
                f"model: {model_name}\n"
                f"checkpoint: {checkpoint.path}\n"
                f"sample_key: {display_key}\n"
                f"prediction_score_threshold: {score_threshold}"
            ),
        )

    aggregate = _aggregate_metrics(sample_metrics)
    aggregate.update(_instance_ap_metrics(raw_sample_values))
    _write_metrics(
        checkpoint_output_dir,
        aggregate,
        header=(
            f"model: {model_name}\n"
            f"checkpoint: {checkpoint.path}\n"
            f"epoch: {checkpoint.epoch}\n"
            f"samples: {len(sample_metrics)}\n"
            f"prediction_score_threshold: {score_threshold}"
        ),
    )
    print(
        f"[{model_name}] {checkpoint.name}: "
        f"InstF1={aggregate['instance_f1']:.4f}, "
        f"SemF1={aggregate['semantic_f1']:.4f}, "
        f"AP50={aggregate['average_precision_50']:.4f}, "
        f"mAP={aggregate['mean_average_precision']:.4f}, samples={len(sample_metrics)}",
        flush=True,
    )
    return aggregate


def _write_model_summary(model_output_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    model_output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = model_output_dir / "checkpoint_metrics_summary.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        header = ["checkpoint_name", "epoch", "checkpoint_path", *METRIC_NAMES]
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write(
                "\t".join(
                    [
                        row["checkpoint_name"],
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
    print(f"Saved model summary to {txt_path}", flush=True)


def _prediction_count(config: InstanceSweepConfig) -> int:
    return config.max_samples if config.max_samples is not None else 10**9


def _make_comparison_args(config: InstanceSweepConfig) -> argparse.Namespace:
    return argparse.Namespace(
        data_root=str(config.data_root),
        base_output_dir=str(config.output_root / "_setup"),
        dino_checkpoint=str(config.dino_checkpoint) if config.dino_checkpoint else None,
        dino_lightning_checkpoint=None,
        graha_pretrain_dir=str(config.graha_pretrain_dir) if config.graha_pretrain_dir else None,
        graha_lightning_checkpoint=None,
        graha_wac_mode=config.graha_wac_mode,
        graha_vis_uv_merge_method=config.graha_vis_uv_merge_method,
        target_size=config.target_size,
        band_filter=config.band_filter,
        max_train_samples=None,
        max_val_samples=None,
        max_test_samples=config.max_samples,
        toy_batch_size=config.toy_batch_size,
        toy_num_workers=config.toy_num_workers,
        graha_stats_batch_size=config.graha_stats_batch_size,
        graha_batch_size=config.graha_batch_size,
        graha_num_workers=config.graha_num_workers,
        max_epochs=1,
        toy_learning_rate=5.0e-5,
        toy_weight_decay=1.0e-3,
        toy_freeze_backbone=False,
        toy_normalize_inputs=config.toy_normalize_inputs,
        toy_architecture=config.toy_architecture,
        toy_gradient_clip_val=1.0,
        disable_toy_gradient_clipping=True,
        graha_backbone_lr=config.graha_backbone_lr,
        graha_head_lr=config.graha_head_lr,
        graha_layer_decay=config.graha_layer_decay,
        graha_weight_decay=config.graha_weight_decay,
        graha_warmup_steps=config.graha_warmup_steps,
        graha_anchor_sizes=config.graha_anchor_sizes,
        graha_anchor_aspect_ratios=config.graha_anchor_aspect_ratios,
        graha_score_threshold=config.graha_score_threshold,
        plot_every_n_epochs=0,
        plot_n_samples=5,
        progress_log_every_n_batches=10**9,
        prediction_split=config.prediction_split,
        prediction_n_samples=_prediction_count(config),
        prediction_score_threshold=config.prediction_score_threshold,
        mask_shift=config.mask_shift,
        skip_toy_fit=True,
        skip_graha_fit=True,
        no_fit=True,
        run_epoch_test_suite=False,
        epoch_test_split=config.prediction_split,
        epoch_test_n_samples=_prediction_count(config),
        epoch_test_every_n_epochs=1,
        seed=config.seed,
    )


def _setup_toy(config: InstanceSweepConfig):
    comparison_config = comparison_workflow.build_config(_make_comparison_args(config))
    with _quiet(not config.verbose):
        datamodule = comparison_workflow.create_toy_datamodule(comparison_config)
        datamodule.setup(config.prediction_split)
        task = comparison_workflow.create_toy_task(
            comparison_config,
            datamodule.weight_assignments or [],
        )
        image_processor = comparison_workflow.create_toy_image_processor(comparison_config)
    task.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return comparison_config, datamodule, task, image_processor


def _setup_graha(config: InstanceSweepConfig):
    comparison_config = comparison_workflow.build_config(_make_comparison_args(config))
    graha_config = comparison_workflow.build_graha_config(
        comparison_config,
        config.output_root / "_graha_setup",
    )
    with _quiet(not config.verbose):
        graha_workflow.configure_proj_environment()
        graha_workflow.configure_python_paths(graha_config)
        graha_workflow.validate_required_paths(graha_config)
        deps = graha_workflow.import_project_dependencies()
        datamodule_cls = deps["LunarObjectDetectionInstanceMaskDatamodule"]
        task_cls = graha_workflow.make_notebook_object_detection_task_class(
            deps["LunarObjectDetectionTask"]
        )
        means, stds = graha_workflow.calculate_train_stats(
            graha_config,
            datamodule_cls,
        )
        datamodule = graha_workflow.create_datamodule(
            graha_config,
            datamodule_cls,
            means,
            stds,
        )
        datamodule.setup(config.prediction_split)
        sample_batch = graha_workflow.inspect_batch(datamodule)
        task = graha_workflow.create_task(graha_config, task_cls, sample_batch)
    task.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return datamodule, task


def run_toy_sweep(config: InstanceSweepConfig) -> list[dict[str, Any]]:
    if config.toy_checkpoint_dir is None:
        raise ValueError("Toy sweep requested but toy_checkpoint_dir is not set.")
    checkpoints = discover_checkpoints(config.toy_checkpoint_dir, max_checkpoints=config.max_checkpoints)
    print(f"[Toy] Found {len(checkpoints)} checkpoint(s).")
    _, datamodule, task, image_processor = _setup_toy(config)
    model_output_dir = config.output_root / "toy_model"
    rows = []
    for checkpoint in tqdm(checkpoints, desc="Toy checkpoints", dynamic_ncols=True):
        _load_lightning_checkpoint_state(task, checkpoint.path)
        if image_processor is None:
            cache_dir = save_graha_instance_prediction_cache(
                task=task,
                datamodule=datamodule,
                output_dir=model_output_dir / checkpoint.name,
                model_name="toy",
                split=config.prediction_split,
                n_samples=_prediction_count(config),
                score_threshold=0.0,
            )
        else:
            cache_dir = save_toy_instance_prediction_cache(
                task=task,
                datamodule=datamodule,
                output_dir=model_output_dir / checkpoint.name,
                image_processor=image_processor,
                model_name="toy",
                split=config.prediction_split,
                n_samples=_prediction_count(config),
                score_threshold=0.0,
            )
        metrics = _write_checkpoint_outputs(
            cache_dir=cache_dir,
            checkpoint_output_dir=model_output_dir / checkpoint.name,
            checkpoint=checkpoint,
            model_name="Toy",
            score_threshold=config.prediction_score_threshold,
        )
        rows.append(
            {
                "checkpoint_name": checkpoint.name,
                "epoch": checkpoint.epoch,
                "checkpoint_path": checkpoint.path,
                **metrics,
            }
        )
    _write_model_summary(model_output_dir, rows)
    del task, datamodule
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def run_graha_sweep(config: InstanceSweepConfig) -> list[dict[str, Any]]:
    if config.graha_checkpoint_dir is None:
        raise ValueError("Graha sweep requested but graha_checkpoint_dir is not set.")
    checkpoints = discover_checkpoints(config.graha_checkpoint_dir, max_checkpoints=config.max_checkpoints)
    print(f"[Graha] Found {len(checkpoints)} checkpoint(s).")
    datamodule, task = _setup_graha(config)
    model_output_dir = config.output_root / "graha_model"
    rows = []
    for checkpoint in tqdm(checkpoints, desc="Graha checkpoints", dynamic_ncols=True):
        _load_lightning_checkpoint_state(task, checkpoint.path)
        cache_dir = save_graha_instance_prediction_cache(
            task=task,
            datamodule=datamodule,
            output_dir=model_output_dir / checkpoint.name,
            model_name="graha",
            split=config.prediction_split,
            n_samples=_prediction_count(config),
            score_threshold=0.0,
        )
        metrics = _write_checkpoint_outputs(
            cache_dir=cache_dir,
            checkpoint_output_dir=model_output_dir / checkpoint.name,
            checkpoint=checkpoint,
            model_name="Graha",
            score_threshold=config.prediction_score_threshold,
        )
        rows.append(
            {
                "checkpoint_name": checkpoint.name,
                "epoch": checkpoint.epoch,
                "checkpoint_path": checkpoint.path,
                **metrics,
            }
        )
    _write_model_summary(model_output_dir, rows)
    del task, datamodule
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def run_sweep(config: InstanceSweepConfig) -> dict[str, list[dict[str, Any]]]:
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
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--band-filter", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--toy-batch-size", type=int, default=2)
    parser.add_argument("--toy-num-workers", type=int, default=4)
    parser.add_argument("--toy-normalize-inputs", action="store_true")
    parser.add_argument(
        "--toy-architecture",
        choices=["mask2former", "dino-mask-rcnn"],
        default="mask2former",
    )
    parser.add_argument("--dino-checkpoint", type=str, default=None)
    parser.add_argument("--graha-pretrain-dir", type=str, default=None)
    parser.add_argument("--graha-wac-mode", choices=["new-wac", "vis-uv"], default="new-wac")
    parser.add_argument("--graha-vis-uv-merge-method", choices=["mean", "max"], default="mean")
    parser.add_argument("--graha-stats-batch-size", type=int, default=16)
    parser.add_argument("--graha-batch-size", type=int, default=2)
    parser.add_argument("--graha-num-workers", type=int, default=4)
    parser.add_argument("--graha-backbone-lr", type=float, default=5.0e-5)
    parser.add_argument("--graha-head-lr", type=float, default=2.0e-4)
    parser.add_argument("--graha-layer-decay", type=float, default=0.75)
    parser.add_argument("--graha-weight-decay", type=float, default=0.05)
    parser.add_argument("--graha-warmup-steps", type=int, default=500)
    parser.add_argument("--graha-anchor-sizes", type=lambda value: [[int(x)] for x in value.split(",")], default=[[8], [16], [32], [64]])
    parser.add_argument("--graha-anchor-aspect-ratios", type=lambda value: [float(x) for x in value.split(",")], default=[0.5, 1.0, 2.0])
    parser.add_argument("--graha-score-threshold", type=float, default=0.5)
    parser.add_argument("--prediction-split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--prediction-score-threshold", type=float, default=0.5)
    parser.add_argument("--mask-shift", type=int, nargs=2, default=(0, 0))
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notebook_dir = Path(__file__).resolve().parents[2] / "notebooks" / "full_model"
    ensure_data_symlink(args.simlink_dest, notebook_dir / "data")
    config = build_config(args)
    print("Output root:", config.output_root)
    print("Data root:", config.data_root)
    run_sweep(config)


if __name__ == "__main__":
    main()
