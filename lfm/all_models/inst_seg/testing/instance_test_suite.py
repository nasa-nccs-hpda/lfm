"""Instance segmentation test-suite output helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from lfm.all_models.all_tasks import CheckpointRecord
from lfm.all_models.all_tasks.utils import (
    plot_instance_cache_predictions,
    save_graha_instance_prediction_cache,
    save_toy_instance_prediction_cache,
)
from lfm.all_models.all_tasks.utils.metrics import _instance_metrics
from lfm.all_models.inst_seg.prediction.instance_prediction_cache import (
    _load_instance_prediction_cache,
)

INSTANCE_TEST_SUITE_METRICS = [
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


def semantic_f1(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = (pred_mask > 0).reshape(-1)
    gt = (gt_mask > 0).reshape(-1)
    tp = float(np.sum(pred & gt))
    fp = float(np.sum(pred & ~gt))
    fn = float(np.sum(~pred & gt))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return float(2 * precision * recall / (precision + recall + 1e-8))


def score_logits(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    eps = np.finfo(np.float32).eps
    clipped = np.clip(scores, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32)


def normalize_prediction_scores(sample: dict[str, Any]) -> np.ndarray:
    pred_mask = np.asarray(sample["pred_mask"])
    num_instances = len([value for value in np.unique(pred_mask) if int(value) != 0])
    scores = np.asarray(sample["pred_scores"], dtype=np.float32).reshape(-1)
    if scores.shape[0] < num_instances:
        scores = np.pad(
            scores, (0, num_instances - scores.shape[0]), constant_values=1.0
        )
    return scores


def threshold_instance_sample(
    sample: dict[str, Any], score_threshold: float
) -> dict[str, Any]:
    scores = normalize_prediction_scores(sample)
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


def instance_binary_masks(instance_mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(instance_mask)
    return [(mask == value) for value in np.unique(mask) if int(value) != 0]


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    return intersection / union if union > 0 else 0.0


def average_precision_from_detections(
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
    return float(
        np.sum((recall[changed + 1] - recall[changed]) * precision[changed + 1])
    )


def instance_ap_at_threshold(
    samples: list[dict[str, Any]], iou_threshold: float
) -> float:
    detections: list[tuple[float, bool]] = []
    total_gt = 0
    for sample in samples:
        gt_masks = instance_binary_masks(sample["gt_mask"])
        pred_masks = instance_binary_masks(sample["pred_mask"])
        pred_scores = normalize_prediction_scores(sample)
        total_gt += len(gt_masks)
        matched_gt: set[int] = set()
        order = (
            np.argsort(-pred_scores, kind="mergesort")
            if pred_scores.size
            else np.asarray([], dtype=np.int64)
        )
        for pred_index in order:
            if pred_index >= len(pred_masks):
                detections.append((float(pred_scores[pred_index]), False))
                continue
            best_iou = 0.0
            best_gt_index = None
            for gt_index, gt_mask in enumerate(gt_masks):
                if gt_index in matched_gt:
                    continue
                iou = mask_iou(pred_masks[pred_index], gt_mask)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = gt_index
            is_tp = best_gt_index is not None and best_iou >= iou_threshold
            if is_tp:
                matched_gt.add(int(best_gt_index))
            detections.append((float(pred_scores[pred_index]), bool(is_tp)))
    return average_precision_from_detections(detections, total_gt)


def instance_ap_metrics(samples: list[dict[str, Any]]) -> dict[str, float]:
    if not samples:
        return {
            "average_precision": 0.0,
            "average_precision_50": 0.0,
            "mean_average_precision": 0.0,
        }
    ap_by_threshold = {
        float(threshold): instance_ap_at_threshold(samples, float(threshold))
        for threshold in AP_IOU_THRESHOLDS
    }
    return {
        "average_precision": float(ap_by_threshold[0.5]),
        "average_precision_50": float(ap_by_threshold[0.5]),
        "mean_average_precision": float(np.mean(list(ap_by_threshold.values()))),
    }


def sample_metrics(sample: dict[str, Any]) -> dict[str, float]:
    inst = _instance_metrics(sample["pred_mask"], sample["gt_mask"])
    ap_metrics = instance_ap_metrics([sample])
    return {
        "semantic_f1": semantic_f1(sample["pred_mask"], sample["gt_mask"]),
        "instance_precision": float(inst["precision"]),
        "instance_recall": float(inst["recall"]),
        "instance_f1": float(inst["f1"]),
        "instance_mean_iou": float(inst["mean_iou"]),
        **ap_metrics,
        "num_pred": float(inst["num_pred"]),
        "num_gt": float(inst["num_gt"]),
    }


def metrics_to_array(metrics: dict[str, float]) -> np.ndarray:
    dtype = [(name, "f8") for name in INSTANCE_TEST_SUITE_METRICS]
    row = np.zeros((), dtype=dtype)
    for name in INSTANCE_TEST_SUITE_METRICS:
        row[name] = float(metrics[name])
    return row


def write_metrics(
    output_dir: Path, metrics: dict[str, float], *, header: str | None = None
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "metrics.npy", metrics_to_array(metrics))
    with (output_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        if header:
            f.write(header.rstrip() + "\n")
        for name in INSTANCE_TEST_SUITE_METRICS:
            f.write(f"{name}: {metrics[name]:.8f}\n")


def aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {name: 0.0 for name in INSTANCE_TEST_SUITE_METRICS}
    return {
        name: float(np.mean([row[name] for row in rows]))
        for name in INSTANCE_TEST_SUITE_METRICS
    }


def write_instance_test_suite_outputs(
    *,
    cache_dir: Path,
    checkpoint_output_dir: Path,
    checkpoint: CheckpointRecord,
    model_name: str,
    score_threshold: float,
) -> dict[str, float]:
    samples = _load_instance_prediction_cache(cache_dir)
    sample_metrics_rows = []
    raw_sample_values = []
    for sample_key, sample in tqdm(
        sorted(samples.items()),
        desc=f"{model_name} {checkpoint.name} samples",
        leave=False,
        dynamic_ncols=True,
        file=sys.stdout,
    ):
        display_key = str(sample_key).split("_input", 1)[0]
        thresholded_sample = threshold_instance_sample(sample, score_threshold)
        sample_dir = checkpoint_output_dir / display_key
        sample_dir.mkdir(parents=True, exist_ok=True)

        np.save(sample_dir / f"{display_key}_input.npy", thresholded_sample["image"])
        np.save(sample_dir / f"{display_key}_label.npy", thresholded_sample["gt_mask"])
        np.save(sample_dir / f"{display_key}_pred.npy", thresholded_sample["pred_mask"])
        np.save(
            sample_dir / f"{display_key}_pred_classes.npy",
            np.ones((thresholded_sample["pred_scores"].shape[0],), dtype=np.int64),
        )
        np.save(
            sample_dir / f"{display_key}_pred_logits.npy",
            score_logits(thresholded_sample["pred_scores"]),
        )
        np.save(
            sample_dir / f"{display_key}_gt_boxes.npy", thresholded_sample["gt_boxes"]
        )
        np.save(
            sample_dir / f"{display_key}_pred_boxes.npy",
            thresholded_sample["pred_boxes"],
        )
        np.save(
            sample_dir / f"{display_key}_pred_scores.npy",
            thresholded_sample["pred_scores"],
        )

        metrics = sample_metrics(thresholded_sample)
        metrics.update(instance_ap_metrics([sample]))
        sample_metrics_rows.append(metrics)
        raw_sample_values.append(sample)
        write_metrics(
            sample_dir,
            metrics,
            header=(
                f"model: {model_name}\n"
                f"checkpoint: {checkpoint.path}\n"
                f"sample_key: {display_key}\n"
                f"prediction_score_threshold: {score_threshold}"
            ),
        )

    aggregate = aggregate_metrics(sample_metrics_rows)
    aggregate.update(instance_ap_metrics(raw_sample_values))
    write_metrics(
        checkpoint_output_dir,
        aggregate,
        header=(
            f"model: {model_name}\n"
            f"checkpoint: {checkpoint.path}\n"
            f"epoch: {checkpoint.epoch}\n"
            f"samples: {len(sample_metrics_rows)}\n"
            f"prediction_score_threshold: {score_threshold}"
        ),
    )
    print(
        f"[{model_name}] {checkpoint.name}: "
        f"InstF1={aggregate['instance_f1']:.4f}, "
        f"SemF1={aggregate['semantic_f1']:.4f}, "
        f"AP50={aggregate['average_precision_50']:.4f}, "
        f"mAP={aggregate['mean_average_precision']:.4f}, "
        f"samples={len(sample_metrics_rows)}",
        flush=True,
    )
    return aggregate


def run_instance_test_suite(
    *,
    task,
    datamodule,
    output_dir: Path,
    model_name: str,
    split: str,
    n_samples: int,
    suite_name: str,
    score_threshold: float,
    epoch: int = -1,
    image_processor=None,
) -> tuple[dict[str, float], Path]:
    suite_dir = output_dir / "test_suite" / model_name / suite_name
    if image_processor is None:
        cache_dir = save_graha_instance_prediction_cache(
            task=task,
            datamodule=datamodule,
            output_dir=suite_dir,
            model_name=model_name,
            split=split,
            n_samples=n_samples,
            score_threshold=score_threshold,
        )
    else:
        cache_dir = save_toy_instance_prediction_cache(
            task=task,
            datamodule=datamodule,
            output_dir=suite_dir,
            image_processor=image_processor,
            model_name=model_name,
            split=split,
            n_samples=n_samples,
            score_threshold=score_threshold,
        )
    metrics = write_instance_test_suite_outputs(
        cache_dir=cache_dir,
        checkpoint_output_dir=suite_dir,
        checkpoint=CheckpointRecord(
            path=Path(f"{model_name}_{suite_name}"),
            epoch=epoch,
            name=suite_name,
        ),
        model_name=model_name,
        score_threshold=score_threshold,
    )
    plot_instance_cache_predictions(
        cache_dir,
        suite_dir,
        model_name=model_name,
        n_samples=min(5, n_samples),
        filename=f"{split}_instance_predictions.png",
    )
    return metrics, cache_dir
