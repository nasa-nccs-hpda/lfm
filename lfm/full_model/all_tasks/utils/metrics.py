"""Metric helpers for segmentation plots and prediction caches."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from lfm.full_model.all_tasks.utils.common import (
    _display_sample_key,
    _model_display_name,
)
from lfm.full_model.all_tasks.utils.prediction_cache import \
    _load_prediction_cache


def calculate_f1_score(pred: np.ndarray, label: np.ndarray) -> float:
    pred = pred.reshape(-1)
    label = label.reshape(-1)
    tp = np.sum((pred == 1) & (label == 1))
    fp = np.sum((pred == 1) & (label == 0))
    fn = np.sum((pred == 0) & (label == 1))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return float(2 * (precision * recall) / (precision + recall + 1e-8))


def _binary_metrics(pred: np.ndarray, label: np.ndarray) -> dict[str, float]:
    pred_bool = pred.astype(bool).reshape(-1)
    label_bool = label.astype(bool).reshape(-1)

    tp = float(np.sum(pred_bool & label_bool))
    fp = float(np.sum(pred_bool & ~label_bool))
    fn = float(np.sum(~pred_bool & label_bool))
    tn = float(np.sum(~pred_bool & ~label_bool))
    eps = 1e-8

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "pixel_accuracy": accuracy,
        "foreground_precision": precision,
        "foreground_recall": recall,
        "foreground_f1": f1,
        "iou": iou,
        "predicted_foreground_fraction": float(np.mean(pred_bool)),
        "ground_truth_foreground_fraction": float(np.mean(label_bool)),
    }


def evaluate_prediction_caches(
    cache_dirs: dict[str, str | Path],
    output_dir: str | Path,
    *,
    filename_prefix: str = "prediction_cache_metrics",
) -> tuple[list[dict], list[dict]]:
    """Compute binary segmentation metrics from prediction caches."""
    loaded = {
        name: _load_prediction_cache(path) for name, path in cache_dirs.items()
    }
    if not loaded:
        raise ValueError("No prediction caches were provided.")

    first_model = next(iter(loaded))
    shared_keys = set(loaded[first_model])
    for samples in loaded.values():
        shared_keys &= set(samples)
    sample_keys = sorted(shared_keys)
    if not sample_keys:
        raise ValueError(
            "Prediction caches do not contain matching sample keys."
        )

    rows = []
    for model_name, samples in loaded.items():
        display_name = _model_display_name(model_name)
        for sample_key in sample_keys:
            sample = samples[sample_key]
            metrics = _binary_metrics(sample["pred"], sample["label"])
            rows.append(
                {
                    "model": display_name,
                    "sample_key": _display_sample_key(sample_key),
                    **metrics,
                }
            )

    summary_rows = []
    metric_names = [
        "pixel_accuracy",
        "foreground_precision",
        "foreground_recall",
        "foreground_f1",
        "iou",
        "predicted_foreground_fraction",
        "ground_truth_foreground_fraction",
    ]
    for model_name in [_model_display_name(name) for name in loaded]:
        model_rows = [row for row in rows if row["model"] == model_name]
        summary = {"model": model_name, "n_samples": len(model_rows)}
        for metric in metric_names:
            metrics = [row[metric] for row in model_rows]
            summary[metric] = float(np.mean(metrics))
        summary_rows.append(summary)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{filename_prefix}.json"
    csv_path = output_dir / f"{filename_prefix}.csv"
    summary_json_path = output_dir / f"{filename_prefix}_summary.json"
    summary_csv_path = output_dir / f"{filename_prefix}_summary.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    fieldnames = ["model", "sample_key", *metric_names]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_fieldnames = ["model", "n_samples", *metric_names]
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved per-sample metrics to {csv_path}")
    print(f"Saved summary metrics to {summary_csv_path}")
    return rows, summary_rows


def _instance_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    pred_ids = [int(x) for x in np.unique(pred_mask) if int(x) != 0]
    gt_ids = [int(x) for x in np.unique(gt_mask) if int(x) != 0]
    if not pred_ids and not gt_ids:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "mean_iou": 1.0,
            "num_pred": 0,
            "num_gt": 0,
        }
    if not pred_ids:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_iou": 0.0,
            "num_pred": 0,
            "num_gt": len(gt_ids),
        }
    if not gt_ids:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_iou": 0.0,
            "num_pred": len(pred_ids),
            "num_gt": 0,
        }

    candidates = []
    for pred_id in pred_ids:
        pred_pixels = pred_mask == pred_id
        for gt_id in gt_ids:
            gt_pixels = gt_mask == gt_id
            intersection = float(np.logical_and(pred_pixels, gt_pixels).sum())
            union = float(np.logical_or(pred_pixels, gt_pixels).sum())
            iou = intersection / union if union > 0 else 0.0
            if iou >= iou_threshold:
                candidates.append((iou, pred_id, gt_id))

    matched_pred = set()
    matched_gt = set()
    matched_ious = []
    for iou, pred_id, gt_id in sorted(candidates, reverse=True):
        if pred_id in matched_pred or gt_id in matched_gt:
            continue
        matched_pred.add(pred_id)
        matched_gt.add(gt_id)
        matched_ious.append(iou)

    tp = len(matched_ious)
    precision = tp / len(pred_ids) if pred_ids else 0.0
    recall = tp / len(gt_ids) if gt_ids else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "num_pred": len(pred_ids),
        "num_gt": len(gt_ids),
    }
