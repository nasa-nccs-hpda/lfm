"""Instance prediction-cache helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from lfm.all_models.all_tasks.utils.common import (
    _extract_paths,
    _get_split_dataloader,
    _sample_key,
)


def _to_cpu_prediction_list(predictions) -> list[dict[str, torch.Tensor]]:
    if hasattr(predictions, "output"):
        predictions = predictions.output
    result = []
    for pred in predictions:
        result.append(
            {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in pred.items()
            }
        )
    return result


def _boxes_from_instance_mask(mask: np.ndarray) -> np.ndarray:
    boxes = []
    for instance_id in np.unique(mask):
        if int(instance_id) == 0:
            continue
        ys, xs = np.where(mask == instance_id)
        if xs.size == 0 or ys.size == 0:
            continue
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


def _binary_maps_to_instance_map(
    binary_masks: np.ndarray | torch.Tensor,
    class_labels: np.ndarray | torch.Tensor | None = None,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    if torch.is_tensor(binary_masks):
        binary_masks = binary_masks.detach().cpu().numpy()
    if class_labels is not None and torch.is_tensor(class_labels):
        class_labels = class_labels.detach().cpu().numpy()
    binary_masks = np.asarray(binary_masks)
    if binary_masks.ndim == 2:
        if (
            np.issubdtype(binary_masks.dtype, np.integer)
            and binary_masks.max(initial=0) > 1
        ):
            return binary_masks.astype(np.int32)
        return (binary_masks > threshold).astype(np.int32)
    if binary_masks.ndim == 4 and binary_masks.shape[1] == 1:
        binary_masks = binary_masks[:, 0]

    h, w = binary_masks.shape[-2:]
    instance_map = np.zeros((h, w), dtype=np.int32)
    next_id = 1
    for idx, mask in enumerate(binary_masks):
        if (
            class_labels is not None
            and idx < len(class_labels)
            and int(class_labels[idx]) == 0
        ):
            continue
        mask_bool = mask > threshold
        if not mask_bool.any():
            continue
        instance_map[mask_bool] = next_id
        next_id += 1
    return instance_map


def _prediction_masks_to_instance_map(
    masks: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    if masks.numel() == 0:
        shape = masks.shape[-2:] if masks.ndim >= 2 else (1, 1)
        return np.zeros(tuple(shape), dtype=np.int32)
    masks = masks.detach().cpu()
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    instance_map = np.zeros(tuple(masks.shape[-2:]), dtype=np.int32)
    for idx, mask in enumerate(masks, start=1):
        mask_bool = mask.numpy() > threshold
        if mask_bool.any():
            instance_map[mask_bool] = idx
    return instance_map


def _save_instance_cache_arrays(
    cache_dir: Path,
    *,
    index: int,
    sample_key: str,
    model_name: str,
    image: np.ndarray,
    gt_mask: np.ndarray,
    gt_boxes: np.ndarray,
    pred_mask: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    image_path: str = "",
) -> dict:
    filename = f"{index:04d}_{sample_key}.npz"
    np.savez_compressed(
        cache_dir / filename,
        image=image,
        gt_mask=gt_mask.astype(np.int32),
        gt_boxes=gt_boxes.astype(np.float32),
        pred_mask=pred_mask.astype(np.int32),
        pred_boxes=pred_boxes.astype(np.float32),
        pred_scores=pred_scores.astype(np.float32),
        sample_key=sample_key,
        model_name=model_name,
        image_path=image_path,
    )
    return {
        "index": index,
        "sample_key": sample_key,
        "file": filename,
        "image_path": image_path,
    }


def save_toy_instance_prediction_cache(
    task,
    datamodule,
    output_dir: str | Path,
    image_processor,
    *,
    model_name: str = "toy",
    split: str = "val",
    n_samples: int = 5,
    score_threshold: float = 0.5,
    setup_datamodule: bool = True,
) -> Path:
    """Save Toy Mask2Former true-instance predictions in a shared cache format."""
    cache_dir = Path(output_dir) / "prediction_cache" / model_name / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    if setup_datamodule:
        datamodule.setup("fit" if split in {"train", "val"} else "test")
    dataloader = _get_split_dataloader(datamodule, split)

    was_training = task.training
    task.eval()
    device = task.device
    manifest = []
    saved = 0
    with torch.no_grad():
        for batch in dataloader:
            x = batch["pixel_values"].to(device)
            outputs = task.model(pixel_values=x)
            target_sizes = [(x.shape[-2], x.shape[-1])] * x.shape[0]
            post_processed = image_processor.post_process_instance_segmentation(
                outputs,
                threshold=score_threshold,
                target_sizes=target_sizes,
                return_binary_maps=True,
            )
            filenames = batch.get("filename", [""] * x.shape[0])
            for i, result in enumerate(post_processed):
                if saved >= n_samples:
                    break
                sample_key = _sample_key(
                    filenames[i] if i < len(filenames) else None, saved
                )
                gt_mask = (
                    batch["instance_mask"][i].detach().cpu().numpy().astype(np.int32)
                )
                gt_boxes = _boxes_from_instance_mask(gt_mask)
                segments_info = result.get("segments_info", [])
                class_labels = np.asarray(
                    [segment.get("label_id", 1) for segment in segments_info],
                    dtype=np.int64,
                )
                pred_mask = _binary_maps_to_instance_map(
                    result["segmentation"],
                    class_labels if class_labels.size else None,
                )
                pred_boxes = _boxes_from_instance_mask(pred_mask)
                pred_scores = np.asarray(
                    [segment.get("score", 1.0) for segment in segments_info],
                    dtype=np.float32,
                )
                if pred_scores.shape[0] != pred_boxes.shape[0]:
                    pred_scores = np.ones((pred_boxes.shape[0],), dtype=np.float32)
                manifest.append(
                    _save_instance_cache_arrays(
                        cache_dir,
                        index=saved,
                        sample_key=sample_key,
                        model_name=model_name,
                        image=x[i].detach().cpu().numpy(),
                        gt_mask=gt_mask,
                        gt_boxes=gt_boxes,
                        pred_mask=pred_mask,
                        pred_boxes=pred_boxes,
                        pred_scores=pred_scores,
                        image_path=str(filenames[i]) if i < len(filenames) else "",
                    )
                )
                saved += 1
            if saved >= n_samples:
                break
    task.train(was_training)
    with (cache_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"Saved {saved} {model_name} instance prediction cache file(s) to {cache_dir}"
    )
    return cache_dir


def save_graha_instance_prediction_cache(
    task,
    datamodule,
    output_dir: str | Path,
    *,
    model_name: str = "graha",
    split: str = "val",
    n_samples: int = 5,
    score_threshold: float = 0.5,
    setup_datamodule: bool = True,
) -> Path:
    """Save Graha Mask R-CNN true-instance predictions in a shared cache format."""
    cache_dir = Path(output_dir) / "prediction_cache" / model_name / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    if setup_datamodule:
        datamodule.setup("fit" if split in {"train", "val"} else "test")
    dataloader = _get_split_dataloader(datamodule, split)

    was_training = task.training
    task.eval()
    device = task.device
    manifest = []
    saved = 0
    with torch.no_grad():
        for batch in dataloader:
            x = batch["image"].to(device)
            predictions = _to_cpu_prediction_list(
                task.predict_step({"image": x}, batch_idx=0)
            )
            filenames, _ = _extract_paths(batch)
            for i, pred in enumerate(predictions):
                if saved >= n_samples:
                    break
                sample_key = _sample_key(
                    filenames[i] if i < len(filenames) else None, saved
                )
                gt_masks = batch["masks"][i].detach().cpu()
                gt_mask = _prediction_masks_to_instance_map(gt_masks, threshold=0.5)
                gt_boxes = batch["boxes"][i].detach().cpu().numpy().astype(np.float32)
                scores = pred.get("scores", torch.zeros((0,), dtype=torch.float32))
                keep = scores >= score_threshold
                pred_boxes_t = pred.get(
                    "boxes", torch.zeros((0, 4), dtype=torch.float32)
                )[keep]
                pred_masks_t = pred.get(
                    "masks",
                    torch.zeros((0, *gt_masks.shape[-2:]), dtype=torch.float32),
                )[keep]
                pred_scores = scores[keep].detach().cpu().numpy().astype(np.float32)
                pred_mask = _prediction_masks_to_instance_map(
                    pred_masks_t, threshold=0.5
                )
                manifest.append(
                    _save_instance_cache_arrays(
                        cache_dir,
                        index=saved,
                        sample_key=sample_key,
                        model_name=model_name,
                        image=batch["image"][i].detach().cpu().numpy(),
                        gt_mask=gt_mask,
                        gt_boxes=gt_boxes,
                        pred_mask=pred_mask,
                        pred_boxes=pred_boxes_t.detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32),
                        pred_scores=pred_scores,
                        image_path=str(filenames[i]) if i < len(filenames) else "",
                    )
                )
                saved += 1
            if saved >= n_samples:
                break
    task.train(was_training)
    with (cache_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"Saved {saved} {model_name} instance prediction cache file(s) to {cache_dir}"
    )
    return cache_dir


def _load_instance_prediction_cache(cache_dir: str | Path) -> dict[str, dict]:
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        files = [cache_dir / item["file"] for item in manifest]
    else:
        files = sorted(cache_dir.glob("*.npz"))
    samples = {}
    for path in files:
        data = np.load(path, allow_pickle=False)
        sample_key = str(data["sample_key"])
        samples[sample_key] = {
            "image": data["image"],
            "gt_mask": data["gt_mask"],
            "gt_boxes": data["gt_boxes"],
            "pred_mask": data["pred_mask"],
            "pred_boxes": data["pred_boxes"],
            "pred_scores": data["pred_scores"],
            "image_path": str(data["image_path"]),
        }
    return samples
