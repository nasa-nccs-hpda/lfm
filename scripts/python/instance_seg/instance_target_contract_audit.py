"""Audit instance target contracts across Toy and Graha datamodules.

This is a diagnostic entrypoint. It does not train or instantiate model
weights. It checks whether the active Toy Mask2Former, Toy Mask R-CNN, and
Graha Mask R-CNN datamodules emit compatible instance target geometry from the
same split-folder dataset.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

LFM_ROOT = Path(__file__).resolve().parents[3]
if str(LFM_ROOT) not in sys.path:
    sys.path.insert(0, str(LFM_ROOT))

from lfm.all_models.inst_seg.instance_data_utils import (
    instance_mask_to_object_detection_targets,
)
from lfm.full_model.inst_seg.instance_mask_datamodule import (
    LunarObjectDetectionInstanceMaskDatamodule,
)
from lfm.toy_model.inst_seg.lightning_wrappers import (
    ToyDinoMaskRCNNSplitDataModule,
    ToyInstanceSegSplitDataModule,
)


def tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu()
    summary: dict[str, Any] = {
        "kind": "tensor",
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
    }
    if tensor.numel() == 0:
        return summary
    numeric = tensor.float() if not tensor.is_floating_point() else tensor
    finite = torch.isfinite(numeric)
    summary["nonfinite_count"] = int((~finite).sum().item())
    if torch.any(finite):
        valid = numeric[finite]
        summary.update(
            {
                "min": round(float(valid.min().item()), 8),
                "max": round(float(valid.max().item()), 8),
                "mean": round(float(valid.mean().item()), 8),
                "std": round(float(valid.std(unbiased=False).item()), 8),
            }
        )
    return summary


def compact_summary(value: Any, *, depth: int = 0) -> Any:
    if depth > 3:
        return {"kind": type(value).__name__}
    if torch.is_tensor(value):
        return tensor_summary(value)
    if isinstance(value, dict):
        return {
            str(key): compact_summary(val, depth=depth + 1)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return {
            "kind": type(value).__name__,
            "length": len(value),
            "items": [compact_summary(item, depth=depth + 1) for item in value[:3]],
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"kind": type(value).__name__, "repr": repr(value)[:200]}


def _sample_image(sample: dict[str, Any]) -> torch.Tensor:
    if "image" in sample:
        return sample["image"]
    return sample["pixel_values"]


def _sample_instance_mask(sample: dict[str, Any]) -> torch.Tensor:
    if "instance_mask" in sample:
        return sample["instance_mask"]
    return sample["mask"]


def _sample_object_targets(
    sample: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if {"boxes", "labels", "masks"}.issubset(sample):
        return sample["boxes"], sample["labels"], sample["masks"]
    return instance_mask_to_object_detection_targets(
        _sample_instance_mask(sample),
        box_format="xyxy",
    )


def validate_object_targets(
    *,
    image: torch.Tensor,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
) -> dict[str, Any]:
    height, width = image.shape[-2:]
    valid: dict[str, Any] = {
        "boxes_shape_ok": list(boxes.shape)[-1:] == [4] if boxes.ndim == 2 else False,
        "boxes_dtype": str(boxes.dtype),
        "labels_dtype": str(labels.dtype),
        "masks_dtype": str(masks.dtype),
        "target_count": int(labels.shape[0]),
        "count_consistent": (
            boxes.ndim == 2
            and masks.ndim == 3
            and boxes.shape[0] == labels.shape[0] == masks.shape[0]
        ),
    }
    if boxes.numel() == 0:
        valid.update(
            {
                "boxes_inside_image": True,
                "boxes_positive_area": True,
            }
        )
        return valid

    x1, y1, x2, y2 = boxes.unbind(dim=1)
    valid["boxes_inside_image"] = bool(
        torch.all(x1 >= 0)
        and torch.all(y1 >= 0)
        and torch.all(x2 <= width)
        and torch.all(y2 <= height)
    )
    valid["boxes_positive_area"] = bool(torch.all(x2 > x1) and torch.all(y2 > y1))
    return valid


def summarize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    image = _sample_image(sample)
    instance_mask = _sample_instance_mask(sample)
    boxes, labels, masks = _sample_object_targets(sample)
    instance_ids = torch.unique(instance_mask)
    instance_ids = instance_ids[instance_ids > 0]
    return {
        "keys": sorted(sample.keys()),
        "image": tensor_summary(image),
        "instance_mask": tensor_summary(instance_mask),
        "instance_count_from_mask": int(instance_ids.numel()),
        "object_targets": {
            "boxes": tensor_summary(boxes),
            "labels": tensor_summary(labels),
            "masks": tensor_summary(masks),
            "validation": validate_object_targets(
                image=image,
                boxes=boxes,
                labels=labels,
                masks=masks,
            ),
        },
        "raw_sample": compact_summary(sample),
    }


def datamodule_sample_summary(datamodule: Any, split: str) -> dict[str, Any]:
    dataset = getattr(datamodule, f"{split}_dataset")
    sample = dataset[0]
    loader = getattr(datamodule, f"{split}_dataloader")()
    batch = next(iter(loader))
    return {
        "dataset_type": f"{type(dataset).__module__}.{type(dataset).__name__}",
        "dataset_len": int(len(dataset)),
        "sample": summarize_sample(sample),
        "batch": compact_summary(batch),
    }


def make_datamodules(args: argparse.Namespace) -> dict[str, Any]:
    common_toy = {
        "data_root": args.data_root,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "target_size": args.target_size,
        "image_glob": args.image_glob,
        "label_glob": args.label_glob,
        "image_suffix": args.image_suffix,
        "label_suffix": args.label_suffix,
        "band_filter": args.band_filter,
        "normalize_inputs": False,
        "scale_inputs": True,
        "mask_shift": tuple(args.mask_shift),
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "max_test_samples": args.max_test_samples,
        "ignore_nodata_in_loss": args.ignore_nodata_in_loss,
        "nodata_ignore_index": args.nodata_ignore_index,
    }
    common_graha = {
        "data_root": args.data_root,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "crop_size": args.target_size,
        "image_glob": args.image_glob,
        "label_glob": args.label_glob,
        "image_suffix": args.image_suffix,
        "label_suffix": args.label_suffix,
        "band_filter": args.band_filter,
        "means": None,
        "stds": None,
        "target_box_format": "xyxy",
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "max_test_samples": args.max_test_samples,
        "no_data_replace": 0.0,
        "no_label_replace": None,
        "mask_shift": tuple(args.mask_shift),
        "ignore_nodata_in_loss": args.ignore_nodata_in_loss,
        "nodata_ignore_index": args.nodata_ignore_index,
    }
    return {
        "toy_mask2former": ToyInstanceSegSplitDataModule(**common_toy),
        "toy_dino_mask_rcnn": ToyDinoMaskRCNNSplitDataModule(**common_toy),
        "graha_mask_rcnn": LunarObjectDetectionInstanceMaskDatamodule(**common_graha),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--image-glob", default="*.tif")
    parser.add_argument("--label-glob", default="*_label.npz")
    parser.add_argument("--image-suffix", default="_input_wac_chip")
    parser.add_argument("--label-suffix", default="_label")
    parser.add_argument(
        "--band-filter", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6]
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=8)
    parser.add_argument("--max-val-samples", type=int, default=8)
    parser.add_argument("--max-test-samples", type=int, default=8)
    parser.add_argument("--mask-shift", type=int, nargs=2, default=(0, 0))
    parser.add_argument("--ignore-nodata-in-loss", action="store_true")
    parser.add_argument("--nodata-ignore-index", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datamodules = make_datamodules(args)
    result: dict[str, Any] = {"data_root": str(args.data_root), "datamodules": {}}

    for name, datamodule in datamodules.items():
        datamodule.setup(None)
        result["datamodules"][name] = {
            "type": f"{type(datamodule).__module__}.{type(datamodule).__name__}",
            "weight_assignments": getattr(datamodule, "weight_assignments", None),
            "splits": {},
        }
        for split in ("train", "val", "test"):
            dataset = getattr(datamodule, f"{split}_dataset", None)
            if dataset is None:
                result["datamodules"][name]["splits"][split] = {"present": False}
                continue
            result["datamodules"][name]["splits"][split] = datamodule_sample_summary(
                datamodule,
                split,
            )

    output_json = args.output_json
    if output_json is None:
        output_json = Path("scripts/outputs/instance_target_contract_audit.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote instance target contract audit to {output_json}", flush=True)


if __name__ == "__main__":
    main()
