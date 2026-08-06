"""Smoke test local TerraTorch registration for Toy DINO Mask R-CNN.

This diagnostic does not train. It checks the narrow contracts needed before a
Toy DINO TerraTorch workflow can be promoted into the normal comparison path:

1. Toy Mask R-CNN datamodule emits object detection targets.
2. The local Toy DINO backbone can be built from TerraTorch's registry.
3. The registered backbone produces an image feature pyramid.
4. The registered backbone can drive a TorchVision Mask R-CNN loss step.
5. TerraTorch's ObjectDetectionTask can, or cannot, consume that same backbone.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import torch

LFM_ROOT = Path(__file__).resolve().parents[3]
if str(LFM_ROOT) not in sys.path:
    sys.path.insert(0, str(LFM_ROOT))

from lfm.all_models.all_tasks.data.normalization import (
    load_terramind_pretraining_stats,
)
from lfm.all_models.all_tasks import config_defaults as defaults
from lfm.toy_model.inst_seg.lightning_wrappers import ToyDinoMaskRCNNSplitDataModule
from lfm.toy_model.inst_seg.terratorch_dino_backbone import require_terratorch_registry

BACKBONE_NAME = "toy_dino_v3_mask_rcnn_backbone"


def tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu()
    summary: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
    }
    if tensor.numel() == 0:
        return summary
    numeric = tensor.float()
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


def compact_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return tensor_summary(value)
    if isinstance(value, dict):
        return {str(key): compact_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [compact_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def safe_step(name: str, fn: Callable[[], Any]) -> dict[str, Any]:
    try:
        return {"ok": True, "result": fn()}
    except Exception as exc:  # noqa: BLE001 - diagnostics must capture all failures.
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-20:],
        }


def get_normalization_stats(
    args: argparse.Namespace,
) -> tuple[list[float] | None, list[float] | None]:
    if not args.normalize_inputs:
        return None, None
    if args.normalization_source != "pretrain":
        raise ValueError(
            "This diagnostic currently supports pretrain normalization only. "
            "Run without --normalize-inputs for an unnormalized shape-only test."
        )
    if args.modality_info is None:
        raise ValueError("--modality-info is required with --normalize-inputs.")
    return load_terramind_pretraining_stats(
        args.modality_info,
        normalization_modality=defaults.normalize_normalization_modality(
            args.normalization_modality
        ),
        band_filter=args.band_filter,
    )


def make_datamodule(args: argparse.Namespace) -> ToyDinoMaskRCNNSplitDataModule:
    means, stds = get_normalization_stats(args)
    datamodule = ToyDinoMaskRCNNSplitDataModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=args.target_size,
        image_glob=args.image_glob,
        label_glob=args.label_glob,
        image_suffix=args.image_suffix,
        label_suffix=args.label_suffix,
        band_filter=args.band_filter,
        normalize_inputs=args.normalize_inputs,
        means=means,
        stds=stds,
        scale_inputs=args.normalization_source != "pretrain",
        mask_shift=tuple(args.mask_shift),
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        ignore_nodata_in_loss=args.ignore_nodata_in_loss,
        nodata_ignore_index=args.nodata_ignore_index,
    )
    datamodule.setup("fit")
    if datamodule.weight_assignments is None:
        raise RuntimeError("Toy DINO datamodule did not infer weight assignments.")
    return datamodule


def object_detection_targets_from_batch(
    batch: dict[str, Any],
) -> list[dict[str, torch.Tensor]]:
    targets = []
    for boxes, labels, masks in zip(batch["boxes"], batch["labels"], batch["masks"]):
        targets.append(
            {
                "boxes": boxes.float(),
                "labels": labels.long(),
                "masks": masks.to(torch.uint8),
            }
        )
    return targets


def summarize_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "keys": sorted(batch.keys()),
        "image": tensor_summary(batch["image"]),
        "boxes_per_image": [list(boxes.shape) for boxes in batch["boxes"]],
        "labels_per_image": [list(labels.shape) for labels in batch["labels"]],
        "masks_per_image": [list(masks.shape) for masks in batch["masks"]],
        "filenames": list(batch["filename"]),
    }


def build_registered_backbone(
    args: argparse.Namespace,
    weight_assignments: list[str],
):
    registry = require_terratorch_registry()
    kwargs = {
        "num_bands": len(weight_assignments),
        "weight_assignments": weight_assignments,
        "out_channels": args.out_channels,
        "layers_to_extract": args.layers_to_extract,
        "output_strides": args.output_strides,
        "return_format": args.backbone_return_format,
        "freeze_encoder": args.freeze_backbone,
        "device": str(args.device),
    }
    if args.backbone_feature_names is not None:
        kwargs["feature_names"] = args.backbone_feature_names
    if args.dino_checkpoint is not None:
        kwargs["checkpoint_path"] = str(args.dino_checkpoint)
    return registry.build(BACKBONE_NAME, **kwargs)


def summarize_features(features: Any, input_shape: list[int]) -> dict[str, Any]:
    if isinstance(features, dict):
        items = features.items()
    else:
        items = [(str(index), value) for index, value in enumerate(features)]
    _, _, image_h, image_w = input_shape
    summary = {}
    for name, value in items:
        feature = value.detach().cpu()
        height, width = feature.shape[-2:]
        summary[str(name)] = {
            **tensor_summary(feature),
            "approx_stride_h": round(float(image_h / height), 4),
            "approx_stride_w": round(float(image_w / width), 4),
        }
    return summary


def make_torchvision_mask_rcnn(
    backbone,
    *,
    num_bands: int,
    target_size: int,
    anchor_sizes: list[list[int]],
    anchor_aspect_ratios: list[float],
):
    from torchvision.models.detection import MaskRCNN
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign

    aspect_ratios = tuple(tuple(anchor_aspect_ratios) for _ in anchor_sizes)
    anchor_generator = AnchorGenerator(
        sizes=tuple(tuple(int(size) for size in level) for level in anchor_sizes),
        aspect_ratios=aspect_ratios,
    )
    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=7,
        sampling_ratio=2,
    )
    mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=14,
        sampling_ratio=2,
    )
    return MaskRCNN(
        backbone,
        num_classes=2,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=box_roi_pool,
        mask_roi_pool=mask_roi_pool,
        min_size=target_size,
        max_size=target_size,
        image_mean=[0.0] * int(num_bands),
        image_std=[1.0] * int(num_bands),
    )


def run_torchvision_loss_step(
    args: argparse.Namespace,
    backbone,
    batch: dict[str, Any],
) -> dict[str, Any]:
    model = make_torchvision_mask_rcnn(
        backbone,
        num_bands=int(batch["image"].shape[1]),
        target_size=args.target_size,
        anchor_sizes=args.anchor_sizes,
        anchor_aspect_ratios=args.anchor_aspect_ratios,
    ).to(args.device)
    model.train()
    images = [image.to(args.device) for image in batch["image"]]
    targets = [
        {key: value.to(args.device) for key, value in target.items()}
        for target in object_detection_targets_from_batch(batch)
    ]
    with torch.set_grad_enabled(args.run_backward):
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        if args.run_backward:
            loss.backward()
    return {
        "loss_terms": {
            key: round(float(value.detach().cpu()), 8)
            for key, value in loss_dict.items()
        },
        "loss": round(float(loss.detach().cpu()), 8),
    }


def unwrap_torchvision_backbone(backbone):
    """Return the inner TorchVision-compatible backbone when wrapped for TerraTorch."""
    return getattr(backbone, "backbone", backbone)


def build_terratorch_task(
    args: argparse.Namespace,
    batch: dict[str, Any],
    weight_assignments: list[str],
):
    from terratorch.tasks import ObjectDetectionTask

    backbone_args = {
        "backbone_num_bands": len(weight_assignments),
        "backbone_weight_assignments": weight_assignments,
        "backbone_out_channels": args.out_channels,
        "backbone_layers_to_extract": args.layers_to_extract,
        "backbone_output_strides": args.output_strides,
        "backbone_return_format": args.backbone_return_format,
        "backbone_freeze_encoder": args.freeze_backbone,
        "backbone_device": str(args.device),
    }
    if args.backbone_feature_names is not None:
        backbone_args["backbone_feature_names"] = args.backbone_feature_names

    task = ObjectDetectionTask(
        model_factory="ObjectDetectionModelFactory",
        model_args={
            "framework": "mask-rcnn",
            "backbone": BACKBONE_NAME,
            **backbone_args,
            "num_classes": 2,
            "in_channels": int(batch["image"].shape[1]),
            "framework_min_size": args.target_size,
            "framework_max_size": args.target_size,
            "necks": [],
        },
        freeze_backbone=False,
        freeze_decoder=False,
        class_names=["Background", "Crater"],
    ).to(args.device)
    return task


def run_terratorch_task_loss_step(
    args: argparse.Namespace,
    task: Any,
    batch: dict[str, Any],
) -> dict[str, Any]:
    task.train()
    images = batch["image"].to(args.device)
    if hasattr(task, "reformat_batch"):
        targets = task.reformat_batch(batch, batch_size=images.shape[0])
    else:
        targets = object_detection_targets_from_batch(batch)
    targets = [
        {key: value.to(args.device) for key, value in target.items()}
        for target in targets
    ]
    with torch.set_grad_enabled(args.run_backward):
        output = task(images, targets)
        loss_dict = output if isinstance(output, dict) else output.output
        loss = sum(loss_dict.values())
        if args.run_backward:
            loss.backward()
    return {
        "task_type": f"{type(task).__module__}.{type(task).__name__}",
        "model_type": f"{type(task.model).__module__}.{type(task.model).__name__}",
        "loss_terms": {
            key: round(float(value.detach().cpu()), 8)
            for key, value in loss_dict.items()
        },
        "loss": round(float(loss.detach().cpu()), 8),
    }


def summarize_terratorch_detection_model(task: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    model = getattr(task, "model", None)
    torchvision_model = getattr(model, "torchvision_model", None)
    summary["model_type"] = (
        f"{type(model).__module__}.{type(model).__name__}"
        if model is not None
        else None
    )
    summary["torchvision_model_type"] = (
        f"{type(torchvision_model).__module__}.{type(torchvision_model).__name__}"
        if torchvision_model is not None
        else None
    )
    if torchvision_model is None:
        return summary
    backbone = getattr(torchvision_model, "backbone", None)
    summary["backbone_type"] = (
        f"{type(backbone).__module__}.{type(backbone).__name__}"
        if backbone is not None
        else None
    )
    summary["backbone_out_channels"] = compact_value(
        getattr(backbone, "out_channels", None)
    )
    roi_heads = getattr(torchvision_model, "roi_heads", None)
    if roi_heads is not None:
        box_roi_pool = getattr(roi_heads, "box_roi_pool", None)
        mask_roi_pool = getattr(roi_heads, "mask_roi_pool", None)
        summary["box_roi_pool_featmap_names"] = compact_value(
            getattr(box_roi_pool, "featmap_names", None)
        )
        summary["mask_roi_pool_featmap_names"] = compact_value(
            getattr(mask_roi_pool, "featmap_names", None)
        )
    return summary


def configure_proj_environment() -> None:
    conda_prefix = Path(sys.executable).parents[1]
    for candidate in (
        conda_prefix / "share" / "proj",
        conda_prefix / "Library" / "share" / "proj",
    ):
        if (candidate / "proj.db").exists():
            os.environ.setdefault("PROJ_LIB", str(candidate))
            os.environ.setdefault("PROJ_DATA", str(candidate))
            os.environ.setdefault("GDAL_DATA", str(conda_prefix / "share" / "gdal"))
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--dino-checkpoint", type=Path, default=None)
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--image-glob", default="*chip*.tif")
    parser.add_argument("--label-glob", default="*label*.npz")
    parser.add_argument("--image-suffix", default=None)
    parser.add_argument("--label-suffix", default=None)
    parser.add_argument(
        "--band-filter", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6]
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=4)
    parser.add_argument("--max-val-samples", type=int, default=4)
    parser.add_argument("--max-test-samples", type=int, default=4)
    parser.add_argument("--mask-shift", type=int, nargs=2, default=(0, 0))
    parser.add_argument("--ignore-nodata-in-loss", action="store_true")
    parser.add_argument("--nodata-ignore-index", type=int, default=-1)
    parser.add_argument("--normalize-inputs", action="store_true")
    parser.add_argument(
        "--normalization-source", choices=["pretrain"], default="pretrain"
    )
    parser.add_argument(
        "--normalization-modality", choices=["vis-uv", "nac"], default="vis-uv"
    )
    parser.add_argument("--modality-info", type=Path, default=None)
    parser.add_argument("--out-channels", type=int, default=256)
    parser.add_argument(
        "--layers-to-extract", type=int, nargs="+", default=[5, 11, 17, 23]
    )
    parser.add_argument(
        "--output-strides", type=int, nargs="+", default=[8, 16, 32, 64]
    )
    parser.add_argument(
        "--backbone-return-format",
        choices=["list", "ordered_dict"],
        default="ordered_dict",
        help="Feature container returned by the registered TerraTorch wrapper.",
    )
    parser.add_argument(
        "--backbone-feature-names",
        nargs="+",
        default=None,
        help=(
            "Optional feature names for the registered OrderedDict output. "
            "Use this to match TerraTorch ROI pool featmap_names."
        ),
    )
    parser.add_argument(
        "--anchor-sizes",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64],
        help="One anchor size per feature level.",
    )
    parser.add_argument(
        "--anchor-aspect-ratios", type=float, nargs="+", default=[0.5, 1.0, 2.0]
    )
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--run-backward", action="store_true")
    parser.add_argument("--skip-terratorch-task", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_proj_environment()
    args.device = torch.device(args.device)
    args.anchor_sizes = [[int(size)] for size in args.anchor_sizes]

    result: dict[str, Any] = {
        "backbone_name": BACKBONE_NAME,
        "data_root": str(args.data_root),
        "device": str(args.device),
        "checks": {},
    }

    datamodule = make_datamodule(args)
    batch = next(iter(datamodule.train_dataloader()))
    weight_assignments = list(datamodule.weight_assignments)
    result["datamodule"] = {
        "type": f"{type(datamodule).__module__}.{type(datamodule).__name__}",
        "weight_assignments": weight_assignments,
        "batch": summarize_batch(batch),
    }

    backbone_holder: dict[str, Any] = {}

    def check_registry_build() -> dict[str, Any]:
        backbone = build_registered_backbone(args, weight_assignments).to(args.device)
        backbone_holder["backbone"] = backbone
        return {
            "type": f"{type(backbone).__module__}.{type(backbone).__name__}",
            "out_channels": compact_value(backbone.out_channels),
            "inner_backbone_type": (
                f"{type(backbone.backbone).__module__}.{type(backbone.backbone).__name__}"
                if hasattr(backbone, "backbone")
                else None
            ),
            "inner_out_channels": (
                compact_value(backbone.backbone.out_channels)
                if hasattr(backbone, "backbone")
                else None
            ),
            "return_format": getattr(backbone, "return_format", None),
            "feature_names": compact_value(getattr(backbone, "feature_names", None)),
        }

    result["checks"]["registry_build"] = safe_step(
        "registry_build", check_registry_build
    )

    def check_backbone_forward() -> dict[str, Any]:
        backbone = backbone_holder.get("backbone")
        if backbone is None:
            backbone = build_registered_backbone(args, weight_assignments).to(
                args.device
            )
            backbone_holder["backbone"] = backbone
        backbone.eval()
        images = batch["image"].to(args.device)
        with torch.no_grad():
            features = backbone(images)
        return summarize_features(features, list(images.shape))

    result["checks"]["backbone_forward"] = safe_step(
        "backbone_forward",
        check_backbone_forward,
    )

    def check_torchvision_loss() -> dict[str, Any]:
        backbone = unwrap_torchvision_backbone(
            build_registered_backbone(args, weight_assignments)
        )
        return run_torchvision_loss_step(args, backbone, batch)

    result["checks"]["torchvision_mask_rcnn_loss"] = safe_step(
        "torchvision_mask_rcnn_loss",
        check_torchvision_loss,
    )

    if not args.skip_terratorch_task:
        task_holder: dict[str, Any] = {}

        def check_terratorch_task_build() -> dict[str, Any]:
            task = build_terratorch_task(args, batch, weight_assignments)
            task_holder["task"] = task
            return {
                "task_type": f"{type(task).__module__}.{type(task).__name__}",
                "model_summary": summarize_terratorch_detection_model(task),
            }

        result["checks"]["terratorch_object_detection_task_build"] = safe_step(
            "terratorch_object_detection_task_build",
            check_terratorch_task_build,
        )

        def check_terratorch_task_loss() -> dict[str, Any]:
            task = task_holder.get("task")
            if task is None:
                task = build_terratorch_task(args, batch, weight_assignments)
            return run_terratorch_task_loss_step(args, task, batch)

        result["checks"]["terratorch_object_detection_task_loss"] = safe_step(
            "terratorch_object_detection_task_loss",
            check_terratorch_task_loss,
        )

    output_json = args.output_json
    if output_json is None:
        output_json = Path("scripts/outputs/terratorch_dino_registration_smoke.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(
        f"Wrote TerraTorch DINO registration smoke output to {output_json}", flush=True
    )

    required_checks = [
        result["checks"]["registry_build"]["ok"],
        result["checks"]["backbone_forward"]["ok"],
        result["checks"]["torchvision_mask_rcnn_loss"]["ok"],
    ]
    if args.strict and not all(required_checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
