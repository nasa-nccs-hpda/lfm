"""Shared split-level COCO mask AP and Mask R-CNN validation inference rules.

Metrics consume independent detections, never flattened plotting rasters. CPU
state stores COCO run-length encoded masks and is discarded after each epoch.
"""

from __future__ import annotations

import io
from contextlib import contextmanager, redirect_stdout
from typing import Any

import numpy as np
import torch

MASK_AP_KEY = "val_mask_ap"
MAX_DETECTIONS = 1000
MASK_AP_PROTOCOL = {
    "metric_key": MASK_AP_KEY,
    "iou_thresholds": [0.5 + 0.05 * i for i in range(10)],
    "recall_thresholds": [i / 100 for i in range(101)],
    "max_detection_thresholds": [1, 10, MAX_DETECTIONS],
    "mask_threshold": 0.5,
    "box_score_threshold": 0.0,
    "box_nms_threshold": 0.5,
    "rpn_pre_nms_top_n_test": 1000,
    "rpn_post_nms_top_n_test": 1000,
    "rpn_nms_threshold": 0.7,
    "aggregation": "whole_split_coco_segm_all_areas_single_crater_class",
    "no_targets_ap": -1.0,
}


def _masks(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    value = value.detach().cpu()
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 3 or value.shape[-2:] != valid.shape:
        raise ValueError(
            "Mask AP requires independent (N,H,W) masks on the valid grid."
        )
    return (value > MASK_AP_PROTOCOL["mask_threshold"]) & valid


class InstanceMaskAP:
    """Accumulate every eligible chip, then compute once over the full split."""

    def __init__(self) -> None:
        self.images: list[dict[str, Any]] = []
        self.predictions: list[dict[str, Any]] = []
        self.targets: list[dict[str, Any]] = []
        self.num_chips = 0

    @staticmethod
    def _append_annotations(destination, masks, image_id, scores=None):
        from pycocotools import mask as coco_mask

        for index, mask in enumerate(masks):
            encoded = coco_mask.encode(np.asfortranarray(mask.numpy(), dtype=np.uint8))
            annotation = {
                "id": len(destination) + 1,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": encoded,
                "area": float(coco_mask.area(encoded)),
                "iscrowd": 0,
            }
            if scores is not None:
                annotation["score"] = float(scores[index])
            destination.append(annotation)

    def update(
        self, predictions: list[dict[str, torch.Tensor]], batch: dict[str, Any]
    ) -> None:
        if len(predictions) != len(batch["masks"]):
            raise ValueError(
                "Prediction count differs from validation batch membership."
            )
        for index, prediction in enumerate(predictions):
            valid = batch["valid_mask"][index].detach().cpu().bool()
            if not valid.any():
                raise ValueError(
                    "Wholly invalid chips must be skipped by the eligible loader."
                )
            masks = _masks(prediction["masks"], valid)
            scores = prediction["scores"].detach().cpu().float()
            labels = prediction["labels"].detach().cpu().long()
            if scores.shape != (len(masks),) or labels.shape != scores.shape:
                raise ValueError(
                    "Every predicted mask requires an independent score and label."
                )
            if not torch.isfinite(scores).all():
                raise ValueError("Mask AP requires finite confidence scores.")
            keep = masks.flatten(1).any(1) & (labels == 1)
            # Stable ranking also defines the common limit for external detections.
            order = scores[keep].argsort(descending=True, stable=True)[:MAX_DETECTIONS]
            image_id = self.num_chips + 1
            self.images.append(
                {"id": image_id, "height": valid.shape[0], "width": valid.shape[1]}
            )
            self._append_annotations(
                self.predictions, masks[keep][order], image_id, scores[keep][order]
            )
            masks = _masks(batch["masks"][index], valid)
            labels = batch["labels"][index].detach().cpu().long()
            keep = masks.flatten(1).any(1) & (labels == 1)
            self._append_annotations(self.targets, masks[keep], image_id)
            self.num_chips += 1

    def compute(self) -> torch.Tensor:
        if self.num_chips == 0:
            raise RuntimeError("No eligible validation chips reached mask AP.")
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        with redirect_stdout(io.StringIO()):
            gt, detections = COCO(), COCO()
            for dataset, annotations in (
                (gt, self.targets),
                (detections, self.predictions),
            ):
                dataset.dataset = {
                    "images": self.images,
                    "categories": [{"id": 1, "name": "crater"}],
                    "annotations": annotations,
                }
                dataset.createIndex()
            evaluation = COCOeval(gt, detections, iouType="segm")
            evaluation.params.iouThrs = np.array(MASK_AP_PROTOCOL["iou_thresholds"])
            evaluation.params.recThrs = np.array(MASK_AP_PROTOCOL["recall_thresholds"])
            evaluation.params.maxDets = MASK_AP_PROTOCOL["max_detection_thresholds"]
            evaluation.params.areaRng = [[0, float("inf")]]
            evaluation.params.areaRngLbl = ["all"]
            evaluation.evaluate()
            evaluation.accumulate()
        # COCO's summarize() hardcodes maxDets=100 for aggregate AP. Read its
        # precision tensor directly to honor our dense-crater detection limit.
        precision = evaluation.eval["precision"][:, :, :, 0, -1]
        defined = precision[precision > -1]
        return torch.tensor(
            float(defined.mean()) if defined.size else MASK_AP_PROTOCOL["no_targets_ap"]
        )


@contextmanager
def mask_ap_inference(model):
    """Apply common detector limits before inference, restoring plotting settings."""
    detector = getattr(model, "torchvision_model", model)
    heads, rpn = detector.roi_heads, detector.rpn
    saved = (
        heads.score_thresh,
        heads.nms_thresh,
        heads.detections_per_img,
        rpn._pre_nms_top_n["testing"],
        rpn._post_nms_top_n["testing"],
        rpn.nms_thresh,
    )
    heads.score_thresh = MASK_AP_PROTOCOL["box_score_threshold"]
    heads.nms_thresh = MASK_AP_PROTOCOL["box_nms_threshold"]
    heads.detections_per_img = MAX_DETECTIONS
    rpn._pre_nms_top_n["testing"] = MASK_AP_PROTOCOL["rpn_pre_nms_top_n_test"]
    rpn._post_nms_top_n["testing"] = MASK_AP_PROTOCOL["rpn_post_nms_top_n_test"]
    rpn.nms_thresh = MASK_AP_PROTOCOL["rpn_nms_threshold"]
    try:
        yield
    finally:
        (
            heads.score_thresh,
            heads.nms_thresh,
            heads.detections_per_img,
            rpn._pre_nms_top_n["testing"],
            rpn._post_nms_top_n["testing"],
            rpn.nms_thresh,
        ) = saved


class MaskAPValidationMixin:
    """Shared lifecycle; intentionally replaces inherited validation metric hooks."""

    def on_validation_epoch_start(self) -> None:
        self._validation_mask_ap = InstanceMaskAP()

    def on_validation_epoch_end(self) -> None:
        evaluator = self._validation_mask_ap
        try:
            self.log(
                MASK_AP_KEY,
                evaluator.compute().to(self.device),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "val_mask_ap_chips",
                float(evaluator.num_chips),
                on_step=False,
                on_epoch=True,
            )
        finally:
            self._validation_mask_ap = None
