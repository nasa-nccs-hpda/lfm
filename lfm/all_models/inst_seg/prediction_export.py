"""Ownership of paired instance prediction artifacts and display membership."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from lfm.all_models.inst_seg.mask_ap import MASK_AP_PROTOCOL
from lfm.all_models.inst_seg.prediction.instance_prediction_cache import (
    _boxes_from_instance_mask,
    _prediction_masks_to_instance_map,
)
from lfm.all_models.instance_comparison_plot import (
    InstanceComparisonChip,
    validate_instance_comparison_sources,
)
from lfm.all_models.plotting import DisplayPolicy, prepare_display
from lfm.all_models.prediction_artifacts import ChipMetadata
from lfm.artifacts import read_json, write_json

DISPLAY_BANDS = [3, 1, 0]
MODELS = ("toy", "graha")


def _rgb(raw: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return prepare_display(
        raw.transpose(1, 2, 0),
        valid=valid,
        policy=DisplayPolicy(bands=(0, 1, 2), stretch="channel_minmax"),
    )


def _sample(batch, index, prediction, chip_id, record, score_threshold):
    valid = batch["valid_mask"][index].cpu().numpy().astype(bool)
    gt = batch["mask"][index].cpu().numpy().astype(np.int32).copy()
    gt[~valid] = -1
    raw = batch["raw_image"][index, DISPLAY_BANDS].cpu().numpy()
    masks = prediction["masks"]
    if masks.ndim == 4:
        masks = masks[:, 0]
    binary = (masks > MASK_AP_PROTOCOL["mask_threshold"]) & torch.from_numpy(valid)
    keep = (prediction["scores"] >= score_threshold) & (prediction["labels"] == 1)
    order = prediction["scores"][keep].argsort(descending=True, stable=True)
    # Paint lower-confidence masks first, keeping stronger detections on top.
    raster = _prediction_masks_to_instance_map(binary[keep][order.flip(0)])
    raster[~valid] = 0
    return {
        "chip_id": np.array(chip_id),
        "image_path": np.array(str(record.image_path)),
        "label_path": np.array(str(record.label_path)),
        "image": raw,
        "image_valid_bits": np.packbits(
            batch["image_valid_mask"][index].cpu().numpy(), axis=-1
        ),
        "valid_mask": valid,
        "gt_mask": gt,
        "pred_mask": raster,
        "pred_boxes": _boxes_from_instance_mask(raster),
        "gt_boxes": batch["boxes"][index].cpu().numpy(),
        # Pack independent masks along width; unpack with count=valid.shape[1].
        "prediction_mask_bits": np.packbits(binary.numpy(), axis=-1),
        "prediction_scores": prediction["scores"].numpy(),
        "prediction_labels": prediction["labels"].numpy(),
        "prediction_boxes": prediction["boxes"].numpy(),
        "target_mask_bits": np.packbits(batch["masks"][index].cpu().numpy(), axis=-1),
        "target_labels": batch["labels"][index].cpu().numpy(),
    }


def _validate_pair(first, second, chip_id):
    validate_instance_comparison_sources(
        {
            model: {
                chip_id: {key: sample[key] for key in ("gt_mask", "image", "pred_mask")}
            }
            for model, sample in (("toy", first), ("graha", second))
        },
        [chip_id],
    )

    def metadata(sample):
        return ChipMetadata(
            str(sample["chip_id"]),
            str(sample["image_path"]),
            str(sample["label_path"]),
            sample["valid_mask"],
        )

    metadata(first).validate_pair(metadata(second))
    if not np.array_equal(first["image_valid_bits"], second["image_valid_bits"]):
        raise ValueError(f"{chip_id}: model exports differ in image_valid_bits.")
    if not np.array_equal(first["image"], second["image"], equal_nan=True):
        raise ValueError(f"{chip_id}: raw RGB crops differ across model exports.")


@dataclass(frozen=True)
class DisplayOptions:
    n_samples: int = 5
    score_threshold: float = 0.5

    def __post_init__(self):
        if type(self.n_samples) is not int or self.n_samples < 1:
            raise ValueError("Display sample count must be a positive integer.")
        if (
            not math.isfinite(self.score_threshold)
            or not 0 <= self.score_threshold <= 1
        ):
            raise ValueError("Display score threshold must be in [0, 1].")


@dataclass(frozen=True)
class SavedPredictions:
    chips: list[InstanceComparisonChip]
    selected_ids: list[str]
    indexes: dict[str, Path]
    encoding: dict
    visualization_caches: dict[str, Path]


class PairedPredictionExport:
    """Own selection before inference, encoding, pairing and complete membership."""

    def __init__(
        self,
        output_dir: Path,
        membership,
        *,
        display: DisplayOptions | None = None,
        width: int = 256,
    ):
        self.output = output_dir
        self.records = {item.chip_id: item for item in membership.splits["test"]}
        self.display, self.width = display or DisplayOptions(), width
        self.selected_ids = None
        self._indexes = {}
        self._active = None
        self._completed = set()
        self._last_chip = {}

    def batches(self, model: str, loader):
        if len(self._indexes) >= len(MODELS) or model != MODELS[len(self._indexes)]:
            raise ValueError("Export Toy then GRAHA exactly once.")
        if self._active is not None:
            raise RuntimeError("Finish one traversal before starting the next.")
        self._active = model
        self._indexes[model] = {}
        (self.output / model / "predictions").mkdir(parents=True)
        iterator = iter(loader)
        buffered = []
        if self.selected_ids is None:
            self.selected_ids = []
            while len(self.selected_ids) < self.display.n_samples:
                batch = next(iterator, None)
                if batch is None:
                    break
                self._check_order(model, batch)
                buffered.append(batch)
                self.selected_ids.extend(
                    batch["chip_id"][: self.display.n_samples - len(self.selected_ids)]
                )
        yield from buffered
        for batch in iterator:
            self._check_order(model, batch)
            yield batch
        self._completed.add(model)
        self._active = None

    def _check_order(self, model: str, batch) -> None:
        ids = batch["chip_id"]
        previous = self._last_chip.get(model)
        if ids and (
            ids != sorted(ids) or (previous is not None and ids[0] <= previous)
        ):
            raise ValueError("Prediction traversal requires sorted unique chip IDs.")
        if ids:
            self._last_chip[model] = ids[-1]

    def write(self, model: str, batch, predictions) -> None:
        if self._active != model:
            raise RuntimeError("Write predictions during the active traversal.")
        if len(predictions) != len(batch["chip_id"]):
            raise ValueError("Prediction count differs from batch membership.")
        exports = self._indexes[model]
        for index, (chip_id, prediction) in enumerate(
            zip(batch["chip_id"], predictions)
        ):
            if chip_id not in self.records or chip_id in exports:
                raise ValueError(f"Unexpected or duplicate test chip: {chip_id}.")
            sample = _sample(
                batch,
                index,
                prediction,
                chip_id,
                self.records[chip_id],
                self.display.score_threshold,
            )
            filename = f"{len(exports):06d}.npz"
            path = self.output / model / "predictions" / filename
            if model == "graha":
                reference_path = self._indexes["toy"].get(chip_id)
                if reference_path is None:
                    raise ValueError(f"{chip_id}: no matching Toy prediction artifact.")
                with np.load(reference_path, allow_pickle=False) as reference:
                    _validate_pair(reference, sample, chip_id)
            np.savez_compressed(path, **sample)
            exports[chip_id] = str(path.resolve())
            if chip_id in self.selected_ids:
                viz_dir = self.output / model / "visualization"
                viz_dir.mkdir(exist_ok=True)
                np.savez_compressed(
                    viz_dir / filename,
                    **{
                        key: sample[key]
                        for key in (
                            "chip_id",
                            "image_path",
                            "image",
                            "valid_mask",
                            "gt_mask",
                            "pred_mask",
                            "pred_boxes",
                            "gt_boxes",
                        )
                    },
                )

    def count(self, model: str) -> int:
        return len(self._indexes[model])

    def finish(self, eligibility_reports: dict[str, Path]) -> SavedPredictions:
        if self._completed != set(MODELS):
            raise ValueError(
                "Both prediction traversals must finish before publication."
            )
        reports = {}
        indexes = {}
        for model in MODELS:
            report = read_json(eligibility_reports[model])
            eligible = set(report["eligible_chip_ids"])
            skipped = {item["chip_id"] for item in report["skipped_chips"]}
            if (
                not report["complete_traversal"]
                or eligible != set(self._indexes[model])
                or eligible & skipped
                or eligible | skipped != set(self.records)
            ):
                raise ValueError(
                    "Test eligibility does not cover the complete split membership."
                )
            reports[model] = report
        if reports["toy"] != reports["graha"]:
            raise ValueError("Toy/GRAHA test eligibility or skip reasons differ.")
        for model in MODELS:
            indexes[model] = self.output / model / "predictions.json"
            write_json(indexes[model], self._indexes[model])
        chips = []
        for chip_id in self.selected_ids:
            with (
                np.load(self._indexes["toy"][chip_id], allow_pickle=False) as first,
                np.load(self._indexes["graha"][chip_id], allow_pickle=False) as second,
            ):
                _validate_pair(first, second, chip_id)
                chips.append(
                    InstanceComparisonChip(
                        key=chip_id,
                        display_image=_rgb(first["image"], first["valid_mask"]),
                        ground_truth=first["gt_mask"],
                        valid_mask=first["valid_mask"],
                        predictions={
                            "toy": first["pred_mask"],
                            "graha": second["pred_mask"],
                        },
                        prediction_boxes={
                            "toy": first["pred_boxes"],
                            "graha": second["pred_boxes"],
                        },
                        ground_truth_boxes=first["gt_boxes"],
                        display_note="Raw crop · RGB bands 3,1,0",
                    )
                )
        return SavedPredictions(
            chips,
            self.selected_ids,
            indexes,
            {
                "masks": "packbits_width_big_endian",
                "width": self.width,
                "threshold": MASK_AP_PROTOCOL["mask_threshold"],
                "overlaps": "independent",
                "validity": "clip_predictions_and_targets_drop_empty_in_shared_evaluator",
            },
            {model: self.output / model / "visualization" for model in MODELS},
        )
