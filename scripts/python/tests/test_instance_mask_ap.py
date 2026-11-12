"""COCO mask AP contract and complete epoch-one validation for both routes."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn

from lfm.all_models.inst_seg.mask_ap import (
    MASK_AP_KEY,
    MAX_DETECTIONS,
    InstanceMaskAP,
    mask_ap_inference,
)
from lfm.all_models.inst_seg.paired_experiment import (
    PairedInstanceExperiment,
    build_split_manifest,
    create_experiment_datamodule,
)
from lfm.full_model.inst_seg.instance_graha_components import (
    make_downstream_object_detection_task_class,
)
from lfm.toy_model.inst_seg.lightning_wrappers.toy_dino_mask_rcnn_lightning import (
    ToyDinoMaskRCNNLightningModule,
)
from scripts.python.tests import test_instance_nodata_training as nodata

input_files = nodata.input_files
cpu_threads = nodata.cpu_threads
_experiment = nodata._experiment
_tiny_maskrcnn = nodata._tiny_maskrcnn


def _prediction(masks, scores=None):
    return {
        "masks": masks[:, None].float(),
        "scores": torch.tensor(scores if scores is not None else [0.9] * len(masks)),
        "labels": torch.ones(len(masks), dtype=torch.long),
    }


def _batch(masks, valid=None):
    return {
        "masks": [masks],
        "labels": [torch.ones(len(masks), dtype=torch.long)],
        "valid_mask": [
            torch.ones(masks.shape[-2:], dtype=torch.bool) if valid is None else valid
        ],
    }


@pytest.fixture
def objects():
    masks = torch.zeros(2, 16, 16, dtype=torch.bool)
    masks[0, 2:10, 2:10] = True
    masks[1, 6:14, 6:14] = True
    return masks


def test_perfect_overlapping_masks_and_low_scores(objects):
    assert (objects[0] & objects[1]).any()
    evaluator = InstanceMaskAP()
    evaluator.update([_prediction(objects, [0.002, 0.001])], _batch(objects))
    assert evaluator.compute().item() == pytest.approx(1)


def test_high_confidence_false_positive_and_missed_instance(objects):
    evaluator = InstanceMaskAP()
    fp = torch.zeros_like(objects[:1])
    fp[:, 0, 14:16] = True
    evaluator.update(
        [_prediction(torch.cat([fp, objects[:1]]), [0.99, 0.8])], _batch(objects[:1])
    )
    assert evaluator.compute().item() == pytest.approx(0.5)
    evaluator = InstanceMaskAP()
    evaluator.update([_prediction(objects[:1])], _batch(objects))
    assert evaluator.compute().item() == pytest.approx(51 / 101)


def test_empty_predictions_targets_and_split_level_aggregation(objects):
    empty = objects[:0]
    evaluator = InstanceMaskAP()
    evaluator.update([_prediction(empty)], _batch(objects))
    assert evaluator.compute().item() == 0
    evaluator = InstanceMaskAP()
    # False positive on a negative chip must contribute to the whole-split curve.
    evaluator.update([_prediction(objects[:1], [0.99])], _batch(empty))
    evaluator.update([_prediction(objects, [0.8, 0.7])], _batch(objects))
    evaluator.update([_prediction(empty)], _batch(empty))
    assert evaluator.num_chips == 3
    assert evaluator.compute().item() == pytest.approx(2 / 3)
    for prediction in (_prediction(empty), _prediction(objects)):
        evaluator = InstanceMaskAP()
        evaluator.update([prediction], _batch(empty))
        assert evaluator.compute().item() == -1  # COCO: AP undefined without GT.


def test_nodata_clips_both_sides_and_drops_wholly_invalid_instances(objects):
    valid = torch.ones(16, 16, dtype=torch.bool)
    valid[:, :6] = False
    target = objects[:1].clone()
    prediction = target.clone()
    prediction[:, :, :6] = True  # Arbitrary errors outside the common footprint.
    invalid_only = torch.zeros_like(target)
    invalid_only[:, :, :6] = True
    evaluator = InstanceMaskAP()
    evaluator.update(
        [_prediction(torch.cat([invalid_only, prediction]), [0.99, 0.01])],
        _batch(torch.cat([target, invalid_only]), valid),
    )
    assert evaluator.compute().item() == pytest.approx(1)
    # Fully clipped GT and predictions still represent an eligible negative chip.
    evaluator.update([_prediction(invalid_only)], _batch(invalid_only, valid))
    assert evaluator.compute().item() == pytest.approx(1)


def test_dense_perfect_detections_are_not_capped_at_100():
    masks = torch.eye(121, dtype=torch.bool).reshape(121, 11, 11)
    evaluator = InstanceMaskAP()
    evaluator.update([_prediction(masks)], _batch(masks))
    assert evaluator.compute().item() == pytest.approx(1)


def test_aggregate_uses_all_ten_iou_thresholds():
    target = torch.ones(1, 2, 4, dtype=torch.bool)
    prediction = target.clone()
    prediction.flatten()[5:] = False  # IoU 0.625 passes 0.50, 0.55, and 0.60.
    evaluator = InstanceMaskAP()
    evaluator.update([_prediction(prediction)], _batch(target))
    assert evaluator.compute().item() == pytest.approx(0.3)


def test_common_limit_keeps_highest_scored_detections(objects):
    # An extra, lowest-scored true positive must fall outside the common cap.
    false_positives = objects[1:].repeat(MAX_DETECTIONS, 1, 1)
    predictions = torch.cat([objects[:1], false_positives])
    evaluator = InstanceMaskAP()
    evaluator.update(
        [_prediction(predictions, [0.001] + [0.9] * MAX_DETECTIONS)],
        _batch(objects[:1]),
    )
    assert len(evaluator.predictions) == MAX_DETECTIONS
    assert evaluator.compute().item() == 0


def test_common_detector_settings_restore_even_on_failure():
    detector = _tiny_maskrcnn()
    before = (
        detector.roi_heads.score_thresh,
        detector.roi_heads.detections_per_img,
        detector.rpn._post_nms_top_n.copy(),
    )
    with (
        pytest.raises(RuntimeError, match="inference failed"),
        mask_ap_inference(detector),
    ):
        assert detector.roi_heads.score_thresh == 0
        assert detector.roi_heads.detections_per_img == MAX_DETECTIONS
        assert detector.roi_heads.nms_thresh == 0.5
        assert detector.rpn._post_nms_top_n["testing"] == 1000
        raise RuntimeError("inference failed")
    assert before == (
        detector.roi_heads.score_thresh,
        detector.roi_heads.detections_per_img,
        detector.rpn._post_nms_top_n,
    )


class _TerraTorchModel(nn.Module):
    """Match TerraTorch's model output wrapper around a real small detector."""

    def __init__(self):
        super().__init__()
        self.torchvision_model = _tiny_maskrcnn()

    def forward(self, images, targets=None):
        output = self.torchvision_model(list(images), targets)
        return SimpleNamespace(output=output)


class _LunarTaskBase(LightningModule):
    """Replace pretrained setup, retaining the GRAHA downstream validation route."""

    boxes_field, labels_field, masks_field = "boxes", "labels", "masks"

    def __init__(self):
        super().__init__()
        self.model = _TerraTorchModel()
        self.cosine_t_max = 1

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        targets = self.reformat_batch(batch, len(batch["image"]))
        return sum(self(batch["image"], targets).output.values())

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

    def validation_step(self, *args):
        raise AssertionError("Inherited score-filtered metrics must be bypassed")

    def on_validation_epoch_end(self):
        raise AssertionError("Inherited aggregation must be bypassed")

    def test_step(self, *args):
        raise AssertionError("Held-out test must not run during training")


@pytest.mark.parametrize("route", ["toy", "graha"])
def test_epoch_one_reports_full_eligible_validation_and_checkpoint_metric(
    input_files, tmp_path, route
):
    experiment = _experiment(input_files, tmp_path)
    dm = create_experiment_datamodule(
        experiment, build_split_manifest(experiment), route
    )
    if route == "toy":
        task = ToyDinoMaskRCNNLightningModule(
            _tiny_maskrcnn(), max_epochs=1, ignore_nodata_in_loss=True
        )
    else:
        task = make_downstream_object_detection_task_class(_LunarTaskBase)(
            ignore_nodata_in_loss=True
        )
        assert task.monitor == MASK_AP_KEY
    checkpoint = ModelCheckpoint(
        dirpath=tmp_path / "checkpoints", monitor=MASK_AP_KEY, mode="max"
    )

    class SeenChips(Callback):
        def __init__(self):
            self.chips = []

        def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx):
            self.chips.extend(batch["chip_id"])

    seen = SeenChips()
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[seen, checkpoint],
    )
    trainer.fit(task, datamodule=dm)
    assert set(seen.chips) == {
        "val/b_partial_chip.tif",
        "val/d_empty_chip.tif",
        "val/f_valid_chip.tif",
    }
    assert len(seen.chips) == 3
    assert trainer.callback_metrics["val_mask_ap_chips"].item() == 3
    assert torch.isfinite(trainer.callback_metrics[MASK_AP_KEY])
    assert checkpoint.best_model_path and checkpoint.best_model_score is not None
    assert task._validation_mask_ap is None
    # No test traversal/eligibility report was produced by training.
    assert not (tmp_path / route / "eligibility/test.json").exists()


@pytest.mark.parametrize("route", ["toy", "graha"])
def test_validation_route_is_independent_of_plotting_score_threshold(objects, route):
    if route == "toy":
        task = ToyDinoMaskRCNNLightningModule(_tiny_maskrcnn())
        task._loss_step = lambda *args: torch.tensor(0.0)
        task.forward = lambda *args: [_prediction(objects, [0.002, 0.001])]
    else:
        task = make_downstream_object_detection_task_class(_LunarTaskBase)()
        task.forward = lambda *args: SimpleNamespace(
            output=[_prediction(objects, [0.002, 0.001])]
        )
    batch = {**_batch(objects), "image": torch.zeros(1, 70, 16, 16)}
    for threshold in (0.0, 0.5, 0.999):
        task.score_threshold = threshold
        task.prediction_score_threshold = threshold
        task.eval()
        task.on_validation_epoch_start()
        task.validation_step(batch, 0)
        assert task._validation_mask_ap.compute().item() == pytest.approx(1)


def test_paired_experiment_rejects_epoch_test_evaluation(input_files, tmp_path):
    experiment = _experiment(input_files, tmp_path)
    with pytest.raises(ValueError, match="held-out test"):
        PairedInstanceExperiment(
            replace(experiment.toy, run_epoch_test_suite=True), experiment.graha
        )


def test_gfft_retains_its_existing_validation_hooks():
    from lfm.full_model.inst_seg import instance_gfft_components

    task_cls = instance_gfft_components.make_downstream_object_detection_task_class(
        _LunarTaskBase
    )
    assert task_cls.validation_step is _LunarTaskBase.validation_step
    assert task_cls.on_validation_epoch_end is _LunarTaskBase.on_validation_epoch_end
