"""NoData crosses real data adapters and both downstream Mask R-CNN wrappers."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
import torch
import yaml
from rasterio.transform import from_origin
from torch import nn

from lfm.all_models.inst_seg.data.eligibility import (
    NoEligibleInstanceSamplesError,
    SkippedInstanceChip,
)
from lfm.all_models.inst_seg.data.instance_dataset import minmax_scale_per_band
from lfm.all_models.inst_seg.mask_rcnn_nodata import (
    MASK_NODATA_VALUE,
    nodata_aware_maskrcnn_loss,
)
from lfm.all_models.inst_seg.paired_experiment import (
    build_split_manifest,
    create_experiment_datamodule,
    resolve_paired_experiment,
)
from lfm.full_model.inst_seg.instance_graha_components import (
    make_downstream_object_detection_task_class,
)
from lfm.toy_model.inst_seg.lightning_wrappers.toy_dino_mask_rcnn_lightning import (
    ToyDinoMaskRCNNLightningModule,
)


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _write_pair(root, split, key, raw, mask):
    chips, labels = root / split / "chips", root / split / "labels"
    chips.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        chips / f"{key}_chip.tif",
        "w",
        driver="GTiff",
        height=raw.shape[1],
        width=raw.shape[2],
        count=raw.shape[0],
        dtype="float32",
        compress="deflate",
        predictor=3,
        nodata=-9999,
        transform=from_origin(100, 200, 1, 1),
    ) as dst:
        dst.write(raw)
    np.savez(
        labels / f"{key}_label.npz",
        mask=mask,
        bboxes=np.array([[0, 0, 256, 256]]),
        num_craters=999,
    )


@pytest.fixture(scope="module")
def input_files(tmp_path_factory):
    root = tmp_path_factory.mktemp("instance-nodata")
    data = root / "data"
    raw = (
        20
        + np.arange(70, dtype=np.float32)[:, None, None]
        + np.arange(258, dtype=np.float32)[None, :, None] / 2
        + np.arange(260, dtype=np.float32)[None, None, :] / 4
    )
    raw[0] = 7  # Constant valid bands retain their original values.
    mask = np.zeros((258, 260), dtype=np.int32)
    mask[20:40, 10:40] = 11
    mask[25:30, 25:30] = 22  # This object is wholly inside missing data.
    mask[70:90, 100:120] = 33
    partial = raw.copy()
    partial[5, 20:45, 20:45] = -9999  # Raster metadata sentinel.
    partial[6, 40:45, 40:45] = np.nan
    partial[7, 70:80, 110:120] = np.inf
    partial[69, 60:75, 70:75] = -12345  # Explicit excluded sentinel.
    invalid = np.full_like(raw, -9999)
    invalid_band = raw.copy()
    invalid_band[69] = np.nan
    invalid_crop = invalid.copy()
    invalid_crop[:, 0, :] = raw[:, 0, :]  # Valid pixels only outside the center crop.
    examples = {
        "a_invalid": (invalid, mask),
        "b_partial": (partial, mask),
        "c_invalid_band": (invalid_band, mask),
        "d_empty": (raw, np.zeros_like(mask)),
        "e_invalid_crop": (invalid_crop, mask),
        "f_valid": (raw, mask),
    }
    for split in ("train", "val", "test"):
        for key, (image, label) in examples.items():
            _write_pair(data, split, key, image, label)
    pretrain = root / "pretrain"
    pretrain.mkdir()
    means = np.arange(70, dtype=np.float32) + 100
    stds = np.full(70, 2, dtype=np.float32)
    info = {}
    for modality, start, end in (("vis", 0, 5), ("uv", 5, 7), ("static", 7, 70)):
        info[modality] = {
            "stats": {
                modality: {
                    "mean": means[start:end].tolist(),
                    "std": stds[start:end].tolist(),
                }
            }
        }
    (pretrain / "modality_info.yaml").write_text(yaml.safe_dump(info))
    return data, pretrain, partial, mask, means, stds


def _experiment(input_files, root):
    data, pretrain, *_ = input_files
    return resolve_paired_experiment(
        run_root=root,
        data_root=data,
        graha_pretrain_dir=pretrain,
        max_epochs=1,
        toy_batch_size=2,
        graha_batch_size=2,
        toy_num_workers=0,
        graha_num_workers=0,
        excluded_nodata_values=[-12345],
    )


def test_scaling_uses_valid_raw_values_and_zeros_invalid_or_missing_bands():
    image = torch.tensor(
        [
            [[10.0, 20.0], [float("nan"), -9999.0]],
            [[7.0, 7.0], [float("inf"), 7.0]],
            [[float("nan"), float("inf")], [-9999.0, -9999.0]],
        ]
    )
    invalid = ~torch.isfinite(image) | (image == -9999)
    scaled = minmax_scale_per_band(image, invalid)
    torch.testing.assert_close(
        scaled,
        torch.tensor(
            [
                [[0.0, 1.0], [0.0, 0.0]],
                [[7.0, 7.0], [0.0, 7.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ),
    )
    assert torch.isfinite(minmax_scale_per_band(image)).all()


def test_real_adapters_preserve_validity_scaling_and_clipped_geometry(
    input_files, tmp_path
):
    experiment = _experiment(input_files, tmp_path)
    membership = build_split_manifest(experiment)
    _, _, raw, mask, means, stds = input_files
    valid = np.isfinite(raw) & (raw != -9999) & (raw != -12345)
    expected_image_valid = torch.from_numpy(valid[:, 1:257, 2:258].copy())
    expected_valid = expected_image_valid.all(dim=0)
    expected_mask = torch.from_numpy(mask[1:257, 2:258].copy()).long()
    expected_mask[~expected_valid] = -1
    samples = {}
    for model in ("toy", "graha"):
        dm = create_experiment_datamodule(experiment, membership, model)
        assert not (tmp_path / model / "eligibility").exists()
        sample = dm.train_dataset[1]
        samples[model] = sample
        assert torch.isfinite(sample["image"]).all()
        torch.testing.assert_close(sample["image_valid_mask"], expected_image_valid)
        torch.testing.assert_close(sample["valid_mask"], expected_valid)
        torch.testing.assert_close(sample["mask"], expected_mask)
        assert sample["masks"].shape[0] == sample["num_craters"] == 2
        assert 22 not in sample["mask"].unique()
        assert not sample["masks"][:, ~expected_valid].any()
        torch.testing.assert_close(
            sample["boxes"],
            torch.tensor(
                [
                    [8.0, 19.0, 18.0, 39.0],
                    [98.0, 69.0, 118.0, 89.0],
                ]
            ),
        )
        if "crater_boxes" in sample:
            torch.testing.assert_close(sample["crater_boxes"], sample["boxes"])
        assert not sample["image"][~expected_image_valid].any()
        for index in (0, 2, 4):
            assert isinstance(dm.train_dataset[index], SkippedInstanceChip)
        assert dm.train_dataset[3]["masks"].shape == (0, 256, 256)
    expected_graha = (
        torch.from_numpy(raw[:, 1:257, 2:258].copy())
        - torch.from_numpy(means)[:, None, None]
    ) / torch.from_numpy(stds)[:, None, None]
    expected_graha[~expected_image_valid] = 0
    torch.testing.assert_close(samples["graha"]["image"], expected_graha)
    for band in (0, 5, 6, 7, 69):
        values = raw[band][valid[band]]
        transformed = raw[band].copy()
        if values.max() > values.min():
            transformed = (transformed - values.min()) / (values.max() - values.min())
        transformed[~valid[band]] = 0
        torch.testing.assert_close(
            samples["toy"]["image"][band],
            torch.from_numpy(transformed[1:257, 2:258].copy()),
        )


@pytest.mark.parametrize("model", ["toy", "graha"])
def test_loader_refills_batches_and_records_the_same_skips(
    input_files, tmp_path, model
):
    experiment = _experiment(input_files, tmp_path)
    membership = build_split_manifest(experiment)
    dm = create_experiment_datamodule(experiment, membership, model)
    loader = dm.val_dataloader()
    with pytest.raises(TypeError, match="determined during iteration"):
        len(loader)
    assert loader.batch_count_upper_bound == 3
    batches = list(loader)
    assert [len(batch["image"]) for batch in batches] == [2, 1]
    assert [key for batch in batches for key in batch["chip_id"]] == [
        "val/b_partial_chip.tif",
        "val/d_empty_chip.tif",
        "val/f_valid_chip.tif",
    ]
    assert all(
        batch["image_valid_mask"].shape[1:] == (70, 256, 256) for batch in batches
    )
    report = json.loads((tmp_path / model / "eligibility/val.json").read_text())
    assert report["complete_traversal"]
    assert len(report["eligible_chip_ids"]) == 3
    assert {item["chip_id"] for item in report["skipped_chips"]} == {
        "val/a_invalid_chip.tif",
        "val/c_invalid_band_chip.tif",
        "val/e_invalid_crop_chip.tif",
    }
    assert all(
        item["reason"] == "no_valid_pixels_in_crop" for item in report["skipped_chips"]
    )
    # A later preview must keep the complete eligibility report intact.
    iterator = iter(dm.val_dataloader())
    next(iterator)
    iterator.close()
    assert json.loads((tmp_path / model / "eligibility/val.json").read_text()) == report


@pytest.mark.parametrize("model", ["toy", "graha"])
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_all_invalid_required_split_fails_during_iteration(
    input_files, tmp_path, model, split
):
    experiment = _experiment(input_files, tmp_path)
    membership = build_split_manifest(experiment)
    membership = replace(
        membership,
        splits={
            key: records[:1] if key == split else records
            for key, records in membership.splits.items()
        },
    )
    dm = create_experiment_datamodule(experiment, membership, model)
    method = "val_dataloader" if split == "val" else f"{split}_dataloader"
    with pytest.raises(
        NoEligibleInstanceSamplesError, match=f"required split '{split}'"
    ):
        list(getattr(dm, method)())
    report = json.loads((tmp_path / model / f"eligibility/{split}.json").read_text())
    assert report["complete_traversal"] and not report["eligible_chip_ids"]
    assert len(report["skipped_chips"]) == 1


class _TinyBackbone(nn.Module):
    out_channels = 8

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(70, 8, 3, stride=4, padding=1), nn.ReLU())

    def forward(self, images):
        return {"0": self.layers(images)}


def _tiny_maskrcnn():
    from torchvision.models.detection import MaskRCNN
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
    from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
    from torchvision.models.detection.rpn import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign

    return MaskRCNN(
        _TinyBackbone(),
        num_classes=None,
        min_size=256,
        max_size=256,
        image_mean=[0.0] * 70,
        image_std=[1.0] * 70,
        rpn_anchor_generator=AnchorGenerator(((8, 16, 32),), ((0.5, 1.0, 2.0),)),
        box_roi_pool=MultiScaleRoIAlign(["0"], 7, 2),
        box_head=TwoMLPHead(8 * 7 * 7, 32),
        box_predictor=FastRCNNPredictor(32, 2),
        mask_roi_pool=MultiScaleRoIAlign(["0"], 8, 2),
        mask_head=MaskRCNNHeads(8, [8], 1),
        mask_predictor=MaskRCNNPredictor(8, 8, 2),
        rpn_pre_nms_top_n_train=32,
        rpn_post_nms_top_n_train=16,
        rpn_pre_nms_top_n_test=16,
        rpn_post_nms_top_n_test=8,
        box_batch_size_per_image=16,
        box_detections_per_img=8,
    )


class _DetectionTaskBase(nn.Module):
    """Drive the production GRAHA downstream adapter with a real small detector.

    Pretrained GRAHA/TerraTorch initialization is intentionally outside this CPU test.
    """

    masks_field = "masks"
    boxes_field = "boxes"
    labels_field = "labels"

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, targets=None):
        return self.model(images, targets)


@pytest.mark.parametrize("route", ["toy", "graha"])
@pytest.mark.parametrize("empty_only", [False, True])
def test_real_maskrcnn_forward_backward_for_both_adapters(
    input_files, tmp_path, route, empty_only
):
    from torchvision.models.detection import roi_heads

    torch.manual_seed(42)
    experiment = _experiment(input_files, tmp_path)
    dm = create_experiment_datamodule(
        experiment, build_split_manifest(experiment), route
    )
    batch = next(iter(dm.val_dataloader()))  # Partial NoData + empty-instance chip.
    if empty_only:
        batch = {key: value[1:] for key, value in batch.items()}
    model = _tiny_maskrcnn()
    original_loss = roi_heads.maskrcnn_loss
    if route == "toy":
        task = ToyDinoMaskRCNNLightningModule(
            model, ignore_nodata_in_loss=True, max_epochs=1
        )
        images, targets = task._images_and_targets(batch)
    else:
        task_cls = make_downstream_object_detection_task_class(_DetectionTaskBase)
        task = task_cls(model, ignore_nodata_in_loss=True)
        images = list(batch["image"])
        targets = task.reformat_batch(batch, batch_size=len(images))
    assert (targets[0]["masks"][:, ~batch["valid_mask"][0]] == MASK_NODATA_VALUE).all()
    assert targets[-1]["masks"].shape[0] == 0
    logits_seen = []

    def capture_logits(module, inputs, output):
        output.retain_grad()
        logits_seen.append(output)

    hook = model.roi_heads.mask_predictor.register_forward_hook(capture_logits)
    losses = task(images, targets)
    assert "loss_mask" in losses and all(
        torch.isfinite(value) for value in losses.values()
    )
    sum(losses.values()).backward()
    hook.remove()
    assert roi_heads.maskrcnn_loss is original_loss
    assert model.backbone.layers[0].weight.grad is not None
    assert torch.isfinite(model.backbone.layers[0].weight.grad).all()
    assert logits_seen and torch.isfinite(logits_seen[0].grad).all()
    # Eval targets remain binary and inference also completes on both input policies.
    task.eval()
    with torch.no_grad():
        predictions = task(images)
    assert len(predictions) == len(images)


def test_actual_roi_mask_loss_has_zero_gradient_for_invalid_bins():
    from torchvision.models.detection.roi_heads import project_masks_on_boxes

    masks = torch.ones(1, 8, 8, dtype=torch.uint8)
    masks[:, :4, :4] = MASK_NODATA_VALUE
    boxes = [torch.tensor([[0.0, 0.0, 8.0, 8.0]])]
    matched = [torch.tensor([0])]
    logits = torch.zeros(1, 2, 8, 8, requires_grad=True)
    loss = nodata_aware_maskrcnn_loss(
        logits, boxes, [masks], [torch.tensor([1])], matched
    )
    loss.backward()
    validity = project_masks_on_boxes(
        (masks != MASK_NODATA_VALUE).to(torch.uint8), boxes[0], matched[0], 8
    )
    invalid = validity == 0
    assert invalid.any()
    assert not logits.grad[:, 1][invalid].any()
    assert logits.grad[:, 1][~invalid].abs().sum() > 0


def test_loader_worker_skips_are_visible_in_parent_process(tmp_path):
    from lfm.toy_model.inst_seg.lightning_wrappers.toy_dino_mask_rcnn_datamodule import (
        ToyDinoMaskRCNNSplitDataModule,
    )

    raw = np.arange(2 * 40 * 40, dtype=np.float32).reshape(2, 40, 40)
    mask = np.zeros((40, 40), dtype=np.int32)
    for split in ("train", "val", "test"):
        for key, image in (
            ("a_invalid", np.full_like(raw, -9999)),
            ("b_valid", raw),
            ("c_valid", raw),
        ):
            _write_pair(tmp_path / "data", split, key, image, mask)
    dm = ToyDinoMaskRCNNSplitDataModule(
        tmp_path / "data",
        batch_size=2,
        num_workers=2,
        target_size=32,
        image_glob="*.tif",
        label_glob="*_label.npz",
        band_filter=[0, 1],
        ignore_nodata_in_loss=True,
        skip_invalid_chips=True,
        skip_report_dir=tmp_path / "eligibility",
        pin_memory=False,
    )
    dm.setup("fit")
    loader = dm.val_dataloader()
    batches = list(loader)
    assert len(batches) == 1 and len(batches[0]["image"]) == 2
    assert set(loader.skipped_chips) == {"val/a_invalid_chip.tif"}
    report = json.loads((tmp_path / "eligibility/val.json").read_text())
    assert report["skipped_chips"][0]["chip_id"] == "val/a_invalid_chip.tif"
    assert report["complete_traversal"]


def test_cache_roundtrip_preserves_common_and_per_band_validity(input_files, tmp_path):
    from lfm.all_models.inst_seg.prediction.instance_prediction_cache import (
        _load_instance_prediction_cache,
        save_graha_instance_prediction_cache,
    )

    class FixedPredictions(nn.Module):
        device = torch.device("cpu")

        def predict_step(self, batch, batch_idx):
            return [
                {
                    "masks": torch.ones(1, 1, *image.shape[-2:]),
                    "scores": torch.tensor([0.9]),
                    "boxes": torch.tensor([[0.0, 0.0, 256.0, 256.0]]),
                }
                for image in batch["image"]
            ]

    experiment = _experiment(input_files, tmp_path)
    dm = create_experiment_datamodule(
        experiment, build_split_manifest(experiment), "toy"
    )
    expected = dm.val_dataset[1]
    directory = save_graha_instance_prediction_cache(
        FixedPredictions(),
        dm,
        tmp_path,
        model_name="toy",
        n_samples=1,
        setup_datamodule=False,
    )
    cached = next(iter(_load_instance_prediction_cache(directory).values()))
    np.testing.assert_array_equal(cached["valid_mask"], expected["valid_mask"].numpy())
    np.testing.assert_array_equal(
        cached["image_valid_mask"], expected["image_valid_mask"].numpy()
    )
    assert cached["chip_id"] == expected["chip_id"]
    assert (cached["gt_mask"][~cached["valid_mask"]] == -1).all()
    assert (cached["pred_mask"][~cached["valid_mask"]] == 0).all()


def test_lightning_epoch_consumes_all_eligible_batches(input_files, tmp_path):
    from lightning.pytorch import Callback, Trainer

    class SeenChips(Callback):
        def __init__(self):
            self.train = []
            self.val = []

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            self.train.extend(batch["chip_id"])

        def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx):
            self.val.extend(batch["chip_id"])

    experiment = _experiment(input_files, tmp_path)
    dm = create_experiment_datamodule(
        experiment, build_split_manifest(experiment), "toy"
    )
    task = ToyDinoMaskRCNNLightningModule(
        _tiny_maskrcnn(), ignore_nodata_in_loss=True, max_epochs=1
    )
    seen = SeenChips()
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[seen],
    )
    trainer.fit(task, datamodule=dm)
    assert len(seen.train) == len(seen.val) == 3
    assert len(set(seen.train)) == len(set(seen.val)) == 3
    assert trainer.global_step == 2
    assert torch.isfinite(trainer.callback_metrics["val_loss"])


def test_graha_scheduler_records_nominal_budget_without_a_pixel_scan(
    input_files, tmp_path
):
    experiment = _experiment(input_files, tmp_path)
    dm = create_experiment_datamodule(
        experiment, build_split_manifest(experiment), "graha"
    )
    loader = dm.train_dataloader()

    class OptimizerTaskBase(nn.Module):
        def __init__(self):
            super().__init__()
            self.cosine_t_max = None
            self.warmup_steps = 1
            self.trainer = SimpleNamespace(
                estimated_stepping_batches=float("inf"),
                train_dataloader=loader,
                accumulate_grad_batches=2,
                max_epochs=3,
            )
            self.resolved = {}

        def save_hyperparameters(self, values):
            self.resolved.update(values)

        def configure_optimizers(self):
            return self.cosine_t_max

    task = make_downstream_object_detection_task_class(OptimizerTaskBase)(
        ignore_nodata_in_loss=True
    )
    assert (
        task.configure_optimizers() == 5
    )  # ceil(3 nominal batches / 2) * 3 epochs - warmup.
    assert task.resolved["cosine_t_max"] == 5
    assert not (tmp_path / "graha/eligibility").exists()
    task.cosine_t_max = 123
    assert task.configure_optimizers() == 123
