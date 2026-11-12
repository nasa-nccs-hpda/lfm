"""Standalone fresh checkpoint comparison, including actual downstream wrappers."""

import json
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lfm.all_models.inst_seg import comparison, prediction_export, training
from lfm.all_models.inst_seg.mask_ap import InstanceMaskAP
from scripts.python.tests import test_instance_checkpoint_lifecycle as lifecycle
from scripts.python.tests import test_instance_mask_ap as ap
from scripts.python.tests.instance_experiment_support import route_tasks

input_files = ap.input_files
cpu_threads = ap.cpu_threads


def _task(config, dm, model):
    if model == "toy":
        task = ap.ToyDinoMaskRCNNLightningModule(
            ap._tiny_maskrcnn(),
            max_epochs=config.max_epochs,
            ignore_nodata_in_loss=True,
        )
    else:

        class Base(ap._LunarTaskBase):
            configure_optimizers = lifecycle.SmallTask.configure_optimizers

        task = ap.make_downstream_object_detection_task_class(Base)(
            ignore_nodata_in_loss=True
        )
        task.budget = config.max_epochs

    def nonmonotonic_validation():
        task.log(training.MASK_AP_KEY, (0.9, 0.4)[task.current_epoch])
        task._validation_mask_ap = None

    task.on_validation_epoch_end = nonmonotonic_validation
    return task


@pytest.fixture(scope="module")
def trained_pair(input_files, tmp_path_factory):
    root = tmp_path_factory.mktemp("comparison-training") / "run"
    lifecycle._prepare(input_files, root, budget=50)
    with pytest.MonkeyPatch.context() as patch:
        route_tasks(patch, _task)
        for model in comparison.MODELS:
            training.train_prepared_experiment(root, model, max_epochs=2)
    return root


def _compare(root, output, **kwargs):
    return comparison.compare_checkpoints(
        root / "toy/result.json", root / "graha/result.json", output, **kwargs
    )


def _replay_metrics(index):
    evaluator = InstanceMaskAP()
    for filename in json.loads(index.read_text()).values():
        with np.load(filename, allow_pickle=False) as sample:
            width = sample["valid_mask"].shape[1]

            def unpack(key, width=width):
                return torch.from_numpy(
                    np.unpackbits(sample[key], axis=-1, count=width)
                )

            evaluator.update(
                [
                    {
                        "masks": unpack("prediction_mask_bits"),
                        "scores": torch.from_numpy(sample["prediction_scores"]),
                        "labels": torch.from_numpy(sample["prediction_labels"]),
                    }
                ],
                {
                    "masks": [unpack("target_mask_bits")],
                    "labels": [torch.from_numpy(sample["target_labels"])],
                    "valid_mask": [torch.from_numpy(sample["valid_mask"])],
                },
            )
    return evaluator.num_chips, evaluator.compute().item()


def test_full_test_fresh_loads_independent_normalization_and_four_rows(
    trained_pair, input_files, tmp_path, monkeypatch, capsys
):
    loaded, forwarded, weak_tasks = [], {}, []
    actual_load = torch.load
    expected_ids = [
        "test/b_partial_chip.tif",
        "test/d_empty_chip.tif",
        "test/f_valid_chip.tif",
    ]

    def tracked_load(path, *args, **kwargs):
        loaded.append(str(path))
        return actual_load(path, *args, **kwargs)

    def builder(config, dm, model):
        assert config.max_epochs == 2  # Recorded override, not prepared budget 50.
        assert not weak_tasks or weak_tasks[-1]() is None
        task = _task(config, dm, model)
        del task.on_validation_epoch_end  # Discard training-only closure.
        weak_tasks.append(weakref.ref(task))
        forward = task.forward

        def inference(images):
            assert not task.training and not torch.is_grad_enabled()
            assert task.model.training is False
            detector = getattr(task.model, "torchvision_model", task.model)
            assert detector.roi_heads.score_thresh == 0
            assert detector.roi_heads.detections_per_img == 1000
            assert detector.rpn._post_nms_top_n["testing"] == 1000
            forwarded[model] = images.detach().cpu().clone()
            return forward(images)

        task.forward = inference
        return task

    rendered = []
    render = comparison.plot_instance_comparison

    def plot(chips, output_path, **kwargs):
        assert [chip.key for chip in chips] == expected_ids[:2]
        assert list(kwargs["model_labels"]) == ["toy", "graha"]
        assert "Test split" in kwargs["title"]
        assert "epoch 0" in kwargs["model_labels"]["toy"]
        assert all(set(chip.predictions) == {"toy", "graha"} for chip in chips)
        assert not chips[0].valid_mask.all()
        assert not chips[1].ground_truth.any()
        _, _, raw, *_ = input_files
        valid = chips[0].valid_mask
        crop = raw[[3, 1, 0], 1:257, 2:258]
        np.testing.assert_allclose(
            chips[0].display_image, prediction_export._rgb(crop, valid)
        )
        rendered.append(chips)
        return render(chips, output_path, **kwargs)

    monkeypatch.setattr(comparison.torch, "load", tracked_load)
    route_tasks(monkeypatch, builder)
    monkeypatch.setattr(comparison, "plot_instance_comparison", plot)
    output = tmp_path / "comparison"
    manifest = json.loads(_compare(trained_pair, output, n_samples=2).read_text())
    assert rendered
    assert (output / "test_comparison.png").stat().st_size > 0
    assert manifest["display"]["chip_ids"] == expected_ids[:2]
    assert manifest["display"]["stored_bands"] == [3, 1, 0]
    assert manifest["inference"]["batch_size"] == 32
    assert manifest["status"] == "complete"
    for model in comparison.MODELS:
        result = json.loads((trained_pair / model / "result.json").read_text())
        assert result["best_checkpoint"]["epoch"] == 0
        assert result["latest_checkpoint"]["epoch"] == 1
        assert loaded.count(result["best_checkpoint"]["path"]) == 1
        assert forwarded[model].shape == (3, 70, 256, 256)
        count, score = _replay_metrics(output / model / "predictions.json")
        assert count == manifest["metrics"][model]["eligible_chips"] == 3
        assert score == pytest.approx(manifest["metrics"][model]["test_mask_ap"])
        assert len(list((output / model / "visualization").glob("*.npz"))) == 2
        report = json.loads((output / model / "eligibility/test.json").read_text())
        assert report["eligible_chip_ids"] == expected_ids
        assert len(report["skipped_chips"]) == 3
        assert not (trained_pair / model / "eligibility/test.json").exists()
    assert not torch.equal(forwarded["toy"], forwarded["graha"])
    assert (
        torch.isfinite(forwarded["toy"]).all()
        and torch.isfinite(forwarded["graha"]).all()
    )
    torch.testing.assert_close(forwarded["toy"][:, 0], torch.full((3, 256, 256), 7.0))
    torch.testing.assert_close(
        forwarded["graha"][:, 0], torch.full((3, 256, 256), (7.0 - 100) / 2)
    )
    progress = capsys.readouterr().out
    for model in comparison.MODELS:
        assert f"{model} test inference; batches=1;" in progress
        assert f"{model} test evaluation complete; mask_AP=" in progress
    assert "Rendering four-row test comparison" in progress
    assert "Comparison figure saved:" in progress
    assert "Evaluation and plot pipeline complete;" in progress


def test_rerun_without_training_or_training_cache_source(
    trained_pair, tmp_path, monkeypatch
):
    route_tasks(monkeypatch, _task)
    for name, threshold in (("first", 0.0), ("second", 1.0)):
        output = tmp_path / name
        manifest = json.loads(
            _compare(
                trained_pair, output, batch_size=1, display_score_threshold=threshold
            ).read_text()
        )
        assert len(manifest["display"]["chip_ids"]) == 3
        assert manifest["display"]["requested_count"] == 5
    first = json.loads((tmp_path / "first/metrics.json").read_text())
    second = json.loads((tmp_path / "second/metrics.json").read_text())
    assert first == second  # Plotting confidence never truncates AP input.
    with pytest.raises(FileExistsError):
        _compare(trained_pair, tmp_path / "first")


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "running"),
        ("architecture", "wrong"),
        ("split_manifest_id", "stale"),
        ("config_id", "stale"),
        ("completed_epochs", 1),
        ("normalization", {}),
        ("configuration", {}),
        ("selection", {}),
        ("validity_policy", {}),
    ],
)
def test_rejects_stale_or_mismatched_result(trained_pair, tmp_path, field, value):
    path = trained_pair / "toy/result.json"
    original = path.read_text()
    try:
        payload = json.loads(original)
        payload[field] = value
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="Mismatched or stale"):
            _compare(trained_pair, tmp_path / "output")
    finally:
        path.write_text(original)
    assert not (tmp_path / "output").exists()


def test_rejects_best_replaced_by_latest_and_strict_wrong_weights(
    trained_pair, tmp_path, monkeypatch
):
    result, _ = training.load_completed_training(
        trained_pair / "toy/result.json", "toy"
    )
    best = Path(result.payload["best_checkpoint"]["path"])
    original = best.read_bytes()
    route_tasks(monkeypatch, _task)
    try:
        best.write_bytes(Path(result.payload["latest_checkpoint"]["path"]).read_bytes())
        with pytest.raises(ValueError, match="provenance"):
            _compare(trained_pair, tmp_path / "latest")
        best.write_bytes(original)
        state = torch.load(best, map_location="cpu", weights_only=False)
        state["state_dict"]["unexpected"] = torch.zeros(1)
        torch.save(state, best)
        with pytest.raises(RuntimeError, match="Unexpected key"):
            _compare(trained_pair, tmp_path / "weights")
        assert not (tmp_path / "weights/comparison.json").exists()
    finally:
        best.write_bytes(original)


def test_rejects_mixed_experiments(trained_pair, input_files, tmp_path, monkeypatch):
    other = tmp_path / "other"
    lifecycle._prepare(input_files, other, budget=50)
    route_tasks(monkeypatch, _task)
    training.train_prepared_experiment(other, "graha", max_epochs=2)
    with pytest.raises(ValueError, match="mixed experiments"):
        comparison.compare_checkpoints(
            trained_pair / "toy/result.json",
            other / "graha/result.json",
            tmp_path / "output",
        )
    assert not (tmp_path / "output").exists()


def test_parser_defaults_and_invalid_settings():
    parser = comparison.create_comparison_parser()
    base = [
        "--toy-result",
        "toy/result.json",
        "--graha-result",
        "graha/result.json",
        "--output-dir",
        "comparison",
    ]
    args = parser.parse_args(base)
    assert args.batch_size == 32 and args.n_samples == 5
    for option in ("--batch-size", "--n-samples"):
        with pytest.raises(SystemExit):
            parser.parse_args(base + [option, "0"])
    for kwargs in (
        {"batch_size": 0},
        {"n_samples": False},
        {"display_score_threshold": float("nan")},
    ):
        with pytest.raises(ValueError):
            comparison.compare_checkpoints("unused", "unused", "unused", **kwargs)


@pytest.mark.parametrize(
    "case",
    [
        "missing_field",
        "duplicate_key",
        "missing_checkpoint",
        "ambiguous_callback",
        "stale_selection",
    ],
)
def test_rejects_incomplete_missing_ambiguous_and_stale_artifacts(trained_pair, case):
    path = trained_pair / "toy/result.json"
    original = path.read_text()
    payload = json.loads(original)
    best = Path(payload["best_checkpoint"]["path"])
    checkpoint_bytes = best.read_bytes()
    try:
        if case == "missing_field":
            payload.pop("initialization")
            path.write_text(json.dumps(payload))
        elif case == "duplicate_key":
            path.write_text(original.rstrip()[:-1] + ', "model": "toy"}')
        elif case == "missing_checkpoint":
            best.unlink()
        else:
            state = torch.load(best, map_location="cpu", weights_only=False)
            callback = next(
                value for value in state["callbacks"].values() if "best_epoch" in value
            )
            if case == "ambiguous_callback":
                state["callbacks"]["duplicate_selector"] = dict(callback)
            else:
                callback["best_model_score"] = torch.tensor(0.1)
            torch.save(state, best)
        with pytest.raises(ValueError):
            result, _ = training.load_completed_training(path, "toy")
            with result.best_weights():
                pass
    finally:
        path.write_text(original)
        best.write_bytes(checkpoint_bytes)


def _export_predictions(output, batch, prediction, record):
    chip_id = "test/example.tif"
    record = SimpleNamespace(
        chip_id=chip_id, image_path=record.image_path, label_path=record.label_path
    )
    exports = prediction_export.PairedPredictionExport(
        output, SimpleNamespace(splits={"test": [record]})
    )
    batch = {**batch, "chip_id": [chip_id]}
    for item in exports.batches("toy", [batch]):
        exports.write("toy", item, [prediction])
    with np.load(output / "toy/predictions/000000.npz", allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}, exports, batch


def test_metric_exports_preserve_overlaps_scores_and_empty_detections(tmp_path):

    masks = torch.zeros(3, 16, 16)
    masks[0, 2:10, 2:10] = 1
    masks[1, 6:14, 6:14] = 1
    masks[2, :, 0] = 1  # Wholly outside the common footprint.
    valid = torch.ones(16, 16, dtype=torch.bool)
    valid[:, 0] = False
    batch = {
        "mask": torch.zeros(1, 16, 16, dtype=torch.long),
        "valid_mask": valid[None],
        "raw_image": torch.ones(1, 70, 16, 16),
        "image_valid_mask": valid[None, None].expand(1, 70, 16, 16),
        "boxes": [torch.tensor([[2, 2, 10, 10], [6, 6, 14, 14]])],
        "masks": [masks[:2].byte()],
        "labels": [torch.ones(2, dtype=torch.long)],
    }
    prediction = {
        "masks": masks[:, None],
        "scores": torch.tensor([0.002, 0.001, 0.99]),
        "labels": torch.ones(3, dtype=torch.long),
        "boxes": torch.tensor([[2, 2, 10, 10], [6, 6, 14, 14], [0, 0, 1, 16]]),
    }
    record = SimpleNamespace(
        image_path=Path("test/example.tif"), label_path=Path("example.npz")
    )
    sample = prediction_export._sample(
        batch, 0, prediction, "test/example.tif", record, 0.5
    )
    restored = np.unpackbits(sample["prediction_mask_bits"], axis=-1, count=16)
    assert (restored[0] & restored[1]).any()
    assert not restored[2].any()
    np.testing.assert_array_equal(
        sample["prediction_scores"], prediction["scores"].numpy()
    )
    assert not sample["pred_mask"].any()  # Plot threshold excludes low scores.
    evaluator = InstanceMaskAP()
    evaluator.update(
        [
            {
                "masks": torch.from_numpy(restored),
                "scores": prediction["scores"],
                "labels": prediction["labels"],
            }
        ],
        batch,
    )
    assert evaluator.compute().item() == pytest.approx(1)
    empty = {key: value[:0] for key, value in prediction.items()}
    sample, _, _ = _export_predictions(tmp_path / "empty", batch, empty, record)
    assert sample["prediction_mask_bits"].shape == (0, 16, 2)
    assert sample["pred_boxes"].shape == (0, 4)


@pytest.mark.parametrize(
    "field",
    [
        "chip_id",
        "image_path",
        "label_path",
        "valid_mask",
        "image_valid_bits",
        "image",
        "gt_mask",
        "pred_mask",
    ],
)
def test_rejects_export_identity_geometry_partitions_and_validity_mismatch(
    field, tmp_path
):
    record = SimpleNamespace(
        image_path=Path("test/chips/example.tif"),
        label_path=Path("test/labels/example.npz"),
    )
    batch = {
        "mask": torch.ones(1, 16, 16, dtype=torch.long),
        "valid_mask": torch.ones(1, 16, 16, dtype=torch.bool),
        "raw_image": torch.zeros(1, 70, 16, 16),
        "image_valid_mask": torch.ones(1, 70, 16, 16, dtype=torch.bool),
        "boxes": [torch.tensor([[0, 0, 16, 16]])],
        "masks": [torch.ones(1, 16, 16, dtype=torch.uint8)],
        "labels": [torch.ones(1, dtype=torch.long)],
    }
    prediction = {
        "masks": torch.zeros(0, 1, 16, 16),
        "scores": torch.zeros(0),
        "labels": torch.ones(0, dtype=torch.long),
        "boxes": torch.zeros(0, 4),
    }
    sample, exports, batch = _export_predictions(
        tmp_path / "mismatch", batch, prediction, record
    )
    changed = {key: value.copy() for key, value in sample.items()}
    if field in {"chip_id", "image_path", "label_path"}:
        changed[field] = np.array("other")
    elif field == "pred_mask":
        changed[field] = np.zeros((8, 8))
    else:
        changed[field].flat[0] = 0 if field in {"valid_mask", "image_valid_bits"} else 2
    np.savez_compressed(tmp_path / "mismatch/toy/predictions/000000.npz", **changed)
    with pytest.raises(ValueError):
        for item in exports.batches("graha", [batch]):
            exports.write("graha", item, [prediction])
    # Equivalent target partitions may use different positive instance IDs.
    sample, exports, batch = _export_predictions(
        tmp_path / "renumbered", batch, prediction, record
    )
    sample["gt_mask"] *= 3
    np.savez_compressed(tmp_path / "renumbered/toy/predictions/000000.npz", **sample)
    for item in exports.batches("graha", [batch]):
        exports.write("graha", item, [prediction])


def test_oom_is_not_retried_with_smaller_batch(trained_pair, tmp_path, monkeypatch):
    sizes = []

    def builder(config, dm, model):
        assert dm.batch_size == 2
        task = _task(config, dm, model)

        def oom(images):
            sizes.append(len(images))
            raise torch.OutOfMemoryError("requested test OOM")

        task.forward = oom
        return task

    route_tasks(monkeypatch, builder)
    with pytest.raises(torch.OutOfMemoryError, match="requested test OOM"):
        _compare(trained_pair, tmp_path / "oom", batch_size=2, n_samples=1)
    assert sizes == [2]
    assert not (tmp_path / "oom/comparison.json").exists()
