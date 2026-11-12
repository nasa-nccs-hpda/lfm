"""The paired baseline reaches both real data adapters without model weights."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
import yaml
from rasterio.transform import from_origin

from lfm.all_models.all_tasks import config_defaults as defaults
from lfm.all_models.all_tasks.cli_args import create_instance_experiment_parser
from lfm.all_models.inst_seg.paired_experiment import (
    build_split_manifest,
    create_experiment_datamodule,
    create_paired_experiment_parser,
    load_experiment,
    prepare_experiment,
    resolve_paired_experiment,
)


def _pair_dirs(root: Path, split: str) -> tuple[Path, Path]:
    chips = root / split / "chips"
    labels = root / split / "labels"
    chips.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    return chips, labels


@pytest.fixture
def paired_data(tmp_path):
    data_root = tmp_path / "dataset"
    raw = (
        np.arange(70, dtype=np.float32)[:, None, None] * 10
        + np.arange(258, dtype=np.float32)[None, :, None] / 2
        + np.arange(260, dtype=np.float32)[None, None, :] / 4
    )
    mask = np.zeros((258, 260), dtype=np.int32)
    mask[20:40, 30:60] = 7
    mask[90:120, 100:130] = 19
    for split in ("train", "val", "test"):
        chips, labels = _pair_dirs(data_root, split)
        with rasterio.open(
            chips / "lunar_input_wac_static_chip.tif",
            "w",
            driver="GTiff",
            width=260,
            height=258,
            count=70,
            dtype="float32",
            transform=from_origin(100, 200, 1, 1),
        ) as dst:
            dst.write(raw)
            for index in range(70):
                description = f"UV {index - 5}" if index in (5, 6) else f"Band {index}"
                dst.set_band_description(index + 1, description)
        np.savez(labels / "lunar_label.npz", mask=mask)
    pretrain = tmp_path / "pretrain"
    pretrain.mkdir()
    means = np.arange(70, dtype=np.float32) + 5
    stds = np.arange(70, dtype=np.float32) / 10 + 2
    modality_info = {}
    for modality, start, end in (("vis", 0, 5), ("uv", 5, 7), ("static", 7, 70)):
        modality_info[modality] = {
            "stats": {
                modality: {
                    "mean": means[start:end].tolist(),
                    "std": stds[start:end].tolist(),
                }
            }
        }
    (pretrain / "modality_info.yaml").write_text(yaml.safe_dump(modality_info))
    experiment = resolve_paired_experiment(
        run_root=tmp_path / "run",
        data_root=data_root,
        dino_checkpoint=tmp_path / "dino.pth",
        graha_pretrain_dir=pretrain,
        max_epochs=1,
        toy_num_workers=0,
        graha_num_workers=0,
    )
    return experiment, raw, mask, means, stds


def test_baseline_resolves_independently_without_changing_legacy_defaults(tmp_path):
    experiment = resolve_paired_experiment(run_root=tmp_path / "run")
    for config in (experiment.toy, experiment.graha):
        assert config.max_epochs == 50
        assert config.seed == 42
        assert config.toy_batch_size == config.graha_batch_size == 32
        assert config.band_filter == list(range(70))
        assert config.target_size == 256
        assert config.toy_architecture == "dino-mask-rcnn"
        assert config.graha_backend_modalities == ["vis", "uv", "static"]
        assert config.graha_input_modality_mode == "vis-uv-static"
        assert not config.toy_freeze_backbone and not config.graha_freeze_backbone
        assert (
            config.max_train_samples
            is config.max_val_samples
            is config.max_test_samples
            is None
        )
        assert not config.run_epoch_test_suite
    assert experiment.toy.normalization_source == "finetune"
    assert not experiment.toy.toy_normalize_inputs
    assert experiment.graha.normalization_source == "pretrain"
    assert experiment.graha.normalization_modality == "vis_uv_static"
    assert experiment.toy.toy_learning_rate == defaults.DEFAULT_INSTANCE_LEARNING_RATE
    assert experiment.graha.graha_head_lr == defaults.DEFAULT_GRAHA_HEAD_LR
    assert create_instance_experiment_parser().parse_args([]).max_epochs == 100
    payload = experiment.payload()
    assert len(payload["band_layout"]) == 70
    assert payload["band_layout"][7] == {
        "stored_index": 7,
        "modality": "static",
        "modality_index": 0,
    }
    assert payload["initialization"]["graha"]["new_modalities"] is None
    assert payload["feature_pyramids"]["toy"]["layers"] == [5, 11, 17, 23]


@pytest.mark.parametrize("epochs", [0, -1, 1.5, True])
def test_epoch_budget_must_be_positive_integer(tmp_path, epochs):
    with pytest.raises(ValueError, match="max_epochs"):
        resolve_paired_experiment(run_root=tmp_path, max_epochs=epochs)


def test_overrides_preserve_initial_protocol_and_native_embeddings(tmp_path):
    experiment = resolve_paired_experiment(
        run_root=tmp_path,
        seed=123,
        max_epochs=1,
        toy_batch_size=4,
        graha_batch_size=8,
        toy_learning_rate=1e-5,
        graha_head_lr=1e-4,
    )
    assert experiment.toy.toy_batch_size == 4
    assert experiment.graha.graha_batch_size == 8
    assert experiment.toy.toy_learning_rate == 1e-5
    assert experiment.graha.graha_head_lr == 1e-4
    assert experiment.toy.seed == experiment.graha.seed == 123
    assert experiment.toy.max_epochs == experiment.graha.max_epochs == 1
    with pytest.raises(TypeError, match="Unsupported baseline override"):
        resolve_paired_experiment(run_root=tmp_path, band_filter=[0, 1])
    with pytest.raises(ValueError, match="native vis/uv/static"):
        replace(
            experiment,
            graha=replace(experiment.graha, graha_backend_modalities=["wac"]),
        )


def test_preparation_only_pairs_files_and_persists_membership(tmp_path):
    # Invalid raster/label bytes are intentional: preparation must not open them.
    for split in ("train", "val", "test"):
        chips, labels = _pair_dirs(tmp_path / "data", split)
        for key in ("b", "a"):
            (chips / f"{key}_input_wac_static_chip.tif").write_bytes(b"not a raster")
            (labels / f"{key}_label.npz").write_bytes(b"not an array")
    root = tmp_path / "run"
    experiment = resolve_paired_experiment(run_root=root, data_root=tmp_path / "data")
    prepare_experiment(experiment, root)
    loaded, manifest = load_experiment(root)
    assert loaded == experiment
    assert manifest == build_split_manifest(experiment)
    assert [r.chip_id for r in manifest.splits["train"]] == [
        "train/a_input_wac_static_chip.tif",
        "train/b_input_wac_static_chip.tif",
    ]
    assert all(len(records) == 2 for records in manifest.splits.values())
    assert (root / "experiment.json").read_text().endswith("\n")
    with pytest.raises(FileExistsError, match="already exist"):
        prepare_experiment(experiment, root)
    payload = json.loads((root / "split_manifest.json").read_text())
    payload["splits"]["test"][0]["chip_id"] = "another-chip"
    (root / "split_manifest.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="identities do not match"):
        load_experiment(root)
    membership = build_split_manifest(experiment)
    (root / "split_manifest.json").write_text(
        json.dumps(
            {
                **membership.payload(),
                "manifest_id": membership.manifest_id,
            }
        )
    )
    payload = json.loads((root / "experiment.json").read_text())
    payload["normalization"]["toy"]["method"] = "incorrect_shared_statistics"
    (root / "experiment.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="identities do not match"):
        load_experiment(root)


def test_existing_graha_builder_retains_native_pretrained_modalities(tmp_path):
    from lfm.full_model.inst_seg.instance_graha_components import (
        build_comparison_config,
        create_task,
    )

    experiment = resolve_paired_experiment(run_root=tmp_path, max_epochs=1)
    config = build_comparison_config(experiment.graha, tmp_path / "graha")
    # Capture actual task construction arguments without creating/loading a model.
    args = create_task(config, dict, {"image": torch.empty(0, 70, 256, 256)})
    model_args = args["model_args"]
    assert model_args["backbone_modalities"] == ["vis", "uv", "static"]
    assert model_args["backbone_new_modalities"] is None
    assert model_args["in_channels"] == 70
    assert model_args["framework"] == "mask-rcnn"
    assert (
        model_args["backbone_checkpoint_path"]
        == experiment.payload()["initialization"]["graha"]["checkpoint"]
    )
    assert not args["freeze_backbone"]
    assert config.max_epochs == 1


def test_missing_label_fails_during_ordinary_pairing(tmp_path):
    chips, _ = _pair_dirs(tmp_path / "data", "train")
    (chips / "unpaired_chip.tif").touch()
    experiment = resolve_paired_experiment(
        run_root=tmp_path / "run", data_root=tmp_path / "data"
    )
    with pytest.raises(FileNotFoundError, match="no matching label"):
        build_split_manifest(experiment)


def test_real_adapters_load_same_crops_targets_and_frozen_membership(
    paired_data, tmp_path
):
    experiment, raw, mask, means, stds = paired_data
    root = tmp_path / "run"
    prepare_experiment(experiment, root)
    experiment, manifest = load_experiment(root)
    # A later unmatched file must not alter membership or Toy's metadata source.
    (experiment.toy.data_root / "train/chips/aaa_unmatched.tif").write_bytes(
        b"not a raster"
    )
    toy = create_experiment_datamodule(
        experiment, manifest, "toy", include_raw_image=True
    )
    graha = create_experiment_datamodule(
        experiment, manifest, "graha", include_raw_image=True
    )
    assert type(toy).__name__ == "ToyDinoMaskRCNNSplitDataModule"
    assert type(graha).__name__ == "GrahaObjectDetectionInstanceDataModule"
    assert len(toy.weight_assignments) == 70
    assert toy.weight_assignments[5:7] == ["blue", "blue"]
    expected_raw = torch.from_numpy(raw[:, 1:257, 2:258].copy())
    expected_mask = torch.from_numpy(mask[1:257, 2:258].astype(np.int64))
    expected_toy = (
        torch.from_numpy(raw) - torch.from_numpy(raw).amin((1, 2))[:, None, None]
    ) / (torch.from_numpy(raw).amax((1, 2)) - torch.from_numpy(raw).amin((1, 2)))[
        :, None, None
    ]
    expected_graha = (
        expected_raw - torch.from_numpy(means)[:, None, None]
    ) / torch.from_numpy(stds)[:, None, None]
    for split in ("train", "val", "test"):
        toy_dataset = getattr(toy, f"{split}_dataset")
        graha_dataset = getattr(graha, f"{split}_dataset")
        assert toy_dataset.records == graha_dataset.records == manifest.splits[split]
        a, b = toy_dataset[0], graha_dataset[0]
        assert a["chip_id"] == b["chip_id"] == manifest.splits[split][0].chip_id
        for item in (a, b):
            torch.testing.assert_close(item["raw_image"], expected_raw)
            torch.testing.assert_close(item["mask"], expected_mask)
            assert item["image"].shape == (70, 256, 256)
        for key in ("mask", "boxes", "labels", "masks"):
            torch.testing.assert_close(a[key], b[key])
        torch.testing.assert_close(a["image"], expected_toy[:, 1:257, 2:258])
        torch.testing.assert_close(b["image"], expected_graha)
        assert not torch.allclose(a["image"], b["image"])
    for dm in (toy, graha):
        batch = next(iter(dm.val_dataloader()))
        assert batch["chip_id"] == [manifest.splits["val"][0].chip_id]
        torch.testing.assert_close(batch["raw_image"][0], expected_raw)


def test_prepare_cli_forwards_one_epoch_without_loading_inputs(tmp_path):
    for split in ("train", "val", "test"):
        chips, labels = _pair_dirs(tmp_path / "data", split)
        (chips / "one_chip.tif").touch()
        (labels / "one_label.npz").touch()
    script = (
        Path(__file__).resolve().parents[1]
        / "instance_seg/prepare_instance_experiment.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--run-root",
            str(tmp_path / "run"),
            "--data-root",
            str(tmp_path / "data"),
            "--max-epochs",
            "1",
        ],
        env={**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(tmp_path / "mpl")},
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads((tmp_path / "run/experiment.json").read_text())
    assert (
        payload["models"]["toy"]["max_epochs"]
        == payload["models"]["graha"]["max_epochs"]
        == 1
    )
    assert payload["models"]["toy"]["toy_batch_size"] == 32
    assert "no training or jobs submitted" in result.stdout
    parser = create_paired_experiment_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--run-root", str(tmp_path / "other"), "--max-epochs", "0"])
