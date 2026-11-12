"""Exercise the shared lifecycle through both real semantic construction adapters."""

import json
from dataclasses import replace
from types import MethodType

import numpy as np
import pytest
import rasterio
import torch
import yaml
from lightning.pytorch import Callback, LightningModule
from rasterio.transform import from_origin
from torch import nn

from lfm.all_models import recorded_training
from lfm.all_models.recorded_training import (
    CompletedTraining,
    Resume,
    RunSpec,
    WeightsOnly,
)
from lfm.all_models.sem_seg.config import build_config
from lfm.all_models.sem_seg.recorded_training import begin_training
from lfm.full_model.sem_seg import semantic_graha_components as graha
from lfm.full_model.sem_seg.semantic_from_instance_datamodule import (
    LunarSemanticFromInstanceDatamodule,
)
from lfm.toy_model.sem_seg import semantic_toy_components as toy


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def semantic_inputs(tmp_path):
    data, pretrain = tmp_path / "data", tmp_path / "pretrain"
    raw = np.broadcast_to(
        np.arange(70, dtype=np.float32)[:, None, None] + 20, (70, 80, 80)
    ).copy()
    raw += np.arange(80, dtype=np.float32)[None, None, :]
    mask = np.zeros((80, 80), dtype=np.int32)
    mask[8:24, 8:24] = 7
    for split in ("train", "val", "test"):
        chips, labels = data / split / "chips", data / split / "labels"
        chips.mkdir(parents=True)
        labels.mkdir()
        with rasterio.open(
            chips / "one_chip.tif",
            "w",
            driver="GTiff",
            height=80,
            width=80,
            count=70,
            dtype="float32",
            transform=from_origin(100, 200, 1, 1),
        ) as dst:
            dst.write(raw)
        np.savez(labels / "one_label.npz", mask=mask, bboxes=np.array([[8, 8, 16, 16]]))
    (pretrain / "checkpoints").mkdir(parents=True)
    (pretrain / "full_config.yaml").write_text("{}")
    (pretrain / "checkpoints/checkpoint_weights_final.pt").touch()
    info = {
        name: {"stats": {name: {"mean": [100] * count, "std": [2] * count}}}
        for name, count in (("vis", 5), ("uv", 2), ("static", 63))
    }
    (pretrain / "modality_info.yaml").write_text(yaml.safe_dump(info))
    return data, pretrain


class TinyGrahaTask(LightningModule):
    """Local replacement for the external pretrained backbone, with real batches."""

    def __init__(self, **kwargs):
        super().__init__()
        self.model = nn.Conv2d(70, 2, 1)

    def training_step(self, batch, batch_idx):
        assert "num_craters" not in batch  # The production downstream wrapper runs.
        return nn.functional.cross_entropy(
            self.model(batch["image"]), batch["mask"].long()
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.StepLR(optimizer, 1),
        }


@pytest.fixture
def small_backbones(monkeypatch):
    monkeypatch.setattr(
        toy, "create_model", lambda config, assignments: nn.Conv2d(70, 2, 1)
    )
    monkeypatch.setattr(graha, "configure_proj_environment", lambda: None)
    monkeypatch.setattr(
        graha,
        "import_project_dependencies",
        lambda: {
            "LunarSemanticFromInstanceDatamodule": LunarSemanticFromInstanceDatamodule,
            "LunarShapeSegmentationTask": TinyGrahaTask,
        },
    )


def _config(semantic_inputs, output):
    data, pretrain = semantic_inputs
    return build_config(
        data_root=data,
        base_output_dir=output,
        graha_pretrain_dir=pretrain,
        band_filter=list(range(70)),
        target_size=80,
        semantic_label_source="instance",
        batch_size=1,
        num_workers=0,
        graha_batch_size=1,
        graha_num_workers=0,
        normalization_source="pretrain",
        normalization_modality="vis_uv_static",
        normalize_inputs=True,
        toy_loss_type="cross_entropy",
        max_epochs=3,
        graha_input_modality_mode="vis-uv-static",
        graha_backend_modalities=["vis", "uv", "static"],
    )


@pytest.mark.parametrize("model,metric", [("toy", "val_loss"), ("graha", "val/loss")])
def test_real_semantic_adapters_minimize_losses_above_one_and_load_best(
    semantic_inputs, tmp_path, small_backbones, monkeypatch, model, metric
):
    config = _config(semantic_inputs, tmp_path / model)
    attempt = begin_training(config, RunSpec(model, "semantic", 3))
    prepared = attempt.prepare()
    assert prepared.run.configuration.max_epochs == 3
    assert prepared.datamodule.batch_size == 1
    batch = next(iter(prepared.datamodule.val_dataloader()))
    image = batch[0] if isinstance(batch, tuple) else batch["image"]
    assert torch.isfinite(image).all()
    torch.testing.assert_close(image[0, 0, 0, 0], torch.tensor(-40.0))

    def validation_step(task, batch, batch_idx):
        task.log(metric, (3.0, 2.0, 4.0)[task.current_epoch], batch_size=1)

    prepared.task.validation_step = MethodType(validation_step, prepared.task)
    result = attempt.fit(prepared)
    assert result.payload["selection"] == {
        "metric": metric,
        "mode": "min",
        "split": "val",
    }
    assert result.payload["best_checkpoint"]["epoch"] == 1
    assert result.payload["best_checkpoint"]["score"] == 2.0
    assert result.payload["latest_checkpoint"]["epoch"] == 2
    completed = CompletedTraining.load(result.path, attempt.run)
    with completed.best_weights() as weights:
        prepared.task.load_state_dict(weights, strict=True)
        assert weights
    assert not weights
    assert json.loads(result.path.read_text())["task"] == "semantic"
    with pytest.raises(ValueError, match="already reached"):
        begin_training(
            config,
            attempt.run.spec,
            start=Resume(result.payload["latest_checkpoint"]["path"]),
        )

    class StopAfterFirst(Callback):
        def on_train_epoch_end(self, trainer, module):
            trainer.should_stop = trainer.current_epoch == 0

    original_trainer = recorded_training.create_training_trainer

    def interrupted_trainer(run, checkpoint):
        trainer = original_trainer(run, checkpoint)
        trainer.callbacks.insert(0, StopAfterFirst())
        return trainer

    interrupted_config = replace(
        config, base_output_dir=tmp_path / f"{model}-interrupted"
    )
    interrupted = begin_training(interrupted_config, attempt.run.spec)
    interrupted_prepared = interrupted.prepare()
    interrupted_prepared.task.validation_step = MethodType(
        validation_step, interrupted_prepared.task
    )
    with monkeypatch.context() as patch:
        patch.setattr(recorded_training, "create_training_trainer", interrupted_trainer)
        with pytest.raises(RuntimeError, match="complete best/latest"):
            interrupted.fit(interrupted_prepared)
    resumed = begin_training(
        interrupted_config,
        attempt.run.spec,
        start=Resume(interrupted_config.base_output_dir / "checkpoints/last.ckpt"),
    )
    assert resumed.initial_epoch == 1
    resumed_prepared = resumed.prepare()
    resumed_prepared.task.validation_step = MethodType(
        validation_step, resumed_prepared.task
    )
    resumed_result = resumed.fit(resumed_prepared)
    full_state = torch.load(
        result.payload["latest_checkpoint"]["path"],
        map_location="cpu",
        weights_only=False,
    )
    resumed_state = torch.load(
        resumed_result.payload["latest_checkpoint"]["path"],
        map_location="cpu",
        weights_only=False,
    )
    assert resumed_result.payload["best_checkpoint"]["epoch"] == 1
    assert full_state["global_step"] == resumed_state["global_step"] == 3
    for key, value in full_state["state_dict"].items():
        torch.testing.assert_close(
            resumed_state["state_dict"][key], value, rtol=0, atol=0
        )

    # A legacy weights-only file is valid initialization, never full-state resume.
    legacy = tmp_path / f"{model}-legacy.ckpt"
    torch.save({"state_dict": prepared.task.state_dict()}, legacy)
    fresh = replace(config, base_output_dir=tmp_path / f"{model}-fresh")
    with pytest.raises(ValueError, match="incompatible|full-state"):
        begin_training(fresh, attempt.run.spec, start=Resume(legacy))
    initialization = begin_training(fresh, attempt.run.spec, start=WeightsOnly(legacy))
    loaded = initialization.prepare()
    assert initialization.initial_epoch == 0
    for key, value in prepared.task.state_dict().items():
        torch.testing.assert_close(loaded.task.state_dict()[key], value)


def test_prepared_model_cannot_cross_attempts(
    semantic_inputs, tmp_path, small_backbones
):
    config = _config(semantic_inputs, tmp_path / "one")
    first = begin_training(config, RunSpec("toy", "semantic", 1))
    second = begin_training(
        replace(config, base_output_dir=tmp_path / "two"), first.run.spec
    )
    with pytest.raises(ValueError, match="another training attempt"):
        second.fit(first.prepare())
    assert not (tmp_path / "two/result.json").exists()
