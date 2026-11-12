"""Development checks for selection, publication, and explicit full-state recovery."""

import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from lightning.pytorch import Callback, LightningDataModule, LightningModule
from torch.utils.data import DataLoader

from lfm.all_models import recorded_training
from lfm.all_models.inst_seg import training
from lfm.all_models.inst_seg.data.sampling import EpochShuffleSampler
from lfm.all_models.inst_seg.paired_experiment import (
    create_experiment_datamodule,
    load_experiment,
    prepare_experiment,
)
from scripts.python.tests import test_instance_mask_ap as ap
from scripts.python.tests.instance_experiment_support import route_tasks

input_files = ap.input_files
cpu_threads = ap.cpu_threads


class SmallData(LightningDataModule):
    def __init__(self, seed, initial_epoch):
        super().__init__()
        self.seed, self.initial_epoch = seed, initial_epoch

    def train_dataloader(self):
        data = torch.arange(7).float()
        return DataLoader(
            data,
            batch_size=2,
            sampler=EpochShuffleSampler(data, self.seed, self.initial_epoch),
            generator=torch.Generator().manual_seed(self.seed),
        )

    def val_dataloader(self):
        return DataLoader([0], generator=torch.Generator().manual_seed(self.seed))


class SmallTask(LightningModule):
    def __init__(self, budget, scores=(0.4, 0.9, 0.6)):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(()))
        self.budget, self.scores = budget, scores
        self.seen = []

    def training_step(self, batch, batch_idx):
        self.seen.extend((self.current_epoch, int(item)) for item in batch)
        noise = torch.rand(()) + random.random() + float(np.random.random())
        return (self.weight - batch.mean() - noise).square()

    def validation_step(self, *args):
        pass

    def on_validation_epoch_end(self):
        score = self.scores[self.current_epoch]
        if score is not None:
            self.log(training.MASK_AP_KEY, score)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.budget
            ),
        }


def _prepare(input_files, root, budget=3):
    experiment = ap._experiment(input_files, root)
    experiment = replace(
        experiment,
        toy=replace(experiment.toy, max_epochs=budget),
        graha=replace(experiment.graha, max_epochs=budget),
    )
    prepare_experiment(experiment, root)
    return experiment


@pytest.fixture
def small_route(monkeypatch):
    tasks = []

    def builder(config, dm, model):
        task = SmallTask(config.max_epochs)
        tasks.append(task)
        return task

    route_tasks(monkeypatch, builder)
    monkeypatch.setattr(
        "lfm.all_models.inst_seg.paired_experiment.create_experiment_datamodule",
        lambda experiment, manifest, model, initial_epoch, **kwargs: SmallData(
            experiment.toy.seed, initial_epoch
        ),
    )
    return tasks


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_nonmonotonic_selection_and_full_state_publication(
    input_files, tmp_path, small_route
):
    root = tmp_path / "run"
    experiment = _prepare(input_files, root)
    result = json.loads(training.train_prepared_experiment(root, "toy").read_text())
    assert result["best_checkpoint"]["epoch"] == 1
    assert result["best_checkpoint"]["score"] == pytest.approx(0.9)
    assert result["latest_checkpoint"]["epoch"] == 2
    assert result["completed_epochs"] == result["max_epochs"] == 3
    assert result["config_id"] == experiment.config_id
    assert result["split_manifest_id"] == load_experiment(root)[1].manifest_id
    assert result["normalization"]["method"] == "per_chip_per_band_minmax"
    assert result["sampler"]["seed"] == 42
    assert len(list((root / "toy/checkpoints").glob("*.ckpt"))) == 2
    best = _load(result["best_checkpoint"]["path"])
    latest = _load(result["latest_checkpoint"]["path"])
    assert best["epoch"] == 1 and latest["epoch"] == 2
    assert latest["optimizer_states"][0]["state"]
    assert latest["lr_schedulers"][0]["last_epoch"] == 3
    assert latest[training.STATE_KEY]["rng"]
    assert any(
        key.startswith("InstanceTrainingCheckpoint{") for key in latest["callbacks"]
    )
    assert not torch.equal(best["state_dict"]["weight"], latest["state_dict"]["weight"])
    with pytest.raises(FileExistsError, match="explicit --resume"):
        training.train_prepared_experiment(root, "toy")


@pytest.mark.parametrize("stop_epoch", [0, 1])
def test_resume_matches_uninterrupted_optimizer_scheduler_rng_and_sample_order(
    input_files, tmp_path, small_route, monkeypatch, stop_epoch
):
    full, interrupted = tmp_path / "full", tmp_path / "interrupted"
    for root in (full, interrupted):
        _prepare(input_files, root)
    full_result = json.loads(
        training.train_prepared_experiment(full, "toy").read_text()
    )
    full_task = small_route[-1]
    create_trainer = recorded_training.create_training_trainer

    class StopAfterOneEpoch(Callback):
        def on_train_epoch_end(self, trainer, module):
            trainer.should_stop = trainer.current_epoch == stop_epoch

    def stop_trainer(config, checkpoint):
        trainer = create_trainer(config, checkpoint)
        trainer.callbacks.insert(0, StopAfterOneEpoch())
        return trainer

    monkeypatch.setattr(recorded_training, "create_training_trainer", stop_trainer)
    with pytest.raises(RuntimeError, match="complete best/latest"):
        training.train_prepared_experiment(interrupted, "toy")
    first_task = small_route[-1]
    assert not (interrupted / "toy/result.json").exists()
    latest_path = interrupted / "toy/checkpoints/last.ckpt"
    assert _load(latest_path)["epoch"] == stop_epoch
    monkeypatch.setattr(recorded_training, "create_training_trainer", create_trainer)
    resumed_result = json.loads(
        training.train_prepared_experiment(
            interrupted,
            "toy",
            resume=latest_path,
            max_epochs=3,
        ).read_text()
    )
    resumed_task = small_route[-1]
    assert resumed_task.seen[0][0] == stop_epoch + 1
    assert first_task.seen + resumed_task.seen == full_task.seen
    assert resumed_result["best_checkpoint"]["epoch"] == 1
    full_state = _load(full_result["latest_checkpoint"]["path"])
    resumed_state = _load(resumed_result["latest_checkpoint"]["path"])
    assert resumed_state["global_step"] == full_state["global_step"] == 12
    torch.testing.assert_close(
        full_state["state_dict"]["weight"],
        resumed_state["state_dict"]["weight"],
        rtol=0,
        atol=0,
    )
    for key in ("exp_avg", "exp_avg_sq", "step"):
        torch.testing.assert_close(
            full_state["optimizer_states"][0]["state"][0][key],
            resumed_state["optimizer_states"][0]["state"][0][key],
            rtol=0,
            atol=0,
        )
    assert full_state["lr_schedulers"] == resumed_state["lr_schedulers"]


@pytest.mark.parametrize("metric", [None, float("nan"), float("inf"), -1.0])
def test_invalid_selection_metric_fails_without_success(
    input_files, tmp_path, small_route, monkeypatch, metric
):
    root = tmp_path / "run"
    _prepare(input_files, root, budget=1)
    route_tasks(monkeypatch, lambda *args: SmallTask(1, [metric]))
    with pytest.raises(RuntimeError, match="finite val_mask_ap"):
        training.train_prepared_experiment(root, "toy")
    assert not (root / "toy/result.json").exists()


def test_resume_rejects_wrong_identity_and_incomplete_state(
    input_files, tmp_path, small_route
):
    root = tmp_path / "run"
    experiment = _prepare(input_files, root)
    result = json.loads(training.train_prepared_experiment(root, "toy").read_text())
    state = _load(result["latest_checkpoint"]["path"])
    identity = training.training_identity(
        experiment, load_experiment(root)[1], "toy", 3
    )
    state["epoch"] = 0
    checkpoint = tmp_path / "mutated.ckpt"
    for key in ("config_id", "split_manifest_id", "model", "max_epochs"):
        wrong = {**identity, key: "incompatible"}
        torch.save(state, checkpoint)
        with pytest.raises(ValueError, match="incompatible"):
            recorded_training.load_resume_state(
                checkpoint,
                wrong,
                root / "toy/checkpoints",
                state_key=training.STATE_KEY,
            )
    state.pop("optimizer_states")
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="full-state"):
        recorded_training.load_resume_state(
            checkpoint, identity, root / "toy/checkpoints", state_key=training.STATE_KEY
        )
    with pytest.raises(ValueError, match="already reached"):
        training.train_prepared_experiment(
            root, "toy", resume=result["latest_checkpoint"]["path"]
        )


def test_weights_initialization_starts_new_optimizer_and_progress(
    input_files, tmp_path, small_route
):
    source, fresh = tmp_path / "source", tmp_path / "fresh"
    for root in (source, fresh):
        _prepare(input_files, root, budget=1)
    source_result = json.loads(
        training.train_prepared_experiment(source, "toy").read_text()
    )
    weights = source_result["best_checkpoint"]["path"]
    result = json.loads(
        training.train_prepared_experiment(
            fresh, "toy", initialize_weights=weights
        ).read_text()
    )
    assert result["initialization"]["method"] == "weights_only"
    assert result["resume_from"] is None
    assert result["completed_epochs"] == 1 and result["global_step"] == 4
    assert small_route[-1].seen[0][0] == 0
    with pytest.raises(ValueError, match="resume or weights-only"):
        training.train_prepared_experiment(
            fresh, "toy", resume=weights, initialize_weights=weights
        )


@pytest.mark.parametrize("route", ["toy", "graha"])
def test_real_routes_publish_one_epoch_artifacts(
    input_files, tmp_path, monkeypatch, route, capsys
):
    root = tmp_path / "run"
    _prepare(input_files, root, budget=50)
    tasks = []

    def builder(config, dm, model):
        assert config.max_epochs == 1
        if model == "toy":
            task = ap.ToyDinoMaskRCNNLightningModule(
                ap._tiny_maskrcnn(),
                max_epochs=config.max_epochs,
                ignore_nodata_in_loss=True,
            )
            assert task.max_epochs_for_scheduler == 1
        else:

            class ScheduledBase(ap._LunarTaskBase):
                configure_optimizers = SmallTask.configure_optimizers

            task = ap.make_downstream_object_detection_task_class(ScheduledBase)(
                ignore_nodata_in_loss=True
            )
            task.budget = config.max_epochs
        tasks.append(task)
        return task

    route_tasks(monkeypatch, builder)
    result = json.loads(
        training.train_prepared_experiment(root, route, max_epochs=1).read_text()
    )
    assert result["architecture"] == (
        "dino-mask-rcnn" if route == "toy" else "graha-mask-rcnn"
    )
    assert result["max_epochs"] == result["completed_epochs"] == 1
    assert (
        result["best_checkpoint"]["epoch"] == result["latest_checkpoint"]["epoch"] == 0
    )
    assert Path(result["best_checkpoint"]["path"]).is_file()
    assert Path(result["latest_checkpoint"]["path"]).is_file()
    assert not (root / route / "eligibility/test.json").exists()
    assert _load(result["latest_checkpoint"]["path"])["lr_schedulers"][0]["T_max"] == 1
    experiment, manifest = load_experiment(root)
    dm = create_experiment_datamodule(experiment, manifest, route, initial_epoch=2)
    assert dm.train_dataloader().sampler.epoch == 2
    progress = capsys.readouterr().out
    assert "Training epoch 1/1 started" in progress
    assert "Training epoch 1/1; batch=1" in progress
    assert "Full validation epoch 1/1 complete;" in progress
    assert "Epoch 1/1 checkpoints saved; best_score=" in progress


def test_missing_artifact_prevents_result_publication(
    input_files, tmp_path, small_route, monkeypatch
):
    root = tmp_path / "run"
    _prepare(input_files, root, budget=1)
    create_trainer = recorded_training.create_training_trainer

    def trainer_with_lost_checkpoint(run, checkpoint):
        trainer = create_trainer(run, checkpoint)
        fit = trainer.fit

        def fit_then_remove(*args, **kwargs):
            fit(*args, **kwargs)
            Path(checkpoint.last_model_path).unlink()

        trainer.fit = fit_then_remove
        return trainer

    monkeypatch.setattr(
        recorded_training, "create_training_trainer", trainer_with_lost_checkpoint
    )
    with pytest.raises(RuntimeError, match="complete best/latest"):
        training.train_prepared_experiment(root, "toy")
    assert not (root / "toy/result.json").exists()


def test_latest_write_failure_preserves_previous_recoverable_best(
    input_files, tmp_path, small_route, monkeypatch
):
    root = tmp_path / "run"
    experiment = _prepare(input_files, root)
    save_last = recorded_training.TrainingCheckpoint._save_last_checkpoint

    def fail_second_write(callback, trainer, metrics):
        if trainer.current_epoch == 1:
            raise OSError("latest write failed")
        return save_last(callback, trainer, metrics)

    monkeypatch.setattr(
        recorded_training.TrainingCheckpoint, "_save_last_checkpoint", fail_second_write
    )
    with pytest.raises(OSError, match="latest write failed"):
        training.train_prepared_experiment(root, "toy")
    assert not (root / "toy/result.json").exists()
    state = recorded_training.load_resume_state(
        root / "toy/checkpoints/last.ckpt",
        training.training_identity(experiment, load_experiment(root)[1], "toy", 3),
        root / "toy/checkpoints",
        state_key=training.STATE_KEY,
    )
    assert state["epoch"] == 0


def test_manifest_write_failure_cannot_report_success(
    input_files, tmp_path, small_route, monkeypatch
):
    root = tmp_path / "run"
    _prepare(input_files, root, budget=1)
    original = Path.replace

    def fail_result_publication(path, target):
        if Path(target).name == "result.json":
            raise OSError("manifest publication failed")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", fail_result_publication)
    with pytest.raises(OSError, match="manifest publication failed"):
        training.train_prepared_experiment(root, "toy")
    assert not (root / "toy/result.json").exists()


def test_incompatible_model_state_fails_strict_resume(
    input_files, tmp_path, small_route, monkeypatch
):
    root = tmp_path / "run"
    _prepare(input_files, root)
    create_trainer = recorded_training.create_training_trainer

    class StopAfterOneEpoch(Callback):
        def on_train_epoch_end(self, trainer, module):
            trainer.should_stop = True

    def stop_trainer(config, checkpoint):
        trainer = create_trainer(config, checkpoint)
        trainer.callbacks.insert(0, StopAfterOneEpoch())
        return trainer

    monkeypatch.setattr(recorded_training, "create_training_trainer", stop_trainer)
    with pytest.raises(RuntimeError, match="complete best/latest"):
        training.train_prepared_experiment(root, "toy")
    latest = root / "toy/checkpoints/last.ckpt"
    state = _load(latest)
    state["state_dict"]["unexpected_parameter"] = torch.zeros(1)
    torch.save(state, latest)
    monkeypatch.setattr(recorded_training, "create_training_trainer", create_trainer)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        training.train_prepared_experiment(root, "toy", resume=latest)
    assert not (root / "toy/result.json").exists()


@pytest.mark.parametrize("route", ["toy", "graha"])
def test_resume_with_actual_eligible_loaders(input_files, tmp_path, monkeypatch, route):
    tasks = []

    class ChipTask(SmallTask):
        def training_step(self, batch, batch_idx):
            self.seen.extend((self.current_epoch, chip) for chip in batch["chip_id"])
            noise = torch.rand(()) + random.random() + float(np.random.random())
            return (self.weight - batch["image"].mean() - noise).square()

    def builder(config, dm, model):
        task = ChipTask(config.max_epochs)
        tasks.append(task)
        return task

    route_tasks(monkeypatch, builder)
    full, interrupted = tmp_path / "full", tmp_path / "interrupted"
    for root in (full, interrupted):
        _prepare(input_files, root)
    full_result = json.loads(
        training.train_prepared_experiment(full, route).read_text()
    )
    full_task = tasks[-1]
    create_trainer = recorded_training.create_training_trainer

    class StopAfterOneEpoch(Callback):
        def on_train_epoch_end(self, trainer, module):
            trainer.should_stop = True

    def stop_trainer(config, checkpoint):
        trainer = create_trainer(config, checkpoint)
        trainer.callbacks.insert(0, StopAfterOneEpoch())
        return trainer

    monkeypatch.setattr(recorded_training, "create_training_trainer", stop_trainer)
    with pytest.raises(RuntimeError, match="complete best/latest"):
        training.train_prepared_experiment(interrupted, route)
    first_task = tasks[-1]
    monkeypatch.setattr(recorded_training, "create_training_trainer", create_trainer)
    resumed_result = json.loads(
        training.train_prepared_experiment(
            interrupted,
            route,
            resume=interrupted / route / "checkpoints/last.ckpt",
        ).read_text()
    )
    assert first_task.seen + tasks[-1].seen == full_task.seen
    assert len(full_task.seen) == 9  # Three eligible chips in each of three epochs.
    torch.testing.assert_close(
        _load(full_result["latest_checkpoint"]["path"])["state_dict"]["weight"],
        _load(resumed_result["latest_checkpoint"]["path"])["state_dict"]["weight"],
        rtol=0,
        atol=0,
    )


def test_production_builders_receive_budget_without_pretrained_loading(
    input_files, tmp_path, monkeypatch
):
    from lfm.full_model.inst_seg import instance_graha_components as graha
    from lfm.toy_model.inst_seg import instance_toy_components as toy

    experiment = ap._experiment(input_files, tmp_path)
    monkeypatch.setattr(toy, "load_dinov3_encoder", lambda **kwargs: None)
    monkeypatch.setattr(
        toy, "create_dino_mask_rcnn_model", lambda **kwargs: ap._tiny_maskrcnn()
    )
    toy_config = replace(experiment.toy, max_epochs=17)
    dm = create_experiment_datamodule(
        experiment, ap.build_split_manifest(experiment), "toy"
    )
    task = training.resolve_model_adapter("toy", "instance").create_recorded_task(
        toy_config, dm
    )
    assert task.max_epochs_for_scheduler == 17
    scheduler = task.configure_optimizers()["lr_scheduler"]["scheduler"]
    assert scheduler._schedulers[-1].T_max == 15  # Two warmup + 15 cosine epochs.
    checkpoint = recorded_training.TrainingCheckpoint(
        tmp_path, {}, {}, training.INSTANCE_SELECTION, training.STATE_KEY
    )
    assert (
        recorded_training.create_training_trainer(
            training.describe_training(
                experiment, ap.build_split_manifest(experiment), "toy", 17
            ),
            checkpoint,
        ).max_epochs
        == 17
    )
    captured = {}

    def capture_task(config, task_cls, sample):
        captured["config"] = config
        return sample

    monkeypatch.setattr(graha, "configure_python_paths", lambda config: None)
    monkeypatch.setattr(
        graha,
        "import_project_dependencies",
        lambda: {"LunarObjectDetectionTask": LightningModule},
    )
    monkeypatch.setattr(graha, "create_task", capture_task)
    graha_config = replace(experiment.graha, max_epochs=17)
    training.resolve_model_adapter("graha", "instance").create_recorded_task(
        graha_config, dm
    )
    assert captured["config"].max_epochs == 17
    assert (
        recorded_training.create_training_trainer(
            training.describe_training(
                experiment, ap.build_split_manifest(experiment), "graha", 17
            ),
            checkpoint,
        ).max_epochs
        == 17
    )


def test_training_cli_rejects_invalid_budget_and_ambiguous_initialization():
    parser = training.create_training_parser()
    base = ["--run-root", "/tmp/example", "--model", "toy"]
    assert parser.parse_args(base + ["--max-epochs", "1"]).max_epochs == 1
    assert parser.parse_args(base).max_epochs is None
    for extra in (
        ["--max-epochs", "0"],
        ["--resume", "a", "--initialize-weights", "b"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(base + extra)
