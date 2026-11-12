"""Recorded training lifecycle shared by model families and segmentation tasks.

Task adapters describe configuration, preparation and selection. This module owns
resume, RNG ordering, checkpoint provenance and atomic success publication.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Generic, Literal, Protocol, TypeVar

import numpy as np
import torch
from lightning.pytorch import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from lightning.pytorch.callbacks import ModelCheckpoint

from lfm.artifacts import json_ready, read_json, write_json

Configuration = TypeVar("Configuration")


@dataclass(frozen=True)
class RunSpec:
    model: Literal["toy", "graha"]
    task: Literal["instance", "semantic"]
    max_epochs: int

    def __post_init__(self):
        if self.model not in ("toy", "graha") or self.task not in (
            "instance",
            "semantic",
        ):
            raise ValueError("Unsupported model or segmentation task.")
        if type(self.max_epochs) is not int or self.max_epochs < 1:
            raise ValueError("max_epochs must be a positive integer.")


@dataclass(frozen=True)
class SelectionPolicy:
    metric: str
    mode: Literal["min", "max"]
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self):
        if not self.metric or self.mode not in ("min", "max"):
            raise ValueError("Selection needs a metric and min/max direction.")

    def valid(self, score) -> bool:
        try:
            return (
                math.isfinite(float(score))
                and (self.minimum is None or float(score) >= self.minimum)
                and (self.maximum is None or float(score) <= self.maximum)
            )
        except (TypeError, ValueError):
            return False

    def improves(self, score: float, previous: float) -> bool:
        return score < previous if self.mode == "min" else score > previous


@dataclass(frozen=True)
class Pretrained:
    pass


@dataclass(frozen=True)
class WeightsOnly:
    path: Path


@dataclass(frozen=True)
class Resume:
    path: Path


@dataclass(frozen=True)
class TrainingRun(Generic[Configuration]):
    spec: RunSpec
    output_dir: Path
    seed: int
    configuration: Configuration
    identity: dict
    metadata: dict
    initialization: dict
    selection: SelectionPolicy
    state_key: str = "recorded_training"
    callback_name: str = "TrainingCheckpoint"

    def expected_result(self) -> dict:
        return json_ready(
            {
                **self.metadata,
                "schema_version": 1,
                "status": "complete",
                "model": self.spec.model,
                "configuration": asdict(self.configuration),
                "training_identity": self.identity,
                "max_epochs": self.spec.max_epochs,
                "completed_epochs": self.spec.max_epochs,
                "epoch_indexing": "zero_based",
            }
        )


@dataclass(frozen=True)
class PreparedModel(Generic[Configuration]):
    run: TrainingRun[Configuration]
    task: LightningModule
    datamodule: LightningDataModule
    owner: object


class RecordedModelAdapter(Protocol):
    def prepare(
        self, run: TrainingRun, initial_epoch: int, owner: object
    ) -> PreparedModel: ...


@dataclass(frozen=True)
class TrainingResult:
    path: Path
    payload: dict


class TrainingProgress(Callback):
    """Readable progress for non-interactive logs, without a metrics service."""

    def __init__(self, metric: str):
        self.metric = metric
        self._last_report = 0.0

    def on_train_epoch_start(self, trainer, pl_module):
        self._last_report = time.monotonic()
        print(
            f"Training epoch {trainer.current_epoch + 1}/{trainer.max_epochs} started",
            flush=True,
        )

    def _batch(self, trainer, batch_idx, phase, outputs=None):
        now = time.monotonic()
        if batch_idx == 0 or (batch_idx + 1) % 50 == 0 or now - self._last_report >= 60:
            self._last_report = now
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
            detail = (
                f"; loss={float(loss.detach()):.6g}"
                if isinstance(loss, torch.Tensor) and loss.numel() == 1
                else ""
            )
            print(
                f"{phase} epoch {trainer.current_epoch + 1}/{trainer.max_epochs}; batch={batch_idx + 1}; step={trainer.global_step}{detail}",
                flush=True,
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._batch(trainer, batch_idx, "Training", outputs)

    def on_validation_epoch_start(self, trainer, pl_module):
        print(
            f"Full validation epoch {trainer.current_epoch + 1}/{trainer.max_epochs} started",
            flush=True,
        )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._batch(trainer, batch_idx, "Validation")

    def on_validation_end(self, trainer, pl_module):
        score = trainer.callback_metrics.get(self.metric)
        print(
            f"Full validation epoch {trainer.current_epoch + 1}/{trainer.max_epochs} complete; {self.metric}={score}",
            flush=True,
        )


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError(
                "Resume checkpoint CUDA RNG state requires the same device count."
            )
        torch.cuda.set_rng_state_all(state["cuda"])


class TrainingCheckpoint(ModelCheckpoint):
    """Retain selected best and latest full state, with strict run provenance."""

    def __init__(
        self,
        output_dir: Path,
        identity: dict,
        initialization: dict,
        selection: SelectionPolicy,
        state_key: str,
        callback_name: str = "TrainingCheckpoint",
    ):
        self.callback_name = callback_name
        self.selection = selection
        self.metadata_key = state_key
        super().__init__(
            dirpath=output_dir / "checkpoints",
            monitor=selection.metric,
            mode=selection.mode,
            filename="best-epoch-{epoch:03d}",
            auto_insert_metric_name=False,
            save_top_k=1,
            save_last=True,
            save_weights_only=False,
            every_n_epochs=1,
            save_on_train_epoch_end=True,
            enable_version_counter=False,
        )
        self.identity = identity
        self.initialization = initialization
        self.best_epoch: int | None = None
        self._resume_rng = None
        self._pending_removals: list[str] = []

    @property
    def state_key(self):
        return super().state_key.replace(type(self).__qualname__, self.callback_name, 1)

    def on_validation_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            metric = trainer.callback_metrics.get(self.selection.metric)
            if metric is None or not self.selection.valid(metric):
                raise RuntimeError(
                    f"Checkpoint selection requires finite {self.selection.metric} satisfying {self.selection}; got {metric}."
                )
        super().on_validation_end(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        score = trainer.callback_metrics.get(self.selection.metric)
        if score is None or not self.selection.valid(score):
            raise RuntimeError(
                f"Missing or invalid full-validation {self.selection.metric}."
            )
        if self.best_model_score is None or self.selection.improves(
            float(score), float(self.best_model_score)
        ):
            self.best_epoch = trainer.current_epoch
        print(
            f"Saving epoch {trainer.current_epoch + 1}/{trainer.max_epochs} best/latest checkpoints",
            flush=True,
        )
        super().on_train_epoch_end(trainer, pl_module)
        # Some Lightning releases update save_last only when top-k improved.
        # A worse validation epoch still needs its own resumable latest state.
        if self._last_global_step_saved != trainer.global_step:
            self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))
        # Until latest is durable, its previous best must remain available for
        # recovery if publication is interrupted between the two writes.
        for path in self._pending_removals:
            super()._remove_checkpoint(trainer, path)
        self._pending_removals.clear()
        print(
            f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} checkpoints saved; best_score={float(self.best_model_score):.6g}; best={self.best_model_path}; latest={self.last_model_path}",
            flush=True,
        )

    def _remove_checkpoint(self, trainer, filepath):
        self._pending_removals.append(filepath)

    def state_dict(self):
        return {**super().state_dict(), "best_epoch": self.best_epoch}

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.best_epoch = state_dict["best_epoch"]

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint[self.metadata_key] = {
            "identity": self.identity,
            "initialization": self.initialization,
            "rng": _rng_state(),
        }

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        state = checkpoint.get(self.metadata_key, {})
        if state.get("identity") != self.identity:
            raise ValueError(
                "Resume checkpoint has incompatible experiment/model/epoch identity."
            )
        self.initialization = state["initialization"]
        self._resume_rng = state["rng"]

    def on_train_start(self, trainer, pl_module):
        # Restore after model/optimizer creation and loader setup consume RNG.
        if self._resume_rng is not None:
            _restore_rng(self._resume_rng)
            self._resume_rng = None


def load_resume_state(
    path: Path,
    identity: dict,
    checkpoint_dir: Path,
    *,
    state_key: str = "recorded_training",
) -> dict:
    """Load only an explicitly supplied, trusted full-state local checkpoint."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    metadata = state.get(state_key, {})
    if metadata.get("identity") != identity:
        raise ValueError(
            "Resume checkpoint has incompatible experiment/model/epoch identity."
        )
    required = (
        "state_dict",
        "optimizer_states",
        "lr_schedulers",
        "loops",
        "epoch",
        "global_step",
    )
    if any(key not in state for key in required) or not state["optimizer_states"]:
        raise ValueError("Explicit resume requires a full-state training checkpoint.")
    if (
        not state["lr_schedulers"]
        or not metadata.get("rng")
        or not metadata.get("initialization")
    ):
        raise ValueError(
            "Resume checkpoint lacks scheduler, RNG, or initialization state."
        )
    callbacks = [
        value
        for value in state.get("callbacks", {}).values()
        if "best_epoch" in value and "best_model_path" in value
    ]
    if len(callbacks) != 1 or callbacks[0].get("dirpath") != str(
        checkpoint_dir.resolve()
    ):
        raise ValueError(
            "Resume must use the original model output/checkpoint directory."
        )
    best = Path(callbacks[0]["best_model_path"])
    if not best.is_file() or best.stat().st_size == 0:
        raise FileNotFoundError(f"Resume requires the retained best checkpoint: {best}")
    if state["epoch"] + 1 >= identity["max_epochs"]:
        raise ValueError(
            "Resume checkpoint already reached the total target epoch budget."
        )
    return state


def create_training_trainer(
    run: TrainingRun, checkpoint: TrainingCheckpoint
) -> Trainer:
    return Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="32",
        max_epochs=run.spec.max_epochs,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        logger=False,
        callbacks=[TrainingProgress(run.selection.metric), checkpoint],
        enable_progress_bar=False,
        log_every_n_steps=5,
    )


class TrainingAttempt:
    """Validated attempt; seed before preparation and publish only after fit."""

    def __init__(
        self,
        run: TrainingRun,
        adapter: RecordedModelAdapter,
        *,
        start: Pretrained | WeightsOnly | Resume | None = None,
    ):
        start = Pretrained() if start is None else start
        if not isinstance(start, (Pretrained, WeightsOnly, Resume)):
            raise TypeError("Choose Pretrained, WeightsOnly or Resume initialization.")
        self.run, self.adapter, self.start = run, adapter, start
        self.initial_epoch = 0
        self.initialization = run.initialization
        self.resume = None
        self._prepared = False
        self._fitted = False
        if isinstance(start, Resume):
            self.resume = Path(start.path).expanduser().resolve()
            state = load_resume_state(
                self.resume,
                run.identity,
                run.output_dir / "checkpoints",
                state_key=run.state_key,
            )
            self.initialization = state[run.state_key]["initialization"]
            self.initial_epoch = state["epoch"] + 1
        elif (run.output_dir / "result.json").exists() or any(
            (run.output_dir / "checkpoints").glob("*.ckpt")
        ):
            raise FileExistsError(
                "Existing training artifacts require explicit --resume or a fresh run root."
            )
        run.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare(self) -> PreparedModel:
        if self._prepared:
            raise RuntimeError("A training attempt may be prepared only once.")
        self._prepared = True
        seed_everything(self.run.seed, workers=True)
        prepared = self.adapter.prepare(self.run, self.initial_epoch, self)
        self._validate_prepared(prepared)
        if isinstance(self.start, WeightsOnly):
            path = Path(self.start.path).expanduser().resolve()
            weights = torch.load(path, map_location="cpu", weights_only=False)
            prepared.task.load_state_dict(
                weights.get("state_dict", weights), strict=True
            )
            self.initialization = {
                "mode": "weights_only",
                "method": "weights_only",
                "checkpoint": str(path),
                "base": self.initialization,
            }
        return prepared

    def _validate_prepared(self, prepared: PreparedModel):
        if prepared.owner is not self or prepared.run is not self.run:
            raise ValueError("Prepared model belongs to another training attempt.")

    def fit(self, prepared: PreparedModel) -> TrainingResult:
        self._validate_prepared(prepared)
        if self._fitted or not self._prepared:
            raise RuntimeError("Prepare and fit each training attempt once.")
        self._fitted = True
        checkpoint = TrainingCheckpoint(
            self.run.output_dir,
            self.run.identity,
            self.initialization,
            self.run.selection,
            self.run.state_key,
            self.run.callback_name,
        )
        trainer = create_training_trainer(self.run, checkpoint)
        (self.run.output_dir / "result.json").unlink(missing_ok=True)
        trainer.fit(
            prepared.task,
            datamodule=prepared.datamodule,
            ckpt_path=str(self.resume) if self.resume else None,
            weights_only=False,
        )
        return self._publish(trainer, checkpoint)

    def _publish(self, trainer, checkpoint) -> TrainingResult:
        best, latest = (
            Path(checkpoint.best_model_path),
            Path(checkpoint.last_model_path),
        )
        if (
            not checkpoint.best_model_path
            or not checkpoint.last_model_path
            or not best.is_file()
            or not latest.is_file()
            or best.stat().st_size == 0
            or latest.stat().st_size == 0
            or not self.run.selection.valid(checkpoint.best_model_score)
            or checkpoint.best_epoch is None
            or trainer.current_epoch != self.run.spec.max_epochs
        ):
            raise RuntimeError(
                "Training did not produce complete best/latest checkpoints and epoch progress."
            )
        payload = {
            **self.run.expected_result(),
            "global_step": trainer.global_step,
            "initialization": checkpoint.initialization,
            "resume_from": str(self.resume) if self.resume else None,
            "best_checkpoint": {
                "path": str(best.resolve()),
                "epoch": checkpoint.best_epoch,
                "score": float(checkpoint.best_model_score),
            },
            "latest_checkpoint": {
                "path": str(latest.resolve()),
                "epoch": trainer.current_epoch - 1,
                "full_state": True,
            },
        }
        path = self.run.output_dir / "result.json"
        write_json(path, payload)
        return TrainingResult(path, payload)


@dataclass(frozen=True)
class CompletedTraining:
    path: Path
    payload: dict
    run: TrainingRun

    @classmethod
    def load(cls, path: Path, run: TrainingRun) -> CompletedTraining:
        path = Path(path).expanduser().resolve()
        payload = read_json(path)
        try:
            if path != run.output_dir / "result.json":
                raise ValueError(
                    "Result must belong to its recorded model output directory."
                )
            for key, expected in run.expected_result().items():
                if payload[key] != expected:
                    raise ValueError(f"Mismatched or stale result field: {key}.")
            best, latest = payload["best_checkpoint"], payload["latest_checkpoint"]
            if (
                type(best["epoch"]) is not int
                or not 0 <= best["epoch"] < run.spec.max_epochs
                or latest["epoch"] != run.spec.max_epochs - 1
                or latest["full_state"] is not True
                or not run.selection.valid(best["score"])
                or type(payload["global_step"]) is not int
                or payload["global_step"] < 1
                or not payload["initialization"]
            ):
                raise ValueError("Incomplete checkpoint selection/progress metadata.")
            for item in (best, latest):
                checkpoint = Path(item["path"])
                if (
                    not checkpoint.is_absolute()
                    or checkpoint.parent != run.output_dir / "checkpoints"
                    or not checkpoint.is_file()
                    or checkpoint.stat().st_size == 0
                ):
                    raise ValueError(f"Missing or mismatched checkpoint: {checkpoint}.")
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError(f"Incomplete model-result manifest: {path}.") from error
        return cls(path, payload, run)

    def _checkpoint_state(self, kind: str) -> dict:
        item = self.payload[f"{kind}_checkpoint"]
        state = torch.load(item["path"], map_location="cpu", weights_only=False)
        metadata = state.get(self.run.state_key, {})
        callbacks = [
            value
            for value in state.get("callbacks", {}).values()
            if "best_epoch" in value and "best_model_path" in value
        ]
        best = self.payload["best_checkpoint"]
        if (
            metadata.get("identity") != self.payload["training_identity"]
            or metadata.get("initialization") != self.payload["initialization"]
            or state.get("epoch") != item["epoch"]
            or not state.get("state_dict")
            or not state.get("optimizer_states")
            or not state.get("lr_schedulers")
            or not state.get("loops")
            or len(callbacks) != 1
        ):
            raise ValueError(f"Incomplete or mismatched {kind} checkpoint provenance.")
        callback = callbacks[0]
        score = callback.get("best_model_score")
        if (
            callback["best_model_path"] != best["path"]
            or callback["best_epoch"] != best["epoch"]
            or score is None
            or float(score) != best["score"]
            or callback.get("dirpath") != str(self.run.output_dir / "checkpoints")
            or (
                kind == "latest"
                and state.get("global_step") != self.payload["global_step"]
            )
        ):
            raise ValueError(f"Stale or ambiguous {kind} checkpoint selection.")
        return state

    @contextmanager
    def best_weights(self) -> Iterator[Mapping[str, torch.Tensor]]:
        # Validate one model at a time. Do not retain latest optimizer/RNG state.
        latest = self._checkpoint_state("latest")
        del latest
        state = self._checkpoint_state("best")
        weights = state.pop("state_dict")
        del state
        try:
            yield weights
        finally:
            weights.clear()
