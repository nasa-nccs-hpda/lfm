"""Shared experiment orchestration helpers."""

from __future__ import annotations

import json
import re
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def json_ready(value: Any) -> Any:
    """Convert common config values into JSON-serializable objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    return value


def save_config_json(config: Any, path: Path) -> None:
    """Save a dataclass-like config as JSON."""
    payload = asdict(config) if is_dataclass(config) else dict(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)


def save_single_timing_json(
    *,
    model: str,
    started_at: float,
    path: Path,
) -> None:
    """Save a simple elapsed-time summary for one model run."""
    elapsed = time.perf_counter() - started_at
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"model": model, "seconds": round(elapsed, 3)}, f, indent=2)
    print(f"{model.title()} elapsed seconds: {elapsed:.3f}", flush=True)


@dataclass(frozen=True)
class CheckpointRecord:
    path: Path
    epoch: int | None
    name: str


def parse_checkpoint_epoch(path: Path) -> int | None:
    text = str(path)
    for pattern in [r"epoch[=_-](\d+)", r"model-(\d+)-", r"epoch_(\d+)"]:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def checkpoint_output_name(path: Path, epoch: int | None) -> str:
    if epoch is not None:
        return f"epoch_{epoch:03d}"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_")
    return stem or "checkpoint"


def discover_checkpoints(
    checkpoint_dir: Path,
    *,
    max_checkpoints: int | None = None,
) -> list[CheckpointRecord]:
    checkpoint_dir = Path(checkpoint_dir).resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoint_dir}"
        )
    paths = sorted(path for path in checkpoint_dir.rglob("*.ckpt") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No .ckpt files found under {checkpoint_dir}")

    records = []
    for path in paths:
        epoch = parse_checkpoint_epoch(path)
        records.append(
            CheckpointRecord(
                path=path,
                epoch=epoch,
                name=checkpoint_output_name(path, epoch),
            )
        )
    records.sort(
        key=lambda item: (
            item.epoch is None,
            item.epoch if item.epoch is not None else 10**9,
            str(item.path),
        )
    )

    unique_records = []
    used_names: dict[str, int] = {}
    for record in records:
        count = used_names.get(record.name, 0)
        used_names[record.name] = count + 1
        if count:
            record = CheckpointRecord(
                path=record.path,
                epoch=record.epoch,
                name=f"{record.name}_{count + 1}",
            )
        unique_records.append(record)

    if max_checkpoints is not None:
        unique_records = unique_records[:max_checkpoints]
    return unique_records


def load_lightning_checkpoint_state(
    module: Any,
    checkpoint_path: Path,
) -> None:
    import torch

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*You are using `torch.load` with `weights_only=False`.*",
            category=FutureWarning,
        )
        checkpoint = torch.load(
            Path(checkpoint_path).resolve(),
            map_location="cpu",
            weights_only=False,
        )
    module.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)


def write_checkpoint_metrics_summary(
    model_output_dir: Path,
    rows: list[dict[str, Any]],
    *,
    metric_names: Sequence[str],
) -> None:
    if not rows:
        return
    model_output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = model_output_dir / "checkpoint_metrics_summary.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        header = ["checkpoint_name", "epoch", "checkpoint_path", *metric_names]
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write(
                "\t".join(
                    [
                        str(row["checkpoint_name"]),
                        "" if row["epoch"] is None else str(row["epoch"]),
                        str(row["checkpoint_path"]),
                        *[f"{row[name]:.8f}" for name in metric_names],
                    ]
                )
                + "\n"
            )

    dtype = [
        ("checkpoint_name", "U128"),
        ("epoch", "i8"),
        ("checkpoint_path", "U1024"),
        *[(name, "f8") for name in metric_names],
    ]
    arr = np.zeros(len(rows), dtype=dtype)
    for i, row in enumerate(rows):
        arr[i]["checkpoint_name"] = str(row["checkpoint_name"])
        arr[i]["epoch"] = -1 if row["epoch"] is None else int(row["epoch"])
        arr[i]["checkpoint_path"] = str(row["checkpoint_path"])
        for name in metric_names:
            arr[i][name] = row[name]
    np.save(model_output_dir / "checkpoint_metrics_summary.npy", arr)
    print(f"Saved model summary to {txt_path}", flush=True)


class SingleModelExperiment:
    """Common runner for one Toy or Graha model workflow."""

    def __init__(
        self,
        *,
        model: str,
        config: Any,
        output_dir: Path,
        run_model: Callable[[], Any],
        checkpoint_subdirs: Sequence[str | Path] = (),
        on_complete: Callable[[float, Any], None] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.run_model = run_model
        self.checkpoint_subdirs = tuple(Path(path) for path in checkpoint_subdirs)
        self.on_complete = on_complete

    def prepare(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for subdir in self.checkpoint_subdirs:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)
        save_config_json(
            self.config,
            self.output_dir / f"config_{self.model}_model.json",
        )

    def run(self) -> Any:
        started_at = time.perf_counter()
        self.prepare()
        result = self.run_model()
        if self.on_complete is not None:
            self.on_complete(started_at, result)
        else:
            save_single_timing_json(
                model=self.model,
                started_at=started_at,
                path=self.output_dir / f"timing_summary_{self.model}_model.json",
            )
        return result


class ComparisonExperiment:
    """Common runner for paired Toy and Graha comparison workflows."""

    def __init__(
        self,
        *,
        config: Any,
        output_dir: Path,
        run_toy: Callable[[], Any],
        run_graha: Callable[[], Any],
        checkpoint_subdirs: Sequence[str | Path] = (),
        on_complete: Callable[[float, dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.run_toy = run_toy
        self.run_graha = run_graha
        self.checkpoint_subdirs = tuple(Path(path) for path in checkpoint_subdirs)
        self.on_complete = on_complete

    def prepare(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for subdir in self.checkpoint_subdirs:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)
        save_config_json(self.config, self.output_dir / "config.json")

    def run(self) -> dict[str, Any]:
        started_at = time.perf_counter()
        self.prepare()
        results = {
            "toy": self.run_toy(),
            "graha": self.run_graha(),
        }
        if self.on_complete is not None:
            self.on_complete(started_at, results)
        else:
            elapsed = time.perf_counter() - started_at
            with (self.output_dir / "timing_summary.json").open(
                "w", encoding="utf-8"
            ) as f:
                json.dump({"seconds": round(elapsed, 3)}, f, indent=2)
            print(f"Comparison elapsed seconds: {elapsed:.3f}", flush=True)
        return results


class CheckpointSweepExperiment:
    """Common runner for Toy/Graha checkpoint sweep orchestration."""

    def __init__(
        self,
        *,
        output_root: Path,
        models: Sequence[str],
        checkpoint_dirs: dict[str, Path | None],
        run_model_sweep: Callable[[str, list[CheckpointRecord]], list[dict[str, Any]]],
        max_checkpoints: int | None = None,
        seed: int | None = None,
        seed_fn: Callable[[int], Any] | None = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.models = tuple(models)
        self.checkpoint_dirs = checkpoint_dirs
        self.run_model_sweep = run_model_sweep
        self.max_checkpoints = max_checkpoints
        self.seed = seed
        self.seed_fn = seed_fn

    def prepare(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        if self.seed is not None and self.seed_fn is not None:
            self.seed_fn(self.seed)

    def _discover_model_checkpoints(self, model: str) -> list[CheckpointRecord]:
        checkpoint_dir = self.checkpoint_dirs.get(model)
        if checkpoint_dir is None:
            raise ValueError(
                f"{model.title()} sweep requested but {model}_checkpoint_dir is not set."
            )
        checkpoints = discover_checkpoints(
            checkpoint_dir,
            max_checkpoints=self.max_checkpoints,
        )
        print(f"[{model.title()}] Found {len(checkpoints)} checkpoint(s).")
        return checkpoints

    def run(self) -> dict[str, list[dict[str, Any]]]:
        self.prepare()
        results: dict[str, list[dict[str, Any]]] = {}
        for model in self.models:
            checkpoints = self._discover_model_checkpoints(model)
            results[model] = self.run_model_sweep(model, checkpoints)
        return results
