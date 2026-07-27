"""Shared experiment orchestration helpers."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


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
