"""Run comparison training, then sweep the checkpoints from that run."""

# ruff: noqa: E402

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

LFM_ROOT = Path(__file__).resolve().parents[3]
if str(LFM_ROOT) not in sys.path:
    sys.path.insert(0, str(LFM_ROOT))

from lfm.all_models.all_tasks.checkpoint_pipeline_config import (
    CheckpointPipelineConfig,
    CheckpointPipelineResult,
    build_checkpoint_pipeline_config_from_args,
    build_sweep_command,
    command_to_display,
)
from lfm.all_models.all_tasks.cli_args import parse_checkpoint_pipeline_args


def _run_command(command: Sequence[str]) -> float:
    print("\nRunning command:", flush=True)
    print(command_to_display(command), flush=True)
    started_at = time.perf_counter()
    subprocess.run(command, check=True)
    elapsed = time.perf_counter() - started_at
    print(f"Command finished in {elapsed:.1f}s", flush=True)
    return elapsed


def _snapshot_children(path: Path) -> set[Path]:
    if not path.exists():
        return set()
    return {child.resolve() for child in path.iterdir() if child.is_dir()}


def _latest_child(path: Path, *, exclude: set[Path]) -> Path:
    candidates = [
        child
        for child in path.iterdir()
        if child.is_dir() and child.resolve() not in exclude
    ]
    if not candidates:
        candidates = [child for child in path.iterdir() if child.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No output directory was created under {path}")
    return max(candidates, key=lambda child: child.stat().st_mtime).resolve()


def run_pipeline(config: CheckpointPipelineConfig) -> CheckpointPipelineResult:
    config.base_output_dir.mkdir(parents=True, exist_ok=True)

    training_seconds: float | None = None
    if config.existing_training_output_dir:
        training_output_dir = config.existing_training_output_dir
        print(f"Using existing training output dir: {training_output_dir}", flush=True)
    else:
        before = _snapshot_children(config.base_output_dir)
        training_seconds = _run_command(config.training_command)
        training_output_dir = _latest_child(config.base_output_dir, exclude=before)
        print(f"Detected training output dir: {training_output_dir}", flush=True)

    if config.skip_sweep:
        result = CheckpointPipelineResult(
            task=config.task,
            training_output_dir=str(training_output_dir),
            sweep_output_dir=None,
            training_seconds=training_seconds,
            sweep_seconds=None,
        )
    else:
        sweep_output_dir = (
            config.sweep_output_root
            if config.sweep_output_root
            else training_output_dir / "checkpoint_sweep"
        )
        command = build_sweep_command(config, training_output_dir, sweep_output_dir)
        sweep_seconds = _run_command(command)
        result = CheckpointPipelineResult(
            task=config.task,
            training_output_dir=str(training_output_dir),
            sweep_output_dir=str(sweep_output_dir),
            training_seconds=training_seconds,
            sweep_seconds=sweep_seconds,
        )

    summary_path = training_output_dir / "train_then_checkpoint_sweep_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"Saved pipeline summary to {summary_path}", flush=True)
    return result


def parse_args():
    return parse_checkpoint_pipeline_args(description=__doc__)


def main() -> None:
    args = parse_args()
    config = build_checkpoint_pipeline_config_from_args(
        args,
        repo_root=LFM_ROOT,
        python_executable=sys.executable,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
