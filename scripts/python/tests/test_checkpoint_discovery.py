from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


EXPERIMENTS_PATH = (
    Path(__file__).resolve().parents[3]
    / "lfm"
    / "all_models"
    / "all_tasks"
    / "experiments.py"
)
SPEC = importlib.util.spec_from_file_location(
    "lfm_checkpoint_discovery_test_module",
    EXPERIMENTS_PATH,
)
assert SPEC is not None and SPEC.loader is not None
EXPERIMENTS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPERIMENTS
SPEC.loader.exec_module(EXPERIMENTS)
resolve_inference_checkpoint = EXPERIMENTS.resolve_inference_checkpoint


def _write_checkpoint(path: Path, contents: bytes = b"checkpoint") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return path


def test_explicit_checkpoint_takes_precedence(tmp_path: Path) -> None:
    explicit = _write_checkpoint(tmp_path / "manual" / "model.pt")

    resolved = resolve_inference_checkpoint(
        explicit,
        task_subdir="semantic_seg_finetuning",
        outputs_root=tmp_path / "missing_outputs",
    )

    assert resolved == explicit.resolve()


def test_discovers_checkpoint_from_latest_valid_experiment(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    task_dir = tmp_path / "outputs" / "semantic_seg_finetuning"
    older = _write_checkpoint(
        task_dir / "date_2026_08_29-time_10_00_00" / "checkpoints" / "old.ckpt"
    )
    newest_empty = task_dir / "date_2026_08_31-time_10_00_00"
    newest_empty.mkdir(parents=True)
    latest_valid = _write_checkpoint(
        task_dir
        / "date_2026_08_30-time_10_00_00"
        / "checkpoints"
        / "full_model"
        / "latest.pt"
    )

    resolved = resolve_inference_checkpoint(
        "",
        task_subdir="semantic_seg_finetuning",
        outputs_root=tmp_path / "outputs",
    )

    assert resolved == latest_valid.resolve()
    assert resolved != older.resolve()
    output = capsys.readouterr().out
    assert f"Discovered experiment directory: {latest_valid.parents[2]}" in output
    assert "Discovered checkpoint: latest.pt" in output


def test_chooses_newest_checkpoint_within_latest_experiment(tmp_path: Path) -> None:
    experiment_dir = (
        tmp_path
        / "outputs"
        / "instance_seg_finetuning"
        / "date_2026_08_31-time_10_00_00"
    )
    older = _write_checkpoint(experiment_dir / "checkpoints" / "epoch_1.ckpt")
    newer = _write_checkpoint(experiment_dir / "artifacts" / "final.pt")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    resolved = resolve_inference_checkpoint(
        None,
        task_subdir="instance_seg_finetuning",
        outputs_root=tmp_path / "outputs",
    )

    assert resolved == newer.resolve()


def test_raises_when_no_valid_checkpoint_exists(tmp_path: Path) -> None:
    experiment_dir = (
        tmp_path
        / "outputs"
        / "instance_seg_finetuning"
        / "date_2026_08_31-time_10_00_00"
    )
    _write_checkpoint(experiment_dir / "empty.ckpt", contents=b"")
    _write_checkpoint(experiment_dir / "notes.txt")

    with pytest.raises(FileNotFoundError, match=r"No non-empty \.ckpt or \.pt"):
        resolve_inference_checkpoint(
            "",
            task_subdir="instance_seg_finetuning",
            outputs_root=tmp_path / "outputs",
        )
