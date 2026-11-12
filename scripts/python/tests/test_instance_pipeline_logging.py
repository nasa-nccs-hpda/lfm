"""Exercise live streaming and concurrent writers without GPUs or Slurm."""

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from lfm.all_models.inst_seg.experiment_logging import log_pipeline, run_logged_process


def test_output_is_visible_before_process_exit_and_error_status_is_preserved(tmp_path):
    log = tmp_path / "pipeline.log"
    release = tmp_path / "release"
    script = """
import pathlib, sys, time
print('first training batch', flush=True)
deadline = time.monotonic() + 15
while not pathlib.Path(sys.argv[1]).exists() and time.monotonic() < deadline:
    time.sleep(0.01)
print('CUDA out of memory', file=sys.stderr, flush=True)
sys.exit(7)
"""
    with ThreadPoolExecutor() as pool:
        future = pool.submit(
            run_logged_process,
            [sys.executable, "-u", "-c", script, str(release)],
            log,
            "toy",
            env=dict(os.environ),
        )
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if log.exists() and "first training batch" in log.read_text():
                    break
                time.sleep(0.01)
            assert "first training batch" in log.read_text()
            assert not future.done(), "Output must arrive before the child finishes"
        finally:
            release.touch()
        assert future.result(timeout=10) == 7
    text = log.read_text()
    assert "[toy" in text and "CUDA out of memory" in text and "FAILED (exit=7)" in text


def test_independent_processes_append_complete_tagged_lines(tmp_path):
    log = tmp_path / "pipeline.log"
    repo = Path(__file__).resolve().parents[3]
    script = """
import sys
from pathlib import Path
from lfm.all_models.inst_seg.experiment_logging import log_pipeline
for index in range(25):
    log_pipeline(Path(sys.argv[1]), sys.argv[2], f'event={index} ' + 'moon' * 512)
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-S", "-c", script, str(log), f"writer-{number}"],
            cwd=repo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        for number in range(4)
    ]
    for process in processes:
        _, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, stderr
    lines = log.read_text().splitlines()
    assert len(lines) == 100
    for number in range(4):
        for index in range(25):
            matches = [
                line
                for line in lines
                if f"[writer-{number}" in line and f"] event={index} " in line
            ]
            assert len(matches) == 1
            assert matches[0].endswith("moon" * 512)


def test_terminal_output_is_normalized_and_job_identity_is_recorded(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    log = tmp_path / "pipeline.log"
    log_pipeline(log, "graha", "\x1b[31mepoch 1\x1b[0m\repoch 2\ncheckpoint saved")
    lines = log.read_text().splitlines()
    assert len(lines) == 3
    assert all("[graha job=42]" in line for line in lines)
    assert "\x1b" not in log.read_text()
    assert [line.split("] ", 1)[1] for line in lines] == [
        "epoch 1",
        "epoch 2",
        "checkpoint saved",
    ]


def test_container_launch_error_reaches_central_log(tmp_path):
    log = tmp_path / "pipeline.log"
    with pytest.raises(FileNotFoundError):
        run_logged_process(
            [str(tmp_path / "missing-apptainer")],
            log,
            "comparison",
            env=dict(os.environ),
        )
    assert "Launch failed: FileNotFoundError" in log.read_text()
