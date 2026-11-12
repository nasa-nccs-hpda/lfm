"""Line-oriented experiment logs shared by independent Slurm allocations."""

from __future__ import annotations

import fcntl
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def log_pipeline(path: Path, stage: str, message: str) -> None:
    """Flush tagged lines under a shared-filesystem lock, then echo to Slurm."""
    job_id = os.environ.get("SLURM_JOB_ID")
    context = f"{stage} job={job_id}" if job_id else stage
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    lines = _ANSI_ESCAPE.sub("", str(message)).splitlines() or [""]
    record = "".join(f"{timestamp} [{context}] {line}\n" for line in lines)
    # Separate allocations append concurrently. Hold the lock through flush so
    # buffered writes and multi-line tracebacks cannot splice each other's lines.
    with Path(path).open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(record)
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    print(record, end="", flush=True)


def run_logged_process(command: list[str], path: Path, stage: str, *, env: dict) -> int:
    """Tee merged stdout/stderr as it arrives; preserve the child's exit status."""
    log_pipeline(path, stage, "Starting container process")
    try:
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
            env=env,
        ) as process:
            for line in process.stdout:
                log_pipeline(path, stage, line.rstrip("\r\n"))
            returncode = process.wait()
    except BaseException as exc:
        log_pipeline(path, stage, f"Launch failed: {type(exc).__name__}: {exc}")
        raise
    log_pipeline(
        path,
        stage,
        f"Container process {'succeeded' if returncode == 0 else 'FAILED'} (exit={returncode})",
    )
    return returncode
