"""Host submission, container launch and allocated execution ownership.

The host import path is standard-library only. Model dependencies load solely
inside run_allocated, after the container has started.
"""

from __future__ import annotations

import os
import re
import runpy
import shlex
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from lfm.all_models.inst_seg.experiment_logging import log_pipeline, run_logged_process
from lfm.artifacts import read_json, write_json

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTAINER = "/explore/nobackup/projects/lfm/containers/lfm-container-ipyleaflet"
Stage = Literal["toy", "graha", "comparison"]
STAGES = ("toy", "graha", "comparison")


@dataclass(frozen=True)
class PreparationSettings:
    data_root: Path
    dino_checkpoint: Path
    graha_pretrain_dir: Path
    max_epochs: int = 50
    seed: int = 42
    toy_batch_size: int = 32
    graha_batch_size: int = 32
    toy_num_workers: int = 10
    graha_num_workers: int = 10
    image_glob: str = "*.tif"
    label_glob: str = "*_label.npz"
    image_suffix: str | None = None
    label_suffix: str | None = None
    excluded_nodata_values: list[float] | None = None
    toy_learning_rate: float | None = None
    toy_weight_decay: float | None = None
    toy_gradient_clip_val: float | None = None
    graha_backbone_lr: float | None = None
    graha_head_lr: float | None = None
    graha_layer_decay: float | None = None
    graha_weight_decay: float | None = None
    graha_warmup_steps: int | None = None


@dataclass(frozen=True)
class ContainerSettings:
    path: str = DEFAULT_CONTAINER
    identity: str | None = None
    executable: str = "apptainer"


@dataclass(frozen=True)
class Resources:
    memory: str = "128G"
    cpus: int = 10
    time: str = "24:00:00"


@dataclass(frozen=True)
class SubmissionPaths:
    run_root: Path | None = None
    output_root: Path = REPO_ROOT / "scripts/outputs/instance_grace"
    cache_dir: Path | None = None
    log_dir: Path | None = None
    config_source: Path | None = None


@dataclass(frozen=True)
class SchedulerSettings:
    executable: str = "sbatch"
    partition: str = "grace"
    account: str | None = None
    qos: str | None = None


@dataclass(frozen=True)
class InferenceSettings:
    batch_size: int = 32
    n_samples: int = 5


@dataclass(frozen=True)
class SubmissionRequest:
    preparation: PreparationSettings
    paths: SubmissionPaths = field(default_factory=SubmissionPaths)
    container: ContainerSettings = field(default_factory=ContainerSettings)
    scheduler: SchedulerSettings = field(default_factory=SchedulerSettings)
    training: Resources = field(default_factory=Resources)
    comparison: Resources = field(default_factory=Resources)
    inference: InferenceSettings = field(default_factory=InferenceSettings)

    def __post_init__(self):
        for value in (
            self.preparation.max_epochs,
            self.preparation.toy_batch_size,
            self.preparation.graha_batch_size,
            self.training.cpus,
            self.comparison.cpus,
            self.inference.batch_size,
            self.inference.n_samples,
        ):
            if type(value) is not int or value < 1:
                raise ValueError(
                    "Epoch, batch, sample and CPU settings must be positive integers."
                )


@dataclass(frozen=True)
class SubmissionReceipt:
    manifest: Path
    pipeline_log: Path
    job_ids: dict[str, str]


class _Submission:
    def __init__(self, path: Path, payload: dict):
        self.path, self.payload = path, payload

    @classmethod
    def load(cls, path: Path):
        payload = read_json(path)
        if payload["max_epochs"] != payload["preparation_settings"]["max_epochs"]:
            raise ValueError("Submission contains inconsistent epoch budgets.")
        return cls(path, payload)

    def save(self):
        write_json(self.path, self.payload)

    def command(self, stage: str) -> list[str]:
        """Build the actual allocated launch, preserving Slurm CUDA visibility."""
        state = self.payload
        if stage not in (*STAGES, "prepare"):
            raise ValueError(f"Unknown stage: {stage}")
        repo = Path(state["repository"]["path"])
        root = Path(state["run_root"])
        cache = Path(state["cache_dir"])
        command = [
            state["apptainer"],
            "exec",
            "--cleanenv",
            "--containall",
            "--no-mount",
            "home,cwd,hostfs",
        ]
        if stage != "prepare":
            command.append("--nv")
        binds = [
            (repo, "ro"),
            *((Path(path), "ro") for path in state["inputs"]),
            (root, "rw"),
            (cache, "rw"),
        ]
        if stage != "prepare":
            binds.extend(
                [
                    (Path(state["configuration"]), "ro"),
                    (Path(state["split_manifest"]), "ro"),
                ]
            )
        for path, mode in binds:
            command.extend(["--bind", f"{path}:{path}:{mode}"])
        command.extend(["--pwd", str(repo)])
        environment = {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(cache / "matplotlib" / stage),
            "TMPDIR": str(cache / "tmp" / stage),
            "HOME": str(cache / "home" / stage),
            "HF_HOME": str(cache / "huggingface"),
            "TORCH_HOME": str(cache / "torch"),
            "XDG_CACHE_HOME": str(cache / "xdg"),
        }
        for name in ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID", "SLURM_JOB_NODELIST"):
            if name in os.environ:
                environment[name] = os.environ[name]
        for key, value in environment.items():
            command.extend(["--env", f"{key}={value}"])
        if stage == "prepare":
            command.extend(
                [
                    state["container"]["path"],
                    "python",
                    "-u",
                    str(
                        repo
                        / "scripts/python/instance_seg/prepare_instance_experiment.py"
                    ),
                    "--run-root",
                    str(root),
                ]
            )
            for key, value in state["preparation_settings"].items():
                if value is not None:
                    command.append("--" + key.replace("_", "-"))
                    command.extend(
                        str(item)
                        for item in (value if isinstance(value, list) else [value])
                    )
            return command
        command.extend(
            [
                state["container"]["path"],
                "python",
                "-u",
                str(
                    repo / "scripts/python/instance_seg/run_instance_grace_gpu_stage.py"
                ),
                "--metadata",
                str(root / "runtime" / f"{stage}.json"),
                "--stage",
                stage,
            ]
        )
        if stage in ("toy", "graha"):
            command.extend(
                [
                    "--run-root",
                    str(root),
                    "--model",
                    stage,
                    "--max-epochs",
                    str(state["max_epochs"]),
                ]
            )
        else:
            command.extend(
                [
                    "--toy-result",
                    str(root / "toy/result.json"),
                    "--graha-result",
                    str(root / "graha/result.json"),
                    "--output-dir",
                    str(root / "comparison"),
                    "--batch-size",
                    str(state["inference"]["batch_size"]),
                    "--n-samples",
                    str(state["inference"]["n_samples"]),
                ]
            )
        return command


def _readable(path: Path) -> None:
    if not path.exists() or not os.access(path, os.R_OK):
        raise FileNotFoundError(f"Required path is missing or unreadable: {path}")


def _repository_state() -> dict:
    def git(*args):
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        return result.stdout.strip()

    return {
        "path": str(REPO_ROOT),
        "revision": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def submit(request: SubmissionRequest) -> SubmissionReceipt:
    repository = _repository_state()
    supplied_container = request.container.path
    container = Path(request.container.path).expanduser().resolve()
    _readable(container)
    run_root = (
        (
            request.paths.run_root
            or request.paths.output_root
            / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
        )
        .expanduser()
        .resolve()
    )
    cache = (request.paths.cache_dir or run_root / "cache").expanduser().resolve()
    logs = (request.paths.log_dir or run_root / "logs").expanduser().resolve()
    settings = {
        key: value
        for key, value in asdict(request.preparation).items()
        if value is not None
    }
    for key in ("data_root", "dino_checkpoint", "graha_pretrain_dir"):
        settings[key] = Path(settings[key]).expanduser().resolve()
    inputs = [
        settings["data_root"],
        settings["dino_checkpoint"],
        settings["graha_pretrain_dir"],
    ]
    for path in inputs + [
        settings["graha_pretrain_dir"] / "full_config.yaml",
        settings["graha_pretrain_dir"] / "modality_info.yaml",
        settings["graha_pretrain_dir"] / "checkpoints/checkpoint_weights_final.pt",
    ]:
        _readable(path)
    # Bind specifications cannot safely represent these characters. Keep all
    # writable destinations outside artifacts that must stay read-only.
    for path in [REPO_ROOT, run_root, cache, *inputs]:
        if any(character in str(path) for character in ":,\n"):
            raise ValueError(f"Unsupported character in container bind path: {path}")
    for destination in (run_root, cache, logs):
        if any(
            destination == source
            or source in destination.parents
            or destination in source.parents
            for source in inputs
        ):
            raise ValueError(
                f"Writable destination overlaps a read-only input: {destination}"
            )
        if destination == REPO_ROOT or destination in REPO_ROOT.parents:
            raise ValueError(
                f"Writable destination would expose the repository: {destination}"
            )
    run_root.mkdir(parents=True, exist_ok=False)
    for path in [
        logs,
        run_root / "runtime",
        run_root / "toy",
        run_root / "graha",
        cache,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    for stage in (*STAGES, "prepare"):
        for name in ("tmp", "matplotlib", "home"):
            (cache / name / stage).mkdir(parents=True, exist_ok=True)
    resources = {
        "training": {
            "memory": request.training.memory,
            "cpus": request.training.cpus,
            "time": request.training.time,
        },
        "comparison": {
            "memory": request.comparison.memory,
            "cpus": request.comparison.cpus,
            "time": request.comparison.time,
        },
    }
    state = {
        "schema_version": 1,
        "status": "preparing",
        "created_at": datetime.now(UTC).isoformat(),
        "repository": repository,
        "container": {
            "supplied": supplied_container,
            "path": str(container),
            "identity": request.container.identity,
        },
        "run_root": str(run_root),
        "configuration": str(run_root / "experiment.json"),
        "split_manifest": str(run_root / "split_manifest.json"),
        "config_source": str(request.paths.config_source.expanduser().resolve())
        if request.paths.config_source
        else None,
        "preparation_settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in settings.items()
        },
        "cache_dir": str(cache),
        "log_dir": str(logs),
        "pipeline_log": str(run_root / "pipeline.log"),
        "inputs": [str(path) for path in inputs],
        "scheduler": {
            "executable": request.scheduler.executable,
            "partition": request.scheduler.partition,
            "account": request.scheduler.account,
            "qos": request.scheduler.qos,
            "gpus": 1,
            "resources": resources,
        },
        "apptainer": request.container.executable,
        "max_epochs": request.preparation.max_epochs,
        "inference": {
            "batch_size": request.inference.batch_size,
            "n_samples": request.inference.n_samples,
        },
        "jobs": {},
        "dependency": None,
    }
    manifest = run_root / "submission.json"
    submission = _Submission(manifest, state)
    submission.save()
    pipeline_log = Path(state["pipeline_log"])
    log_pipeline(
        pipeline_log,
        "submit",
        f"Experiment: {run_root}; max_epochs={request.preparation.max_epochs}",
    )
    print(f"Follow progress: tail -F -- {shlex.quote(str(pipeline_log))}", flush=True)
    stage = None
    try:
        state["preparation_command"] = submission.command("prepare")
        submission.save()
        returncode = run_logged_process(
            state["preparation_command"],
            pipeline_log,
            "prepare",
            env=container_host_environment(),
        )
        if returncode:
            raise RuntimeError(f"Container preparation failed ({returncode}).")
        configuration = read_json(state["configuration"])
        _readable(Path(state["split_manifest"]))
        state["config_id"] = configuration["config_id"]
        state["split_manifest_id"] = configuration["split_manifest_id"]
        state["status"] = "prepared"
        submission.save()
        log_pipeline(
            pipeline_log,
            "submit",
            "Configuration and shared splits prepared; submitting training jobs",
        )
        for stage in STAGES:
            if stage == "comparison":
                state["dependency"] = "afterok:" + ":".join(
                    state["jobs"][model]["job_id"] for model in ("toy", "graha")
                )
            resource = resources["comparison" if stage == "comparison" else "training"]
            command = [
                request.scheduler.executable,
                "--parsable",
                f"--job-name=iseg_{stage}",
                f"--partition={request.scheduler.partition}",
                "--gpus=1",
                "--ntasks=1",
                f"--mem={resource['memory']}",
                f"--cpus-per-task={resource['cpus']}",
                f"--time={resource['time']}",
                f"--chdir={REPO_ROOT}",
                f"--output={logs / (stage + '_%j.out')}",
                f"--error={logs / (stage + '_%j.err')}",
            ]
            for option in ("account", "qos"):
                if getattr(request.scheduler, option):
                    command.append(f"--{option}={getattr(request.scheduler, option)}")
            if stage == "comparison":
                command.append("--dependency=" + state["dependency"])
            command.extend(
                [
                    str(
                        REPO_ROOT
                        / "scripts/shell/instance_seg/sbatch_instance_grace_stage.sh"
                    ),
                    sys.executable,
                    str(REPO_ROOT),
                    str(manifest),
                    stage,
                ]
            )
            job = {"status": "submitting", "command": command, "job_id": None}
            state["jobs"][stage] = job
            state["status"] = "submitting"
            submission.save()
            log_pipeline(
                pipeline_log,
                "submit",
                f"Submitting {stage}"
                + (
                    f"; dependency={state['dependency']}"
                    if stage == "comparison"
                    else ""
                ),
            )
            response = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            job.update(
                returncode=response.returncode,
                stdout=response.stdout,
                stderr=response.stderr,
            )
            # Even on a scheduler error retain any parsable ID that was returned.
            raw_id = response.stdout.strip()
            if re.fullmatch(r"[0-9]+(?:;[^\s;]+)?", raw_id):
                job["job_id"] = raw_id.split(";", 1)[0]
                job["scheduler_id"] = raw_id
            submission.save()
            if response.stderr.strip():
                log_pipeline(
                    pipeline_log,
                    "submit",
                    f"{stage} scheduler stderr: {response.stderr.strip()}",
                )
            if response.returncode:
                raise RuntimeError(
                    f"{stage} submission failed ({response.returncode}): {response.stderr.strip()}"
                )
            if job["job_id"] is None:
                raise RuntimeError(
                    f"{stage} returned an invalid parsable scheduler ID: {raw_id!r}"
                )
            job["status"] = "submitted"
            submission.save()
            log_pipeline(
                pipeline_log,
                "submit",
                f"Submitted {stage}: job={job['job_id']}"
                + (
                    f"; waiting for {state['dependency']}"
                    if stage == "comparison"
                    else "; awaiting allocation"
                ),
            )
        state["status"] = "submitted"
        submission.save()
        log_pipeline(
            pipeline_log,
            "submit",
            "All three jobs submitted; waiting for their allocations and completion",
        )
    except BaseException as exc:
        state["status"] = "submission_failed"
        state["error"] = {
            "stage": stage or "prepare",
            "type": type(exc).__name__,
            "message": str(exc),
        }
        if stage:
            state["jobs"][stage]["status"] = "submission_failed"
        submission.save()
        log_pipeline(
            pipeline_log,
            "submit",
            f"Submission FAILED at {stage or 'prepare'}: {type(exc).__name__}: {exc}",
        )
        print(f"Submission state retained at {manifest}", file=sys.stderr, flush=True)
        raise
    return SubmissionReceipt(
        manifest,
        pipeline_log,
        {stage: job["job_id"] for stage, job in state["jobs"].items()},
    )


def container_host_environment() -> dict[str, str]:
    # --cleanenv does not suppress APPTAINERENV_* or implicit bind variables.
    # Remove injection channels before exec, without changing the job's CUDA mask.
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("APPTAINER", "SINGULARITY"))
    }


def launch(submission: Path, stage: Stage) -> int:
    saved = _Submission.load(submission)
    state = saved.payload
    pipeline_log = Path(
        state.get("pipeline_log", str(submission.parent / "pipeline.log"))
    )
    returncode = run_logged_process(
        saved.command(stage),
        pipeline_log,
        stage,
        env=container_host_environment(),
    )
    if stage == "comparison" and returncode == 0:
        log_pipeline(
            pipeline_log,
            stage,
            f"Pipeline complete; results: {state['run_root']}/comparison/comparison.json; figure: {state['run_root']}/comparison/test_comparison.png",
        )
    elif stage in ("toy", "graha") and returncode != 0:
        log_pipeline(
            pipeline_log,
            stage,
            "Training failed; the afterok comparison cannot start until both parents succeed",
        )
    return returncode


def run_allocated(stage: Stage, metadata_path: Path, arguments: Sequence[str]) -> None:
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    metadata = {
        "stage": stage,
        "started_at": datetime.now(UTC).isoformat(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "nodes": os.environ.get("SLURM_JOB_NODELIST"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "arguments": arguments,
        "status": "starting",
    }
    write_json(metadata_path, metadata)
    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("CUDA is not visible inside the allocated container.")
        properties = torch.cuda.get_device_properties(0)
        metadata["device"] = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "visible_count": torch.cuda.device_count(),
        }
        metadata["status"] = "running"
        write_json(metadata_path, metadata)
        print(
            f"CUDA ready: {properties.name}; memory_bytes={properties.total_memory}; visible_devices={metadata['device']['visible_count']}",
            flush=True,
        )
        script = (
            "compare_instance_experiment.py"
            if stage == "comparison"
            else "train_instance_experiment.py"
        )
        script_path = REPO_ROOT / "scripts/python/instance_seg" / script
        original_argv = sys.argv
        try:
            sys.argv = [str(script_path), *arguments]
            runpy.run_path(str(script_path), run_name="__main__")
        finally:
            sys.argv = original_argv
    except BaseException as exc:
        metadata.update(
            status="failed", error={"type": type(exc).__name__, "message": str(exc)}
        )
        write_json(metadata_path, metadata)
        raise
    metadata.update(status="complete", completed_at=datetime.now(UTC).isoformat())
    write_json(metadata_path, metadata)
