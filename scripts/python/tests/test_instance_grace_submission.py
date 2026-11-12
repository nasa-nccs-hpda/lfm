"""Local scheduler/container doubles; these checks never submit cluster jobs."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from lfm.all_models import experiment_execution as execution
from lfm.all_models.inst_seg import grace_experiment as grace


@pytest.fixture
def cluster(tmp_path):
    # Deliberately invalid contents establish that preparation only pairs names.
    data = tmp_path / "dataset with spaces"
    for split in ("train", "val", "test"):
        chips, labels = data / split / "chips", data / split / "labels"
        chips.mkdir(parents=True)
        labels.mkdir(parents=True)
        (chips / "one_chip.tif").write_text("not raster data")
        (labels / "one_label.npz").write_text("not label data")
    dino = tmp_path / "dino.pth"
    dino.write_text("not weights")
    pretrain = tmp_path / "pretrain"
    (pretrain / "checkpoints").mkdir(parents=True)
    for name in (
        "full_config.yaml",
        "modality_info.yaml",
        "checkpoints/checkpoint_weights_final.pt",
    ):
        (pretrain / name).write_text("not pretrained data")
    container = tmp_path / "container.sif"
    container.touch()
    scheduler = tmp_path / "sbatch"
    scheduler.write_text(
        f"#!{sys.executable}\n"
        + """
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
manifest = Path(args[-2])
state = json.loads(manifest.read_text())
root = Path(state["run_root"])
assert (root / "experiment.json").is_file()
assert (root / "split_manifest.json").is_file()
assert all((root / name).is_dir() for name in ("toy", "graha", "runtime"))
for arg in args:
    if arg.startswith(("--output=", "--error=")):
        assert Path(arg.split("=", 1)[1]).parent.is_dir()
calls = manifest.parent / "scheduler_calls.json"
history = json.loads(calls.read_text()) if calls.exists() else []
history.append(args)
calls.write_text(json.dumps(history))
number = len(history)
if number == int(os.environ.get("FAIL_SUBMISSION", "0")):
    print("account rejected", file=sys.stderr)
    sys.exit(17)
if number == int(os.environ.get("BAD_ID_SUBMISSION", "0")):
    print("unexpected scheduler response")
else:
    print(f"{1000 + number};grace-cluster")
"""
    )
    scheduler.chmod(0o755)
    apptainer = tmp_path / "apptainer"
    apptainer.write_text(
        f"#!{sys.executable}\n"
        + """
import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
assert "--cleanenv" in args
assert not any(key.startswith(("APPTAINER", "SINGULARITY")) for key in os.environ)
env = dict(os.environ)
for index, arg in enumerate(args):
    if arg == "--env":
        key, value = args[index + 1].split("=", 1)
        env[key] = value
python = args.index("python")
if Path(args[python + 2]).name in {
    "submit_instance_grace_experiment.py", "run_instance_grace_stage.py"
}:
    # Controller bootstrap uses container Python; its bridge reaches the host
    # doubles below without attempting nested container execution.
    sys.exit(subprocess.run([sys.executable, *args[python + 1:]], env=env).returncode)
calls = Path(os.environ["CONTAINER_CALLS"])
history = json.loads(calls.read_text()) if calls.exists() else []
history.append(args)
calls.write_text(json.dumps(history))
if "prepare_instance_experiment.py" in args[python + 2]:
    assert "--nv" not in args
    sys.exit(subprocess.run([sys.executable, *args[python + 1:]], env=env).returncode)
assert "--nv" in args
print("model stage output", flush=True)
if os.environ.get("GPU_STAGE_STATUS", "0") != "0":
    print("CUDA out of memory", file=sys.stderr, flush=True)
sys.exit(int(os.environ.get("GPU_STAGE_STATUS", "0")))
"""
    )
    apptainer.chmod(0o755)
    return SimpleNamespace(
        root=tmp_path / "run with spaces",
        data=data,
        dino=dino,
        pretrain=pretrain,
        container=container,
        scheduler=scheduler,
        apptainer=apptainer,
        container_calls=tmp_path / "container_calls.json",
    )


def _arguments(cluster):
    return [
        "--run-root",
        str(cluster.root),
        "--data-root",
        str(cluster.data),
        "--dino-checkpoint",
        str(cluster.dino),
        "--graha-pretrain-dir",
        str(cluster.pretrain),
        "--container",
        str(cluster.container),
        "--sbatch",
        str(cluster.scheduler),
        "--apptainer",
        str(cluster.apptainer),
    ]


def _environment(cluster, **overrides):
    return {
        **os.environ,
        "CONTAINER_CALLS": str(cluster.container_calls),
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(cluster.root.parent / "host-mpl"),
        **overrides,
    }


def _submit(cluster, *arguments, **environment):
    return subprocess.run(
        [
            "bash",
            str(
                grace.REPO_ROOT
                / "scripts/shell/instance_seg/submit_instance_grace_experiment.sh"
            ),
            *_arguments(cluster),
            *arguments,
        ],
        env=_environment(cluster, **environment),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _state(cluster):
    return json.loads((cluster.root / "submission.json").read_text())


def test_one_epoch_submission_and_all_allocated_container_commands(cluster):
    result = _submit(cluster, "--max-epochs", "1", "--container-identity", "my-build")
    assert result.returncode == 0, result.stderr
    state = _state(cluster)
    calls = json.loads((cluster.root / "scheduler_calls.json").read_text())
    assert len(calls) == 3
    assert [call[-1] for call in calls] == ["toy", "graha", "comparison"]
    assert state["status"] == "submitted"
    assert state["dependency"] == "afterok:1001:1002"
    assert [state["jobs"][stage]["job_id"] for stage in execution.STAGES] == [
        "1001",
        "1002",
        "1003",
    ]
    assert state["jobs"]["toy"]["scheduler_id"] == "1001;grace-cluster"
    assert all(
        not any(item.startswith("--dependency=") for item in call) for call in calls[:2]
    )
    assert "--dependency=afterok:1001:1002" in calls[2]
    for call in calls:
        assert (
            "--parsable" in call and "--gpus=1" in call and "--partition=grace" in call
        )
        assert (
            "--mem=128G" in call
            and "--cpus-per-task=10" in call
            and "--time=24:00:00" in call
        )
    config = json.loads(Path(state["configuration"]).read_text())
    assert state["max_epochs"] == 1
    for model, batch_key in (("toy", "toy_batch_size"), ("graha", "graha_batch_size")):
        resolved = config["models"][model]
        assert resolved["max_epochs"] == 1 and resolved[batch_key] == 32
        assert resolved["band_filter"] == list(range(70))
        assert all(
            resolved[f"max_{split}_samples"] is None
            for split in ("train", "val", "test")
        )
    assert state["repository"]["revision"]
    assert isinstance(state["repository"]["dirty"], bool)
    assert state["container"]["identity"] == "my-build"
    assert state["inference"]["batch_size"] == 32
    assert state["pipeline_log"] == str(cluster.root / "pipeline.log")
    assert "tail -F" in result.stdout
    # Execute the actual sbatch wrapper on each recorded command. The Apptainer
    # double stops at the container boundary; no GPU test or training is run.
    environment = _environment(
        cluster,
        CUDA_VISIBLE_DEVICES="GPU-assigned-by-slurm",
        SLURM_JOB_ID="1001",
        APPTAINERENV_PYTHONPATH="/host/conda",
        SINGULARITYENV_GDAL_DATA="/host/gdal",
        APPTAINER_BIND="/host:/leak",
        PROJ_LIB="/host/proj",
        PYTHONPATH="/host/python",
    )
    for call in calls:
        script = next(
            index
            for index, item in enumerate(call)
            if item.endswith("sbatch_instance_grace_stage.sh")
        )
        launched = subprocess.run(
            ["bash", *call[script:]],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert launched.returncode == 0, launched.stderr
    containers = json.loads(cluster.container_calls.read_text())
    central = Path(state["pipeline_log"]).read_text()
    assert "[prepare]" in central and "Prepared experiment" in central
    assert (
        "Submitted toy: job=1001" in central
        and "waiting for afterok:1001:1002" in central
    )
    for stage in execution.STAGES:
        assert f"[{stage} job=1001] model stage output" in central
    assert "Pipeline complete; results:" in central
    assert len(containers) == 4  # CPU preparation + exactly three GPU launches.
    for index, command in enumerate(containers[1:]):
        assert "--cleanenv" in command and "--nv" in command
        assert "MPLBACKEND=Agg" in command
        assert "CUDA_VISIBLE_DEVICES=GPU-assigned-by-slurm" in command
        assert "SLURM_JOB_ID=1001" in command
        binds = [command[i + 1] for i, arg in enumerate(command) if arg == "--bind"]
        for path in (
            grace.REPO_ROOT,
            cluster.data,
            cluster.dino,
            cluster.pretrain,
            Path(state["configuration"]),
            Path(state["split_manifest"]),
        ):
            assert f"{path}:{path}:ro" in binds
        assert f"{cluster.root}:{cluster.root}:rw" in binds
        assert not any("/host/" in arg for arg in command)
        if index < 2:
            assert command[command.index("--max-epochs") + 1] == "1"
            assert command[command.index("--model") + 1] == ("toy", "graha")[index]
        else:
            assert command[command.index("--batch-size") + 1] == "32"
            assert command[command.index("--toy-result") + 1] == str(
                cluster.root / "toy/result.json"
            )
            assert command[command.index("--graha-result") + 1] == str(
                cluster.root / "graha/result.json"
            )


def test_resource_config_and_independent_inference_overrides(cluster):
    source = cluster.root.parent / "settings.json"
    source.write_text(json.dumps({"max_epochs": 7, "seed": 19, "toy_batch_size": 12}))
    result = _submit(
        cluster,
        "--config",
        str(source),
        "--max-epochs",
        "2",
        "--account",
        "science",
        "--qos",
        "long",
        "--train-memory",
        "200G",
        "--train-cpus",
        "12",
        "--train-time",
        "12:00:00",
        "--comparison-memory",
        "64G",
        "--comparison-cpus",
        "4",
        "--comparison-time",
        "02:00:00",
        "--inference-batch-size",
        "8",
        "--log-dir",
        str(cluster.root.parent / "external logs"),
        "--cache-dir",
        str(cluster.root.parent / "external cache"),
    )
    assert result.returncode == 0, result.stderr
    state = _state(cluster)
    for stage in ("toy", "graha"):
        command = state["jobs"][stage]["command"]
        assert all(
            option in command
            for option in (
                "--mem=200G",
                "--cpus-per-task=12",
                "--time=12:00:00",
                "--account=science",
                "--qos=long",
            )
        )
    comparison = state["jobs"]["comparison"]["command"]
    assert all(
        option in comparison
        for option in ("--mem=64G", "--cpus-per-task=4", "--time=02:00:00")
    )
    assert state["max_epochs"] == 2
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CONTAINER_CALLS", str(cluster.container_calls))
        assert execution.launch(cluster.root / "submission.json", "comparison") == 0
    command = json.loads(cluster.container_calls.read_text())[-1]
    assert command[command.index("--batch-size") + 1] == "8"
    configuration = json.loads(Path(state["configuration"]).read_text())
    assert configuration["models"]["toy"]["toy_batch_size"] == 12
    assert configuration["models"]["toy"]["seed"] == 19
    assert state["config_source"] == str(source)


@pytest.mark.parametrize("failure", [1, 2, 3])
def test_partial_scheduler_failure_preserves_ids_responses_and_dependency(
    cluster, failure
):
    result = _submit(cluster, FAIL_SUBMISSION=str(failure))
    assert result.returncode != 0
    assert "account rejected" in result.stderr
    state = _state(cluster)
    assert state["max_epochs"] == 50
    assert state["status"] == "submission_failed"
    assert state["error"]["stage"] == execution.STAGES[failure - 1]
    assert len(state["jobs"]) == failure
    for index, stage in enumerate(execution.STAGES[: failure - 1], start=1):
        assert state["jobs"][stage]["job_id"] == str(1000 + index)
        assert state["jobs"][stage]["status"] == "submitted"
    failed = state["jobs"][execution.STAGES[failure - 1]]
    assert failed["returncode"] == 17 and "account rejected" in failed["stderr"]
    assert state["dependency"] == ("afterok:1001:1002" if failure == 3 else None)
    central = Path(state["pipeline_log"]).read_text()
    assert "Submission FAILED" in central and "account rejected" in central
    assert (
        len(json.loads((cluster.root / "scheduler_calls.json").read_text())) == failure
    )


def test_invalid_scheduler_response_is_retained_and_stops_submission(cluster):
    result = _submit(cluster, BAD_ID_SUBMISSION="2")
    assert result.returncode != 0
    state = _state(cluster)
    assert state["jobs"]["toy"]["job_id"] == "1001"
    assert state["jobs"]["graha"]["stdout"] == "unexpected scheduler response\n"
    assert "comparison" not in state["jobs"]


def test_container_preparation_failure_submits_nothing(cluster):
    (cluster.data / "train/labels/one_label.npz").unlink()
    result = _submit(cluster)
    assert (
        result.returncode != 0 and "no matching label" in result.stdout + result.stderr
    )
    assert _state(cluster)["jobs"] == {}
    assert "no matching label" in (cluster.root / "pipeline.log").read_text()
    assert not (cluster.root / "scheduler_calls.json").exists()


def test_allocated_container_failure_is_logged_without_pipeline_success(cluster):
    result = _submit(cluster)
    assert result.returncode == 0, result.stderr
    command = _state(cluster)["jobs"]["toy"]["command"]
    script = next(
        index
        for index, arg in enumerate(command)
        if arg.endswith("sbatch_instance_grace_stage.sh")
    )
    result = subprocess.run(
        ["bash", *command[script:]],
        env=_environment(cluster, GPU_STAGE_STATUS="137", SLURM_JOB_ID="1001"),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 137
    central = (cluster.root / "pipeline.log").read_text()
    assert "CUDA out of memory" in central and "FAILED (exit=137)" in central
    assert (
        "afterok comparison cannot start" in central
        and "Pipeline complete" not in central
    )


def test_host_help_and_epoch_validation_use_only_standard_library(tmp_path):
    script = (
        grace.REPO_ROOT
        / "scripts/python/instance_seg/submit_instance_grace_experiment.py"
    )
    result = subprocess.run(
        [sys.executable, "-S", str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--max-epochs" in result.stdout
    assert grace.parse_submission_args([]).max_epochs == 50
    for value in ("0", "-1", "1.5"):
        with pytest.raises(SystemExit):
            grace.parse_submission_args(["--max-epochs", value])
    settings = tmp_path / "settings.json"
    settings.write_text('{"max_epochs": 0}')
    with pytest.raises(SystemExit):
        grace.parse_submission_args(["--config", str(settings)])


@pytest.mark.parametrize("stage", execution.STAGES)
@pytest.mark.parametrize("failure", [None, "cuda", "oom"])
def test_gpu_entry_records_device_forwards_arguments_and_surfaces_failures(
    tmp_path, monkeypatch, stage, failure
):
    script = (
        grace.REPO_ROOT / "scripts/python/instance_seg/run_instance_grace_gpu_stage.py"
    )
    spec = importlib.util.spec_from_file_location("gpu_stage", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cuda = SimpleNamespace(
        is_available=lambda: failure != "cuda",
        device_count=lambda: 1,
        get_device_properties=lambda index: SimpleNamespace(
            name="Assigned H100", total_memory=80_000_000_000
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    arguments = (
        ["--run-root", "/run", "--model", stage, "--max-epochs", "1"]
        if stage != "comparison"
        else [
            "--toy-result",
            "/toy/result.json",
            "--graha-result",
            "/graha/result.json",
            "--output-dir",
            "/comparison",
            "--batch-size",
            "8",
        ]
    )
    metadata = tmp_path / "metadata.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(script), "--metadata", str(metadata), "--stage", stage, *arguments],
    )
    entered = []

    def run(path, run_name):
        entered.append((Path(path).name, sys.argv[1:]))
        assert run_name == "__main__"
        if failure == "oom":
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(execution.runpy, "run_path", run)
    if failure:
        with pytest.raises(
            RuntimeError, match="not visible" if failure == "cuda" else "out of memory"
        ):
            module.main()
    else:
        module.main()
    state = json.loads(metadata.read_text())
    assert state["arguments"] == arguments
    assert state["status"] == ("failed" if failure else "complete")
    if failure != "cuda":
        assert entered == [
            (
                "compare_instance_experiment.py"
                if stage == "comparison"
                else "train_instance_experiment.py",
                arguments,
            )
        ]
        assert state["device"] == {
            "name": "Assigned H100",
            "total_memory_bytes": 80_000_000_000,
            "visible_count": 1,
        }
    else:
        assert entered == []
