"""Compatible CLI settings for the paired Grace experiment."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

from lfm.all_models.experiment_execution import (
    DEFAULT_CONTAINER,
    REPO_ROOT,
    ContainerSettings,
    InferenceSettings,
    PreparationSettings,
    Resources,
    SchedulerSettings,
    SubmissionPaths,
    SubmissionRequest,
)
from lfm.all_models.inst_seg.paired_experiment_cli import (
    create_paired_experiment_parser,
    positive_integer,
)


def create_submission_parser():
    parser = create_paired_experiment_parser(require_run_root=False)
    parser.description = (
        "Submit two Grace container training jobs and an afterok comparison."
    )
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "scripts/outputs/instance_grace"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON defaults using preparation option names; CLI values override them.",
    )
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument(
        "--container-identity",
        help="Optional supplied container tag/build identity (no hashing).",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--account")
    parser.add_argument("--qos")
    parser.add_argument("--partition", default="grace")
    parser.add_argument("--train-memory", default="128G")
    parser.add_argument("--train-cpus", type=positive_integer, default=10)
    parser.add_argument("--train-time", default="24:00:00")
    parser.add_argument("--comparison-memory", default="128G")
    parser.add_argument("--comparison-cpus", type=positive_integer, default=10)
    parser.add_argument("--comparison-time", default="24:00:00")
    parser.add_argument("--inference-batch-size", type=positive_integer, default=32)
    parser.add_argument("--n-samples", type=positive_integer, default=5)
    parser.add_argument(
        "--sbatch",
        default="sbatch",
        help="Scheduler executable (also supports local fake schedulers).",
    )
    parser.add_argument("--apptainer", default="apptainer")
    return parser


def parse_submission_args(argv=None):
    """Validate file defaults through the same argparse types as CLI options."""
    parser = create_submission_parser()
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path)
    config, _ = bootstrap.parse_known_args(argv)
    defaults = []
    if config.config:
        payload = json.loads(config.config.read_text())
        allowed = {
            action.dest for action in create_paired_experiment_parser()._actions
        } - {"help", "run_root"}
        if not isinstance(payload, dict) or set(payload) - allowed:
            parser.error(
                "--config must be an object containing preparation option names (except run_root)"
            )
        for key, value in payload.items():
            if value is None:
                continue
            defaults.append("--" + key.replace("_", "-"))
            defaults.extend(
                str(item) for item in (value if isinstance(value, list) else [value])
            )
    return parser.parse_args(defaults + list(sys.argv[1:] if argv is None else argv))


def submission_request(args) -> SubmissionRequest:
    return SubmissionRequest(
        preparation=PreparationSettings(
            **{
                item.name: getattr(args, item.name)
                for item in fields(PreparationSettings)
                if hasattr(args, item.name)
            }
        ),
        paths=SubmissionPaths(
            args.run_root, args.output_root, args.cache_dir, args.log_dir, args.config
        ),
        container=ContainerSettings(
            args.container, args.container_identity, args.apptainer
        ),
        scheduler=SchedulerSettings(
            args.sbatch, args.partition, args.account, args.qos
        ),
        training=Resources(args.train_memory, args.train_cpus, args.train_time),
        comparison=Resources(
            args.comparison_memory, args.comparison_cpus, args.comparison_time
        ),
        inference=InferenceSettings(args.inference_batch_size, args.n_samples),
    )
