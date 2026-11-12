"""Instance experiment adapter for shared recorded training and preparation."""

from __future__ import annotations

import argparse
import gc
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from lfm.all_models.all_tasks.model_adapters import resolve_model_adapter
from lfm.all_models.inst_seg.mask_ap import MASK_AP_KEY, MAX_DETECTIONS
from lfm.all_models.inst_seg.paired_experiment import (
    InstanceSplitManifest,
    PairedInstanceExperiment,
    load_experiment,
    positive_integer,
)
from lfm.all_models.recorded_training import (
    CompletedTraining,
    PreparedModel,
    Pretrained,
    Resume,
    RunSpec,
    SelectionPolicy,
    TrainingAttempt,
    TrainingRun,
    WeightsOnly,
)

STATE_KEY = "paired_instance_training"
INSTANCE_SELECTION = SelectionPolicy(MASK_AP_KEY, "max", 0, 1)


def training_identity(
    experiment: PairedInstanceExperiment,
    manifest: InstanceSplitManifest,
    model: str,
    max_epochs: int,
) -> dict:
    if model not in {"toy", "graha"}:
        raise ValueError(f"Unsupported model: {model}")
    return {
        "schema_version": 1,
        "config_id": experiment.config_id,
        "split_manifest_id": manifest.manifest_id,
        "model": model,
        "max_epochs": max_epochs,
        "sampler": {
            "method": "epoch_seeded_randperm",
            "seed": getattr(experiment, model).seed,
            "resume_boundary": "completed_epoch",
        },
    }


def describe_training(
    experiment, manifest, model: str, max_epochs: int | None = None
) -> TrainingRun:
    if model not in ("toy", "graha"):
        raise ValueError(f"Unsupported model: {model}")
    config = getattr(experiment, model)
    spec = RunSpec(
        model, "instance", config.max_epochs if max_epochs is None else max_epochs
    )
    config = replace(config, max_epochs=spec.max_epochs)
    payload = experiment.payload()
    metadata = {
        "architecture": "dino-mask-rcnn" if model == "toy" else "graha-mask-rcnn",
        "config_id": experiment.config_id,
        "split_manifest_id": manifest.manifest_id,
        "band_layout": payload["band_layout"],
        "feature_pyramid": payload["feature_pyramids"][model],
        "normalization": payload["normalization"][model],
        "validity_policy": {
            "footprint": "all_selected_bands_valid",
            "empty_crop": "skip_during_loading",
            "targets": "clip_masks_drop_empty_recompute_boxes",
            "invalid_mask_loss_pixels": "excluded",
            "image_nodata_policy": config.image_nodata_policy,
            "excluded_nodata_values": config.excluded_nodata_values,
            "eligibility_reports": str(config.base_output_dir / "eligibility"),
        },
        "selection": {
            "metric": MASK_AP_KEY,
            "mode": "max",
            "split": "val",
            "max_detections": MAX_DETECTIONS,
        },
        "sampler": {
            "method": "epoch_seeded_randperm",
            "seed": config.seed,
            "epoch_seed": "seed + zero_based_epoch",
            "workers": "deterministic preprocessing; separate seed generator",
            "resume_boundary": "completed_epoch",
        },
    }
    return TrainingRun(
        spec,
        config.base_output_dir,
        config.seed,
        config,
        training_identity(experiment, manifest, model, spec.max_epochs),
        metadata,
        {"mode": "pretrained", **payload["initialization"][model]},
        INSTANCE_SELECTION,
        STATE_KEY,
        "InstanceTrainingCheckpoint",
    )


class InstancePreparation:
    def __init__(
        self, experiment: PairedInstanceExperiment, membership: InstanceSplitManifest
    ):
        self.experiment, self.membership = experiment, membership

    def prepare(
        self, run: TrainingRun, initial_epoch: int, owner: object
    ) -> PreparedModel:
        from lfm.all_models.inst_seg.paired_experiment import (
            create_experiment_datamodule,
        )

        # Use the validated effective configuration for data as well as model.
        dm = create_experiment_datamodule(
            self.experiment,
            self.membership,
            run.spec.model,
            initial_epoch=initial_epoch,
            configuration=run.configuration,
        )
        task = resolve_model_adapter(run.spec.model, "instance").create_recorded_task(
            run.configuration, dm
        )
        return PreparedModel(run, task, dm, owner)


def begin_training(
    run_root: str | Path,
    model: str,
    *,
    max_epochs: int | None = None,
    start: Pretrained | WeightsOnly | Resume | None = None,
) -> TrainingAttempt:
    experiment, membership = load_experiment(run_root)
    run = describe_training(experiment, membership, model, max_epochs)
    return TrainingAttempt(
        run, InstancePreparation(experiment, membership), start=start
    )


def train_prepared_experiment(
    run_root: str | Path,
    model: str,
    *,
    max_epochs: int | None = None,
    resume: str | Path | None = None,
    initialize_weights: str | Path | None = None,
) -> Path:
    if resume is not None and initialize_weights is not None:
        raise ValueError("Choose explicit resume or weights-only initialization.")
    start = (
        Resume(Path(resume))
        if resume is not None
        else WeightsOnly(Path(initialize_weights))
        if initialize_weights is not None
        else Pretrained()
    )
    attempt = begin_training(run_root, model, max_epochs=max_epochs, start=start)
    print(
        f"Loading {model} data and pretrained model; max_epochs={attempt.run.spec.max_epochs}; output={attempt.run.output_dir}",
        flush=True,
    )
    prepared = attempt.prepare()
    print(f"{model} data and model loaded; starting training", flush=True)
    return attempt.fit(prepared).path


def load_completed_training(
    path: str | Path, model: str
) -> tuple[CompletedTraining, InstancePreparation]:
    from lfm.artifacts import read_json

    path = Path(path).expanduser().resolve()
    experiment, membership = load_experiment(path.parent.parent)
    budget = read_json(path).get("max_epochs")
    run = describe_training(experiment, membership, model, budget)
    return CompletedTraining.load(path, run), InstancePreparation(
        experiment, membership
    )


def create_training_parser():
    parser = argparse.ArgumentParser(
        description="Train one model from a prepared paired experiment."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model", choices=("toy", "graha"), required=True)
    parser.add_argument(
        "--max-epochs",
        type=positive_integer,
        default=None,
        help="Total target epochs; defaults to prepared configuration (50). Must match the original budget on resume.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        type=Path,
        help="Restore full training state from this explicit checkpoint.",
    )
    mode.add_argument(
        "--initialize-weights",
        type=Path,
        help="Strictly load model weights for a fresh run with new optimizer/progress.",
    )
    return parser


@dataclass(frozen=True)
class CompletedPair:
    toy: CompletedTraining
    graha: CompletedTraining
    preparation: dict[str, InstancePreparation]


def load_completed_pair(
    toy_result: str | Path, graha_result: str | Path
) -> CompletedPair:
    toy, toy_preparation = load_completed_training(toy_result, "toy")
    graha, graha_preparation = load_completed_training(graha_result, "graha")
    if any(
        toy.payload[key] != graha.payload[key]
        for key in ("config_id", "split_manifest_id", "max_epochs")
    ):
        raise ValueError(
            "Model results belong to mixed experiments/configurations/splits or epoch budgets."
        )
    records = toy_preparation.membership.splits["test"]
    if len({record.chip_id for record in records}) != len(records):
        raise ValueError("Ambiguous duplicate test chip identifiers.")
    return CompletedPair(
        toy, graha, {"toy": toy_preparation, "graha": graha_preparation}
    )


@contextmanager
def prepare_for_test(
    completed: CompletedTraining,
    preparation: InstancePreparation,
    *,
    batch_size: int = 32,
    eligibility_dir: Path,
    device,
):
    from lfm.all_models.inst_seg.paired_experiment import create_experiment_datamodule

    membership = replace(
        preparation.membership,
        splits={
            **preparation.membership.splits,
            "test": sorted(
                preparation.membership.splits["test"], key=lambda item: item.chip_id
            ),
        },
    )
    gc.collect()
    task = None
    try:
        with completed.best_weights() as weights:
            dm = create_experiment_datamodule(
                preparation.experiment,
                membership,
                completed.run.spec.model,
                configuration=completed.run.configuration,
                include_raw_image=True,
                batch_size=batch_size,
                eligibility_dir=eligibility_dir,
            )
            task = resolve_model_adapter(
                completed.run.spec.model, "instance"
            ).create_recorded_task(completed.run.configuration, dm)
            task.load_state_dict(weights, strict=True)
        task.to(device).eval()
        yield PreparedModel(completed.run, task, dm, completed)
    finally:
        if task is not None:
            task.to("cpu")
        del task
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
