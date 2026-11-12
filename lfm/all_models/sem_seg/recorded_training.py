"""Explicit semantic adoption of recorded training, preserving legacy workflows.

The existing semantic entry points retain their weights-only checkpoints and
final-model evaluation. This opt-in path records full training state and selected
validation loss; it performs no test evaluation automatically.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace

from lfm.all_models.all_tasks.model_adapters import resolve_model_adapter
from lfm.all_models.recorded_training import (
    Pretrained,
    Resume,
    RunSpec,
    TrainingAttempt,
    TrainingRun,
    WeightsOnly,
)
from lfm.all_models.sem_seg.config import SemanticSegmentationExperimentConfig
from lfm.artifacts import json_ready


def begin_training(
    config: SemanticSegmentationExperimentConfig,
    spec: RunSpec,
    *,
    start: Pretrained | WeightsOnly | Resume | None = None,
) -> TrainingAttempt:
    if spec.task != "semantic":
        raise ValueError("Semantic training requires a semantic RunSpec.")
    if (
        config.toy_lightning_checkpoint is not None
        or config.graha_lightning_checkpoint is not None
    ):
        raise ValueError(
            "Recorded training requires explicit WeightsOnly or Resume, with legacy checkpoint fields unset."
        )
    config = replace(config, max_epochs=spec.max_epochs)
    adapter = resolve_model_adapter(spec.model, spec.task)
    selection = adapter.recorded_selection
    config_id = hashlib.sha256(
        json.dumps(json_ready(asdict(config)), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    identity = {"schema_version": 1, "config_id": config_id, **asdict(spec)}
    pretrained = (
        config.dino_checkpoint if spec.model == "toy" else config.graha_pretrain_dir
    )
    run = TrainingRun(
        spec,
        config.base_output_dir,
        config.seed,
        config,
        identity,
        {
            "task": "semantic",
            "config_id": config_id,
            "architecture": "dino-semantic"
            if spec.model == "toy"
            else "graha-semantic",
            "selection": {
                "metric": selection.metric,
                "mode": selection.mode,
                "split": "val",
            },
        },
        {
            "mode": "pretrained",
            "checkpoint": str(pretrained) if pretrained is not None else None,
        },
        selection,
    )
    return TrainingAttempt(run, adapter, start=start)
