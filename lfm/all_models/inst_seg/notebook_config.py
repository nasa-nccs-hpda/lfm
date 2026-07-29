"""Notebook-facing instance segmentation configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lfm.all_models.inst_seg.config import InstanceSegmentationExperimentConfig
from lfm.all_models.inst_seg.config import build_config as build_experiment_config
from lfm.full_model.inst_seg.instance_model_adapter import GrahaInstanceModelAdapter


@dataclass(frozen=True)
class GrahaInstanceNotebookConfigs:
    experiment_config: InstanceSegmentationExperimentConfig
    graha_config: Any
    dependencies: dict[str, Any]


def build_graha_notebook_configs(
    *,
    output_dir: str | Path | None = None,
    configure_environment: bool = True,
    configure_python_paths: bool = True,
    validate_paths: bool = True,
    import_dependencies: bool = True,
    **config_kwargs: Any,
) -> GrahaInstanceNotebookConfigs:
    """Build instance experiment and Graha configs for notebook workflows."""
    experiment_config = build_experiment_config(**config_kwargs)
    graha_output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else experiment_config.base_output_dir
    )
    adapter = GrahaInstanceModelAdapter()
    graha_config = adapter.build_comparison_config(
        experiment_config,
        graha_output_dir,
    )

    if configure_environment:
        adapter.configure_environment()
    if configure_python_paths:
        adapter.configure_python_paths(graha_config)
    if validate_paths:
        adapter.validate_required_paths(graha_config)
    dependencies = adapter.import_project_dependencies() if import_dependencies else {}

    return GrahaInstanceNotebookConfigs(
        experiment_config=experiment_config,
        graha_config=graha_config,
        dependencies=dependencies,
    )
