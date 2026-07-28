"""Graha instance segmentation model adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lfm.all_models.all_tasks import InstanceModelAdapter
from lfm.full_model.inst_seg import instance_graha_components as components


class GrahaInstanceModelAdapter(InstanceModelAdapter):
    """Adapter for Graha/Lunar-FM instance segmentation workflow construction."""

    @property
    def display_name(self) -> str:
        return "Graha"

    def build_config(self, args: Any) -> Any:
        return components.build_config(args)

    def build_comparison_config(self, config: Any, output_dir: Path) -> Any:
        return components.build_comparison_config(config, output_dir)

    def configure_environment(self) -> None:
        components.configure_proj_environment()

    def configure_python_paths(self, config: Any) -> None:
        components.configure_python_paths(config)

    def validate_required_paths(self, config: Any) -> None:
        components.validate_required_paths(config)

    def import_project_dependencies(self) -> dict[str, Any]:
        return components.import_project_dependencies()

    def make_task_class(self, lunar_object_detection_task_cls: Any) -> Any:
        return components.make_downstream_object_detection_task_class(
            lunar_object_detection_task_cls
        )

    def get_normalization_stats(
        self,
        config: Any,
        datamodule_cls: Any,
    ) -> tuple[list[float], list[float]]:
        return components.get_normalization_stats(config, datamodule_cls)

    def create_datamodule(self, config: Any, *args: Any, **kwargs: Any) -> Any:
        return components.create_datamodule(config, *args, **kwargs)

    def inspect_batch(self, datamodule: Any) -> dict[str, Any]:
        return components.inspect_batch(datamodule)

    def create_task(
        self, config: Any, task_cls: Any, sample_batch: dict[str, Any]
    ) -> Any:
        return components.create_task(config, task_cls, sample_batch)

    def create_model_or_task(
        self,
        config: Any,
        datamodule: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        sample_batch = self.inspect_batch(datamodule)
        return self.create_task(config, *args, sample_batch=sample_batch, **kwargs)

    def create_trainer(
        self,
        config: Any,
        output_dir: Path,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return components.create_trainer(config, output_dir, *args, **kwargs)

    def load_checkpoint_state(
        self,
        model_or_task: Any,
        checkpoint_path: Path,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return components.load_lightning_checkpoint_state(
            model_or_task,
            checkpoint_path,
            *args,
            **kwargs,
        )
