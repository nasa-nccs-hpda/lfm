"""Graha semantic segmentation model adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lfm.all_models.all_tasks import SemanticModelAdapter
from lfm.full_model.sem_seg import semantic_seg_finetuning as workflow


class GrahaSemanticModelAdapter(SemanticModelAdapter):
    """Adapter for Graha/Lunar-FM semantic segmentation workflow construction."""

    @property
    def display_name(self) -> str:
        return "Graha"

    def build_comparison_config(self, config: Any, output_dir: Path) -> Any:
        return workflow.build_comparison_config(config, output_dir)

    def build_config(self, args: Any) -> Any:
        return workflow.build_config(args)

    def configure_environment(self) -> None:
        workflow.configure_proj_environment()

    def configure_python_paths(self, config: Any) -> None:
        workflow.configure_python_paths(config)

    def validate_required_paths(self, config: Any) -> None:
        workflow.validate_required_paths(config)

    def import_project_dependencies(self) -> dict[str, Any]:
        return workflow.import_project_dependencies()

    def make_task_class(self, lunar_shape_segmentation_task_cls: Any) -> Any:
        return workflow.make_downstream_shape_segmentation_task_class(
            lunar_shape_segmentation_task_cls
        )

    def get_normalization_stats(
        self,
        config: Any,
        datamodule_cls: Any,
    ) -> tuple[list[float], list[float]]:
        return workflow.get_normalization_stats(config, datamodule_cls)

    def create_datamodule(self, config: Any, *args: Any, **kwargs: Any) -> Any:
        return workflow.create_datamodule(config, *args, **kwargs)

    def inspect_batch(self, datamodule: Any) -> dict[str, Any]:
        return workflow.inspect_batch(datamodule)

    def create_task(
        self, config: Any, task_cls: Any, sample_batch: dict[str, Any]
    ) -> Any:
        return workflow.create_task(config, task_cls, sample_batch)

    def create_model_or_task(
        self,
        config: Any,
        datamodule: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        sample_batch = workflow.inspect_batch(datamodule)
        return self.create_task(config, *args, sample_batch=sample_batch, **kwargs)

    def create_trainer(
        self,
        config: Any,
        output_dir: Path,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return workflow.create_trainer(config, output_dir, *args, **kwargs)

    def load_checkpoint_state(
        self,
        model_or_task: Any,
        checkpoint_path: Path,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return workflow.load_lightning_checkpoint_state(
            model_or_task,
            checkpoint_path,
            *args,
            **kwargs,
        )
