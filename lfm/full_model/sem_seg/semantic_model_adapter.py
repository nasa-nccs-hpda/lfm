"""Graha semantic segmentation model adapter."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from lfm.all_models.all_tasks import SemanticModelAdapter
from lfm.full_model.sem_seg import semantic_graha_components as components

if TYPE_CHECKING:
    from lfm.all_models.recorded_training import (
        PreparedModel,
        SelectionPolicy,
        TrainingRun,
    )
    from lfm.all_models.sem_seg.config import SemanticSegmentationExperimentConfig


class GrahaSemanticModelAdapter(SemanticModelAdapter):
    """Adapter for Graha/Lunar-FM semantic segmentation workflow construction."""

    @property
    def display_name(self) -> str:
        return "Graha"

    def build_comparison_config(self, config: Any, output_dir: Path) -> Any:
        return components.build_comparison_config(config, output_dir)

    def build_config(self, args: Any) -> Any:
        return components.build_config(args)

    def configure_environment(self) -> None:
        components.configure_proj_environment()

    def configure_python_paths(self, config: Any) -> None:
        components.configure_python_paths(config)

    def validate_required_paths(self, config: Any) -> None:
        components.validate_required_paths(config)

    def import_project_dependencies(self) -> dict[str, Any]:
        return components.import_project_dependencies()

    def make_task_class(self, lunar_shape_segmentation_task_cls: Any) -> Any:
        return components.make_downstream_shape_segmentation_task_class(
            lunar_shape_segmentation_task_cls
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
        sample_batch = components.inspect_batch(datamodule)
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

    @property
    def recorded_selection(self) -> SelectionPolicy:
        from lfm.all_models.recorded_training import SelectionPolicy

        return SelectionPolicy("val/loss", "min")

    def prepare(
        self,
        run: TrainingRun[SemanticSegmentationExperimentConfig],
        initial_epoch: int,
        owner: object,
    ) -> PreparedModel[SemanticSegmentationExperimentConfig]:
        import torch

        from lfm.all_models.recorded_training import PreparedModel

        config = run.configuration
        self.configure_environment()
        effective = self.build_comparison_config(config, run.output_dir)
        self.configure_python_paths(effective)
        self.validate_required_paths(effective)
        deps = self.import_project_dependencies()
        dm_cls = deps[
            "LunarSemanticFromInstanceDatamodule"
            if config.semantic_label_source == "instance"
            else "LunarSemanticMaskSegmentationDatamodule"
        ]
        means, stds = self.get_normalization_stats(effective, dm_cls)
        dm = self.create_datamodule(effective, dm_cls, means, stds)
        dm.setup("fit")
        task_cls = self.make_task_class(deps["LunarShapeSegmentationTask"])
        # Build from configured geometry without consuming the training loader.
        task = self.create_task(
            effective,
            task_cls,
            {"image": torch.empty(0, len(config.band_filter), *config.target_size)},
        )
        return PreparedModel(run, task, dm, owner)
