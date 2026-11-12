"""Toy semantic segmentation model adapter."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from lfm.all_models.all_tasks import SemanticModelAdapter
from lfm.toy_model.sem_seg import semantic_toy_components as components

if TYPE_CHECKING:
    from lfm.all_models.recorded_training import (
        PreparedModel,
        SelectionPolicy,
        TrainingRun,
    )
    from lfm.all_models.sem_seg.config import SemanticSegmentationExperimentConfig


class ToySemanticModelAdapter(SemanticModelAdapter):
    """Adapter for Toy/DINO semantic segmentation workflow construction."""

    @property
    def display_name(self) -> str:
        return "Toy"

    def create_datamodule(self, config: Any, *args: Any, **kwargs: Any) -> Any:
        return components.create_datamodule(config, *args, **kwargs)

    def create_model(self, config: Any, weight_assignments: list[str]) -> Any:
        return components.create_model(config, weight_assignments)

    def create_lightning_module(self, config: Any, model: Any, **kwargs: Any) -> Any:
        return components.create_lightning_module(config, model, **kwargs)

    def create_model_or_task(
        self,
        config: Any,
        datamodule: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if datamodule.weight_assignments is None:
            raise RuntimeError("Toy datamodule did not create weight assignments.")
        model = self.create_model(config, datamodule.weight_assignments)
        return self.create_lightning_module(config, model, **kwargs)

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

        return SelectionPolicy("val_loss", "min")

    def prepare(
        self,
        run: TrainingRun[SemanticSegmentationExperimentConfig],
        initial_epoch: int,
        owner: object,
    ) -> PreparedModel[SemanticSegmentationExperimentConfig]:
        from lfm.all_models.recorded_training import PreparedModel

        config = run.configuration
        info = (
            config.graha_pretrain_dir / "modality_info.yaml"
            if config.graha_pretrain_dir
            else None
        )
        dm = self.create_datamodule(
            config, run.output_dir, normalization_modality_info=info
        )
        return PreparedModel(run, self.create_model_or_task(config, dm), dm, owner)
