"""Toy semantic segmentation model adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lfm.all_models.all_tasks import SemanticModelAdapter
from lfm.toy_model.sem_seg import semantic_toy_components as components


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
