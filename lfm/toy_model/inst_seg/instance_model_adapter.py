"""Toy instance segmentation model adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lfm.all_models.all_tasks import InstanceModelAdapter
from lfm.toy_model.inst_seg import instance_seg_finetuning as workflow


class ToyInstanceModelAdapter(InstanceModelAdapter):
    """Adapter for Toy/DINO instance segmentation workflow construction."""

    @property
    def display_name(self) -> str:
        return "Toy"

    def create_datamodule(self, config: Any, *args: Any, **kwargs: Any) -> Any:
        return workflow.create_datamodule(config, *args, **kwargs)

    def create_task(self, config: Any, weight_assignments: list[str]) -> Any:
        return workflow.create_task(config, weight_assignments)

    def create_model_or_task(
        self,
        config: Any,
        datamodule: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return self.create_task(config, datamodule.weight_assignments or [])

    def create_image_processor(self, config: Any) -> Any:
        return workflow.create_image_processor(config)

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

    def run_workflow(self, config: Any, output_dir: Path, **kwargs: Any) -> Path | None:
        return workflow.run_toy_workflow(config, output_dir, **kwargs)
