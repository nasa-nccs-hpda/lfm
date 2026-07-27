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

    def create_datamodule(self, config: Any, *args: Any, **kwargs: Any) -> Any:
        return workflow.create_datamodule(config, *args, **kwargs)

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

    def run_workflow(self, config: Any, **kwargs: Any) -> tuple[Path, Path | None]:
        return workflow.run_graha_workflow(config, **kwargs)
