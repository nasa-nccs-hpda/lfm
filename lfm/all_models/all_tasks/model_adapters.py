"""Shared model-adapter contracts for Toy and Graha workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class ModelAdapter(Protocol):
    """Common workflow boundary for model-family-specific behavior."""

    @property
    def display_name(self) -> str:
        """User-facing model family name used in logs and output metadata."""
        ...

    def create_datamodule(self, config: Any, *args: Any, **kwargs: Any) -> Any:
        """Build the task datamodule for this model family."""
        ...

    def create_trainer(
        self, config: Any, output_dir: Path, *args: Any, **kwargs: Any
    ) -> Any:
        """Build the trainer for this model family."""
        ...

    def load_checkpoint_state(
        self,
        model_or_task: Any,
        checkpoint_path: Path,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Load a model-family-specific checkpoint into a model or task."""
        ...


class SemanticModelAdapter(ModelAdapter, Protocol):
    """Adapter contract for semantic segmentation workflows."""

    def create_model_or_task(
        self,
        config: Any,
        datamodule: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Build the semantic model or Lightning task."""
        ...


class InstanceModelAdapter(ModelAdapter, Protocol):
    """Adapter contract for instance segmentation workflows."""

    def create_model_or_task(
        self,
        config: Any,
        datamodule: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Build the instance model or Lightning task."""
        ...
