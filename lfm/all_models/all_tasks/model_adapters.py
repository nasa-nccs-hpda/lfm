"""Shared model-adapter contracts for Toy and Graha workflows."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload

if TYPE_CHECKING:
    from lfm.full_model.inst_seg.instance_model_adapter import GrahaInstanceModelAdapter
    from lfm.full_model.sem_seg.semantic_model_adapter import GrahaSemanticModelAdapter
    from lfm.toy_model.inst_seg.instance_model_adapter import ToyInstanceModelAdapter
    from lfm.toy_model.sem_seg.semantic_model_adapter import ToySemanticModelAdapter


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


@overload
def resolve_model_adapter(
    model: str, task: Literal["instance"]
) -> ToyInstanceModelAdapter | GrahaInstanceModelAdapter: ...


@overload
def resolve_model_adapter(
    model: str, task: Literal["semantic"]
) -> ToySemanticModelAdapter | GrahaSemanticModelAdapter: ...


@overload
def resolve_model_adapter(
    model: str, task: str
) -> (
    ToyInstanceModelAdapter
    | GrahaInstanceModelAdapter
    | ToySemanticModelAdapter
    | GrahaSemanticModelAdapter
): ...


def resolve_model_adapter(
    model: str, task: str
) -> (
    ToyInstanceModelAdapter
    | GrahaInstanceModelAdapter
    | ToySemanticModelAdapter
    | GrahaSemanticModelAdapter
):
    """Resolve the four existing concrete adapters only when model work starts."""
    if (model, task) == ("toy", "instance"):
        from lfm.toy_model.inst_seg.instance_model_adapter import (
            ToyInstanceModelAdapter,
        )

        return ToyInstanceModelAdapter()
    if (model, task) == ("graha", "instance"):
        from lfm.full_model.inst_seg.instance_model_adapter import (
            GrahaInstanceModelAdapter,
        )

        return GrahaInstanceModelAdapter()
    if (model, task) == ("toy", "semantic"):
        from lfm.toy_model.sem_seg.semantic_model_adapter import ToySemanticModelAdapter

        return ToySemanticModelAdapter()
    if (model, task) == ("graha", "semantic"):
        from lfm.full_model.sem_seg.semantic_model_adapter import (
            GrahaSemanticModelAdapter,
        )

        return GrahaSemanticModelAdapter()
    raise ValueError(f"Unsupported model/task: {model}/{task}")
