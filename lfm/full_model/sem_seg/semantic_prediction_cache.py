"""Semantic prediction-cache helpers.

Semantic prediction caches currently use the shared all-task implementation
without task-specific adaptation. This module exists to keep semantic and
instance package structure mirrored.
"""

from __future__ import annotations

from lfm.full_model.all_tasks.utils.prediction_cache import (
    _load_prediction_cache,
    save_prediction_cache,
)

__all__ = [
    "_load_prediction_cache",
    "save_prediction_cache",
]
