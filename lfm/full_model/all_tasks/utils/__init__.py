"""Shared utility helpers for full-model experiments.

Task plotting helpers are exposed lazily for backward compatibility. New code
should prefer the focused modules directly.
"""

from __future__ import annotations

from .utils import (
    create_timestamped_output_dir,
    ensure_data_symlink,
    load_terramind_nac_pretraining_stats,
    load_terramind_pretraining_stats,
    load_terramind_wac_pretraining_stats,
)

__all__ = [
    "ValidationPlotCallback",
    "calculate_f1_score",
    "create_overlay_image",
    "create_timestamped_output_dir",
    "ensure_data_symlink",
    "evaluate_prediction_caches",
    "load_terramind_nac_pretraining_stats",
    "load_terramind_pretraining_stats",
    "load_terramind_wac_pretraining_stats",
    "plot_instance_batch_sanity",
    "plot_instance_cache_comparison",
    "plot_instance_cache_predictions",
    "plot_instance_label_comparison",
    "plot_instance_predictions",
    "plot_prediction_cache_comparison",
    "plot_validation_predictions",
    "prepare_image_for_display",
    "save_graha_instance_prediction_cache",
    "save_prediction_cache",
    "save_toy_instance_prediction_cache",
]


def __getattr__(name: str):
    if name == "ValidationPlotCallback":
        from .callbacks import ValidationPlotCallback

        return ValidationPlotCallback
    if name in {"create_overlay_image", "prepare_image_for_display"}:
        from . import display

        return getattr(display, name)
    if name in {"calculate_f1_score", "evaluate_prediction_caches"}:
        from . import metrics

        return getattr(metrics, name)
    if name in {
        "save_graha_instance_prediction_cache",
        "save_prediction_cache",
        "save_toy_instance_prediction_cache",
    }:
        from . import prediction_cache

        return getattr(prediction_cache, name)
    if name in {"plot_prediction_cache_comparison", "plot_validation_predictions"}:
        from lfm.full_model.sem_seg import semantic_plotting as plotting

        return getattr(plotting, name)
    if name in {
        "plot_instance_batch_sanity",
        "plot_instance_cache_comparison",
        "plot_instance_cache_predictions",
        "plot_instance_label_comparison",
        "plot_instance_predictions",
    }:
        from lfm.full_model.inst_seg import instance_plotting as plotting

        return getattr(plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
