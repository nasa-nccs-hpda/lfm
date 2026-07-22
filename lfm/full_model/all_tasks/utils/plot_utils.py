"""Compatibility facade for segmentation plotting helpers.

New code should import shared helpers from `display`, `metrics`,
`prediction_cache`, and `callbacks`, and task-specific plotting helpers from
`lfm.full_model.sem_seg.plotting` or `lfm.full_model.inst_seg.plotting`.
"""

from __future__ import annotations

from .callbacks import ValidationPlotCallback
from .display import (
    create_colored_overlay_image,
    create_overlay_image,
    prepare_image_for_display,
)
from lfm.full_model.inst_seg.plotting import (
    plot_instance_batch_sanity,
    plot_instance_cache_comparison,
    plot_instance_cache_predictions,
    plot_instance_label_comparison,
    plot_instance_predictions,
)
from .metrics import (
    _binary_metrics,
    _instance_metrics,
    calculate_f1_score,
    evaluate_prediction_caches,
)
from .prediction_cache import (
    _load_instance_prediction_cache,
    _load_prediction_cache,
    save_graha_instance_prediction_cache,
    save_prediction_cache,
    save_toy_instance_prediction_cache,
)
from lfm.full_model.sem_seg.plotting import (
    plot_prediction_cache_comparison,
    plot_validation_predictions,
)

__all__ = [
    "ValidationPlotCallback",
    "_binary_metrics",
    "_instance_metrics",
    "_load_instance_prediction_cache",
    "_load_prediction_cache",
    "calculate_f1_score",
    "create_colored_overlay_image",
    "create_overlay_image",
    "evaluate_prediction_caches",
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
