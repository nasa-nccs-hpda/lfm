"""Shared experiment configuration validation."""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Sequence
from typing import Any

from lfm.all_models.all_tasks import config_defaults as defaults


TOY_SEMANTIC_LOSS_CHOICES = {
    "cross_entropy",
    "focal",
    "dice",
    "combined",
    "focal_dice",
    "full",
}


def validate_experiment_config(config: Any, *, task: str) -> None:
    """Validate shared semantic/instance experiment config values.

    Hard failures raise ``ValueError``. Suspicious but technically valid values
    emit warnings so unusual experiments can still run intentionally.
    """

    errors: list[str] = []

    _validate_choices(config, errors)
    _validate_common_numbers(config, errors)
    _validate_task_numbers(config, task=task, errors=errors)
    _validate_band_filter(config, errors)

    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Invalid {task} experiment configuration:\n{details}")

    _warn_common_scales(config, task=task)


def _validate_choices(config: Any, errors: list[str]) -> None:
    _require_choice(
        config,
        "dataset_modality",
        defaults.DATASET_MODALITY_CHOICES,
        errors,
    )
    _require_choice(
        config,
        "graha_input_modality_mode",
        defaults.GRAHA_INPUT_MODALITY_CHOICES,
        errors,
    )
    _require_choice(
        config,
        "graha_vis_uv_merge_method",
        defaults.GRAHA_VIS_UV_MERGE_CHOICES,
        errors,
    )
    _validate_optional_choices(
        config,
        "graha_backend_modalities",
        defaults.GRAHA_BACKEND_MODALITY_CHOICES,
        errors,
    )
    _require_choice(
        config,
        "normalization_source",
        defaults.NORMALIZATION_SOURCE_CHOICES,
        errors,
    )
    _require_choice(
        config,
        "normalization_modality",
        defaults.NORMALIZATION_MODALITY_CHOICES,
        errors,
    )
    _require_choice(
        config,
        "image_nodata_policy",
        defaults.IMAGE_NODATA_POLICY_CHOICES,
        errors,
    )
    _require_choice(config, "prediction_split", defaults.SPLIT_CHOICES, errors)
    _require_choice(config, "epoch_test_split", defaults.SPLIT_CHOICES, errors)

    if hasattr(config, "semantic_label_source"):
        _require_choice(
            config,
            "semantic_label_source",
            ("semantic", "instance"),
            errors,
        )
    if hasattr(config, "spatial_transform"):
        _require_choice(config, "spatial_transform", ("crop", "resize"), errors)
    if hasattr(config, "toy_loss_type"):
        _require_choice(config, "toy_loss_type", TOY_SEMANTIC_LOSS_CHOICES, errors)
    if hasattr(config, "toy_architecture"):
        _require_choice(
            config,
            "toy_architecture",
            defaults.TOY_INSTANCE_ARCHITECTURE_CHOICES,
            errors,
        )

    for name in ("image_glob", "label_glob"):
        if hasattr(config, name):
            value = getattr(config, name)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{name} must be a non-empty string.")


def _validate_common_numbers(config: Any, errors: list[str]) -> None:
    for name in (
        "max_epochs",
        "plot_n_samples",
        "prediction_n_samples",
        "epoch_test_n_samples",
    ):
        _require_int(config, name, errors, minimum=1)

    for name in (
        "num_workers",
        "toy_num_workers",
        "graha_num_workers",
        "progress_log_every_n_batches",
        "graha_warmup_steps",
    ):
        _require_int(config, name, errors, minimum=0)

    for name in (
        "plot_every_n_epochs",
        "epoch_test_every_n_epochs",
        "batch_size",
        "toy_batch_size",
        "graha_batch_size",
        "graha_stats_batch_size",
    ):
        _require_int(config, name, errors, minimum=1)

    for name in ("max_train_samples", "max_val_samples", "max_test_samples"):
        _require_optional_int(config, name, errors, minimum=1)

    for name in (
        "learning_rate",
        "toy_learning_rate",
        "graha_backbone_lr",
        "graha_head_lr",
    ):
        _require_float(config, name, errors, minimum_exclusive=0.0)

    for name in (
        "weight_decay",
        "toy_weight_decay",
        "graha_weight_decay",
        "toy_shape_loss_weight",
        "graha_shape_loss_weight",
    ):
        _require_float(config, name, errors, minimum=0.0)

    for name in ("toy_shape_loss_pad_frac", "graha_shape_loss_pad_frac"):
        _require_float(config, name, errors, minimum=0.0, maximum=1.0)

    _require_float(config, "graha_layer_decay", errors, minimum_exclusive=0.0, maximum=1.0)
    _require_optional_float(config, "toy_gradient_clip_val", errors, minimum_exclusive=0.0)
    _require_float(config, "prediction_score_threshold", errors, minimum=0.0, maximum=1.0)
    _require_float(config, "graha_score_threshold", errors, minimum=0.0, maximum=1.0)


def _validate_task_numbers(config: Any, *, task: str, errors: list[str]) -> None:
    target_size = getattr(config, "target_size", None)
    if isinstance(target_size, tuple):
        if len(target_size) != 2:
            errors.append("target_size tuple must contain exactly two values.")
        for idx, value in enumerate(target_size):
            _require_int_value(f"target_size[{idx}]", value, errors, minimum=1)
    elif target_size is not None:
        _require_int_value("target_size", target_size, errors, minimum=1)

    if task == "instance":
        _validate_anchor_sizes(getattr(config, "graha_anchor_sizes", None), errors)
        _validate_positive_float_sequence(
            "graha_anchor_aspect_ratios",
            getattr(config, "graha_anchor_aspect_ratios", None),
            errors,
        )
        mask_shift = getattr(config, "mask_shift", None)
        if not isinstance(mask_shift, Sequence) or isinstance(mask_shift, str):
            errors.append("mask_shift must be a two-value integer sequence.")
        elif len(mask_shift) != 2:
            errors.append("mask_shift must contain exactly two values.")
        else:
            for idx, value in enumerate(mask_shift):
                _require_int_value(f"mask_shift[{idx}]", value, errors)


def _validate_band_filter(config: Any, errors: list[str]) -> None:
    if not hasattr(config, "band_filter"):
        return
    band_filter = getattr(config, "band_filter")
    if band_filter is None:
        return
    if isinstance(band_filter, str) or not isinstance(band_filter, Sequence):
        errors.append("band_filter must be a sequence of nonnegative integer indices.")
        return
    if len(band_filter) == 0:
        errors.append("band_filter must not be empty when provided.")
        return
    for idx, value in enumerate(band_filter):
        _require_int_value(f"band_filter[{idx}]", value, errors, minimum=0)


def _warn_common_scales(config: Any, *, task: str) -> None:
    _warn_if_gt(config, "learning_rate", 1.0e-2, task)
    _warn_if_gt(config, "toy_learning_rate", 1.0e-2, task)
    _warn_if_gt(config, "graha_backbone_lr", 1.0e-3, task)
    _warn_if_gt(config, "graha_head_lr", 1.0e-2, task)
    _warn_if_gt(config, "weight_decay", 0.2, task)
    _warn_if_gt(config, "toy_weight_decay", 0.2, task)
    _warn_if_gt(config, "graha_weight_decay", 0.2, task)
    _warn_if_gt(config, "toy_shape_loss_weight", 1.0, task)
    _warn_if_gt(config, "graha_shape_loss_weight", 1.0, task)
    _warn_if_gt(config, "max_epochs", 500, task)
    _warn_if_gt(config, "graha_warmup_steps", 50_000, task)
    _warn_if_gt(config, "batch_size", 64, task)
    _warn_if_gt(config, "toy_batch_size", 64, task)
    _warn_if_gt(config, "graha_batch_size", 64, task)


def _require_choice(
    config: Any,
    name: str,
    choices: Iterable[str],
    errors: list[str],
) -> None:
    if not hasattr(config, name):
        return
    value = getattr(config, name)
    choices_tuple = tuple(choices)
    if value not in choices_tuple:
        errors.append(f"{name} must be one of {choices_tuple}, got {value!r}.")


def _validate_optional_choices(
    config: Any,
    name: str,
    choices: Iterable[str],
    errors: list[str],
) -> None:
    if not hasattr(config, name):
        return
    value = getattr(config, name)
    if value is None:
        return
    if isinstance(value, str) or not isinstance(value, Sequence):
        errors.append(f"{name} must be a sequence of values from {tuple(choices)}.")
        return
    choices_tuple = tuple(choices)
    for idx, item in enumerate(value):
        if item not in choices_tuple:
            errors.append(
                f"{name}[{idx}] must be one of {choices_tuple}, got {item!r}."
            )


def _require_int(
    config: Any,
    name: str,
    errors: list[str],
    *,
    minimum: int | None = None,
) -> None:
    if hasattr(config, name):
        _require_int_value(name, getattr(config, name), errors, minimum=minimum)


def _require_optional_int(
    config: Any,
    name: str,
    errors: list[str],
    *,
    minimum: int | None = None,
) -> None:
    if not hasattr(config, name):
        return
    value = getattr(config, name)
    if value is None:
        return
    _require_int_value(name, value, errors, minimum=minimum)


def _require_int_value(
    name: str,
    value: Any,
    errors: list[str],
    *,
    minimum: int | None = None,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{name} must be an integer, got {value!r}.")
        return
    if minimum is not None and value < minimum:
        errors.append(f"{name} must be >= {minimum}, got {value}.")


def _require_float(
    config: Any,
    name: str,
    errors: list[str],
    *,
    minimum: float | None = None,
    minimum_exclusive: float | None = None,
    maximum: float | None = None,
) -> None:
    if hasattr(config, name):
        _require_float_value(
            name,
            getattr(config, name),
            errors,
            minimum=minimum,
            minimum_exclusive=minimum_exclusive,
            maximum=maximum,
        )


def _require_optional_float(
    config: Any,
    name: str,
    errors: list[str],
    *,
    minimum_exclusive: float | None = None,
) -> None:
    if not hasattr(config, name):
        return
    value = getattr(config, name)
    if value is None:
        return
    _require_float_value(
        name,
        value,
        errors,
        minimum_exclusive=minimum_exclusive,
    )


def _require_float_value(
    name: str,
    value: Any,
    errors: list[str],
    *,
    minimum: float | None = None,
    minimum_exclusive: float | None = None,
    maximum: float | None = None,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{name} must be numeric, got {value!r}.")
        return
    value_float = float(value)
    if not math.isfinite(value_float):
        errors.append(f"{name} must be finite, got {value!r}.")
        return
    if minimum is not None and value_float < minimum:
        errors.append(f"{name} must be >= {minimum}, got {value_float}.")
    if minimum_exclusive is not None and value_float <= minimum_exclusive:
        errors.append(f"{name} must be > {minimum_exclusive}, got {value_float}.")
    if maximum is not None and value_float > maximum:
        errors.append(f"{name} must be <= {maximum}, got {value_float}.")


def _validate_anchor_sizes(value: Any, errors: list[str]) -> None:
    if value is None:
        errors.append("graha_anchor_sizes must be provided.")
        return
    if isinstance(value, str) or not isinstance(value, Sequence) or len(value) == 0:
        errors.append("graha_anchor_sizes must be a non-empty sequence of size groups.")
        return
    for group_idx, group in enumerate(value):
        if isinstance(group, int):
            group_values = [group]
        elif isinstance(group, Sequence) and not isinstance(group, str):
            group_values = list(group)
        else:
            errors.append(
                f"graha_anchor_sizes[{group_idx}] must be an integer or integer sequence."
            )
            continue
        if not group_values:
            errors.append(f"graha_anchor_sizes[{group_idx}] must not be empty.")
        for size_idx, size in enumerate(group_values):
            _require_int_value(
                f"graha_anchor_sizes[{group_idx}][{size_idx}]",
                size,
                errors,
                minimum=1,
            )


def _validate_positive_float_sequence(
    name: str,
    value: Any,
    errors: list[str],
) -> None:
    if value is None:
        errors.append(f"{name} must be provided.")
        return
    if isinstance(value, str) or not isinstance(value, Sequence) or len(value) == 0:
        errors.append(f"{name} must be a non-empty numeric sequence.")
        return
    for idx, item in enumerate(value):
        _require_float_value(f"{name}[{idx}]", item, errors, minimum_exclusive=0.0)


def _warn_if_gt(config: Any, name: str, threshold: float, task: str) -> None:
    if not hasattr(config, name):
        return
    value = getattr(config, name)
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return
    value_float = float(value)
    if math.isfinite(value_float) and value_float > threshold:
        warnings.warn(
            f"{task} config {name}={value_float:g} is unusually high "
            f"(typical upper warning threshold: {threshold:g}).",
            UserWarning,
            stacklevel=3,
        )
