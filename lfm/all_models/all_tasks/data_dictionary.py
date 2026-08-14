"""Resolve notebook data dictionaries into experiment config overrides."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from lfm.all_models.all_tasks import config_defaults as defaults


DataDictionary = Mapping[str, Any]


_PASSTHROUGH_KEYS = {
    "dataset_modality",
    "image_glob",
    "label_glob",
    "image_suffix",
    "label_suffix",
    "semantic_label_source",
    "normalization_source",
    "graha_input_modality_mode",
    "graha_vis_uv_merge_method",
    "ignore_nodata_in_loss",
    "nodata_ignore_index",
    "excluded_nodata_values",
}


_DEFAULT_CHIP_LAYOUTS = {
    "wac": {
        "vis": [0, 1, 2, 3, 4],
        "uv": [5, 6],
    },
    "wac_static": {
        "vis": [0, 1, 2, 3, 4],
        "uv": [5, 6],
        "static": list(range(7, 70)),
    },
    "nac": {
        "pho": [0],
    },
    "nac_dtm": {
        "pho": [0],
        "dtm": [1],
    },
}


def _as_int_list(value: Any, *, name: str) -> list[int]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of integer channel indices.")
    return [int(item) for item in value]


def _as_float_list(value: Any, *, name: str) -> list[float]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of numeric values.")
    return [float(item) for item in value]


def _resolve_layout_channels(
    data_dict: DataDictionary,
) -> dict[str, list[int]]:
    raw_layout = data_dict.get("chip_layout", {})
    if raw_layout is None:
        raw_layout = {}
    if not isinstance(raw_layout, Mapping):
        raise TypeError("DATA_DICT['chip_layout'] must be a mapping.")
    if not raw_layout:
        dataset_modality = data_dict.get("dataset_modality")
        selected = data_dict.get("selected_modalities")
        band_filters = data_dict.get("band_filters")
        if dataset_modality is None:
            selected_has_static = (
                not isinstance(selected, str)
                and isinstance(selected, Sequence)
                and "static" in {str(item) for item in selected}
            )
            filters_have_static = (
                isinstance(band_filters, Mapping) and "static" in band_filters
            )
            dataset_modality = (
                "wac_static"
                if selected_has_static or filters_have_static
                else defaults.DEFAULT_DATASET_MODALITY
            )
        dataset_modality = str(dataset_modality)
        raw_layout = _DEFAULT_CHIP_LAYOUTS.get(dataset_modality, {})

    layout: dict[str, list[int]] = {}
    for modality, value in raw_layout.items():
        if isinstance(value, Mapping):
            if "channels" not in value:
                raise KeyError(
                    f"DATA_DICT['chip_layout'][{modality!r}] must contain 'channels'."
                )
            value = value["channels"]
        layout[str(modality)] = _as_int_list(value, name=f"chip_layout[{modality!r}]")
    return layout


def _resolve_dataset_modality(
    data_dict: DataDictionary,
    layout: Mapping[str, list[int]],
) -> str:
    raw_modality = data_dict.get("dataset_modality")
    if raw_modality is not None:
        return str(raw_modality)
    if "static" in layout:
        return "wac_static"
    return defaults.DEFAULT_DATASET_MODALITY


def _resolve_band_filter(
    data_dict: DataDictionary,
    *,
    layout: Mapping[str, list[int]] | None = None,
) -> list[int] | None:
    if "band_filter" in data_dict and data_dict["band_filter"] is not None:
        return _as_int_list(data_dict["band_filter"], name="band_filter")

    if layout is None:
        layout = _resolve_layout_channels(data_dict)
    raw_band_filters = data_dict.get("band_filters")
    raw_selected = data_dict.get("selected_modalities")

    if raw_selected is not None:
        if isinstance(raw_selected, str) or not isinstance(raw_selected, Sequence):
            raise TypeError("DATA_DICT['selected_modalities'] must be a sequence.")
        selected_modalities = [str(item) for item in raw_selected]
    elif isinstance(raw_band_filters, Mapping) and raw_band_filters:
        selected_modalities = [str(item) for item in raw_band_filters]
    else:
        selected_modalities = list(layout)

    if raw_band_filters is not None and not isinstance(raw_band_filters, Mapping):
        raise TypeError("DATA_DICT['band_filters'] must be a mapping.")

    channels: list[int] = []
    for modality in selected_modalities:
        if modality == "nac" and "nac" not in layout and "pho" in layout:
            modality = "pho"
        if modality not in layout:
            raise KeyError(
                f"DATA_DICT selected modality {modality!r} is missing from chip_layout."
            )
        modality_channels = layout[modality]
        if raw_band_filters is None or modality not in raw_band_filters:
            channels.extend(modality_channels)
            continue
        local_indices = _as_int_list(
            raw_band_filters[modality],
            name=f"band_filters[{modality!r}]",
        )
        if not local_indices:
            raise ValueError(
                f"DATA_DICT['band_filters'][{modality!r}] must select at least "
                "one channel. Remove the modality key to use all channels for "
                "that modality, or remove DATA_DICT['band_filters'] to use the "
                "dataset defaults."
            )
        for local_index in local_indices:
            try:
                channels.append(modality_channels[local_index])
            except IndexError as exc:
                raise IndexError(
                    f"band_filters[{modality!r}] contains local index "
                    f"{local_index}, but chip_layout[{modality!r}] has "
                    f"{len(modality_channels)} channel(s)."
                ) from exc

    return channels or None


def resolve_data_dictionary(data_dict: DataDictionary | None) -> dict[str, Any]:
    """Convert a notebook DATA_DICT into build_config keyword overrides.

    The preferred notebook format is intentionally plain Python:

    ``chip_layout`` maps logical modality names to absolute stored chip channels.
    ``band_filters`` optionally selects modality-local channel indices.
    """
    if data_dict is None:
        return {}
    if not isinstance(data_dict, Mapping):
        raise TypeError("data_dict must be a mapping or None.")

    overrides: dict[str, Any] = {}
    for key in _PASSTHROUGH_KEYS:
        if key in data_dict and data_dict[key] is not None:
            overrides[key] = data_dict[key]
    if "excluded_nodata_values" in overrides:
        overrides["excluded_nodata_values"] = _as_float_list(
            overrides["excluded_nodata_values"],
            name="excluded_nodata_values",
        )
    elif "nodata_values" in data_dict and data_dict["nodata_values"] is not None:
        overrides["excluded_nodata_values"] = _as_float_list(
            data_dict["nodata_values"],
            name="nodata_values",
        )

    if "data_dir" not in data_dict or data_dict["data_dir"] is None:
        raise KeyError("DATA_DICT must include 'data_dir'.")
    overrides["data_root"] = data_dict["data_dir"]

    layout = _resolve_layout_channels(data_dict)
    dataset_modality = _resolve_dataset_modality(data_dict, layout)
    overrides.setdefault("dataset_modality", dataset_modality)

    band_filter = _resolve_band_filter(data_dict, layout=layout)
    if band_filter is not None:
        overrides["band_filter"] = band_filter

    if "graha_input_modalities" in data_dict:
        graha_modalities = tuple(str(item) for item in data_dict["graha_input_modalities"])
        if graha_modalities == ("vis", "uv"):
            overrides["graha_input_modality_mode"] = "vis-uv"
        elif graha_modalities == ("vis", "uv", "static"):
            overrides["graha_input_modality_mode"] = "single"
        elif graha_modalities in {("nac",), ("pho",)}:
            overrides["graha_input_modality_mode"] = "single"
        elif graha_modalities in {("nac", "dtm"), ("pho", "dtm")}:
            overrides["graha_input_modality_mode"] = "nac-dtm"
        else:
            raise ValueError(
                "Unsupported DATA_DICT['graha_input_modalities']: "
                f"{graha_modalities!r}."
            )
    overrides.setdefault(
        "normalization_modality",
        defaults.normalization_modality_for_dataset(dataset_modality),
    )
    if "normalization_modality" in data_dict and data_dict["normalization_modality"]:
        overrides["normalization_modality"] = defaults.normalize_normalization_modality(
            str(data_dict["normalization_modality"])
        )
    overrides.setdefault(
        "graha_input_modality_mode",
        defaults.graha_input_modality_mode_for_dataset(dataset_modality),
    )
    return overrides
