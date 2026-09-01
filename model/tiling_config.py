"""Configuration objects for modality-neutral lunar tiling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


SelectionMode = Literal["product_id", "all_intersecting"]

SELECTION_MODES = ("product_id", "all_intersecting")
RESAMPLING_METHODS = ("nearest", "bilinear", "cubic", "average")
SUPPORTED_INDEX_SUFFIXES = (".gpkg", ".shp")


def _non_empty_text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty.")
    return text


def _optional_float(value: object, *, field_name: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be numeric or None.") from exc


def _string_tuple(value: object, *, field_name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings or None.")
    result = tuple(
        _non_empty_text(item, field_name=f"{field_name} item") for item in value
    )
    if not result:
        raise ValueError(f"{field_name} must not be empty when provided.")
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must not contain duplicates.")
    return result


def _index_tuple(value: object, *, field_name: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence of 1-based indices.")
    result = tuple(int(item) for item in value)
    if not result:
        raise ValueError(f"{field_name} must not be empty when provided.")
    if any(index < 1 for index in result):
        raise ValueError(f"{field_name} values must be 1-based positive integers.")
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must not contain duplicates.")
    return result


def _require_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")
    return value


def _reject_unknown_keys(
    values: Mapping[str, Any],
    *,
    allowed: set[str],
    field_name: str,
) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise TypeError(f"Unknown {field_name} option(s): {', '.join(unknown)}")


@dataclass(frozen=True)
class BandNoDataOverride:
    """Override NoData handling for one named output band."""

    band_name: str
    source_value: float | None = None
    output_value: float | None = None
    preserve_source: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "band_name",
            _non_empty_text(self.band_name, field_name="band_name"),
        )
        object.__setattr__(
            self,
            "source_value",
            _optional_float(self.source_value, field_name="source_value"),
        )
        object.__setattr__(
            self,
            "output_value",
            _optional_float(self.output_value, field_name="output_value"),
        )


@dataclass(frozen=True)
class TileSourceConfig:
    """Describe one raster modality consumed by the LTM tiling pipeline."""

    name: str
    data_dir: Path
    index_path: Path
    index_layer: str | None = None
    location_field: str = "location"
    selection_mode: SelectionMode = "all_intersecting"
    band_names: tuple[str, ...] | None = None
    band_indices: tuple[int, ...] | None = None
    resampling: str = "bilinear"
    source_nodata: float | None = None
    output_nodata: float | None = None
    preserve_source_nodata: bool = False
    band_nodata_overrides: tuple[BandNoDataOverride, ...] = ()
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _non_empty_text(self.name, field_name="source name"),
        )
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "index_path", Path(self.index_path))
        if not str(self.data_dir):
            raise ValueError("data_dir must not be empty.")
        if not str(self.index_path):
            raise ValueError("index_path must not be empty.")
        if self.index_path.suffix.lower() not in SUPPORTED_INDEX_SUFFIXES:
            supported = ", ".join(SUPPORTED_INDEX_SUFFIXES)
            raise ValueError(
                f"index_path must end with one of {supported}, got "
                f"{self.index_path}."
            )
        if self.index_layer is not None:
            object.__setattr__(
                self,
                "index_layer",
                _non_empty_text(self.index_layer, field_name="index_layer"),
            )
        object.__setattr__(
            self,
            "location_field",
            _non_empty_text(self.location_field, field_name="location_field"),
        )
        if self.selection_mode not in SELECTION_MODES:
            valid = ", ".join(SELECTION_MODES)
            raise ValueError(
                f"selection_mode must be one of {valid}, got "
                f"{self.selection_mode!r}."
            )
        object.__setattr__(
            self,
            "band_names",
            _string_tuple(self.band_names, field_name="band_names"),
        )
        object.__setattr__(
            self,
            "band_indices",
            _index_tuple(self.band_indices, field_name="band_indices"),
        )
        if self.band_names is not None and self.band_indices is not None:
            raise ValueError("Configure band_names or band_indices, not both.")
        if self.resampling not in RESAMPLING_METHODS:
            valid = ", ".join(RESAMPLING_METHODS)
            raise ValueError(
                f"resampling must be one of {valid}, got {self.resampling!r}."
            )
        object.__setattr__(
            self,
            "source_nodata",
            _optional_float(self.source_nodata, field_name="source_nodata"),
        )
        object.__setattr__(
            self,
            "output_nodata",
            _optional_float(self.output_nodata, field_name="output_nodata"),
        )
        overrides = tuple(self.band_nodata_overrides)
        if any(not isinstance(item, BandNoDataOverride) for item in overrides):
            raise TypeError(
                "band_nodata_overrides must contain BandNoDataOverride objects."
            )
        override_names = [item.band_name for item in overrides]
        if len(set(override_names)) != len(override_names):
            raise ValueError("band_nodata_overrides must have unique band names.")
        object.__setattr__(self, "band_nodata_overrides", overrides)

    def nodata_override_for(self, band_name: str) -> BandNoDataOverride | None:
        """Return the exact-name NoData override for a band, if configured."""
        return next(
            (
                override
                for override in self.band_nodata_overrides
                if override.band_name == band_name
            ),
            None,
        )


@dataclass(frozen=True)
class TileConfig:
    """Configuration shared by tile-index, point, and geographic-AOI queries."""

    output_dir: Path
    zoom_level: int
    sources: tuple[TileSourceConfig, ...]
    debug: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not str(self.output_dir):
            raise ValueError("output_dir must not be empty.")
        if isinstance(self.zoom_level, bool) or int(self.zoom_level) < 1:
            raise ValueError("zoom_level must be a positive integer.")
        object.__setattr__(self, "zoom_level", int(self.zoom_level))
        sources = tuple(self.sources)
        if not sources:
            raise ValueError("TileConfig requires at least one source.")
        if any(not isinstance(source, TileSourceConfig) for source in sources):
            raise TypeError("sources must contain TileSourceConfig objects.")
        source_names = [source.name for source in sources]
        if len(set(source_names)) != len(source_names):
            raise ValueError("TileConfig source names must be unique.")
        object.__setattr__(self, "sources", sources)

    def source(self, name: str) -> TileSourceConfig:
        """Return a configured source by name."""
        for source in self.sources:
            if source.name == name:
                return source
        raise KeyError(f"Unknown tile source: {name!r}")


def _band_overrides_from_dict(
    value: object,
    *,
    source_name: str,
) -> tuple[BandNoDataOverride, ...]:
    if value is None:
        return ()
    overrides = _require_mapping(
        value,
        field_name=f"sources[{source_name!r}].nodata.band_overrides",
    )
    result: list[BandNoDataOverride] = []
    for band_name, raw_override in overrides.items():
        values = _require_mapping(
            raw_override,
            field_name=f"band override {band_name!r}",
        )
        _reject_unknown_keys(
            values,
            allowed={"source_value", "output_value", "preserve_source"},
            field_name=f"band override {band_name!r}",
        )
        result.append(
            BandNoDataOverride(
                band_name=str(band_name),
                source_value=values.get("source_value"),
                output_value=values.get("output_value"),
                preserve_source=bool(values.get("preserve_source", False)),
            )
        )
    return tuple(result)


def _source_from_dict(name: str, value: object) -> TileSourceConfig:
    values = _require_mapping(value, field_name=f"sources[{name!r}]")
    _reject_unknown_keys(
        values,
        allowed={
            "data_dir",
            "index",
            "selection_mode",
            "bands",
            "resampling",
            "nodata",
            "required",
        },
        field_name=f"sources[{name!r}]",
    )
    if "data_dir" not in values:
        raise KeyError(f"sources[{name!r}] must include 'data_dir'.")
    if "index" not in values:
        raise KeyError(f"sources[{name!r}] must include 'index'.")

    index = _require_mapping(
        values["index"],
        field_name=f"sources[{name!r}].index",
    )
    _reject_unknown_keys(
        index,
        allowed={"path", "layer", "location_field"},
        field_name=f"sources[{name!r}].index",
    )
    if "path" not in index:
        raise KeyError(f"sources[{name!r}].index must include 'path'.")

    bands = values.get("bands")
    band_names = None
    band_indices = None
    if bands not in (None, "all"):
        band_values = _require_mapping(
            bands,
            field_name=f"sources[{name!r}].bands",
        )
        _reject_unknown_keys(
            band_values,
            allowed={"names", "indices"},
            field_name=f"sources[{name!r}].bands",
        )
        band_names = band_values.get("names")
        band_indices = band_values.get("indices")

    nodata = values.get("nodata", {})
    nodata_values = _require_mapping(
        nodata,
        field_name=f"sources[{name!r}].nodata",
    )
    _reject_unknown_keys(
        nodata_values,
        allowed={
            "source_value",
            "output_value",
            "preserve_source",
            "band_overrides",
        },
        field_name=f"sources[{name!r}].nodata",
    )

    return TileSourceConfig(
        name=name,
        data_dir=Path(values["data_dir"]),
        index_path=Path(index["path"]),
        index_layer=index.get("layer"),
        location_field=index.get("location_field", "location"),
        selection_mode=values.get("selection_mode", "all_intersecting"),
        band_names=band_names,
        band_indices=band_indices,
        resampling=values.get("resampling", "bilinear"),
        source_nodata=nodata_values.get("source_value"),
        output_nodata=nodata_values.get("output_value"),
        preserve_source_nodata=bool(
            nodata_values.get("preserve_source", False)
        ),
        band_nodata_overrides=_band_overrides_from_dict(
            nodata_values.get("band_overrides"),
            source_name=name,
        ),
        required=bool(values.get("required", True)),
    )


def tile_config_from_dict(value: Mapping[str, Any]) -> TileConfig:
    """Build a :class:`TileConfig` from a plain notebook-friendly mapping."""
    values = _require_mapping(value, field_name="tile config")
    _reject_unknown_keys(
        values,
        allowed={"output_dir", "zoom_level", "sources", "debug"},
        field_name="tile config",
    )
    for required in ("output_dir", "zoom_level", "sources"):
        if required not in values:
            raise KeyError(f"Tile config must include {required!r}.")
    raw_sources = _require_mapping(values["sources"], field_name="sources")
    sources = tuple(
        _source_from_dict(str(name), source_values)
        for name, source_values in raw_sources.items()
    )
    return TileConfig(
        output_dir=Path(values["output_dir"]),
        zoom_level=values["zoom_level"],
        sources=sources,
        debug=bool(values.get("debug", False)),
    )


__all__ = [
    "BandNoDataOverride",
    "RESAMPLING_METHODS",
    "SELECTION_MODES",
    "SUPPORTED_INDEX_SUFFIXES",
    "SelectionMode",
    "TileConfig",
    "TileSourceConfig",
    "tile_config_from_dict",
]
