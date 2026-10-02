"""Easy automatic-routing workflow for configured lunar tile sources."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import logging
from pathlib import Path
from types import MappingProxyType
from typing import Literal, TextIO, TypeAlias

from .grid_registry import GridFamily
from .grid_router import GridPointRoute, GridQueryPart, route_aoi, route_point
from .source_modes import compose_tile_sources
from .static_band_contract import (
    MINIRF_SOURCE_NODATA,
    MINIRF_SOURCE_NODATA_BANDS,
    STATIC_BAND_NAMES,
    STATIC_OUTPUT_NODATA,
)
from .tiling import create_tiles_for_index
from .tiling_config import BandNoDataOverride, TileConfig, TileSourceConfig
from .tiling_policy import group_source_rasters_by_product
from .tiling_preparation import (
    TilePreparationResult,
    TileSourcePreparation,
    prepare_tile_config,
)
from .tiling_results import TileCubeRecord, TileSourceError, safe_filename_component
from .vector_index import (
    IndexedRaster,
    query_source_index,
    query_source_index_envelopes,
)
from .vector_index_builder import DEFAULT_RASTER_GLOBS


LOGGER = logging.getLogger(__name__)

SourceRole: TypeAlias = Literal["dynamic", "static"]
ModalityPreset: TypeAlias = Literal["wac", "nac", "static", "custom"]
ProductRequest: TypeAlias = str | Mapping[str, str | None] | None

_SOURCE_ROLES = frozenset({"dynamic", "static"})
_MODALITY_PRESETS = frozenset({"wac", "nac", "static", "custom"})
_POLAR_GRID_IDS = frozenset({"LPS_N", "LPS_S"})
WAC_DEFAULT_ZOOMS = MappingProxyType(
    {
        GridFamily.LTM: 5,
        GridFamily.LPS_N: 4,
        GridFamily.LPS_S: 4,
    }
)
NAC_DEFAULT_ZOOMS = MappingProxyType(
    {
        GridFamily.LTM: 11,
        GridFamily.LPS_N: 10,
        GridFamily.LPS_S: 10,
    }
)
_DEFAULT_ZOOMS = {
    "wac": WAC_DEFAULT_ZOOMS,
    "nac": NAC_DEFAULT_ZOOMS,
}


def _tile_definition_for_grid(grid_id: str, zoom_level: int):
    from .grid_tile_def import tile_definition_for_grid

    return tile_definition_for_grid(grid_id, zoom_level)


def _text(value: object, *, field_name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{field_name} must not be empty.")
    return result


def _boolean(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean.")
    return value


def _family_key(value: object) -> GridFamily:
    if isinstance(value, GridFamily):
        return value
    normalized = _text(value, field_name="zoom override family").casefold()
    aliases = {
        "ltm": GridFamily.LTM,
        "lps_n": GridFamily.LPS_N,
        "lps_s": GridFamily.LPS_S,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        valid = ", ".join(aliases)
        raise ValueError(
            f"Unknown zoom override family {value!r}; expected one of {valid}."
        ) from exc


def _zoom(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def default_zoom_for_modality(
    modality: str,
    family: GridFamily | str,
) -> int | None:
    """Return the built-in near-native family zoom for WAC or 1 m NAC.

    WAC uses LTM zoom 5 (about 75.824 m/pixel) and polar zoom 4 (about
    73.775 m/pixel). Processed one-metre NAC uses LTM zoom 11 (about
    1.185 m/pixel) and polar zoom 10 (about 1.153 m/pixel). Static and custom
    modalities have no independent default because static adopts its dynamic
    acquisition group and custom resolution policy must be explicit.
    """
    normalized_modality = _text(modality, field_name="modality").casefold()
    if normalized_modality not in _MODALITY_PRESETS:
        valid = ", ".join(sorted(_MODALITY_PRESETS))
        raise ValueError(f"modality must be one of {valid}.")
    return _DEFAULT_ZOOMS.get(normalized_modality, {}).get(_family_key(family))


@dataclass(frozen=True)
class TilePointQuery:
    """One geographic point routed automatically to an LTM or polar grid."""

    lat: float
    lon: float


@dataclass(frozen=True)
class TileAOIQuery:
    """One geographic AOI in repository IAU:30100 coordinate order."""

    ul_lat: float
    ul_lon: float
    lr_lat: float
    lr_lon: float


TileQuery: TypeAlias = TilePointQuery | TileAOIQuery


@dataclass(frozen=True)
class TileSourceDefinition:
    """Source policy, preparation settings, role, and zoom behavior."""

    source: TileSourceConfig
    role: SourceRole
    modality: ModalityPreset = "custom"
    image_glob: str | None = None
    image_globs: tuple[str, ...] = DEFAULT_RASTER_GLOBS
    zoom_overrides: Mapping[GridFamily | str, int] = field(default_factory=dict)
    polar_supported: bool = False
    rebuild_invalid_index: bool = False
    index_worker_count: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, TileSourceConfig):
            raise TypeError("source must be a TileSourceConfig.")
        role = _text(self.role, field_name="source role").casefold()
        if role not in _SOURCE_ROLES:
            raise ValueError("source role must be 'dynamic' or 'static'.")
        modality = _text(self.modality, field_name="modality").casefold()
        if modality not in _MODALITY_PRESETS:
            valid = ", ".join(sorted(_MODALITY_PRESETS))
            raise ValueError(f"modality must be one of {valid}.")
        if role == "static" and self.source.selection_mode != "all_intersecting":
            raise ValueError(
                "Static source definitions must use "
                "selection_mode='all_intersecting'."
            )
        if modality in {"wac", "nac"} and role != "dynamic":
            raise ValueError(f"Built-in {modality.upper()} must be dynamic.")
        if modality == "static" and role != "static":
            raise ValueError("The built-in static modality must have role='static'.")
        polar_supported = _boolean(
            self.polar_supported,
            field_name="polar_supported",
        )
        rebuild_invalid_index = _boolean(
            self.rebuild_invalid_index,
            field_name="rebuild_invalid_index",
        )

        normalized_overrides: dict[GridFamily, int] = {}
        for raw_family, raw_zoom in self.zoom_overrides.items():
            family = _family_key(raw_family)
            if family in normalized_overrides:
                raise ValueError(f"Duplicate zoom override for {family.value!r}.")
            normalized_overrides[family] = _zoom(
                raw_zoom,
                field_name=f"zoom override for {family.value}",
            )

        preparation = TileSourcePreparation(
            self.source,
            image_glob=self.image_glob,
            image_globs=self.image_globs,
            rebuild_invalid_index=rebuild_invalid_index,
            worker_count=self.index_worker_count,
        )
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "image_glob", preparation.image_glob)
        object.__setattr__(self, "image_globs", preparation.image_globs)
        object.__setattr__(
            self,
            "zoom_overrides",
            MappingProxyType(normalized_overrides),
        )
        object.__setattr__(self, "polar_supported", polar_supported)
        object.__setattr__(
            self,
            "rebuild_invalid_index",
            rebuild_invalid_index,
        )

    def preparation(self) -> TileSourcePreparation:
        """Return the index-preparation declaration for this source."""
        return TileSourcePreparation(
            self.source,
            image_glob=self.image_glob,
            image_globs=self.image_globs,
            rebuild_invalid_index=self.rebuild_invalid_index,
            worker_count=self.index_worker_count,
        )

    def configured_zoom(self, family: GridFamily) -> int | None:
        """Return an override or built-in family default, if one exists."""
        override = self.zoom_overrides.get(family)
        if override is not None:
            return override
        return default_zoom_for_modality(self.modality, family)


class AutomaticTilingError(RuntimeError):
    """A high-level workflow stage failed with structured tiling context."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        source_name: str | None = None,
        product_id: str | None = None,
        grid_id: str | None = None,
        zoom_level: int | None = None,
        tile_x: int | None = None,
        tile_y: int | None = None,
        completed_records: tuple[TileCubeRecord, ...] = (),
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.source_name = source_name
        self.product_id = product_id
        self.grid_id = grid_id
        self.zoom_level = zoom_level
        self.tile_x = tile_x
        self.tile_y = tile_y
        self.completed_records = completed_records


@dataclass(frozen=True, order=True)
class _TileAddress:
    grid_id: str
    zoom_level: int
    tile_y: int
    tile_x: int


def resolve_tile_index_path(
    data_dir: str | Path,
    *,
    index_path: str | Path | None = None,
    canonical_static: bool = False,
) -> Path:
    """Resolve the contract index path without creating or scanning for one."""
    root = Path(data_dir)
    if index_path is not None:
        return Path(index_path)
    default = root / "output_index.shp"
    if default.is_file():
        return default
    legacy_static = root / "db2.shp"
    if canonical_static and legacy_static.is_file():
        return legacy_static
    return default


def make_wac_tile_source(
    *,
    data_dir: str | Path,
    index_path: str | Path | None = None,
    name: str = "wac",
    index_layer: str | None = None,
    location_field: str = "location",
    required: bool = True,
    image_glob: str | None = None,
    image_globs: tuple[str, ...] = DEFAULT_RASTER_GLOBS,
    zoom_overrides: Mapping[GridFamily | str, int] | None = None,
    rebuild_invalid_index: bool = False,
    index_worker_count: int | None = None,
) -> TileSourceDefinition:
    """Build the WAC product-scoped source preset."""
    root = Path(data_dir)
    source = TileSourceConfig(
        name=name,
        data_dir=root,
        index_path=resolve_tile_index_path(root, index_path=index_path),
        index_layer=index_layer,
        location_field=location_field,
        selection_mode="product_id",
        resampling="bilinear",
        preserve_source_nodata=True,
        required=required,
    )
    return TileSourceDefinition(
        source=source,
        role="dynamic",
        modality="wac",
        image_glob=image_glob,
        image_globs=image_globs,
        zoom_overrides=zoom_overrides or {},
        polar_supported=True,
        rebuild_invalid_index=rebuild_invalid_index,
        index_worker_count=index_worker_count,
    )


def make_nac_tile_source(
    *,
    data_dir: str | Path,
    index_path: str | Path | None = None,
    name: str = "nac",
    index_layer: str | None = None,
    location_field: str = "location",
    required: bool = True,
    image_glob: str | None = None,
    image_globs: tuple[str, ...] = DEFAULT_RASTER_GLOBS,
    zoom_overrides: Mapping[GridFamily | str, int] | None = None,
    rebuild_invalid_index: bool = False,
    index_worker_count: int | None = None,
) -> TileSourceDefinition:
    """Build the processed one-metre NAC product-scoped source preset."""
    root = Path(data_dir)
    source = TileSourceConfig(
        name=name,
        data_dir=root,
        index_path=resolve_tile_index_path(root, index_path=index_path),
        index_layer=index_layer,
        location_field=location_field,
        selection_mode="product_id",
        resampling="bilinear",
        preserve_source_nodata=True,
        required=required,
    )
    return TileSourceDefinition(
        source=source,
        role="dynamic",
        modality="nac",
        image_glob=image_glob,
        image_globs=image_globs,
        zoom_overrides=zoom_overrides or {},
        polar_supported=True,
        rebuild_invalid_index=rebuild_invalid_index,
        index_worker_count=index_worker_count,
    )


def make_static_tile_source(
    *,
    data_dir: str | Path,
    index_path: str | Path | None = None,
    name: str = "static",
    index_layer: str | None = None,
    location_field: str = "location",
    required: bool = True,
    image_glob: str | None = None,
    image_globs: tuple[str, ...] = DEFAULT_RASTER_GLOBS,
    zoom_overrides: Mapping[GridFamily | str, int] | None = None,
    polar_supported: bool = False,
    rebuild_invalid_index: bool = False,
    index_worker_count: int | None = None,
) -> TileSourceDefinition:
    """Build the canonical 63-band contextual static-source preset."""
    root = Path(data_dir)
    source = TileSourceConfig(
        name=name,
        data_dir=root,
        index_path=resolve_tile_index_path(
            root,
            index_path=index_path,
            canonical_static=True,
        ),
        index_layer=index_layer,
        location_field=location_field,
        selection_mode="all_intersecting",
        band_names=STATIC_BAND_NAMES,
        resampling="bilinear",
        output_nodata=STATIC_OUTPUT_NODATA,
        band_nodata_overrides=tuple(
            BandNoDataOverride(
                band_name=band_name,
                source_value=MINIRF_SOURCE_NODATA,
            )
            for band_name in MINIRF_SOURCE_NODATA_BANDS
        ),
        required=required,
    )
    return TileSourceDefinition(
        source=source,
        role="static",
        modality="static",
        image_glob=image_glob,
        image_globs=image_globs,
        zoom_overrides=zoom_overrides or {},
        polar_supported=polar_supported,
        rebuild_invalid_index=rebuild_invalid_index,
        index_worker_count=index_worker_count,
    )


def _enabled_definitions(
    *,
    dynamic_sources: Iterable[TileSourceDefinition] | None,
    static_sources: Iterable[TileSourceDefinition] | None,
    include_dynamic: bool,
    include_static: bool,
) -> tuple[TileSourceDefinition, ...]:
    use_dynamic = _boolean(include_dynamic, field_name="include_dynamic")
    use_static = _boolean(include_static, field_name="include_static")
    dynamic = tuple(dynamic_sources or ()) if use_dynamic else ()
    static = tuple(static_sources or ()) if use_static else ()
    for definition in dynamic:
        if not isinstance(definition, TileSourceDefinition):
            raise TypeError(
                "dynamic_sources must contain TileSourceDefinition objects."
            )
        if definition.role != "dynamic":
            raise ValueError(
                f"Dynamic collection contains role={definition.role!r} source "
                f"{definition.source.name!r}."
            )
    for definition in static:
        if not isinstance(definition, TileSourceDefinition):
            raise TypeError(
                "static_sources must contain TileSourceDefinition objects."
            )
        if definition.role != "static":
            raise ValueError(
                f"Static collection contains role={definition.role!r} source "
                f"{definition.source.name!r}."
            )
    composed = compose_tile_sources(
        dynamic_sources=(item.source for item in dynamic),
        static_sources=(item.source for item in static),
        include_dynamic=use_dynamic,
        include_static=use_static,
    )
    by_name = {item.source.name: item for item in dynamic + static}
    definitions = tuple(by_name[source.name] for source in composed)
    by_component: dict[str, str] = {}
    for definition in definitions:
        name = definition.source.name
        component = safe_filename_component(name)
        previous = by_component.get(component)
        if previous is not None and previous != name:
            raise ValueError(
                f"Source names {previous!r} and {name!r} map to the same "
                f"filename component {component!r}."
            )
        by_component[component] = name
    return definitions


def _route_query(
    query: TileQuery,
) -> tuple[GridPointRoute | None, tuple[GridQueryPart, ...]]:
    if isinstance(query, TilePointQuery):
        point = route_point(lat=query.lat, lon=query.lon)
        return point, ()
    if isinstance(query, TileAOIQuery):
        parts = route_aoi(
            ul_lat=query.ul_lat,
            ul_lon=query.ul_lon,
            lr_lat=query.lr_lat,
            lr_lon=query.lr_lon,
        )
        return None, parts
    raise TypeError("query must be a TilePointQuery or TileAOIQuery.")


def _grid_families(
    point: GridPointRoute | None,
    parts: tuple[GridQueryPart, ...],
) -> dict[str, GridFamily]:
    if point is not None:
        return {point.grid_id: point.family}
    return {part.grid_id: part.family for part in parts}


def _source_groups(
    definitions: tuple[TileSourceDefinition, ...],
    families: Mapping[str, GridFamily],
) -> dict[tuple[str, int], tuple[TileSourceDefinition, ...]]:
    result: dict[tuple[str, int], tuple[TileSourceDefinition, ...]] = {}
    dynamic = tuple(item for item in definitions if item.role == "dynamic")
    static = tuple(item for item in definitions if item.role == "static")
    global_order = {
        item.source.name: position for position, item in enumerate(definitions)
    }
    for grid_id, family in sorted(families.items()):
        if grid_id in _POLAR_GRID_IDS:
            unsupported = [
                item.source.name
                for item in definitions
                if not item.polar_supported
            ]
            if unsupported:
                raise AutomaticTilingError(
                    "Polar tiling was requested for source(s) without verified "
                    f"polar coverage: {unsupported}.",
                    stage="preflight",
                    grid_id=grid_id,
                )

        groups: dict[int, list[TileSourceDefinition]] = {}
        for definition in dynamic:
            zoom_level = definition.configured_zoom(family)
            if zoom_level is None:
                raise AutomaticTilingError(
                    f"Dynamic source {definition.source.name!r} has no default "
                    f"or override for grid family {family.value!r}.",
                    stage="preflight",
                    source_name=definition.source.name,
                    grid_id=grid_id,
                )
            groups.setdefault(zoom_level, []).append(definition)

        for definition in static:
            configured = definition.configured_zoom(family)
            target_zooms = (configured,) if configured is not None else tuple(groups)
            if not target_zooms:
                raise AutomaticTilingError(
                    f"Static-only source {definition.source.name!r} must define "
                    f"a zoom override for grid family {family.value!r}.",
                    stage="preflight",
                    source_name=definition.source.name,
                    grid_id=grid_id,
                )
            for zoom_level in target_zooms:
                groups.setdefault(zoom_level, []).append(definition)

        for zoom_level, members in groups.items():
            try:
                _tile_definition_for_grid(grid_id, zoom_level)
            except Exception as exc:
                raise AutomaticTilingError(
                    f"Grid {grid_id!r} does not support zoom {zoom_level}: {exc}",
                    stage="preflight",
                    grid_id=grid_id,
                    zoom_level=zoom_level,
                ) from exc
            result[(grid_id, zoom_level)] = tuple(
                sorted(
                    {member.source.name: member for member in members}.values(),
                    key=lambda item: global_order[item.source.name],
                )
            )
    return result


def _prepare_sources(
    definitions: tuple[TileSourceDefinition, ...],
    *,
    output_dir: Path,
    representative_zoom: int,
    debug: bool,
    logger: logging.Logger | None,
    stdout: TextIO | None,
) -> TilePreparationResult:
    indexes = []
    for definition in definitions:
        try:
            prepared = prepare_tile_config(
                output_dir=output_dir,
                zoom_level=representative_zoom,
                sources=(definition.preparation(),),
                debug=debug,
                logger=logger,
                stdout=stdout,
            )
        except Exception as exc:
            raise AutomaticTilingError(
                f"Index preparation failed for source "
                f"{definition.source.name!r}: {exc}",
                stage="index_preparation",
                source_name=definition.source.name,
            ) from exc
        indexes.extend(prepared.indexes)
    return TilePreparationResult(
        config=TileConfig(
            output_dir=output_dir,
            zoom_level=representative_zoom,
            sources=tuple(item.source for item in definitions),
            debug=debug,
        ),
        indexes=tuple(indexes),
    )


def _normalize_product_request(
    definitions: tuple[TileSourceDefinition, ...],
    product_ids: ProductRequest,
) -> dict[str, str | None]:
    product_sources = tuple(
        item.source
        for item in definitions
        if item.source.selection_mode == "product_id"
    )
    product_names = {source.name for source in product_sources}
    if isinstance(product_ids, str):
        product_id = _text(product_ids, field_name="product_id")
        if len(product_sources) != 1:
            raise ValueError(
                "A scalar product ID requires exactly one enabled "
                "product-scoped dynamic source."
            )
        return {product_sources[0].name: product_id}
    if product_ids is None:
        return {source.name: None for source in product_sources}
    if not isinstance(product_ids, Mapping):
        raise TypeError("product_ids must be a string, mapping, or None.")
    normalized: dict[str, str | None] = {}
    for raw_name, raw_product_id in product_ids.items():
        name = _text(raw_name, field_name="product-ID source name")
        if name in normalized:
            raise ValueError(f"Duplicate product-ID source name: {name!r}.")
        if name not in product_names:
            raise ValueError(
                f"Product IDs may only target enabled product-scoped sources; "
                f"invalid source {name!r}."
            )
        normalized[name] = (
            None
            if raw_product_id is None
            else _text(raw_product_id, field_name=f"product ID for {name!r}")
        )
    for source in product_sources:
        normalized.setdefault(source.name, None)
    if not product_sources and normalized:
        raise ValueError("This request has no enabled product-scoped sources.")
    return normalized


def _addresses(
    *,
    point: GridPointRoute | None,
    parts: tuple[GridQueryPart, ...],
    groups: Mapping[tuple[str, int], tuple[TileSourceDefinition, ...]],
) -> tuple[_TileAddress, ...]:
    addresses: set[_TileAddress] = set()
    for grid_id, zoom_level in groups:
        tile_definition = _tile_definition_for_grid(grid_id, zoom_level)
        if point is not None:
            tile_index = tile_definition.llToTileIndex(point.lat, point.lon)
            if tile_index is None:
                raise AutomaticTilingError(
                    f"Point ({point.lat}, {point.lon}) is outside grid "
                    f"{grid_id!r} zoom {zoom_level}.",
                    stage="routing",
                    grid_id=grid_id,
                    zoom_level=zoom_level,
                )
            tile_x, tile_y = tile_index
            addresses.add(
                _TileAddress(grid_id, zoom_level, tile_y, tile_x)
            )
            continue
        for part in parts:
            if part.grid_id != grid_id:
                continue
            for tile_x, tile_y in tile_definition.getOverlappingTiles(
                part.ul_lat,
                part.ul_lon,
                part.lr_lat,
                part.lr_lon,
            ):
                addresses.add(
                    _TileAddress(grid_id, zoom_level, tile_y, tile_x)
                )
    return tuple(sorted(addresses))


def _query_records(
    definition: TileSourceDefinition,
    *,
    point: GridPointRoute | None,
    parts: tuple[GridQueryPart, ...],
    groups: Mapping[tuple[str, int], tuple[TileSourceDefinition, ...]],
    addresses: tuple[_TileAddress, ...],
) -> tuple[IndexedRaster, ...]:
    source = definition.source
    records: dict[str, IndexedRaster] = {}
    if point is None:
        for part in parts:
            for record in query_source_index(
                source,
                ul_lat=part.ul_lat,
                ul_lon=part.ul_lon,
                lr_lat=part.lr_lat,
                lr_lon=part.lr_lon,
            ):
                records.setdefault(str(record.path), record)
    else:
        envelopes = []
        for address in addresses:
            if definition not in groups[(address.grid_id, address.zoom_level)]:
                continue
            tile_definition = _tile_definition_for_grid(
                address.grid_id,
                address.zoom_level,
            )
            envelopes.extend(
                tile_definition.geographic_query_envelopes(
                    address.tile_x,
                    address.tile_y,
                )
            )
        for record in query_source_index_envelopes(source, tuple(envelopes)):
            records.setdefault(str(record.path), record)
    return tuple(
        sorted(
            records.values(),
            key=lambda item: (
                str(item.path),
                item.feature_id if item.feature_id is not None else -1,
            ),
        )
    )


def _discover_products(
    definitions: tuple[TileSourceDefinition, ...],
    *,
    requested: Mapping[str, str | None],
    point: GridPointRoute | None,
    parts: tuple[GridQueryPart, ...],
    groups: Mapping[tuple[str, int], tuple[TileSourceDefinition, ...]],
    addresses: tuple[_TileAddress, ...],
    logger: logging.Logger | None,
) -> dict[str, tuple[str, ...]]:
    active_logger = logger or LOGGER
    result: dict[str, tuple[str, ...]] = {}
    routed_grids = ", ".join(sorted({address.grid_id for address in addresses}))
    for definition in definitions:
        source = definition.source
        if source.selection_mode != "product_id":
            continue
        try:
            indexed = _query_records(
                definition,
                point=point,
                parts=parts,
                groups=groups,
                addresses=addresses,
            )
            grouped = group_source_rasters_by_product(source, indexed)
        except Exception as exc:
            raise AutomaticTilingError(
                f"Product discovery failed for source {source.name!r} across "
                f"grid(s) {routed_grids or 'none'}: {exc}",
                stage="product_discovery",
                source_name=source.name,
                grid_id=routed_grids or None,
            ) from exc
        explicit = requested[source.name]
        selected = (
            tuple(grouped)
            if explicit is None
            else ((explicit,) if explicit in grouped else ())
        )
        if explicit is None:
            active_logger.info(
                "Discovered %d product ID(s) for source %r: %s",
                len(selected),
                source.name,
                ", ".join(selected) if selected else "none",
            )
        else:
            active_logger.info(
                "Using explicit product ID %r for source %r.",
                explicit,
                source.name,
            )
        components: dict[str, str] = {}
        for product_id in selected:
            component = safe_filename_component(product_id)
            previous = components.get(component)
            if previous is not None and previous != product_id:
                raise ValueError(
                    f"Source {source.name!r} product IDs {previous!r} and "
                    f"{product_id!r} map to the same filename component."
                )
            components[component] = product_id
        if not selected and source.required:
            detail = (
                "no products intersect the query"
                if explicit is None
                else f"product {explicit!r} does not intersect the query"
            )
            raise AutomaticTilingError(
                f"Required source {source.name!r} has {detail}; routed grids: "
                f"{routed_grids or 'none'}.",
                stage="product_discovery",
                source_name=source.name,
                product_id=explicit,
                grid_id=routed_grids or None,
            )
        result[source.name] = selected
    return result


def _ordered_records(
    definitions: Sequence[TileSourceDefinition],
    records: list[TileCubeRecord],
) -> list[TileCubeRecord]:
    source_order = {
        definition.source.name: position
        for position, definition in enumerate(definitions)
    }
    return sorted(
        records,
        key=lambda record: (
            record.grid_id,
            record.tile_y,
            record.tile_x,
            source_order[record.source_name],
            record.product_id or "",
        ),
    )


def _validate_unique_records(records: Sequence[TileCubeRecord]) -> None:
    identities: set[tuple[object, ...]] = set()
    paths: set[Path] = set()
    for record in records:
        identity = (
            record.grid_id,
            record.zoom_level,
            record.tile_x,
            record.tile_y,
            record.source_name,
            record.product_id,
        )
        if identity in identities:
            raise RuntimeError(
                f"Automatic tiling produced duplicate record {identity}."
            )
        if record.path in paths:
            raise RuntimeError(
                f"Automatic tiling produced a duplicate output path: {record.path}."
            )
        identities.add(identity)
        paths.add(record.path)


def create_tiles_for_query(
    *,
    query: TileQuery,
    output_dir: str | Path,
    dynamic_sources: Iterable[TileSourceDefinition] | None = None,
    static_sources: Iterable[TileSourceDefinition] | None = None,
    product_ids: ProductRequest = None,
    include_dynamic: bool = True,
    include_static: bool = True,
    debug: bool = False,
    logger: logging.Logger | None = None,
    stdout: TextIO | None = None,
) -> list[TileCubeRecord]:
    """Prepare indexes and create automatically routed point or AOI tiles."""
    definitions = _enabled_definitions(
        dynamic_sources=dynamic_sources,
        static_sources=static_sources,
        include_dynamic=include_dynamic,
        include_static=include_static,
    )
    point, parts = _route_query(query)
    families = _grid_families(point, parts)
    groups = _source_groups(definitions, families)
    requested_products = _normalize_product_request(definitions, product_ids)
    addresses = _addresses(point=point, parts=parts, groups=groups)
    if not addresses:
        raise AutomaticTilingError(
            "The geographic query did not select any lunar grid tiles.",
            stage="routing",
        )

    output_path = Path(output_dir)
    representative_zoom = min(zoom for _, zoom in groups)
    _prepare_sources(
        definitions,
        output_dir=output_path,
        representative_zoom=representative_zoom,
        debug=debug,
        logger=logger,
        stdout=stdout,
    )
    selections = _discover_products(
        definitions,
        requested=requested_products,
        point=point,
        parts=parts,
        groups=groups,
        addresses=addresses,
        logger=logger,
    )

    records: list[TileCubeRecord] = []
    produced: set[tuple[str, str]] = set()
    for address in addresses:
        members = groups[(address.grid_id, address.zoom_level)]
        for definition in members:
            source = definition.source
            if source.selection_mode == "product_id":
                runs = tuple(
                    (replace(source, required=False), product_id)
                    for product_id in selections[source.name]
                )
            else:
                runs = ((source, None),)
            for run_source, product_id in runs:
                config = TileConfig(
                    output_dir=output_path,
                    zoom_level=address.zoom_level,
                    sources=(run_source,),
                    debug=debug,
                )
                selectors = (
                    {source.name: product_id}
                    if product_id is not None
                    else None
                )
                try:
                    created = create_tiles_for_index(
                        config,
                        grid_id=address.grid_id,
                        tile_x=address.tile_x,
                        tile_y=address.tile_y,
                        selectors=selectors,
                    )
                except TileSourceError as exc:
                    raise AutomaticTilingError(
                        "Automatic tile generation failed for "
                        f"source {source.name!r}, product {product_id!r}, grid "
                        f"{address.grid_id!r}, zoom {address.zoom_level}, tile "
                        f"({address.tile_x}, {address.tile_y}): {exc}",
                        stage="tile_generation",
                        source_name=source.name,
                        product_id=product_id,
                        grid_id=address.grid_id,
                        zoom_level=address.zoom_level,
                        tile_x=address.tile_x,
                        tile_y=address.tile_y,
                        completed_records=(
                            tuple(records) + tuple(exc.completed_records)
                        ),
                    ) from exc
                for record in created:
                    if record.source_name != source.name:
                        raise RuntimeError(
                            "Automatic tiling received a record for the wrong "
                            f"source: {record}."
                        )
                    if record.product_id != product_id:
                        raise RuntimeError(
                            "Automatic tiling received inconsistent product "
                            f"metadata: {record}."
                        )
                    if product_id is not None:
                        produced.add((source.name, product_id))
                records.extend(created)

    for definition in definitions:
        source = definition.source
        if source.selection_mode != "product_id" or not source.required:
            continue
        for product_id in selections[source.name]:
            if (source.name, product_id) not in produced:
                source_addresses = tuple(
                    address
                    for address in addresses
                    if definition
                    in groups[(address.grid_id, address.zoom_level)]
                )
                attempted = ", ".join(
                    f"{address.grid_id} zoom {address.zoom_level} tile "
                    f"({address.tile_x}, {address.tile_y})"
                    for address in source_addresses
                )
                one_address = (
                    source_addresses[0] if len(source_addresses) == 1 else None
                )
                raise AutomaticTilingError(
                    f"Required source {source.name!r} product {product_id!r} "
                    "produced no tile cubes for the geographic query; "
                    f"attempted {attempted or 'no tile addresses'}.",
                    stage="tile_generation",
                    source_name=source.name,
                    product_id=product_id,
                    grid_id=(
                        one_address.grid_id
                        if one_address is not None
                        else ", ".join(
                            sorted(
                                {
                                    address.grid_id
                                    for address in source_addresses
                                }
                            )
                        )
                        or None
                    ),
                    zoom_level=(
                        one_address.zoom_level
                        if one_address is not None
                        else None
                    ),
                    tile_x=(
                        one_address.tile_x if one_address is not None else None
                    ),
                    tile_y=(
                        one_address.tile_y if one_address is not None else None
                    ),
                    completed_records=tuple(records),
                )

    ordered = _ordered_records(definitions, records)
    _validate_unique_records(ordered)
    return ordered


__all__ = [
    "AutomaticTilingError",
    "ModalityPreset",
    "ProductRequest",
    "SourceRole",
    "TileAOIQuery",
    "TilePointQuery",
    "TileQuery",
    "TileSourceDefinition",
    "create_tiles_for_query",
    "default_zoom_for_modality",
    "make_nac_tile_source",
    "make_static_tile_source",
    "make_wac_tile_source",
    "resolve_tile_index_path",
    "NAC_DEFAULT_ZOOMS",
    "WAC_DEFAULT_ZOOMS",
]
