"""Preparation and validation of raster vector indexes used by tiling."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from pathlib import Path
import sys
from typing import Any, TextIO

from .lunar_crs import LUNAR_GEOGRAPHIC_WKT_PATH, load_lunar_geographic_wkt
from .vector_index import resolve_indexed_raster_path


LOGGER = logging.getLogger(__name__)
SUPPORTED_INDEX_SUFFIXES = (".gpkg", ".shp")


class VectorIndexValidationError(ValueError):
    """A raster vector index does not satisfy its declared contract."""


class StaleVectorIndexError(VectorIndexValidationError):
    """A raster vector index does not match the current raster inventory."""


@dataclass(frozen=True)
class VectorIndexBuildConfig:
    """Describe a Shapefile or GeoPackage raster-index preparation.

    ``index_path`` defaults to ``<data_dir>/output_index.shp``. Supplying a
    path remains useful for canonical static data (currently ``db2.shp``), a
    GeoPackage, or a collection with a project-specific index name.
    """

    data_dir: Path
    index_path: Path | None = None
    image_glob: str = "*.tif"
    layer_name: str | None = None
    location_field: str = "location"
    output_srs_path: Path = LUNAR_GEOGRAPHIC_WKT_PATH

    def __post_init__(self) -> None:
        data_dir = Path(self.data_dir)
        index_path = (
            data_dir / "output_index.shp"
            if self.index_path is None
            else Path(self.index_path)
        )
        object.__setattr__(self, "data_dir", data_dir)
        object.__setattr__(self, "index_path", index_path)
        object.__setattr__(self, "output_srs_path", Path(self.output_srs_path))
        if index_path.suffix.lower() not in SUPPORTED_INDEX_SUFFIXES:
            raise ValueError("index_path must end with .shp or .gpkg.")
        if not self.image_glob.strip():
            raise ValueError("image_glob must not be empty.")
        if self.layer_name is not None and not self.layer_name.strip():
            raise ValueError("layer_name must not be empty when provided.")
        if not self.location_field.strip():
            raise ValueError("location_field must not be empty.")


@dataclass(frozen=True)
class VectorIndexValidationResult:
    """Validated metadata and inventory for one reusable raster index."""

    index_path: Path
    driver_name: str
    layer_name: str
    location_field: str
    feature_count: int
    raster_paths: tuple[Path, ...]


def _output_srs_wkt(config: VectorIndexBuildConfig) -> str:
    if config.output_srs_path == LUNAR_GEOGRAPHIC_WKT_PATH:
        return load_lunar_geographic_wkt()
    if not config.output_srs_path.is_file():
        raise FileNotFoundError(
            f"Output CRS WKT file does not exist: {config.output_srs_path}"
        )
    wkt = config.output_srs_path.read_text(encoding="utf-8").strip()
    if not wkt:
        raise ValueError(f"Output CRS WKT file is empty: {config.output_srs_path}")
    return wkt


def _normalized_path(path: Path) -> Path:
    """Return a comparison-safe absolute path without requiring it to exist."""
    return path.expanduser().resolve(strict=False)


def _announce(
    message: str,
    *,
    logger: logging.Logger,
    stdout: TextIO | None,
) -> None:
    logger.info(message)
    if stdout is not None:
        print(message, file=stdout, flush=True)


def discover_raster_paths(config: VectorIndexBuildConfig) -> tuple[Path, ...]:
    """Return the deterministic raster inventory declared by ``config``."""
    if not config.data_dir.is_dir():
        raise FileNotFoundError(
            f"Raster data directory does not exist: {config.data_dir}"
        )
    raster_paths = tuple(
        sorted(
            (path for path in config.data_dir.glob(config.image_glob) if path.is_file()),
            key=lambda path: str(path),
        )
    )
    if not raster_paths:
        raise FileNotFoundError(
            f"No rasters matched {config.image_glob!r} in {config.data_dir}"
        )
    return raster_paths


def _expected_driver_name(index_path: Path) -> str:
    return "GPKG" if index_path.suffix.lower() == ".gpkg" else "ESRI Shapefile"


def _progress_bar(*, total: int, stdout: TextIO, enabled: bool):
    """Return a stdout tqdm bar, or a no-op compatible fallback."""
    if enabled:
        try:
            from tqdm.auto import tqdm

            return tqdm(
                total=total,
                desc="Building raster index",
                unit="raster",
                file=stdout,
                dynamic_ncols=True,
            )
        except ImportError:
            LOGGER.warning(
                "tqdm is unavailable; raster index creation will continue "
                "without a progress bar."
            )

    class NullProgress:
        def update(self, amount: int = 1) -> None:
            del amount

        def set_postfix_str(self, value: str, *, refresh: bool = True) -> None:
            del value, refresh

        def close(self) -> None:
            return None

    return NullProgress()


def _raster_footprint(path: Path, *, output_srs, gdal: Any, ogr: Any, osr: Any):
    """Return a four-corner raster footprint transformed to ``output_srs``."""
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError(f"Could not open raster while indexing: {path}")
    try:
        if dataset.RasterXSize < 1 or dataset.RasterYSize < 1:
            raise ValueError(f"Raster has invalid dimensions: {path}")
        geotransform = dataset.GetGeoTransform(can_return_null=True)
        if geotransform is None:
            raise ValueError(f"Raster has no affine transform: {path}")
        source_srs = dataset.GetSpatialRef()
        if source_srs is None:
            raise ValueError(f"Raster has no embedded CRS: {path}")
        source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        width = float(dataset.RasterXSize)
        height = float(dataset.RasterYSize)

        def coordinate(pixel: float, line: float) -> tuple[float, float]:
            return (
                geotransform[0]
                + pixel * geotransform[1]
                + line * geotransform[2],
                geotransform[3]
                + pixel * geotransform[4]
                + line * geotransform[5],
            )

        ring = ogr.Geometry(ogr.wkbLinearRing)
        for pixel, line in (
            (0.0, 0.0),
            (width, 0.0),
            (width, height),
            (0.0, height),
            (0.0, 0.0),
        ):
            x, y = coordinate(pixel, line)
            ring.AddPoint_2D(x, y)
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometry(ring)

        with osr.ExceptionMgr(useExceptions=False):
            transformation = osr.CoordinateTransformation(source_srs, output_srs)
        if transformation is None:
            raise RuntimeError(
                f"Could not transform raster footprint to the index CRS: {path}"
            )
        if polygon.Transform(transformation) != 0:
            raise RuntimeError(
                f"Raster footprint transformation failed while indexing: {path}"
            )
        if polygon.IsEmpty() or not polygon.IsValid():
            raise ValueError(
                f"Raster produced an empty or invalid index footprint: {path}"
            )
        return polygon
    finally:
        dataset = None


def _create_vector_index_with_ogr(
    config: VectorIndexBuildConfig,
    raster_paths: tuple[Path, ...],
    *,
    progress: bool,
    stdout: TextIO,
) -> Path:
    """Write one validated footprint feature per raster using Python OGR."""
    from osgeo import gdal, ogr, osr

    gdal.UseExceptions()
    ogr.UseExceptions()
    osr.UseExceptions()

    output_srs = osr.SpatialReference()
    if output_srs.ImportFromWkt(_output_srs_wkt(config)) != 0:
        raise ValueError(f"Could not import output CRS WKT: {config.output_srs_path}")
    output_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    driver_name = _expected_driver_name(config.index_path)
    driver = ogr.GetDriverByName(driver_name)
    if driver is None:
        raise RuntimeError(f"OGR driver is unavailable: {driver_name}")
    dataset = driver.CreateDataSource(str(config.index_path))
    if dataset is None:
        raise RuntimeError(f"Could not create raster vector index: {config.index_path}")

    layer = None
    bar = _progress_bar(total=len(raster_paths), stdout=stdout, enabled=progress)
    try:
        layer = dataset.CreateLayer(
            config.layer_name or config.index_path.stem,
            srs=output_srs,
            geom_type=ogr.wkbPolygon,
        )
        if layer is None:
            raise RuntimeError(
                f"Could not create index layer in {config.index_path}"
            )
        location_field = ogr.FieldDefn(config.location_field, ogr.OFTString)
        if config.index_path.suffix.lower() == ".shp":
            location_field.SetWidth(254)
        if layer.CreateField(location_field) != 0:
            raise RuntimeError(
                f"Could not create location field {config.location_field!r} "
                f"in {config.index_path}"
            )

        for path in raster_paths:
            stored_path = str(path)
            if (
                config.index_path.suffix.lower() == ".shp"
                and len(stored_path.encode("utf-8")) > 254
            ):
                raise ValueError(
                    f"Raster path exceeds the Shapefile location-field limit: {path}. "
                    "Use a GeoPackage index or a shorter data path."
                )
            footprint = _raster_footprint(
                path,
                output_srs=output_srs,
                gdal=gdal,
                ogr=ogr,
                osr=osr,
            )
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetField(config.location_field, stored_path)
            feature.SetGeometry(footprint)
            if layer.CreateFeature(feature) != 0:
                raise RuntimeError(f"Could not index raster: {path}")
            feature = None
            footprint = None
            bar.set_postfix_str(path.name, refresh=False)
            bar.update(1)
    finally:
        bar.close()
        layer = None
        dataset = None
    return config.index_path


def _rebuild_guidance(config: VectorIndexBuildConfig) -> str:
    return (
        "Rebuild it explicitly after reviewing the difference; tiling will not "
        f"overwrite {config.index_path} automatically."
    )


def _spatial_references_equivalent(
    actual,
    expected,
    *,
    index_suffix: str,
) -> bool:
    """Compare CRS semantics while allowing Shapefile WKT1 metadata loss."""
    if actual.IsSame(expected):
        return True
    if index_suffix.casefold() != ".shp":
        return False
    if not actual.IsGeographic() or not expected.IsGeographic():
        return False

    def prime_meridian(spatial_reference) -> float:
        # GDAL 3.8's Python SpatialReference wrapper does not expose the newer
        # GetPrimeMeridian convenience method. PRIMEM child 1 is the numeric
        # longitude in both WKT1 and WKT2.
        value = spatial_reference.GetAttrValue("PRIMEM", 1)
        return math.nan if value is None else float(value)

    # An ESRI Shapefile .prj serializes the repository's modern IAU WKT as
    # WKT1. That representation drops authority, usage, and datum metadata, so
    # OSR IsSame() returns false even when the coordinate space is unchanged.
    # Validate the numeric geographic coordinate system that the format can
    # actually persist instead of weakening validation for richer formats.
    comparisons = (
        (actual.GetSemiMajor(), expected.GetSemiMajor(), 1e-6),
        (actual.GetSemiMinor(), expected.GetSemiMinor(), 1e-6),
        (actual.GetInvFlattening(), expected.GetInvFlattening(), 1e-12),
        (prime_meridian(actual), prime_meridian(expected), 1e-12),
        (actual.GetAngularUnits(), expected.GetAngularUnits(), 1e-18),
    )
    return all(
        math.isclose(
            float(actual_value),
            float(expected_value),
            rel_tol=0.0,
            abs_tol=tolerance,
        )
        for actual_value, expected_value, tolerance in comparisons
    )


def validate_vector_index(
    config: VectorIndexBuildConfig,
    *,
    expected_raster_paths: tuple[Path, ...] | None = None,
) -> VectorIndexValidationResult:
    """Validate schema, CRS, geometry, paths, and optional inventory freshness."""
    from osgeo import gdal, osr

    gdal.UseExceptions()
    if not config.index_path.is_file():
        raise FileNotFoundError(
            f"Raster vector index does not exist: {config.index_path}"
        )

    dataset = gdal.OpenEx(
        str(config.index_path),
        gdal.OF_VECTOR | gdal.OF_READONLY,
    )
    if dataset is None:
        raise VectorIndexValidationError(
            f"Could not open raster vector index: {config.index_path}"
        )

    layer = None
    try:
        driver = dataset.GetDriver()
        driver_name = driver.ShortName if driver is not None else ""
        expected_driver = _expected_driver_name(config.index_path)
        if driver_name != expected_driver:
            raise VectorIndexValidationError(
                f"Raster vector index {config.index_path} uses driver "
                f"{driver_name!r}; expected {expected_driver!r}."
            )

        layer = (
            dataset.GetLayer(0)
            if config.layer_name is None
            else dataset.GetLayerByName(config.layer_name)
        )
        if layer is None:
            available = [
                dataset.GetLayer(index).GetName()
                for index in range(dataset.GetLayerCount())
            ]
            requested = config.layer_name or "layer 0"
            raise VectorIndexValidationError(
                f"Could not find {requested!r} in {config.index_path}; "
                f"available layers: {available}."
            )

        layer_name = layer.GetName()
        definition = layer.GetLayerDefn()
        location_index = definition.GetFieldIndex(config.location_field)
        if location_index < 0:
            available = [
                definition.GetFieldDefn(index).GetName()
                for index in range(definition.GetFieldCount())
            ]
            raise VectorIndexValidationError(
                f"Raster vector index {config.index_path} layer {layer_name!r} "
                f"does not contain location field {config.location_field!r}; "
                f"available fields: {available}."
            )

        actual_srs = layer.GetSpatialRef()
        if actual_srs is None:
            raise VectorIndexValidationError(
                f"Raster vector index {config.index_path} has no layer CRS."
            )
        expected_srs = osr.SpatialReference()
        if expected_srs.ImportFromWkt(_output_srs_wkt(config)) != 0:
            raise ValueError(
                f"Could not import output CRS WKT: {config.output_srs_path}"
            )
        actual_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        expected_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if not _spatial_references_equivalent(
            actual_srs,
            expected_srs,
            index_suffix=config.index_path.suffix,
        ):
            raise VectorIndexValidationError(
                f"Raster vector index {config.index_path} does not use the "
                f"configured output CRS {config.output_srs_path}."
            )

        indexed_paths: list[Path] = []
        invalid_geometry_fids: list[int] = []
        missing_paths: list[Path] = []
        layer.ResetReading()
        for feature in layer:
            fid = int(feature.GetFID()) if feature.GetFID() is not None else -1
            geometry = feature.GetGeometryRef()
            if geometry is None or geometry.IsEmpty() or not geometry.IsValid():
                invalid_geometry_fids.append(fid)
            stored_path = feature.GetField(config.location_field)
            try:
                raster_path = resolve_indexed_raster_path(
                    config.data_dir,
                    stored_path,
                )
            except ValueError as exc:
                raise VectorIndexValidationError(
                    f"Raster vector index {config.index_path} feature {fid} "
                    f"has an invalid {config.location_field!r}: {exc}"
                ) from exc
            indexed_paths.append(raster_path)
            if not raster_path.is_file():
                missing_paths.append(raster_path)

        if invalid_geometry_fids:
            raise VectorIndexValidationError(
                f"Raster vector index {config.index_path} contains empty or "
                f"invalid geometry for feature IDs {invalid_geometry_fids[:10]}."
            )
        if missing_paths:
            preview = ", ".join(str(path) for path in missing_paths[:5])
            raise StaleVectorIndexError(
                f"Raster vector index {config.index_path} references "
                f"{len(missing_paths)} missing raster(s), including {preview}. "
                f"{_rebuild_guidance(config)}"
            )

        normalized_indexed = tuple(_normalized_path(path) for path in indexed_paths)
        duplicate_count = len(normalized_indexed) - len(set(normalized_indexed))
        if duplicate_count:
            raise VectorIndexValidationError(
                f"Raster vector index {config.index_path} contains "
                f"{duplicate_count} duplicate raster path record(s)."
            )

        if expected_raster_paths is not None:
            normalized_expected = {
                _normalized_path(path) for path in expected_raster_paths
            }
            normalized_actual = set(normalized_indexed)
            missing_from_index = sorted(
                normalized_expected - normalized_actual,
                key=str,
            )
            unexpected_in_index = sorted(
                normalized_actual - normalized_expected,
                key=str,
            )
            if missing_from_index or unexpected_in_index:
                raise StaleVectorIndexError(
                    f"Raster vector index {config.index_path} is stale: "
                    f"{len(missing_from_index)} current raster(s) are missing "
                    f"from the index and {len(unexpected_in_index)} indexed "
                    f"raster(s) are absent from the current inventory. "
                    f"{_rebuild_guidance(config)}"
                )

        return VectorIndexValidationResult(
            index_path=config.index_path,
            driver_name=driver_name,
            layer_name=layer_name,
            location_field=config.location_field,
            feature_count=len(indexed_paths),
            raster_paths=tuple(sorted(indexed_paths, key=str)),
        )
    finally:
        layer = None
        dataset = None


def create_vector_index(
    config: VectorIndexBuildConfig,
    *,
    raster_paths: tuple[Path, ...] | None = None,
    progress: bool = True,
    stdout: TextIO | None = None,
) -> Path:
    """Create a new raster vector index; never overwrite an existing index.

    The supported Explore GDAL 3.8.4 Python bindings do not expose
    ``gdal.TileIndex``. Writing through OGR keeps creation Python-native and
    supplies genuine per-raster progress without shelling out to
    ``gdaltindex``.
    """
    if config.index_path.exists():
        raise FileExistsError(
            f"Raster vector index already exists: {config.index_path}. "
            "Remove or archive it explicitly before rebuilding."
        )
    paths = discover_raster_paths(config) if raster_paths is None else raster_paths
    if not paths:
        raise FileNotFoundError(
            f"No rasters matched {config.image_glob!r} in {config.data_dir}"
        )
    config.index_path.parent.mkdir(parents=True, exist_ok=True)
    active_stdout = sys.stdout if stdout is None else stdout
    return _create_vector_index_with_ogr(
        config,
        paths,
        progress=progress,
        stdout=active_stdout,
    )


def ensure_vector_index(
    config: VectorIndexBuildConfig,
    *,
    logger: logging.Logger | None = None,
    stdout: TextIO | None = None,
) -> VectorIndexValidationResult:
    """Validate and reuse an index, or create it when it does not exist."""
    active_logger = logger or LOGGER
    active_stdout = sys.stdout if stdout is None else stdout
    raster_paths = discover_raster_paths(config)
    _announce(
        f"Found {len(raster_paths)} raster(s) matching {config.image_glob!r} "
        f"in {config.data_dir}.",
        logger=active_logger,
        stdout=active_stdout,
    )

    if config.index_path.exists():
        _announce(
            f"Validating existing raster index: {config.index_path}",
            logger=active_logger,
            stdout=active_stdout,
        )
        result = validate_vector_index(
            config,
            expected_raster_paths=raster_paths,
        )
        _announce(
            f"Reusing validated raster index with {result.feature_count} "
            f"feature(s): {config.index_path}",
            logger=active_logger,
            stdout=active_stdout,
        )
        return result

    _announce(
        f"Raster index does not exist and will be created: {config.index_path}",
        logger=active_logger,
        stdout=active_stdout,
    )
    _announce(
        "Indexing a large lunar raster directory can take several minutes.",
        logger=active_logger,
        stdout=active_stdout,
    )
    create_vector_index(
        config,
        raster_paths=raster_paths,
        progress=True,
        stdout=active_stdout,
    )
    result = validate_vector_index(
        config,
        expected_raster_paths=raster_paths,
    )
    _announce(
        f"Created and validated raster index with {result.feature_count} "
        f"feature(s): {config.index_path}",
        logger=active_logger,
        stdout=active_stdout,
    )
    return result


__all__ = [
    "StaleVectorIndexError",
    "VectorIndexBuildConfig",
    "VectorIndexValidationError",
    "VectorIndexValidationResult",
    "create_vector_index",
    "discover_raster_paths",
    "ensure_vector_index",
    "validate_vector_index",
]
