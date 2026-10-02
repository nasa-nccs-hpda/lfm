"""Preparation and validation of raster vector indexes used by tiling."""

from __future__ import annotations

from contextlib import contextmanager
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, replace
import logging
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, TextIO

from .lunar_crs import LUNAR_GEOGRAPHIC_WKT_PATH, load_lunar_geographic_wkt
from .vector_index import resolve_indexed_raster_path


LOGGER = logging.getLogger(__name__)
SUPPORTED_INDEX_SUFFIXES = (".gpkg", ".shp")
DEFAULT_RASTER_GLOBS = ("*.tif", "*.tiff", "*.nc", "*.vrt")
FOOTPRINT_EDGE_SAMPLES = 21


class VectorIndexValidationError(ValueError):
    """A raster vector index does not satisfy its declared contract."""


class StaleVectorIndexError(VectorIndexValidationError):
    """A raster vector index does not match the current raster inventory."""


class VectorIndexLockError(RuntimeError):
    """Another process is already creating the requested raster index."""


def resolve_index_worker_count(worker_count: int | None = None) -> int:
    """Resolve explicit workers or default to the Slurm CPU allocation."""
    if worker_count is not None:
        if isinstance(worker_count, bool) or not isinstance(worker_count, int):
            raise TypeError("worker_count must be an integer or None.")
        if worker_count < 1:
            raise ValueError("worker_count must be at least 1.")
        return worker_count

    slurm_value = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_value is None or not slurm_value.strip():
        return 1
    try:
        resolved = int(slurm_value)
    except ValueError as exc:
        raise ValueError(
            "SLURM_CPUS_PER_TASK must be a positive integer when worker_count "
            "is not configured explicitly."
        ) from exc
    if resolved < 1:
        raise ValueError(
            "SLURM_CPUS_PER_TASK must be a positive integer when worker_count "
            "is not configured explicitly."
        )
    return resolved


@dataclass(frozen=True)
class VectorIndexBuildConfig:
    """Describe a Shapefile or GeoPackage raster-index preparation.

    ``index_path`` defaults to ``<data_dir>/output_index.shp``. Supplying a
    path remains useful for canonical static data (currently ``db2.shp``), a
    GeoPackage, or a collection with a project-specific index name.

    Discovery defaults to GeoTIFF, NetCDF, and VRT files. ``image_glob`` is
    retained as a backward-compatible single-pattern override;
    ``image_globs`` configures an ordered set of patterns. A NetCDF file must
    open as a directly readable GDAL raster. For a subdataset-only container,
    create a VRT that selects the intended variable.

    ``rebuild_invalid_index`` is an explicit ownership declaration for a
    disposable application-managed GeoPackage cache. It is false by default
    and cannot be enabled for Shapefiles such as shared legacy indexes.

    ``worker_count`` controls parallel raster-footprint inspection. ``None``
    (the default) uses ``SLURM_CPUS_PER_TASK`` and falls back to one worker
    outside Slurm. Set it to ``1`` to force serial indexing.
    """

    data_dir: Path
    index_path: Path | None = None
    image_glob: str | None = None
    layer_name: str | None = None
    location_field: str = "location"
    output_srs_path: Path = LUNAR_GEOGRAPHIC_WKT_PATH
    image_globs: tuple[str, ...] = DEFAULT_RASTER_GLOBS
    rebuild_invalid_index: bool = False
    worker_count: int | None = None

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
        if not isinstance(self.rebuild_invalid_index, bool):
            raise TypeError("rebuild_invalid_index must be a boolean.")
        if self.rebuild_invalid_index and index_path.suffix.lower() != ".gpkg":
            raise ValueError(
                "rebuild_invalid_index is supported only for application-owned "
                "GeoPackage caches."
            )
        if self.worker_count is not None:
            resolve_index_worker_count(self.worker_count)
        image_glob = (
            None if self.image_glob is None else str(self.image_glob).strip()
        )
        if self.image_glob is not None and not image_glob:
            raise ValueError("image_glob must not be empty when provided.")
        if isinstance(self.image_globs, str):
            raise TypeError("image_globs must be a sequence of glob patterns.")
        image_globs = tuple(str(pattern).strip() for pattern in self.image_globs)
        if not image_globs or any(not pattern for pattern in image_globs):
            raise ValueError(
                "image_globs must contain at least one non-empty pattern."
            )
        image_globs = tuple(dict.fromkeys(image_globs))
        if image_glob is not None and image_globs != DEFAULT_RASTER_GLOBS:
            raise ValueError("Provide image_glob or image_globs, not both.")
        object.__setattr__(self, "image_glob", image_glob)
        object.__setattr__(self, "image_globs", image_globs)
        if self.layer_name is not None and not self.layer_name.strip():
            raise ValueError("layer_name must not be empty when provided.")
        if not self.location_field.strip():
            raise ValueError("location_field must not be empty.")

    @property
    def raster_globs(self) -> tuple[str, ...]:
        """Return the effective raster patterns, including the legacy override."""
        return (self.image_glob,) if self.image_glob is not None else self.image_globs


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
            {
                path
                for pattern in config.raster_globs
                for path in config.data_dir.glob(pattern)
                if path.is_file()
            },
            key=lambda path: str(path),
        )
    )
    if not raster_paths:
        raise FileNotFoundError(
            f"No rasters matched {config.raster_globs!r} in {config.data_dir}"
        )
    return raster_paths


def _expected_driver_name(index_path: Path) -> str:
    return "GPKG" if index_path.suffix.lower() == ".gpkg" else "ESRI Shapefile"


def _progress_bar(*, total: int, stdout: TextIO, enabled: bool):
    """Return a stdout tqdm bar, or a no-op compatible fallback."""
    if enabled:
        try:
            # Force tqdm's text renderer so notebook and batch execution write
            # progress directly to the configured stdout stream without
            # requiring an ipywidgets frontend.
            from tqdm import tqdm

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


def _perimeter_pixels(
    width: float,
    height: float,
    *,
    samples_per_edge: int = FOOTPRINT_EDGE_SAMPLES,
) -> tuple[tuple[float, float], ...]:
    """Return a clockwise, corner-deduplicated raster perimeter in pixel space."""
    if isinstance(samples_per_edge, bool) or not isinstance(samples_per_edge, int):
        raise TypeError("samples_per_edge must be an integer.")
    if samples_per_edge < 2:
        raise ValueError("samples_per_edge must be at least 2.")
    if width <= 0.0 or height <= 0.0:
        raise ValueError("Raster width and height must be positive.")
    fractions = tuple(
        index / (samples_per_edge - 1) for index in range(samples_per_edge)
    )
    return (
        *((fraction * width, 0.0) for fraction in fractions),
        *((width, fraction * height) for fraction in fractions[1:]),
        *(
            ((1.0 - fraction) * width, height)
            for fraction in fractions[1:]
        ),
        *((0.0, (1.0 - fraction) * height) for fraction in fractions[1:-1]),
    )


def _enclosed_geographic_pole(transformed_polygon) -> float | None:
    """Infer a pole enclosed by a geographic ring from longitude winding."""
    ring = transformed_polygon.GetGeometryRef(0)
    if ring is None or ring.GetPointCount() < 4:
        return None
    coordinates = tuple(
        (ring.GetX(index), ring.GetY(index))
        for index in range(ring.GetPointCount())
    )
    if any(
        not math.isfinite(longitude) or not math.isfinite(latitude)
        for longitude, latitude in coordinates
    ):
        return None

    longitude_winding = 0.0
    previous_longitude = coordinates[0][0]
    for longitude, _ in coordinates[1:]:
        delta = longitude - previous_longitude
        delta = (delta + 180.0) % 360.0 - 180.0
        longitude_winding += delta
        previous_longitude = longitude
    if round(longitude_winding / 360.0) == 0:
        return None

    _, _, minimum_latitude, maximum_latitude = transformed_polygon.GetEnvelope()
    if minimum_latitude >= 0.0:
        return 90.0
    if maximum_latitude <= 0.0:
        return -90.0
    return 90.0 if abs(maximum_latitude) >= abs(minimum_latitude) else -90.0


def _full_longitude_polar_cap(transformed_polygon, *, pole_latitude: float, ogr: Any):
    """Return a conservative valid geographic polygon for a pole-containing raster."""
    _, _, minimum_latitude, maximum_latitude = transformed_polygon.GetEnvelope()
    if pole_latitude > 0.0:
        lower_latitude = max(-90.0, min(90.0, minimum_latitude))
        coordinates = (
            (-180.0, lower_latitude),
            (180.0, lower_latitude),
            (180.0, 90.0),
            (-180.0, 90.0),
            (-180.0, lower_latitude),
        )
    else:
        upper_latitude = max(-90.0, min(90.0, maximum_latitude))
        coordinates = (
            (-180.0, -90.0),
            (180.0, -90.0),
            (180.0, upper_latitude),
            (-180.0, upper_latitude),
            (-180.0, -90.0),
        )
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for longitude, latitude in coordinates:
        ring.AddPoint_2D(longitude, latitude)
    cap = ogr.Geometry(ogr.wkbPolygon)
    cap.AddGeometry(ring)
    if cap.IsEmpty() or not cap.IsValid():
        raise ValueError("Could not construct a valid polar-cap index footprint.")
    return cap


def _full_longitude_band(
    *,
    minimum_latitude: float,
    maximum_latitude: float,
    ogr: Any,
):
    """Return a valid geographic footprint spanning every longitude."""
    lower_latitude = max(-90.0, min(90.0, minimum_latitude))
    upper_latitude = max(-90.0, min(90.0, maximum_latitude))
    if (
        not math.isfinite(lower_latitude)
        or not math.isfinite(upper_latitude)
        or upper_latitude <= lower_latitude
    ):
        raise ValueError(
            "Could not construct a full-longitude footprint from the raster's "
            "latitude range."
        )
    band = _polygon_from_coordinates(
        (
            (-180.0, lower_latitude),
            (180.0, lower_latitude),
            (180.0, upper_latitude),
            (-180.0, upper_latitude),
            (-180.0, lower_latitude),
        ),
        ogr=ogr,
    )
    if band.IsEmpty() or not band.IsValid():
        raise ValueError("Could not construct a valid full-longitude footprint.")
    return band


def _unwrap_longitudes(
    coordinates: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    """Return one continuous longitude sequence without +/-180 jumps."""
    if not coordinates:
        raise ValueError("Geographic footprint contains no coordinates.")
    unwrapped = [coordinates[0]]
    for raw_longitude, latitude in coordinates[1:]:
        longitude = raw_longitude
        previous_longitude = unwrapped[-1][0]
        while longitude - previous_longitude > 180.0:
            longitude -= 360.0
        while longitude - previous_longitude < -180.0:
            longitude += 360.0
        unwrapped.append((longitude, latitude))
    return tuple(unwrapped)


def _polygon_from_coordinates(coordinates, *, ogr: Any):
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for longitude, latitude in coordinates:
        ring.AddPoint_2D(longitude, latitude)
    if coordinates[0] != coordinates[-1]:
        ring.AddPoint_2D(*coordinates[0])
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)
    return polygon


def _polygon_parts(geometry) -> tuple[Any, ...]:
    """Return cloned polygon members from an OGR polygonal geometry."""
    geometry_name = geometry.GetGeometryName().upper()
    if geometry_name == "POLYGON":
        return (geometry.Clone(),)
    if geometry_name in {"MULTIPOLYGON", "GEOMETRYCOLLECTION"}:
        return tuple(
            part
            for index in range(geometry.GetGeometryCount())
            for part in _polygon_parts(geometry.GetGeometryRef(index))
        )
    return ()


def _shift_longitude(geometry, offset: float):
    shifted = geometry.Clone()

    def shift(part) -> None:
        for index in range(part.GetPointCount()):
            point = part.GetPoint(index)
            part.SetPoint_2D(index, point[0] + offset, point[1])
        for index in range(part.GetGeometryCount()):
            shift(part.GetGeometryRef(index))

    shift(shifted)
    return shifted


def _canonical_geographic_footprint(transformed_polygon, *, ogr: Any):
    """Canonicalize a geographic footprint, including seams and global bands."""
    ring = transformed_polygon.GetGeometryRef(0)
    if ring is None or ring.GetPointCount() < 4:
        raise ValueError("Transformed raster footprint has no exterior ring.")
    coordinates = tuple(
        (float(ring.GetX(index)), float(ring.GetY(index)))
        for index in range(ring.GetPointCount())
    )
    if any(
        not math.isfinite(longitude) or not math.isfinite(latitude)
        for longitude, latitude in coordinates
    ):
        raise ValueError("Transformed raster footprint has non-finite coordinates.")

    unwrapped = _polygon_from_coordinates(
        _unwrap_longitudes(coordinates),
        ogr=ogr,
    )
    if unwrapped.IsEmpty() or not unwrapped.IsValid():
        raise ValueError(
            "Raster footprint remains invalid after longitude unwrapping."
        )
    minimum_longitude, maximum_longitude, minimum_latitude, maximum_latitude = (
        unwrapped.GetEnvelope()
    )
    if minimum_latitude < -90.0 - 1e-9 or maximum_latitude > 90.0 + 1e-9:
        raise ValueError("Transformed raster footprint exceeds latitude bounds.")
    longitude_span = maximum_longitude - minimum_longitude
    if longitude_span > 360.0 + 1e-9:
        raise ValueError(
            "A non-polar raster footprint spans more than the full longitude "
            "range."
        )
    if longitude_span >= 360.0 - 1e-9:
        return _full_longitude_band(
            minimum_latitude=minimum_latitude,
            maximum_latitude=maximum_latitude,
            ogr=ogr,
        )
    if minimum_longitude >= -180.0 and maximum_longitude <= 180.0:
        return unwrapped

    first_window = math.floor((minimum_longitude + 180.0) / 360.0)
    last_window = math.floor(
        (maximum_longitude + 180.0 - 1e-12) / 360.0
    )
    parts = []
    for window_index in range(first_window, last_window + 1):
        western_edge = -180.0 + 360.0 * window_index
        window = _polygon_from_coordinates(
            (
                (western_edge, -90.0),
                (western_edge + 360.0, -90.0),
                (western_edge + 360.0, 90.0),
                (western_edge, 90.0),
                (western_edge, -90.0),
            ),
            ogr=ogr,
        )
        intersection = unwrapped.Intersection(window)
        for polygon in _polygon_parts(intersection):
            if polygon.IsEmpty() or polygon.GetArea() <= 0.0:
                continue
            shifted = _shift_longitude(polygon, -360.0 * window_index)
            if shifted.IsEmpty() or not shifted.IsValid():
                raise ValueError(
                    "Could not construct a valid antimeridian-safe footprint."
                )
            parts.append(shifted)

    if not parts:
        raise ValueError("Raster footprint has no area after antimeridian splitting.")
    if len(parts) == 1:
        return parts[0]
    footprint = ogr.Geometry(ogr.wkbMultiPolygon)
    for part in parts:
        footprint.AddGeometry(part)
    if footprint.IsEmpty() or not footprint.IsValid():
        raise ValueError("Could not construct a valid multipart raster footprint.")
    return footprint


def _as_multipolygon(geometry, *, ogr: Any):
    """Return a MultiPolygon suitable for the index layer contract."""
    parts = _polygon_parts(geometry)
    if not parts:
        raise ValueError("Raster footprint is not polygonal.")
    multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)
    for part in parts:
        multipolygon.AddGeometry(part)
    return multipolygon


def _raster_footprint(
    path: Path,
    *,
    output_srs,
    gdal: Any,
    ogr: Any,
    osr: Any,
    samples_per_edge: int = FOOTPRINT_EDGE_SAMPLES,
):
    """Return a densified raster perimeter transformed to ``output_srs``."""
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError(f"Could not open raster while indexing: {path}")
    try:
        if dataset.RasterXSize < 1 or dataset.RasterYSize < 1:
            subdatasets = dataset.GetSubDatasets()
            if subdatasets:
                raise ValueError(
                    f"Raster container exposes only subdatasets: {path}. "
                    "Create a VRT selecting the intended subdataset before indexing."
                )
            raise ValueError(f"Raster has invalid dimensions: {path}")
        if dataset.RasterCount < 1:
            raise ValueError(f"Raster has no readable bands: {path}")
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

        perimeter = _perimeter_pixels(
            width,
            height,
            samples_per_edge=samples_per_edge,
        )
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for pixel, line in (*perimeter, perimeter[0]):
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
        pole_latitude = _enclosed_geographic_pole(polygon)
        if pole_latitude is not None:
            LOGGER.info(
                "Raster %s contains the geographic pole at latitude %.0f; "
                "using a conservative full-longitude cap in the source index.",
                path,
                pole_latitude,
            )
            return _full_longitude_polar_cap(
                polygon,
                pole_latitude=pole_latitude,
                ogr=ogr,
            )
        try:
            return _canonical_geographic_footprint(polygon, ogr=ogr)
        except ValueError as exc:
            raise ValueError(
                f"Raster produced an empty or invalid index footprint: {path}. "
                f"{exc}"
            ) from exc
    finally:
        dataset = None


_FOOTPRINT_WORKER_STATE: tuple[Any, Any, Any, Any] | None = None


def _initialize_footprint_worker(output_srs_wkt: str) -> None:
    """Initialize process-local GDAL objects for footprint computation."""
    from osgeo import gdal, ogr, osr

    gdal.UseExceptions()
    ogr.UseExceptions()
    osr.UseExceptions()
    output_srs = osr.SpatialReference()
    if output_srs.ImportFromWkt(output_srs_wkt) != 0:
        raise ValueError("Could not import output CRS WKT in footprint worker.")
    output_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    global _FOOTPRINT_WORKER_STATE
    _FOOTPRINT_WORKER_STATE = (output_srs, gdal, ogr, osr)


def _raster_footprint_wkb(path: Path) -> bytes:
    """Return one process-safe MultiPolygon footprint serialized as WKB."""
    if _FOOTPRINT_WORKER_STATE is None:
        raise RuntimeError("Raster-footprint worker was not initialized.")
    output_srs, gdal, ogr, osr = _FOOTPRINT_WORKER_STATE
    footprint = _raster_footprint(
        path,
        output_srs=output_srs,
        gdal=gdal,
        ogr=ogr,
        osr=osr,
    )
    footprint = _as_multipolygon(footprint, ogr=ogr)
    return bytes(footprint.ExportToWkb())


@contextmanager
def _parallel_footprint_results(
    raster_paths: tuple[Path, ...],
    *,
    output_srs_wkt: str,
    worker_count: int,
):
    """Yield ordered WKB results from a bounded pool of spawned workers."""
    executor = ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_footprint_worker,
        initargs=(output_srs_wkt,),
    )
    pending: deque[tuple[Path, Future[bytes]]] = deque()
    paths = iter(raster_paths)

    def submit_next() -> bool:
        try:
            path = next(paths)
        except StopIteration:
            return False
        pending.append((path, executor.submit(_raster_footprint_wkb, path)))
        return True

    for _ in range(min(len(raster_paths), worker_count * 2)):
        submit_next()

    def ordered_results():
        while pending:
            path, future = pending.popleft()
            footprint_wkb = future.result()
            submit_next()
            yield path, footprint_wkb

    try:
        yield ordered_results()
    finally:
        for _, future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


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

    output_srs_wkt = _output_srs_wkt(config)
    output_srs = osr.SpatialReference()
    if output_srs.ImportFromWkt(output_srs_wkt) != 0:
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
    active_worker_count = min(
        len(raster_paths),
        resolve_index_worker_count(config.worker_count),
    )
    print(
        f"Using {active_worker_count} raster-footprint worker(s).",
        file=stdout,
        flush=True,
    )
    bar = _progress_bar(total=len(raster_paths), stdout=stdout, enabled=progress)
    try:
        layer = dataset.CreateLayer(
            config.layer_name or config.index_path.stem,
            srs=output_srs,
            geom_type=ogr.wkbMultiPolygon,
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

        def write_feature(path: Path, footprint) -> None:
            stored_path = str(path)
            if (
                config.index_path.suffix.lower() == ".shp"
                and len(stored_path.encode("utf-8")) > 254
            ):
                raise ValueError(
                    f"Raster path exceeds the Shapefile location-field limit: {path}. "
                    "Use a GeoPackage index or a shorter data path."
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

        if active_worker_count == 1:
            for path in raster_paths:
                footprint = _raster_footprint(
                    path,
                    output_srs=output_srs,
                    gdal=gdal,
                    ogr=ogr,
                    osr=osr,
                )
                write_feature(
                    path,
                    _as_multipolygon(footprint, ogr=ogr),
                )
        else:
            with _parallel_footprint_results(
                raster_paths,
                output_srs_wkt=output_srs_wkt,
                worker_count=active_worker_count,
            ) as footprint_results:
                for path, footprint_wkb in footprint_results:
                    footprint = ogr.CreateGeometryFromWkb(footprint_wkb)
                    if footprint is None:
                        raise RuntimeError(
                            f"Could not deserialize raster footprint: {path}"
                        )
                    write_feature(path, footprint)
    finally:
        bar.close()
        layer = None
        dataset = None
    return config.index_path


def _rebuild_guidance(config: VectorIndexBuildConfig) -> str:
    if config.rebuild_invalid_index:
        return "This application-owned GeoPackage cache is eligible for rebuilding."
    return (
        "Rebuild it explicitly after reviewing the difference; tiling will not "
        f"overwrite {config.index_path} automatically."
    )


def _index_lock_path(index_path: Path) -> Path:
    return index_path.with_name(f".{index_path.name}.lock")


@contextmanager
def _exclusive_creation_lock(index_path: Path):
    """Hold a process-scoped, sibling lock while an index is being published."""
    lock_path = _index_lock_path(index_path)
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o664,
        )
    except FileExistsError as exc:
        raise VectorIndexLockError(
            f"Raster index creation is already in progress for {index_path}; "
            f"lock file: {lock_path}. If no creation job is active, inspect "
            "and remove the stale lock explicitly."
        ) from exc

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as lock_file:
            lock_file.write(f"pid={os.getpid()}\n")
            lock_file.flush()
        yield lock_path
    finally:
        lock_path.unlink(missing_ok=True)


def _staged_artifacts(staging_directory: Path, index_name: str) -> tuple[Path, ...]:
    """Return every regular file emitted for one staged vector index."""
    prefix = f"{index_name}.".casefold()
    artifacts = tuple(
        sorted(
            (
                path
                for path in staging_directory.iterdir()
                if path.is_file()
                and (
                    path.name.casefold() == index_name.casefold()
                    or path.name.casefold().startswith(prefix)
                    or path.stem.casefold() == Path(index_name).stem.casefold()
                )
            ),
            key=lambda path: path.name,
        )
    )
    if not artifacts:
        raise RuntimeError(
            f"Vector index creation produced no files in {staging_directory}."
        )
    return artifacts


def _publish_staged_index(
    staged_index_path: Path,
    destination_index_path: Path,
) -> None:
    """Publish a validated index, moving its primary file last."""
    artifacts = _staged_artifacts(
        staged_index_path.parent,
        staged_index_path.name,
    )
    if staged_index_path not in artifacts:
        raise RuntimeError(
            f"Staged vector index is missing its primary file: {staged_index_path}"
        )

    destinations = tuple(
        destination_index_path.parent / artifact.name for artifact in artifacts
    )
    collisions = tuple(path for path in destinations if path.exists())
    if collisions:
        preview = ", ".join(str(path) for path in collisions[:5])
        raise FileExistsError(
            "Raster vector index publication would overwrite existing file(s): "
            f"{preview}. Remove or archive them explicitly before rebuilding."
        )

    publication_order = sorted(
        zip(artifacts, destinations),
        key=lambda pair: pair[0] == staged_index_path,
    )
    published: list[Path] = []
    try:
        for source, destination in publication_order:
            os.replace(source, destination)
            published.append(destination)
    except Exception:
        for destination in reversed(published):
            destination.unlink(missing_ok=True)
        raise


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

    try:
        dataset = gdal.OpenEx(
            str(config.index_path),
            gdal.OF_VECTOR | gdal.OF_READONLY,
        )
    except RuntimeError as exc:
        raise VectorIndexValidationError(
            f"Could not open raster vector index: {config.index_path}"
        ) from exc
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
    """Create and validate a staged raster index, then publish it once complete.

    The supported Explore GDAL 3.8.4 Python bindings do not expose
    ``gdal.TileIndex``. Writing through OGR keeps creation Python-native and
    supplies genuine per-raster progress without shelling out to
    ``gdaltindex``. A sibling lock prevents cooperating jobs from publishing
    the same destination concurrently, and existing files are never silently
    overwritten.
    """
    if config.index_path.exists():
        raise FileExistsError(
            f"Raster vector index already exists: {config.index_path}. "
            "Remove or archive it explicitly before rebuilding."
        )
    paths = discover_raster_paths(config) if raster_paths is None else raster_paths
    if not paths:
        raise FileNotFoundError(
            f"No rasters matched {config.raster_globs!r} in {config.data_dir}"
        )
    config.index_path.parent.mkdir(parents=True, exist_ok=True)
    active_stdout = sys.stdout if stdout is None else stdout

    with _exclusive_creation_lock(config.index_path):
        if config.index_path.exists():
            raise FileExistsError(
                f"Raster vector index already exists: {config.index_path}. "
                "Remove or archive it explicitly before rebuilding."
            )

        staging_directory = Path(
            tempfile.mkdtemp(
                prefix=f".{config.index_path.name}.staging-",
                dir=config.index_path.parent,
            )
        )
        staged_config = replace(
            config,
            index_path=staging_directory / config.index_path.name,
        )
        try:
            _create_vector_index_with_ogr(
                staged_config,
                paths,
                progress=progress,
                stdout=active_stdout,
            )
            validate_vector_index(
                staged_config,
                expected_raster_paths=paths,
            )
            _publish_staged_index(
                staged_config.index_path,
                config.index_path,
            )
        finally:
            shutil.rmtree(staging_directory, ignore_errors=True)
    return config.index_path


def _rebuild_invalid_geopackage(
    config: VectorIndexBuildConfig,
    *,
    raster_paths: tuple[Path, ...],
    stdout: TextIO,
) -> tuple[VectorIndexValidationResult, bool]:
    """Atomically replace an invalid application-owned GeoPackage cache."""
    if not config.rebuild_invalid_index or config.index_path.suffix.lower() != ".gpkg":
        raise ValueError(
            "Invalid-index rebuilding requires an application-owned GeoPackage."
        )
    config.index_path.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_creation_lock(config.index_path):
        if config.index_path.exists():
            try:
                result = validate_vector_index(
                    config,
                    expected_raster_paths=raster_paths,
                )
            except VectorIndexValidationError:
                pass
            else:
                return result, False

        staging_directory = Path(
            tempfile.mkdtemp(
                prefix=f".{config.index_path.name}.replacement-",
                dir=config.index_path.parent,
            )
        )
        staged_config = replace(
            config,
            index_path=staging_directory / config.index_path.name,
            rebuild_invalid_index=False,
        )
        try:
            _create_vector_index_with_ogr(
                staged_config,
                raster_paths,
                progress=True,
                stdout=stdout,
            )
            validate_vector_index(
                staged_config,
                expected_raster_paths=raster_paths,
            )
            for suffix in ("-wal", "-shm", "-journal"):
                Path(f"{config.index_path}{suffix}").unlink(missing_ok=True)
            os.replace(staged_config.index_path, config.index_path)
        finally:
            shutil.rmtree(staging_directory, ignore_errors=True)

    return (
        validate_vector_index(
            config,
            expected_raster_paths=raster_paths,
        ),
        True,
    )


def ensure_vector_index(
    config: VectorIndexBuildConfig,
    *,
    logger: logging.Logger | None = None,
    stdout: TextIO | None = None,
) -> VectorIndexValidationResult:
    """Reuse or create an index, rebuilding only an opted-in managed cache."""
    active_logger = logger or LOGGER
    active_stdout = sys.stdout if stdout is None else stdout
    raster_paths = discover_raster_paths(config)
    _announce(
        f"Found {len(raster_paths)} raster(s) matching {config.raster_globs!r} "
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
        try:
            result = validate_vector_index(
                config,
                expected_raster_paths=raster_paths,
            )
        except VectorIndexValidationError as exc:
            if not config.rebuild_invalid_index:
                raise
            _announce(
                f"Managed raster index is invalid or stale and will be "
                f"rebuilt: {config.index_path}. Reason: {exc}",
                logger=active_logger,
                stdout=active_stdout,
            )
            result, rebuilt = _rebuild_invalid_geopackage(
                config,
                raster_paths=raster_paths,
                stdout=active_stdout,
            )
            action = "Rebuilt" if rebuilt else "Reused concurrently rebuilt"
            _announce(
                f"{action} raster index with {result.feature_count} "
                f"feature(s): {config.index_path}",
                logger=active_logger,
                stdout=active_stdout,
            )
            return result
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
    "DEFAULT_RASTER_GLOBS",
    "FOOTPRINT_EDGE_SAMPLES",
    "StaleVectorIndexError",
    "VectorIndexBuildConfig",
    "VectorIndexLockError",
    "VectorIndexValidationError",
    "VectorIndexValidationResult",
    "create_vector_index",
    "discover_raster_paths",
    "ensure_vector_index",
    "resolve_index_worker_count",
    "validate_vector_index",
]
