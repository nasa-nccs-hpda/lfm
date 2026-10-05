"""Request discovery and target-grid/AOI construction for chip preflight."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import json
import math
from pathlib import Path
import warnings

from .chip_config import SplitName
from .chip_types import (
    ChipRequest,
    GeographicAOI,
    LabelInput,
    ReferenceSample,
    SourceSelector,
    TargetGrid,
    validate_request_contracts,
)
from .lunar_crs import load_lunar_geographic_wkt


DEFAULT_SAMPLE_ID_SUFFIXES = (
    "_input_nac_static_chip",
    "_input_wac_static_chip",
    "_input_nac_chip",
    "_input_wac_chip",
    "_input_chip",
    "_mask_orig",
    "_label",
    "_mask",
    "_chip",
    "_img",
)
REFERENCE_SUFFIXES = (".tif", ".tiff")
DEFAULT_EDGE_SAMPLES = 21
LTM_MIN_LATITUDE = -82.0
LTM_MAX_LATITUDE = 82.0


class UnsupportedCoverageError(ValueError):
    """A target footprint extends outside numbered-LTM coverage."""

    status = "unsupported_polar_coverage"


class AmbiguousLongitudeError(ValueError):
    """A geographic footprint does not have a unique sub-180-degree envelope."""

    status = "ambiguous_longitude_span"


def normalize_sample_id(
    path_or_name: str | Path,
    *,
    suffixes: Sequence[str] = DEFAULT_SAMPLE_ID_SUFFIXES,
) -> str:
    """Strip only a documented terminal role suffix while preserving case."""
    stem = Path(path_or_name).stem
    for suffix in sorted(suffixes, key=len, reverse=True):
        if stem.casefold().endswith(str(suffix).casefold()):
            stem = stem[: -len(suffix)]
            break
    if not stem:
        raise ValueError(f"Could not derive a sample ID from {path_or_name!r}.")
    return stem


def product_id_from_sample_id(sample_id: str) -> str:
    """Return the first filename component used by built-in WAC/NAC selectors."""
    product_id = str(sample_id).split("_", 1)[0].strip()
    if not product_id:
        raise ValueError(f"Could not derive a product ID from {sample_id!r}.")
    return product_id


def discover_reference_tiffs(
    directory: str | Path,
    *,
    recursive: bool = False,
) -> tuple[Path, ...]:
    """Return deterministic TIFF references from one directory."""
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Reference directory does not exist: {root}")
    candidates = root.rglob("*") if recursive else root.iterdir()
    paths = tuple(
        sorted(
            (
                path
                for path in candidates
                if path.is_file() and path.suffix.casefold() in REFERENCE_SUFFIXES
            ),
            key=lambda path: str(path).casefold(),
        )
    )
    if not paths:
        raise FileNotFoundError(f"No .tif or .tiff references found in {root}.")
    return paths


def materialize_requests(
    requests: Iterable[ChipRequest],
    *,
    sample_limit: int | None = None,
) -> tuple[ChipRequest, ...]:
    """Validate and sort a request iterable independently of discovery order."""
    materialized = tuple(requests)
    validate_request_contracts(materialized)
    ordered = tuple(
        sorted(
            materialized,
            key=lambda request: (request.sample_id.casefold(), request.sample_id),
        )
    )
    if sample_limit is None:
        return ordered
    if isinstance(sample_limit, bool) or not isinstance(sample_limit, int):
        raise TypeError("sample_limit must be an integer or None.")
    if sample_limit < 1:
        raise ValueError("sample_limit must be positive when configured.")
    return ordered[:sample_limit]


def pixel_to_projected(
    transform: Sequence[float],
    pixel: float,
    line: float,
) -> tuple[float, float]:
    """Apply a six-element GDAL affine geotransform."""
    x = transform[0] + pixel * transform[1] + line * transform[2]
    y = transform[3] + pixel * transform[4] + line * transform[5]
    return float(x), float(y)


def perimeter_pixels(
    width: int,
    height: int,
    samples_per_edge: int = DEFAULT_EDGE_SAMPLES,
) -> tuple[tuple[float, float], ...]:
    """Densify all four raster edges without duplicating their corners."""
    if isinstance(samples_per_edge, bool) or not isinstance(samples_per_edge, int):
        raise TypeError("samples_per_edge must be an integer.")
    if samples_per_edge < 2:
        raise ValueError("samples_per_edge must be at least 2.")
    fractions = tuple(
        index / (samples_per_edge - 1) for index in range(samples_per_edge)
    )
    return (
        *((fraction * width, 0.0) for fraction in fractions),
        *((float(width), fraction * height) for fraction in fractions[1:]),
        *(
            ((1.0 - fraction) * width, float(height))
            for fraction in fractions[1:]
        ),
        *((0.0, (1.0 - fraction) * height) for fraction in fractions[1:-1]),
    )


def raster_bounds(
    transform: Sequence[float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    corners = (
        pixel_to_projected(transform, 0.0, 0.0),
        pixel_to_projected(transform, float(width), 0.0),
        pixel_to_projected(transform, float(width), float(height)),
        pixel_to_projected(transform, 0.0, float(height)),
    )
    xs = tuple(point[0] for point in corners)
    ys = tuple(point[1] for point in corners)
    return min(xs), min(ys), max(xs), max(ys)


def validate_target_grid_consistency(grid: TargetGrid) -> None:
    """Require the affine grid perimeter to agree with its declared bounds."""
    if not isinstance(grid, TargetGrid):
        raise TypeError("grid must be a TargetGrid.")
    derived = raster_bounds(grid.transform, grid.width, grid.height)
    scale = max(*(abs(value) for value in (*grid.bounds, *derived)), 1.0)
    tolerance = scale * 1e-10
    if any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)
        for actual, expected in zip(derived, grid.bounds)
    ):
        raise ValueError(
            "Target grid transform, dimensions, and bounds are inconsistent: "
            f"affine-derived bounds are {derived}, declared bounds are {grid.bounds}."
        )


def target_grid_from_bounds(
    *,
    crs_wkt: str,
    bounds: Sequence[float],
    width: int,
    height: int,
    transform: Sequence[float] | None = None,
) -> TargetGrid:
    """Build a complete grid, deriving a north-up affine when omitted."""
    bounds_tuple = tuple(bounds)
    if len(bounds_tuple) != 4:
        raise ValueError("bounds must contain left, bottom, right, and top.")
    if transform is None:
        if isinstance(width, bool) or not isinstance(width, int) or width < 1:
            raise ValueError("width must be a positive integer.")
        if isinstance(height, bool) or not isinstance(height, int) or height < 1:
            raise ValueError("height must be a positive integer.")
        left, bottom, right, top = (float(value) for value in bounds_tuple)
        transform = (
            left,
            (right - left) / width,
            0.0,
            top,
            0.0,
            -(top - bottom) / height,
        )
    grid = TargetGrid(
        crs_wkt=crs_wkt,
        transform=transform,  # type: ignore[arg-type]
        bounds=bounds_tuple,  # type: ignore[arg-type]
        width=width,
        height=height,
    )
    validate_target_grid_consistency(grid)
    return grid


def longitude_envelope(
    longitudes: Sequence[float],
) -> tuple[float, float, float, bool]:
    """Return the smallest continuous longitude interval containing all points."""
    if not longitudes:
        raise ValueError("At least one longitude is required.")
    normalized = sorted(float(value) % 360.0 for value in longitudes)
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("Longitudes must be finite.")
    if len(normalized) == 1:
        west = normalized[0] if normalized[0] <= 180.0 else normalized[0] - 360.0
        return west, west, 0.0, False
    gaps = [
        normalized[index + 1] - normalized[index]
        for index in range(len(normalized) - 1)
    ]
    gaps.append(normalized[0] + 360.0 - normalized[-1])
    largest_gap_index = max(range(len(gaps)), key=gaps.__getitem__)
    span = 360.0 - gaps[largest_gap_index]
    if span >= 180.0:
        raise AmbiguousLongitudeError(
            "Target footprint longitude span must be less than 180 degrees; "
            f"computed span was {span}."
        )
    start = normalized[(largest_gap_index + 1) % len(normalized)]
    west = start if start <= 180.0 else start - 360.0
    east = west + span
    return west, east, span, east > 180.0


def _create_transformation(source_srs, target_srs):
    from osgeo import osr

    source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    with osr.ExceptionMgr(useExceptions=False):
        transformation = osr.CoordinateTransformation(source_srs, target_srs)
    if transformation is None:
        raise RuntimeError("Could not construct the requested CRS transformation.")
    return transformation


def _spatial_reference(crs_wkt: str):
    try:
        from osgeo import osr
    except ImportError as exc:
        raise RuntimeError(
            "GDAL Python bindings are required to transform target grids."
        ) from exc
    spatial_reference = osr.SpatialReference()
    if spatial_reference.ImportFromWkt(str(crs_wkt)) != 0:
        raise ValueError("Could not parse target-grid CRS WKT.")
    return spatial_reference


def _transform_point(transformation, x: float, y: float) -> tuple[float, float]:
    transformed = transformation.TransformPoint(float(x), float(y))
    if transformed is None or len(transformed) < 2:
        raise RuntimeError(f"Coordinate transformation failed for ({x}, {y}).")
    first, second = float(transformed[0]), float(transformed[1])
    if not math.isfinite(first) or not math.isfinite(second):
        raise RuntimeError(
            "Coordinate transformation returned non-finite coordinates for "
            f"({x}, {y})."
        )
    return first, second


def geographic_aoi_from_target_grid(
    grid: TargetGrid,
    *,
    edge_samples: int = DEFAULT_EDGE_SAMPLES,
) -> GeographicAOI:
    """Transform a densified grid perimeter to one logical lunar-geographic AOI."""
    validate_target_grid_consistency(grid)
    source_srs = _spatial_reference(grid.crs_wkt)
    target_srs = _spatial_reference(load_lunar_geographic_wkt())
    forward = _create_transformation(source_srs, target_srs)
    reverse = _create_transformation(target_srs.Clone(), source_srs.Clone())
    latitudes: list[float] = []
    longitudes: list[float] = []
    projected_points: list[tuple[float, float]] = []
    for pixel, line in perimeter_pixels(grid.width, grid.height, edge_samples):
        projected_x, projected_y = pixel_to_projected(
            grid.transform,
            pixel,
            line,
        )
        longitude, latitude = _transform_point(forward, projected_x, projected_y)
        round_trip_x, round_trip_y = _transform_point(reverse, longitude, latitude)
        projected_points.append((projected_x, projected_y))
        latitudes.append(latitude)
        longitudes.append(longitude)
        pixel_scale = max(
            abs(grid.transform[1]),
            abs(grid.transform[2]),
            abs(grid.transform[4]),
            abs(grid.transform[5]),
            1.0,
        )
        tolerance = pixel_scale * 1e-6
        x_matches = math.isclose(
            projected_x,
            round_trip_x,
            rel_tol=1e-10,
            abs_tol=tolerance,
        )
        if not x_matches and source_srs.IsGeographic():
            wrapped_difference = (
                (projected_x - round_trip_x + 180.0) % 360.0
            ) - 180.0
            x_matches = math.isclose(
                wrapped_difference,
                0.0,
                rel_tol=0.0,
                abs_tol=tolerance,
            )
        if not (
            x_matches
            and math.isclose(
                projected_y,
                round_trip_y,
                rel_tol=1e-10,
                abs_tol=tolerance,
            )
        ):
            raise ValueError(
                "Target-grid perimeter failed geographic CRS round-trip "
                f"coverage at ({projected_x}, {projected_y})."
            )
    if not projected_points:
        raise ValueError("Target-grid perimeter is empty.")
    west, east, _, _ = longitude_envelope(longitudes)
    aoi = GeographicAOI(max(latitudes), west, min(latitudes), east)
    validate_numbered_ltm_coverage(aoi)
    return aoi


def validate_numbered_ltm_coverage(aoi: GeographicAOI) -> None:
    """Reject footprints outside the supported -82 to +82 degree LTM path."""
    if not isinstance(aoi, GeographicAOI):
        raise TypeError("aoi must be a GeographicAOI.")
    if (
        aoi.upper_left_latitude > LTM_MAX_LATITUDE
        or aoi.lower_right_latitude < LTM_MIN_LATITUDE
    ):
        raise UnsupportedCoverageError(
            "Target footprint extends outside numbered LTM coverage "
            f"[{LTM_MIN_LATITUDE}, {LTM_MAX_LATITUDE}] degrees latitude."
        )


def geographic_query_parts(aoi: GeographicAOI) -> tuple[GeographicAOI, ...]:
    """Convert one logical AOI into one or two non-wrapping tiler queries."""
    validate_numbered_ltm_coverage(aoi)
    logical_west = aoi.upper_left_longitude
    logical_east = aoi.lower_right_longitude
    while logical_east <= logical_west:
        logical_east += 360.0
    span = logical_east - logical_west
    if span >= 180.0:
        raise AmbiguousLongitudeError(
            "Logical AOI longitude span must be less than 180 degrees; "
            f"computed span was {span}."
        )
    west = ((logical_west + 180.0) % 360.0) - 180.0
    east = west + span
    if east <= 180.0:
        return (
            GeographicAOI(
                aoi.upper_left_latitude,
                west,
                aoi.lower_right_latitude,
                east,
            ),
        )
    return (
        GeographicAOI(
            aoi.upper_left_latitude,
            west,
            aoi.lower_right_latitude,
            180.0,
        ),
        GeographicAOI(
            aoi.upper_left_latitude,
            -180.0,
            aoi.lower_right_latitude,
            east - 360.0,
        ),
    )


def validate_request_geographic_aoi(
    request: ChipRequest,
    *,
    edge_samples: int = DEFAULT_EDGE_SAMPLES,
    tolerance: float = 1e-8,
) -> tuple[GeographicAOI, ...]:
    """Verify a request AOI is the transformed envelope of its target grid."""
    if not isinstance(request, ChipRequest):
        raise TypeError("request must be a ChipRequest.")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative.")
    derived = geographic_aoi_from_target_grid(
        request.target_grid,
        edge_samples=edge_samples,
    )
    provided_parts = geographic_query_parts(request.geographic_aoi)
    derived_parts = geographic_query_parts(derived)
    if len(provided_parts) != len(derived_parts):
        raise ValueError(
            "Request geographic AOI antimeridian behavior does not match its "
            "target-grid footprint."
        )
    for provided, expected in zip(provided_parts, derived_parts):
        provided_values = (
            provided.upper_left_latitude,
            provided.upper_left_longitude,
            provided.lower_right_latitude,
            provided.lower_right_longitude,
        )
        expected_values = (
            expected.upper_left_latitude,
            expected.upper_left_longitude,
            expected.lower_right_latitude,
            expected.lower_right_longitude,
        )
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)
            for left, right in zip(provided_values, expected_values)
        ):
            raise ValueError(
                "Request geographic AOI does not match the densified transformed "
                f"target-grid footprint: provided {request.geographic_aoi}, "
                f"derived {derived}."
            )
    return provided_parts


def reference_sample_from_tiff(
    path: str | Path,
    *,
    sample_id: str | None = None,
    edge_samples: int = DEFAULT_EDGE_SAMPLES,
) -> ReferenceSample:
    """Read one reference TIFF's exact grid and derive its geographic AOI."""
    try:
        from osgeo import gdal, gdalconst
    except ImportError as exc:
        raise RuntimeError(
            "GDAL Python bindings are required to read reference TIFFs."
        ) from exc
    reference_path = Path(path)
    if reference_path.suffix.casefold() not in REFERENCE_SUFFIXES:
        raise ValueError("Reference path must end with .tif or .tiff.")
    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference TIFF does not exist: {reference_path}")
    gdal.UseExceptions()
    dataset = gdal.Open(str(reference_path), gdalconst.GA_ReadOnly)
    if dataset is None:
        raise ValueError(f"GDAL could not open reference TIFF: {reference_path}")
    try:
        width = int(dataset.RasterXSize)
        height = int(dataset.RasterYSize)
        band_count = int(dataset.RasterCount)
        if width < 1 or height < 1 or band_count < 1:
            raise ValueError(
                "Reference TIFF requires positive dimensions and at least one band."
            )
        source_srs = dataset.GetSpatialRef()
        if source_srs is None:
            raise ValueError(f"Reference TIFF has no embedded CRS: {reference_path}")
        crs_wkt = source_srs.ExportToWkt().strip()
        if not crs_wkt:
            raise ValueError(f"Reference TIFF has an empty CRS: {reference_path}")
        transform = dataset.GetGeoTransform(can_return_null=True)
        if transform is None:
            raise ValueError(
                f"Reference TIFF has no affine transform: {reference_path}"
            )
        transform = tuple(float(value) for value in transform)
        bounds = raster_bounds(transform, width, height)
        grid = TargetGrid(
            crs_wkt=crs_wkt,
            transform=transform,
            bounds=bounds,
            width=width,
            height=height,
        )
        validate_target_grid_consistency(grid)
        aoi = geographic_aoi_from_target_grid(grid, edge_samples=edge_samples)
        descriptions = tuple(
            dataset.GetRasterBand(index).GetDescription() or None
            for index in range(1, band_count + 1)
        )
        return ReferenceSample(
            path=reference_path,
            sample_id=normalize_sample_id(reference_path)
            if sample_id is None
            else sample_id,
            target_grid=grid,
            geographic_aoi=aoi,
            band_count=band_count,
            band_descriptions=descriptions,
        )
    finally:
        dataset = None


def chip_request_from_reference(
    path: str | Path,
    *,
    split_group_key: str,
    sample_id: str | None = None,
    label_path: str | Path | None = None,
    label_grid: TargetGrid | None = None,
    assigned_split: SplitName | None = None,
    source_selectors: Sequence[SourceSelector] = (),
    edge_samples: int = DEFAULT_EDGE_SAMPLES,
    label_input: LabelInput | None = None,
) -> ChipRequest:
    """Construct one request from a validated reference TIFF."""
    reference = reference_sample_from_tiff(
        path,
        sample_id=sample_id,
        edge_samples=edge_samples,
    )
    return ChipRequest(
        sample_id=reference.sample_id,
        target_grid=reference.target_grid,
        geographic_aoi=reference.geographic_aoi,
        split_group_key=split_group_key,
        label_path=label_path,
        label_grid=label_grid,
        assigned_split=assigned_split,
        source_selectors=tuple(source_selectors),
        reference_path=reference.path,
        label_input=label_input,
    )


def _projected_to_pixel(grid: TargetGrid, x: float, y: float) -> tuple[float, float]:
    origin_x, a, b, origin_y, c, d = grid.transform
    determinant = a * d - b * c
    x, y = x - origin_x, y - origin_y
    return (d * x - b * y) / determinant, (-c * x + a * y) / determinant


def target_grid_from_pixel_bounds(
    source_grid: TargetGrid, pixel_bounds: Sequence[float],
) -> TargetGrid:
    """Round outward on the original pixel lattice, retaining rotated affines.

    Bounds are min-column, min-row, max-column, max-row. Negative windows are
    allowed: imagery coverage/NoData is evaluated separately, never by cropping
    the requested output to the source extent.
    """
    validate_target_grid_consistency(source_grid)
    if len(pixel_bounds) != 4 or any(not math.isfinite(v) for v in pixel_bounds):
        raise ValueError("pixel_bounds must contain four finite values.")
    left, top, right, bottom = pixel_bounds
    if left >= right or top >= bottom:
        raise ValueError("pixel_bounds must have positive width and height.")

    def snap(value):
        nearest = round(value)
        return nearest if abs(value - nearest) <= 1e-8 else value

    col, row = math.floor(snap(left)), math.floor(snap(top))
    end_col, end_row = math.ceil(snap(right)), math.ceil(snap(bottom))
    origin_x, origin_y = pixel_to_projected(source_grid.transform, col, row)
    _, a, b, _, c, d = source_grid.transform
    affine = (origin_x, a, b, origin_y, c, d)
    width, height = end_col - col, end_row - row
    return TargetGrid(source_grid.crs_wkt, affine, raster_bounds(affine, width, height), width, height)


def static_grid_reference(aoi: GeographicAOI) -> TargetGrid:
    """100 m, zero-anchored LTM lattice in the zone containing the AOI center."""
    geographic_query_parts(aoi)  # also checks polar coverage and ambiguous spans
    span = (aoi.lower_right_longitude - aoi.upper_left_longitude) % 360.0
    longitude = ((aoi.upper_left_longitude + span / 2 + 180) % 360) - 180
    latitude = (aoi.upper_left_latitude + aoi.lower_right_latitude) / 2
    zone = math.floor((longitude + 180) / 8) + 1
    hemisphere = "N" if latitude >= 0 else "S"
    path = Path(__file__).resolve().parents[1] / "TMS" / "RG" / f"tms_LTM_{zone}{hemisphere}RG.json"
    crs = json.loads(path.read_text())["crs"]
    # The one-pixel extent is only a lattice reference, not a coverage limit.
    return TargetGrid(crs, (0, 100, 0, 0, 0, -100), (0, -100, 100, 0), 1, 1)


def _select_source_grid(source_grids: Mapping[str, Sequence[TargetGrid]]) -> TargetGrid:
    """Resolve modality precedence; competing lattices need an explicit choice.

    WAC candidates must be the original VIS rasters, not UV or tiled datacubes.
    """
    candidates = {str(name).lower(): tuple(grids) for name, grids in source_grids.items()}
    if len(candidates) != len(source_grids) or not candidates or any(not grids for grids in candidates.values()):
        raise ValueError("source_grids must have unique modality names and nonempty candidate lists.")
    if any(not isinstance(grid, TargetGrid) for grids in candidates.values() for grid in grids):
        raise TypeError("source_grids candidates must be TargetGrid records.")
    if len(candidates) > 1:
        if "wac" not in candidates:
            raise ValueError("Multiple dynamic modalities require an explicit source_grid when WAC is absent.")
        warnings.warn("Multiple dynamic modalities: WAC takes precedence for the output grid.", UserWarning, stacklevel=3)
        grids = candidates["wac"]
    else:
        grids = next(iter(candidates.values()))
    chosen = grids[0]
    for grid in grids:
        validate_target_grid_consistency(grid)
        # Equal WKT is the dependency-free fast path; equivalent WKT encodings
        # are accepted using the same GDAL CRS comparison as the rest of chips.
        same_crs = chosen.crs_wkt == grid.crs_wkt or bool(
            _spatial_reference(chosen.crs_wkt).IsSame(_spatial_reference(grid.crs_wkt)))
        same_linear = all(math.isclose(chosen.transform[i], grid.transform[i], rel_tol=0, abs_tol=1e-10)
                          for i in (1, 2, 4, 5))
        origin = _projected_to_pixel(chosen, grid.transform[0], grid.transform[3])
        if not same_crs or not same_linear or any(abs(v - round(v)) > 1e-8 for v in origin):
            raise ValueError("Competing source pixel lattices require an explicit source_grid reference.")
    return chosen


def target_grid_from_geographic_aoi(
    aoi: GeographicAOI, source_grid: TargetGrid,
) -> TargetGrid:
    """Transform an IAU:30100 boundary and round outward on a native lattice."""
    geographic_query_parts(aoi)
    validate_target_grid_consistency(source_grid)
    transformation = _create_transformation(
        _spatial_reference(load_lunar_geographic_wkt()),
        _spatial_reference(source_grid.crs_wkt),
    )
    west = ((aoi.upper_left_longitude + 180) % 360) - 180
    east = west + ((aoi.lower_right_longitude - aoi.upper_left_longitude) % 360)
    north, south = aoi.upper_left_latitude, aoi.lower_right_latitude

    def project(point):
        return _projected_to_pixel(source_grid, *_transform_point(transformation, *point))

    points = []

    def refine(start, end, first, last, depth=0):
        middle = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        mapped = project(middle)
        error = math.hypot(mapped[0] - (first[0] + last[0]) / 2,
                           mapped[1] - (first[1] + last[1]) / 2)
        if error > 1e-4:
            if depth >= 20:
                raise ValueError("AOI boundary transformation did not converge within 20 subdivisions.")
            refine(start, middle, first, mapped, depth + 1)
            refine(middle, end, mapped, last, depth + 1)
        else:
            points.extend((first, mapped, last))

    corners = ((west, north), (east, north), (east, south), (west, south))
    for start, end in zip(corners, corners[1:] + corners[:1]):
        samples = [(start[0] + (end[0] - start[0]) * i / 20,
                    start[1] + (end[1] - start[1]) * i / 20) for i in range(21)]
        for first, last in zip(samples, samples[1:]):
            refine(first, last, project(first), project(last))
    return target_grid_from_pixel_bounds(source_grid, (
        min(p[0] for p in points), min(p[1] for p in points),
        max(p[0] for p in points), max(p[1] for p in points),
    ))


def chip_request_from_aoi(
    *,
    sample_id: str,
    crs_wkt: str | None = None,
    bounds: Sequence[float] | None = None,
    width: int | None = None,
    height: int | None = None,
    split_group_key: str,
    transform: Sequence[float] | None = None,
    label_path: str | Path | None = None,
    label_grid: TargetGrid | None = None,
    assigned_split: SplitName | None = None,
    source_selectors: Sequence[SourceSelector] = (),
    edge_samples: int = DEFAULT_EDGE_SAMPLES,
    geographic_aoi: GeographicAOI | None = None,
    source_grid: TargetGrid | None = None,
    source_grids: Mapping[str, Sequence[TargetGrid]] | None = None,
    static_only: bool = False,
    label_input: LabelInput | None = None,
) -> ChipRequest:
    """Construct a geographic AOI request, or preserve the explicit-grid API.

    Geographic mode requires original pre-tiling source metadata (WAC VIS),
    resolved candidates, or ``static_only=True``. It does not read imagery or
    labels. ``geographic_aoi`` is IAU:30100; output pixels use the native grid.
    """
    requested_aoi = geographic_aoi
    if not isinstance(static_only, bool):
        raise TypeError("static_only must be a boolean.")
    if geographic_aoi is not None:
        if any(value is not None for value in (crs_wkt, bounds, width, height, transform)):
            raise ValueError("Geographic AOI and explicit target-grid arguments cannot be mixed.")
        if sum((source_grid is not None, source_grids is not None, static_only)) != 1:
            raise ValueError("Choose exactly one of source_grid, source_grids, or static_only.")
        reference = (static_grid_reference(geographic_aoi) if static_only else
                     _select_source_grid(source_grids) if source_grids is not None else source_grid)
        grid = target_grid_from_geographic_aoi(geographic_aoi, reference)
    else:
        if source_grid is not None or source_grids is not None or static_only:
            raise ValueError("Source grid selection requires geographic_aoi.")
        if any(value is None for value in (crs_wkt, bounds, width, height)):
            raise ValueError("Explicit-grid mode requires crs_wkt, bounds, width, and height.")
        grid = target_grid_from_bounds(crs_wkt=crs_wkt, bounds=bounds, width=width,
                                       height=height, transform=transform)
    realized_aoi = geographic_aoi_from_target_grid(
        grid,
        edge_samples=edge_samples,
    )
    return ChipRequest(
        sample_id=sample_id,
        target_grid=grid,
        geographic_aoi=realized_aoi,
        split_group_key=split_group_key,
        label_path=label_path,
        label_grid=label_grid,
        assigned_split=assigned_split,
        source_selectors=tuple(source_selectors),
        label_input=label_input,
        requested_aoi=requested_aoi,
    )


def chip_requests_from_reference_paths(
    paths: Iterable[str | Path],
    *,
    split_group_key: Callable[[ReferenceSample], str],
    edge_samples: int = DEFAULT_EDGE_SAMPLES,
    sample_limit: int | None = None,
) -> tuple[ChipRequest, ...]:
    """Build deterministic requests from an explicit path or glob iterable."""
    if not callable(split_group_key):
        raise TypeError("split_group_key must be a callable.")
    ordered_paths = tuple(
        sorted((Path(path) for path in paths), key=lambda path: str(path).casefold())
    )
    references = tuple(
        reference_sample_from_tiff(path, edge_samples=edge_samples)
        for path in ordered_paths
    )
    requests = tuple(
        ChipRequest(
            sample_id=reference.sample_id,
            target_grid=reference.target_grid,
            geographic_aoi=reference.geographic_aoi,
            split_group_key=split_group_key(reference),
            reference_path=reference.path,
        )
        for reference in references
    )
    return materialize_requests(requests, sample_limit=sample_limit)


def chip_requests_from_reference_directory(
    directory: str | Path,
    *,
    split_group_key: Callable[[ReferenceSample], str],
    recursive: bool = False,
    edge_samples: int = DEFAULT_EDGE_SAMPLES,
    sample_limit: int | None = None,
) -> tuple[ChipRequest, ...]:
    """Discover reference TIFFs and build requests using the same path API."""
    return chip_requests_from_reference_paths(
        discover_reference_tiffs(directory, recursive=recursive),
        split_group_key=split_group_key,
        edge_samples=edge_samples,
        sample_limit=sample_limit,
    )


__all__ = [
    "AmbiguousLongitudeError",
    "DEFAULT_EDGE_SAMPLES",
    "DEFAULT_SAMPLE_ID_SUFFIXES",
    "LTM_MAX_LATITUDE",
    "LTM_MIN_LATITUDE",
    "REFERENCE_SUFFIXES",
    "UnsupportedCoverageError",
    "chip_request_from_aoi",
    "chip_request_from_reference",
    "chip_requests_from_reference_directory",
    "chip_requests_from_reference_paths",
    "discover_reference_tiffs",
    "geographic_aoi_from_target_grid",
    "geographic_query_parts",
    "longitude_envelope",
    "materialize_requests",
    "normalize_sample_id",
    "perimeter_pixels",
    "pixel_to_projected",
    "product_id_from_sample_id",
    "raster_bounds",
    "reference_sample_from_tiff",
    "target_grid_from_bounds",
    "target_grid_from_geographic_aoi",
    "target_grid_from_pixel_bounds",
    "static_grid_reference",
    "validate_numbered_ltm_coverage",
    "validate_request_geographic_aoi",
    "validate_target_grid_consistency",
]
