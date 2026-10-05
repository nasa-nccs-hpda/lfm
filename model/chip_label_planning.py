"""Read-only source-label validation and compact worker preparation plans.

No dataset/intermediate directories, rasterized masks, or output labels are
created here. GDAL and NumPy are imported only on paths that need them.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

from .chip_labels import (
    _crs_is_same, _grid_from_metadata, _label_error, _numpy, _sidecar_path,
    _validate_grid_match, _validate_instance_archive, _validate_mask, validate_label,
)
from .chip_requests import (
    _create_transformation, _projected_to_pixel, _spatial_reference,
    _transform_point, pixel_to_projected, raster_bounds, validate_target_grid_consistency,
)
from .chip_types import (
    ChipRequest, LabelInput, LabelMismatchError, LabelPreparationPlan,
    LabelValidationDiagnostic, TargetGrid,
)

if TYPE_CHECKING:
    from .chip_preflight import PreparedChipRequest


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lunar_srs(request: ChipRequest, wkt: str):
    try:
        srs = _spatial_reference(wkt)
        if not (srs.IsGeographic() or srs.IsProjected()):
            raise ValueError("A geographic or projected lunar CRS is required.")
        if not all(math.isclose(radius, 1737400, rel_tol=0, abs_tol=1e-5)
                   for radius in (srs.GetSemiMajor(), srs.GetSemiMinor())):
            raise ValueError("Label CRS must use the IAU:30100 lunar sphere.")
        return srs
    except Exception as exc:
        raise _label_error(request, code="incompatible_label_crs", message=str(exc)) from exc


def _resolve_grid(
    request: ChipRequest, source: LabelInput, embedded: TargetGrid | None = None,
    *, allow_missing: bool = False,
) -> LabelInput:
    """Read one selected sidecar; never reinterpret legacy target_grid as a scene."""
    try:
        sidecar = source.sidecar_path or _sidecar_path(source.path)
    except ValueError as exc:
        raise _label_error(request, code="ambiguous_label_sidecar", message=str(exc)) from exc
    grids = [grid for grid in (source.source_grid, embedded) if grid is not None]
    if sidecar is not None:
        try:
            document = json.loads(sidecar.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise ValueError("Sidecar must contain an object.")
            grid = _grid_from_metadata(document)
            if "source_grid" not in document:
                # Old target_grid (including bare-grid JSON) is an exact-chip
                # association, never implicit permission to clip a parent label.
                _validate_grid_match(request, grid)
            grids.append(grid)
        except LabelMismatchError:
            raise
        except Exception as exc:
            raise _label_error(request, code="malformed_label_metadata", message=str(exc)) from exc
    if not grids:
        if allow_missing:
            return source
        raise _label_error(request, code="missing_label_grid",
                           message="Array clipping requires source_grid metadata or a source_grid sidecar.")
    for grid in grids:
        validate_target_grid_consistency(grid)
    for grid in grids[1:]:
        first = grids[0]
        matches = (_crs_is_same(first.crs_wkt, grid.crs_wkt)
                   and (first.width, first.height) == (grid.width, grid.height))
        for col, row in ((0, 0), (grid.width, 0), (0, grid.height), (grid.width, grid.height)):
            x, y = _projected_to_pixel(first, *pixel_to_projected(grid.transform, col, row))
            matches = matches and abs(x - col) <= 1e-8 and abs(y - row) <= 1e-8
        if not matches:
            raise _label_error(request, code="label_grid_mismatch",
                               message="Explicit, embedded, and sidecar label grids disagree.")
    return replace(source, source_grid=grids[0], sidecar_path=sidecar)


def _pixel_mapper(request: ChipRequest, source_grid: TargetGrid):
    """Target pixel coordinates -> source pixels, with invertibility checks."""
    target = request.target_grid
    source_srs = _lunar_srs(request, source_grid.crs_wkt)
    target_srs = _lunar_srs(request, target.crs_wkt)
    same = _crs_is_same(source_grid.crs_wkt, target.crs_wkt)
    forward = None if same else _create_transformation(target_srs, source_srs)
    reverse = None if same else _create_transformation(source_srs.Clone(), target_srs.Clone())

    def branch(x, grid):
        center, _ = pixel_to_projected(grid.transform, grid.width / 2, grid.height / 2)
        return x + 360 * round((center - x) / 360)

    def map_pixel(col, row):
        x, y = pixel_to_projected(target.transform, col, row)
        sx, sy = (x, y) if same else _transform_point(forward, x, y)
        if source_srs.IsGeographic():
            sx = branch(sx, source_grid)
        pixel = _projected_to_pixel(source_grid, sx, sy)
        rx, ry = (sx, sy) if same else _transform_point(reverse, sx, sy)
        if target_srs.IsGeographic():
            rx = branch(rx, target)
        back = _projected_to_pixel(target, rx, ry)
        error = math.hypot(back[0] - col, back[1] - row)
        if not all(math.isfinite(value) for value in (*pixel, *back)) or error > 1e-4:
            raise _label_error(
                request, code="invalid_label_transform",
                message=(
                    "Label-to-target CRS mapping failed its pixel-space round trip "
                    f"at ({col:.9g}, {row:.9g}): returned "
                    f"({back[0]:.9g}, {back[1]:.9g}), error={error:.9g} pixels "
                    "(tolerance=0.0001)."
                ),
                expected=(col, row), actual=back,
            )
        return pixel

    return map_pixel, same


def _covered_footprint(request: ChipRequest, source_grid: TargetGrid):
    """Check a densified footprint and retain its ordered source-pixel boundary."""
    validate_target_grid_consistency(source_grid)
    validate_target_grid_consistency(request.target_grid)
    target = request.target_grid
    mapper, same_crs = _pixel_mapper(request, source_grid)
    tolerance = 1e-8
    boundary = []

    def check(point):
        pixel = mapper(*point)
        if not (-tolerance <= pixel[0] <= source_grid.width + tolerance
                and -tolerance <= pixel[1] <= source_grid.height + tolerance):
            raise _label_error(request, code="incomplete_label_coverage",
                               message="Realized target footprint extends beyond the source label grid.")
        return pixel

    def refine(start, end, first, last, depth=0):
        middle = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        mapped = check(middle)
        # Measure curvature in target pixels, not map units or source pixels.
        # Probe toward the farther raster edge, capped at one pixel. Always
        # adding one steps outside bottom/right edges (even an entire one-row
        # chip), unnecessarily testing the CRS outside the required footprint.
        # Divide by the signed steps to preserve the Jacobian's scale/direction.
        def inward_step(coordinate, extent):
            return (min(1.0, extent - coordinate) if coordinate <= extent / 2
                    else -min(1.0, coordinate))

        col_step = inward_step(middle[0], target.width)
        row_step = inward_step(middle[1], target.height)
        dx = mapper(middle[0] + col_step, middle[1])
        dy = mapper(middle[0], middle[1] + row_step)
        a, c = (dx[0] - mapped[0]) / col_step, (dx[1] - mapped[1]) / col_step
        b, d = (dy[0] - mapped[0]) / row_step, (dy[1] - mapped[1]) / row_step
        determinant = a * d - b * c
        if not math.isfinite(determinant) or abs(determinant) < 1e-20:
            raise _label_error(request, code="invalid_label_transform", message="Label mapping is locally singular.")
        ex, ey = mapped[0] - (first[0] + last[0]) / 2, mapped[1] - (first[1] + last[1]) / 2
        if math.hypot((d * ex - b * ey) / determinant, (-c * ex + a * ey) / determinant) > 1e-4:
            if depth >= 20:
                raise _label_error(request, code="invalid_label_transform",
                                   message="Label footprint densification did not converge.")
            refine(start, middle, first, mapped, depth + 1)
            refine(middle, end, mapped, last, depth + 1)
        else:
            boundary.extend((first, mapped, last))

    corners = ((0, 0), (target.width, 0), (target.width, target.height), (0, target.height))
    for start, end in zip(corners, corners[1:] + corners[:1]):
        points = [(start[0] + (end[0] - start[0]) * i / 20,
                   start[1] + (end[1] - start[1]) * i / 20) for i in range(21)]
        for first, last in zip(points, points[1:]):
            refine(first, last, check(first), check(last))
    return mapper, same_crs, boundary


def classify_label_grid(
    request: ChipRequest, source_grid: TargetGrid,
) -> tuple[str, tuple[int, int, int, int] | None]:
    """Return exact/aligned/warp method and optional covered integer window."""
    mapper, same_crs, _ = _covered_footprint(request, source_grid)
    target = request.target_grid
    tolerance = 1e-8
    corners = ((0, 0), (target.width, 0), (target.width, target.height), (0, target.height))
    origin = mapper(0, 0)
    x_axis, y_axis = mapper(1, 0), mapper(0, 1)
    aligned = same_crs and all(abs(a - b) <= tolerance for a, b in zip(
        (x_axis[0] - origin[0], x_axis[1] - origin[1],
         y_axis[0] - origin[0], y_axis[1] - origin[1]), (1, 0, 0, 1)))
    for col, row in corners:
        x, y = mapper(col, row)
        aligned = aligned and abs(x - origin[0] - col) <= tolerance and abs(y - origin[1] - row) <= tolerance
    if aligned and all(abs(v - round(v)) <= tolerance for v in origin):
        window = (round(origin[0]), round(origin[1]), target.width, target.height)
        if window == (0, 0, source_grid.width, source_grid.height):
            return "exact", None
        return "aligned_window", window
    return "nearest_warp", None


def _validate_raster_coverage_mask(request, grid, mask_band):
    """Reject unknown source cells intersecting the chip, even when downsampling.

    Checking only nearest target centers would hide small unlabelled gaps.
    Validity is read in bounded blocks; no full-scene mask is allocated.
    """
    from osgeo import ogr

    np = _numpy()
    _, _, boundary = _covered_footprint(request, grid)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for point in boundary:
        x, y = (round(v) if abs(v - round(v)) <= 1e-8 else v for v in point)
        ring.AddPoint_2D(x, y)
    ring.CloseRings()
    footprint = ogr.Geometry(ogr.wkbPolygon)
    footprint.AddGeometry(ring)
    if not footprint.IsValid():
        raise _label_error(request, code="invalid_label_transform", message="Transformed label footprint is not a valid polygon.")
    left, right, top, bottom = footprint.GetEnvelope()
    left, right = max(0, math.floor(left)), min(grid.width, math.ceil(right))
    top, bottom = max(0, math.floor(top)), min(grid.height, math.ceil(bottom))
    for row in range(top, bottom, 256):
        for col in range(left, right, 256):
            values = mask_band.ReadAsArray(col, row, min(256, right - col), min(256, bottom - row))
            if values is None:
                raise ValueError("Could not read label validity mask.")
            for y, x in np.argwhere(values == 0):
                x, y = col + int(x), row + int(y)
                cell_ring = ogr.Geometry(ogr.wkbLinearRing)
                for px, py in ((x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1), (x, y)):
                    cell_ring.AddPoint_2D(px, py)
                cell = ogr.Geometry(ogr.wkbPolygon)
                cell.AddGeometry(cell_ring)
                if footprint.Intersects(cell) and footprint.Intersection(cell).GetArea() > 0:
                    raise _label_error(request, code="label_nodata_in_target",
                                       message="Source label contains unknown/NoData pixels inside the target footprint.")


def _validate_raster(request: ChipRequest, source: LabelInput):
    from osgeo import gdal

    dataset = gdal.OpenEx(str(source.path), gdal.OF_RASTER | gdal.OF_READONLY)
    if dataset is None:
        raise ValueError(f"Could not open label raster: {source.path}")
    try:
        if dataset.GetDriver().ShortName != "GTiff":
            raise _label_error(request, code="unsupported_label_type", message="Raster label input must be a GeoTIFF.")
        if dataset.RasterCount != 1:
            raise _label_error(request, code="malformed_label", message="Semantic GeoTIFF labels must have exactly one band.")
        band = dataset.GetRasterBand(1)
        if gdal.GetDataTypeName(band.DataType) not in (
            "Byte", "Int8", "UInt16", "Int16", "UInt32", "Int32", "UInt64", "Int64",
        ):
            raise _label_error(request, code="invalid_label_dtype", message="Semantic GeoTIFF labels must contain integers.")
        affine = dataset.GetGeoTransform(can_return_null=True)
        wkt = dataset.GetProjection()
        if affine is None or not wkt:
            raise _label_error(request, code="missing_label_grid", message="GeoTIFF labels require embedded CRS and affine metadata.")
        grid = TargetGrid(wkt, affine, raster_bounds(affine, dataset.RasterXSize, dataset.RasterYSize),
                          dataset.RasterXSize, dataset.RasterYSize)
        source = _resolve_grid(request, source, embedded=grid)
        method, window = classify_label_grid(request, grid)
        if band.GetMaskFlags() != gdal.GMF_ALL_VALID:
            _validate_raster_coverage_mask(request, grid, band.GetMaskBand())
        return source, method, window, ()
    finally:
        dataset = None


def _validate_vector(request: ChipRequest, source: LabelInput):
    from osgeo import gdal, ogr

    if source.relation != "clip_to_target":
        raise _label_error(request, code="invalid_label_relation", message="GeoPackage labels require clip_to_target mode.")
    dataset = gdal.OpenEx(str(source.path), gdal.OF_VECTOR | gdal.OF_READONLY)
    if dataset is None:
        raise ValueError(f"Could not open GeoPackage: {source.path}")
    try:
        if dataset.GetDriver().ShortName != "GPKG":
            raise _label_error(request, code="unsupported_label_type", message="Vector label input must be a GeoPackage.")
        layer = dataset.GetLayerByName(source.layer)
        if layer is None:
            raise _label_error(request, code="missing_label_layer", message=f"GeoPackage has no layer {source.layer!r}.")
        srs = layer.GetSpatialRef()
        if srs is None:
            raise _label_error(request, code="missing_label_grid", message="GeoPackage layer must declare a lunar CRS.")
        source_srs = _lunar_srs(request, srs.ExportToWkt())
        target_srs = _lunar_srs(request, request.target_grid.crs_wkt)
        transform = _create_transformation(source_srs, target_srs)
        definition = layer.GetLayerDefn()
        index = definition.GetFieldIndex("crater_id")
        if index < 0 or definition.GetFieldDefn(index).GetType() not in (ogr.OFTInteger, ogr.OFTInteger64):
            raise _label_error(request, code="invalid_instance_ids", message="crater_id must be an integer field.")
        ids = set()
        for feature in layer:
            instance = feature.GetField(index)
            if not isinstance(instance, int) or instance <= 0 or instance in ids:
                raise _label_error(request, code="invalid_instance_ids", message="crater_id values must be unique positive integers.")
            ids.add(instance)
            geometry = feature.GetGeometryRef()
            if (geometry is None or geometry.IsEmpty()
                    or ogr.GT_Flatten(geometry.GetGeometryType()) not in (ogr.wkbPolygon, ogr.wkbMultiPolygon)
                    or not geometry.IsValid() or geometry.GetArea() <= 0):
                raise _label_error(request, code="invalid_label_geometry", message=f"Crater {instance} must have valid nonempty polygon geometry.")
            projected = geometry.Clone()
            if projected.Transform(transform) != 0 or not all(math.isfinite(v) for v in projected.GetEnvelope()):
                raise _label_error(request, code="invalid_label_transform", message=f"Could not transform crater {instance} onto the target CRS.")
        # Empty layers and polygons outside the chip are valid. Neither implies
        # unlabelled coverage: the scientist supplied a finished full-scene set.
        return source, "vector_rasterize", None, ()
    finally:
        dataset = None


def require_materialized_label(prepared: PreparedChipRequest) -> None:
    """Prevent direct acquisition/publication of an unmaterialized source."""
    plan = prepared.preflight.label_plan
    artifact = prepared.prepared_label
    if artifact is not None:
        if artifact.plan != plan or artifact.target_grid != prepared.request.target_grid:
            raise _label_error(prepared.request, code="label_plan_mismatch",
                               message="Prepared label does not match this sample's plan.")
        try:
            intact = artifact.path.is_file() and _hash_file(artifact.path) == artifact.sha256
        except OSError:
            intact = False
        if not intact:
            raise _label_error(prepared.request, code="label_artifact_changed",
                               message="Prepared label is missing or changed before acquisition/publication.")
        return
    if (plan is not None and plan.requires_materialization) or (
        plan is None and prepared.request.label_input is not None
        and prepared.request.label_input.relation == "clip_to_target"
    ):
        raise _label_error(
            prepared.request, code="label_materialization_required",
            message="Materialize the label before calling acquisition/publication directly.",
        )


def plan_label_preparation(request: ChipRequest, path: str | Path) -> LabelPreparationPlan:
    """Validate source structure/coverage and hash it without creating files."""
    path = Path(path)
    source = request.label_input or LabelInput(path, source_grid=request.label_grid)
    try:
        if source.path != path:
            raise ValueError("Resolved label path conflicts with explicit association.")
        if not path.is_file():
            raise _label_error(request, code="missing_label", message=f"Label does not exist: {path}")
        suffix = path.suffix.lower()
        expected_kind = {".npy": "semantic", ".npz": "raster_instance",
                         ".tif": "semantic", ".tiff": "semantic", ".gpkg": "vector_instance"}.get(suffix)
        if expected_kind is None or expected_kind != source.kind:
            raise _label_error(request, code="unsupported_label_type", message="Label kind and file format are incompatible.")
        before = _hash_file(path)
        if suffix == ".gpkg":
            source, method, window, diagnostics = _validate_vector(request, source)
        elif suffix in (".tif", ".tiff"):
            source, method, window, diagnostics = _validate_raster(request, source)
        elif source.relation == "exact":
            # Keep the existing shape-only compatibility warning for old exact
            # arrays lacking georeferencing; clipping never uses this fallback.
            source = _resolve_grid(request, source, allow_missing=True)
            validation_request = replace(request, label_input=source, label_grid=source.source_grid)
            diagnostics = validate_label(validation_request, path)
            source = replace(source, source_grid=source.source_grid or request.target_grid)
            method, window = "exact", None
        else:
            source = _resolve_grid(request, source)
            source_request = replace(request, target_grid=source.source_grid)
            np = _numpy()
            if suffix == ".npy":
                mask = np.load(path, mmap_mode="r", allow_pickle=False)
                _validate_mask(source_request, mask)
                diagnostics = ()
            else:
                with np.load(path, allow_pickle=False) as archive:
                    diagnostics = _validate_instance_archive(source_request, archive)
            method, window = classify_label_grid(request, source.source_grid)
        if source.relation == "exact" and method != "exact":
            raise _label_error(request, code="label_grid_mismatch", message="Exact label mode requires the target grid to match the source.")
        after = _hash_file(path)
        if before != after:
            raise _label_error(request, code="label_source_changed", message="Label source changed during validation.")
        diagnostic = LabelValidationDiagnostic(
            "label_preparation_planned",
            f"Validated {source.kind} source; preparation method: {method}.",
            "info",
        )
        return LabelPreparationPlan(source, request.target_grid, method, after, window,
                                    (diagnostic, *diagnostics))
    except LabelMismatchError as exc:
        exc.label_path = path
        raise
    except Exception as exc:
        raise _label_error(request, code="malformed_label", message=f"Could not plan label {path}: {exc}") from exc


__all__ = ["classify_label_grid", "plan_label_preparation"]
