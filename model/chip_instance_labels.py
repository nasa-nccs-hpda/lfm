"""Worker-local instance conversion; no imagery acquisition or publication.

Geometry operations use target pixel coordinates. Pixel centers on outlines
are included explicitly, rather than relying on a rasterizer's edge tie rule.
"""

from dataclasses import dataclass, replace
import math
import os
from pathlib import Path
import tempfile
import zipfile

from .chip_label_materialization import (
    _MaskReader, _nearest_indices, _staging_path, _verify_source, _windows,
)
from .chip_label_planning import _hash_file, _lunar_srs, _pixel_mapper, plan_label_preparation
from .chip_labels import _crs_is_same, _label_error, _numpy, _validate_instance_archive
from .chip_requests import _create_transformation, _projected_to_pixel, _transform_point, pixel_to_projected
from .chip_types import (
    ChipRequest, GeographicAOI, LabelInput, LabelMismatchError,
    LabelPreparationPlan, LabelValidationDiagnostic, PreparedLabelArtifact,
)


@dataclass(frozen=True)
class InstanceLabelConversion:
    """Arrays stay inside the worker; provenance uses compact ID pairs."""

    mask: object
    bboxes: object
    num_craters: int
    id_mapping: tuple
    diagnostics: tuple = ()

    def arrays(self):
        np = _numpy()
        return dict(mask=self.mask, bboxes=self.bboxes,
                    num_craters=np.asarray(self.num_craters, dtype=np.int64))


def _diagnostic(code, instance, message, severity="info"):
    return LabelValidationDiagnostic(code, f"Source instance {instance}: {message}", severity)


def _rectangle(left, top, right, bottom):
    from osgeo import ogr

    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in ((left, top), (right, top), (right, bottom), (left, bottom), (left, top)):
        ring.AddPoint_2D(float(x), float(y))
    result = ogr.Geometry(ogr.wkbPolygon)
    result.AddGeometry(ring)
    return result


def _edge(first, last, mapper, nonlinear):
    """21 initial samples, adaptive error <= 1e-4 target pixel, max depth 20."""
    def refine(a, b, pa, pb, depth=0):
        middle = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        pm = mapper(*middle)
        error = math.hypot(pm[0] - (pa[0] + pb[0]) / 2,
                           pm[1] - (pa[1] + pb[1]) / 2)
        if error <= 1e-4:
            return [pa]
        if depth >= 20:
            raise ValueError("Label outline transformation did not converge.")
        return refine(a, middle, pa, pm, depth + 1) + refine(middle, b, pm, pb, depth + 1)

    if not nonlinear:
        return [mapper(*first)]
    points = [(first[0] + (last[0] - first[0]) * i / 20,
               first[1] + (last[1] - first[1]) * i / 20) for i in range(21)]
    result = []
    for a, b in zip(points, points[1:]):
        result.extend(refine(a, b, mapper(*a), mapper(*b)))
    return result


def _pixel_geometry(geometry, mapper, nonlinear):
    from osgeo import ogr

    kind = ogr.GT_Flatten(geometry.GetGeometryType())
    if kind == ogr.wkbMultiPolygon:
        result = ogr.Geometry(ogr.wkbMultiPolygon)
        for child in geometry:
            result.AddGeometry(_pixel_geometry(child, mapper, nonlinear))
        if not result.IsValid():
            raise ValueError("Transformed multipart label outline is invalid.")
        return result
    result = ogr.Geometry(ogr.wkbPolygon)
    for original in geometry:
        ring = ogr.Geometry(ogr.wkbLinearRing)
        points = [original.GetPoint(i)[:2] for i in range(original.GetPointCount())]
        for first, last in zip(points, points[1:]):
            for x, y in _edge(first, last, mapper, nonlinear):
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError("Nonfinite transformed label outline.")
                ring.AddPoint_2D(x, y)
        ring.CloseRings()
        result.AddGeometry(ring)
    if not result.IsValid():
        raise ValueError("Transformed label outline is invalid.")
    return result


def _outline_mapper(request, source_wkt, source_grid=None):
    target = request.target_grid
    same = _crs_is_same(source_wkt, target.crs_wkt)
    transform = None if same else _create_transformation(
        _lunar_srs(request, source_wkt), _lunar_srs(request, target.crs_wkt))

    def mapper(x, y):
        if source_grid is not None:
            x, y = pixel_to_projected(source_grid.transform, x, y)
        if transform is not None:
            x, y = _transform_point(transform, x, y)
        return _projected_to_pixel(target, x, y)

    return mapper, not same


def _clip(geometry, target, instance, diagnostics):
    from osgeo import ogr

    footprint = _rectangle(0, 0, target.width, target.height)
    clipped = geometry.Intersection(footprint)
    if clipped is None:
        raise ValueError("Label outline intersection failed.")
    # A multipart intersection may include isolated touching lines or points.
    # Those are not crater outlines and must not enlarge the retained box.
    polygons = ogr.Geometry(ogr.wkbMultiPolygon)

    def collect(part):
        kind = ogr.GT_Flatten(part.GetGeometryType())
        if kind == ogr.wkbPolygon and part.GetArea() > 0:
            polygons.AddGeometry(part)
        elif kind in (ogr.wkbMultiPolygon, ogr.wkbGeometryCollection):
            for child in part:
                collect(child)

    collect(clipped)
    clipped = polygons
    if clipped.IsEmpty() or clipped.GetArea() <= 0:
        diagnostics.append(_diagnostic("outside_instance", instance, "no positive area inside chip."))
        return None
    if not geometry.Within(footprint):
        diagnostics.append(_diagnostic("clipped_instance", instance, "outline clipped at raster edges."))
    left, right, top, bottom = clipped.GetEnvelope()
    # Clamp numerical noise from geometric intersection, not the original outline.
    left, right = max(0., left), min(float(target.width), right)
    top, bottom = max(0., top), min(float(target.height), bottom)
    return clipped, (left, top, right - left, bottom - top)


def _box_window(box, target):
    x, y, width, height = box
    return (max(0, math.floor(x)), max(0, math.floor(y)),
            min(target.width, math.ceil(x + width)), min(target.height, math.ceil(y + height)))


def _finish(request, mask, boxes, ids, diagnostics):
    np = _numpy()
    result = InstanceLabelConversion(mask, np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
                                     len(ids), tuple((old, new) for new, old in enumerate(ids, 1)),
                                     tuple(diagnostics))
    validation = _validate_instance_archive(request, result.arrays())
    return replace(result, diagnostics=(*result.diagnostics, *validation))


def _convert_vector(request, plan):
    from osgeo import gdal, ogr

    np = _numpy()
    target = request.target_grid
    mask = np.zeros((target.height, target.width), dtype=np.int64)
    dataset = gdal.OpenEx(str(plan.source.path), gdal.OF_VECTOR | gdal.OF_READONLY)
    try:
        layer = dataset.GetLayerByName(plan.source.layer)
        mapper, nonlinear = _outline_mapper(request, layer.GetSpatialRef().ExportToWkt())
        features = sorted((feature.GetField("crater_id"), feature.GetGeometryRef().Clone()) for feature in layer)
        boxes, ids, diagnostics = [], [], []
        for instance, geometry in features:
            prepared = _clip(_pixel_geometry(geometry, mapper, nonlinear), target, instance, diagnostics)
            if prepared is None:
                continue
            clipped, box = prepared
            left, top, right, bottom = _box_window(box, target)
            # Intersects includes outline centers and excludes hole interiors.
            # Only the feature's bounded window is visited, never the full scene.
            support = np.zeros((bottom - top, right - left), dtype=bool)
            point = ogr.Geometry(ogr.wkbPoint)
            for row in range(top, bottom):
                for col in range(left, right):
                    point.SetPoint_2D(0, col + .5, row + .5)
                    support[row - top, col - left] = clipped.Intersects(point)
            if not support.any():
                diagnostics.append(_diagnostic("subpixel_instance", instance,
                                               "positive area but no covered pixel centers; omitted.", "warning"))
                continue
            ids.append(instance)
            boxes.append(box)
            mask[top:bottom, left:right][support] = len(ids)
        visible = set(int(i) for i in np.unique(mask))
        for output_id, source_id in enumerate(ids, 1):
            if output_id not in visible:
                diagnostics.append(_diagnostic("fully_occluded_instance", source_id,
                                               "independent pixel support overwritten by higher source IDs.", "warning"))
        return _finish(request, mask, boxes, ids, diagnostics)
    finally:
        dataset = None


def _convert_archive(request, plan):
    np = _numpy()
    target, source = request.target_grid, plan.source.source_grid
    sampled = np.zeros((target.height, target.width), dtype=np.int64)
    with np.load(plan.source.path, allow_pickle=False) as archive:
        source_boxes = archive["bboxes"]
    with _MaskReader(request, plan.source) as reader:
        mapper = _pixel_mapper(request, source)[0] if plan.method == "nearest_warp" else None
        for window in _windows(target.width, target.height):
            col, row, width, height = window
            if plan.method == "nearest_warp":
                rows, cols = _nearest_indices(request, source, window, mapper)
                values = reader.gather(rows, cols)
            else:
                x, y = (0, 0) if plan.source_window is None else plan.source_window[:2]
                values = reader.read(x + col, y + row, width, height)
            sampled[row:row + height, col:col + width] = values
    mapper, nonlinear = _outline_mapper(request, source.crs_wkt, source)
    candidates, diagnostics = {}, []
    for instance, (x, y, width, height) in enumerate(source_boxes, 1):
        prepared = _clip(_pixel_geometry(_rectangle(x, y, x + width, y + height), mapper, nonlinear),
                         target, instance, diagnostics)
        if prepared is not None:
            candidates[instance] = prepared[1]
    # A source pixel can straddle a fractional annotation box. Nearest sampling
    # can therefore carry its ID into a target whose actual annotation outline
    # is fully outside. Geometry exclusion wins; that annotation is dropped.
    eligible = np.zeros(len(source_boxes) + 1, dtype=bool)
    for instance in candidates:
        eligible[instance] = True
    removed = (sampled > 0) & ~eligible[sampled]
    if removed.any():
        diagnostics.append(LabelValidationDiagnostic(
            "excluded_instance_pixels", "Discarded nearest samples belonging to fully excluded annotation boxes.",
            "warning", actual=int(np.count_nonzero(removed))))
        sampled[removed] = 0
    visible = set(int(i) for i in np.unique(sampled) if i > 0)
    ids, boxes = [], []
    for instance, box in candidates.items():
        if instance not in visible:
            left, top, right, bottom = _box_window(box, target)
            if not np.any(sampled[top:bottom, left:right] > 0):
                diagnostics.append(_diagnostic("unsupported_instance", instance,
                                               "clipped box has no remaining pixel/overlap support; omitted.", "warning"))
                continue
            diagnostics.append(_diagnostic("fully_occluded_instance", instance,
                                           "clipped box contains other instance pixels (overlap heuristic).", "warning"))
        ids.append(instance)
        boxes.append(box)
    lookup = np.zeros(len(source_boxes) + 1, dtype=np.int64)
    for output_id, source_id in enumerate(ids, 1):
        lookup[source_id] = output_id
    return _finish(request, lookup[sampled], boxes, ids, diagnostics)


def convert_crater_labels(path, *, target_grid, layer="craters") -> InstanceLabelConversion:
    """Read a finished GeoPackage and return target-sized arrays, without writes."""
    # Label planning uses only target_grid; this internal geographic placeholder
    # is never queried, returned as provenance, or used to acquire imagery.
    request = ChipRequest("label_conversion", target_grid, GeographicAOI(1, 0, 0, 1),
                          "label_conversion", label_input=LabelInput(
                              Path(path), relation="clip_to_target", layer=layer))
    try:
        plan = plan_label_preparation(request, request.label_path)
        if plan.source.kind != "vector_instance":
            raise ValueError("convert_crater_labels requires a GeoPackage.")
        result = _convert_vector(request, plan)
        _verify_source(request, plan)
        return result
    except LabelMismatchError:
        raise
    except Exception as exc:
        raise _label_error(request, code="instance_conversion_failed", message=str(exc)) from exc


def _write_archive(path, result):
    """Fixed member order, timestamps, and NPY version give reproducible bytes."""
    np = _numpy()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, array in result.arrays().items():
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o600 << 16
            with archive.open(info, "w", force_zip64=True) as member:
                np.lib.format.write_array(member, array, version=(1, 0), allow_pickle=False)


def materialize_instance_label(request, plan, *, staging_root=None) -> PreparedLabelArtifact:
    """Reuse exact NPZ bytes or atomically stage a validated derived archive.

    Source masks/boxes and output arrays live only in this worker. The caller
    owns empty staging directories and subsequent publication/retention (A5).
    """
    if not isinstance(request, ChipRequest) or not isinstance(plan, LabelPreparationPlan):
        raise TypeError("Expected ChipRequest and LabelPreparationPlan records.")
    temporary, linked = None, False
    try:
        if plan.source.kind not in ("raster_instance", "vector_instance"):
            raise ValueError("Expected an instance label preparation plan.")
        if request.target_grid != plan.target_grid:
            raise ValueError("Request and plan target grids differ.")
        if request.label_path is not None and request.label_path != plan.source.path:
            raise ValueError("Request and plan source paths differ.")
        if request.label_grid is not None and request.label_grid != plan.source.source_grid:
            raise ValueError("Request and plan source grids differ.")
        if request.label_input is not None:
            source = request.label_input
            if ((source.kind, source.relation, source.layer) !=
                    (plan.source.kind, plan.source.relation, plan.source.layer)
                    or (source.sidecar_path is not None and source.sidecar_path != plan.source.sidecar_path)):
                raise ValueError("Request and plan label contracts differ.")
        _verify_source(request, plan)
        current = plan_label_preparation(replace(request, label_input=plan.source,
                                                label_path=plan.source.path,
                                                label_grid=plan.source.source_grid), plan.source.path)
        if (current.source, current.target_grid, current.method, current.source_sha256, current.source_window) != (
                plan.source, plan.target_grid, plan.method, plan.source_sha256, plan.source_window):
            raise ValueError("Source metadata or preparation plan changed.")
        _verify_source(request, plan)
        if not plan.requires_materialization:
            return PreparedLabelArtifact(plan.source.path, plan, plan.source_sha256, diagnostics=plan.diagnostics)
        destination = _staging_path(request, plan, staging_root)
        result = (_convert_vector(request, plan) if plan.source.kind == "vector_instance"
                  else _convert_archive(request, plan))
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f".{request.sample_id}-", suffix=".npz", dir=destination.parent)
        temporary = Path(name)
        os.close(fd)
        _write_archive(temporary, result)
        np = _numpy()
        with np.load(temporary, allow_pickle=False) as archive:
            _validate_instance_archive(request, archive)
            for key, expected in result.arrays().items():
                actual = archive[key]
                if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
                    raise ValueError(f"Staged {key} differs from conversion result.")
        _verify_source(request, plan)
        artifact = PreparedLabelArtifact(destination, plan, _hash_file(temporary), result.id_mapping,
                                         (*plan.diagnostics, *result.diagnostics))
        os.link(temporary, destination)
        linked = True
        temporary.unlink()
        temporary = None
        return artifact
    except Exception as exc:
        cleanup_errors = []
        try:
            if linked and temporary is not None and destination.exists() and os.path.samefile(temporary, destination):
                destination.unlink()
        except OSError as cleanup_exc:
            cleanup_errors.append(str(cleanup_exc))
        try:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        except OSError as cleanup_exc:
            cleanup_errors.append(str(cleanup_exc))
        failure = exc if isinstance(exc, LabelMismatchError) else _label_error(
            request, code="instance_conversion_failed", message=str(exc))
        failure.label_path = plan.source.path
        if cleanup_errors:
            failure.diagnostics += (LabelValidationDiagnostic("label_staging_cleanup_failed", "; ".join(cleanup_errors)),)
        if failure is exc:
            raise
        raise failure from exc


__all__ = ["InstanceLabelConversion", "convert_crater_labels", "materialize_instance_label"]
