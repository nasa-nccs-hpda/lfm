"""Worker-side semantic label preparation, independent of imagery acquisition.

The artifact carries the target grid and source provenance; NPY stores only
the integer mask. Orchestration, retention, and dataset publication are A5.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
import tempfile

from .chip_label_planning import _hash_file, _pixel_mapper, plan_label_preparation
from .chip_labels import _crs_is_same, _label_error, _numpy, _validate_mask
from .chip_types import (
    ChipRequest, LabelInput, LabelMismatchError, LabelPreparationPlan,
    LabelValidationDiagnostic, PreparedLabelArtifact,
)


_BLOCK_SIZE = 256


def _windows(width, height):
    for row in range(0, height, _BLOCK_SIZE):
        for col in range(0, width, _BLOCK_SIZE):
            yield col, row, min(_BLOCK_SIZE, width - col), min(_BLOCK_SIZE, height - row)


def _close_array(array):
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


class _MaskReader:
    """Read native integer values; shared NPZ-mask access is also useful to A4.

    NPY uses a read-only mapping. NPZ must decompress its mask in memory.
    GeoTIFF reads windows; gathering for warps reads bounded source blocks.
    """

    def __init__(self, request: ChipRequest, source: LabelInput):
        self.request, self.source = request, source
        self.array = self.dataset = self.band = None

    def __enter__(self):
        np = _numpy()
        try:
            suffix = self.source.path.suffix.lower()
            if suffix == ".npy":
                self.array = np.load(self.source.path, mmap_mode="r", allow_pickle=False)
            elif suffix == ".npz":
                with np.load(self.source.path, allow_pickle=False) as archive:
                    self.array = archive["mask"]
            elif suffix in (".tif", ".tiff"):
                from osgeo import gdal

                self.dataset = gdal.OpenEx(str(self.source.path), gdal.OF_RASTER | gdal.OF_READONLY)
                if self.dataset is None or self.dataset.RasterCount != 1:
                    raise ValueError("Could not open a single-band source label.")
                self.band = self.dataset.GetRasterBand(1)
                sample = self.band.ReadAsArray(0, 0, 1, 1)
                if sample is None:
                    raise ValueError("Could not read source label pixels.")
                self.dtype = sample.dtype
            else:
                raise ValueError(f"Unsupported mask format: {suffix}")
            if self.array is not None:
                source_request = replace(self.request, target_grid=self.source.source_grid)
                _validate_mask(source_request, self.array)
                self.dtype = self.array.dtype
            if not np.issubdtype(self.dtype, np.integer):
                raise _label_error(self.request, code="invalid_label_dtype",
                                   message="Label materialization requires integer class IDs.")
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        _close_array(self.array)
        self.array = self.band = self.dataset = None

    def read(self, col, row, width, height):
        grid = self.source.source_grid
        if col < 0 or row < 0 or col + width > grid.width or row + height > grid.height:
            raise _label_error(self.request, code="incomplete_label_coverage",
                               message="Label read window extends beyond its source grid.")
        values = (self.array[row:row + height, col:col + width] if self.array is not None
                  else self.band.ReadAsArray(col, row, width, height))
        if values is None or values.shape != (height, width) or values.dtype != self.dtype:
            raise _label_error(self.request, code="invalid_label_read",
                               message="Source label read returned unexpected shape or dtype.")
        return values

    def gather(self, rows, cols):
        np = _numpy()
        if self.array is not None:
            return self.array[rows, cols]
        result = np.empty(rows.shape, dtype=self.dtype)
        block_rows, block_cols = rows // _BLOCK_SIZE, cols // _BLOCK_SIZE
        blocks = np.unique(np.column_stack((block_rows.ravel(), block_cols.ravel())), axis=0)
        grid = self.source.source_grid
        for block_row, block_col in blocks:
            row, col = int(block_row) * _BLOCK_SIZE, int(block_col) * _BLOCK_SIZE
            selected = (block_rows == block_row) & (block_cols == block_col)
            values = self.read(col, row, min(_BLOCK_SIZE, grid.width - col),
                               min(_BLOCK_SIZE, grid.height - row))
            result[selected] = values[rows[selected] - row, cols[selected] - col]
        return result


def _nearest_indices(request, source_grid, window, mapper):
    """Inverse-map target centers, then select the containing source pixel.

    Only coordinates are floating-point. Values are gathered in their original
    integer dtype, including uint64/int64 IDs beyond float64's exact range.
    """
    np = _numpy()
    col, row, width, height = window
    if mapper is None:
        columns, rows = np.meshgrid(np.arange(col, col + width, dtype=float) + 0.5,
                                    np.arange(row, row + height, dtype=float) + 0.5)
        x0, a, b, y0, c, d = request.target_grid.transform
        x, y = x0 + a * columns + b * rows, y0 + c * columns + d * rows
        sx0, sa, sb, sy0, sc, sd = source_grid.transform
        determinant = sa * sd - sb * sc
        dx, dy = x - sx0, y - sy0
        source_cols = (sd * dx - sb * dy) / determinant
        source_rows = (-sc * dx + sa * dy) / determinant
    else:
        coordinates = np.asarray([
            mapper(col + x + 0.5, row + y + 0.5)
            for y in range(height) for x in range(width)
        ]).reshape(height, width, 2)
        source_cols, source_rows = coordinates[..., 0], coordinates[..., 1]
    # Resolve numerical noise at nearest-neighbor ties consistently: an exact
    # source pixel boundary belongs to the pixel on its right/bottom side.
    source_cols = np.where(np.abs(source_cols - np.rint(source_cols)) <= 1e-8,
                           np.rint(source_cols), source_cols)
    source_rows = np.where(np.abs(source_rows - np.rint(source_rows)) <= 1e-8,
                           np.rint(source_rows), source_rows)
    if (not np.isfinite(source_cols).all() or not np.isfinite(source_rows).all()
            or (source_cols < 0).any() or (source_cols >= source_grid.width).any()
            or (source_rows < 0).any() or (source_rows >= source_grid.height).any()):
        raise _label_error(request, code="incomplete_label_coverage",
                           message="A target pixel center has no source label sample; background padding is forbidden.")
    return np.floor(source_rows).astype(np.int64), np.floor(source_cols).astype(np.int64)


def _verify_source(request, plan):
    try:
        actual = _hash_file(plan.source.path)
    except OSError as exc:
        raise _label_error(request, code="label_source_changed",
                           message=f"Validated source label is no longer readable: {exc}") from exc
    if actual != plan.source_sha256:
        raise _label_error(request, code="label_source_changed",
                           message="Source label changed since its preparation plan was validated.")


def _validate_staged(request, path, dtype, expected_content_hash):
    np = _numpy()
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        _validate_mask(request, array)
        if array.dtype != dtype:
            raise ValueError("Staged label dtype differs from the source dtype.")
        digest = hashlib.sha256()
        for col, row, width, height in _windows(request.target_grid.width, request.target_grid.height):
            digest.update(np.ascontiguousarray(array[row:row + height, col:col + width]).tobytes())
        if digest.hexdigest() != expected_content_hash:
            raise ValueError("Staged label pixels differ from the materialized values.")
    finally:
        _close_array(array)


def _staging_path(request, plan, staging_root):
    if staging_root is None:
        raise _label_error(request, code="missing_label_staging_root",
                           message="Derived labels require an explicit staging_root.")
    root = Path(staging_root).resolve()
    sample_root = root / request.sample_id
    directory = sample_root / "labels"
    destination = directory / f"{request.sample_id}_label{plan.output_suffix}"
    for path in (sample_root, directory, destination):
        if path.is_symlink() or path.resolve() != path:
            raise _label_error(request, code="unsafe_label_staging_path",
                               message="Sample label staging paths must not traverse symlinks.")
    for source in (plan.source.path, plan.source.sidecar_path):
        if source is not None and (
            source.resolve().is_relative_to(sample_root) or source.absolute().is_relative_to(sample_root)
        ):
            raise _label_error(request, code="unsafe_label_staging_path",
                               message="Sample staging must not contain source labels or their sidecars.")
    if destination.exists():
        raise _label_error(request, code="label_artifact_exists",
                           message=f"Refusing to overwrite an existing staged label: {destination}")
    return destination


def materialize_semantic_label(
    request: ChipRequest,
    plan: LabelPreparationPlan,
    *,
    staging_root: str | Path | None = None,
) -> PreparedLabelArtifact:
    """Reuse an exact NPY or stage a verified semantic NPY without publication.

    Derived artifacts use ``<staging_root>/<sample_id>/labels/<sample_id>_label.npy``.
    Existing artifacts are never overwritten. Invalid plans fail before writes;
    runtime failure removes this call's temporary file (empty directories may
    remain for the later retention policy). Instance conversion belongs to A4.
    """
    if not isinstance(request, ChipRequest) or not isinstance(plan, LabelPreparationPlan):
        raise TypeError("Expected ChipRequest and LabelPreparationPlan records.")
    temporary = None
    linked = False
    output = None
    try:
        if plan.source.kind != "semantic":
            raise _label_error(request, code="unsupported_label_materialization",
                               message="Instance labels require joint mask/box/count preparation in A4.")
        if request.target_grid != plan.target_grid:
            raise _label_error(request, code="label_plan_mismatch", message="Request and label-plan target grids differ.")
        if request.label_path is not None and request.label_path != plan.source.path:
            raise _label_error(request, code="label_plan_mismatch", message="Request and label-plan source paths differ.")
        if request.label_grid is not None and request.label_grid != plan.source.source_grid:
            raise _label_error(request, code="label_plan_mismatch", message="Request and label-plan source grids differ.")
        if request.label_input is not None and (
            request.label_input.kind != plan.source.kind or request.label_input.relation != plan.source.relation
        ):
            raise _label_error(request, code="label_plan_mismatch", message="Request and label-plan kind/relation differ.")
        if (request.label_input is not None and request.label_input.sidecar_path is not None
                and request.label_input.sidecar_path != plan.source.sidecar_path):
            raise _label_error(request, code="label_plan_mismatch", message="Request and label-plan sidecars differ.")
        _verify_source(request, plan)
        source_request = replace(request, label_input=plan.source, label_path=plan.source.path,
                                 label_grid=plan.source.source_grid)
        current = plan_label_preparation(source_request, plan.source.path)
        if (current.method, current.source_window, current.source.source_grid) != (
            plan.method, plan.source_window, plan.source.source_grid
        ):
            raise _label_error(request, code="label_plan_mismatch",
                               message="Source grid/relation changed or the supplied plan is inconsistent.")
        _verify_source(request, plan)
        if not plan.requires_materialization:
            return PreparedLabelArtifact(plan.source.path, plan, plan.source_sha256,
                                         diagnostics=plan.diagnostics)

        destination = _staging_path(request, plan, staging_root)
        np = _numpy()
        with _MaskReader(request, plan.source) as reader:
            mapper = None
            if plan.method == "nearest_warp":
                # OSR handles CRS changes and longitude branch normalization.
                # Projected, equal-CRS affines can use vectorized pixel mapping.
                from .chip_requests import _spatial_reference

                same = _crs_is_same(plan.source.source_grid.crs_wkt, request.target_grid.crs_wkt)
                if not same or _spatial_reference(plan.source.source_grid.crs_wkt).IsGeographic():
                    mapper, _ = _pixel_mapper(request, plan.source.source_grid)
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=f".{request.sample_id}-", suffix=".npy", dir=destination.parent)
            temporary = Path(name)
            os.close(fd)
            target = request.target_grid
            output = np.lib.format.open_memmap(temporary, mode="w+", dtype=reader.dtype,
                                               shape=(target.height, target.width), version=(1, 0))
            content_hash = hashlib.sha256()
            for window in _windows(target.width, target.height):
                col, row, width, height = window
                if plan.method in ("exact", "aligned_window"):
                    x, y = (0, 0) if plan.source_window is None else plan.source_window[:2]
                    values = reader.read(x + col, y + row, width, height)
                else:
                    rows, cols = _nearest_indices(request, plan.source.source_grid, window, mapper)
                    values = reader.gather(rows, cols)
                if values.dtype != reader.dtype:
                    raise ValueError("Label resampling changed the integer dtype.")
                output[row:row + height, col:col + width] = values
                content_hash.update(np.ascontiguousarray(values).tobytes())
            output.flush()
            _close_array(output)
            output = None
            _validate_staged(request, temporary, reader.dtype, content_hash.hexdigest())
        _verify_source(request, plan)
        digest = _hash_file(temporary)
        diagnostic = LabelValidationDiagnostic("semantic_label_materialized",
                                               f"Validated {plan.method} label on the exact target grid.", "info")
        artifact = PreparedLabelArtifact(destination, plan, digest,
                                         diagnostics=(*plan.diagnostics, diagnostic))
        # Atomic no-clobber install: another worker's artifact is never replaced.
        os.link(temporary, destination)
        linked = True
        temporary.unlink()
        temporary = None
        return artifact
    except Exception as exc:
        cleanup_errors = []
        try:
            if output is not None:
                _close_array(output)
        except Exception as cleanup_exc:
            cleanup_errors.append(str(cleanup_exc))
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
            request, code="label_materialization_failed",
            message=f"Could not materialize semantic label: {exc}",
        )
        failure.label_path = plan.source.path
        if cleanup_errors:
            failure.diagnostics += (LabelValidationDiagnostic(
                "label_staging_cleanup_failed", "; ".join(cleanup_errors),
                actual=str(temporary),
            ),)
        if failure is exc:
            raise
        raise failure from exc


__all__ = ["materialize_semantic_label"]
