"""Assemble reprojected modalities and write verified model-ready chips."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import math
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from .chip_config import ChipConfig, OutputModalityConfig
from .chip_reprojection import ChipReprojectionResult, ReprojectedModality
from .chip_requests import raster_bounds, validate_target_grid_consistency
from .chip_types import TargetGrid


_SAFE_SAMPLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class ChipAssemblyError(RuntimeError):
    """A reprojected sample could not satisfy the final chip contract."""

    def __init__(
        self,
        message: str,
        *,
        sample_id: str,
        code: str = "assembly_error",
        modality_alias: str | None = None,
        band_name: str | None = None,
    ) -> None:
        super().__init__(message)
        self.sample_id = sample_id
        self.code = code
        self.modality_alias = modality_alias
        self.band_name = band_name


@dataclass(frozen=True)
class AssembledChip:
    """Ordered, target-grid-aligned bands ready for deterministic encoding."""

    reprojection: ChipReprojectionResult
    config: ChipConfig
    target_grid: TargetGrid
    band_names: tuple[str, ...]
    band_origins: tuple[tuple[str, str], ...]
    required_bands: tuple[bool, ...]
    pixels: Any
    valid_mask: Any
    common_nodata: float

    @property
    def sample_id(self) -> str:
        """Return the request's validated dataset-compatible identifier."""
        return self.reprojection.acquisition.prepared_request.request.sample_id


@dataclass(frozen=True)
class ChipWriteValidation:
    """Facts verified by reopening one staged model-ready GeoTIFF."""

    path: Path
    band_count: int
    shape: tuple[int, int]
    dtype: str
    band_names: tuple[str, ...]
    valid_pixel_counts: tuple[int, ...]
    compression: str


@dataclass(frozen=True)
class WrittenChip:
    """One atomically staged chip and its successful reopen validation."""

    assembled: AssembledChip
    path: Path
    validation: ChipWriteValidation


@dataclass(frozen=True)
class _SelectedModality:
    alias: str
    source_name: str
    names: tuple[str, ...]
    names_are_explicit: bool
    required: bool
    pixels: Any
    valid_mask: Any


def _libraries():
    try:
        import numpy as np
        from osgeo import gdal, osr
    except ImportError as exc:
        raise RuntimeError(
            "Chip assembly requires NumPy and the GDAL Python bindings."
        ) from exc
    gdal.UseExceptions()
    return np, gdal, osr


def _error(
    reprojection: ChipReprojectionResult,
    message: str,
    *,
    code: str,
    modality_alias: str | None = None,
    band_name: str | None = None,
) -> ChipAssemblyError:
    return ChipAssemblyError(
        message,
        sample_id=(
            reprojection.acquisition.prepared_request.request.sample_id
        ),
        code=code,
        modality_alias=modality_alias,
        band_name=band_name,
    )


def staged_chip_path(config: ChipConfig, sample_id: str) -> Path:
    """Return C5's deterministic pre-publication chip path."""
    if not isinstance(config, ChipConfig):
        raise TypeError("config must be a ChipConfig.")
    request_ids = str(sample_id).strip()
    if not request_ids:
        raise ValueError("sample_id must not be empty.")
    # ChipRequest has already constrained IDs to safe filename characters. Keep
    # this public helper from weakening that contract when called independently.
    if not _SAFE_SAMPLE_ID.fullmatch(request_ids):
        raise ValueError("sample_id is not a dataset-compatible sample ID.")
    return (
        config.intermediate_root
        / request_ids
        / "assembled"
        / f"{request_ids}{config.final_output_suffix}"
    )


def _available_missing_names(
    result: ReprojectedModality,
    modality: OutputModalityConfig,
    source: Any,
) -> tuple[str, ...]:
    if result.band_names:
        return result.band_names
    if source.band_names:
        return tuple(source.band_names)
    if source.band_indices:
        return tuple(f"band_{index}" for index in source.band_indices)
    if modality.band_names:
        return modality.band_names
    if modality.band_indices:
        return tuple(
            f"band_{index}" for index in range(1, max(modality.band_indices) + 1)
        )
    if modality.output_band_names:
        return tuple(
            f"band_{index}"
            for index in range(1, len(modality.output_band_names) + 1)
        )
    return ()


def _selection_indices(
    reprojection: ChipReprojectionResult,
    modality: OutputModalityConfig,
    available_names: tuple[str, ...],
) -> tuple[int, ...]:
    if modality.band_names is not None:
        folded: dict[str, list[int]] = {}
        for index, name in enumerate(available_names):
            folded.setdefault(name.casefold(), []).append(index)
        selected: list[int] = []
        for requested in modality.band_names:
            matches = folded.get(requested.casefold(), [])
            if not matches:
                raise _error(
                    reprojection,
                    f"Modality {modality.alias!r} has no band {requested!r}; "
                    f"available bands are {available_names!r}.",
                    code="missing_selected_band",
                    modality_alias=modality.alias,
                    band_name=requested,
                )
            if len(matches) != 1:
                raise _error(
                    reprojection,
                    f"Band {requested!r} is ambiguous in modality "
                    f"{modality.alias!r}.",
                    code="ambiguous_selected_band",
                    modality_alias=modality.alias,
                    band_name=requested,
                )
            selected.append(matches[0])
        return tuple(selected)
    if modality.band_indices is not None:
        if max(modality.band_indices) > len(available_names):
            raise _error(
                reprojection,
                f"Modality {modality.alias!r} selects band indices "
                f"{modality.band_indices!r} from only {len(available_names)} bands.",
                code="selected_band_index_out_of_range",
                modality_alias=modality.alias,
            )
        return tuple(index - 1 for index in modality.band_indices)
    return tuple(range(len(available_names)))


def _select_modality(
    reprojection: ChipReprojectionResult,
    result: ReprojectedModality,
    modality: OutputModalityConfig,
    config: ChipConfig,
    np,
) -> _SelectedModality:
    if (
        result.acquisition_group != modality.acquisition_group
        or result.source_name != modality.source_name
        or result.alias != modality.alias
    ):
        raise _error(
            reprojection,
            "Reprojected modality order or identity does not match ChipConfig.",
            code="modality_identity_mismatch",
            modality_alias=modality.alias,
        )
    if result.target_grid != reprojection.target_grid:
        raise _error(
            reprojection,
            f"Modality {modality.alias!r} is not on the authoritative target grid.",
            code="modality_target_grid_mismatch",
            modality_alias=modality.alias,
        )
    source = config.acquisition_group(
        modality.acquisition_group
    ).tile_config.source(modality.source_name)

    if result.status == "missing_optional":
        if source.required:
            raise _error(
                reprojection,
                f"Required modality {modality.alias!r} is missing.",
                code="missing_required_modality",
                modality_alias=modality.alias,
            )
        available = _available_missing_names(result, modality, source)
        if not available:
            raise _error(
                reprojection,
                f"Missing optional modality {modality.alias!r} has no "
                "configured band contract for placeholder channels.",
                code="unknown_optional_band_contract",
                modality_alias=modality.alias,
            )
        indices = _selection_indices(reprojection, modality, available)
        selected_names = tuple(available[index] for index in indices)
        shape = (
            len(indices),
            reprojection.target_grid.height,
            reprojection.target_grid.width,
        )
        pixels = np.full(shape, config.common_nodata, dtype=np.float64)
        mask = np.zeros(shape, dtype=bool)
    elif result.status == "complete":
        available = tuple(result.band_names)
        if not available:
            raise _error(
                reprojection,
                f"Complete modality {modality.alias!r} has no band metadata.",
                code="empty_modality_bands",
                modality_alias=modality.alias,
            )
        pixels = np.asarray(result.pixels)
        mask = np.asarray(result.valid_mask, dtype=bool)
        expected_shape = (
            len(available),
            reprojection.target_grid.height,
            reprojection.target_grid.width,
        )
        if pixels.shape != expected_shape or mask.shape != expected_shape:
            raise _error(
                reprojection,
                f"Modality {modality.alias!r} arrays do not match "
                f"{expected_shape}.",
                code="modality_array_shape_mismatch",
                modality_alias=modality.alias,
            )
        indices = _selection_indices(reprojection, modality, available)
        selected_names = tuple(available[index] for index in indices)
        pixels = pixels[np.asarray(indices), :, :]
        mask = mask[np.asarray(indices), :, :]
    else:
        raise _error(
            reprojection,
            f"Unknown modality status {result.status!r}.",
            code="unknown_modality_status",
            modality_alias=modality.alias,
        )

    names = modality.output_band_names or selected_names
    if len(names) != len(selected_names):
        raise _error(
            reprojection,
            f"Modality {modality.alias!r} has {len(selected_names)} selected "
            f"bands but {len(names)} output names.",
            code="output_band_name_count_mismatch",
            modality_alias=modality.alias,
        )
    return _SelectedModality(
        alias=modality.alias,
        source_name=modality.source_name,
        names=tuple(names),
        names_are_explicit=modality.output_band_names is not None,
        required=source.required,
        pixels=pixels,
        valid_mask=mask,
    )


def _resolve_output_names(
    reprojection: ChipReprojectionResult,
    selected: tuple[_SelectedModality, ...],
) -> tuple[str, ...]:
    provisional = tuple(
        (item.alias, name, item.names_are_explicit)
        for item in selected
        for name in item.names
    )
    counts = Counter(name.casefold() for _, name, _ in provisional)
    names = tuple(
        (
            f"{alias}_{name}"
            if not explicit and counts[name.casefold()] > 1
            else name
        )
        for alias, name, explicit in provisional
    )
    folded = [name.casefold() for name in names]
    if len(set(folded)) != len(folded):
        duplicates = sorted(
            name for name, count in Counter(folded).items() if count > 1
        )
        raise _error(
            reprojection,
            "Final band names remain ambiguous after modality qualification: "
            + ", ".join(duplicates)
            + ". Configure unique output_band_names.",
            code="duplicate_output_band_names",
        )
    return names


def assemble_chip(
    reprojection: ChipReprojectionResult,
    config: ChipConfig,
) -> AssembledChip:
    """Select and concatenate modality bands in ``ChipConfig`` order."""
    if not isinstance(reprojection, ChipReprojectionResult):
        raise TypeError("reprojection must be a ChipReprojectionResult.")
    if not isinstance(config, ChipConfig):
        raise TypeError("config must be a ChipConfig.")
    prepared = reprojection.acquisition.prepared_request
    if not prepared.eligible_for_acquisition:
        raise _error(
            reprojection,
            "Only requests with a passed label preflight and assigned split may "
            "be assembled.",
            code="ineligible_preflight",
        )
    if prepared.preflight.resolved_label_path is None:
        raise _error(
            reprojection,
            "A passed label preflight must retain its resolved label path.",
            code="missing_preflight_label",
        )
    if reprojection.acquisition.status != "complete":
        raise _error(
            reprojection,
            "Only complete acquisitions may be assembled.",
            code="incomplete_acquisition",
        )
    if reprojection.target_grid != prepared.request.target_grid:
        raise _error(
            reprojection,
            "Reprojection target does not match the request target grid.",
            code="target_grid_mismatch",
        )
    validate_target_grid_consistency(reprojection.target_grid)
    if len(reprojection.modalities) != len(config.output_modalities):
        raise _error(
            reprojection,
            "Reprojection does not contain every configured output modality.",
            code="modality_count_mismatch",
        )

    np, _, _ = _libraries()
    selected = tuple(
        _select_modality(reprojection, result, modality, config, np)
        for result, modality in zip(
            reprojection.modalities,
            config.output_modalities,
            strict=True,
        )
    )
    if not selected or not any(item.names for item in selected):
        raise _error(
            reprojection,
            "At least one output band is required.",
            code="empty_output_bands",
        )
    band_names = _resolve_output_names(reprojection, selected)
    pixels = np.concatenate(tuple(item.pixels for item in selected), axis=0)
    valid_mask = np.concatenate(
        tuple(item.valid_mask for item in selected), axis=0
    ).astype(bool, copy=False)
    expected_shape = (
        len(band_names),
        reprojection.target_grid.height,
        reprojection.target_grid.width,
    )
    if pixels.shape != expected_shape or valid_mask.shape != expected_shape:
        raise _error(
            reprojection,
            f"Assembled arrays do not match {expected_shape}.",
            code="assembled_shape_mismatch",
        )
    return AssembledChip(
        reprojection=reprojection,
        config=config,
        target_grid=reprojection.target_grid,
        band_names=band_names,
        band_origins=tuple(
            (item.alias, item.source_name)
            for item in selected
            for _ in item.names
        ),
        required_bands=tuple(
            item.required for item in selected for _ in item.names
        ),
        pixels=pixels,
        valid_mask=valid_mask,
        common_nodata=float(config.common_nodata),
    )


def _cast_pixels(assembled: AssembledChip, dtype_name: str, np):
    dtype = np.dtype(dtype_name)
    pixels = np.asarray(assembled.pixels)
    mask = np.asarray(assembled.valid_mask, dtype=bool)
    valid = pixels[mask]
    if valid.size and not np.isfinite(valid).all():
        raise _error(
            assembled.reprojection,
            "Valid output pixels must all be finite.",
            code="nonfinite_valid_pixels",
        )

    nodata = assembled.common_nodata
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        if not float(nodata).is_integer() or not limits.min <= nodata <= limits.max:
            raise _error(
                assembled.reprojection,
                f"NoData {nodata!r} is not representable by {dtype_name}.",
                code="nodata_not_representable",
            )
        if valid.size and (
            np.any(valid < limits.min)
            or np.any(valid > limits.max)
            or not np.equal(valid, np.rint(valid)).all()
        ):
            raise _error(
                assembled.reprojection,
                f"Valid pixels cannot be losslessly represented by {dtype_name}.",
                code="integer_cast_not_lossless",
            )
    else:
        limits = np.finfo(dtype)
        if abs(nodata) > limits.max:
            raise _error(
                assembled.reprojection,
                f"NoData {nodata!r} is outside the {dtype_name} range.",
                code="nodata_not_representable",
            )
        if valid.size and np.any(np.abs(valid) > limits.max):
            raise _error(
                assembled.reprojection,
                f"Valid pixels exceed the {dtype_name} range.",
                code="floating_cast_out_of_range",
            )

    output = np.full(pixels.shape, nodata, dtype=dtype)
    output[mask] = valid.astype(dtype, copy=False)
    typed_nodata = np.asarray(nodata, dtype=dtype).item()
    if float(typed_nodata) != nodata:
        raise _error(
            assembled.reprojection,
            f"NoData {nodata!r} cannot be represented exactly by {dtype_name}.",
            code="nodata_not_exactly_representable",
        )
    if mask.any() and np.equal(output[mask], typed_nodata).any():
        raise _error(
            assembled.reprojection,
            "A valid pixel becomes indistinguishable from the output NoData value.",
            code="valid_nodata_collision",
        )
    return output, typed_nodata


def _spatial_reference(wkt: str, osr):
    spatial_reference = osr.SpatialReference()
    if spatial_reference.ImportFromWkt(wkt) != 0:
        raise ValueError("Invalid CRS WKT.")
    spatial_reference.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return spatial_reference


def _same_crs(first: str, second: str, osr) -> bool:
    try:
        return bool(
            _spatial_reference(first, osr).IsSame(
                _spatial_reference(second, osr)
            )
        )
    except (RuntimeError, ValueError):
        return False


def _gdal_dtype(dtype_name: str, gdal) -> int:
    return {
        "uint8": gdal.GDT_Byte,
        "uint16": gdal.GDT_UInt16,
        "int16": gdal.GDT_Int16,
        "uint32": gdal.GDT_UInt32,
        "int32": gdal.GDT_Int32,
        "float32": gdal.GDT_Float32,
        "float64": gdal.GDT_Float64,
    }[dtype_name]


def validate_written_chip(
    path: Path,
    assembled: AssembledChip,
    *,
    dtype_name: str,
) -> ChipWriteValidation:
    """Reopen a staged chip and verify its complete raster contract."""
    np, gdal, osr = _libraries()
    path = Path(path)
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None:
        raise _error(
            assembled.reprojection,
            f"Could not reopen staged chip {path}.",
            code="chip_reopen_failed",
        )
    try:
        target = assembled.target_grid
        if (
            dataset.RasterCount != len(assembled.band_names)
            or dataset.RasterXSize != target.width
            or dataset.RasterYSize != target.height
        ):
            raise _error(
                assembled.reprojection,
                "Written chip dimensions or channel count do not match assembly.",
                code="written_shape_mismatch",
            )
        transform = tuple(float(value) for value in dataset.GetGeoTransform())
        scale = max(*(abs(value) for value in (*transform, *target.transform)), 1.0)
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=scale * 1e-10)
            for actual, expected in zip(transform, target.transform, strict=True)
        ) or any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=scale * 1e-10)
            for actual, expected in zip(
                raster_bounds(transform, target.width, target.height),
                target.bounds,
                strict=True,
            )
        ):
            raise _error(
                assembled.reprojection,
                "Written chip transform or extent does not match the target grid.",
                code="written_grid_mismatch",
            )
        if not _same_crs(dataset.GetProjectionRef(), target.crs_wkt, osr):
            raise _error(
                assembled.reprojection,
                "Written chip CRS does not match the target grid.",
                code="written_crs_mismatch",
            )
        compression = (
            dataset.GetMetadataItem("COMPRESSION", "IMAGE_STRUCTURE") or ""
        ).upper()
        if compression != "LZW":
            raise _error(
                assembled.reprojection,
                f"Written chip compression is {compression!r}, expected 'LZW'.",
                code="written_compression_mismatch",
            )

        expected_mask = np.asarray(assembled.valid_mask, dtype=bool)
        names: list[str] = []
        valid_counts: list[int] = []
        expected_gdal_dtype = _gdal_dtype(dtype_name, gdal)
        for index, expected_name in enumerate(assembled.band_names, start=1):
            band = dataset.GetRasterBand(index)
            if band.DataType != expected_gdal_dtype:
                raise _error(
                    assembled.reprojection,
                    f"Band {expected_name!r} has the wrong dtype.",
                    code="written_dtype_mismatch",
                    band_name=expected_name,
                )
            nodata = band.GetNoDataValue()
            if nodata is None or not math.isclose(
                float(nodata),
                assembled.common_nodata,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise _error(
                    assembled.reprojection,
                    f"Band {expected_name!r} does not expose common NoData "
                    f"{assembled.common_nodata!r}.",
                    code="written_nodata_mismatch",
                    band_name=expected_name,
                )
            name = band.GetMetadataItem("Name") or band.GetDescription() or ""
            if name != expected_name or band.GetDescription() != expected_name:
                raise _error(
                    assembled.reprojection,
                    f"Band {index} metadata does not match {expected_name!r}.",
                    code="written_band_name_mismatch",
                    band_name=expected_name,
                )
            actual_mask = np.asarray(band.GetMaskBand().ReadAsArray()) != 0
            band_expected_mask = expected_mask[index - 1]
            if not np.array_equal(actual_mask, band_expected_mask):
                raise _error(
                    assembled.reprojection,
                    f"Band {expected_name!r} validity mask changed on disk.",
                    code="written_mask_mismatch",
                    band_name=expected_name,
                )
            values = np.asarray(band.ReadAsArray())
            if actual_mask.any() and not np.isfinite(values[actual_mask]).all():
                raise _error(
                    assembled.reprojection,
                    f"Band {expected_name!r} has nonfinite valid pixels.",
                    code="written_nonfinite_pixels",
                    band_name=expected_name,
                )
            valid_count = int(actual_mask.sum())
            if assembled.required_bands[index - 1] and valid_count == 0:
                raise _error(
                    assembled.reprojection,
                    f"Required band {expected_name!r} has no finite coverage.",
                    code="empty_required_band",
                    band_name=expected_name,
                )
            names.append(name)
            valid_counts.append(valid_count)
        return ChipWriteValidation(
            path=path,
            band_count=dataset.RasterCount,
            shape=(dataset.RasterYSize, dataset.RasterXSize),
            dtype=dtype_name,
            band_names=tuple(names),
            valid_pixel_counts=tuple(valid_counts),
            compression=compression,
        )
    finally:
        dataset = None


def write_model_ready_chip(
    assembled: AssembledChip,
    config: ChipConfig,
    *,
    output_path: Path | None = None,
) -> WrittenChip:
    """Atomically stage a compressed chip and validate it by reopening."""
    if not isinstance(assembled, AssembledChip):
        raise TypeError("assembled must be an AssembledChip.")
    if not isinstance(config, ChipConfig):
        raise TypeError("config must be a ChipConfig.")
    if assembled.config != config:
        raise _error(
            assembled.reprojection,
            "Assembly and write configurations differ.",
            code="assembly_config_mismatch",
        )
    if assembled.common_nodata != config.common_nodata:
        raise _error(
            assembled.reprojection,
            "Assembly NoData does not match ChipConfig.",
            code="assembly_config_nodata_mismatch",
        )
    path = (
        staged_chip_path(config, assembled.sample_id)
        if output_path is None
        else Path(output_path)
    )
    if path.exists() or path.is_symlink():
        raise _error(
            assembled.reprojection,
            f"Refusing to overwrite existing staged chip {path}.",
            code="output_exists",
        )

    np, gdal, _ = _libraries()
    output, typed_nodata = _cast_pixels(assembled, config.output_dtype, np)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid4().hex}.tmp.tif")
    dataset = None
    try:
        dataset = gdal.GetDriverByName("GTiff").Create(
            str(temporary),
            assembled.target_grid.width,
            assembled.target_grid.height,
            len(assembled.band_names),
            _gdal_dtype(config.output_dtype, gdal),
            options=[
                "TILED=YES",
                "COMPRESS=LZW",
                "BIGTIFF=IF_SAFER",
            ],
        )
        if dataset is None:
            raise _error(
                assembled.reprojection,
                f"Could not create staged chip {temporary}.",
                code="chip_create_failed",
            )
        dataset.SetProjection(assembled.target_grid.crs_wkt)
        dataset.SetGeoTransform(assembled.target_grid.transform)
        for index, name in enumerate(assembled.band_names, start=1):
            band = dataset.GetRasterBand(index)
            band.SetNoDataValue(float(typed_nodata))
            band.SetDescription(name)
            band.SetMetadataItem("Name", name)
            band.WriteArray(output[index - 1])
        dataset.FlushCache()
        dataset = None
        validation = validate_written_chip(
            temporary,
            assembled,
            dtype_name=config.output_dtype,
        )
        temporary.replace(path)
        validation = replace(validation, path=path)
        return WrittenChip(assembled=assembled, path=path, validation=validation)
    except Exception:
        dataset = None
        temporary.unlink(missing_ok=True)
        raise


def assemble_and_write_chip(
    reprojection: ChipReprojectionResult,
    config: ChipConfig,
) -> WrittenChip:
    """Run C5 assembly and verified staging for one eligible sample."""
    return write_model_ready_chip(assemble_chip(reprojection, config), config)


__all__ = [
    "AssembledChip",
    "ChipAssemblyError",
    "ChipWriteValidation",
    "WrittenChip",
    "assemble_and_write_chip",
    "assemble_chip",
    "staged_chip_path",
    "validate_written_chip",
    "write_model_ready_chip",
]
