"""Structured, target-sample-isolated acquisition through the public tiler."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from .chip_config import AcquisitionGroupConfig, ChipConfig
from .chip_preflight import PreparedChipRequest
from .chip_requests import geographic_query_parts, product_id_from_sample_id
from .chip_types import ChipRequest, GeographicAOI, SourceSelector
from .tiling import create_tiles_for_aoi
from .tiling_results import (
    MissingRequiredSourceError,
    TileCubeRecord,
    TileSourceError,
)


AcquisitionStatus = Literal["complete", "failed"]
AcquisitionSeverity = Literal["info", "warning", "error"]


class SelectorResolutionError(ValueError):
    """A request cannot supply the selectors required by its source configs."""

    def __init__(
        self,
        message: str,
        *,
        sample_id: str,
        acquisition_group: str | None = None,
        source_name: str | None = None,
    ) -> None:
        super().__init__(message)
        self.sample_id = sample_id
        self.acquisition_group = acquisition_group
        self.source_name = source_name


@dataclass(frozen=True, order=True)
class CubeRecordKey:
    """Structured cube identity qualified by its acquisition group."""

    acquisition_group: str
    source_name: str
    zone: str
    zoom_level: int
    tile_x: int
    tile_y: int


@dataclass(frozen=True)
class CubeRecordGroup:
    """Records sharing one structured group/source/tile identity."""

    key: CubeRecordKey
    records: tuple[TileCubeRecord, ...]


@dataclass(frozen=True)
class AcquisitionDiagnostic:
    """One reproducible acquisition observation or failure."""

    code: str
    message: str
    severity: AcquisitionSeverity
    acquisition_group: str | None = None
    source_name: str | None = None
    zone: str | None = None
    zoom_level: int | None = None
    tile_x: int | None = None
    tile_y: int | None = None


@dataclass(frozen=True)
class AcquisitionGroupResult:
    """Structured outcome from one sample/acquisition-group directory."""

    sample_id: str
    acquisition_group: str
    zoom_level: int
    output_dir: Path
    logical_aoi: GeographicAOI
    query_parts: tuple[GeographicAOI, ...]
    selectors: tuple[SourceSelector, ...]
    status: AcquisitionStatus
    records: tuple[TileCubeRecord, ...] = ()
    record_groups: tuple[CubeRecordGroup, ...] = ()
    inventory_paths: tuple[Path, ...] = ()
    diagnostics: tuple[AcquisitionDiagnostic, ...] = ()
    attempted_query_parts: tuple[GeographicAOI, ...] = ()
    failed_query_part: GeographicAOI | None = None

    @property
    def selector_mapping(self) -> dict[str, str]:
        """Return the exact source-name mapping accepted by the public tiler."""
        return {
            selector.source_name: selector.product_id
            for selector in self.selectors
        }


@dataclass(frozen=True)
class ChipAcquisitionResult:
    """Acquisition outcomes for one request that passed label preflight."""

    prepared_request: PreparedChipRequest
    status: AcquisitionStatus
    group_results: tuple[AcquisitionGroupResult, ...] = ()
    diagnostics: tuple[AcquisitionDiagnostic, ...] = ()

    @property
    def records(self) -> tuple[TileCubeRecord, ...]:
        """Return records from all attempted acquisition groups in order."""
        return tuple(
            record
            for group_result in self.group_results
            for record in group_result.records
        )


def _configured_sources(
    config: ChipConfig,
) -> dict[tuple[str, str], tuple[str, str]]:
    result: dict[tuple[str, str], tuple[str, str]] = {}
    for group in config.acquisition_groups:
        for source in group.tile_config.sources:
            result[(group.name.casefold(), source.name.casefold())] = (
                group.name,
                source.name,
            )
    return result


def derive_source_selectors(
    request: ChipRequest,
    config: ChipConfig,
) -> tuple[SourceSelector, ...]:
    """Resolve and validate every effective product selector for one request."""
    if not isinstance(request, ChipRequest):
        raise TypeError("request must be a ChipRequest.")
    if not isinstance(config, ChipConfig):
        raise TypeError("config must be a ChipConfig.")

    configured = _configured_sources(config)
    explicit: dict[tuple[str, str], SourceSelector] = {}
    for selector in request.source_selectors:
        key = (
            selector.acquisition_group.casefold(),
            selector.source_name.casefold(),
        )
        if key not in configured:
            raise SelectorResolutionError(
                f"Selector references unknown acquisition source "
                f"{selector.acquisition_group!r}/{selector.source_name!r}.",
                sample_id=request.sample_id,
                acquisition_group=selector.acquisition_group,
                source_name=selector.source_name,
            )
        explicit[key] = selector

    resolved: list[SourceSelector] = []
    for group in config.acquisition_groups:
        for source in group.tile_config.sources:
            key = (group.name.casefold(), source.name.casefold())
            override = explicit.get(key)
            if source.selection_mode == "all_intersecting":
                if override is not None:
                    raise SelectorResolutionError(
                        f"All-intersecting source {group.name!r}/{source.name!r} "
                        "does not accept a product selector.",
                        sample_id=request.sample_id,
                        acquisition_group=group.name,
                        source_name=source.name,
                    )
                continue
            if override is not None:
                product_id = override.product_id
            elif source.name.casefold() in {"wac", "nac"}:
                product_id = product_id_from_sample_id(request.sample_id)
            else:
                raise SelectorResolutionError(
                    f"Product-scoped source {group.name!r}/{source.name!r} "
                    "requires an explicit per-request selector rule.",
                    sample_id=request.sample_id,
                    acquisition_group=group.name,
                    source_name=source.name,
                )
            resolved.append(
                SourceSelector(
                    acquisition_group=group.name,
                    source_name=source.name,
                    product_id=product_id,
                )
            )
    return tuple(resolved)


def selectors_for_group(
    selectors: Sequence[SourceSelector],
    acquisition_group: str,
) -> tuple[SourceSelector, ...]:
    """Select effective selectors for one acquisition group in source order."""
    key = acquisition_group.casefold()
    return tuple(
        selector
        for selector in selectors
        if selector.acquisition_group.casefold() == key
    )


def cube_record_key(
    acquisition_group: str,
    record: TileCubeRecord,
) -> CubeRecordKey:
    """Build the filename-independent identity for one acquired cube."""
    return CubeRecordKey(
        acquisition_group=acquisition_group,
        source_name=record.source_name,
        zone=record.zone,
        zoom_level=record.zoom_level,
        tile_x=record.tile_x,
        tile_y=record.tile_y,
    )


def deduplicate_cube_records(
    acquisition_group: str,
    records: Sequence[TileCubeRecord],
) -> tuple[TileCubeRecord, ...]:
    """Deduplicate antimeridian-query overlap using structured identity."""
    by_key: dict[CubeRecordKey, TileCubeRecord] = {}
    for record in records:
        if not isinstance(record, TileCubeRecord):
            raise TypeError("Acquisition results must contain TileCubeRecord objects.")
        key = cube_record_key(acquisition_group, record)
        previous = by_key.setdefault(key, record)
        if previous != record:
            raise ValueError(
                "Conflicting cube records share structured identity "
                f"{key!r}."
            )
    return tuple(by_key[key] for key in sorted(by_key))


def group_cube_records(
    acquisition_group: str,
    records: Sequence[TileCubeRecord],
) -> tuple[CubeRecordGroup, ...]:
    """Group records without parsing product or tile data from filenames."""
    grouped: dict[CubeRecordKey, list[TileCubeRecord]] = {}
    for record in records:
        key = cube_record_key(acquisition_group, record)
        grouped.setdefault(key, []).append(record)
    return tuple(
        CubeRecordGroup(key=key, records=tuple(grouped[key]))
        for key in sorted(grouped)
    )


def _inventory(output_dir: Path) -> tuple[Path, ...]:
    if not output_dir.is_dir():
        return ()
    return tuple(sorted(path for path in output_dir.rglob("*") if path.is_file()))


def _selector_mapping(selectors: Sequence[SourceSelector]) -> dict[str, str]:
    return {selector.source_name: selector.product_id for selector in selectors}


def _failure_diagnostic(
    exc: Exception,
    *,
    group: AcquisitionGroupConfig,
) -> AcquisitionDiagnostic:
    if isinstance(exc, MissingRequiredSourceError):
        code = "missing_required_source"
    elif isinstance(exc, TileSourceError):
        code = "tile_source_error"
    else:
        code = "acquisition_error"
    return AcquisitionDiagnostic(
        code=code,
        message=str(exc),
        severity="error",
        acquisition_group=group.name,
        source_name=getattr(exc, "source_name", None),
        zone=getattr(exc, "zone", None),
        zoom_level=group.tile_config.zoom_level,
        tile_x=getattr(exc, "tile_x", None),
        tile_y=getattr(exc, "tile_y", None),
    )


def _successful_coverage_diagnostics(
    request: ChipRequest,
    group: AcquisitionGroupConfig,
    records: Sequence[TileCubeRecord],
) -> tuple[AcquisitionDiagnostic, ...]:
    diagnostics: list[AcquisitionDiagnostic] = []
    tile_addresses = {
        (record.zone, record.zoom_level, record.tile_x, record.tile_y)
        for record in records
    }
    for source in group.tile_config.sources:
        source_addresses = {
            (record.zone, record.zoom_level, record.tile_x, record.tile_y)
            for record in records
            if record.source_name.casefold() == source.name.casefold()
        }
        if not source_addresses:
            diagnostics.append(
                AcquisitionDiagnostic(
                    code=(
                        "missing_required_source"
                        if source.required
                        else "missing_optional_source"
                    ),
                    message=(
                        f"{'Required' if source.required else 'Optional'} source "
                        f"{source.name!r} returned no cube records for sample "
                        f"{request.sample_id!r}."
                    ),
                    severity="error" if source.required else "warning",
                    acquisition_group=group.name,
                    source_name=source.name,
                    zoom_level=group.tile_config.zoom_level,
                )
            )
            continue
        for zone, zoom_level, tile_x, tile_y in sorted(
            tile_addresses - source_addresses
        ):
            diagnostics.append(
                AcquisitionDiagnostic(
                    code=(
                        "incomplete_required_source_coverage"
                        if source.required
                        else "incomplete_optional_source_coverage"
                    ),
                    message=(
                        f"{'Required' if source.required else 'Optional'} source "
                        f"{source.name!r} returned no record for LTM{zone} "
                        f"zoom {zoom_level} tile ({tile_x}, {tile_y})."
                    ),
                    severity="error" if source.required else "warning",
                    acquisition_group=group.name,
                    source_name=source.name,
                    zone=zone,
                    zoom_level=zoom_level,
                    tile_x=tile_x,
                    tile_y=tile_y,
                )
            )
    return tuple(diagnostics)


def _acquire_group(
    request: ChipRequest,
    config: ChipConfig,
    group: AcquisitionGroupConfig,
    *,
    selectors: Sequence[SourceSelector],
) -> AcquisitionGroupResult:
    """Acquire one isolated group through ``create_tiles_for_aoi``."""
    output_dir = config.intermediate_root / request.sample_id / group.name
    query_parts = geographic_query_parts(request.geographic_aoi)
    group_selectors = selectors_for_group(selectors, group.name)
    selector_mapping = _selector_mapping(group_selectors)
    tile_config = replace(group.tile_config, output_dir=output_dir)
    records: tuple[TileCubeRecord, ...] = ()
    failure: Exception | None = None
    attempted_query_parts: list[GeographicAOI] = []
    failed_query_part: GeographicAOI | None = None

    for part in query_parts:
        attempted_query_parts.append(part)
        try:
            part_records = create_tiles_for_aoi(
                tile_config,
                ul_lat=part.upper_left_latitude,
                ul_lon=part.upper_left_longitude,
                lr_lat=part.lower_right_latitude,
                lr_lon=part.lower_right_longitude,
                selectors=selector_mapping or None,
            )
            records = deduplicate_cube_records(
                group.name,
                (*records, *part_records),
            )
        except Exception as exc:
            failure = exc
            failed_query_part = part
            partial_records = getattr(exc, "completed_records", ())
            try:
                records = deduplicate_cube_records(
                    group.name,
                    (*records, *partial_records),
                )
            except (TypeError, ValueError):
                pass
            break

    diagnostics: list[AcquisitionDiagnostic] = []
    if failure is not None:
        diagnostics.append(_failure_diagnostic(failure, group=group))
    else:
        diagnostics.extend(
            _successful_coverage_diagnostics(request, group, records)
        )

    status: AcquisitionStatus = (
        "failed"
        if any(item.severity == "error" for item in diagnostics)
        else "complete"
    )
    return AcquisitionGroupResult(
        sample_id=request.sample_id,
        acquisition_group=group.name,
        zoom_level=group.tile_config.zoom_level,
        output_dir=output_dir,
        logical_aoi=request.geographic_aoi,
        query_parts=query_parts,
        selectors=group_selectors,
        status=status,
        records=records,
        record_groups=group_cube_records(group.name, records),
        inventory_paths=_inventory(output_dir),
        diagnostics=tuple(diagnostics),
        attempted_query_parts=tuple(attempted_query_parts),
        failed_query_part=failed_query_part,
    )


def acquire_prepared_request(
    prepared: PreparedChipRequest,
    config: ChipConfig,
) -> ChipAcquisitionResult:
    """Acquire groups for one eligible request, stopping at its first failure."""
    if not isinstance(prepared, PreparedChipRequest):
        raise TypeError("prepared must be a PreparedChipRequest.")
    if not isinstance(config, ChipConfig):
        raise TypeError("config must be a ChipConfig.")
    if not prepared.eligible_for_acquisition:
        raise ValueError(
            f"Sample {prepared.request.sample_id!r} did not pass acquisition "
            "preflight."
        )
    try:
        selectors = derive_source_selectors(prepared.request, config)
    except SelectorResolutionError as exc:
        diagnostic = AcquisitionDiagnostic(
            code="invalid_source_selector",
            message=str(exc),
            severity="error",
            acquisition_group=exc.acquisition_group,
            source_name=exc.source_name,
        )
        return ChipAcquisitionResult(
            prepared_request=prepared,
            status="failed",
            diagnostics=(diagnostic,),
        )

    group_results: list[AcquisitionGroupResult] = []
    for group in config.acquisition_groups:
        group_result = _acquire_group(
            prepared.request,
            config,
            group,
            selectors=selectors,
        )
        group_results.append(group_result)
        if group_result.status == "failed":
            break
    diagnostics = tuple(
        diagnostic
        for group_result in group_results
        for diagnostic in group_result.diagnostics
    )
    status: AcquisitionStatus = (
        "failed"
        if any(result.status == "failed" for result in group_results)
        else "complete"
    )
    return ChipAcquisitionResult(
        prepared_request=prepared,
        status=status,
        group_results=tuple(group_results),
        diagnostics=diagnostics,
    )


__all__ = [
    "AcquisitionDiagnostic",
    "AcquisitionGroupResult",
    "AcquisitionSeverity",
    "AcquisitionStatus",
    "ChipAcquisitionResult",
    "CubeRecordGroup",
    "CubeRecordKey",
    "SelectorResolutionError",
    "acquire_prepared_request",
    "cube_record_key",
    "deduplicate_cube_records",
    "derive_source_selectors",
    "group_cube_records",
    "selectors_for_group",
]
