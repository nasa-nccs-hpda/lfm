# Tiling-to-Chip-Creation Handoff

Last updated: 2026-10-02

This document is for an agent modernizing `notebooks/chip_example.ipynb` and
the chip-creation workflow that consumes lunar datacubes. It summarizes the
current tiling contract, recent index-management changes, and the boundaries
that chip creation must preserve.

This is a point-in-time integration handoff, not the complete implementation
history. Use these canonical references when more detail is needed:

- [`docs/chip_creation_modernization_plan.md`](../docs/chip_creation_modernization_plan.md)
- [`docs/tiling_modernization_plan.md`](../docs/tiling_modernization_plan.md)
- [`TMS/README.md`](../TMS/README.md)
- [`notebooks/tiling_example.ipynb`](../notebooks/tiling_example.ipynb)
- [`lfm/data_processing/tiling/tiling_workflow.py`](../lfm/data_processing/tiling/tiling_workflow.py)
- [`lfm/data_processing/tiling/vector_index_builder.py`](../lfm/data_processing/tiling/vector_index_builder.py)
- [`lfm/data_processing/chip/chip_acquisition.py`](../lfm/data_processing/chip/chip_acquisition.py)

## Current migration status

Core tiling modernization is complete. The automatic workflow supports
numbered LTM grids and `LPS_N`/`LPS_S`, optional WAC/NAC product IDs, source
composition, family-specific zooms, and automatic source-index preparation.

Chip phases C0-C7 are complete. C8 real-data validation and C9 notebook
modernization are in progress. The existing chip notebook is an initial draft;
it predates several recent tiling-index improvements and should not be treated
as the final integration contract.

Both example notebooks now resolve indexes through
`model.resolve_notebook_source_index()`. The canonical Explore WAC, NAC, and
static directories use their existing shared `output_index.gpkg` files with
automatic replacement disabled. A changed data directory receives a
persistent, application-owned cache beneath the clone, where automatic rebuild
is safe. The chip notebook validates or prepares these paths once in
coordinator setup before any chip workers start.

## Stable tiling contract that chip creation may rely on

2026-10-06 chip seam extension: downstream acquisition now splits ±82° seam
queries (and antimeridian parts) into single-family calls to the unchanged
strict tiler. Optional acquisition-group `ltm_zoom_level`/`polar_zoom_level`
override the base scalar zoom; the polar notebook uses WAC/static 5/4 or NAC
11/10. Final-grid compositing prefers the geographic owner family, falling
back to valid other-family data per band. Dynamic native grids remain intact;
static-only uses the AOI-center grid at 100 m. Pole-containing chips remain
unsupported. This supersedes older blanket polar/seam rejections below.
Seam GDAL/HPC and real-data validation are pending in the AOI plan.

2026-10-05 coverage extension (user-reported HPC tests passed): tiling now
retains all-NoData warped channels. Missing named channels in a spatial query
are filled only if a read-only, product-filtered inventory of indexed raster
metadata confirms they exist. Canonical static output retains all 63 bands,
using -32768 for uncovered channels and logging warnings instead of stopping
subsequent tiles. Unknown names, unreadable sources and other processing errors
still fail. An entirely uncovered source with no declared band-name schema is
not fabricated. See `test_tiling_coverage.py` for the targeted acceptance tests.
The user reports the coverage, raster-cube and configured-tiler suites passed
on HPC; NAC-plus-static real-data notebook validation remains pending.

### Grids, routing, and zooms

- Geographic inputs use repository-owned IAU:30100 from
  `TMS/IAU_30100_2015.wkt`.
- The high-level router selects numbered LTM zones below the polar threshold,
  `LPS_N` at latitudes `>= 82`, and `LPS_S` at latitudes `<= -82`.
- AOIs crossing a zone, equator, antimeridian, or polar threshold are
  partitioned into deterministic query parts.
- WAC defaults are LTM zoom 5 and polar zoom 4.
- Processed 1 m NAC defaults are LTM zoom 11 and polar zoom 10.
- Every tiling output is exactly 512 by 512 pixels on its addressed Armstrong
  grid.
- All raster warps performed by tiling use bilinear resampling.

The current chip request/preflight layer intentionally accepts only numbered
LTM coverage. `lfm/data_processing/chip/chip_requests.py` raises the typed
`UnsupportedCoverageError` with status `unsupported_polar_coverage` for polar
targets. Keep this rejection until polar chip reprojection, assembly, and
validation receive their own migration. Upstream polar support is not
permission to silently enable polar chips.

### Source and product behavior

The easy tiling API is exported from `model`:

```python
from lfm.data_processing.tiling import (
    TileAOIQuery,
    TilePointQuery,
    create_tiles_for_query,
    make_nac_tile_source,
    make_static_tile_source,
    make_wac_tile_source,
)
```

- WAC and NAC presets are dynamic, `product_id`-scoped sources.
- A PID string selects one observation. `None` discovers all intersecting
  product IDs and keeps companion files for each product together.
- Static is an `all_intersecting` contextual source and never accepts a PID.
- `create_tiles_for_query()` defaults to dynamic plus static, while
  `include_dynamic` and `include_static` allow either source class to be
  disabled. Disabled sources are not discovered, indexed, validated, queried,
  or written.
- Static context is written once per tile even when multiple dynamic products
  are discovered.

Chip acquisition currently uses the strict low-level
`create_tiles_for_aoi(TileConfig, ..., selectors=...)` path. This is still a
valid boundary: chip requests derive or provide exact selectors and each
`AcquisitionGroupConfig` has one explicit zoom. Do not switch chip acquisition
to omitted-PID discovery without a deliberate change to sample identity,
manifest semantics, and multi-product output handling.

### Valid AOIs with no intersecting products

The high-level automatic workflow no longer treats an otherwise geographically
valid AOI as invalid merely because a product-scoped dynamic source has no
intersecting product. Product discovery:

- logs a warning and emits `ProductAOIWarning` for each unavailable requested
  or discovered product source;
- omits that source rather than raising `AutomaticTilingError` at product
  discovery;
- allows another configured dynamic source with an intersecting product to
  continue; and
- returns an empty `TileCubeRecord` list without starting cube creation when no
  configured dynamic source is runnable. In that case contextual static is
  skipped rather than written by itself for the mixed query.

This does not weaken geographic validation: an AOI that cannot route to any
lunar grid tile still raises a routing error. It also does not remove strict
low-level per-tile required-source errors such as `MissingRequiredSourceError`.

Chip acquisition currently calls the strict low-level API, so this high-level
change may not directly alter every chip request yet. Nevertheless, chip code
must not use “no exception” as evidence that acquisition produced usable data.
It must inspect returned structured records against required output modalities.
An empty record set or absent required modality becomes a typed, per-sample
chip acquisition failure with no published chip/label pair; optional omissions
follow the existing placeholder policy only when their band contract is known.
Capture structured record/coverage diagnostics rather than parsing warning
text. If chip acquisition later adopts the high-level workflow, add focused
empty-result and alternate-dynamic-source tests before changing the API call.

### Structured results

Tiling returns ordered `TileCubeRecord` objects. Use these fields rather than
parsing filenames:

```text
source_name, zone, grid_id, zoom_level, tile_x, tile_y,
product_id, path, band_names, crs_wkt, nodata_values
```

`zone` remains the backward-compatible canonical grid identifier. `grid_id` is
its grid-neutral alias and may contain an LTM ID such as `42N` or a polar ID
such as `LPS_N`.

The chip backend already follows this rule through `CubeRecordKey`,
`deduplicate_cube_records()`, and `group_cube_records()`. Preserve group name,
source name, grid, zoom, tile column, and tile row as the structured identity.

### Static and dynamic raster contracts

- The canonical static cube has 63 bands in the exact order defined by
  `STATIC_BAND_NAMES`.
- Every static output band declares destination NoData `-32768`.
- The two Mini-RF bands named by `MINIRF_SOURCE_NODATA_BANDS` additionally
  declare `-3.4028230607370965e38` as a source-only sentinel. That value must
  be translated during tiling; it is not the static cube's output NoData.
- WAC and NAC presets preserve their native source NoData explicitly.
- Static tiling may produce `Float64` cubes so mixed source dtypes and sentinel
  translation remain safe. Chip assembly may downcast only after its existing
  range and NoData checks succeed.
- Tiling resampling and chip-stage resampling are separate decisions. Tiling is
  always bilinear. Chip output modalities retain their configured bilinear or
  nearest-neighbor reprojection policy.

## Raster-index contract

### Supported preparation behavior

`VectorIndexBuildConfig`, `ensure_vector_index()`, and
`TileSourcePreparation` support:

- `.tif`, `.tiff`, `.nc`, and `.vrt` raster discovery;
- Shapefile and GeoPackage indexes;
- deterministic raster inventory and feature order;
- IAU:30100 geographic footprints with 21 samples per raster edge;
- longitude unwrapping and antimeridian-safe multipart footprints;
- conservative full-longitude caps for pole-containing rasters;
- valid full-longitude bands for global rasters; and
- staged validation, locking, and atomic publication.

Some global products contain one redundant seam column. The builder may clamp
at most one nominal raster pixel of longitude overlap into the canonical
`[-180, 180]` band. Wider spans remain errors. This changes only conservative
index geometry; it does not modify source pixels or tiling output.

### Ownership and replacement policy

- Existing valid indexes are validated and reused byte-for-byte.
- Missing indexes may be created.
- Invalid or stale indexes are rebuilt automatically only when
  `rebuild_invalid_index=True` and the target is an application-owned
  GeoPackage.
- Shared indexes and legacy Shapefiles are protected and must never be
  automatically replaced.
- Index creation uses a sibling lock and staged output, so a failed build must
  not publish a partial index.
- Low-level tile generation remains read-only with respect to every index.

Do not point `rebuild_invalid_index=True` at a shared GeoPackage. A user-owned
cache under the clone may opt in; a shared project index should be validated
and reused or fail with an actionable diagnostic.

### Parallel footprint inspection

`worker_count=None` resolves to `SLURM_CPUS_PER_TASK` and falls back to one
outside Slurm. `worker_count=1` forces serial operation. Only raster reads and
footprint transforms run in spawned worker processes; deterministic OGR writes
remain serial in the parent.

The public WAC, NAC, and static constructors expose this as
`index_worker_count`. Progress uses plain stdout `tqdm`, not notebook widgets.

For chip creation, prepare all required indexes once in coordinator preflight,
before starting the chip `max_workers` process pool. Do not let every chip
worker discover or rebuild the same source index. This avoids lock contention,
nested process pools, repeated full-directory scans, and CPU oversubscription.

### Shared default indexes

The temporary maintenance entry points are:

- `scripts/python/all_tasks/create_shared_default_indexes.py`
- `scripts/shell/all_tasks/sbatch_create_shared_default_indexes.sh`

They build `output_index.gpkg` in the default WAC, NAC, and static directories,
validate existing files instead of replacing them, and use eight Grace CPUs by
default. These are maintainer tools, not notebook APIs.

The implemented notebook resolution policy is:

1. If the configured default/shared directory has an explicit validated shared
   `output_index.gpkg`, use it read-only with automatic rebuilding disabled.
2. Otherwise create or reuse a user-owned GeoPackage under a persistent clone
   cache such as `outputs/chip_creation/indexes/`, where
   `rebuild_invalid_index=True` is safe.
3. For a custom user data directory, create or validate its configured index
   once before chip batching. Prefer a user-writable cache location if the
   raster directory is shared or read-only.
4. Do not silently fall back from a declared invalid shared index to a
   different index; surface the validation error and the paths involved.

`resolve_notebook_source_index()` performs this ownership decision and returns
the selected path plus `uses_shared_default` and `rebuild_invalid_index`
metadata. Do not replace it with bare `resolve_tile_index_path()`, which does
not encode notebook ownership policy.

## How this maps onto the current chip architecture

`ChipConfig` contains ordered `AcquisitionGroupConfig` objects. Each group
contains one `TileConfig`, so all sources in that group share one acquisition
zoom. WAC plus static normally share zoom 5; NAC plus static normally share
zoom 11. If a dataset needs both resolutions, use separate acquisition groups
and qualify every `OutputModalityConfig` by acquisition group and source.

`lfm/data_processing/chip/chip_acquisition.py` currently:

- isolates intermediate cubes under
  `<intermediate_root>/<sample_id>/<acquisition_group>`;
- splits antimeridian AOIs into query parts;
- calls strict `create_tiles_for_aoi()` per part;
- derives WAC/NAC selectors from the sample ID unless explicitly overridden;
- deduplicates records through structured keys; and
- stops later acquisition groups after the first group failure while retaining
  structured partial diagnostics.

Because a valid tiling call may return no records under the high-level
discovery contract, any future migration of this caller must retain an explicit
post-call required-coverage check. A warning-only tiling outcome is not, by
itself, a successful model-ready chip.

Keep index preparation outside those per-sample directories. The prepared
`TileSourceConfig.index_path` values should be stable inputs shared read-only
by all sample workers.

The reference-TIFF workflow remains the preferred chip interface. It derives
the geographic AOI through IAU:30100 while preserving the source TIFF's exact
target CRS, geotransform, width, and height for final model-ready output. A
training GeoDataFrame is not part of the modern contract.

## Recommended notebook refactor sequence

1. Preserve the repository-root and `/panfs` to `/explore` path normalization
   already used by the current notebook.
2. Keep true user inputs together: raster directories, reference/label paths,
   optional selectors, output root, split policy, chip worker count, and index
   worker count.
3. Keep the derived index-resolution/preparation section before constructing
   `TileConfig` and before calling `create_chips()`.
4. Continue resolving shared defaults through
   `resolve_notebook_source_index()` and using persistent per-clone
   GeoPackage caches for overrides. Print whether each index was created,
   rebuilt, or reused.
5. Build `TileSourceConfig` objects from those validated paths. Reuse the
   canonical static constants; do not duplicate the 63-band list or sentinel
   rules in notebook prose.
6. Construct acquisition groups by resolution, then `ChipConfig` and output
   modalities.
7. Keep polar preflight rejection visible in notebook markdown.
8. Run label/request preflight before acquisition, then use structured chip and
   tile results for summaries and plots.
9. Make serial index creation (`INDEX_WORKER_COUNT=1`) and serial chip creation
   (`MAX_WORKERS=1`) easy to select independently.
10. Validate the notebook top-to-bottom in
    `lfm-container-ipyleaflet` on the `grace` partition before marking C9
    complete.

## Validation evidence and open gates

Validated in the supported Explore container:

- The automatic tiling workflow, optional PID behavior, LTM/polar routing,
  source modes, and polar WAC generation passed their focused regression gates.
- WAC/static and NAC/static LTM tiling, deterministic output, static band order,
  bilinear resampling, and NoData behavior passed prior regression testing.
- The vector-index builder passed creation, validation, reuse, stale inventory,
  curved polar footprint, seam, locking, staging, and preparation tests.
- Real-data index creation and reuse passed for the previously failing WAC
  seam raster `M1122410242CE.prj.uv.mos.tif`.
- Real-data global-footprint creation and reuse passed for
  `LDRM_32_N_FLOAT.iau.tif`.
- Invalid application-owned cache replacement passed the 43-test focused suite
  in Explore job 37938161.

Still open as of this handoff:

- The newest redundant-seam handling prompted by
  `WAC_EMP_321NM.iau.tif` passes the dependency-light local suite but still
  needs its exact supported-container real-data confirmation.
- The two-process GDAL index regression still needs final Explore confirmation.
- The tiling notebook still needs its final top-to-bottom P7.8 execution gate.
- The chip notebook needs its C9 top-to-bottom HPC run, and C8 real-data dataset
  validation remains incomplete.

Do not describe these open items as regression-closed until their reports have
been inspected.

## Do-not-regress checklist

- Do not parse cube filenames to recover grid or product metadata.
- Do not mutate or automatically replace a shared source index.
- Do not build an index separately in every chip worker.
- Do not bypass `resolve_notebook_source_index()` when selecting notebook
  indexes.
- Do not standardize dynamic NoData to the static `-32768` value.
- Do not expose Mini-RF source sentinels as canonical static output NoData.
- Do not change tiling away from bilinear resampling.
- Do not combine WAC zoom 5 and NAC zoom 11 in one `TileConfig`.
- Do not enable polar chip creation merely because tiling supports polar grids.
- Do not replace the reference-TIFF target grid with a rounded AOI-derived
  grid.
- Do not start nested index and chip worker pools.
