# Tiling Modernization Plan

## Objective

Replace the WAC-specific, partially hard-coded tiling interface with a
configuration-driven interface that can create LTM datacubes for any declared
lunar raster modality. Each source modality will declare its own raster
directory, vector index, selection behavior, bands, and NoData policy. All
tiling warps use bilinear resampling.

This plan covers tiling only: geographic or tile queries through creation of
LTM datacubes. Reference-image alignment and final training-chip creation are
covered by `chip_creation_modernization_plan.md`.

## Status and sequencing

- `[Planned]`: not started.
- `[In-P]`: currently in progress.
- `[Complete]`: implemented, tested, and documented for the stated scope.

Phases and sub-steps are strictly sequential. Start a sub-step only after the
preceding sub-step is `[Complete]`, and start a phase only after every sub-step
in the preceding phase is `[Complete]`. Only one sub-step should be `[In-P]` at
a time. A phase becomes `[Complete]` when all of its sub-steps are complete.

## Baseline contract and modernization boundary

The legacy implementation has the following observable contract:

- `Pipeline(tileDbPath, outDir, debug=False, targetProductID=None)` accepts one
  dynamic raster index and obtains contextual rasters from the hard-coded
  `Pipeline.STATIC_FILE_DB` path.
- `runTileIndex`, `runPoint`, and `run` are the public query entry points. They
  return `list[Path]`; callers recover modality, zone, zoom, tile coordinates,
  and product identity by parsing filenames.
- TMS JSON files define LTM zones and zoom matrices. Every current tile is
  512×512 pixels; cell size and matrix dimensions vary by zoom.
- The shared lunar geographic CRS is repository data in
  `TMS/IAU_30100_2015.wkt`.
  Code loads that file instead of embedding IAU:30100 WKT string literals. The
  cloned repository is therefore part of the public tiling runtime contract.
- A geographic AOI may cross zones and tiles. `TmsIntersector` finds zones,
  `TmsTileDef` identifies tiles, and `Pipeline` processes them in the returned
  order.
- Dynamic rasters are grouped into one cube per filename-derived product ID.
  `targetProductID`, when present, filters every non-static source before
  warping.
- Contextual rasters are grouped into one multi-band `StaticCube` per tile and
  are never product-filtered.
- Every intersecting source raster is warped directly onto the LTM tile grid
  with bilinear resampling. Empty bands are omitted.
- Output files are tiled, LZW-compressed GeoTIFFs with LTM CRS, tile-grid
  transform, band `Name` metadata, and group-writable permissions.
- Every static output band uses `-32768` NoData. Per-band source NoData remains
  explicit so values such as the Mini-RF `-3.4e38` sentinel are masked before
  bilinear resampling and converted to `-32768` in the output. Dynamic bands
  preserve source NoData.
- Existing tests are predominantly HPC integration tests tied to `/explore`
  WAC, NAC, and static data. Modernization needs local configuration and
  contract tests in addition to retaining those integration checks.

The modern boundary is:

```text
query + TileConfig + per-source query selectors
    -> TMS zone/tile discovery
    -> source-index queries
    -> LTM cube creation
    -> ordered TileCubeRecord results and GeoTIFFs
```

Reference TIFFs, final target-grid reprojection, label handling, and training
dataset layout remain outside this boundary.

Temporary compatibility covers the legacy constructor and its three query
methods while repository callers migrate. New code uses configuration objects,
format-independent indexes, explicit source policies, and structured results.

Acceptance scenarios are applied in this order:

1. WAC-only: a product selector produces only that product's seven-band cubes.
2. NAC-only: the same public API produces product-scoped NAC cubes without a
   WAC alias or static database.
3. WAC plus static: one query produces independently identified WAC and static
   records for each intersecting tile, with declared source order and NoData.

## Phase T0 — Preserve and define the existing contract `[Complete]`

- `[Complete]` **T0.1** Inventory the behavior of `Pipeline`,
  `TmsIntersector`, `TmsZoneDef`, and `TmsTileDef`, including tile dimensions,
  filenames, band metadata, product filtering, permissions, and NoData output.
- `[Complete]` **T0.2** Record the current public entry points for tile-index,
  point, and geographic-AOI queries and identify which behaviors require a
  temporary compatibility adapter.
- `[Complete]` **T0.3** Define the tiling boundary: inputs are a query, a
  `TileConfig`, and optional per-query selectors; outputs are structured cube
  records plus files on disk.
- `[Complete]` **T0.4** Define acceptance examples for WAC-only, NAC-only, and
  WAC-plus-static tiling before changing implementation code.
- `[Complete]` **T0.5** Store the shared IAU:30100 lunar geographic CRS in a
  repository `.wkt` file and use one loader instead of inline WKT strings.

## Phase T1 — Introduce tiling configuration objects `[Complete]`

- `[Complete]` **T1.1** Add a `TileSourceConfig` dataclass containing the source
  name, data directory, vector-index path, optional layer name, raster-location
  field, and selection mode.
- `[Complete]` **T1.2** Add per-source band selection, source NoData, output
  NoData, and any required per-band override fields; validate bilinear as the
  tiling resampling contract.
- `[Complete]` **T1.3** Add a `TileConfig` dataclass containing the ordered source
  configurations, output directory, zoom level, and debug settings.
- `[Complete]` **T1.4** Add a small dictionary-to-config constructor so notebooks
  can express source definitions as plain Python while backend functions
  receive validated config objects.
- `[Complete]` **T1.5** Add focused tests for valid configs, missing required
  fields, duplicate source names, unsupported selection modes, and invalid
  resampling or NoData settings.

## Phase T2 — Generalize vector-index access `[Complete]`

- `[Complete]` **T2.1** Replace the explicitly selected ESRI Shapefile driver
  with format-independent GDAL/OGR dataset opening.
- `[Complete]` **T2.2** Support both `.shp` and `.gpkg` indexes, including an
  optional configured GeoPackage layer and a configurable raster-location
  field.
- `[Complete]` **T2.3** Resolve relative raster paths against the configured data
  directory while preserving support for absolute paths stored in an index.
- `[Complete]` **T2.4** Separate index creation or refresh from tile generation;
  tiling will consume an existing declared index and will not silently rebuild
  it.
- `[Complete]` **T2.5** Add equivalent query tests using minimal Shapefile and
  GeoPackage fixtures.

## Phase T3 — Replace static/dynamic branching with source policies `[Complete]`

- `[Complete]` **T3.1** Replace the hard-coded WAC database and static database
  with ordered iteration over `TileConfig.sources`.
- `[Complete]` **T3.2** Implement `product_id` selection for product-scoped
  modalities such as WAC and NAC.
- `[Complete]` **T3.3** Implement `all_intersecting` selection for contextual
  modalities such as static lunar layers.
- `[Complete]` **T3.4** Pass per-query product IDs or equivalent selectors by
  source name rather than through the pipeline constructor.
- `[Complete]` **T3.5** Apply each source's configured bands and NoData policy
  while warping source rasters to the LTM tile grid with bilinear resampling.
- `[Complete]` **T3.6** Preserve ordered, meaningful band metadata independently
  for every source modality.
- `[Complete]` **T3.7** Verify WAC-only, NAC-only, static-only, and mixed-source
  behavior with focused tests.

## Phase T4 — Introduce structured tiling results `[Complete]`

- `[Complete]` **T4.1** Add a `TileCubeRecord` containing source name, LTM zone,
  zoom level, tile coordinates, optional product ID, output path, band names,
  CRS, and NoData information.
- `[Complete]` **T4.2** Make tile-index, point, and AOI queries return ordered
  `TileCubeRecord` collections instead of requiring callers to parse filenames.
- `[Complete]` **T4.3** Define generic, deterministic filenames that retain zone,
  zoom, tile, source, and optional product identity for human inspection.
- `[Complete]` **T4.4** Ensure an empty intersection, skipped raster, partial
  source result, and fatal source failure have explicit result or exception
  behavior.
- `[Complete]` **T4.5** Add tests proving that structured records and written
  raster metadata agree.

## Phase T5 — Stabilize the public tiling API `[Complete]`

- `[Complete]` **T5.1** Add clear public functions for tile-index, point, and AOI
  creation that accept `TileConfig`.
- `[Complete]` **T5.2** Keep the TMS geometry classes responsible only for zone,
  coordinate, and tile calculations.
- `[Complete]` **T5.3** Isolate GDAL raster reading, warping, and writing from
  query orchestration so each layer can be tested independently.
- `[Complete]` **T5.4** Add temporary compatibility wrappers for legacy callers
  identified in T0, with deprecation guidance pointing to the new API.
- `[Complete]` **T5.5** Run the existing tiling test suite and resolve regressions
  without restoring WAC-specific branching.

T5.5 checkpoint: the Explore run on 2026-09-01 used GDAL 3.8.4 and the
repository `TMS/IAU_30100_2015.wkt`. All 42 modern contract tests passed with
the GDAL-backed tests enabled, all 25 selected legacy geometry and pipeline
regressions passed, and the filtered one-tile legacy integration test passed.
That integration produced one selected WAC product and 63 intersecting static
products. The unfiltered legacy tests that generate hundreds or thousands of
cubes and the test with another user's hard-coded output directory remain
excluded from routine validation.

## Phase T6 — Validate on representative lunar data `[Complete]`

- `[Complete]` **T6.1** Run a WAC product-scoped AOI query on the HPC data and
  inspect zone, tile, band, CRS, transform, shape, and NoData metadata.
- `[Complete]` **T6.2** Run the equivalent NAC query through the same public API.
- `[Complete]` **T6.3** Run a mixed WAC-plus-static query with explicitly ordered
  static bands and per-band NoData considerations.
- `[Complete]` **T6.4** Compare representative new WAC/static cubes against legacy
  outputs and document any intentional differences.
- `[Complete]` **T6.5** Confirm that repeated runs are deterministic and do not
  depend on filename parsing or implicit index creation.

T6.1 checkpoint: `scripts/python/all_tasks/validate_wac_tiling.py` exercises
the modern `TileConfig` and `create_tiles_for_aoi` API with WAC product
`M1187363083CE` over the established two-tile AOI at zoom 5. It validates two
seven-band records in LTM zone `42N` against their written raster CRS,
geotransform, 512×512 shape, band names, and NoData metadata, then writes a
JSON report. Submit it on Explore with
`scripts/shell/all_tasks/sbatch_validate_wac_tiling.sh`.

The first two Explore attempts (jobs 37921020 and 37921024) reached zone `42N`
but failed while PROJ constructed a transformation to the custom-authority LTM
CRS with OSR's Python exception mode enabled; cloning the LTM geographic base
did not change that behavior. The earlier Explore regression run had already
verified the same transformations against expected numeric coordinates with
OSR's non-exception behavior. `TmsTileDef` now scopes that behavior to
transformation construction with `osr.ExceptionMgr`, restores raster exception
handling immediately afterward, uses the repository IAU:30100 WKT directly,
caches both transformations, rejects missing transformation objects, and
rejects non-finite results. A rerun is pending. The validation report also
records and asserts bilinear resampling.

The third attempt (job 37921176) passed coordinate transformation and reached
the WAC vector-index query, then failed because the older `lfm-container`
runtime could not locate `proj.db`. Repository batch scripts now default to the
fixed `lfm-container-ipyleaflet` runtime used by the successful T5 regression
run. A rerun is pending.

The final T6.1 run (job 37921233) passed in 20 seconds with
`lfm-container-ipyleaflet`. It produced deterministically ordered tiles
`(1, 62)` and `(1, 63)`, each with a 512×512 seven-band WAC cube in LTM zone
`42N`. Both records and files agreed on CRS, geotransform, band names, and the
source-preserved NoData value `-3.4028226550889045e+38`; the report confirmed
bilinear resampling.

T6.2 checkpoint: `scripts/python/all_tasks/validate_nac_tiling.py` applies the
same public AOI API and metadata inspection to known NAC product
`M1117899885LE`. Because NAC observations are sparse, an AOI tile without that
product is skipped explicitly while at least one single-band output is
required. Submit it with
`scripts/shell/all_tasks/sbatch_validate_nac_tiling.sh`. The Explore validation
passed, confirming that the product-scoped source policy is modality-neutral
and that sparse NAC coverage is handled through explicit optional-tile skips.

T6.3 checkpoint: the 63-band order from `docs/wac_static_bands.md` is now a
repository contract in `model/static_band_contract.py`. The mixed validator
declares WAC first and static second, uses the staticLinks `db2.shp` index,
requires the exact static band order, applies `-32768` output NoData to all
bands, declares the Mini-RF source sentinel separately for DeltaCPR and
DeltaS1, and asserts bilinear resampling for both sources. Submit
`scripts/shell/all_tasks/sbatch_validate_mixed_wac_static_tiling.sh`. The first
mixed run exposed the GeoTIFF constraint that a
multiband dataset persists only one `TIFFTAG_GDAL_NODATA` value. Before revising
the output contract, use
`scripts/shell/all_tasks/sbatch_diagnose_static_source_ranges.sh` to sample
native static sources, report per-band valid ranges, and identify sampled
collisions with candidate NoData values. This diagnostic intentionally does
not change the tiling policy. Follow that source-level sample with
`scripts/shell/all_tasks/sbatch_diagnose_static_cube_nodata.sh`, which creates
representative static LTM cubes and compares source, intended output,
persisted GeoTIFF, and pixel-level sentinel behavior for every canonical band.

The source diagnostic sampled all 63 canonical bands and found no valid
collision with `-32768`; the cube diagnostic then isolated the mixed-output
metadata problem to DeltaCPR and DeltaS1. The static output contract now uses
`-32768` for every band while retaining `-3.4e38` as the source-only NoData for
those two Mini-RF bands.

The standardized mixed validation rerun passed: all 63 static bands write and
reopen with `-32768` NoData while Mini-RF source pixels are still masked using
their declared source sentinel. T6.3 is complete.

T6.4 checkpoint: `scripts/python/all_tasks/compare_legacy_modern_tiling.py`
runs the representative two-tile AOI through both the legacy compatibility
`Pipeline` and modern `TileConfig` API. It pairs outputs by modality and LTM
tile, aligns bands by `Name`, compares CRS, transform, shape, NoData masks, and
valid pixels, and records filename, band-order, canonical-selection, and dtype
differences separately. Submit it with
`scripts/shell/all_tasks/sbatch_compare_legacy_modern_tiling.sh`.

The first comparison run (job 37921387) passed both 63-band static cubes but
found sentinel-scale differences in all seven WAC bands. The modern path was
passing explicit per-band `srcNodata` to GDAL, which otherwise changes the
multi-band default to `UNIFIED_SRC_NODATA=YES`. Since WAC bands have independent
valid footprints, that policy allowed one band's NoData sentinel to enter
bilinear interpolation whenever another band was valid. The raster backend now
sets `UNIFIED_SRC_NODATA=PARTIAL` to reproduce the legacy intrinsic-NoData
behavior while retaining the explicit modern contract. T6.4 remains in
progress pending a comparison rerun.

The second comparison run (job 37921397) confirmed that `PARTIAL` removed the
sentinel contamination from the later bands of each WAC source, but the leading
UV and VIS bands and a small number of mask pixels still differed from the
legacy intrinsic-metadata path. The backend now omits explicit GDAL NoData
arguments when resolved source and output values exactly match each band's
native metadata. Explicit source overrides and output normalization continue to
use `UNIFIED_SRC_NODATA=PARTIAL`. T6.4 remains in progress pending another
comparison rerun.

The final comparison run (job 37921398) passed all four representative cubes in
22 seconds. Both seven-band WAC cubes matched legacy output with no failed bands,
and both 63-band static cubes matched in band order, grid, CRS, NoData masks, and
valid pixel values. The documented intentional static difference is output
dtype: legacy selects `float32` from its first band, while the modern writer
uses a common safe result dtype (`float64`) for the mixed source-band dtypes.
T6.4 is complete.

T6.5 checkpoint: `scripts/python/all_tasks/validate_tiling_determinism.py` runs
the representative mixed WAC/static AOI twice into independent output
directories. It pairs outputs only through returned `TileCubeRecord` fields,
compares record order and metadata, requires byte-identical GeoTIFF SHA-256
digests, and verifies that declared index files and the visible index inventory
are unchanged. It also rejects any index artifact created in either tiling
output. Submit it with
`scripts/shell/all_tasks/sbatch_validate_tiling_determinism.sh`.

The Explore validation (job 37921401) passed every determinism check. The two
runs returned identical structured record order and metadata and produced
byte-identical GeoTIFF SHA-256 digests. Declared index contents and the visible
index inventory were unchanged, and neither output directory contained a new
index artifact. T6.5 and Phase T6 are complete.

## Phase T7 — Modernize the tiling example notebook `[In-P]`

- `[Complete]` **T7.1** Create `notebooks/tiling_example.ipynb` while retaining the
  legacy notebook until migration is complete.
- `[Complete]` **T7.2** Use the repository-root convention from
  `notebooks/instance_ibm_train.ipynb` and derive repository-owned paths from
  `repo_root`.
- `[Complete]` **T7.3** Put HPC source paths, indexes, output paths, and modality
  declarations in one visible configuration section.
- `[Complete]` **T7.4** Demonstrate tile-index, point, and AOI queries with
  `TileConfig` and structured results.
- `[Complete]` **T7.5** Add compact visual and metadata inspection of generated
  cubes, including band names and NoData counts.
- `[In-P]` **T7.6** Execute the notebook top-to-bottom on the HPC system and
  clear stale outputs before committing the modern copy.

T7.1–T7.5 checkpoint: the new top-level `notebooks/tiling_example.ipynb`
retains the legacy nested notebook, follows the repository-root and
`/panfs`-to-`/explore` discovery convention, and exposes all user-editable data,
index, selector, AOI, and output paths in one section. It uses `TileConfig` for
representative WAC-plus-static and NAC-plus-static AOI runs, pairs outputs from
structured records, and plots masked dynamic and static bands with per-panel
colorbars and robust limits. Optional point and tile-index examples demonstrate
the other public query entry points without running by default. T7.6 remains in
progress pending a top-to-bottom Explore execution.

## Phase T8 — Complete the tiling migration `[Planned]`

- `[Planned]` **T8.1** Update backend docstrings and user documentation with the
  final config schema and public API.
- `[Planned]` **T8.2** Update repository callers to use `TileConfig` or an
  intentional compatibility wrapper.
- `[Planned]` **T8.3** Mark legacy hard-coded constructor arguments and notebook
  paths as deprecated or remove them after all callers migrate.
- `[Planned]` **T8.4** Run formatting, unit tests, and the final HPC smoke test.
- `[Planned]` **T8.5** Mark this plan complete and unblock Phase C0 of the chip
  creation modernization plan.
