# Lunar Tiling Progress Log

This document tracks day-by-day work on the LFM lunar tiling backend,
validation, documentation, and public notebook. It is an execution record, not
the authoritative API specification.

Canonical references:

- [Tiling modernization plan](../../docs/tiling_modernization_plan.md)
- [Armstrong tiling scheme](../../TMS/README.md)
- [Supported tiling notebook](../../notebooks/tiling_example.ipynb)
- [Lunar tiling agent skill](../skills/lunar-tiling/SKILL.md)

## Current status

- Core tiling modernization: **Complete**
- Supported production geometry: numbered LTM zones
- Public examples: WAC/static at zoom 5 and NAC/static at zoom 11
- Tiling resampling: bilinear
- Canonical static cube: 63 bands with `-32768` output NoData
- Downstream chip modernization: tracked separately

Known boundaries and follow-ups:

- LPN/LPS production tiling is not implemented.
- Automatic discovery followed by one output cube per PID is not implemented.
  An `all_intersecting` WAC/NAC query instead stacks all intersecting raster
  bands into one source cube per LTM tile with `product_id=None`.
- The completed tiling plan records a maintainer-accepted exception: a separate
  top-to-bottom execution of the final `.ipynb` after the last NAC AOI and
  differential-zoom edits was not recorded.
- The no-PID alternate AOI query added on 2026-09-30 has passed notebook JSON
  and Python-cell syntax validation but has not yet been run against Explore
  data.

## Daily entries

### 2026-10-01

Goal:

- Establish a persistent daily progress record for tiling work.
- Plan the automatic LTM/`LPS_N`/`LPS_S` workflow and freeze its extended
  public contract before implementation.

Completed:

- Created `.agents/progress_docs/tiling.md`.
- Seeded the log with the completed modernization milestones and current
  validation boundaries.
- Created `.agents/planning_docs/polar_tiling_integration_plan.md` with a
  strictly sequential P0-P10 implementation and validation sequence.
- Completed Phase P0 in
  `.agents/planning_docs/polar_tiling_contract.md`: inventoried current
  LTM-only assumptions and defined automatic grid routing, optional PID
  discovery, dynamic/static source modes, index-path defaults, polar zoom
  defaults, grid-neutral result compatibility, filenames, ordering, and 23
  acceptance cases.

Validation:

- Passed whitespace validation and confirmed that every canonical reference
  resolves to an existing repository file.
- Audited the public tiling API, geometry classes, source selection,
  vector-index code, raster writer, notebook, helpers, visualization, tests,
  TMS metadata, and relevant downstream chip assumptions against the P0
  contract.
- Confirmed the current result field is `TileCubeRecord.zone`, not
  `ltm_zone`, and corrected the integration plan accordingly.
- Confirmed that loading 92 JSON definitions is not working polar support: the
  existing filename parser reduces the polar definitions to `N` and `S` and
  later attempts an LTM file lookup.
- P0 was documentation-only; no runtime or Explore tiling execution was
  performed or claimed.

Decisions:

- Preserve `record.zone` as the backward-compatible canonical grid ID and add
  a grid-neutral `grid_id` alias during implementation.
- Use WAC zoom 4 and processed 1 m NAC zoom 10 on the polar grids, matching the
  physical resolutions of LTM zooms 5 and 11.
- Write static/context cubes once per tile when omitted-PID discovery finds
  multiple dynamic products.
- Keep the current chip workflow's typed polar rejection until downstream
  polar chip processing is separately implemented and validated.

P1 work started:

- Added a derived-or-explicit `VectorIndexBuildConfig` index path,
  deterministic raster discovery, `ensure_vector_index()`, structured
  validation results, and typed malformed/stale index errors.
- Added validation for driver, layer, location field, lunar CRS, geometry,
  raster path resolution, duplicates, missing rasters, and inventory drift.
- Added visible creation/reuse announcements and logging, including the raster
  count and large-directory timing notice. These behaviors remain pending
  their sequential P1.2/P1.3 acceptance checkpoints.
- Added a Grace/`lfm-container-ipyleaflet` GDAL callback probe and focused
  Shapefile/GeoPackage builder test wrapper.
- Local dependency-free tiling tests passed: 43 tests run, with the two
  GDAL-backed vector-index integration tests skipped because local GDAL is not
  installed. Shell syntax, Python syntax, and whitespace checks passed.
- P1.1 is complete. P1.2 is in progress pending the container-backed focused
  tests; no Explore execution is claimed yet.
- Explore job 37937983 reached the capability probe and failed before the
  focused tests because GDAL 3.8.4 does not expose `gdal.TileIndex` in Python.
  This is accepted capability evidence, not a successful validation run.
- Replaced the unavailable Python utility call with a direct Python/OGR writer
  that creates one footprint feature per raster and exposes genuine
  per-raster tqdm progress on stdout. `gdaltindex` remains the later semantic
  comparison oracle rather than the production writer.
- Updated the probe to report missing Python APIs and the installed
  `gdaltindex` executable/version without crashing. The focused wrapper still
  requires a rerun.
- After the fallback change, 44 dependency-free local tiling tests pass; the
  two GDAL-backed index tests remain skipped locally.
- Explore job 37937987 completed the revised capability report and exercised
  the OGR writer. Its two Shapefile cases failed validation because `.prj`
  downgraded the IAU CRS to WKT1 and lost metadata required by exact
  `OSR.IsSame()` identity; this was not a footprint-creation failure.
- Added a Shapefile-only semantic CRS fallback that strictly compares the
  numeric lunar sphere, flattening, prime meridian, and angular unit that WKT1
  can preserve. GeoPackage and other richer formats retain exact CRS identity
  validation. Suppressed the unrelated test-fixture GDAL exception-mode
  warning. The focused wrapper requires another rerun.
- Explore job 37937988 reached the Shapefile fallback but found that GDAL 3.8's
  Python wrapper lacks `SpatialReference.GetPrimeMeridian()`. Replaced it with
  the compatible numeric `PRIMEM` WKT-node lookup and added a dependency-free
  regression for the older method surface.
- Explore job 37938002 passed all 12 focused vector-index builder tests in the
  supported `lfm-container-ipyleaflet` image. Shapefile and GeoPackage
  creation, validation, reuse, and stale-inventory rejection are now confirmed
  with GDAL enabled.
- The job's capability report confirmed that GDAL exposes neither the Python
  `gdal.TileIndex` nor `gdal.TileIndexOptions` API. P1.2 through P1.5 are
  complete: deterministic inventory announcements and the Python/OGR writer's
  genuine per-raster stdout `tqdm` progress are the supported path.
- Began P1.6. New index creation uses an exclusive sibling lock, writes and
  validates in a temporary sibling directory, then publishes all artifacts
  with the Shapefile `.shp` primary file last. Creation and validation failures
  clean staging files and the lock without publishing a partial index.
- Added dependency-free tests for sidecar publication, active-lock rejection,
  writer-failure cleanup, and validation-failure cleanup. The focused local
  vector-index suite passes 14 tests with its two GDAL integration cases
  skipped, and the broader selected tiling suite passes 43 tests with six
  environment-dependent skips.

Next:

- Rerun `scripts/shell/all_tasks/sbatch_probe_gdal_tileindex_progress.sh` on
  Explore to validate P1.6 staging, sidecar publication, locking, cleanup, and
  stdout progress with GDAL enabled.

### 2026-09-30

Completed:

- Reviewed whether the supported notebook could run without PID filtering.
- Confirmed that `selection_mode="product_id"` requires a nonempty selector.
- Added an opt-in WAC AOI query under `RUN_ALTERNATE_QUERIES` that uses
  `selection_mode="all_intersecting"` and passes no `selectors` argument.
- Documented that this no-PID query produces one stacked WAC cube per LTM tile,
  not one cube per discovered PID.
- Created `.agents/skills/lunar-tiling/SKILL.md` to preserve the modern tiling
  contract and operational conventions for future agent work.

Validation:

- Confirmed that `notebooks/tiling_example.ipynb` remains valid JSON.
- Confirmed Python syntax for every code cell after excluding Jupyter magic
  lines.
- Validated the `lunar-tiling` skill with the official skill validator.
- The new alternate query was not submitted to Explore on this date.

### 2026-09-03

Completed:

- Finalized the user-facing structure and explanatory configuration comments in
  `notebooks/tiling_example.ipynb`.
- Kept user-edited paths, WAC/NAC PIDs, and AOIs separate from derived indexes,
  zoom levels, timestamped outputs, display settings, and path validation.
- Confirmed WAC/static at zoom 5 and moved 1 m NAC/static to zoom 11.
- Added NaN-masked plotting and a maximum display of four tile pairs per AOI.
- Documented the Armstrong scheme, IAU:30100, LTM zones, zoom matrices, tile
  addressing, and the unsupported LPN/LPS production path in `TMS/README.md`.
- Updated the tiling and chip modernization plans with the stable handoff
  contract.
- Marked tiling Phases T0-T8 complete by maintainer signoff.
- Clarified the chip plan's static NoData language: successful canonical static
  cubes contain all 63 ordered bands with `-32768` output NoData; the two
  Mini-RF bands use `-3.4028230607370965e38` only as a source sentinel; WAC and
  NAC preservation is explicitly configured.

Validation:

- Recorded the lack of a separate final `.ipynb` execution as an acceptance
  exception rather than claiming that run occurred.
- Passed notebook structure, JSON, and Python syntax checks available locally.

### 2026-09-01

Completed:

- Replaced the WAC-specific, hard-coded tiling interface with the
  configuration-driven `TileConfig`, `TileSourceConfig`, and
  `BandNoDataOverride` contract.
- Added public AOI, point, and explicit tile-index entry points returning
  structured `TileCubeRecord` results.
- Moved the shared lunar geographic CRS to
  `TMS/IAU_30100_2015.wkt` and loaded it from repository data.
- Generalized source indexes to existing `.shp` or `.gpkg` files queried
  read-only.
- Standardized all tiling warps on bilinear resampling.
- Established the canonical 63-band static order and standardized every static
  output band on `-32768` NoData.
- Preserved the two Mini-RF `-3.4028230607370965e38` source sentinels so they
  are masked before interpolation and converted on output.
- Preserved native WAC and NAC source NoData through explicit configuration.
- Added diagnostics for source ranges, static sentinels, written cube NoData,
  modern-versus-legacy comparison, and display-only overflow investigation.
- Standardized new Slurm wrappers on the `grace` partition and the
  `lfm-container-ipyleaflet` Apptainer image.

Validation:

- WAC validation passed.
- NAC validation passed.
- Mixed WAC/static validation passed after static NoData standardization.
- Modern/legacy regression comparisons passed for covered products; products
  without selected WAC coverage produced an intentional explicit modern error
  instead of a legacy static-only result.
- Determinism validation passed for record order, record metadata, output
  SHA-256 hashes, declared-index immutability, visible index inventory, and
  absence of newly created output indexes.

## Entry template

### YYYY-MM-DD

Goal:

- What this day's work intends to accomplish.

Completed:

- Code, notebook, configuration, diagnostic, or documentation changes.

Validation:

- Commands or jobs run, artifacts inspected, and pass/fail results.
- State explicitly when a change received only static validation.

Decisions:

- Contract decisions, tradeoffs, and boundaries established that day.

Next:

- Remaining work or the next safe step.
