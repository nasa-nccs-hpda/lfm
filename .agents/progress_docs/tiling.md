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
- Supported low-level production geometry: numbered LTM, `LPS_N`, and `LPS_S`
- Public examples: WAC/static at zoom 5 and NAC/static at zoom 11
- Tiling resampling: bilinear
- Canonical static cube: 63 bands with `-32768` output NoData
- Automatic raster-index preparation backend: **Complete**
- Default indexed raster formats: `.tif`, `.tiff`, `.nc`, and `.vrt`
- Optional AOI product discovery: **Complete**
- Grid-neutral metadata registry and geographic router: **Complete**
- Polar tile geometry and low-level addressing: **Complete**
- Downstream chip modernization: tracked separately

Known boundaries and follow-ups:

- `LPS_N`/`LPS_S` low-level production is regression-closed. Source-mode
  composition, family-specific default zooms, and the easy automatic workflow
  remain P5/P6 work.
- The legacy opt-in `all_intersecting` WAC alternate query still stacks all
  intersecting rasters into one source cube. The supported high-level AOI path
  instead discovers products and writes one dynamic cube per PID.
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
- The maintainer reported that the expanded supported-container suite passed.
  P1.6 failure-safe publication and P1.7 completed-index validation are now
  complete.
- Began P1.8. Raster footprints are now sampled at 21 points per edge before
  transformation instead of being reduced to four transformed corners. Added
  pure perimeter tests and a GDAL-backed `LPS_N` curvature/vertex-count test.
- The updated focused local builder suite passes 16 tests with three GDAL
  integration tests skipped; the broader selected tiling suite passes 46 tests
  with seven environment-dependent skips.
- Explore job 37938014 passed the 19-test supported-container suite, including
  the `LPS_N` densified-edge regression. P1.8 is complete. Its report confirms
  GDAL 3.8.4, `/usr/bin/gdaltindex`, and no Python TileIndex callback API.
- Began P1.9. Added a semantic comparison diagnostic for LTM, north-polar, and
  paired longitude-seam fixtures plus a dedicated Grace wrapper using
  `lfm-container-ipyleaflet`. The JSON report compares CRS, inventory and order,
  valid geometry, vertices, bounds, area, and AOI query behavior against the
  installed `gdaltindex` CLI.
- Explore job 37938028 passed the LTM and seam fixtures. Its polar fixture
  matched CRS, driver, inventory, order, validity, exact bounds, and all AOI
  queries, while the 81-vertex LFM outline had 9.509 percent more planar area
  than GDAL's five-vertex chord outline and exceeded the provisional
  two-percent threshold.
- Recorded the maintainer decision that the densified curved footprint is
  canonical. Polar GDAL area is now informational rather than an acceptance
  gate; the production 21-sample outline must converge to a higher-resolution
  201-sample reference within 0.001 degrees and 0.1 percent relative area. The
  existing GDAL area tolerance remains active for LTM and seam fixtures.
- Explore job 37938033 passed the canonical-curvature comparison, including
  the 21-sample versus 201-sample polar convergence gate. P1.9 is complete.
- Began P1.10. Added a high-level source preparation contract that ensures
  enabled indexes before constructing the low-level `TileConfig`, returns the
  validation results, and completely skips disabled sources. The existing tile
  generation and vector-query APIs remain read-only.
- Added dependency-free tests for preparation order, derived builder settings,
  disabled-source skipping, and pre-mutation request errors. Added a GDAL
  integration test that creates an enabled GeoPackage while proving a disabled
  nonexistent directory is untouched. The expanded selected local suite passes
  52 tests with eight GDAL-dependent skips.
- Explore job 37938034 passed all 25 focused builder and high-level preparation
  tests. P1.10 is complete.
- Began P1.11 by auditing its required matrix. Added logger delivery assertions,
  explicit archive-then-rebuild coverage, byte/mtime identity checks for reused
  Shapefile and GeoPackage indexes, and a real low-level tile-generation test
  that requires its prepared GeoPackage to remain byte-for-byte and
  timestamp-identical. Automatic destructive rebuild remains intentionally
  unsupported.
- The expanded selected local suite passes 54 tests with ten GDAL-dependent
  skips.
- Explore job 37938035 passed all 27 focused builder and preparation tests.
  P1.11 and all of Phase P1 are complete. This confirms stdout/log progress,
  deterministic creation, reuse and staleness rules, caller-explicit rebuild,
  concurrency locking, failure cleanup, Shapefile sidecars, GeoPackage output,
  disabled-source skipping, and byte/mtime index immutability during low-level
  tile generation.
- Added a post-P1 real-data progress smoke test. It deterministically selects a
  small configurable WAC sample, creates only symlinks in an isolated
  job-specific test directory, builds and validates a GeoPackage with the
  per-raster stdout `tqdm` bar, then verifies read-only reuse and writes a JSON
  report. No production raster or index is modified. The Explore run is
  pending.
- Extended post-P1 raster discovery to support `.tif`, `.tiff`, `.nc`, and
  `.vrt` by default, plus multiple caller-supplied glob patterns with stable
  de-duplication. Preserved the scalar `image_glob` as a compatibility
  override. Added synthetic GDAL integration coverage for VRT and directly
  raster-readable NetCDF creation, validation, and reuse; subdataset-only
  NetCDF inputs receive guidance to select a variable through a VRT.
- The expanded dependency-free builder/preparation suite passes 34 tests with
  eight GDAL-dependent skips. The supported-container test rerun is pending.
- Real-data progress job 37938036 selected three static gravity GeoTIFFs and
  displayed genuine per-raster progress, then stopped on
  `outputNorthPole_20km.tif`. Its transformed densified perimeter crossed the
  longitude seam around the enclosed north pole and became self-intersecting;
  this was unrelated to `.nc` or `.vrt` discovery.
- Added a pole-aware footprint fallback: a source raster proven to contain
  `+90` or `-90` receives a valid, conservative full-longitude geographic cap
  in the index. Other rasters retain the canonical curved footprint. Added
  north/south GDAL regressions. The updated local suite passes 35 tests with
  nine GDAL-dependent skips; the container rerun is pending.
- Real-data rerun 37938046 reached 67 percent before the initial detector tried
  to transform an out-of-domain geographic pole into the source CRS. Replaced
  inverse-projection probing with transformed-ring longitude winding, which
  distinguishes an enclosed pole from a seam-only rectangle without making an
  out-of-domain transform. The local suite now passes 36 tests with nine
  GDAL-dependent skips; another container rerun is pending.
- Real-data rerun 37938048 passed in `lfm-container-ipyleaflet`. It indexed the
  ordinary, north-pole, and south-pole gravity GeoTIFFs, completed the stdout
  `tqdm` bar, created and validated a three-feature GeoPackage, and reused it
  with identical structured metadata. This closes the polar-cap and real-data
  progress validation; GDAL-backed `.nc` and `.vrt` tests remain pending.
- Explore job 37938049 passed all 36 focused builder and preparation tests with
  GDAL enabled. The run includes directly readable NetCDF and VRT
  creation/validation/reuse and both pole-cap regressions. The post-P1 raster
  format extension is fully validated for its stated scope.
- Began Phase P2 and added a final notebook sub-phase plus a supported-container
  regression gate. `TileSourceConfig` now declares a callable product resolver,
  defaulting to the lunar filename prefix before the first period; custom
  modalities may supply their own resolver.
- Added `create_tiles_for_aoi_by_product()` as the high-level optional-PID AOI
  path while preserving strict selector requirements in the existing low-level
  `create_tiles_for_*` APIs. Exact strings select one PID; `None` or an omitted
  entry discovers sorted intersecting PIDs, groups companion rasters, tiles
  each dynamic product separately, and processes contextual sources once.
- Added typed required-product failures, product IDs on structured source
  errors, safe-filename collision checks, deterministic grid/tile/source/PID
  ordering, and completed-record propagation for partial failures.
- Updated notebook plotting to pair several dynamic products on one tile with
  one static record and include PID in panel titles. Updated the public notebook
  so both PID variables accept strings or `None`; Markdown now explains
  discovery, companion grouping, separate outputs, ordering, static reuse,
  missing-product behavior, and expert-query PID reuse.
- Added P2 tests for explicit, omitted, single, multiple, missing, optional,
  malformed, custom-resolver, collision, ordering, companion-file, static-once,
  visualization-pairing, and product-specific error cases. The expanded local
  modern suite passes 91 tests with 14 environment-dependent skips. Notebook
  JSON, unique IDs, clean outputs, Python-cell syntax, shell syntax, Python
  compilation, and whitespace checks pass.
- Explore job 37938065 passed all 91 modern tiling contract tests, all 25 safe
  legacy regression tests, and the filtered one-tile legacy integration test
  under GDAL 3.8.4 in `lfm-container-ipyleaflet`. The job completed in 35
  seconds. Apptainer emitted FUSE cleanup messages after successful child
  processes, but every test command returned successfully and the wrapper
  reached `All tiling modernization checks passed.` This completes Phase P2.
- Began Phase P3. Added an immutable `GridRegistry` that validates all 92
  repository definitions and exposes explicit grid family, CRS, geographic
  coverage, and typed tile-matrix metadata.
- Added point routing with inclusive `+/-82` polar ownership, canonical
  longitudes at the poles, deterministic LTM boundary ownership, and stable
  public `LPS_N`/`LPS_S` identifiers.
- Added AOI validation and routing across polar thresholds, the equator, every
  numbered LTM longitude band, and the antimeridian. Query parts always have
  positive area and non-wrapping longitude bounds; duplicate parts are removed.
- Added 23 registry/router tests covering inventory, metadata, both poles,
  threshold crossing and boundary-only touches, every LTM longitude boundary,
  seam splitting, full polar caps, invalid spans, stable order, and invalid
  coordinates. The expanded local modern suite passes 114 tests with 14
  unchanged environment-dependent skips.
- Explore job 37938083 passed all 114 modern tiling contract tests, all 25 safe
  legacy regression tests, and the filtered one-tile legacy integration test
  under GDAL 3.8.4 in `lfm-container-ipyleaflet`. The wrapper completed in 11
  seconds. Apptainer FUSE cleanup messages occurred after successful commands
  and did not affect the result. This completes Phase P3.
- Began Phase P4. Added shared grid-neutral tile-matrix arithmetic and retained
  `TmsTileDef` as the numbered-LTM projection/overlap implementation.
- Added `PolarTileDef` for `LPS_N` and `LPS_S`, including traditional GIS axis
  order, densified geographic AOI projection, circular full-cap geometry,
  clipped row/column discovery, meaningful-overlap filtering, explicit address
  validation, and conservative seam-safe tile envelopes for source indexes.
- Connected configured AOI tiling to P3 routing parts and path-deduplicated
  multi-envelope source-index queries. Added explicit `grid_id` API aliases,
  record/error aliases, polar filenames, and unchanged LTM filename behavior.
- Added shared-matrix, polar numerical, seam, pole, invalid-address, output CRS,
  exact geotransform, filename, API, and envelope-deduplication tests. The
  expanded local modern suite passes 134 tests with 23 environment-dependent
  skips.
- Added a read-only Slurm diagnostic for finding real polar test candidates
  beneath the project `data`, `processed_data`, and `rawdata` trees. It uses
  one GDAL process per requested Slurm CPU, the canonical densified footprint
  implementation, deterministic optional sampling, stdout `tqdm` progress,
  ranked directory summaries, candidate raster paths, suggested polar points,
  error capture, and a JSON report. A local inventory-only dry run passed; no
  Explore rasters were opened locally.
- Inspected the first P4 container result. All preceding modern checks passed,
  including the polar cube's dimensions and exact geotransform; the sole
  failure was a direct CRS `IsSame()` comparison before GDAL axis strategies
  were normalized. Updated that regression to compare both CRSs using
  traditional GIS axis order, as the downstream cube reader already does, and
  to print both WKT definitions if a substantive mismatch remains.
- Inspected the second P4 container result. Its emitted WKT showed GDAL 3.8's
  GeoTIFF reconstruction drops the custom IAU/USGSLGS identifiers and emits
  projection-native north/north polar axes, while retaining the Moon radius,
  units, stereographic method, scale, origin, central meridian, and false
  offsets. Added a projected-raster CRS equivalence fallback that compares the
  complete PROJ.4 operation after strict OSR comparison fails and deliberately
  ignores only axis/runtime CRS metadata. Connected the downstream chip cube
  check to the same helper so a valid polar cube will not pass tiling and then
  fail acquisition. Added positive axis-loss and negative changed-projection
  tests. The expanded local modern suite passes 136 tests with 23 GDAL skips.
- Explore job 37938093 passed all 136 modern tiling contract tests, all 25 safe
  legacy regression tests, and the filtered one-tile legacy integration test
  under GDAL 3.8.4 in `lfm-container-ipyleaflet`. The wrapper reached its
  success terminus in 9 seconds. Apptainer FUSE cleanup messages occurred only
  after successful child processes. This completes Phase P4.
- Began Phase P5. Added `compose_tile_sources()` with boolean
  `include_dynamic`/`include_static` controls that default to combined mode,
  reject disabling both classes, require a source for each enabled class, and
  preserve dynamic-before-static ordering.
- Disabled source collections are not iterated, so they require no path,
  index, selector, preparation, validation, query, or output. Enabled static
  sources must be contextual `all_intersecting` sources. Required/optional
  flags remain attached to the original source objects.
- Added source-mode regressions for combined, dynamic-only, and static-only
  LTM, `LPS_N`, and `LPS_S` results; invalid controls; missing enabled classes;
  duplicate names; skipped disabled validation; static PID rejection;
  deterministic source order; and preservation of completed records across a
  later static failure. The combined source-mode and preparation check passes
  44 tests with two GDAL-dependent skips. The expanded modern suite passes 156
  tests with 25 GDAL-dependent skips; the legacy subset awaits the supported
  container because GDAL is unavailable locally.
- Updated the GDAL-backed preparation immutability regression to use P4's
  grid-neutral tile-definition factory and added the preparation test module to
  the main modernization wrapper. The container gate will now directly verify
  that disabled data directories and indexes remain untouched.
- Added a focused real-data north-polar WAC validation driver and `grace`
  Slurm wrapper for
  `WAC_GLOBAL_P900N0000_100M.eqc.iau2.LPS_N.vrt`. The focused Explore run uses
  dynamic-only mode at 86 degrees north, builds an isolated one-file index,
  creates one `LPS_N` zoom-4 cube, and reopens it to validate tile geometry,
  CRS, native NoData, valid pixels, LZW compression, and permissions. This is
  exploratory evidence for later P9 and does not advance the sequential phase
  status while P5.6 remains open.
- Explore job 37938102 completed that focused check in three seconds: it built
  and validated the isolated one-feature index, wrote the expected `LPS_N`
  zoom-4 tile `(8, 11)`, and passed the structural cube checks with all 262,144
  pixels valid. Visual inspection raised a possible distortion question that
  the original structural check could not resolve.
- Extended the diagnostic with an equal-aspect three-panel PNG showing the
  native-resolution source VRT window, the written tile with a shared robust
  stretch, and absolute error against a fresh bilinear GDAL warp on the exact
  tile grid. The JSON report now includes native-window geometry, valid-mask
  mismatches, mean absolute error, RMSE, maximum error, and the 99.5th error
  percentile. Local syntax and helper checks pass; the comparison plot awaits
  an Explore run.

Next:

- Run the expanded modernization wrapper in `lfm-container-ipyleaflet` to
  complete P5.6 and close Phase P5.

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
  addressing, and the unsupported `LPS_N`/`LPS_S` production path in
  `TMS/README.md`.
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
