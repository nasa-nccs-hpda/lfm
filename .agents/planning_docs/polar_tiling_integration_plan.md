# Polar Tiling Integration Plan

This document is the sequential implementation plan for adding Armstrong polar
tiling to the modern LFM tiling workflow. It also covers two related usability
changes required by the public workflow: automatic source-index preparation and
optional product-ID discovery.

This is a planning document. The authoritative description of the currently
supported production behavior remains:

- [`docs/tiling_modernization_plan.md`](../../docs/tiling_modernization_plan.md)
- [`TMS/README.md`](../../TMS/README.md)
- [`notebooks/tiling_example.ipynb`](../../notebooks/tiling_example.ipynb)

The prospective contract frozen in Phase P0 is recorded in
[`polar_tiling_contract.md`](polar_tiling_contract.md).

## Status convention

- `[Not Started]`: no implementation work has begun.
- `[In-Progress]`: active work; only one sub-step should have this status.
- `[Complete]`: implemented, tested, and documented for the stated scope.
- `[Deferred]`: intentionally removed from the current implementation sequence;
  the reason and conditions for resuming it must be recorded.

Phases and their sub-steps are strictly sequential. Start a sub-step only after
the preceding sub-step is `[Complete]`. Start a phase only after every required
sub-step in the preceding phase is `[Complete]`. A phase becomes `[Complete]`
only when all of its required sub-steps are complete. A deliberately deferred
item does not block later work once its deferral is documented and accepted.

## Accepted design decisions

- Public polar grid identifiers are `LPS_N` and `LPS_S`.
- Locations at latitude `>= 82` degrees use `LPS_N`; locations at latitude
  `<= -82` degrees use `LPS_S`; locations strictly between those limits use a
  numbered LTM zone.
- The normal user workflow does not require a grid or zone. Users provide
  enabled data directories, a point or AOI, and an optional product ID. Grid
  discovery and routing are automatic.
- An AOI crossing a `+/-82` degree boundary is split automatically and routed
  to the applicable LTM and polar grids. Complete output tiles may extend
  across the routing boundary because tile products retain their full native
  grid footprint.
- Explicit tile-address requests remain an expert workflow and must specify a
  complete grid ID, zoom, tile column, and tile row.
- The default source mode is dynamic plus static. Users can explicitly disable
  either source class. Initial polar examples and validation may run
  dynamic-only while polar static coverage is unavailable or unverified.
- Product IDs are optional in the easy workflow. Omitting a product ID means
  discovering intersecting products and producing separate product-scoped
  outputs; it must not silently stack unrelated dynamic products into one cube.
- A missing raster index is created and validated during workflow preparation.
  The lower-level tile-generation path continues to query indexes read-only.
- Tiling continues to use bilinear resampling, 512 by 512 tiles, deterministic
  record ordering, structured results, and the existing per-source NoData
  contracts.

## Phase P0 — Freeze the extended public contract `[Complete]`

- `[Complete]` **P0.1** Inventory every LTM-specific assumption in the public
  API, TMS geometry classes, result records, filenames, notebook helpers,
  plotting helpers, tests, and documentation.
- `[Complete]` **P0.2** Define the minimal user request: configured source
  directories, exactly one point or AOI, and an optional product ID. Establish
  derived defaults for index paths, enabled sources, grid selection, zoom, run
  ID, and output directory.
- `[Complete]` **P0.3** Define the expert explicit-tile request separately so
  it retains a complete caller-supplied grid address without complicating the
  easy workflow.
- `[Complete]` **P0.4** Define exact automatic-routing behavior for points,
  AOIs wholly inside one grid family, AOIs crossing `+/-82` degrees, AOIs
  crossing the longitude seam, and AOIs containing a pole.
- `[Complete]` **P0.5** Define the optional-product contract: a supplied PID
  selects one observation; an omitted PID discovers intersecting PIDs, groups
  all files belonging to each PID, and creates one product-scoped cube per
  grid tile and PID.
- `[Complete]` **P0.6** Define source-mode behavior for dynamic plus static,
  dynamic-only, and static-only runs. Disabled sources must not be indexed,
  validated, queried, or plotted.
- `[Complete]` **P0.7** Define grid-neutral result metadata and filenames,
  including backward compatibility for the current `zone` field and LTM
  filenames. Specify unambiguous `LPS_N` and `LPS_S` output names.
- `[Complete]` **P0.8** Record acceptance cases and expected outputs before
  implementation: ordinary LTM, north polar, south polar, exact-threshold,
  boundary-crossing, longitude-seam, no-PID multi-product, and each source
  inclusion mode.

P0 completion evidence: `.agents/planning_docs/polar_tiling_contract.md`
records the repository audit, minimal and expert workflows, derived index and
zoom defaults, exact routing and product-discovery semantics, source modes,
backward-compatible structured results, filenames, deterministic ordering, and
23 acceptance cases. This phase changes no runtime behavior; production remains
numbered-LTM-only until the implementation and validation phases complete.

## Phase P1 — Add safe automatic raster-index preparation `[In-Progress]`

- `[Complete]` **P1.1** Extend the existing vector-index builder with an
  explicit ensure-if-missing API and configuration for the data directory,
  derived or explicit index path, raster glob, layer name, location field, and
  output geographic CRS.
- `[Complete]` **P1.2** Define reuse and freshness rules. Existing indexes are
  validated and reused; they are never silently overwritten. A stale or
  malformed index produces a clear diagnosis and explicit rebuild guidance.
- `[Complete]` **P1.3** Enumerate input rasters deterministically and log the
  number found, the destination index, and a clear notice that indexing a large
  lunar directory may take several minutes.
- `[Complete]` **P1.4** Probe `gdal.TileIndex` inside the supported
  `lfm-container-ipyleaflet` image to determine whether its installed GDAL
  binding exposes a usable per-raster progress callback.
- `[Complete]` **P1.5** Add a `tqdm` progress bar written to stdout. Prefer a
  GDAL progress callback when it is reliable; otherwise provide deterministic
  discovery/preflight progress and evaluate a one-raster-at-a-time Python/OGR
  writer for meaningful creation progress.
- `[Complete]` **P1.6** Make index publication failure-safe. Build all
  Shapefile sidecars in a temporary sibling location, validate them, and move
  the complete set into place only after success. Add a scoped lock or other
  guard against two jobs creating the same index concurrently.
- `[Complete]` **P1.7** Validate the completed index: driver, CRS, layer,
  location field, feature count, resolvable raster paths, valid geometries, and
  expected inventory. Clean up staging artifacts after a failure.
- `[Complete]` **P1.8** If a custom OGR writer is required, densify raster
  perimeters before transformation so polar and longitude-seam footprints are
  not reduced incorrectly to four transformed corners.
- `[Complete]` **P1.9** Compare generated indexes with GDAL TileIndex using
  semantic equivalence rather than binary equality: CRS, feature count,
  location values, deterministic order, geometry validity, bounds and area
  tolerances, and AOI query results. Include ordinary LTM, polar, and
  longitude-seam fixtures.
- `[In-Progress]` **P1.10** Integrate index preparation into the high-level
  workflow before source validation. Keep the lower-level tile writer and
  vector-index query APIs read-only, and skip preparation entirely for disabled
  source classes.
- `[Not Started]` **P1.11** Add focused tests for creation, stdout progress,
  logging, reuse, stale-index rejection, explicit rebuild behavior, concurrent
  creation, failure cleanup, Shapefile sidecars, GeoPackage output, and source
  index immutability during tile generation.

P1.1 checkpoint: `VectorIndexBuildConfig.index_path` now defaults to
`<data_dir>/output_index.shp` while preserving explicit Shapefile, GeoPackage,
layer, location-field, raster-glob, and output-WKT configuration.
`discover_raster_paths()` supplies a deterministic file inventory, and
`ensure_vector_index()` is the explicit preparation entry point. The low-level
tiler and `query_source_index()` remain read-only.

P1.2 checkpoint: validation and reuse code now checks the index driver, layer,
location field, repository lunar CRS, feature geometries, resolvable raster
paths, duplicate paths, and exact current raster inventory. Missing or changed
inventory raises `StaleVectorIndexError` with explicit rebuild guidance rather
than overwriting the index. Dependency-free and supported-container tests cover
the ensure/reuse path plus Shapefile and GeoPackage creation, reuse, and stale
inventory rejection. The focused wrapper also records whether the supported
GDAL build supplies useful `gdal.TileIndex` callback events:
`scripts/shell/all_tasks/sbatch_probe_gdal_tileindex_progress.sh`.

The first probe attempt, Explore job 37937983, established that the supported
container's `osgeo.gdal` module has no `TileIndex` attribute; the probe stopped
before its focused tests. Index creation has therefore been switched to a
Python/OGR feature writer, which can report genuine per-raster tqdm progress
without shelling out. The repaired probe now records absent Python APIs and the
`gdaltindex` executable/version without failing. At that checkpoint, a rerun
was still required before P1.2 could close.

The second attempt, job 37937987, completed the capability report and created
the synthetic indexes, but two Shapefile validations failed because its `.prj`
file downgraded the repository's modern IAU WKT to WKT1. That representation
drops authority, usage, and datum metadata, causing `OSR.IsSame()` to reject an
otherwise unchanged lunar geographic coordinate space. Validation now retains
exact `IsSame()` checks for GeoPackage and first attempts them for every format;
only a Shapefile geographic fallback compares the persisted semi-major and
semi-minor axes, inverse flattening, prime meridian, and angular units with
strict tolerances. That fix then required another container rerun.

The third attempt, job 37937988, reached that fallback but exposed a binding
compatibility detail: GDAL 3.8 does not provide the Python
`SpatialReference.GetPrimeMeridian()` convenience method. The comparison now
reads the numeric `PRIMEM` WKT child through the long-standing
`GetAttrValue("PRIMEM", 1)` API. A dependency-free compatibility regression
covers an object with the GDAL 3.8 method surface.

Explore job 37938002 completed successfully: all 12 focused GDAL-backed tests
passed in `lfm-container-ipyleaflet`. This closes P1.2 and confirms creation,
validation, reuse, and stale-inventory rejection for both Shapefile and
GeoPackage output. The capability report also confirms that this supported
GDAL binding exposes neither `gdal.TileIndex` nor `gdal.TileIndexOptions`, so
P1.4 is complete. Deterministic discovery, visible count/destination/timing
messages, and the direct Python/OGR writer's per-raster stdout `tqdm` progress
close P1.3 and P1.5.

P1.6 implementation checkpoint: index creation now holds an exclusive sibling
lock, writes into a temporary sibling directory, validates the staged index
against the exact raster inventory, and publishes all generated artifacts only
after validation. Shapefile sidecars are moved before the primary `.shp` file,
and scoped cleanup removes staging files and the lock after creation or
failure. Dependency-free tests cover complete sidecar publication, concurrent
creation rejection, writer failure cleanup, and validation failure cleanup.
The maintainer-reported supported-container rerun passed all tests, closing
P1.6. The same staged validation plus final `ensure_vector_index()` validation
covers driver, CRS, schema, feature inventory, paths, and geometry; cleanup is
exercised for both writer and staged-validation failures, closing P1.7.

P1.8 implementation checkpoint: raster footprints now use 21 samples per edge
in pixel space before applying the source-to-IAU:30100 transformation, rather
than transforming only four corners. Pure tests cover deterministic perimeter
ordering, corner deduplication, and invalid sampling, and a GDAL integration
test uses the repository's `LPS_N` CRS to verify that a curved polar edge is
retained with the expected densified vertex count. The supported-container
rerun, Explore job 37938014, passed and closes P1.8. Its capability report also
reconfirmed GDAL 3.8.4, `/usr/bin/gdaltindex`, and the absence of both Python
TileIndex APIs and callback support.

P1.9 implementation checkpoint: a new isolated comparison harness builds
ordinary `LTM_1N`, `LPS_N`, and paired east/west longitude-seam fixtures. It
creates one index with the LFM OGR writer and one with `/usr/bin/gdaltindex`,
then writes a JSON report comparing CRS, feature counts, location values,
record order, geometry validity, vertex counts, bounds, area, and positive and
negative AOI query results. Comparisons use an explicit 0.02-degree bounds
tolerance and a two-percent relative-area tolerance where GDAL's geometry is an
acceptance oracle. The dedicated
Grace/`lfm-container-ipyleaflet` wrapper is
`scripts/shell/all_tasks/sbatch_compare_vector_index_builders.sh`. A supported
container run was required before the tolerances or P1.9 completion could be
accepted.

P1.9 first-run checkpoint: Explore job 37938028 passed every LTM and
longitude-seam comparison. Those fixtures had identical bounds, matching AOI
results, and effectively identical areas despite the LFM indexes retaining 81
vertices versus GDAL's five. The polar fixture also matched CRS, driver,
inventory, order, validity, bounds, and every AOI query, but exceeded the
provisional two-percent planar-area threshold: LFM area `137.4897511` versus
GDAL area `124.4154901`, a 9.509-percent relative difference. This is the
expected consequence of retaining transformed curved edges instead of GDAL's
four straight chords, not evidence of an incorrect bound or query result.

Maintainer decision: the densified curved footprint is canonical. The revised
gate retains the two-percent GDAL area tolerance for ordinary LTM and seam
fixtures. Polar GDAL area remains reported but is explicitly not an acceptance
gate because its five-vertex chord geometry is the less accurate reference.
Polar accuracy instead requires the production 21-sample footprint to converge
to a higher-resolution 201-sample footprint within 0.001-degree bounds and
0.1-percent relative area. P1.9 remained in progress until this canonical
convergence check passed in the supported container.

Explore job 37938033 passed the revised canonical-curvature comparison. P1.9
is complete: LTM, polar, and seam fixtures satisfy their structural, inventory,
ordering, validity, bounds, AOI-query, and applicable area gates, and the
21-sample polar perimeter converges to the 201-sample reference.

P1.10 implementation checkpoint: `TileSourcePreparation` now derives an index
build contract from a prospective `TileSourceConfig`, and
`prepare_tile_config()` ensures each enabled source index in caller order before
assembling the low-level `TileConfig`. It returns both the config and structured
index-validation results. Disabled preparations are filtered before raster
discovery, index creation, or validation; an all-disabled or duplicate-enabled
request fails before filesystem mutation. The existing `create_tiles_*`,
`ConfiguredTiler`, and `query_source_index()` paths remain unchanged and
read-only with respect to indexes. Dependency-free tests cover derivation,
ordering, disabled sources, and pre-mutation errors. A GDAL integration test
creates a missing enabled GeoPackage while proving that a disabled nonexistent
source directory remains untouched. The supported-container run remains
required before P1.10 is complete.

## Phase P2 — Restore optional product-ID discovery `[Not Started]`

- `[Not Started]` **P2.1** Define modality-aware product-ID extraction without
  hard-coding WAC or NAC behavior into the generic tiler. Product resolution
  must group companion files, such as WAC UV and VIS inputs, under one PID.
- `[Not Started]` **P2.2** Preserve the current strict low-level
  `product_id` selector for explicit product requests while adding a
  high-level discovery path for an omitted selector.
- `[Not Started]` **P2.3** Query intersecting dynamic index records, derive and
  sort unique product IDs deterministically, and log the discovered count and
  identifiers.
- `[Not Started]` **P2.4** Invoke product-scoped tiling separately for each
  discovered PID so unrelated observations are never combined into one dynamic
  cube. Keep `all_intersecting` for contextual/static sources and explicit
  expert use.
- `[Not Started]` **P2.5** Ensure every resulting dynamic record and filename
  retains its resolved product ID, including partial-coverage and missing-source
  error paths.
- `[Not Started]` **P2.6** Add tests for explicit PID, omitted PID with one and
  several matches, no matches, WAC companion-band grouping, NAC products,
  deterministic ordering, malformed names, and product-specific failures.

## Phase P3 — Introduce a grid registry and automatic router `[Not Started]`

- `[Not Started]` **P3.1** Add a grid-neutral registry that loads the 90 LTM
  definitions plus the repository polar definitions and exposes canonical IDs,
  geographic coverage, CRS, and tile matrices.
- `[Not Started]` **P3.2** Represent numbered LTM, `LPS_N`, and `LPS_S` as
  explicit grid families instead of inferring all behavior from an LTM zone
  name pattern.
- `[Not Started]` **P3.3** Implement point routing with an inclusive 82-degree
  polar threshold and deterministic LTM longitude-zone discovery below it.
- `[Not Started]` **P3.4** Implement AOI partitioning at `+82` and `-82` degrees,
  then route each nonempty part to its applicable grid family without duplicate
  grid/tile requests.
- `[Not Started]` **P3.5** Normalize lunar longitudes and handle antimeridian
  AOIs, pole-containing AOIs, and exact-boundary inputs without invalid or
  world-spanning polygons.
- `[Not Started]` **P3.6** Add router tests covering both poles, every threshold
  edge, LTM longitude edges, the longitude seam, multi-grid AOIs, stable order,
  and invalid geographic inputs.

## Phase P4 — Implement polar tile geometry and addressing `[Not Started]`

- `[Not Started]` **P4.1** Parse the north and south polar TMS JSON structures,
  including their projection definitions, zoom matrices, origins, cell sizes,
  matrix dimensions, and valid tile ranges.
- `[Not Started]` **P4.2** Refactor shared tile-matrix calculations away from
  LTM-only assumptions while preserving the proven LTM implementation and
  numerical behavior.
- `[Not Started]` **P4.3** Transform routed geographic point/AOI geometry into
  each polar stereographic grid using traditional GIS axis order and sufficient
  edge densification for curved geographic boundaries.
- `[Not Started]` **P4.4** Resolve intersecting polar tile rows and columns,
  apply the existing meaningful-overlap policy where applicable, clip indexes
  to valid matrix limits, and deduplicate results deterministically.
- `[Not Started]` **P4.5** Extend explicit tile-index validation and structured
  records to accept complete `LPS_N` and `LPS_S` addresses.
- `[Not Started]` **P4.6** Write grid-neutral cube filenames and metadata while
  retaining unchanged LTM filenames for backward compatibility.
- `[Not Started]` **P4.7** Add numerical tests for transforms, projected bounds,
  exact 512 by 512 geotransforms, matrix edges, both poles, seam behavior,
  output CRS, and invalid polar addresses.

## Phase P5 — Add configurable dynamic/static source modes `[Not Started]`

- `[Not Started]` **P5.1** Add high-level `include_dynamic` and
  `include_static` controls, both defaulting to `True`, and reject requests
  that disable both classes.
- `[Not Started]` **P5.2** Compose the ordered `TileSourceConfig` collection only
  from enabled sources. Preserve dynamic-before-static ordering for combined
  runs.
- `[Not Started]` **P5.3** Ensure disabled sources require no path, index,
  selector, validation, or output and cannot cause an unrelated run to fail.
- `[Not Started]` **P5.4** Preserve required/optional semantics within each
  enabled source class, including structured partial-result errors.
- `[Not Started]` **P5.5** Confirm that static-only operation never expects a
  product ID and that dynamic-only polar operation is a supported initial
  production path.
- `[Not Started]` **P5.6** Add tests for all valid source modes on LTM and polar
  grids, source ordering, skipped validation, missing enabled sources, and
  deterministic results.

## Phase P6 — Assemble the easy public workflow `[Not Started]`

- `[Not Started]` **P6.1** Add a small orchestration API that accepts prepared
  modality/source definitions, a point or AOI, an optional product ID, source
  inclusion controls, and output settings without requiring a zone or grid.
- `[Not Started]` **P6.2** Run its stages in an explicit order: validate request,
  prepare enabled indexes, resolve products, route geometry, construct one or
  more `TileConfig` objects, generate tiles, and return ordered structured
  records.
- `[Not Started]` **P6.3** Select modality-appropriate default zooms per grid
  family while allowing expert overrides. Document the chosen WAC and NAC polar
  zooms and their relationship to source resolution.
- `[Not Started]` **P6.4** Ensure multi-grid and multi-product output remains
  deterministic and collision-free and that errors identify the product,
  source, grid, zoom, and tile involved.
- `[Not Started]` **P6.5** Preserve the existing lower-level AOI, point, and
  explicit-index entry points for advanced callers and downstream compatibility.
- `[Not Started]` **P6.6** Add end-to-end local tests for minimal LTM and polar
  requests, automatic indexing, omitted PIDs, boundary splitting, each source
  mode, and repeatability.

## Phase P7 — Update the public tiling notebook `[Not Started]`

- `[Not Started]` **P7.1** Keep the main user configuration limited to data
  directories, WAC/NAC AOIs or points, optional product IDs, and clearly
  explained source-inclusion controls.
- `[Not Started]` **P7.2** Move derived index paths, automatic grid selection,
  zoom defaults, run IDs, output directories, and other implementation details
  into the setup section.
- `[Not Started]` **P7.3** Replace validate-existing-index-first behavior with
  prepare-if-missing followed by validation, including visible stdout progress
  and clear reuse messages.
- `[Not Started]` **P7.4** Add an opt-in polar dynamic-only example to the
  alternate queries section. The example must supply no polar grid identifier
  and demonstrate automatic routing.
- `[Not Started]` **P7.5** Restore an opt-in AOI query with no product ID using
  the new per-product discovery behavior, replacing the current stacked
  `all_intersecting` demonstration.
- `[Not Started]` **P7.6** Update plotting to support dynamic plus static,
  dynamic-only, and static-only records across multiple grids and products,
  while retaining NaN masking and the four-tile display limit.
- `[Not Started]` **P7.7** Explain the 82-degree routing threshold, full-tile
  footprints, product discovery, index creation time, output locations, and
  expert overrides in notebook Markdown and Python comments.
- `[Not Started]` **P7.8** Validate notebook JSON, unique cell IDs, Python-cell
  syntax, empty committed outputs, and a command-line execution using the
  supported container when representative data is available.

## Phase P8 — Run local regression and contract validation `[Not Started]`

- `[Not Started]` **P8.1** Run the existing LTM unit and regression suite and
  prove unchanged covered LTM metadata, pixels, NoData, filenames, ordering,
  and deterministic hashes where byte identity is expected.
- `[Not Started]` **P8.2** Validate index creation against GDAL TileIndex on
  representative LTM data and validate densified footprint correctness on
  polar and longitude-seam fixtures.
- `[Not Started]` **P8.3** Validate north and south polar routing, transforms,
  tile selection, output CRS/geotransform, filenames, records, and bilinear
  raster values.
- `[Not Started]` **P8.4** Validate exact-threshold and cross-threshold AOIs,
  documenting that selected full tiles may extend beyond the routing boundary.
- `[Not Started]` **P8.5** Validate explicit and omitted product IDs with one and
  multiple WAC/NAC observations, including companion-band grouping.
- `[Not Started]` **P8.6** Validate all three source modes and confirm disabled
  source directories and indexes are never touched.
- `[Not Started]` **P8.7** Repeat representative runs and verify record order,
  metadata, output hashes, index reuse, failure cleanup, and absence of
  unintended source-index modifications.

## Phase P9 — Validate on Explore lunar data `[Not Started]`

- `[Not Started]` **P9.1** Add or update a Slurm validation wrapper using the
  `grace` partition, the `lfm-container-ipyleaflet` image, and the established
  `/panfs/ccds02/nobackup` to `/explore/nobackup` bind mapping.
- `[Not Started]` **P9.2** Measure automatic index creation and reuse on a
  representative large source directory, confirming stdout progress, log
  clarity, feature inventory, and query equivalence with GDAL TileIndex.
- `[Not Started]` **P9.3** Run north-polar and south-polar dynamic examples with
  known WAC or NAC coverage and inspect structured records and reopened cubes.
- `[Not Started]` **P9.4** Run exact-threshold, cross-threshold, and
  longitude-seam cases and inspect both the selected grid/tile inventory and
  plotted raster coverage.
- `[Not Started]` **P9.5** Run an omitted-PID AOI containing multiple products
  and confirm separate, correctly grouped outputs with no filename collisions.
- `[Not Started]` **P9.6** Run LTM control cases through the new easy workflow
  and compare them with the accepted modern LTM results.
- `[Not Started]` **P9.7** Record job IDs, reports, accepted tolerances,
  performance observations, and any explicit exceptions without claiming tests
  that were not executed.

## Phase P10 — Align documentation and close the project `[Not Started]`

- `[Not Started]` **P10.1** Update `TMS/README.md` from its current unsupported
  polar description to the implemented `LPS_N`/`LPS_S` routing, geometry,
  zoom, addressing, and boundary behavior.
- `[Not Started]` **P10.2** Update `docs/tiling_modernization_plan.md` with the
  new stable contract for automatic index preparation, optional products,
  grid-neutral results, source modes, and polar support.
- `[Not Started]` **P10.3** Update the chip-creation modernization plan wherever
  its upstream tiling assumptions change, without expanding this project into
  downstream chip implementation.
- `[Not Started]` **P10.4** Update the lunar-tiling agent skill so future work
  follows the new supported contract rather than the prior LTM-only and
  existing-index-only rules.
- `[Not Started]` **P10.5** Add dated implementation and validation evidence to
  `.agents/progress_docs/tiling.md`.
- `[Not Started]` **P10.6** Confirm all public examples, doc links, source names,
  grid identifiers, statuses, and deferred items are consistent.
- `[Not Started]` **P10.7** Mark this plan complete only after local regression,
  Explore validation, notebook validation, and maintainer acceptance have all
  been recorded.

## Deferred scope

- `[Deferred]` **D1** Canonical polar static-data production. Resume after polar
  coverage, source footprints, band availability, NoData behavior, and expected
  zooms are verified. The implementation must continue supporting static-only
  and combined source modes when suitable polar static inputs become available.
- `[Deferred]` **D2** Automatic destructive refresh of stale indexes. The
  planned workflow diagnoses staleness and requires an explicit rebuild action
  so it cannot unexpectedly replace a shared index.
- `[Deferred]` **D3** Migration of downstream chip creation and model-training
  consumers to polar grids. This plan preserves and documents their upstream
  tiling handoff but does not implement downstream polar processing.
