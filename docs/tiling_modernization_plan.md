# Tiling Modernization Plan

## Objective

Replace the WAC-specific, partially hard-coded tiling interface with a
configuration-driven interface that can create LTM datacubes for any declared
lunar raster modality. Each source modality will declare its own raster
directory, vector index, selection behavior, bands, resampling method, and
NoData policy.

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

## Phase T0 — Preserve and define the existing contract `[Planned]`

- `[Planned]` **T0.1** Inventory the behavior of `Pipeline`,
  `TmsIntersector`, `TmsZoneDef`, and `TmsTileDef`, including tile dimensions,
  filenames, band metadata, product filtering, permissions, and NoData output.
- `[Planned]` **T0.2** Record the current public entry points for tile-index,
  point, and geographic-AOI queries and identify which behaviors require a
  temporary compatibility adapter.
- `[Planned]` **T0.3** Define the tiling boundary: inputs are a query, a
  `TileConfig`, and optional per-query selectors; outputs are structured cube
  records plus files on disk.
- `[Planned]` **T0.4** Define acceptance examples for WAC-only, NAC-only, and
  WAC-plus-static tiling before changing implementation code.

## Phase T1 — Introduce tiling configuration objects `[Planned]`

- `[Planned]` **T1.1** Add a `TileSourceConfig` dataclass containing the source
  name, data directory, vector-index path, optional layer name, raster-location
  field, and selection mode.
- `[Planned]` **T1.2** Add per-source band selection, resampling, source NoData,
  output NoData, and any required per-band override fields.
- `[Planned]` **T1.3** Add a `TileConfig` dataclass containing the ordered source
  configurations, output directory, zoom level, and debug settings.
- `[Planned]` **T1.4** Add a small dictionary-to-config constructor so notebooks
  can express source definitions as plain Python while backend functions
  receive validated config objects.
- `[Planned]` **T1.5** Add focused tests for valid configs, missing required
  fields, duplicate source names, unsupported selection modes, and invalid
  resampling or NoData settings.

## Phase T2 — Generalize vector-index access `[Planned]`

- `[Planned]` **T2.1** Replace the explicitly selected ESRI Shapefile driver
  with format-independent GDAL/OGR dataset opening.
- `[Planned]` **T2.2** Support both `.shp` and `.gpkg` indexes, including an
  optional configured GeoPackage layer and a configurable raster-location
  field.
- `[Planned]` **T2.3** Resolve relative raster paths against the configured data
  directory while preserving support for absolute paths stored in an index.
- `[Planned]` **T2.4** Separate index creation or refresh from tile generation;
  tiling will consume an existing declared index and will not silently rebuild
  it.
- `[Planned]` **T2.5** Add equivalent query tests using minimal Shapefile and
  GeoPackage fixtures.

## Phase T3 — Replace static/dynamic branching with source policies `[Planned]`

- `[Planned]` **T3.1** Replace the hard-coded WAC database and static database
  with ordered iteration over `TileConfig.sources`.
- `[Planned]` **T3.2** Implement `product_id` selection for product-scoped
  modalities such as WAC and NAC.
- `[Planned]` **T3.3** Implement `all_intersecting` selection for contextual
  modalities such as static lunar layers.
- `[Planned]` **T3.4** Pass per-query product IDs or equivalent selectors by
  source name rather than through the pipeline constructor.
- `[Planned]` **T3.5** Apply each source's configured bands, resampling method,
  and NoData policy while warping source rasters to the LTM tile grid.
- `[Planned]` **T3.6** Preserve ordered, meaningful band metadata independently
  for every source modality.
- `[Planned]` **T3.7** Verify WAC-only, NAC-only, static-only, and mixed-source
  behavior with focused tests.

## Phase T4 — Introduce structured tiling results `[Planned]`

- `[Planned]` **T4.1** Add a `TileCubeRecord` containing source name, LTM zone,
  zoom level, tile coordinates, optional product ID, output path, band names,
  CRS, and NoData information.
- `[Planned]` **T4.2** Make tile-index, point, and AOI queries return ordered
  `TileCubeRecord` collections instead of requiring callers to parse filenames.
- `[Planned]` **T4.3** Define generic, deterministic filenames that retain zone,
  zoom, tile, source, and optional product identity for human inspection.
- `[Planned]` **T4.4** Ensure an empty intersection, skipped raster, partial
  source result, and fatal source failure have explicit result or exception
  behavior.
- `[Planned]` **T4.5** Add tests proving that structured records and written
  raster metadata agree.

## Phase T5 — Stabilize the public tiling API `[Planned]`

- `[Planned]` **T5.1** Add clear public functions for tile-index, point, and AOI
  creation that accept `TileConfig`.
- `[Planned]` **T5.2** Keep the TMS geometry classes responsible only for zone,
  coordinate, and tile calculations.
- `[Planned]` **T5.3** Isolate GDAL raster reading, warping, and writing from
  query orchestration so each layer can be tested independently.
- `[Planned]` **T5.4** Add temporary compatibility wrappers for legacy callers
  identified in T0, with deprecation guidance pointing to the new API.
- `[Planned]` **T5.5** Run the existing tiling test suite and resolve regressions
  without restoring WAC-specific branching.

## Phase T6 — Validate on representative lunar data `[Planned]`

- `[Planned]` **T6.1** Run a WAC product-scoped AOI query on the HPC data and
  inspect zone, tile, band, CRS, transform, shape, and NoData metadata.
- `[Planned]` **T6.2** Run the equivalent NAC query through the same public API.
- `[Planned]` **T6.3** Run a mixed WAC-plus-static query with explicitly ordered
  static bands and per-band NoData considerations.
- `[Planned]` **T6.4** Compare representative new WAC/static cubes against legacy
  outputs and document any intentional differences.
- `[Planned]` **T6.5** Confirm that repeated runs are deterministic and do not
  depend on filename parsing or implicit index creation.

## Phase T7 — Modernize the tiling example notebook `[Planned]`

- `[Planned]` **T7.1** Create `notebooks/tiling_example.ipynb` while retaining the
  legacy notebook until migration is complete.
- `[Planned]` **T7.2** Use the repository-root convention from
  `notebooks/instance_ibm_train.ipynb` and derive repository-owned paths from
  `repo_root`.
- `[Planned]` **T7.3** Put HPC source paths, indexes, output paths, and modality
  declarations in one visible configuration section.
- `[Planned]` **T7.4** Demonstrate tile-index, point, and AOI queries with
  `TileConfig` and structured results.
- `[Planned]` **T7.5** Add compact visual and metadata inspection of generated
  cubes, including band names and NoData counts.
- `[Planned]` **T7.6** Execute the notebook top-to-bottom on the HPC system and
  clear stale outputs before committing the modern copy.

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
