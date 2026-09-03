# Chip Creation Modernization Plan

## Objective

Replace the training-GeoDataFrame-driven, WAC/static-specific chip workflow
with a reference-TIFF-driven workflow configured by `ChipConfig`. Each
reference TIFF supplies the authoritative output grid and a geographic query
AOI; one or more composed `TileConfig` acquisition groups supply the LTM
datacubes; and the chip layer merges, reprojects, orders, combines, and writes
model-ready samples.

## Starting point and prerequisite

The tiling modernization plan is `[Complete]` as of 2026-09-03. Its
configuration-driven backend was regression-tested through **Phase T6**, and
the supported notebook and migration closeout were completed in Phases T7 and
T8. The stable T0–T6 contract is the integration prerequisite for this plan,
and chip modernization may start now.

The earlier dependency on tiling **T8.5** was resolved by assigning migration
of the legacy chip acquisition path to this plan. It will consume the stable
`TileConfig`, public `create_tiles_for_*` API, and structured `TileCubeRecord`
contract rather than maintaining a second tiler.

The source of truth for the tiling handoff is the **Stable tiling contract for
chip creation** section in
[`tiling_modernization_plan.md`](tiling_modernization_plan.md). The Armstrong
scheme, IAU:30100 CRS, LTM zone/zoom/tile address, and polar limitation are
documented in [`TMS/README.md`](../TMS/README.md).

## Status and sequencing

- `[Planned]`: not started.
- `[In-P]`: currently in progress.
- `[Complete]`: implemented, tested, and documented for the stated scope.

Phases and sub-steps are strictly sequential. Start a sub-step only after the
preceding sub-step is `[Complete]`, and start a phase only after every sub-step
in the preceding phase is `[Complete]`. Only one sub-step should be `[In-P]` at
a time. A phase becomes `[Complete]` when all of its sub-steps are complete.

## Tiling backend handoff (2026-09-03)

### Stable implementation map

| Concern | Existing implementation to consume |
|---|---|
| Public configuration | `model/tiling_config.py`: `TileConfig`, `TileSourceConfig`, `BandNoDataOverride`, `tile_config_from_dict` |
| Public acquisition API | `model/tiling.py`: `create_tiles_for_aoi`, `create_tiles_for_point`, `create_tiles_for_index` |
| Orchestration | `model/configured_tiler.py`; do not call `ConfiguredTiler` directly from chip code |
| Zone and tile geometry | `model/TmsIntersector.py`, `model/TmsZoneDef.py`, `model/TmsTileDef.py` |
| Lunar geographic CRS | `model/lunar_crs.py` loading repository `TMS/IAU_30100_2015.wkt` |
| Read-only source indexes | `model/vector_index.py` supporting `.shp` and `.gpkg` |
| Selection and NoData policy | `model/tiling_policy.py` |
| Warp and cube writing | `model/raster_cube.py` |
| Structured outputs and errors | `model/tiling_results.py`: `TileCubeRecord`, `TileSourceError`, `MissingRequiredSourceError` |
| Canonical static bands | `model/static_band_contract.py` and `lfm/all_models/all_tasks/tiling_utils.py::make_static_source` |
| Worked example | `notebooks/tiling_example.ipynb` |

### Contract chip code must preserve

- Import the public exports from `lfm.model` in repository scripts that put the
  repository parent on `sys.path`. Do not import private acquisition helpers or
  instantiate the deprecated `lfm.model.Pipeline.Pipeline`.
- Supply AOIs as `ul_lat`, `ul_lon`, `lr_lat`, and `lr_lon` in IAU:30100. Use
  `load_lunar_geographic_wkt()`; do not add another inline lunar WKT string.
- Treat every source index as a read-only input. The configured location field
  can resolve absolute paths or paths relative to that source's `data_dir`.
  Chip creation must not create an index as a side effect of tiling.
- Pass product selectors as a mapping keyed by source name. A source configured
  as `product_id` requires a selector; `all_intersecting` rejects one.
- Group records by structured fields:
  `(zone, zoom_level, tile_x, tile_y, source_name)`. Filenames are for human
  inspection only and are not an orchestration interface.
- Expect partial record sets. Required missing coverage raises
  `MissingRequiredSourceError`; optional sparse coverage can produce no record
  for one source while other cubes have already been written. Record and clean
  up intermediate outputs according to an explicit chip-level status policy.
- Tiling always uses bilinear resampling and writes 512×512 LTM GeoTIFFs. Any
  chip-stage resampling policy is separate and must not be implemented by
  weakening the tiling contract.
- The canonical static cube contains 63 bands in exact contract order and uses
  `-32768` as the output NoData value for every band. The two Mini-RF bands
  declare their `-3.4e38` values as source-only sentinels. WAC and NAC examples
  preserve native source NoData.
- A `TileConfig` has exactly one zoom shared by all sources in that acquisition
  group. The current notebook uses WAC plus static at zoom 5 and 1 m NAC plus
  static at zoom 11. `ChipConfig` must not assume all sensor families should
  share the repository's zoom-5 default.
- Use a separate intermediate output directory for each reference sample and
  acquisition group. Static filenames omit product IDs, so sharing one output
  directory across samples can cause valid tile-address collisions.
- Only the numbered LTM path is currently supported. LPN/LPS polar metadata is
  present but was not integrated because its geometry differs from the LTM
  longitude-band grids. Detect and report unsupported polar references instead
  of routing them through LTM code.

### Validation evidence already completed

- 42 modern backend contract tests and 25 selected legacy geometry/pipeline
  regressions passed in the Explore GDAL environment.
- Product-scoped WAC and NAC AOI acquisition passed with CRS, transform,
  512×512 shape, bands, and NoData metadata verified against written files.
- Mixed WAC/static acquisition passed with the canonical 63 static bands and
  uniform static output NoData.
- Representative modern and legacy WAC/static cubes match in grid, band order,
  masks, and valid pixel values. The intentional static difference is modern
  `float64` output for safely combining mixed source dtypes versus legacy
  first-band-selected `float32`.
- Repeated acquisition produced identical record ordering, metadata, and file
  SHA-256 values, while declared source indexes remained unchanged.

These checks establish the cube acquisition boundary. They do not validate the
legacy chip merge/reprojection code; that work belongs to Phases C0–C8 below.

### Existing chip code is migration input, not the target API

The next agent should inventory `model/chip_making/chip_utils.py`, the legacy
`notebooks/toy_model/chip_example.ipynb`, and current task wrappers such as
`scripts/python/instance_seg/create_instance_wac_static_chips.py`. They contain
useful merge, reprojection, label-copying, and dataset-layout behavior, but also
contain the coupling this plan is intended to remove:

- construction of the deprecated `Pipeline`;
- one hard-coded dynamic tile database plus implicit static behavior;
- WAC/static-specific parameter names and dictionaries;
- filename regexes used to recover LTM zone and tile identity;
- training-GeoDataFrame-driven AOIs; and
- broad tuples/status strings instead of typed sample, grid, and result
  records.

Preserve observable output behavior only after it is written into the C0
contract and covered by a test. Do not preserve an internal legacy technique
merely because it is already present. Conversely, do not change the stable
tiling backend to accommodate old chip assumptions unless a reproducible
tiling defect is demonstrated and added to the tiling regression suite.

## Phase C0 — Define the reference-chip contract `[Planned]`

- `[Planned]` **C0.1** Define which TIFFs qualify as reference training
  examples and which metadata must be present: CRS, transform, bounds, width,
  height, and stable sample identity.
- `[Planned]` **C0.2** Define how sample ID and optional source product IDs are
  extracted from a reference filename without coupling the workflow to one
  sensor's full filename.
- `[Planned]` **C0.3** Define the final chip directory layout, filename suffixes,
  band metadata, and image-label pairing conventions in accordance with
  `docs/dataset_contribution.md`.
- `[Planned]` **C0.4** Define acceptance examples for WAC-only, NAC-only,
  WAC-plus-static, and NAC-plus-static reference-driven chip creation,
  including at least one reference that crosses an LTM tile boundary.
- `[Planned]` **C0.5** Define acquisition groups and their zoom policy. Sources
  within one group share one `TileConfig` zoom; sensor families needing
  different resolutions use separate groups (currently WAC zoom 5 and NAC
  zoom 11).
- `[Planned]` **C0.6** Define clear rejection behavior for reference extents
  outside the currently supported numbered LTM path, including LPN/LPS polar
  coverage.

## Phase C1 — Introduce chip configuration objects `[Planned]`

- `[Planned]` **C1.1** Add a `ChipConfig` dataclass containing the reference TIFF
  directory, output dataset root, one or more named composed `TileConfig`
  acquisition groups, and ordered output modalities.
- `[Planned]` **C1.2** Add final output naming, common NoData, label source,
  optional sample limits, intermediate-cube retention settings, and a
  chip-stage resampling policy distinct from tiling's fixed bilinear policy.
- `[Planned]` **C1.3** Represent target-grid and reference-sample metadata with
  small structured records instead of passing unrelated tuples through helper
  functions.
- `[Planned]` **C1.4** Add a dictionary-to-config constructor consistent with the
  tiling configuration interface.
- `[Planned]` **C1.5** Add focused config tests, including invalid composed
  tiling configurations and duplicate or missing output modalities.

## Phase C2 — Derive AOIs and target grids from reference TIFFs `[Planned]`

- `[Planned]` **C2.1** Discover reference TIFFs deterministically from the
  configured directory without loading a training GeoDataFrame.
- `[Planned]` **C2.2** Read each reference TIFF's CRS, affine transform, bounds,
  width, height, and band metadata into a reference-sample record. Reject
  missing or unusable CRS/grid metadata before creating intermediate cubes.
- `[Planned]` **C2.3** Transform densified reference bounds to lunar geographic
  `IAU:30100` coordinates using the repository WKT loader and express them in
  the tiling API's upper-left and lower-right AOI convention. Follow the
  densified-perimeter behavior already exercised by
  `scripts/python/all_tasks/extract_datacube_aoi.py`; transforming only two
  corners is not sufficient for every projected raster.
- `[Planned]` **C2.4** Retain the original projected grid as the exact destination
  grid for final chip creation.
- `[Planned]` **C2.5** Verify that each geographic AOI falls within supported
  numbered LTM coverage before acquisition and return a specific unsupported
  status for polar-only references.
- `[Planned]` **C2.6** Add tests with projected lunar raster fixtures to verify
  geographic AOIs, round-trip coverage of the reference extent, axis order,
  zone-boundary behavior, and polar rejection.

## Phase C3 — Acquire cubes through the tiling API `[Planned]`

- `[Planned]` **C3.1** Derive per-source query selectors, including product IDs,
  from each reference sample and `ChipConfig`. Do not assume every modality
  uses a product selector; static/context sources use `all_intersecting`.
- `[Planned]` **C3.2** Call the public AOI tiling function with the composed
  `TileConfig`; do not instantiate or reach into legacy pipeline internals.
  Derive a unique sample/acquisition-group output directory and create a
  sample-specific config, for example with `dataclasses.replace`, before each
  call.
- `[Planned]` **C3.3** Group returned `TileCubeRecord` objects by source, zone,
  zoom, and tile coordinates without parsing filenames. Keep records from
  different acquisition groups separate when their zooms differ.
- `[Planned]` **C3.4** Define explicit behavior for absent required modalities,
  optional modalities, multiple LTM zones, incomplete tile coverage, and
  `completed_records` attached to a source error.
- `[Planned]` **C3.5** Return and retain structured acquisition diagnostics,
  including selectors, AOI, acquisition group, zoom, records, and any partial
  files, so a batch failure can be reproduced without parsing log text.
- `[Planned]` **C3.6** Add orchestration tests using synthetic structured tiling
  results, including missing required WAC, optional sparse NAC, colliding tile
  addresses in separate sample directories, and multiple acquisition zooms.

## Phase C4 — Generalize merge and reprojection `[Planned]`

- `[Planned]` **C4.1** Replace fixed `wac_files` and `static_files` parameters
  with ordered source-to-cube mappings.
- `[Planned]` **C4.2** Merge adjacent LTM cubes independently for each configured
  source modality, zone, and zoom. Never place cubes from different LTM CRSs or
  zoom grids into one source-space mosaic merely because their filenames share
  tile numbers.
- `[Planned]` **C4.3** Normalize or preserve source NoData according to each
  `TileSourceConfig`, `TileCubeRecord.nodata_values`, and reopened raster
  metadata before interpolation. Do not infer NoData from pixel magnitude.
- `[Planned]` **C4.4** Reproject each merged modality onto the reference TIFF's
  exact CRS, transform, width, and height. Keep this chip-stage resampling
  choice separate from the tiler's fixed bilinear warp: use bilinear for
  continuous image/context data and nearest-neighbor for categorical data.
- `[Planned]` **C4.5** Verify spatial shape, transform, CRS, and extent against the
  reference grid before assembly.
- `[Planned]` **C4.6** Add tests for continuous and categorical resampling,
  modality-specific NoData, and multi-tile reference extents.
- `[Planned]` **C4.7** Add a reference crossing an LTM zone boundary and verify
  that independently reprojected zone groups composite correctly on the one
  authoritative reference grid.

## Phase C5 — Assemble and write model-ready chips `[Planned]`

- `[Planned]` **C5.1** Select and order bands within each modality from explicit
  configuration or cube metadata.
- `[Planned]` **C5.2** Concatenate modalities in `ChipConfig` order and construct
  the final ordered band-name list.
- `[Planned]` **C5.3** Write a compressed GeoTIFF with the reference grid, common
  output NoData, deterministic dtype, and per-band descriptions. Convert every
  input sentinel to the chosen common value before writing because GeoTIFF's
  persisted `TIFFTAG_GDAL_NODATA` is dataset-wide. Make any downcast from the
  modern static cube's safe `float64` dtype explicit and range-checked.
- `[Planned]` **C5.4** Use dataset-compatible sample IDs and terminal filename
  suffixes such as `_input_wac_static_chip.tif`.
- `[Planned]` **C5.5** Reopen each written chip and verify channel count, shape,
  CRS, transform, dataset-wide NoData, band descriptions, per-band masks, and
  finite-data coverage.
- `[Planned]` **C5.6** Add regression tests for band order, including the legacy
  WAC VIS-then-UV ordering where required.

## Phase C6 — Add label pairing and dataset layout `[Planned]`

- `[Planned]` **C6.1** Resolve a matching label from the configured label source
  using the stable reference sample ID.
- `[Planned]` **C6.2** Preserve semantic `.npy` labels and instance `.npz`
  contents without assuming the legacy `data` key.
- `[Planned]` **C6.3** Validate instance label archives against the documented
  `mask`, `bboxes`, and `num_craters` contract when instance labels are used.
- `[Planned]` **C6.4** Write image-label pairs into the configured
  `train/val/test/{chips,labels}` layout without silently inventing a split.
- `[Planned]` **C6.5** Confirm that the shared semantic and instance datasets can
  discover and load representative generated pairs.

## Phase C7 — Add sequential batch orchestration `[Planned]`

- `[Planned]` **C7.1** Add a single-reference `create_chip` operation returning a
  structured result with paths, status, timing, cube records, and diagnostics.
- `[Planned]` **C7.2** Add deterministic directory-level `create_chips`
  orchestration that processes reference TIFFs in sorted order.
- `[Planned]` **C7.3** Define clear skipped, partial, and failed statuses and
  ensure resources are closed on every path. Preserve tiling error context and
  account for any `completed_records` or files created before failure.
- `[Planned]` **C7.4** Add safe overwrite and intermediate-cube retention
  behavior without deleting output implicitly. Limit cleanup to the current
  sample's explicitly resolved intermediate directory.
- `[Planned]` **C7.5** Validate the serial workflow before introducing any
  optional multiprocessing adapter.
- `[Planned]` **C7.6** If still needed after profiling, add multiprocessing using
  the same single-reference operation and result contract.

## Phase C8 — Validate complete datasets on HPC `[Planned]`

- `[Planned]` **C8.1** Run a small WAC reference directory and inspect every
  generated image-label pair.
- `[Planned]` **C8.2** Run a NAC reference directory through the same API and
  record its product selectors, native NoData configuration, and acquisition
  zoom. Use zoom 11 for the known 1 m processed NAC example unless the selected
  dataset establishes another resolution contract.
- `[Planned]` **C8.3** Run WAC-plus-static creation with the intended full static
  band list and confirm final band order.
- `[Planned]` **C8.4** Compare representative outputs with the legacy chip
  workflow numerically and visually, documenting intentional differences.
- `[Planned]` **C8.5** Run the pre-training diagnostics from
  `docs/dataset_contribution.md` on the resulting dataset layout.

## Phase C9 — Modernize the chip example notebook `[Planned]`

- `[Planned]` **C9.1** Create `notebooks/chip_example.ipynb` while retaining the
  legacy notebook until migration is complete.
- `[Planned]` **C9.2** Use the repository-root convention from
  `notebooks/instance_ibm_train.ipynb` and derive repository-owned paths from
  `repo_root`.
- `[Planned]` **C9.3** Put the reference directory, label source, output root,
  product selectors, and other true user inputs in one visible configuration
  section. Follow the tiling notebook with separate derived-variable and
  path-resolution sections for `TileConfig`, `ChipConfig`, indexes, run output,
  and validation.
- `[Planned]` **C9.4** Demonstrate AOI extraction from one reference TIFF before
  running the configured single-chip workflow.
- `[Planned]` **C9.5** Display structured tiling and chip results rather than
  reconstructing state through filename parsing.
- `[Planned]` **C9.6** Visualize selected output bands, the paired label, NoData
  coverage, and key grid metadata.
- `[Planned]` **C9.7** Execute the notebook top-to-bottom on the HPC system and
  clear stale outputs before committing the modern copy.

## Phase C10 — Complete the chip migration `[Planned]`

- `[Planned]` **C10.1** Update backend docstrings and user documentation with the
  final `ChipConfig` schema and reference-TIFF workflow.
- `[Planned]` **C10.2** Update repository scripts and callers to use the new chip
  API or an intentional compatibility wrapper. Completing this step also
  satisfies the chip-caller portion intentionally excluded from tiling T8.2.
- `[Planned]` **C10.3** Deprecate or remove the training-GeoDataFrame dependency
  after all callers migrate.
- `[Planned]` **C10.4** Run formatting, unit tests, dataset loading smoke tests,
  and the final HPC notebook validation.
- `[Planned]` **C10.5** Mark this plan complete and archive the legacy notebook
  workflow according to the repository's chosen deprecation policy. Confirm
  that the completed chip migration still consumes the stable, completed
  tiling contract.
