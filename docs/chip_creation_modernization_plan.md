# Chip Creation Modernization Plan

## Objective

Replace the training-GeoDataFrame-driven, WAC/static-specific chip workflow
with a reference-TIFF-driven workflow configured by `ChipConfig`. Each
reference TIFF supplies the authoritative output grid and a geographic query
AOI; the composed `TileConfig` supplies the LTM datacubes; and the chip layer
merges, reprojects, orders, combines, and writes model-ready samples.

## Prerequisite

The tiling modernization plan must be `[Complete]` through **T8.5** before this
plan starts. Chip creation will consume the final `TileConfig`, public tiling
API, and structured `TileCubeRecord` contract rather than maintaining a second
tiling path.

## Status and sequencing

- `[Planned]`: not started.
- `[In-P]`: currently in progress.
- `[Complete]`: implemented, tested, and documented for the stated scope.

Phases and sub-steps are strictly sequential. Start a sub-step only after the
preceding sub-step is `[Complete]`, and start a phase only after every sub-step
in the preceding phase is `[Complete]`. Only one sub-step should be `[In-P]` at
a time. A phase becomes `[Complete]` when all of its sub-steps are complete.

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
- `[Planned]` **C0.4** Define acceptance examples for WAC-only, NAC-only, and
  WAC-plus-static reference-driven chip creation.

## Phase C1 — Introduce chip configuration objects `[Planned]`

- `[Planned]` **C1.1** Add a `ChipConfig` dataclass containing the reference TIFF
  directory, output dataset root, composed `TileConfig`, and ordered output
  modalities.
- `[Planned]` **C1.2** Add final output naming, common NoData, label source,
  optional sample limits, and intermediate-cube retention settings.
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
  width, height, and band metadata into a reference-sample record.
- `[Planned]` **C2.3** Transform densified reference bounds to lunar geographic
  `IAU:30100` coordinates and express them in the tiling API's upper-left and
  lower-right AOI convention.
- `[Planned]` **C2.4** Retain the original projected grid as the exact destination
  grid for final chip creation.
- `[Planned]` **C2.5** Add tests with projected lunar raster fixtures to verify
  geographic AOIs and round-trip coverage of the reference extent.

## Phase C3 — Acquire cubes through the tiling API `[Planned]`

- `[Planned]` **C3.1** Derive per-source query selectors, including product IDs,
  from each reference sample and `ChipConfig`.
- `[Planned]` **C3.2** Call the public AOI tiling function with the composed
  `TileConfig`; do not instantiate or reach into legacy pipeline internals.
- `[Planned]` **C3.3** Group returned `TileCubeRecord` objects by source, zone,
  zoom, and tile coordinates without parsing filenames.
- `[Planned]` **C3.4** Define explicit behavior for absent required modalities,
  optional modalities, multiple LTM zones, and incomplete tile coverage.
- `[Planned]` **C3.5** Add orchestration tests using synthetic structured tiling
  results.

## Phase C4 — Generalize merge and reprojection `[Planned]`

- `[Planned]` **C4.1** Replace fixed `wac_files` and `static_files` parameters
  with ordered source-to-cube mappings.
- `[Planned]` **C4.2** Merge adjacent LTM cubes independently for each configured
  source modality.
- `[Planned]` **C4.3** Normalize or preserve source NoData according to each
  `TileSourceConfig` before interpolation.
- `[Planned]` **C4.4** Reproject each merged modality onto the reference TIFF's
  exact CRS, transform, width, and height using its configured resampling
  method.
- `[Planned]` **C4.5** Verify spatial shape, transform, CRS, and extent against the
  reference grid before assembly.
- `[Planned]` **C4.6** Add tests for continuous and categorical resampling,
  modality-specific NoData, and multi-tile reference extents.

## Phase C5 — Assemble and write model-ready chips `[Planned]`

- `[Planned]` **C5.1** Select and order bands within each modality from explicit
  configuration or cube metadata.
- `[Planned]` **C5.2** Concatenate modalities in `ChipConfig` order and construct
  the final ordered band-name list.
- `[Planned]` **C5.3** Write a compressed GeoTIFF with the reference grid, common
  output NoData, deterministic dtype, and per-band descriptions.
- `[Planned]` **C5.4** Use dataset-compatible sample IDs and terminal filename
  suffixes such as `_input_wac_static_chip.tif`.
- `[Planned]` **C5.5** Reopen each written chip and verify channel count, shape,
  CRS, transform, NoData, band descriptions, and finite-data coverage.
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
  ensure resources are closed on every path.
- `[Planned]` **C7.4** Add safe overwrite and intermediate-cube retention
  behavior without deleting output implicitly.
- `[Planned]` **C7.5** Validate the serial workflow before introducing any
  optional multiprocessing adapter.
- `[Planned]` **C7.6** If still needed after profiling, add multiprocessing using
  the same single-reference operation and result contract.

## Phase C8 — Validate complete datasets on HPC `[Planned]`

- `[Planned]` **C8.1** Run a small WAC reference directory and inspect every
  generated image-label pair.
- `[Planned]` **C8.2** Run a NAC reference directory through the same API and
  record any source-specific selectors or NoData configuration.
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
  `TileConfig`, and `ChipConfig` in one visible user-configuration section.
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
  API or an intentional compatibility wrapper.
- `[Planned]` **C10.3** Deprecate or remove the training-GeoDataFrame dependency
  after all callers migrate.
- `[Planned]` **C10.4** Run formatting, unit tests, dataset loading smoke tests,
  and the final HPC notebook validation.
- `[Planned]` **C10.5** Mark this plan complete and archive the legacy notebook
  workflow according to the repository's chosen deprecation policy.
