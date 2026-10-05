# Chip Creation Backend Structure

At a high level, the backend is a deterministic batch coordinator around an
isolated, sequential per-chip pipeline.

```text
Reference TIFFs or explicit AOIs
              |
              v
         ChipRequest objects
              |
              v
  Sort -> geographic validation -> split planning -> label validation
              |
              v
    PreparedChipRequest objects
              |
       +------+------+
       | worker pool |  one request per worker task
       +------+------+
              |
              v
 Tiling -> mosaic/reproject/clip -> assemble/write -> publish
              |
              v
 ChipResult + diagnostic JSON per sample
              |
              v
        Dataset manifest
```

Each worker processes one chip serially. Parallelism happens by running
multiple independent chips simultaneously.

AOI-first extension status (A1–A4): requests can now derive an outward-rounded
output grid from a geographic IAU:30100 AOI plus original source-grid metadata,
or the static-only 100 m rule. Typed `LabelInput`, `LabelPreparationPlan`, and
`PreparedLabelArtifact` records describe full-scene label preparation without
carrying arrays. The diagram above still describes the executable pipeline:
automatic worker materialization remains A5. A2 validates full-scene sources
and returns read-only preparation plans. A3 semantic materialization is complete;
A4 adds standalone instance/GeoPackage conversion (pending HPC validation). Plans
needing conversion fail safely at execution with `label_preparation_not_available`
before tiling or overwrite cleanup; exact array plans remain executable.
See the [AOI implementation plan](planning_docs/aoi_chip_creation_and_label_clipping_plan.md)
for the API and validation status. Existing exact-label workflows remain active.

## Behavior map for files under `model/`

The `model/` directory contains the modern chip pipeline, the tiling backend it
uses, and older or offline TMS utilities. Not every file participates directly
in a modern chip-creation run.

### Modern chip-creation files

- `model/__init__.py` is the public import surface. It re-exports the supported
  chip, tiling, CRS, index, and band-contract objects; it does not implement a
  processing stage.
- `model/chip_types.py` owns the shared request, grid, AOI, preflight,
  diagnostic, result, and typed-error records, plus invariants involving more
  than one request.
- `model/chip_config.py` owns acquisition-group, output-modality, split,
  output-format, NoData, sample-limit, and intermediate-retention policies. It
  also applies built-in zoom and WAC band defaults.
- `model/chip_requests.py` owns reference-TIFF discovery, sample/product
  identity, reference-to-request conversion, explicit-AOI request creation,
  target-grid derivation, geographic validation, and antimeridian query
  splitting.
- `model/chip_splits.py` owns deterministic, group-atomic dataset assignment,
  fixed-count priorities, percentage assignment, prior-manifest locks, no-split
  assignment, and nonfatal target-shortfall warnings.
- `model/chip_labels.py` owns label lookup (identity matching for directories
  only), final semantic/instance archive validation, shape checks,
  instance-occlusion handling, and optional label-grid/sidecar comparison.
- `model/chip_label_planning.py` owns read-only source validation, source-grid
  resolution, raster coverage/relation checks, GeoPackage validation, hashes,
  and compact label-preparation plans.
- `model/chip_label_materialization.py` owns standalone worker-side semantic
  preparation: exact NPY reuse, source windows, integer-preserving nearest
  reprojection, verified no-clobber NPY staging, and artifact provenance. Shared
  mask reading also supports NPZ for the instance converter.
- `model/chip_instance_labels.py` owns standalone GeoPackage rasterization and
  joint NPZ mask/box/count conversion: clipped-outline boxes, center inclusion,
  highest-source-ID overlap priority, compact IDs, occlusion/omission diagnostics,
  deterministic archive staging, source verification, and safe rollback.
- `model/chip_preflight.py` owns the non-writing batch gate: deterministic
  request materialization, geographic checks, split planning, conditional label
  validation, and construction of `PreparedChipRequest` objects.
- `model/chip_acquisition.py` owns source-selector derivation, acquisition-group
  execution, calls into the tiling API, antimeridian result combination,
  structured cube-record grouping, source-coverage checks, and preservation of
  partial tiling results.
- `model/chip_reprojection.py` owns cube validation, grouping by LTM zone,
  per-zone mosaicking, source NoData normalization, warping onto the exact
  target grid, cross-zone compositing, and reprojected modality results.
- `model/chip_assembly.py` owns configured band selection and ordering,
  product-qualified band-name matching, modality concatenation, common-NoData
  conversion, safe dtype conversion, staged GeoTIFF writing, and reopen
  validation.
- `model/chip_publication.py` owns final split-directory selection,
  checksum-verified and rollback-safe chip/label publication, byte-preserving
  label copies, dataset membership validation, and deterministic manifest
  creation.
- `model/chip_creation.py` owns the public single-chip, batch, and
  reference-directory APIs. It coordinates stages, multiprocessing, worker
  progress, per-sample failure isolation, diagnostic JSON files, intermediate
  cleanup/retention, deterministic result ordering, and final manifest writing.
- `model/wac_band_contract.py` owns the canonical five-VIS-then-two-UV WAC
  selection and output order. It contains constants rather than processing
  behavior.
- `model/static_band_contract.py` owns the canonical static band order and the
  special source/output NoData rules used for static data. It also contains
  constants rather than a processing stage.

### Modern tiling files used by chip acquisition

- `model/tiling.py` is the small public tiling facade. Its index, point, and AOI
  functions construct a `ConfiguredTiler` and return structured records.
- `model/tiling_config.py` owns modality-neutral tile/source configuration,
  including paths, source selection mode, bands, resampling, required/optional
  behavior, and per-band NoData overrides.
- `model/tiling_policy.py` owns dependency-light policies for product-ID versus
  all-intersecting source selection, selector validation, and effective
  per-band NoData values.
- `model/tiling_results.py` owns tile-cube filenames, `TileCubeRecord`, and typed
  source errors that preserve completed records when later tile work fails.
- `model/configured_tiler.py` is the configuration-driven tiling engine. It
  resolves intersecting LTM tiles, queries each configured source, creates one
  source cube per tile, tracks partial completion, and implements index, point,
  and AOI execution.
- `model/raster_cube.py` owns the low-level GDAL operations for selecting source
  bands, warping them to one LTM tile, preserving/normalizing NoData according
  to policy, writing a cube, and returning its structured record.
- `model/vector_index.py` owns format-independent Shapefile/GeoPackage access,
  spatial source-footprint queries, location-field validation, and resolution
  of stored raster paths.
- `model/vector_index_builder.py` owns explicit creation of source-raster vector
  indexes. It is a data-preparation utility, not a per-chip processing stage.
- `model/lunar_crs.py` owns the repository path and loader for the bundled lunar
  geographic CRS used by both request and tiling code.

### Low-level, legacy, or offline TMS files

- `model/TmsIntersector.py` loads the numbered LTM zones and returns the zones
  and tile definitions intersecting an AOI. The modern configured tiler still
  uses this low-level geometry component.
- `model/TmsZoneDef.py` represents one LTM zone, including its CRS, bounds,
  intersection logic, and access to zoom-specific tile definitions. It is used
  through `TmsIntersector`.
- `model/TmsTileDef.py` implements LTM tile-matrix geometry: coordinate
  transforms, tile indices, overlapping-tile queries, tile bounds, matrix
  dimensions, resolution, and origin. The modern configured tiler and raster
  cube writer still use it.
- `model/Pipeline.py` is the older stateful tiling/cube pipeline. It contains
  legacy query, clipping, static handling, and cube-writing behavior, but the
  modern chip path calls `model/tiling.py` and `ConfiguredTiler` instead.
- `model/create_gpkg.py` is an offline utility for generating GeoPackage tile
  geometry/index data across LTM zones. It is not run for each chip.
- `model/parallel_quadtree.py` contains alternate parallel/quadtree tile-index
  generation and query utilities. It is not part of the normal modern chip
  execution path.

## Core contracts

`model/chip_types.py` defines the vocabulary shared across the pipeline:

- `ChipRequest`: the requested output chip, including its exact grid,
  geographic AOI, sample ID, split-group identity, optional label path, and
  source selectors.
- `TargetGrid`: authoritative CRS, affine transform, bounds, width, and height.
- `GeographicAOI`: the geographic query envelope corresponding to that grid.
- `ChipPreflight`: label and split eligibility established before tiling.
- `ChipResult`: final structured outcome, including paths, status,
  diagnostics, selectors, timing, and tiling records.

`validate_request_contracts()` is here because it validates relationships among
those core request objects, such as duplicate sample IDs or conflicting split
assignments within one group.

## Configuration

`model/chip_config.py` defines policies shared by the batch:

- `AcquisitionGroupConfig`: a group of sources tiled at one zoom and grid
  policy. In the current WAC-plus-static workflow, WAC and static belong to
  `wac_grid` at zoom 5.
- `OutputModalityConfig`: selects a source's bands, output ordering, output
  names, and resampling.
- `ChipConfig`: output paths, label source, acquisition groups, modalities,
  split configuration, NoData value, dtype, sample limit, and
  intermediate-retention policy.
- Split configurations include simple percentage, mixed number/percentage,
  number-only, and no-split modes.
- Built-in WAC defaults select VIS first, then UV; static follows because the
  static output modality is declared after WAC.

The WAC and static band lists are maintained separately in
`model/wac_band_contract.py` and `model/static_band_contract.py`.

## Request construction

`model/chip_requests.py` turns user inputs into `ChipRequest` objects.

There are two primary routes:

- Reference-driven: read the CRS, transform, shape, and bounds from an existing
  TIFF.
- AOI-driven: construct an exact target grid from an explicitly supplied AOI
  and dimensions.

The reference TIFF is not the imagery copied into the final dataset. It defines
the output goal: exact location, grid, width, and height.

This module also:

- Discovers and sorts reference TIFFs.
- Preserves offset-qualified IDs such as `M109..._r7650_c750`.
- Derives WAC/NAC product IDs from sample IDs.
- Converts target-grid bounds into geographic query AOIs.
- Divides an antimeridian-crossing query into two ordinary tiling queries.

## Split planning

`model/chip_splits.py` assigns requests to dataset partitions before labels or
imagery are processed.

Important properties:

- Assignment is deterministic from the configured seed/hash policy.
- `split_group_key` keeps related samples together. The full WAC workflow groups
  all offsets from the same product ID.
- Explicit assignments and prior manifests take precedence.
- Fixed-number shortages produce warnings rather than terminating the batch.
- The approved default attempts 100 test samples, then assigns 90%/10% of the
  remainder to train/validation.
- `NoSplitConfig` assigns everything to an `unsplit` logical assignment
  published directly under `output_root/chips` and `output_root/labels`.

## Label validation and preflight

`model/chip_labels.py` owns label resolution and final-target validation:

- Resolves directories by full sample identity, including row/column offsets;
  explicit file associations do not require identity matching.
- Requires final training-label shape to equal the target chip's height and width.
- Validates semantic masks and instance archives.
- Checks instance counts, IDs, bounding boxes, and the accepted occlusion
  heuristic.
- Validates label grid metadata against the requested chip grid when a sidecar
  or explicit label grid is available.

`model/chip_label_planning.py` validates source labels separately from final
labels. It reads source-grid sidecars or GeoTIFF metadata, classifies exact,
aligned-window, and nearest-warp raster relations, checks lunar CRS/coverage
and NoData gaps, and validates GeoPackage crater layers. It produces compact,
hashed `LabelPreparationPlan` records without writing any label, dataset, or
intermediate files. `model/chip_label_materialization.py` separately implements
semantic preparation on that plan and returns a verified `PreparedLabelArtifact`.
`model/chip_instance_labels.py` implements the corresponding instance adapter
and independent GeoPackage converter (A4; HPC validation pending). Automatic
execution remains A5; pending orchestration materialization is explicitly guarded.

`model/chip_preflight.py` coordinates the batch-level preflight:

1. Materialize and deterministically sort requests.
2. Validate target grids and AOIs.
3. Plan splits.
4. Skip requests left unassigned by a number-only policy.
5. Resolve and validate source labels and build preparation plans for assigned requests.
6. Produce one `PreparedChipRequest` per input.

A failed label never reaches tiling, and no chip is written for it.

## Acquisition

`model/chip_acquisition.py` adapts prepared chip requests to the modern tiling
API.

For each acquisition group it:

- Derives product selectors for WAC/NAC.
- Uses all intersecting rasters for sources such as static.
- Sends each geographic query part to `create_tiles_for_aoi()`.
- Deduplicates structured records when an antimeridian AOI has two query parts.
- Records coverage gaps and preserves completed tile records if later work
  fails.

Tiling may also complete a geographically valid high-level AOI query with a
`ProductAOIWarning` and zero records when no dynamic product intersects; mixed
queries skip contextual static in that case. Chip acquisition must therefore
judge success from structured record coverage, not from the absence of a
tiling exception. Missing required chip modalities remain a typed per-sample
failure that publishes neither artifact, while optional omissions may use the
existing known-band placeholder policy. The current chip caller still uses the
strict low-level AOI API, where per-tile required-source failures can raise.

Intermediate tiling output is isolated under:

```text
<intermediate_root>/<sample_id>/<acquisition_group>/...
```

The tiling backend still owns the nested zone, zoom, and tile structure. Chip
acquisition consumes its structured `TileCubeRecord` results rather than
reconstructing information from filenames.

## Reprojection

`model/chip_reprojection.py` converts tiling cubes into arrays on the
authoritative target grid.

For each output modality it:

- Groups records by LTM zone.
- Validates opened rasters against their structured tiling metadata.
- Mosaics adjacent tiles within a zone.
- Normalizes source-specific NoData before resampling.
- Warps each zone independently onto the exact requested grid.
- Composites zone results into a `ReprojectedModality`.

The output dimensions always come from `ChipRequest.target_grid`, not from
whichever source raster happens to be opened.

## Assembly and staged writing

`model/chip_assembly.py` combines the reprojected modalities.

It:

- Selects and reorders bands according to `OutputModalityConfig`.
- Matches canonical WAC names against product-qualified descriptions using
  regex.
- Concatenates modalities in configuration order.
- Converts all invalid pixels to the configured common NoData value.
- Writes a temporary model-ready GeoTIFF and reopens it to validate its grid,
  dtype, names, count, compression, and valid pixels.

For the current run, the final order is:

```text
5 VIS -> 2 UV -> 63 static bands
```

## Publication

`model/chip_publication.py` moves the validated staged chip and original label
into the dataset.

Publication is pair-atomic:

1. Revalidate the chip and label.
2. Stage both artifacts in their destination directories.
3. Verify checksums.
4. Publish the chip and label together.
5. If either publication fails, roll back the other.
6. Never leave a chip without its matching label.

Depending on the split policy, the destination is either:

```text
<output_root>/<train|val|test>/{chips,labels}
```

or:

```text
<output_root>/{chips,labels}
```

The label is byte-preserved; it is validated but not rewritten.

This module also creates and validates `dataset_manifest.json`, including
configuration, split policy, assignments, selectors, output paths, statuses,
and failures.

## Orchestration and multiprocessing

`model/chip_creation.py` connects everything.

The three important entrypoints are:

- `create_chip(prepared, config)`: low-level sequential worker operation for
  one already-prepared request.
- `create_chips(requests, config, max_workers=...)`: normal batch API.
- `create_chips_from_reference_directory(...)`: convenience API that discovers
  TIFFs, builds requests, and invokes `create_chips()`.

For a parallel batch:

- All preflight and split planning happen once in the coordinator.
- A spawn-based process pool receives independent `PreparedChipRequest`
  objects.
- Each worker runs one chip through tiling, reprojection, assembly, and
  publication.
- GDAL internal threading is held to one thread per worker to avoid nested
  oversubscription.
- Worker progress events return through a queue.
- Completed results are restored to deterministic request order regardless of
  worker completion order.
- The coordinator writes the manifest after all workers finish.

Failures are isolated per sample. A failed worker operation becomes a `failed`
or `partial` `ChipResult`, gets a diagnostic JSON, and does not stop later
samples. The main exception is direct use of `create_chip()`, where an invalid
assigned label raises `LabelMismatchError`; the batch wrapper catches and
converts that into a failed result.

For the full WAC-plus-static run, the Slurm script is only an outer operational
wrapper. It builds the WAC-plus-static `ChipConfig`, constructs all reference
requests, and calls `create_chips(..., max_workers=16)`. The actual geospatial
behavior remains in the backend modules above.
