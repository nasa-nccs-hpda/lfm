---
name: lunar-tiling
description: Develop, review, diagnose, document, or run the LFM configuration-driven lunar raster tiling workflow. Use for Armstrong LTM zones and tiles, WAC/NAC/static datacubes, TileConfig APIs, source indexes, NoData and resampling behavior, the tiling notebook, or Explore tiling validation. Do not use for downstream chip reprojection, label processing, or model training except to preserve their tiling input contract.
---

# LFM Lunar Tiling

Work from the cloned repository and preserve the modern, modality-neutral lunar
tiling contract. Inspect current code before changing behavior; the paths below
are routing aids, not substitutes for the implementation.

## Read the relevant source of truth

- Read `TMS/README.md` for IAU:30100, LTM and polar grid geometry, zoom
  matrices, tile addressing, and the current polar support boundary.
- Read `docs/tiling_modernization_plan.md`, especially **Stable tiling contract
  for chip creation**, when changing the public contract or handing output to
  chip creation.
- Read `notebooks/tiling_example.ipynb` and
  `lfm/all_models/all_tasks/tiling_utils.py` for the supported user workflow and
  canonical static-source helper.
- Read `docs/chip_creation_modernization_plan.md` only when the task crosses the
  boundary into downstream chip acquisition.

## Stay inside the supported boundary

The tiling boundary is:

```text
query + TileConfig + per-source selectors
    -> numbered-LTM or polar grid/tile discovery
    -> read-only source-index queries
    -> per-source lunar-grid GeoTIFF cubes
    -> ordered TileCubeRecord results
```

Reference-TIFF alignment, final target-grid reprojection, labels, chip layout,
and training are downstream concerns. New code must use:

- `model/tiling_config.py`: `TileConfig`, `TileSourceConfig`, and
  `BandNoDataOverride`;
- `model/tiling.py`: `create_tiles_for_aoi`, `create_tiles_for_point`, and
  `create_tiles_for_index`; and
- `model/tiling_results.py`: `TileCubeRecord` and structured source errors.

Do not build new behavior on `model/Pipeline.py`. It is a deprecated regression
and temporary compatibility adapter. Do not parse filenames when the returned
record already provides source, product, zone, zoom, tile, bands, CRS, and
NoData metadata.

## Preserve tiling invariants

- Load lunar geographic CRS data from `TMS/IAU_30100_2015.wkt`; do not embed a
  duplicate WKT string.
- Geographic AOIs use IAU:30100 bounds in this order: `ul_lat`, `ul_lon`,
  `lr_lat`, `lr_lon`.
- Numbered LTM and low-level `LPS_N`/`LPS_S` production are regression-closed.
  The easy source-mode/default-zoom workflow remains unfinished; check the
  polar integration plan before presenting polar operation as notebook-ready.
- A tile is 512 x 512 pixels. Its complete address is
  `(zone, zoom_level, tile_x, tile_y)`.
- One `TileConfig` has one zoom shared by all its sources. Use separate configs
  when modalities need different resolutions; the notebook uses WAC/static at
  zoom 5 and 1 m NAC/static at zoom 11.
- Tiling resampling is always bilinear. Chip-stage resampling is a separate
  policy and must not weaken this invariant.
- Each modality supplies a `.shp` or `.gpkg` raster-index path and a location
  field. The high-level workflow validates and reuses that index or prepares it
  when missing; low-level tile generation queries prepared indexes read-only.
  Automatic replacement must be explicitly enabled and is allowed only for an
  application-owned GeoPackage cache. Never replace a shared or legacy index.
- Write one tiled, LZW-compressed BigTIFF per source and lunar-grid tile, with
  the routed grid CRS, exact tile transform, band names, output NoData
  metadata, and group-writable permissions.
- Preserve deterministic record order: zone, tile row, tile column, then source
  configuration order.

## Apply source-selection semantics exactly

- Compose high-level source modes with `compose_tile_sources()`. Both inclusion
  controls default to `True`; reject disabling both or enabling a class without
  at least one source. Do not inspect a disabled collection. Preserve
  dynamic-before-static order and require static sources to use
  `all_intersecting`.
- The strict `create_tiles_for_*` entry points require a nonempty selector for
  each `selection_mode="product_id"` source. The high-level
  `create_tiles_for_aoi_by_product` entry point accepts an exact PID or `None`;
  `None` discovers intersecting PIDs and invokes the strict path separately for
  each resolved product.
- `selection_mode="all_intersecting"` rejects a selector and includes every
  indexed raster intersecting that tile.
- Keep WAC and NAC configured as `product_id` sources. Optional discovery
  groups companion files such as WAC UV/VIS inputs under their filename prefix
  and writes separate cubes for unrelated PIDs. `all_intersecting` remains an
  explicit expert/context policy and still stacks every selected raster.
- Static context uses `all_intersecting` and is never product-filtered.
- Static-only operation takes no product ID. Dynamic-only operation is valid on
  numbered LTM and polar grids.
- A missing required source raises `MissingRequiredSourceError`. An optional
  sparse source may be skipped. Preserve and report `completed_records` when a
  later source fails; do not silently treat a partial result as a complete
  sample.

## Preserve NoData and band contracts

- Build the canonical static source with `make_static_source()` unless the task
  explicitly changes that contract.
- A successful canonical static cube contains the 63 names in
  `model/static_band_contract.py`, in exact order, and every output band uses
  `-32768` NoData.
- The two Mini-RF bands declare the exact source-only sentinel
  `-3.4028230607370965e38`; mask it before bilinear interpolation and convert it
  to static output NoData.
- WAC and NAC notebook sources explicitly set
  `preserve_source_nodata=True`. This is configured behavior, not behavior
  inferred from modality names.
- Resolve source and output NoData per band. Never infer invalidity from pixel
  magnitude. Reopen written GeoTIFFs when validating persisted NoData metadata,
  especially if a new multiband source can declare different per-band values;
  the GeoTIFF GDAL NoData tag is dataset-wide.

## Maintain the notebook as the public example

- Keep user-editable project/data paths, WAC/NAC PIDs, and AOIs in the **User
  configuration** section. Keep derived indexes, zooms, output paths, display
  settings, validation, and static-source construction below it.
- Preserve repository discovery from the top-level `notebooks/` directory,
  including `/panfs/ccds02/nobackup` to `/explore/nobackup` normalization and
  insertion of `repo_root` into `sys.path`.
- Keep shared WAC/NAC raster directories read-only. Cache their modern
  GeoPackage indexes persistently under `outputs/tiling/indexes/` in each
  user's clone; do not adopt or overwrite legacy indexes in shared data paths.
  The notebook may automatically rebuild only those per-clone caches when
  validation fails. Continue using the declared canonical static index.
- Write each run beneath `outputs/tiling/<RUN_ID>/` without reusing a directory.
- Plot with sentinel pixels converted to `float64` NaN and display no more than
  four tile pairs per AOI unless the user changes that display-only limit.
- Keep expensive or illustrative alternate queries behind
  `RUN_ALTERNATE_QUERIES`. The main WAC/NAC AOI examples may set their PID to
  `None` for per-product discovery. Strict point and explicit-index examples
  should reuse a concrete PID resolved by the AOI query.
- After editing the notebook, validate its JSON, unique cell IDs, and Python
  syntax while accounting for Jupyter magics. Keep committed execution counts
  null and outputs empty unless the user explicitly requests saved outputs.

## Validate proportionally

- Add or update focused tests under `model/tests/` for contract changes. Check
  configuration validation, selection behavior, record metadata, filenames,
  band order, NoData, error behavior, deterministic ordering, and index
  immutability as applicable.
- Use `scripts/shell/all_tasks/sbatch_tiling_modernization_tests.sh` as the
  current template for full Explore validation rather than inventing a second
  environment contract.
- Every new Slurm script uses `#SBATCH --partition=grace` and the
  `lfm-container-ipyleaflet` Apptainer image. Preserve the Explore bind mapping
  used by current repository wrappers.
- Do not claim an HPC or notebook execution occurred when only static or local
  checks ran. Record test evidence and acceptance exceptions accurately in the
  modernization plan when that plan is in scope.
- For modern-versus-legacy regression, compare structured metadata, reopened
  raster properties, masks and values, record order, output hashes when
  determinism is expected, and the before/after source-index inventory.

Keep `TMS/README.md`, the tiling plan, the notebook, helpers, and tests aligned
when a public contract changes. Preserve unrelated user changes in a dirty
worktree.
