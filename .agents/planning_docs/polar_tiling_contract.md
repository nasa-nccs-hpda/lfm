# Polar Tiling Extended Contract

Package migration (2026-10-06): backend imports now use
`lfm.data_processing.chip`, `lfm.data_processing.tiling`, and
`lfm.data_processing.labeling`, with the checkout root on `sys.path`.
Historical phase records retain their original paths. See
[repo_restructure.md](repo_restructure.md) for the file mapping and new test
commands.

This document is the Phase P0 design contract for extending the modern LFM
tiler from numbered Lunar Transverse Mercator (LTM) grids to automatic LTM and
polar routing. It is prospective: until the later implementation and validation
phases are complete, the production behavior documented in
[`TMS/README.md`](../../TMS/README.md) remains authoritative.

The contract covers the public tiling boundary only:

```text
prepared sources + geographic query + optional product selectors
    -> source-index preparation
    -> product discovery
    -> grid routing and tile discovery
    -> per-source grid-aligned cubes
    -> ordered TileCubeRecord results
```

Final chip reprojection, labels, model-ready band assembly, and training remain
downstream concerns.

## P0.1 — Current LTM assumptions and migration inventory

The following inventory records the assumptions that later phases must either
generalize or deliberately preserve.

| Area | Current assumption or behavior | Required migration |
|---|---|---|
| `model/TmsIntersector.py` | Globs all 92 TMS JSON files into one zone dictionary and labels every match as an LTM intersection. The two polar files are therefore loaded, but not correctly supported. | Load definitions through an explicit grid registry, route only eligible query parts to each family, and return canonical grid IDs. |
| `model/TmsZoneDef.py` | Derives a zone by splitting a filename. `tms_LPS_NRG.json` and `tms_LPS_SRG.json` become `N` and `S`; their tile definitions are then looked up as nonexistent `tms_LTM_NRG.json` and `tms_LTM_SRG.json`. Zone intersection assumes an ordinary latitude/longitude rectangle from CRS area-of-use metadata. | Parse grid family and ID explicitly. Do not pass polar definitions through LTM filename or rectangular-zone logic. |
| `model/TmsTileDef.py` | Hard-codes `tms_LTM_<zone>RG.json`, LTM method names/messages, and an LTM zone field. AOI tile discovery transforms only the upper-left and lower-right geographic corners and checks a two-corner geographic tile envelope. | Load by registered grid definition, use grid-neutral projected transforms, densify AOI/tile edges where needed, and preserve the existing LTM numerical path. |
| `model/tiling.py` | Point and explicit-tile entry points require a caller-supplied LTM zone. AOI discovery is automatic but LTM-only. | Add a high-level point/AOI workflow with automatic routing. Retain an expert explicit-address API that requires a complete grid ID. |
| `model/configured_tiler.py` | Instantiates `TmsTileDef` directly, reports `LTM<zone>` in errors, derives source-query bounds from two projected tile corners, and sorts AOI results by `zone`, row, and column. | Consume grid-neutral tile definitions, use seam-safe geographic query parts from densified tile footprints, use grid-neutral diagnostics, and extend deterministic ordering. |
| `model/raster_cube.py` | Type hints and docstrings are LTM-specific, but the GDAL warp/write implementation already consumes a projected tile definition and is mostly grid-neutral. | Generalize its interface language without changing bilinear resampling, exact grid alignment, NoData behavior, compression, permissions, or 512 by 512 output. |
| `model/tiling_results.py` | `TileCubeRecord` stores the address in `zone`; there is no `ltm_zone` field. Filenames always insert an `LTM` prefix, and record/error documentation is LTM-specific. | Preserve `zone` for compatibility, broaden it to the canonical grid ID, add a grid-neutral `grid_id` alias, and generate family-correct filenames. |
| `model/tiling_config.py` | Every `TileSourceConfig` requires an existing index path. Selection is only `product_id` or `all_intersecting`; one `TileConfig` has one zoom for all its sources. | Prepare missing indexes before constructing/running the low-level config. Keep one zoom per generated config and compose separate configs when routed grid families need different default zooms. |
| `model/tiling_policy.py` | A `product_id` source requires a nonempty selector. Product IDs are the filename stem before the first period. `all_intersecting` stacks all matches and does not group products. | Preserve strict explicit selection in the low-level API. Add high-level, modality-aware PID discovery and companion-file grouping before product-scoped tiling. |
| `model/vector_index.py` | Opens indexes read-only and applies one rectangular geographic spatial filter. It assumes a non-wrapping geographic extent. | Keep read-only queries, but allow one or more non-wrapping envelopes derived from a densified grid-tile perimeter and deduplicate returned raster records. |
| `model/vector_index_builder.py` | Explicitly builds a new `.shp` or `.gpkg`, refuses overwrite, and has no ensure/reuse validation, progress bar, logging, staging, or concurrency guard. | Add a high-level ensure-if-missing preparation step with progress, validation, atomic publication, and explicit rebuild semantics. |
| `model/__init__.py` | Exports the low-level config, result, index-builder, and three tiling entry points. | Export new high-level source/request/routing APIs without removing the current low-level exports. |
| `notebooks/tiling_example.ipynb` | Requires existing indexes and nonempty WAC/NAC PIDs; always validates and configures static data; describes only LTM; point and explicit-address examples obtain a zone from an AOI record. | Prepare enabled indexes, accept optional PIDs, expose source inclusion controls, and add automatic polar point/AOI examples under alternate queries. |
| `lfm/all_models/all_tasks/tiling_utils.py` | Provides WAC display defaults, one general LTM zoom default, path validation, run IDs, and a canonical static source requiring an explicit index. | Add grid-family zoom defaults and high-level source preparation while preserving the canonical static band and NoData policy. |
| `lfm/all_models/all_tasks/viz/tiling_viz.py` | Keys and labels records as LTM. Pairing permits only one record per source/tile and requires a dynamic/static pair. | Display canonical grid IDs, distinguish products in keys where necessary, and support combined, dynamic-only, and static-only record sets. |
| Tiling tests | Geometry fixtures are almost entirely LTM `42N`. `test_TmsIntersector` asserts that 92 JSON files load but does not prove that either polar definition can generate a tile. Filename tests require the `LTM` prefix. | Retain all LTM regressions and add north/south polar, threshold, seam, pole, routing, filename, product-discovery, index-preparation, and source-mode coverage. |
| Tiling documentation | `TMS/README.md`, the modernization plan, the agent skill, and the notebook correctly describe production as LTM-only with pre-existing indexes. | Update them only after implementation and validation make the extended path production-supported. |
| Modern chip workflow | Chip preflight rejects geographic coverage beyond `+/-82` degrees. Acquisition/reprojection structures group by `record.zone` and are structurally reusable, but several messages and policies say LTM explicitly. | Keep chip rejection in place during this project. A later, separately validated chip migration may consume polar records through the compatible `zone` field. |
| Deprecated legacy paths | `model/Pipeline.py` and `model/chip_making/chip_utils.py` hard-code LTM filenames, paths, and filename parsing. | Do not extend them. They remain regression or migration inputs, not the target polar API. |

### Confirmed false-positive support signal

`TmsIntersector.zones` currently contains 92 entries, but the polar entries are
stored as `N` and `S`. A low-latitude query does not intersect their areas of
use, so existing LTM tests pass. A polar query can reach their broken LTM file
lookup. The count of 92 is therefore an inventory assertion, not evidence of
polar tiling support.

## P0.2 — Minimal user request and derived defaults

### Easy point/AOI workflow

The high-level public workflow accepts:

- an ordered collection of source descriptions;
- exactly one geographic point or AOI in repository IAU:30100 coordinates;
- an output directory;
- an optional scalar product ID or per-source product-ID mapping;
- `include_dynamic` and `include_static` controls; and
- optional expert overrides for index paths and zooms.

Each source description declares at least:

- a stable source name;
- a source role: `dynamic` or `static`;
- its raster data directory; and
- a built-in modality preset or an explicit equivalent band, selection,
  NoData, and product-resolution policy.

The workflow never infers a modality from a directory name. Built-in WAC, NAC,
and canonical-static presets supply their established band and NoData policy.
An arbitrary modality supplies that policy explicitly.

### Index-path resolution

For each enabled source, resolve its index in this order:

1. Use an explicitly configured `.shp` or `.gpkg` path.
2. Reuse `<data_dir>/output_index.shp` when it exists.
3. For the canonical static preset only, reuse the legacy
   `<data_dir>/db2.shp` when it exists.
4. Otherwise prepare `<data_dir>/output_index.shp`.

Do not scan for and guess among arbitrary index files. Existing indexes are
validated and reused; missing indexes are created before source validation.
The low-level tile-generation and index-query code remains read-only.

Raster discovery accepts an ordered set of glob patterns and defaults to
`*.tif`, `*.tiff`, `*.nc`, and `*.vrt`. Overlapping patterns are deduplicated
before deterministic sorting. The legacy scalar `image_glob` remains a
single-pattern override. NetCDF inputs must be directly readable GDAL raster
datasets; a NetCDF container that exposes only subdatasets must be represented
by a VRT selecting the intended variable. When a VRT and its component rasters
share one directory, callers should narrow the patterns so the logical VRT and
its physical components are not both indexed unintentionally.

New index footprints densify every raster edge before transforming it to
IAU:30100. This curved transformed perimeter is canonical, especially for polar
stereographic sources. A four-corner/five-vertex `gdaltindex` footprint remains
a structural comparison baseline but is not the polar-area truth; polar
accuracy is established by convergence against a higher-resolution densified
perimeter plus matching bounds and AOI query behavior.

A raster whose projected footprint contains either geographic pole cannot be
represented by one ordinary longitude-wrapped perimeter ring without a seam
self-intersection. For index filtering, represent that case as a valid,
conservative full-longitude cap from the densified perimeter's limiting
latitude to `+90` or `-90`. This may admit harmless index false positives at
some longitudes, but it must never exclude actual polar raster coverage. It
does not replace the canonical curved perimeter for rasters that do not contain
a pole.

### Product-selector input

- A scalar `product_id` is a convenience only when exactly one enabled dynamic
  source is product-scoped.
- Multiple product-scoped dynamic sources use a mapping keyed by source name.
- A supplied ID is normalized as nonempty text but otherwise preserved.
- An omitted ID triggers product discovery for each product-scoped dynamic
  source.
- A product ID supplied to a static-only or wholly `all_intersecting` request
  is rejected instead of ignored.

### Source inclusion

`include_dynamic=True` and `include_static=True` are the defaults. At least one
must remain true. An enabled class requires at least one configured source of
that role. Disabled classes are not indexed, validated, queried, written, or
plotted.

Polar static production is not initially available. A polar or cross-threshold
request with `include_static=True` and no verified polar static source fails in
preflight before writing any cubes. It never silently degrades to dynamic-only.
The notebook polar example explicitly sets `include_static=False`.

### Derived zooms and outputs

Built-in modality defaults are selected by grid family:

| Modality | Numbered LTM | `LPS_N` / `LPS_S` | Reason |
|---|---:|---:|---|
| WAC | 5 | 4 | Approximately 75.824 m versus 73.775 m per pixel. |
| Processed 1 m NAC | 11 | 10 | Approximately 1.185 m versus 1.153 m per pixel. |

Canonical static adopts the zoom of the acquisition group in which it is
enabled. A custom modality without a declared default must provide a positive
zoom for every routed grid family. Expert overrides take precedence.

A cross-family request may therefore create multiple internal `TileConfig`
objects at different zoom numbers while retaining comparable physical
resolution. Outputs remain in one caller-supplied run directory with unique,
flat filenames. A timestamped `outputs/tiling/<RUN_ID>/` directory remains the
notebook default.

## P0.3 — Expert explicit-address workflow

An explicit tile request is deliberately separate from automatic routing. It
requires:

```text
(grid_id, zoom_level, tile_x, tile_y)
```

Rules:

- `grid_id` is a numbered LTM ID such as `42N`, or `LPS_N`, or `LPS_S`.
- The zoom must exist in that grid's repository TMS definition.
- Tile indices are zero-based and must fall inside that zoom matrix.
- Explicit-address requests do not apply the 82-degree routing threshold; the
  complete address is authoritative.
- Source preparation and product selection use the same contract as automatic
  queries.
- Existing low-level `zone=` calls remain accepted for compatibility. New
  grid-neutral APIs and documentation use `grid_id`.

The normal notebook does not ask users for this address. It remains an
alternate, advanced example.

## P0.4 — Geographic validation and automatic routing

### Geographic inputs

- Coordinates use the repository `TMS/IAU_30100_2015.wkt` definition and
  traditional GIS axis order.
- Latitude must be finite and inside `[-90, 90]`.
- Longitude must be finite and is normalized to `[-180, 180]`, retaining the
  two endpoints when they communicate a full-longitude AOI.
- A point supplies `lat` and `lon`.
- An AOI retains the existing order: `ul_lat`, `ul_lon`, `lr_lat`, `lr_lon`,
  with `ul_lat > lr_lat` for a nondegenerate area.
- `ul_lon <= lr_lon` is a non-wrapping interval. `ul_lon > lr_lon` is a short
  antimeridian-crossing interval split at `+180/-180`.
- A longitude span of exactly 180 degrees is ambiguous and rejected. A span
  greater than 180 degrees is rejected unless it is the explicit
  `[-180, 180]` full-longitude polar-cap form.
- At an exact pole, all longitudes describe the same point. Point routing uses
  a canonical longitude of zero after validation.

### Region ownership

The routing regions are:

```text
LPS_N:  +82 <= latitude <= +90
LTM:    -82 <  latitude <  +82
LPS_S:  -90 <= latitude <= -82
```

Although the repository TMS metadata overlaps between 80 and 82 degrees, the
automatic workflow intentionally assigns that overlap to numbered LTM grids.
Exactly `+82` and `-82` belong to the corresponding polar grid.

### Points

- `lat >= +82` routes to `LPS_N`.
- `lat <= -82` routes to `LPS_S`.
- Other points route to the one numbered LTM longitude zone in the point's
  hemisphere.

### AOIs

An AOI is intersected with the three routing regions and split at `+82` and
`-82`. Only components having positive area are tiled; a boundary-only touch
does not create an extra family request. Consequently, an AOI whose northern
edge is exactly `+82` but whose interior lies south of it remains LTM-only,
while an AOI extending on both sides is split between LTM and `LPS_N`.

Each LTM part is further split by numbered longitude-zone boundaries. Each
antimeridian-crossing part is represented as two non-wrapping geographic
query pieces. Polar pieces stay in their single polar grid and are projected as
densified polygons, not treated as longitude-band rectangles. Duplicate tile
addresses produced by query splitting are removed.

The selected products are always complete native grid tiles. Pixels in a tile
may extend across the 82-degree routing line or outside the input AOI; the
routing boundary selects grids and tiles but does not clip cube files.

### Tile-to-source index queries

The projected perimeter of a selected tile is densified and transformed to
IAU:30100. Its geographic coverage is expressed as one or more non-wrapping
envelopes for read-only source-index filtering. Envelope false positives are
acceptable because the warp removes non-overlapping pixels; seam-related false
negatives are not. Returned source paths are deduplicated deterministically.

## P0.5 — Optional product discovery and grouping

### Explicit products

The current low-level rule remains unchanged: a source configured with
`selection_mode="product_id"` requires a nonempty selector, and
`all_intersecting` rejects one. Explicit selection produces only the requested
product.

### Omitted products

The high-level workflow handles an omitted PID before invoking the low-level
tiler:

1. Query each enabled product-scoped dynamic index across every routed query
   part.
2. Resolve a PID for every intersecting raster with the source's declared
   product resolver.
3. Group companion rasters by PID. The built-in lunar resolver uses the
   filename stem before the first period, so WAC UV and VIS rasters sharing an
   ID remain one seven-band product.
4. Sort unique PIDs deterministically and log their count and values.
5. Tile each source/PID only where its indexed rasters intersect.

Unrelated dynamic products are never stacked into one cube. The existing
`all_intersecting` behavior remains available for contextual sources and
explicit expert usage, but it is not the meaning of an omitted PID in the easy
workflow.

For a grid tile containing several discovered products, write one dynamic cube
per product-scoped source and PID. Write each static/contextual cube only once
per source and tile, not once per dynamic PID. This avoids duplicate records
and static filename overwrites.

An unresolvable or conflicting dynamic filename is reported with its source
and path. A required dynamic source with no discovered product fails clearly;
an optional source may return no records according to its existing sparse
coverage policy.

## P0.6 — Source-mode behavior

| Mode | Required request behavior | Output behavior |
|---|---|---|
| Dynamic plus static | Both inclusion flags true and at least one configured source in each role. | Dynamic outputs are product-scoped; contextual outputs are written once per source/tile. Built-in composition orders dynamic sources before static. |
| Dynamic-only | `include_dynamic=True`, `include_static=False`. | No static directory or index is resolved, validated, queried, or plotted. This is the initial supported polar example. |
| Static-only | `include_dynamic=False`, `include_static=True`. | No product ID is accepted or discovered. Each contextual source is written once per grid/tile. |
| Neither | Both flags false. | Reject during request validation before filesystem mutation. |

Caller-declared source order remains authoritative within each enabled role.
Built-in notebook composition places enabled dynamic sources before canonical
static. Required and optional source behavior remains explicit, and structured
errors retain records completed before a later source fails.

## P0.7 — Grid-neutral results, ordering, and filenames

### Structured address

`TileCubeRecord.zone` remains the stored constructor field for backward
compatibility. Its widened meaning is the canonical grid ID:

- numbered LTM: `1N` through `45N` and `1S` through `45S`;
- polar north: `LPS_N`;
- polar south: `LPS_S`.

A read-only `record.grid_id` alias returns `record.zone`. New public examples
prefer `grid_id`; existing chip and tiling code using `zone` continues to work.
The same compatibility rule applies to structured source errors. There is no
existing `ltm_zone` record field to preserve.

A complete structured cube identity is:

```text
(grid_id, zoom_level, tile_x, tile_y, source_name, product_id)
```

Contextual records use `product_id=None`. Callers must use these fields rather
than parse filenames.

### Deterministic result order

The high-level result list is sorted by:

1. canonical `grid_id` text;
2. tile row (`tile_y`);
3. tile column (`tile_x`);
4. configured source order; and
5. product ID text within a product-scoped source, with `None` sorting before
   text only when a source legitimately produces both forms.

For existing LTM requests with one PID per dynamic source, this retains the
current zone/row/column/source order.

### Filenames

Numbered LTM filenames remain byte-for-byte compatible in form:

```text
Cube-<source>-LTM<zone>_Zoom-<zoom>_Tile-<x>-<y>[_Product-<pid>].tif
```

Polar filenames use the canonical polar grid ID directly and do not prepend
`LTM`:

```text
Cube-<source>-LPS_N_Zoom-<zoom>_Tile-<x>-<y>[_Product-<pid>].tif
Cube-<source>-LPS_S_Zoom-<zoom>_Tile-<x>-<y>[_Product-<pid>].tif
```

Source names and PIDs continue through the existing safe filename-component
normalization. Output paths must be collision-free across grid, zoom, tile,
source, and PID.

### Preserved raster contract

Every output remains a 512 by 512 tiled, LZW-compressed BigTIFF with the exact
selected grid CRS, transform, cell size, band names, and configured output
NoData. Tiling resampling remains bilinear. The canonical 63-band static and
WAC/NAC source-NoData contracts do not change.

## P0.8 — Acceptance cases

Later phases must turn these cases into focused automated fixtures before the
extended workflow is considered complete.

| ID | Request | Expected contract result |
|---|---|---|
| A1 | Existing WAC AOI below 82 degrees, explicit PID, dynamic plus static | Only numbered LTM grids; WAC at zoom 5; unchanged LTM filenames, pixels, NoData, structured fields, and order; one static record per tile. |
| A2 | Existing 1 m NAC AOI below 82 degrees, explicit PID, dynamic plus static | Only numbered LTM grids; NAC/static at zoom 11; sparse optional NAC behavior remains explicit. |
| A3 | Point at `81.999999` north or south | Numbered LTM route selected from longitude and hemisphere. |
| A4 | Point exactly at `+82` and point exactly at `-82` | `LPS_N` and `LPS_S`, respectively; no LTM duplicate. |
| A5 | Point at each pole | Correct polar grid; input longitude validates but canonical longitude zero is used for the coincident pole location. |
| A6 | AOI wholly north of `+82`, dynamic-only WAC | `LPS_N` only at polar zoom 4; polar filename and CRS; no static path touched. |
| A7 | AOI wholly south of `-82`, dynamic-only NAC | `LPS_S` only at polar zoom 10; polar filename and CRS; no static path touched. |
| A8 | AOI crossing `+82` with dynamic-only WAC | Numbered northern LTM part at zoom 5 plus `LPS_N` part at zoom 4; duplicate addresses removed; full tiles retained. |
| A9 | AOI crossing `-82` with dynamic-only WAC | Numbered southern LTM part at zoom 5 plus `LPS_S` part at zoom 4. |
| A10 | AOI whose edge touches but does not cross `+82` or `-82` | Only the grid family containing the AOI's positive-area interior. |
| A11 | Non-polar AOI crossing `+180/-180` | Split across the applicable numbered LTM edge zones; structured duplicates removed; no world-spanning envelope. |
| A12 | Polar AOI crossing `+180/-180` | Two geographic query pieces routed to one polar grid; polar tile addresses deduplicated. |
| A13 | Explicit full-longitude polar-cap AOI | One polar family, densified projected geometry, finite valid tile addresses, and no ambiguous-longitude rejection. |
| A14 | Omitted PID with one intersecting WAC product | One seven-band WAC cube per covered tile, with UV/VIS companions grouped and PID populated. |
| A15 | Omitted PID with several intersecting products | Separate dynamic cube per PID/tile in deterministic order; one static cube per tile when static is enabled; no overwrite. |
| A16 | Omitted PID with no required dynamic match | Clear required-source/product-discovery failure before a misleading successful result. |
| A17 | Dynamic-only, static-only, and combined requests | Only enabled roles are prepared and processed; disabling both fails before filesystem mutation. |
| A18 | Polar request with static enabled but no verified polar static source | Preflight failure before cube output; no silent static omission. |
| A19 | Explicit `LPS_N` and `LPS_S` tile addresses | Requested valid tile is produced; invalid zoom or matrix index is rejected. |
| A20 | Missing source index | Deterministic raster inventory, stdout progress, validated staged index publication, then read-only tiling query. |
| A21 | Existing valid source index | Validation and reuse with no content or timestamp mutation caused by tile generation. |
| A22 | Repeated identical request | Identical ordered record metadata and byte-identical outputs wherever the existing determinism contract applies. |
| A23 | Current chip workflow receives a polar target before downstream migration | Existing typed unsupported-polar preflight remains in force; this tiling project does not silently expand chip support. |

## P0 completion boundary

Phase P0 freezes behavior and test expectations; it does not claim that the
runtime implements them. Runtime production support remains numbered-LTM-only
until Phases P1 through P9 are implemented and validated. Any later contract
change must update this document and the corresponding planning status before
dependent implementation proceeds.
