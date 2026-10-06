# Armstrong Lunar Tiling Scheme

This directory contains the coordinate reference system (CRS) and tile matrix
metadata used by the Lunar Foundation Model (LFM) implementation of the
Armstrong tiling scheme. The scheme gives a lunar location a stable,
multiresolution address:

```text
(projection zone, zoom level, tile column, tile row)
```

Raster products that have the same address are written on the same projected
pixel grid. This lets LFM pair modalities such as WAC or NAC imagery with the
corresponding static lunar context without requiring the source rasters to
share a native projection, resolution, or footprint.

This document describes the scheme as represented and executed by this
repository. The runnable example is
[`notebooks/tiling_example.ipynb`](../notebooks/tiling_example.ipynb).

## Lunar geographic CRS

Geographic inputs use the Moon (2015) spherical, ocentric latitude/longitude
CRS identified as **IAU:30100 (2015)**. Its sphere has a radius of 1,737,400
meters. The repository-owned definition is
[`IAU_30100_2015.wkt`](IAU_30100_2015.wkt); tiling code loads this file rather
than embedding another WKT definition in Python.

The public AOI API accepts geographic bounds in this order:

```python
{
    "ul_lat": upper_left_latitude,
    "ul_lon": upper_left_longitude,
    "lr_lat": lower_right_latitude,
    "lr_lon": lower_right_longitude,
}
```

Longitude is the projected `x` axis and latitude is the projected `y` axis.
The code explicitly selects traditional GIS axis order when it constructs GDAL
coordinate transformations.

## LTM zones

Most of the Moon is divided into Lunar Transverse Mercator (LTM) zones. The
repository contains 90 LTM zone definitions:

- 45 northern zones named `1N` through `45N`, covering 0° to 82° latitude.
- 45 southern zones named `1S` through `45S`, covering -82° to 0° latitude.
- Each numbered longitude band is 8° wide.

Zone 1 covers -180° to -172° longitude and has a central meridian of -176°.
Each subsequent zone advances 8°. For numbered zone `n`:

```text
west longitude    = -180 + 8 × (n - 1)
east longitude    = west longitude + 8
central meridian  = west longitude + 4
```

For example, zone `42N` covers 0° to 82° latitude and 148° to 156° longitude,
with a central meridian of 152°. An address must include the hemisphere:
`42N` and `42S` are different projected grids.

Every LTM CRS uses the IAU:30100 lunar sphere as its geographic base, a
Transverse Mercator scale factor of 0.999, and a false easting of 250,000
meters. Northern and southern definitions use different false northings as
recorded in their JSON CRS definitions.

## Polar grids: LPS_N and LPS_S

The Armstrong metadata also defines northern and southern Lunar Polar
Stereographic grids. LFM uses the canonical public IDs `LPS_N` and `LPS_S`:

- `RG/tms_LPS_NRG.json` is the `LPS_N` definition. It covers 80° to
  90° latitude and uses a Polar Stereographic projection centered on +90°.
- `RG/tms_LPS_SRG.json` is the `LPS_S` definition. It covers -90° to
  -80° latitude and uses a Polar Stereographic projection centered on -90°.

Both use a central meridian of 0°, a scale factor of 0.994, and false easting
and northing values of 500,000 meters. Their coverage overlaps the LTM zones
between 80° and 82°, so the metadata provides continuity into both polar
regions.

The polar tile matrices use 512×512-pixel tiles and define zoom levels 1
through 15. At zoom 1, each polar matrix is 2×2 tiles; both dimensions double
at each subsequent zoom.

The polar grids were not used by the original LFM tiling workflow because their
geometry differs from the numbered LTM zones. LTM zones are rectangular
longitude bands represented by separate northern and southern Transverse
Mercator grids. A polar stereographic grid instead surrounds a pole, where
meridians converge and its geographic footprint does not behave like an LTM
longitude-band rectangle. The original AOI intersection, zone identifiers,
output naming, notebook examples, and regression tests were implemented around
the LTM geometry and `number + hemisphere` addresses such as `42N`.

The grid registry and geographic router recognize `LPS_N` and `LPS_S`, use an
inclusive 82-degree automatic-routing threshold, and partition geographic AOIs
without passing polar definitions through the LTM filename logic. The low-level
tiler now also implements polar projection, tile intersection, explicit
addresses, seam-safe source-index envelopes, and polar cube filenames. This
low-level path passed its supported-container regression gate. Source-mode
composition, per-family default zooms, and the public easy workflow remain
later phases.

## Zoom levels and tile matrices

Each LTM zone JSON file defines zoom levels 1 through 26. A tile is always
**512×512 pixels**. Increasing the zoom by one:

- halves the projected cell size;
- halves the ground span of one tile in both dimensions; and
- doubles the matrix width and height.

For an LTM zoom level `z`, the matrix contains `2^z` columns and `2^(z+1)`
rows. The exact cell size, scale denominator, origin, and matrix dimensions are
authoritative in the selected zone JSON file. Northern and southern values
have very small numerical differences, so code reads the metadata rather than
recomputing it.

Representative northern LTM values are:

| Zoom | Cell size (m/pixel) | Tile span | Matrix (columns × rows) |
|---:|---:|---:|---:|
| 1 | 1,213.1889 | 621.153 km | 2 × 4 |
| 4 | 151.6486 | 77.644 km | 16 × 32 |
| 5 | 75.8243 | 38.822 km | 32 × 64 |
| 8 | 9.4780 | 4.853 km | 256 × 512 |
| 9 | 4.7390 | 2.426 km | 512 × 1,024 |
| 10 | 2.3695 | 1.213 km | 1,024 × 2,048 |
| 11 | 1.1848 | 606.594 m | 2,048 × 4,096 |
| 12 | 0.5924 | 303.297 m | 4,096 × 8,192 |
| 26 | 0.0000362 | 0.0185 m | 67,108,864 × 134,217,728 |

The existence of a zoom in the metadata does not mean it is appropriate for a
particular sensor. Select a zoom close to the source resolution unless a
downstream alignment contract requires otherwise. The example notebook uses
zoom 5 for WAC and zoom 11 for processed NAC imagery with a native 1 m pixel
size. A single `TileConfig` has one zoom level shared by all sources in that
configuration; modalities that need different grids should use separate
configurations.

## Tile addressing

Tiles use zero-based `(tile_x, tile_y)` indices:

- `tile_x` is the matrix column and increases eastward from the top-left
  origin.
- `tile_y` is the matrix row and increases southward from the top-left origin.

Given the projected top-left matrix origin `(origin_x, origin_y)`, cell size,
and the fixed 512-pixel tile size, the projected tile bounds are calculated as:

```text
xmin = origin_x + tile_x       × 512 × cell_size
xmax = origin_x + (tile_x + 1) × 512 × cell_size
ymax = origin_y - tile_y       × 512 × cell_size
ymin = origin_y - (tile_y + 1) × 512 × cell_size
```

Consequently, `(zone, zoom, tile_x, tile_y)` completely determines a tile's
CRS, extent, resolution, dimensions, and geotransform. A tile index without its
zone and zoom is not a complete address.

## Files in this directory

- [`IAU_30100_2015.wkt`](IAU_30100_2015.wkt) is the shared repository geographic
  CRS definition.
- [`RG/tms_LTM_*RG.json`](RG/) contains the 90 LTM zone and tile matrix
  definitions.
- `RG/tms_LPS_NRG.json` and `RG/tms_LPS_SRG.json` contain the `LPS_N` and
  `LPS_S` tile matrix definitions described above.
- [`RG/tile_database.gpkg`](RG/tile_database.gpkg) is an auxiliary geographic
  inventory of the 728 zoom-1 tiles across the 90 LTM and two polar grids. The
  current configuration-driven tiler does not use this GeoPackage to resolve
  normal AOI queries. The registry and grid-neutral tile-definition factory
  read the JSON files directly.

Do not confuse `tile_database.gpkg` with a raster source index. Each configured
data modality has its own `.shp` or `.gpkg` index whose features describe
source-raster coverage and whose location field identifies the raster file.
The high-level workflow validates and reuses an existing declared index or
creates it when it is missing. A caller may explicitly mark an
application-owned GeoPackage cache as replaceable when invalid or stale;
shared and legacy indexes remain protected. Low-level tile queries consume the
prepared index read-only.

Missing-index creation parallelizes raster footprint inspection with isolated
worker processes while retaining deterministic, serial GeoPackage writes. The
worker count defaults to `SLURM_CPUS_PER_TASK` and falls back to one outside a
Slurm allocation. Expert callers can set `index_worker_count=1` on a source
definition to force the earlier serial behavior.

The example notebooks treat shared raster collections and indexes as
read-only. The canonical WAC, NAC, and static directories use their existing
`output_index.gpkg` files. If a user changes one of those data directories,
the notebook resolves a persistent GeoPackage cache beneath its own output
tree instead. Invalid or stale user-owned caches are rebuilt atomically only
after a replacement validates; shared source indexes are never automatically
replaced.

## How LFM implements the scheme

The modern entry points are exported from [`lfm.data_processing.tiling`](../lfm/data_processing/tiling/__init__.py). The
strict functions live in [`lfm/data_processing/tiling/tiling.py`](../lfm/data_processing/tiling/tiling.py), optional AOI
product discovery lives in
[`lfm/data_processing/tiling/product_tiling.py`](../lfm/data_processing/tiling/product_tiling.py), and the easiest
automatic entry point lives in
[`lfm/data_processing/tiling/tiling_workflow.py`](../lfm/data_processing/tiling/tiling_workflow.py):

- `create_tiles_for_aoi(...)` routes geographic bounds and is the strict
  low-level path: every `product_id` source requires an explicit selector and
  the one configured zoom is applied to each routed grid.
- `create_tiles_for_point(...)` processes the tile containing a point in an
  explicitly supplied `zone` or `grid_id`.
- `create_tiles_for_index(...)` processes an explicit grid/zoom/tile address.
- `create_tiles_for_aoi_by_product(...)` is the high-level optional-product
  path. A configured PID selects one observation; `None` or an omitted mapping
  entry discovers every intersecting PID for that product-scoped source.
- `create_tiles_for_query(...)` prepares enabled indexes, routes AOI or point
  queries, applies family-specific zooms, and accepts exact or omitted product
  IDs. A product source that does not intersect the query emits a
  `ProductAOIWarning` and is skipped. If no dynamic product source remains,
  the call returns an empty list without generating contextual static tiles.

[`lfm/data_processing/tiling/grid_registry.py`](../lfm/data_processing/tiling/grid_registry.py) validates the 90 numbered
LTM definitions plus `LPS_N` and `LPS_S` and exposes their CRS, geographic
coverage, and tile matrices without inferring every grid from an LTM filename.
[`lfm/data_processing/tiling/grid_router.py`](../lfm/data_processing/tiling/grid_router.py) validates and normalizes
geographic requests, routes points at `>= +82` to `LPS_N` and at `<= -82` to
`LPS_S`, and partitions AOIs at the polar thresholds, equator, longitude-zone
edges, and antimeridian. The grid-neutral tile-definition factory and automatic
workflow consume these routing results.

The implementation follows this sequence:

1. `TileConfig` declares the output directory, zoom level, and ordered raster
   sources. Each `TileSourceConfig` declares its data directory, vector index,
   location field, raster selection rule, bands, NoData policy, and whether the
   source is required.
2. [`lfm/data_processing/tiling/grid_router.py`](../lfm/data_processing/tiling/grid_router.py) partitions the AOI into
   canonical numbered LTM, `LPS_N`, and `LPS_S` query parts.
3. [`lfm/data_processing/tiling/grid_tile_def.py`](../lfm/data_processing/tiling/grid_tile_def.py) retains
   [`TmsTileDef`](../lfm/data_processing/tiling/TmsTileDef.py) for proven LTM geometry and uses a
   dedicated polar definition for densified stereographic intersection. Both
   paths require at least 10 meters of overlap in both projected dimensions,
   avoiding tiles touched only by insignificant boundary effects.
4. For each tile, its projected perimeter is transformed back to lunar
   longitude and latitude. Polar perimeters are densified and expressed as one
   or more non-wrapping envelopes. [`lfm/data_processing/tiling/vector_index.py`](../lfm/data_processing/tiling/vector_index.py)
   applies those envelopes as read-only OGR spatial filters and deduplicates
   returned raster paths. The low-level tiler treats source indexes as
   read-only. The automatic workflow may create or atomically rebuild a
   user-owned index during preparation, but protected shared indexes are only
   validated and reused.
5. The strict API makes `product_id` sources select the requested observation.
   The high-level AOI API can instead resolve product IDs through each source's
   configured resolver, group companion rasters such as WAC UV/VIS files, and
   run the strict path separately for every PID. Unrelated observations are
   never stacked. `all_intersecting` contextual sources run only once per tile,
   even when several dynamic products are discovered. If a geographically
   valid query has no intersecting product for one dynamic source, the
   high-level workflow emits `ProductAOIWarning` and may continue another
   runnable dynamic source. If none is runnable, it returns no records and
   skips contextual static for that mixed query; callers must not interpret the
   absence of an exception as complete downstream coverage.
6. [`lfm/data_processing/tiling/raster_cube.py`](../lfm/data_processing/tiling/raster_cube.py) uses GDAL to warp every
   selected raster onto the exact 512×512 routed tile grid. Tiling uses bilinear
   resampling, preserves or normalizes NoData according to each source's
   configuration, and maintains deterministic band ordering.
7. One multiband GeoTIFF is written per source and tile. Files use tiled,
   LZW-compressed BigTIFF output and store the grid CRS, geotransform, band
   names, and output NoData metadata.
8. Results are returned as ordered `TileCubeRecord` objects. Records are sorted
   by zone, tile row, and tile column, with sources processed in configuration
   order. Downstream code therefore does not need to recover metadata by
   parsing filenames.

Numbered LTM filenames retain their existing contract:

```text
Cube-<source>-LTM<zone>_Zoom-<zoom>_Tile-<x>-<y>[_Product-<id>].tif
```

For example:

```text
Cube-wac-LTM42N_Zoom-5_Tile-1-62_Product-M1187363083CE.tif
Cube-static-LTM42N_Zoom-5_Tile-1-62.tif
```

Polar filenames use the canonical grid identifier directly:

```text
Cube-<source>-LPS_N_Zoom-<zoom>_Tile-<x>-<y>[_Product-<id>].tif
Cube-<source>-LPS_S_Zoom-<zoom>_Tile-<x>-<y>[_Product-<id>].tif
```

The matching address shows that these two source cubes share a pixel grid. A
"datacube" here is the multiband file for one configured source on one tile;
different source modalities remain separate files so their band and NoData
contracts remain explicit.

## Configuration example

```python
from pathlib import Path

from lfm.data_processing.tiling import (
    TileConfig,
    TileSourceConfig,
    compose_tile_sources,
    create_tiles_for_aoi_by_product,
)

nac = TileSourceConfig(
    name="nac",
    data_dir=Path("/path/to/nac"),
    index_path=Path("/path/to/nac/output_index.shp"),
    location_field="location",
    selection_mode="product_id",
    resampling="bilinear",
    preserve_source_nodata=True,
    required=False,
)

config = TileConfig(
    output_dir=Path("/path/to/output"),
    zoom_level=11,
    sources=compose_tile_sources(
        dynamic_sources=(nac,),
        include_static=False,
    ),
)

records = create_tiles_for_aoi_by_product(
    config,
    ul_lat=1.0786543156953,
    ul_lon=149.752054273755,
    lr_lat=1.0586543156953,
    lr_lon=149.772054273755,
    # Use an exact string for one product, or None to discover all matches.
    product_ids={"nac": None},
)
```

`compose_tile_sources()` is the source-mode boundary used by the developing
easy workflow. `include_dynamic=True` and `include_static=True` are its
defaults, and at least one must remain enabled. Each enabled class requires at
least one source; disabled collections are not inspected, indexed, validated,
queried, or written. Combined configurations always place dynamic sources
before static sources. Static sources must use `all_intersecting` and never
accept a product ID. Dynamic-only operation is supported on numbered LTM and
polar grids; automatic multi-grid/default-zoom orchestration is Phase P6.

The notebook adds the canonical 63-band static source to this configuration so
that NAC and static cubes are written at the same zoom-11 addresses. Static
output bands use the repository's standardized `-32768` destination NoData
value; dynamic sources can preserve their own source NoData value.

## Relationship to model-ready chips

Tile cubes retain bands that warp entirely to NoData. Named bands excluded by
spatial coverage are filled only after the tiler verifies their existence in
readable indexed raster metadata. This preserves the canonical 63-band static
layout (all empty channels use -32768), warns about coverage and lets later
tiles continue. Unknown band names and unreadable/missing indexed rasters still
fail. The lazy metadata inventory is read-only and cached within a tiler run.

Lunar-grid cubes are intermediate, spatially standardized products. They are
not necessarily the final training samples. The chip workflow accepts numbered
LTM and now has initial single-region polar support, pending HPC acceptance.
See `notebooks/chip_polar_example.ipynb`: dynamic chips retain the original
source lattice; static-only polar chips use a zero-anchored 100 m LPS grid.
Polar antimeridian chips use two non-wrapping acquisition queries and one
continuous projected output grid. Structured records are deduplicated before
mosaicking; the tiler also deduplicates addresses within each AOI call.
Cross-82 seams and pole-containing chips remain rejected. Antimeridian chip
HPC validation remains pending.
It can group matching cube addresses, merge adjacent tiles, reproject them onto
a label or reference-image grid, clip them to the desired area, and select or
combine bands for a particular machine-learning dataset.

For implementation history and regression details, see
[`docs/tiling_modernization_plan.md`](../docs/tiling_modernization_plan.md).
