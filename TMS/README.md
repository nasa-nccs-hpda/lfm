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

## Polar grids: LPN and LPS

The Armstrong metadata also defines Lunar Polar North (LPN) and Lunar Polar
South (LPS) grids. The repository groups both definitions under the `LPS`
filename prefix and distinguishes them with `N` and `S`:

- `RG/tms_LPS_NRG.json` is the northern, or LPN, definition. It covers 80° to
  90° latitude and uses a Polar Stereographic projection centered on +90°.
- `RG/tms_LPS_SRG.json` is the southern, or LPS, definition. It covers -90° to
  -80° latitude and uses a Polar Stereographic projection centered on -90°.

Both use a central meridian of 0°, a scale factor of 0.994, and false easting
and northing values of 500,000 meters. Their coverage overlaps the LTM zones
between 80° and 82°, so the metadata provides continuity into both polar
regions.

The polar tile matrices use 512×512-pixel tiles and define zoom levels 1
through 15. At zoom 1, each polar matrix is 2×2 tiles; both dimensions double
at each subsequent zoom.

LPN and LPS were not used by the current LFM tiling workflow because their
geometry differs from the numbered LTM zones. LTM zones are rectangular
longitude bands represented by separate northern and southern Transverse
Mercator grids. A polar stereographic grid instead surrounds a pole, where
meridians converge and its geographic footprint does not behave like an LTM
longitude-band rectangle. The current AOI intersection, zone identifiers,
output naming, notebook examples, and regression tests were implemented around
the LTM geometry and `number + hemisphere` addresses such as `42N`.

The polar JSON files and their zoom-1 features remain in the repository as
Armstrong scheme metadata, but their presence does not mean the modern public
tiling API supports polar production. Adding that support requires a dedicated
polar intersection and addressing path, followed by separate LPN/LPS
regression tests; the files should not simply be passed through the LTM path.

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
- `RG/tms_LPS_NRG.json` and `RG/tms_LPS_SRG.json` contain the LPN and LPS tile
  matrix definitions described above.
- [`RG/tile_database.gpkg`](RG/tile_database.gpkg) is an auxiliary geographic
  inventory of the 728 zoom-1 tiles across the 90 LTM and two polar grids. The
  current configuration-driven tiler does not use this GeoPackage to resolve
  normal AOI queries; it reads the zone JSON files through `TmsIntersector`.

Do not confuse `tile_database.gpkg` with a raster source index. Each configured
data modality has its own existing `.shp` or `.gpkg` index whose features
describe source-raster coverage and whose location field identifies the raster
file.

## How LFM implements the scheme

The modern entry points are defined in [`model/tiling.py`](../model/tiling.py):

- `create_tiles_for_aoi(...)` discovers all LTM tiles intersecting geographic
  bounds.
- `create_tiles_for_point(...)` processes the tile containing a point in an
  explicitly supplied zone.
- `create_tiles_for_index(...)` processes an explicit zone/zoom/tile address.

The implementation follows this sequence:

1. `TileConfig` declares the output directory, zoom level, and ordered raster
   sources. Each `TileSourceConfig` declares its data directory, vector index,
   location field, raster selection rule, bands, NoData policy, and whether the
   source is required.
2. [`TmsIntersector`](../model/TmsIntersector.py) loads the zone JSON metadata
   and finds every zone intersecting an AOI.
3. [`TmsZoneDef`](../model/TmsZoneDef.py) delegates the AOI to the requested
   zoom matrix. [`TmsTileDef`](../model/TmsTileDef.py) constructs the LTM/IAU
   transformations and resolves the intersecting tile columns and rows. The
   AOI path requires at least 10 meters of overlap in both projected
   dimensions, avoiding tiles touched only by insignificant boundary effects.
4. For each tile, the LTM extent is transformed back to lunar longitude and
   latitude. [`model/vector_index.py`](../model/vector_index.py) applies that
   extent as a read-only OGR spatial filter to each source index. The tiler
   never creates, refreshes, or modifies these source indexes.
5. `product_id` sources select the requested observation, while
   `all_intersecting` sources include all indexed rasters intersecting the tile.
   This is how sparse dynamic imagery and global contextual layers can use the
   same tiling code.
6. [`model/raster_cube.py`](../model/raster_cube.py) uses GDAL to warp every
   selected raster onto the exact 512×512 LTM tile grid. Tiling uses bilinear
   resampling, preserves or normalizes NoData according to each source's
   configuration, and maintains deterministic band ordering.
7. One multiband GeoTIFF is written per source and tile. Files use tiled,
   LZW-compressed BigTIFF output and store the LTM CRS, geotransform, band
   names, and output NoData metadata.
8. Results are returned as ordered `TileCubeRecord` objects. Records are sorted
   by zone, tile row, and tile column, with sources processed in configuration
   order. Downstream code therefore does not need to recover metadata by
   parsing filenames.

The generic filename contract is:

```text
Cube-<source>-LTM<zone>_Zoom-<zoom>_Tile-<x>-<y>[_Product-<id>].tif
```

For example:

```text
Cube-wac-LTM42N_Zoom-5_Tile-1-62_Product-M1187363083CE.tif
Cube-static-LTM42N_Zoom-5_Tile-1-62.tif
```

The matching address shows that these two source cubes share a pixel grid. A
"datacube" here is the multiband file for one configured source on one tile;
different source modalities remain separate files so their band and NoData
contracts remain explicit.

## Configuration example

```python
from pathlib import Path

from model import TileConfig, TileSourceConfig, create_tiles_for_aoi

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
    sources=(nac,),
)

records = create_tiles_for_aoi(
    config,
    ul_lat=1.0786543156953,
    ul_lon=149.752054273755,
    lr_lat=1.0586543156953,
    lr_lon=149.772054273755,
    selectors={"nac": "M1117899885LE"},
)
```

The notebook adds the canonical 63-band static source to this configuration so
that NAC and static cubes are written at the same zoom-11 addresses. Static
output bands use the repository's standardized `-32768` destination NoData
value; dynamic sources can preserve their own source NoData value.

## Relationship to model-ready chips

LTM cubes are intermediate, spatially standardized products. They are not
necessarily the final training samples. The chip-creation workflow can group
matching cube addresses, merge adjacent tiles, reproject them onto a label or
reference-image grid, clip them to the desired area, and select or combine
bands for a particular machine-learning dataset.

For implementation history and regression details, see
[`docs/tiling_modernization_plan.md`](../docs/tiling_modernization_plan.md).
