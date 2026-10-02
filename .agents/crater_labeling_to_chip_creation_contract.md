# Crater Labeling to Chip Creation Contract

This document specifies the interoperability boundary between
`notebooks/crater_labeling.ipynb`, `lfm/labeling/craters.py`, and the modern
chip-creation pipeline. It is a target contract, not a description of behavior
that is already implemented.

The intended user workflow is:

```text
projected lunar source GeoTIFF
  -> crater-labeling notebook
  -> reviewed crater GeoPackage
  -> AOI-first chip-creation notebook
  -> target-sized chip GeoTIFF + instance-label NPZ
```

The GeoPackage is a reusable vector annotation source. It is not the final
training label. Chip creation clips and rasterizes it onto each request's exact
`TargetGrid`, validates the resulting instance archive, and atomically
publishes that `.npz` beside the chip.

## Current labeling output

The current crater-labeling backend:

- opens one projected lunar GeoTIFF and keeps accepted crater geometry in
  source pixel coordinates while the notebook is active;
- converts accepted polygons to the source raster's native projected CRS when
  it writes them;
- atomically replaces one file named
  `<raster-stem>_label_craters.gpkg`;
- writes one `craters` polygon layer with `crater_id`, `method`, `source`,
  `band`, `seed_col`, `seed_row`, and `area_native`; and
- reloads that file only when it contains exactly the `craters` layer, has the
  same CRS and source-raster path, and contains valid polygons.

This preserves annotation geometry and CRS, but it is not yet sufficient for
safe chip creation. The extent of the crater features is not the extent that a
scientist reviewed. In particular, an AOI with no crater features must not be
interpreted as verified background merely because it does not intersect a
polygon.

## GeoPackage v1 contract

### Required layers

A chip-ready crater GeoPackage contains exactly these application layers:

1. `craters`: zero or more crater annotations.
2. `label_coverage`: one or more polygons describing areas explicitly reviewed
   for crater presence or absence.

Normal GeoPackage system tables do not count as application layers. Unknown
application layers are rejected by default so a misspelled layer cannot be
silently ignored. A future typed configuration may explicitly select a known
layer, but the notebook default remains `craters` plus `label_coverage`.

Both layers must have an embedded, projected, lunar-compatible CRS. Their CRS
definitions must be equivalent, not merely share an authority-code string.

### `craters` layer

Geometry must be non-empty, valid `Polygon` or `MultiPolygon` data with positive
native area. The following properties are required:

| Property | Type | Meaning |
|---|---|---|
| `crater_id` | integer | Stable, positive, unique source annotation ID |
| `method` | string | Labeling method such as `circle`, `ellipse`, or `edge_circle` |
| `source` | string | Original source-raster path for provenance and notebook resume checks |
| `band` | integer | One-based source band used while labeling |
| `seed_col` | float | Source-raster seed column |
| `seed_row` | float | Source-raster seed row |
| `area_native` | float | Polygon area in squared native CRS units |

`crater_id` controls deterministic rasterization order. File/driver iteration
order must not affect the output. Existing fields retain their current meaning
and are not repurposed for chip-space coordinates.

### `label_coverage` layer

Coverage geometry is the scientific assertion that the enclosed area has been
reviewed. No crater polygon inside certified coverage means background; no
crater polygon outside coverage means unknown.

The first implementation supports one coverage polygon equal to the full
source-raster footprint. The footprint is formed from the four transformed
raster corners, not from an axis-aligned bounds envelope, so rotated grids are
represented accurately. The schema permits multiple non-overlapping reviewed
polygons later without changing the chip-side containment rule.

Each coverage feature has these required properties:

| Property | Type | Meaning |
|---|---|---|
| `schema_version` | string | Contract version; initially `1` |
| `review_status` | string | `in_progress` or `complete` |
| `source_id` | string | Stable source-scene identity, normally the source raster stem |
| `product_id` | string/null | Labeling-backdrop WAC/NAC product ID when known |
| `source` | string | Original source-raster path for provenance |
| `source_width` | integer | Source raster width at labeling time |
| `source_height` | integer | Source raster height at labeling time |
| `transform` | string | JSON array of the six affine coefficients |
| `coverage_kind` | string | Initially `full_raster` |

The layer CRS already carries the spatial reference, so CRS WKT need not be
duplicated in a property. The source path is provenance, not portable
identity; moving the GeoPackage or raster does not change `source_id`.

### Review lifecycle

Accepted-crater autosaves remain recoverable working state. They write
`review_status=in_progress`. Chip creation rejects that status by default.

The labeling notebook must expose an explicit **Mark ready for chip creation**
operation. It asks the scientist to attest that the declared coverage was
reviewed, then atomically writes `review_status=complete`. Accepting, editing,
or undoing a crater after that operation returns the file to `in_progress`
until it is marked ready again.

This is deliberately separate from ordinary autosave/export. A partially
annotated file must be easy to resume without being accidentally consumed as
complete training truth.

Legacy GeoPackages containing only `craters` remain loadable by the labeling
notebook for migration, but chip creation reports
`label/missing_certified_coverage`. The scientist can reopen the original
raster, review it, and mark it ready. Chip creation must not infer reviewed
coverage from the union or bounds of crater polygons.

## Label association and identity

GeoPackage input uses the typed `LabelInput` association planned for AOI-first
chip creation:

- `path` points to the `.gpkg`;
- `kind` is `instance_vector`;
- `relation` is `clip_to_target`;
- `vector_layer` defaults to `craters`;
- `coverage_layer` defaults to `label_coverage`;
- `source_scene_identity` defaults from the coverage metadata but may be
  explicitly asserted by the caller; and
- `identity_policy` is either `explicit_spatial` or `same_product`.

The GeoPackage filename is a convenience, not the identity contract. Chip
creation validates metadata inside the file:

- an explicit `source_scene_identity` must equal `source_id`;
- all coverage features used by one input must agree on source identity,
  product identity, schema version, source grid metadata, and review status;
  and
- one parent GeoPackage may feed any number of uniquely identified AOI
  requests whose target footprints are fully covered.

`explicit_spatial` is the GeoPackage default. The label path, embedded CRS, and
certified coverage explicitly associate annotations with the AOI, while the
labeling-backdrop `product_id` remains provenance. This permits, for example,
craters drawn over a NAC-derived raster to supervise a WAC-plus-static chip of
the same place. `same_product` additionally requires the metadata `product_id`
to equal the request product prefix or explicit source selector. Chip creation
must never infer `same_product` merely because both sides happen to have a
WAC/NAC-looking filename.

The current absolute `source` field remains useful for reopening a labeling
session, but chip creation does not require that path to exist and never uses
it as the sole identity check.

## Coverage and geometry validation

Before tiling, chip preflight must:

1. Open the required GeoPackage layers without modifying them.
2. Validate schema, CRS, identities, review status, and unique positive crater
   IDs.
3. Transform a densified target-grid footprint into the layer CRS.
4. Require the union of `complete` coverage polygons to contain the entire
   target footprint, within the same explicit numerical tolerance used by
   other label-coverage checks.
5. Select crater geometries with positive-area intersection with the target
   footprint. Boundary-only line or point contact is excluded.
6. Validate selected geometry and produce a compact, serializable preparation
   plan; do not read all polygons into a multiprocessing request and do not
   invoke tiling for a rejected label.

An AOI fully inside certified coverage with zero selected craters is valid and
produces an empty instance label. An AOI partly outside certified coverage is a
per-sample failure; the uncovered portion must never be filled as background.

Invalid geometry is a typed failure by default. Automatic geometry repair is
not silent: if a future option applies `make_valid`, diagnostics and the
manifest must record the repair and the source/derived geometry hashes.

## Deterministic target-grid rasterization

The worker materializes vector labels before its first tiling call:

1. Reopen the GeoPackage and verify the source hash/metadata captured by
   preflight.
2. Reproject selected crater geometry into the exact `TargetGrid` CRS.
3. Intersect it with the exact target-grid footprint and discard empty or
   zero-area results with a structured reason.
4. Sort by source `crater_id` and assign provisional compact IDs `1..N`.
5. Rasterize onto the exact target affine transform, width, and height with
   background `0`, pixel-center inclusion (`all_touched=False`), and an integer
   dtype large enough for `N`.
6. Resolve overlaps with a fixed painter rule: increasing source IDs are
   burned in order, so the larger source ID owns an overlapping output pixel.
7. Compute each clipped geometry's COCO `(x, y, width, height)` box in target
   pixel coordinates, clip the box to the target extent, and preserve float
   precision.
8. Distinguish a fully overwritten instance from a subpixel instance by also
   rasterizing/checking that geometry's own pixel support. A feature with
   support that is entirely overwritten may use the established occlusion
   rule; a feature with no pixel-center support is dropped with a structured
   warning.
9. Remove dropped annotations, remap mask IDs and boxes together to final
   compact IDs `1..N`, and validate the complete archive.

The canonical result is `<sample-id>_label.npz` containing:

- `mask`: 2D integer instance IDs with target shape;
- `bboxes`: `(N, 4)` COCO boxes in target pixel coordinates; and
- `num_craters`: scalar `N`.

The GeoPackage itself is never copied into `train/labels`, `val/labels`,
`test/labels`, or the no-split `labels` directory. Dataset loaders therefore
continue to consume the existing `.npz` contract without Fiona/GDAL vector
dependencies.

Diagnostics and the dataset manifest record the GeoPackage path and hash,
layer names, schema version, coverage status/relation, source-to-target ID map,
dropped/occluded features, rasterization policy, and derived archive hash.

## Failure isolation and publication

All GeoPackage structural, identity, CRS, coverage, and source-change failures
remain per-sample failures. They occur before imagery tiling whenever possible,
publish neither chip nor label, and do not stop later samples.

The derived `.npz` enters the same pair-atomic publication protocol as array
labels. If label materialization, chip creation, or either publication step
fails, neither final artifact remains visible. Source GeoPackages are always
read-only and are never removed by intermediate cleanup.

## Notebook handoff

The crater-labeling notebook must display:

- the raster-specific GeoPackage path;
- current crater count and `in_progress`/`complete` state;
- the meaning of certified coverage and the current coverage extent;
- a deliberate **Mark ready for chip creation** action; and
- a concise next-step cell showing the values to copy into the chip notebook:
  `LABEL_PATH`, `LABEL_KIND="instance_vector"`, `SAMPLE_ID`, AOI CRS/bounds,
  and output width/height.

The AOI-first chip notebook accepts that `.gpkg` directly. Its default
visualization shows the generated chip, the rasterized instance label, and an
overlay; it also reports source crater IDs dropped, remapped, or fully
occluded. It must not require the original labeling raster as a reference chip.

## Validation matrix

| Case | Expected result |
|---|---|
| Complete GeoPackage; AOI contains craters | Target-sized valid `.npz` |
| Complete GeoPackage; AOI contains no craters | Valid empty `.npz` |
| One GeoPackage reused by two covered AOIs | Independent deterministic labels with shared provenance |
| AOI partly outside certified coverage | Pre-tiling per-sample failure |
| Legacy `craters`-only GeoPackage | Loadable for migration; rejected by chip creation |
| `review_status=in_progress` | Pre-tiling per-sample failure |
| Explicit `same_product` mismatch | Identity failure before tiling |
| NAC-authored labels + WAC/static chip under `explicit_spatial` | Accepted when certified coverage contains the AOI |
| Rotated source raster footprint | Coverage uses transformed corners, not bounds envelope |
| Overlapping craters | Higher source ID owns overlap; occluded ID handled explicitly |
| Subpixel crater | Dropped with warning, not misclassified as occluded |
| Invalid or duplicate crater ID | Structural failure before tiling |
| Source changes after preflight | Worker source-hash failure; no publication |
| Serial versus multiprocessing | Same mask, boxes, IDs, diagnostics, and hashes |

## Implementation boundary

Expected labeling-side changes belong in `lfm/labeling/craters.py`, its focused
tests, and `notebooks/crater_labeling.ipynb`. Expected chip-side changes belong
in label types/preflight/materialization, orchestration, publication provenance,
notebook helpers, and `notebooks/chip_example.ipynb`.

No GeoPackage-label behavior belongs in `model/tiling.py`,
`model/configured_tiler.py`, or the raster cube writer. Tiling continues to
acquire imagery only after label preparation succeeds.
