# Crater Labeling to Chip Creation Contract

Accepted user decisions, updated 2026-10-05. This is an implementation contract;
A0 is frozen, and A1 and A2 source planning are complete with user-reported
HPC validation. A3 semantic materialization is accepted as complete. A4
instance/GeoPackage conversion is implemented pending HPC validation;
orchestration integration remains A5.
Declarative acceptance fixtures are
stored in [aoi_chip_contract_fixtures.json](planning_docs/aoi_chip_contract_fixtures.json).
Tracked in
[the AOI plan](planning_docs/aoi_chip_creation_and_label_clipping_plan.md).

## Inputs and output grid

The notebook takes one geographic IAU:30100 AOI; batch creation iterates over
multiple AOIs. Each request explicitly supplies its label path and output
sample ID. Assume supplied GeoPackages represent finished full-raster labels.
No readiness state, certified-coverage layer, filename matching, product-ID
matching, or source-scene matching is required. Processed-area selection is
deferred. Technical format, geometry, CRS, and final grid validation remain.
Output sample IDs and imagery selectors identify outputs and imagery, not label
identity. Preserve descriptive source paths/IDs without requiring the original
labeling raster path to exist on the chip machine. No optional `same_product`
label-matching policy or mandatory `label_coverage` layer is introduced.

Use the original imagery source raster's CRS and native grid from before
tiling. WAC VIS supplies the WAC reference grid; UV follows it. Multiple dynamic
modalities including WAC emit a warning that WAC takes precedence and all
modalities match its grid. Static-only uses the 100 m WAC reference resolution
in the LTM zone containing the geographic AOI center, aligned to projected
easting/northing multiples of 100 m (anchor `(0, 0)`). The historical
`7_band_preprocessing.zip` notebooks establish the VIS-source-to-reference-chip
grid inheritance. The final grid is not determined by intermediate LTM zoom.

Clip labels to final raster edges. Round the source pixel window outward:
floor minimum column/row and ceil maximum column/row, then preserve the source
affine linear coefficients and translate its origin to the window start.
Negative pixel indices are allowed when the AOI extends beyond source coverage;
do not shrink the request to the available source. Partial imagery NoData warns.

Transform the geographic perimeter into reference pixel coordinates before
finding extrema. Densify curved edges; the result is a containing pixel window,
so projected-envelope corners may extend beyond the geographic rectangle.
The user accepts outward coverage; preserve both requested AOI and realized
footprint in provenance. Use the realized footprint for label clipping.

For static-only, normalize longitude to `[-180, 180)`, take the midpoint along
the AOI's eastward longitude span (including seam crossing), and average its
north/south latitudes. Select LTM zone `floor((lon + 180) / 8) + 1`; zone-edge
ties go east, and latitude zero selects the northern hemisphere. Load that
zone's repository CRS. Polar chip coverage remains outside this plan.

Numerical conventions: snap pixel coordinates within `1e-8` pixel of an integer
before floor/ceil to prevent floating-point extra rows/columns. Coverage and
grid comparisons use pixel-space tolerances, not absolute map-coordinate
magnitude. Adaptively densify transformed edges until midpoint chord deviation
is at most `1e-4` target pixel, starting with 21 samples per edge. Cap refinement
at 20 levels; nonconvergence or invalid transforms fail explicitly. Verify
containment and conservatively expand an edge window if refinement reveals
additional extent. Validate these transforms with GDAL in later runtime tests.

Select WAC VIS as reference when present; otherwise use the selected dynamic
modality. For multiple candidate rasters, equivalent CRS/pixel lattices share
a grid; conflicting grids require an explicit reference-source choice and
produce a diagnostic instead of arbitrary file-order selection. Multiple
non-WAC dynamic modalities likewise require an explicit grid reference.
Static-only has no dependency on intersecting WAC data.

## Conversion interface

This plan owns label conversion; no coworker handoff or real notebook run is
required to design it. Input formatting can be inferred from
`lfm/labeling/craters.py`, which exports `<raster-stem>_label_craters.gpkg`
with a native-CRS `craters` polygon layer
with `crater_id`, `method`, `source`, `band`, `seed_col`, `seed_row`, and
`area_native`. Treat annotation provenance as descriptive, not a matching gate.

Implemented standalone interface (A4; HPC validation pending):

```text
convert_crater_labels(path, layer="craters", target_grid=...)
    -> InstanceLabelConversion(mask, bboxes, num_craters, id_mapping, diagnostics)
```

The source is read-only. The target grid specifies CRS, affine transform,
width, and height. Conversion transforms source geometry into that grid and
returns a validated result. The worker adapter stages the NPZ and hands it to
pair publication; conversion itself does not acquire imagery or publish files.
`materialize_instance_label(request, plan, staging_root=...)` is the worker-side
adapter returning a verified `PreparedLabelArtifact`; automatic worker invocation
and publication wiring remain A5. Exact NPZ files retain original bytes. Derived
NPZ files contain canonical `mask`, `bboxes`, and `num_craters`, with the ID map
and diagnostics carried by the artifact, not embedded in training arrays.

## Accepted instance conversion rules

1. Require unambiguous positive source instance IDs (`crater_id` for the current
   exporter). Reproject polygons to the output CRS and intersect them with the
   raster footprint. Ignore outside/boundary-only features with no remaining
   polygon area. Preserve holes and multipart geometry belonging to one ID.
2. Include a pixel when its center is inside the clipped crater outline.
   Do not use an all-touched rule to enlarge small features. An exactly
   boundary-coincident center counts as covered; hole interiors do not.
   Do not introduce a geometric buffer to resolve these ties. A rasterizer
   adapter must satisfy the fixtures rather than assume its default tie rule.
3. Determine each feature's pixel support independently before compositing.
   If it has positive clipped area but no included pixel centers, omit it with
   a warning identifying the source instance. Do not mistake overlap-induced
   disappearance for a subpixel crater.
4. At overlaps, the highest original instance number owns the pixel. Process
   source IDs in ascending order with later IDs overwriting earlier ones.
5. Retain supported instances even if their pixels are subsequently fully
   overwritten. Their existence and boxes are defined by clipped outlines.
   Record fully occluded IDs and validate them using explicit overlap evidence.
6. Sort retained source IDs ascending and renumber them `1..N` for each chip.
   Apply the mapping consistently to mask values and box rows, and preserve
   the source-to-output map in diagnostics. Highest-ID precedence is preserved
   by this monotonic renumbering.
7. Compute each COCO `(x, y, width, height)` box from its original outline
   **after clipping**, transformed into target pixel coordinates. Do not shrink
   the box according to visible pixels or subtract overlapping crater shapes.
   Boxes stay within raster edges and retain coordinate precision.
8. Emit integer `mask` of shape `(height, width)`, `bboxes` of shape `(N, 4)`,
   and scalar `num_craters=N`. Box row `i-1` belongs to output instance `i`.
   Background is zero. No retained craters means an all-zero mask, `(0, 4)`
   boxes, and count zero. An empty background is valid training data.

Use an integer dtype that can hold all retained IDs. Diagnostics distinguish
outside features, clipped features, subpixel omissions, full occlusions, and
invalid inputs. Typed conversion failures affect only the current sample.

## Pipeline requirements

Prepare and validate labels before tiling; publish target-sized instance NPZ
and chip GeoTIFF atomically. Partial imagery NoData warns with per-band counts
and percentages and a spatial union summary, but usable chips can proceed.
No imagery fails the sample with a clear message and no published pair.
NoData values must not be confused with valid zero-background label pixels.

## Imagery NoData and absent imagery

Partial chip NoData is allowed. Preserve the configured NoData value, emit a
warning, and report counts and percentages in the notebook and per-sample
diagnostics/manifest. Measure on the final chip grid. Report per-band counts
and percentages, plus the union of spatial pixels invalid in any output band;
the union percentage uses `width * height` as its denominator and counts each
spatial pixel once. Include per-band detail so optional placeholder channels do
not obscure required imagery coverage. Use declared masks/NoData and nonfinite
values, never pixel magnitude, to determine validity.

If no imagery is available, fail the sample with a clear message such as
“No imagery was available for this AOI.” An empty tiling return is not a chip
success. Completely missing required imagery cannot be rescued by static-only
context or placeholder bands; partially valid imagery can proceed with a
warning. The A0 fixtures explicitly include the all-NoData-required-band case.
Failed samples publish neither artifact and later batch samples continue.

This NoData decision concerns imagery. It does not authorize converting unknown
label values into background or altering categorical label values.

## Fixtures and validation scope

A0 includes deterministic fixture definitions for arbitrary valid, nonconsecutive source
IDs compacted to `1..N` in ascending source-ID order. No particular numeric
source IDs are required or special-cased. Test multiple ID sets and shuffled
feature order to verify that the rule is general and independent of read order.
Additional fixtures cover
partial and complete overlap, clipped-outline boxes unaffected by overlap,
subpixel omission with warning, raster-edge clipping, empty output, boundary
ties, differing CRS, invalid geometry/IDs, and serial/parallel equivalence.
A real notebook export is a later integration check. A4 implements conversion;
A5 integrates staging/publication. Dependency-free fixture checks establish
expected arithmetic and label content, not production GDAL correctness.
