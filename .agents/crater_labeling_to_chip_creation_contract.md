# Crater Labeling to Chip Creation Contract

This document records the user's A0 decisions for integrating the crater
labeling notebook, its backend, and chip creation. These are planned
requirements, not implementation evidence. They supersede the earlier proposed
certification and label-identity policies.
Implementation is tracked in
[the AOI plan](planning_docs/aoi_chip_creation_and_label_clipping_plan.md).

## User workflow and geographic contract

The chip notebook takes one geographic AOI in repository IAU:30100. The full
workflow iterates over multiple geographic AOIs, each with a unique output
sample ID and an explicitly supplied label path. One GeoPackage can be reused
across AOIs. A reference chip is unnecessary.

Geographic AOI is the public input contract. Selected dynamic imagery supplies
the native output resolution (WAC or NAC); static-only requests use WAC
resolution without requiring WAC imagery. Derive dimensions from that policy
and the AOI. All imagery and labels share the resulting grid. Native resolution
is distinct from intermediate tiling zoom/spacing.

Multiple dynamic modalities emit a warning that WAC takes precedence; align
all modalities and labels to the WAC grid. The old chip code inherited its CRS
and affine from a reference chip rather than imposing a universal output CRS.
For AOI requests, the accepted rule is to inherit the original source raster's
CRS and pixel lattice from before tiling. WAC VIS supplies that grid when WAC
is selected; UV and other modalities are aligned to it. The supplied
`7_band_preprocessing.zip` confirms that the original seven-band reference
chips inherited the VIS source CRS and 100 m grid, with UV resampled onto it.
Read source metadata rather than choosing the CRS from intermediate cubes.
Static-only uses the 100 m WAC fallback but still needs a standalone CRS/lattice
selection rule that does not require WAC imagery.

Clip crater geometry to the final raster edges. This plan now owns instance
encoding after clipping. Integer pixel-window and transformed-AOI
footprint rules still need definition; clipping labels does not itself decide
which partially intersecting edge pixels the raster includes. Source label CRS
may differ from the geographic query CRS and must be transformed correctly.

## Full-raster labels and scientist-supplied association

- Assume annotations cover the full labeling raster. Selecting a smaller
  processed/reviewed region is deferred.
- Supplying a GeoPackage means the scientist considers it finished. There is
  no readiness button, `in_progress`/`complete` field, or certification gate.
- Do not require a `label_coverage` layer or migrate existing files solely to
  add certification metadata.
- The scientist explicitly chooses the labels. Chip creation must not reject
  them because their filenames, sample IDs, product IDs, or provenance source
  differ from the acquired imagery. Neither `same_product` nor an equivalent
  optional label-matching policy is part of this contract.
- Output sample IDs and imagery source selectors still identify outputs and
  select imagery; they do not establish label identity.
- Preserve useful source paths/IDs as provenance when available, without
  requiring the original labeling raster path to exist on the chip machine.

Geometry, CRS, file readability, supported format, and final mask/grid checks
remain technical requirements. Scientist-selected association does not remove
the need for georeferencing to clip an array or transform vector geometry.
Crater feature bounds must not be used as the full source raster footprint.
If a source footprint is needed for geometric containment, A0 must agree its
metadata representation within the conversion interface; it is not a review-status
requirement or a filename-matching test.

## Label-conversion interface and ownership

The current labeling backend exports `<raster-stem>_label_craters.gpkg` with a
native-CRS `craters` polygon layer. Current fields are `crater_id`, `method`,
`source`, `band`, `seed_col`, `seed_row`, and `area_native`. These describe the
existing producer, not a newly imposed schema for every supplied GeoPackage.

This plan owns the conversion interface and its semantics, superseding the
earlier delegation to the labeling coworker. Infer input formatting from the
existing export code and build synthetic GeoPackages locally. Running the
labeling notebook is useful for later acceptance but is not a prerequisite.

Proposed interface (names to finalize in A0):

```text
convert_crater_labels(path, layer="craters", target_grid=..., options=...)
    -> InstanceLabelConversion
```

- Inputs: a read-only GeoPackage path, selected layer, complete target grid
  (CRS, affine, dimensions), and explicit conversion options. Read source CRS
  from the layer; no imagery product match, readiness field, or reference chip
  is required.
- Result: integer target-sized mask, COCO boxes, instance count, source-to-output
  ID mapping, and structured diagnostics for clipped, excluded, dropped, or
  occluded instances, with conversion policy and source provenance.
- Errors: typed file/schema/CRS/geometry/conversion errors that the batch adapter
  converts into per-sample failures. Distinguish an invalid input from a valid
  AOI with no intersecting instances.
- Persistence: the converter does not acquire imagery or publish a dataset.
  A worker adapter writes the result to a staged NPZ, validates it, and returns
  the existing planned `PreparedLabelArtifact`. Arrays stay in that worker;
  parent preflight plans contain only serializable metadata.

A0 must freeze feature inclusion, rasterization pixel rules, overlap order,
source-ID handling, compact output IDs, box calculation, empty results,
subpixel instances, and occlusion using deterministic fixtures. Earlier
suggestions such as `all_touched=False` and higher-ID painter priority remain
proposals until that step is complete.

The requested integration outcome remains a target-sized instance `.npz`
compatible with the existing training loaders: `mask`, `(N, 4)` `bboxes`, and
scalar `num_craters`. The interface must establish how conversion
produces those values, including an AOI with no crater instances. Any changes
to that output contract must be coordinated before vector integration.

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
warning. Define the all-NoData-required-band case explicitly in A0 fixtures.
Failed samples publish neither artifact and later batch samples continue.

This NoData decision concerns imagery. It does not authorize converting unknown
label values into background or altering categorical label values.

## Processing, publication, and remaining A0 work

Prepare labels before tiling and validate the final label against the exact
chip grid. Keep source files read-only and publish the chip/label pair with
the existing atomic publication behavior. Record source paths/hashes, output
grid, the agreed conversion method, and NoData diagnostics for reproducibility.
The notebook plots the saved chip, label, and overlay.

A0 still needs static-only grid, integer pixel-window and tolerance rules,
our vector-conversion and source-footprint contract, and synthetic
fixtures. Include explicit differently named labels, GeoPackages with no
review metadata, reuse across AOIs, partial NoData warnings, no imagery,
unusable required imagery, invalid georeferencing, and locally defined
instance edge cases. Later tests cover multiprocessing, source changes,
training-loader compatibility, and publication rollback.
