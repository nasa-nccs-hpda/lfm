# AOI-First Chip Creation and Label Clipping Plan

This document is the sequential implementation plan for changing the modern
chip-creation workflow so the public notebook creates chips from an explicit
user AOI instead of requiring a reference chip. It also adds safe support for a
label covering a larger source raster: the label is clipped or nearest-neighbor
warped to the authoritative chip grid before tiling starts, and only the
target-sized label is published with the chip.

This is a planning document. It does not change runtime behavior. The current
production contracts remain documented in:

- [`tiling_to_chip_creation_handoff.md`](../tiling_to_chip_creation_handoff.md)
- [`docs/chip_creation_modernization_plan.md`](../../docs/chip_creation_modernization_plan.md)
- [`docs/tiling_modernization_plan.md`](../../docs/tiling_modernization_plan.md)
- [`TMS/README.md`](../../TMS/README.md)

The accepted crater GeoPackage extension is specified in
[`crater_labeling_to_chip_creation_contract.md`](../crater_labeling_to_chip_creation_contract.md).
Its user-approved geographic AOI, full-raster label assumption, explicit label
association, and imagery NoData policies govern A0. This plan now owns the
label-conversion interface and semantics; no external handoff is required.

## Accepted A0 decisions

1. Public AOIs use repository IAU:30100. The notebook demonstrates one AOI;
   the full workflow iterates over multiple AOIs.
2. Assume labels cover the full labeling raster. A smaller processed-area
   selection is deferred.
3. Partial chip imagery NoData emits a warning with pixel counts and
   percentages and permits publication when required imagery is usable.
4. Supplying a GeoPackage implies that the scientist considers it finished;
   no readiness state or certification workflow is required.
5. Explicitly supplied labels are trusted associations. Do not match their
   filenames, sample IDs, products, or provenance against imagery.
6. Define crater clipping/rasterization, overlap, IDs, boxes, subpixel handling,
   and occlusion semantics within this plan. This supersedes the earlier
   delegation to the labeling coworker.
7. No available imagery fails the sample with an actionable message and no
   published pair; later samples continue.
8. Derive output resolution from the native resolution of the selected dynamic
   imagery (WAC or NAC). When only static modalities are selected, use WAC
   resolution. Static bands are resampled to the chosen output grid.
9. Multiple selected dynamic modalities including WAC emit a warning that WAC
   takes precedence. Without WAC, require an explicit grid reference.
10. Clip crater annotations to the final raster edges. Do not enlarge the
    raster to accommodate a crater crossing its boundary. Detailed instance
    encoding is defined by this plan's label-conversion interface.
11. Output CRS comes from the selected original source raster before tiling.
    For WAC, use the VIS source CRS/grid; UV and other modalities follow it.
    Read this CRS from source metadata, not from the intermediate LTM cubes.

These decisions supersede earlier proposed review and identity gates. A0 is
complete; the supplement and linked contract freeze the grid, numerical,
conversion, and fixture details. A1 and A2 are complete, with successful HPC
validation recorded in the phase evidence below.

## Status convention

### Accepted decisions supplement (2026-10-05)

The user-approved
[crater conversion contract](../crater_labeling_to_chip_creation_contract.md)
defines the frozen A0 conversion and grid rules. A0 is complete as a design
and fixture phase; backend implementation starts at A1.

- Public inputs are geographic IAU:30100 AOIs: one in the example, multiple in
  batch processing. Output CRS/grid comes from original source imagery before
  tiling, with WAC VIS taking precedence (with a warning) over other dynamic
  modalities. Static-only uses WAC's 100 m reference resolution in the LTM zone
  containing the AOI center, anchored at multiples of 100 m in projected units.
- Round outward on the chosen pixel lattice, preserving source spacing and
  including the whole requested AOI. The realized footprint may be larger;
  record both extents and clip labels to the realized raster edges.
- Trust explicitly supplied label associations without filename/product checks.
  GeoPackages are assumed finished and full-raster; no review-status gate or
  mandatory coverage-certification layer is required.
- Partial imagery NoData warns with counts/percentages and permits usable
  chips. No available imagery fails only the sample and publishes neither pair
  member.
- We own the conversion interface. Clip outlines at raster edges, include
  pixels by their centers, let highest source instance ID win overlaps, and
  compact retained IDs in source order. Boxes follow clipped original outlines
  regardless of overlap. Omit unsupported subpixel craters with a warning;
  fully overwritten supported craters retain their boxes. Empty labels are zero.
- A0 freezes and tests these semantics; A4 implements them through a reusable
  GeoPackage/layer/target-grid to instance-result interface.

### Phase status definitions

- `[Not Started]`: no implementation work has begun.
- `[Implemented]`: code and local checks are in place; the phase may still
  require its recorded HPC validation gate.
- `[In Progress]`: active work; only one sub-step should have this status.
- `[Complete]`: implemented, tested, documented, and accepted for its stated
  scope.
- `[Deferred]`: intentionally removed from the current sequence, with its
  reason and restart condition recorded.

Phases and sub-steps are sequential. A phase is complete only when all required
sub-steps and validation gates in that phase are complete.

## Starting behavior and remaining gap

The backend already supports `chip_request_from_aoi()`, but the active path in
`notebooks/chip_example.ipynb` still:

- requires `REFERENCE_CHIP` during path validation;
- extracts the exact target grid from that TIFF;
- derives the request sample ID and product selector from its filename; and
- originally expected a label whose identity, dimensions, and optional grid
  exactly matched the final chip.

Before A1/A2, the label pipeline accepted semantic `.npy` and instance `.npz` files.
Final-label validation in `model/chip_labels.py` rejects any mask whose shape differs from the target
grid, and optional label-grid metadata must match the target exactly.
`model/chip_publication.py` then byte-copies that source label into the dataset.
These are deliberate safeguards, but they prevent reuse of a georeferenced
label covering a full source scene or other larger parent AOI. A2 now plans
larger array, GeoTIFF, and GeoPackage labels separately, without identity checks
for explicit paths. Materialization and notebook integration remain pending.

The phrase **full-TIFF label** in this plan means a label mask covering the full
spatial extent of a source TIFF. The label may be:

1. a semantic `.npy` mask or instance `.npz` archive with an explicit or JSON
   sidecar source grid; or
2. a single-band, integer, georeferenced `.tif`/`.tiff` semantic mask; or
3. a scientist-supplied crater `.gpkg` assumed to label the full source raster,
   rasterized directly onto the target grid under the linked contract.

A standalone instance GeoTIFF is not sufficient because it does not contain the
required COCO boxes and annotation count. Raster instance input remains `.npz`;
vector GeoPackage input derives its mask, boxes, and count together.

## Required invariants

### Tiling boundary

- Do not change tiling's AOI, routing, tile-size, ordering, resampling, source
  selection, index ownership, or structured-result contracts.
- Continue calling strict `create_tiles_for_aoi()` from chip acquisition with
  an explicit product selector for every `product_id` source.
- Continue splitting antimeridian-crossing geographic queries in chip
  acquisition and deduplicating `TileCubeRecord` objects by structured fields.
- Treat an empty record collection as a valid possible tiling return, not as
  proof that a chip can be assembled. The high-level tiling workflow warns and
  returns no records when no dynamic product intersects a geographically valid
  AOI, and it skips contextual static for that mixed query. Chip acquisition
  must explicitly fail only that sample when a required output modality lacks
  coverage, while preserving existing optional-modality placeholder rules.
- Do not parse or depend on `ProductAOIWarning` text. Base chip decisions and
  diagnostics on returned structured records, requested modalities, and their
  required/optional policy. Retain strict routing errors for geographically
  invalid/unroutable AOIs and strict low-level required-source exceptions.
- Keep all label work downstream of the target request and upstream of tiling.
  Never add label cropping to `model/tiling.py`.
- Preserve the current numbered-LTM chip restriction. Upstream polar tiling
  support does not enable polar chip creation.

### Target grid and AOI

- The public geographic IAU:30100 AOI selects the study area. Single examples
  and batches share this contract. Each request materializes one
  complete `TargetGrid`: lunar-compatible CRS, finite rectangular bounds,
  invertible affine transform, positive width, and positive height.
- The final chip and final label must have exactly that grid and shape. LTM tile
  boundaries remain intermediate acquisition geometry and must not replace the
  target extent.
- Derive dimensions from the original dynamic source grid, using WAC VIS when
  WAC is selected. No reference chip is required. For static-only, load the
  repository CRS of the AOI-center LTM zone and use a north-up 100 m lattice
  anchored at projected `(0, 0)`. The linked contract defines zone-edge ties,
  antimeridian midpoint, outward rounding, and numeric tolerances.
- Outward rounding and transformed-perimeter envelopes may enlarge the output
  footprint. Preserve native pixel spacing, record requested/realized extents,
  and clip labels at the final raster edges. Do not silently crop the grid to
  the available imagery; partial imagery NoData is permitted with diagnostics.
- Initial scope remains rectangular AOIs. Do not silently replace an arbitrary
  polygon with its envelope. Polygon masking is a separate feature.

### Output-resolution policy

| Selected imagery | Output resolution source |
|---|---|
| WAC, with or without static | Native resolution of the configured WAC imagery |
| NAC, with or without static | Native resolution of the configured NAC imagery |
| WAC and other dynamic modalities, with or without static | Warn; WAC grid takes precedence for every modality |
| Static only | WAC resolution, without requiring WAC imagery to intersect the AOI |

Determine native resolution from configured source metadata or a documented
modality definition. Do not substitute intermediate LTM tile spacing or zoom
for native resolution, and do not require a reference chip. Record the source
of the chosen resolution and the realized target transform in diagnostics.
All bands and labels use the same final grid; categorical labels retain their
appropriate categorical conversion policy.

### Accepted source CRS and historical evidence

The legacy implementation in `model/chip_making/chip_utils.py`, in the section
"Load reference training chip for target grid", reads CRS, affine transform,
height, width, and bounds from `train_ds`. `merge_and_reproject_datasets()`
reprojects both WAC and static to that grid. Thus it has no fixed universal
output CRS and does not select output CRS from the geographic query or LTM
acquisition tiles. The modern reference-request and reprojection modules
likewise preserve the supplied `TargetGrid`.

The supplied `7_band_preprocessing.zip`, specifically
`7_band_preprocessing/7_band_preprocessing_3_26.ipynb`, establishes the earlier
reference-chip provenance. Its `clip_and_stack_multimodal()` reads `vis_src.crs`
and `vis_src.transform`, crops a VIS pixel window from the source, and reprojects
UV onto that window. VIS is described as 100 m and UV as 400 m. The resulting
seven-band reference chip inherits VIS CRS and pixel alignment. At commit
`c07f01f`, the later chip workflow then inherits that reference grid.

The accepted reference-free rule is therefore to use the selected original
source raster before tiling as the grid reference. Its native CRS, resolution,
orientation, and pixel lattice determine the output grid; the IAU:30100 AOI
selects its spatial window. For WAC, VIS supplies the grid and UV follows it.
Do not force the final chip into the intermediate tile CRS or assume that all
source rasters use LTM 22S merely because the inspected sample does.
WAC wins when multiple dynamic modalities are selected, with a visible warning
and a recorded grid-reference choice. Reproject NAC/static and labels to that
same target. Source metadata inspection must precede label preparation.

Crater geometry crossing the final raster boundary is clipped at that boundary.
A0 froze outward integer windows and densified geographic-footprint handling:
preserve the native lattice and record requested bounds and realized footprint,
which may include area outside the geographic rectangle. The linked contract
specifies the numerical tolerances and boundary rules.

The formerly open grid decisions are resolved as follows:

- Static-only uses 100 m pixels in the AOI-center LTM zone, anchored at projected
  `(0, 0)`, without opening or acquiring WAC imagery.
- Equivalent source lattices share a grid. Conflicting candidates within the
  winning modality require an explicit reference rather than file-order choice.
- Multiple non-WAC dynamic modalities require an explicit grid reference.
- Outward rounding preserves native spacing; label geometry clips to the
  realized raster edges rather than changing the selected output grid.

Changing final pixel spacing cannot recover detail already lost in acquisition
at a coarser LTM zoom. Check the acquisition/output resolution relationship
when implementing this policy, without treating the two settings as identical.

### Imagery NoData

Partial NoData on the final chip grid warns and continues. Preserve the output
NoData encoding and report per-band invalid-pixel counts and percentages plus
the union of spatial pixels invalid in any output band. The union denominator
is width times height, with each location counted once. Persist these summaries
in diagnostics/manifests and display them in the notebook. Use validity masks,
declared NoData, and nonfinite values rather than pixel magnitude.

No available imagery is a per-sample failure with a clear message. Required
imagery cannot be replaced by static-only context or placeholders. Retain the
required-band nonempty-valid-data check and test its all-NoData case; partial
valid coverage must not fail merely because other pixels are NoData. This
policy does not turn unknown categorical label values into background.

### Label safety

- A label larger than the target is accepted only when its source grid is
  independently known. Array shape alone cannot locate the chip AOI within a
  full-scene mask.
- Raster/array labels must cover the realized target grid. Missing coverage
  cannot be padded as background. For vector GeoPackages, the user asserts
  full-raster annotation; no certified footprint or review status is required.
  Do not infer a source raster's extent from crater polygon bounds.
- Do not validate explicitly supplied label filenames, sample IDs, or product IDs
  against imagery. Preserve technical file, CRS, geometry, and final-grid checks.
- Label reprojection is categorical and uses nearest-neighbor only. Tiling
  remains bilinear; the two policies must not be conflated.
- Label planning and source validation occur before tiling. A label failure is
  isolated to its sample, invokes no tiling call, and publishes neither chip
  nor label.
- Exact-grid labels retain the existing byte-identical publication fast path.
  Derived labels are target-sized artifacts with recorded provenance and are
  validated immediately before atomic pair publication.

## Proposed target contract

### AOI request

The notebook's active request supplies:

- `SAMPLE_ID`: unique output identity. For WAC/NAC it retains the product ID as
  the first underscore-delimited component so selector derivation remains
  deterministic.
- `AOI_BOUNDS`: geographic `(west, south, east, north)` in repository IAU:30100.
  Convert explicitly to tiling's named corner arguments.
- Output CRS, affine, width, and height are derived using the selected source
  grid and outward rounding, or the static-only grid rule. An explicit
  reference-source choice resolves competing source grids within one modality.
- AOI CRS is loaded from repository IAU:30100 rather than user-selected.
- `LABEL_PATH`: explicit source-label association.
- `LABEL_SOURCE_GRID`: optional structured source grid when the label does not
  embed one. This may be read from a sidecar or derived explicitly from the
  corresponding full-scene TIFF without making that TIFF the chip target.
- `SPLIT_GROUP_KEY`: normally the WAC/NAC product ID or another scientifically
  meaningful leakage group.

Extend request construction to accept the geographic AOI and resolved grid
reference before label preflight. Preserve existing explicit target-grid and
reference-TIFF entry points for compatibility. The notebook no longer requires
`REFERENCE_DIR` or `REFERENCE_CHIP` for the active example.

### Explicit label input and provenance

Add a typed label-input contract rather than overloading a bare path. The
recommended record contains:

- source path;
- semantic, raster-instance, or vector-instance kind;
- optional source sample/scene provenance (not an association gate);
- source `TargetGrid` when not embedded in the file;
- requested relation: `exact` or `clip_to_target`; and
- optional sidecar/provenance path.

Keep `ChipRequest.label_path` and `label_grid` as backward-compatible inputs,
normalizing them into this record. New AOI callers should use the typed form.

The scientist explicitly associates each request with a label path. Do not
require label filename, sample-ID, or product equality with the imagery.
One parent label may feed multiple AOIs. Source identity is optional provenance;
technical file/georeferencing and target-grid validation still apply.

New AOI/batch workflows supply associations explicitly, including labels reused
with another imagery modality. Legacy directory lookup remains a compatibility
convenience using the full offset-qualified sample ID; it is not a validation
gate for explicit paths. Imagery selectors and unique output sample IDs remain
independent requirements.

For `.npy` and `.npz`, accept source-grid metadata from either:

1. the typed request association; or
2. one unambiguous JSON sidecar containing `source_grid` and optional provenance.

Continue reading legacy sidecar `target_grid` as an exact-grid association.
Do not reinterpret it as a larger source grid silently.

For a semantic label GeoTIFF, read its CRS, affine transform, bounds, width,
height, band count, dtype, and NoData directly. Require exactly one integer
band. The canonical published training label remains `.npy`, so existing
dataset loaders do not acquire a GeoTIFF dependency.

### Spatial-relation plan

Raster-label preflight classifies one of three relations:

1. `exact`: source and target grids match under the existing strict tolerance;
2. `aligned_window`: same CRS and pixel lattice, with target boundaries mapping
   to an integer source window; or
3. `nearest_warp`: target is fully covered but differs in CRS, resolution,
   rotation, or pixel alignment.

Use the aligned-window path whenever possible because it preserves categorical
values exactly. Otherwise create the target label with nearest-neighbor
reprojection. Coverage testing must use transformed/densified target
footprints, not bounds alone. Reject a target that is partially outside the
source label, crosses an unrepresented gap, has an incompatible lunar CRS, or
cannot be mapped invertibly.

GeoPackages use a separate `vector_rasterize` preparation method under the
linked contract. File readability, lunar CRS, positive unique instance IDs,
and valid polygon geometry are required; identity matching and review status
are not. Empty annotation layers are valid.

### Semantic-label output

- Read only the needed source window when the storage format permits it.
- Produce one 2D integer mask with shape `(AOI_HEIGHT, AOI_WIDTH)`.
- Preserve class IDs. Map declared source NoData only according to an explicit
  label NoData policy; do not infer invalid pixels from magnitude.
- Initially reject source label NoData inside a derived target unless an
  existing explicit label encoding accounts for it; do not invent background
  or an ignore class. This differs from permitted partial imagery NoData.
- Validate the derived mask against the target grid and publish it as
  `<sample-id>_label.npy`.

### Instance-label output

For vector GeoPackages, use the accepted conversion rules in the linked
contract, including highest-source-ID overlap priority, pixel-center inclusion,
clipped-outline boxes, subpixel warnings, and fully occluded annotations.
The following algorithm describes raster archive conversion.

Instance clipping must update the mask, boxes, and count as one transaction:

1. Window or nearest-warp the integer instance mask to the target grid.
2. Transform each source COCO `(x, y, width, height)` box through source pixel
   coordinates into the target pixel grid and clip it to the target extent.
3. Drop annotations whose clipped boxes have no positive target area.
4. Keep remaining annotations in original-ID order and remap them to compact
   IDs `1..N`; apply the same mapping to mask pixels.
5. Preserve an annotation with no visible own-mask pixels only when the current
   overlap/occlusion rule finds another positive instance in its clipped box.
   Otherwise drop it with a structured warning before final validation rather
   than publishing a malformed archive.
6. Write `mask`, clipped `bboxes`, and scalar `num_craters=N` to
   `<sample-id>_label.npz` and run the existing target-sized archive validator.

Record old-to-new ID mappings, annotations removed outside the AOI, annotations
removed after resampling, and accepted occlusions in diagnostics and the
dataset manifest. An AOI containing no instances is a valid empty instance
archive with mask background only, `bboxes.shape == (0, 4)`, and
`num_craters == 0`.

### Pipeline order and publication

The intended per-sample order is:

```text
request/grid validation
  -> deterministic split assignment
  -> source-label resolution and non-writing clip plan
  -> target-label materialization and validation
  -> tiling acquisition
  -> chip reprojection/assembly
  -> atomic chip + target-label publication
```

The parent-process preflight remains non-writing. It validates source
structure, georeferencing, and applicable raster coverage, and produces a compact
`LabelPreparationPlan`. Each chip worker materializes and validates its label
before its first tiling call. This avoids sending large arrays between
processes and prevents nested worker pools.

Add a worker progress stage such as `label/clip` between `preflight` and
`tiling`. Derived labels live under the sample's bounded intermediate root and
follow the existing `always`, `on_failure`, and `never` retention policy.

Publication must accept a validated `PreparedLabelArtifact`:

- exact labels may point at the original source and retain checksum/byte-copy
  behavior;
- clipped labels point at the validated target-sized staging artifact;
- source and derived hashes are checked for changes before publication;
- chip and label remain rollback-safe and appear together or not at all; and
- the manifest records source label, source grid, relation, clip/warp method,
  derived label, target grid, ID mapping summary, hashes, and diagnostics.

## Phase A0 — Freeze AOI and label-clipping contracts `[Complete]`

- `[Complete]` **A0.1** Confirm the initial input formats: semantic `.npy`,
  instance `.npz`, crater GeoPackage, and single-band integer semantic GeoTIFF. Explicitly defer
  instance GeoTIFFs without box/count metadata.
- `[Complete]` **A0.2** Freeze the notebook's AOI inputs, output-grid
  derivation, sample/product identity rules, and rectangular-only scope.
- `[Complete]` **A0.3** Trust explicit label association without identity
  matching, including reuse of one parent label by multiple AOIs. Retain
  legacy directory lookup as a compatibility convenience, not an identity gate
  for explicitly supplied paths.
- `[Complete]` **A0.4** Freeze exact, aligned-window, nearest-warp, and vector
  relations; full-coverage requirements; categorical NoData behavior; and
  numerical tolerances.
- `[Complete]` **A0.5** Formalize and fixture-test the accepted vector
  conversion interface and semantics in the linked contract. Highest source ID
  wins overlap; IDs compact in ascending order; boxes follow clipped outlines;
  unsupported subpixel features warn and drop; empty masks are zero; pixel
  centers determine inclusion. No external labeling handoff is required.
- `[Complete]` **A0.6** Add small repository fixtures or fixture builders for
  exact, larger aligned, differently projected, partial-coverage, semantic,
  overlapping-instance, and the linked GeoPackage acceptance cases before
  implementation.
- `[Complete]` **A0.7** Incorporate the accepted GeoPackage full-raster
  assumption, no readiness state, no label matching, partial imagery NoData
  warnings, and no-imagery failures. Define geometric metadata and conversion
  within this plan and carry integration/test work into phases A1–A8.

Exit gate: the accepted contract and fixtures make every expected output,
warning, and failure deterministic without relying on a real Explore dataset.

A0 completion evidence (2026-10-05): the linked conversion contract specifies
grid choice, outward windows, zero-anchor static grids, boundary ties, numeric
tolerances, source ambiguity handling, and label semantics.
`aoi_chip_contract_fixtures.json` records explicit expected windows, masks,
boxes, IDs, semantic windows/nearest samples, NoData summaries, and failure
outcomes. Six dependency-free tests in `model/tests/test_chip_a0_fixtures.py`
passed, including every permutation of feature order for the vector fixtures.
Command: `python3 -m unittest discover -s model/tests -p test_chip_a0_fixtures.py -v`.
These are analytic fixture checks, not runtime converter tests. GDAL/Fiona/NumPy
are unavailable locally; actual CRS transformations, GeoPackage materialization,
and production serial/parallel tests remain A1–A7 work. A1 implementation and
subsequent HPC validation are recorded below.

## Phase A1 — Extend request and label types `[Complete]`

- `[Complete]` **A1.1** Add immutable `LabelInput`,
  `LabelPreparationPlan`, and `PreparedLabelArtifact` records in the chip type
  layer with path, kind, grid, relation, hash, and diagnostic
  validation.
- `[Complete]` **A1.2** Extend `ChipRequest` compatibly so legacy
  `label_path`/`label_grid` requests normalize to exact mode while AOI callers
  can explicitly request clipping.
- `[Complete]` **A1.3** Keep `chip_request_from_aoi()` as the canonical
  constructor and support geographic inputs with source-grid metadata or the
  static-only rule, deriving dimensions by outward rounding. Preserve existing
  explicit-grid callers and report ambiguous grid-reference choices.
- `[Complete]` **A1.4** Extend result, diagnostic, progress-stage, and
  manifest schemas with label-preparation provenance without placing arrays in
  serializable request objects.
- `[Complete]` **A1.5** Add dictionary/config round-trip tests and
  backward-compatibility tests for existing exact-label callers.

Exit gate: old requests behave identically; new requests can describe a
full-scene label and exact target grid without a reference TIFF.

### A1 implementation and validation

`model/chip_types.py` now provides frozen, JSON/pickle-compatible metadata
records. `LabelInput` uses `semantic`, `raster_instance`, or `vector_instance`
kinds (known suffixes resolve `auto`), `exact`/`clip_to_target` relations, optional
source grid, layer, source identity, and sidecar path. Legacy `label_path` and
`label_grid` normalize to exact input; conflicting legacy/typed fields fail.
`LabelPreparationPlan` records the method, target grid, source SHA-256, optional
aligned window, and diagnostics. `PreparedLabelArtifact` records the plan,
output SHA-256, compact instance-ID mapping, and diagnostics. Exact NPY/NPZ
artifacts require matching source/output hashes; an exact-grid semantic TIFF
still needs conversion to NPY and may therefore have different bytes.

`chip_request_from_aoi()` retains the old explicit-grid arguments. Geographic
mode accepts `geographic_aoi=GeographicAOI(...)` and exactly one of:

- `source_grid=TargetGrid(...)`: an explicit original, pre-tiling raster grid;
- `source_grids={"wac": (vis_grid, ...), ...}`: metadata candidates, with WAC
  precedence and a warning for multiple dynamic modalities. WAC candidates
  must be VIS source grids, not UV grids. Different pixel lattices within the
  selected modality require an explicit `source_grid`; or
- `static_only=True`: the approved 100 m, zero-anchored, midpoint-LTM lattice.

The constructor transforms/densifies the geographic boundary and rounds
outward while preserving native CRS/affine. `requested_aoi` stores the original
geographic input; existing `geographic_aoi` continues to mean the realized
output-grid envelope used for acquisition. Both extents and the target grid
are retained in diagnostics/manifests. Source discovery and automatic metadata
loading for the notebook are not part of A1.

Preflight/result records now have optional `label_plan`/`prepared_label` fields.
Diagnostics accept `label_preparation`; progress accepts `label/clip`.
Provenance fields are additive to the existing version-1 JSON documents; old
keys retain their meanings. A1 does not emit materialization events or write
derived labels. A1 initially rejected `clip_to_target` at preflight to prevent
legacy copying. A2 below replaces that gate with source planning and an
execution-time guard for plans needing materialization. The notebook is unchanged.

Local chip-suite validation: 144 tests discovered, 103 passed, 41 skipped for
missing runtime dependencies. `test_chip_a1_contracts` includes A0 production
rounding/zone fixtures, JSON/pickle round trips, legacy dictionaries, invalid
metadata and hashes, grid-choice ambiguity, adaptive-boundary failure, and
manifest/diagnostic provenance. Its real-GDAL tests cover native CRS offsets,
static-only grids, rotated affines, and antimeridian AOIs; HPC validation is
recorded below.
Full model discovery also encounters four legacy modules that import absent
GDAL unconditionally; those import errors are an environment limitation, not
a passing full-suite result.

Run from the directory containing the `lfm` checkout inside the HPC container:

```bash
python -m unittest discover -s lfm/model/tests -t . -p 'test_chip*.py' -v
```

HPC validation: the user reports all chip-suite tests pass in the container
after setting its working directory to the checkout's parent and using the
discovery command above. This closes A1's environment-dependent validation
gate. The exact HPC test count was not supplied.

## Phase A2 — Resolve source labels and plan clipping `[Complete]`

- `[Complete]` **A2.1** Refactor label validation into source-structure,
  source-grid/relation, and final-target validation rather than applying the
  target shape check while opening the source.
- `[Complete]` **A2.2** Extend resolution to supported GeoTIFF/GeoPackage labels and
  the new sidecar schema while retaining exact full-sample-ID lookup for legacy
  directory-based labels.
- `[Complete]` **A2.3** Accept explicit label paths without filename/product
  equality checks; preserve structural and geospatial validation.
- `[Complete]` **A2.4** Compute exact/aligned/warp relations, source windows,
  densified coverage, and output encoding without writing files.
- `[Complete]` **A2.5** Return typed per-sample failures for missing grid
  metadata, incompatible CRS, incomplete coverage, ambiguous sidecars,
  malformed source contents, and unsupported formats.
- `[Complete]` **A2.6** Prove preflight remains read-only and never invokes
  tiling for a rejected label.

Exit gate: preflight deterministically accepts or rejects every A0 fixture and
produces no dataset/intermediate output.

### A2 implementation and validation

`model/chip_label_planning.py` owns read-only `plan_label_preparation()` and
`classify_label_grid()`. `preflight_label()` now returns a `label_plan` for a
valid source. Existing NPY/NPZ target-sized validation remains separately
available through `validate_label()` for publication; larger array sources
are validated against their own source grid, not the chip dimensions.

- Explicit request paths and file-valued `label_source` associations bypass
  filename/product/sample matching. Directory lookup still requires one full
  offset-qualified sample-ID match. Sidecar `sample_id` is provenance only.
- Arrays in clipping mode require a source grid. Sidecars may use
  `{"source_grid": {...}}`; automatic discovery considers `<label.ext>.json`
  and `<label>.json` and rejects ambiguity. An explicit `sidecar_path` selects
  one deliberately. Conflicting embedded/typed/sidecar grids fail. Legacy
  `target_grid` or bare-grid sidecars remain exact-chip associations, not
  silent full-scene clipping metadata. Old exact arrays without a grid retain
  their `label_grid_unverified` warning for compatibility.
- Raster relations use lunar CRS compatibility, pixel-space alignment,
  round-trip checks, and adaptively densified footprint containment. GeoTIFFs
  must have one integer band and embedded georeferencing. Unknown/NoData cells
  intersecting the realized target footprint fail, even if downsampling would
  otherwise hide the gap. Cells outside the footprint do not invalidate it;
  mask checks read bounded blocks without allocating a full-scene mask.
- GeoPackages validate the selected layer (default `craters`), lunar CRS,
  integer positive unique `crater_id`, and valid polygon/multipolygon geometry.
  Empty layers, arbitrary nonconsecutive IDs, overlaps, and outside features
  are valid. No completion flag, coverage certification, or imagery identity
  is required. No clipping/rasterization occurs yet.
- Plans retain resolved grids, selected sidecars, preparation method/window,
  source SHA-256, and diagnostics. Computed `output_kind`/`output_suffix`
  properties describe semantic NPY versus instance NPZ output. Source hashes
  are checked before and after validation to detect concurrent changes.

Until A3–A5 are implemented, plans requiring materialization are blocked with
`label_preparation_not_available` before acquisition or overwrite cleanup;
direct acquisition/publication entry points are guarded too. Exact NPY/NPZ
plans remain executable. This guard does not turn valid source plans into
preflight failures. The notebook and converter implementations are unchanged.

Local verification: chip discovery ran 161 tests (112 passed, 49 dependency
skips); the dependency-aware model suite ran 361 (274 passed, 87 skips).
`test_chip_label_planning.py` adds 17 tests, including nine locally runnable
metadata/affine/control-flow checks and eight GDAL-backed checks pending HPC.
Source snapshots prove read-only behavior in the applicable tests. A0 vector
fixtures are used as valid source inputs; converted masks/boxes are A3/A4
checks, and imagery-availability outcomes remain outside A2.

Run the same chip-suite discovery command documented under A1, from the
checkout's parent directory inside the HPC container. The new module is also
individually runnable with:

```bash
python -m unittest -v lfm.model.tests.test_chip_label_planning
```

HPC follow-up: the user reported one error in
`GridRelationTestCase.test_different_crs_uses_nearest_warp`, during the
round-trip check on a curvature derivative probe. Review found that the
unconditional `row + 1`/`column + 1` probes could leave the target footprint,
including this fixture's one-row target. Derivative probes now stay inside
the grid and account for signed, potentially fractional steps. The `1e-4`
pixel round-trip tolerance is unchanged; failures now report coordinates and
error magnitude. Two dependency-free regression tests cover one-row/column
grids and continued rejection of genuinely bad round trips. The updated local
chip suite ran 163 tests (114 passed, 49 dependency skips). The original GDAL
test is unchanged. The user subsequently reports all tests pass on HPC after
this correction, closing A2's runtime validation gate. The exact HPC test count
was not supplied.

A2 and A3 are complete; the user accepted A3 before authorizing A4.

## Phase A3 — Materialize semantic labels `[Complete]`

- `[Complete]` **A3.1** Implement the exact no-copy plan and integer-window
  slicing for aligned `.npy`, `.npz` masks, and semantic GeoTIFFs.
- `[Complete]` **A3.2** Implement nearest-neighbor warp to the exact target
  transform, CRS, width, and height for non-aligned semantic labels.
- `[Complete]` **A3.3** Preserve integer class IDs and apply only declared
  NoData/background rules; reject values or conversions that cannot be
  represented safely.
- `[Complete]` **A3.4** Write derived semantic labels to deterministic
  per-sample staging paths, validate their shape/content/grid provenance, and
  compute hashes.
- `[Complete]` **A3.5** Add window-versus-warp equivalence, CRS, rotated
  grid, edge, empty-background, partial-coverage, dtype, and NoData tests.

Exit gate: every accepted semantic source produces one target-sized integer
`.npy` label before tiling, with exact-grid inputs still byte-preservable.

### A3 implementation and validation

`model/chip_label_materialization.py` exposes
`materialize_semantic_label(request, plan, staging_root=...)`. It returns a
`PreparedLabelArtifact` with source/target-grid provenance, hashes, and
diagnostics, without invoking imagery acquisition or dataset publication.

- Exact NPY plans reuse the original file without copying or creating staging
  directories. Exact GeoTIFFs still need conversion to canonical NPY.
- Aligned NPY and GeoTIFF inputs use integer source windows. The shared mask
  reader also supports NPZ mask windows for A4; the semantic entry point rejects
  instance sources rather than separating their masks from boxes/counts.
- Nearest-neighbor reprojection inverse-maps target pixel centers into source
  pixels. Equal projected CRSs use vectorized affine mapping; differing CRSs
  and geographic longitude branches use the validated OSR mapping. Only
  coordinates use floating point: class values retain their source integer
  dtype, including signed/unsigned 64-bit IDs beyond float64's exact range.
  Ties within `1e-8` source pixel of an integer boundary select its right/bottom
  pixel. Rotated target/source affines are preserved.
- Source hashes, grids, relation/window plans, and sidecar/embedded metadata
  are rechecked before writing. A2's full-coverage and declared label-NoData
  checks still apply; unknown areas are never filled with background. Zero
  background and negative integer class encodings remain valid when not
  declared NoData. Changed/unreadable sources fail with typed diagnostics.
- Derived paths are
  `<staging_root>/<sample_id>/labels/<sample_id>_label.npy`. Reads and output
  processing use bounded blocks; NPY uses read-only memory mapping and GeoTIFF
  reads source windows. NPZ mask access necessarily decompresses its mask.
  Output uses a disk-backed NPY, then is reopened to verify shape, dtype, and
  a digest of the generated pixel content. The complete file is also hashed.
- Temporary files are installed with atomic no-clobber links. Existing or
  concurrently installed artifacts are never overwritten. Sample symlink
  traversal and staging that would own source labels/sidecars are rejected.
  Failures remove only this call's files; cleanup failures have diagnostics,
  and empty directories may remain for A5's retention policy.

Grid provenance is retained in the immutable artifact plan, not a new output
sidecar. A5 will persist it through the existing diagnostic/manifest schemas
and integrate worker execution, cleanup, and atomic dataset publication.
The existing orchestration materialization guard remains in place; notebook
and publication behavior have not been changed. A4 instance conversion follows.

Local chip discovery: 188 tests ran, 120 passed and 68 were skipped for absent
dependencies. The new `test_chip_label_materialization` module contains 25
tests: six locally runnable metadata/safety checks, three NumPy checks, and
sixteen NumPy/GDAL integration checks. Runtime tests cover exact reuse,
windows, differing CRS, rotation, categorical ties, wide integers (including
GeoTIFF UInt64), empty background, multi-block processing, NoData and coverage
failures, stale source/sidecar/plan rejection, corruption detection, output
collisions, and shared-parent sample isolation. A3 was subsequently accepted
by the user as complete before starting A4; no additional HPC test count was
provided. Local skips alone are not evidence of raster correctness.

Run from the checkout's parent directory in the HPC container:

```bash
python -m unittest -v lfm.model.tests.test_chip_label_materialization
python -m unittest discover -s lfm/model/tests -t . -p 'test_chip*.py' -v
```

## Phase A4 — Materialize instance labels `[Complete]`

Implement the interface and semantics frozen in A0 for raster archives and
GeoPackage input. The converter remains usable independently of the notebook
and imagery acquisition.

- `[Complete]` **A4.1** Implement aligned-window and nearest-neighbor mask
  preparation for `.npz` archives.
- `[Complete]` **A4.2** Transform and clip COCO boxes into target pixel
  coordinates, including rotated/different-CRS grids.
- `[Complete]` **A4.3** Drop fully outside annotations, remap retained IDs
  stably, and update every mask pixel, box row, and `num_craters` together.
- `[Complete]` **A4.4** Reapply the established overlap/occlusion heuristic
  after clipping; distinguish valid occlusion from disappearance caused by AOI
  exclusion or resampling.
- `[Complete]` **A4.5** Validate and hash the target archive, including the
  valid empty-label case.
- `[Complete]` **A4.6** Add focused tests for partial boxes, fully outside
  instances, ID gaps, overlapping/occluded instances, subpixel instances,
  empty AOIs, malformed archives, and deterministic output bytes.
- `[Complete]` **A4.7** Implement GeoPackage layer reading, CRS transformation,
  clipping, and rasterization through the A0 interface. Return mask, boxes,
  count, ID mapping, and diagnostics; the worker adapter stages and validates
  the resulting NPZ before tiling. Keep source files read-only.
- `[Complete]` **A4.8** Test synthetic GeoPackages matching the current
  labeling export, including source-grid differences, empty intersections,
  overlap, edge clipping, and invalid geometry. Later validate one real export.

Exit gate: every accepted instance source publishes a self-consistent
target-sized archive whose IDs and boxes are valid in target pixel space.

### A4 implementation and validation

`model/chip_instance_labels.py` adds independently callable public APIs:

- `convert_crater_labels(path, target_grid=..., layer="craters")` returns
  `InstanceLabelConversion(mask, bboxes, num_craters, id_mapping, diagnostics)`.
  It validates the finished GeoPackage and never writes or acquires imagery.
- `materialize_instance_label(request, plan, staging_root=...)` returns a
  `PreparedLabelArtifact`. Exact NPZ inputs retain their original bytes and
  extra arrays; derived NPZs contain the three canonical training arrays.
  Array data stays in the worker, not the request or preparation plan.

Vector outlines (including holes/multipart geometry) are transformed into
target pixel coordinates, adaptively densified for CRS changes, and clipped
to the realized raster footprint. Isolated line/point intersections are
discarded before computing boxes. Explicit point/geometry intersection tests
include boundary-coincident pixel centers without all-touched enlargement or
buffering. Only each clipped feature's pixel window is visited. Supported
features are painted in ascending original ID order; full occlusions retain
their original clipped-outline boxes. Subpixel omissions warn. Source IDs are
compacted monotonically and retained in the artifact map.

Raster archives reuse A3's integer window/nearest-center sampling. Boxes are
transformed as outlines before clipping, including rotated and differing-CRS
grids. Fully excluded annotations are dropped even if a straddling source
pixel carried their ID into the nearest sample (reported separately). Absent
IDs with clipped boxes are retained only when another surviving instance has
pixels in their box region; otherwise they are omitted with a warning. This
remains the accepted raster overlap heuristic, not proof of vector occlusion.

Derived outputs use `<staging_root>/<sample_id>/labels/<sample_id>_label.npz`.
Archives have fixed member order, timestamps, NPY version, and uncompressed ZIP
storage for deterministic bytes. The adapter revalidates source metadata and
hashes, reopens the archive to validate every array against the conversion,
hashes the artifact, and installs it with a no-clobber link. Failures remove
only files owned by that call; existing/concurrent outputs and source labels
are protected. Empty staging directories remain the caller's responsibility.

`test_chip_instance_labels.py` covers the A0 vector oracles in both feature
orders, holes, multipart/touch-only geometry, boundary centers, rotated and
different-CRS grids, larger scenes, partial boxes, compacted IDs, raster
occlusion and vanished support, empty labels, malformed inputs, exact-byte
reuse, deterministic bytes, stale sources/sidecars, reopen corruption,
symlinks, source ownership, and concurrent installation. NumPy/GDAL integration
checks run on HPC; local skipped tests do not establish geospatial
correctness. A real labeling-notebook export remains a later integration check.

Local validation: the new module has 27 tests (six passed, 21 dependency skips).
Chip discovery ran 215 tests (126 passed, 89 skipped). Broader dependency-aware
model discovery (`test_[a-z]*.py`) ran 414 tests (288 passed, 126 skipped).
NumPy and GDAL are unavailable locally. The user subsequently reported the HPC
chip suite passing: **215 tests in 2.060 seconds, OK**, with no skips reported.
This satisfies A4's runtime validation gate; A4 is complete. The real notebook
export check remains deferred as described above.

Run from the checkout's parent directory in the HPC container:

```bash
python -m unittest -v lfm.model.tests.test_chip_instance_labels
python -m unittest discover -s lfm/model/tests -t . -p 'test_chip*.py' -v
```

A4's standalone entry points are now called by A5 worker orchestration below.
Direct low-level acquisition still requires a materialized artifact for derived
labels. A6 notebook integration remains pending.

## Phase A5 — Integrate orchestration and publication `[HPC tests passed; real-data validation pending]`

- `[Implemented]` **A5.1** Insert `label/clip` materialization after successful
  preflight and before `acquire_prepared_request()` in both serial and process
  worker paths.
- `[Implemented]` **A5.2** Ensure materialization failure records a failed
  sample, emits structured diagnostics, starts no tiling, and lets later batch
  samples continue.
- `[Implemented]` **A5.3** Pass the validated label artifact explicitly to
  publication rather than recovering it indirectly from the source-label path.
- `[Implemented]` **A5.4** Preserve byte-identical exact-label publication and
  add rollback-safe publication of derived labels.
- `[Implemented]` **A5.5** Integrate intermediate cleanup/retention and protect
  source labels, shared indexes, and unrelated sample intermediates from
  mutation.
- `[Implemented]` **A5.6** Extend dataset-manifest configuration and sample
  documents with source/derived label provenance and stable configuration IDs.
- `[Complete; HPC tests passed]` **A5.7** Add serial/parallel equivalence, overwrite,
  source-changed-during-run, publication rollback, failure isolation, and
  manifest validation tests.
- `[Implemented]` **A5.8** Implement partial imagery NoData warnings with
  final-grid counts and percentages per band and spatial union. No available
  imagery or wholly invalid required imagery fails the sample; do not publish
  placeholders as a substitute for missing required imagery.
- `[Harness implemented; real data pending]` **A5.9** Run a focused HPC smoke
  test using a finished labeling-notebook GeoPackage, two small geographic AOIs,
  the original pre-tiling WAC VIS grid, and canonical static bands. Compare
  serial/two-worker output pixels, hashes, grids and instance maps; check failure
  isolation with a deliberately missing label. Verify source/index preservation,
  final band order, NoData summaries, cleanup, and visually review saved overlays.

Exit gate: exact and clipped labels both participate in the same atomic
chip-label publication contract under serial and multiprocessing execution;
supported-container tests and the focused real-data smoke test pass.

### A5 implementation and validation

`create_chip()` dispatches semantic/instance preparation inside each worker,
emits `label/clip` progress before tiling, and carries the immutable artifact on
the worker's `PreparedChipRequest`. Parent request/preflight records stay intact;
only artifact metadata returns in `ChipResult`. Direct low-level acquisition of
an unmaterialized derived plan fails with `label_materialization_required`.
Materialization failures become failed sample results and never start tiling.
Preflight/source-hash failures before worker staging retain the typed exception
contract and are isolated by the batch adapter.

Publication receives `prepared_label` explicitly, verifies the source and
prepared hashes plus raster sidecar metadata, validates the final-sized label,
and uses the existing rollback-safe pair publisher. Derived format determines
the expected dataset suffix (GeoPackage -> NPZ, semantic GeoTIFF -> NPY).
Exact arrays retain their original bytes. Source-grid/target-grid provenance,
ID maps, label diagnostics and stable label-plan IDs are stored in schema-v2
manifests/diagnostics, with a versioned label-preparation policy in configuration
identity. A staged artifact path may no longer exist after successful cleanup;
the final `label_path` identifies its published copy.

Cleanup checks protected inputs and rejects sample-directory symlinks. Original
labels/sidecars must be outside the entire intermediate root, preventing another
worker's cleanup from owning them. Overwrite checks the source hash before
clearing old sample intermediates. `never`, `on_failure`, and `always` retention
apply to derived labels as well as imagery. When a batch overwrite fails but its
prior pair survives unchanged, the failed result records `preserved_pair` paths
and hashes; directory validation permits only that verified prior pair and does
not count it as a newly successful sample. Failed replacement never silently
deletes the old pair to satisfy membership checks.

Partial final-grid NoData logs a warning, appears in progress output, and records
per-band counts/percentages plus the spatial union in `imagery_nodata`. Fully
invalid required bands fail with `no_valid_required_imagery` before chip writing;
an empty record collection becomes `no_imagery`. Static/optional placeholders cannot substitute
for required dynamic coverage. Tiling behavior itself is unchanged.

`model/tests/test_chip_a5_integration.py` adds protected-cleanup checks and
NumPy/GDAL stage integration with synthetic acquisition only: real label
materialization, chip reprojection, assembly, pair publication, spawned workers,
manifest equivalence, source-change detection, retention, rollback, GeoPackage
publication, and NoData accounting. Local dependency-aware discovery ran 427
tests (290 passed, 137 skipped); the 13 new tests had two local passes and eleven
NumPy/GDAL skips. Chip-only discovery ran 228 tests (128 passed, 100 skipped).
These skips do not validate raster behavior. Run the full chip
suite in the supported container from the checkout's parent:

```bash
python -m unittest discover -s lfm/model/tests -t . -p 'test_chip*.py' -v
```

#### Focused A5 real-data run

HPC follow-up (2026-10-05): the user reports all tests ran successfully on HPC,
closing A5's unit/integration test gate. The exact test count, duration, and skip
breakdown were not supplied. A5.9's focused real-data run and visual overlay
review remain pending; A5 as a whole is not yet complete.

The supplied WAC example is now
`test_outputs/M1412665711CE.prj.vis.mos_label_craters.gpkg` (three crater
polygons, IDs 1–3, lunar Transverse Mercator with central meridian 24 degrees).
Its recorded source is
`/explore/nobackup/projects/lfm/processed_data/Lunar/LRO_WAC_Pho_Sites/M1412665711CE.prj.vis.mos.tif`.
To run this example from the HPC checkout root:

```bash
mkdir -p scripts/logs
LABEL_GPKG="$PWD/test_outputs/M1412665711CE.prj.vis.mos_label_craters.gpkg" \
SOURCE_RASTER=/explore/nobackup/projects/lfm/processed_data/Lunar/LRO_WAC_Pho_Sites/M1412665711CE.prj.vis.mos.tif \
WAC_PRODUCT_ID=M1412665711CE \
AOI_1= AOI_2= \
sbatch scripts/shell/all_tasks/sbatch_validate_aoi_chip_creation.sh
```

Copy the GeoPackage into that HPC checkout first if it is only present locally.
Omitting both AOIs (or clearing inherited values as above) enables `--auto-aoi`:
select the largest annotation by native geometry area, break ties by lowest
source ID, densify and transform its outline to IAU:30100, then use a 20%-padded
envelope and a second envelope ending at its longitude midpoint. These are
test-selected geographic AOIs, not a new production AOI or label-matching policy.
Automatic selection deliberately excludes polar and antimeridian cases; the
existing pixel cap still applies. Both runs assert the selected crater survives,
with `clipped_instance` only in the second case. Coordinates, selected source ID,
and materialization diagnostics are recorded in the report. The test still runs
WAC + 63 static bands serially and with two workers, checks failure isolation,
output equivalence and source/index preservation, and saves visual overlays.
The wrapper uses Grace, two CPUs, 32 GB, and a one-hour limit. Execution and
visual acceptance remain pending; local syntax/unit checks are not HPC evidence.

The user is preparing a finished GeoPackage from the crater-labeling notebook.
`scripts/python/all_tasks/validate_aoi_chip_creation.py` and
`scripts/shell/all_tasks/sbatch_validate_aoi_chip_creation.sh` take that file,
the original five-band WAC VIS source TIFF, an explicit imagery product ID, and
optionally two explicit AOIs in **NORTH WEST SOUTH EAST** order (IAU:30100),
instead of automatic AOI selection. No reference chip or
label-to-product filename matching is used. Choose distinct small AOIs that
intersect annotations, ideally with a crater crossing one chip edge. Empty
labels remain valid backend outputs, but two empty examples do not satisfy this
visual smoke test. The default cap is 512*512 output pixels per chip.

Submit from the repository root; substitute actual paths/coordinates:

```bash
mkdir -p scripts/logs
LABEL_GPKG=/path/to/finished_label_craters.gpkg \
SOURCE_RASTER=/path/to/PRODUCT.prj.vis.mos.tif \
WAC_PRODUCT_ID=PRODUCT \
AOI_1="NORTH WEST SOUTH EAST" \
AOI_2="NORTH WEST SOUTH EAST" \
sbatch scripts/shell/all_tasks/sbatch_validate_aoi_chip_creation.sh
```

These are inline environment assignments: do not put `&&` between them. Optional
overrides are `LABEL_LAYER`, `WAC_DATA_DIR`, `STATIC_DATA_DIR`, `OUTPUT_ROOT`, and
`CONTAINER_PATH`. The wrapper uses Grace, two CPUs, and
`lfm-container-ipyleaflet`; outputs default to
`notebooks/outputs/chip_a5_<job-id>/`. It refuses to reuse that run directory.
Canonical shared indexes are validated/reused read-only; custom collections get
an application-owned index prepared once in the coordinator before workers.

`a5_validation.json` records successes/failures, timings, paths, hashes, grids,
ID mappings, NoData counts and source/index preservation. `inspection_plots/`
contains chip/instance/overlay panels with clipped boxes. Numerical success is
reported separately from pending human visual review. A7 retains the larger 16-worker,
broader regression and notebook validation; this smoke test does not replace it.

HPC follow-up, job **37938962**: the serial full-crater AOI published successfully,
with 78.63% spatial-union NoData (not a WAC-only statistic). The edge-clipping
AOI completed vector conversion, tiling and reprojection but failed assembly:
required static band `GlobeNoPolesDeltaCPR_v2-offsetto49d.iau` had no valid pixels.
The intentional missing-label request failed before acquisition as expected.
The harness stopped before parallel comparison. Logs are in
`test_outputs/validate_aoi_chips_37938962.{out,err}`. A5.9 remains open; the user
explicitly deferred the static-policy decision and authorized proceeding to A6.
No static requiredness or NoData policy was changed to bypass this failure.

## Phase A6 — Make the notebook AOI-first `[Implemented; HPC Run All pending]`

- `[Implemented]` **A6.1** Replace `REFERENCE_DIR` and `REFERENCE_CHIP` in the
  active configuration with sample ID, geographic IAU:30100 bounds, imagery
  configuration, label path, and label source-grid/provenance inputs. Derive
  dimensions and print the selected CRS/grid and requested/realized footprints.
- `[Implemented]` **A6.2** Keep source directories, split behavior, chip worker
  count, index worker count, output root, and index ownership in the same
  user/derived separation established by the tiling notebook and handoff.
- `[Implemented]` **A6.3** Construct the active request only through
  `chip_request_from_aoi()` and display the materialized target grid and
  transformed geographic query AOI before execution.
- `[Implemented]` **A6.4** Remove reference-TIFF validation from the active
  path. Keep the reference-directory API only as a clearly labeled optional
  compatibility/batch example if it still provides instructional value.
- `[Implemented]` **A6.5** Update the commented full workflow to accept a
  deterministic iterable of AOI requests (including GeoDataFrame-derived
  rectangular requests constructed outside the backend) rather than scanning
  a reference directory.
- `[Implemented]` **A6.6** Update visualization for a request with no reference
  image: show generated chip, clipped label, overlay, and concise source-label
  clipping provenance without leaving a blank reference panel.
- `[Implemented]` **A6.7** State prominently that arbitrary polygons are not
  silently converted to bounding boxes and that larger array labels require
  geospatial source-grid metadata.
- `[Implemented]` **A6.8** Preserve one-time coordinator index preparation via
  `resolve_notebook_source_index()` and `ensure_vector_index()` before chip
  workers start.

Exit gate: a clean-kernel **Run All** creates and visualizes one WAC-plus-static
chip from explicit AOI inputs and a larger source label without reading a
reference chip.

Implementation: `notebooks/chip_example.ipynb` uses the supplied WAC GeoPackage,
explicit product selector and the full-crater geographic AOI from job 37938962.
`read_source_grid()` reads original raster metadata only; output dimensions,
CRS, affine and requested/realized bounds are displayed before acquisition.
Labels use explicit `LabelInput(..., relation="clip_to_target")`. The commented
batch workflow constructs sorted AOI requests and preserves product-grouped
splits, warning that one product cannot populate independent dataset splits.
AOI plots have three panels; reference workflows retain the four-panel layout.
Source-label provenance, instance mappings and clipping diagnostics are printed;
failed requests show diagnostics without attempting a plot. Shared-index setup
and static policy are unchanged. Notebook outputs and execution counts are clear.

Local helper validation: three tests passed, one NumPy/Matplotlib plotting test
skipped. Checks cover metadata-only reads (including rotated affines), missing
CRS rejection, notebook cell syntax, uncommented batch-example syntax and unique
cell IDs. The supported-container plotting test and clean-kernel Run All remain
pending; no notebook execution or visual acceptance is claimed. Full local chip
discovery then ran 235 tests: 133 passed, 102 dependency skips. The first full
run exposed a test-isolation issue: importing the standalone smoke CLI prepended
the checkout to `sys.path`, preventing other tests' spawned workers from finding
`lfm.model`. The smoke-harness tests now restore the import path after loading
the CLI; the full suite passed on rerun. Production import behavior is unchanged.

## Phase A7 — Regression and real-data validation `[Not Started]`

- `[Not Started]` **A7.1** Run the focused request, label, preflight,
  acquisition, creation, publication, notebook-helper, and dataset-loader unit
  suites in the supported container.
- `[Not Started]` **A7.2** Re-run existing exact-label reference workflows and
  require unchanged chip pixels, byte-identical labels, split assignments,
  filenames, and manifests except for explicitly versioned added fields.
- `[Not Started]` **A7.3** Run one real WAC AOI whose label covers a larger
  source TIFF. Verify source/target grids, coverage relation, nearest/window
  method, final shape, VIS-then-UV-plus-static band order, label values, and
  visual alignment.
- `[Not Started]` **A7.4** Run at least two AOIs against the same full-scene
  label and confirm unique sample outputs, shared source provenance, product
  grouping, and no cross-sample intermediate collisions.
- `[Not Started]` **A7.5** Run one negative AOI partially outside its label and
  prove it fails before tiling while a later valid AOI succeeds.
- `[Not Started]` **A7.6** Compare serial and 16-worker results, including
  output hashes, manifests, failure ordering, progress events, and peak memory.
- `[Not Started]` **A7.7** Execute `notebooks/chip_example.ipynb` top-to-bottom
  on Grace with `lfm-container-ipyleaflet`, inspect the saved plot, and record
  the Slurm job, elapsed time, report paths, and any accepted exceptions.

Exit gate: focused tests and real-data runs pass, outputs are visually reviewed,
and no validation claim exceeds the recorded evidence.

## Phase A8 — Documentation and migration closure `[Not Started]`

- `[Not Started]` **A8.1** Update the chip modernization plan with the accepted
  AOI-first and larger-label contracts, implementation evidence, and C8/C9
  status.
- `[Not Started]` **A8.2** Update dataset contribution documentation with
  source-label grid/sidecar requirements and canonical derived-label outputs.
- `[Not Started]` **A8.3** Update backend architecture documentation and public
  docstrings for request construction, label preparation, diagnostics,
  publication, and manifest provenance.
- `[Not Started]` **A8.4** Document retained compatibility for reference TIFFs
  and exact-size labels, plus any intentionally deferred polygon or label-format
  support.

Exit gate: the notebook, backend docs, dataset guide, handoff, and canonical
modernization plan describe the same implemented behavior.

## Validation matrix

| Case | Expected result |
|---|---|
| Explicit AOI + exact `.npy` label | Existing byte-preserving path; no reference TIFF |
| Explicit AOI + larger aligned `.npy` label | Integer window, target-sized `.npy` |
| Explicit AOI + larger projected semantic label | Nearest warp, exact target grid |
| Explicit AOI + semantic label GeoTIFF | Embedded grid, canonical `.npy` output |
| Explicit AOI + larger instance `.npz` | Clipped mask/boxes, compact IDs, valid count |
| Two AOIs sharing one parent label | Two unique pairs with shared source provenance |
| AOI outside or partly outside a raster label grid | Per-sample failure before tiling; no pair |
| Explicit label with a different filename/product | Accepted if technically valid; no identity comparison |
| GeoPackage without readiness metadata | Accepted under the full-raster assumption |
| Partial imagery NoData | Warn with final-grid counts/percentages; publish usable pair |
| WAC or NAC plus static | Dynamic imagery determines output resolution; static follows that grid |
| Multiple dynamic modalities including WAC | Emit precedence warning; all bands/labels match the WAC grid |
| Crater crossing a raster boundary | Clip at the raster edge using the A0 conversion contract |
| Static only | AOI-center LTM zone, 100 m pixels, projected zero anchor; no intersecting WAC required |
| No imagery or unusable required imagery | Clear per-sample failure; no pair; later samples continue |
| Outward grid rounding | Preserve native affine lattice; contain AOI; clip labels at realized raster edges |
| Larger array with no source grid | Typed failure; never infer location from shape |
| Arbitrary nonrectangular geometry | Explicit rejection; no silent envelope |
| Antimeridian AOI | Existing split-query acquisition; one logical target request |
| Polar AOI | Existing `unsupported_polar_coverage` preflight failure |
| Exact legacy reference request | Unchanged chip and byte-identical label behavior |
| Publication failure after clipping | Both derived label and chip rolled back |
| Serial versus multiprocessing | Same ordered results, outputs, IDs, and diagnostics |

## Expected implementation surface

Primary files likely to change:

- `model/chip_types.py`
- `model/chip_requests.py`
- `model/chip_labels.py`
- `model/chip_preflight.py`
- `model/chip_creation.py`
- `model/chip_publication.py`
- `model/chip_notebook_utils.py`
- `model/tests/test_chip_types.py`
- `model/tests/test_chip_requests.py`
- `model/tests/test_chip_labels.py`
- `model/tests/test_chip_creation.py`
- `model/tests/test_chip_publication.py`
- `notebooks/chip_example.ipynb`
- `docs/chip_creation_modernization_plan.md`
- `docs/dataset_contribution.md`

`model/tiling.py`, `model/tiling_config.py`, and
`model/tiling_results.py` should not require behavior changes. If implementation
appears to require one, stop and re-evaluate the chip/tiling boundary before
editing them.

## Principal risks and mitigations

| Risk | Mitigation |
|---|---|
| A larger array is clipped without trustworthy georeferencing | Require embedded, explicit, or one unambiguous sidecar source grid |
| AOI is rounded or snapped away from the scientist's study extent | Make the request target grid authoritative; warp labels to it, never replace it with tile or source-label bounds |
| Categorical IDs are corrupted by interpolation | Nearest-neighbor only; validate integer values after materialization |
| Instance boxes, mask IDs, and count diverge | Transform, clip, remap, and validate them as one artifact transaction |
| Raster label grid does not cover the realized output | Densified containment; fail before tiling; no background padding; vector inputs use the full-raster assumption |
| Differently named labels are rejected despite explicit association | Remove identity gates for supplied paths; retain technical format/georeferencing checks |
| Supplied labels cannot be transformed to the target grid | Validate georeferencing and output alignment without filename/product checks |
| Full `.npz` masks create high worker memory | Keep plans small, materialize in workers, window formats that support it, measure `.npz` peak memory, and document limits |
| Derived labels weaken publication atomicity | Hash validated artifacts and retain the existing staged pair/rollback protocol |
| Batch workers rebuild indexes | Preserve coordinator-only index preparation from the handoff |
| New AOI notebook silently enables polar chips | Retain and display the existing typed polar rejection |

## Completion definition

This plan is complete only when a user can run the public notebook from a clean
kernel with no reference chip, provide a complete rectangular AOI/grid and a
larger georeferenced label, and obtain an atomically published chip/label pair
whose two artifacts exactly match the requested target grid. Exact legacy
reference workflows must remain regression-safe, label failures must remain
per-sample and pre-tiling, and the tiling contract must remain unchanged.
