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

The proposed crater GeoPackage extension is specified in
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
9. Multiple selected dynamic modalities emit a warning that WAC takes
   precedence; align all selected modalities to the WAC grid.
10. Clip crater annotations to the final raster edges. Do not enlarge the
    raster to accommodate a crater crossing its boundary. Detailed instance
    encoding is defined by this plan's label-conversion interface.
11. Output CRS comes from the selected original source raster before tiling.
    For WAC, use the VIS source CRS/grid; UV and other modalities follow it.
    Read this CRS from source metadata, not from the intermediate LTM cubes.

These decisions supersede earlier proposed review and identity gates. A0 is
not complete: static-only grid definition, pixel-window rules, numerical tolerances, the
conversion interface, and fixtures remain to be finalized.

## Status convention

- `[Not Started]`: no implementation work has begun.
- `[In Progress]`: active work; only one sub-step should have this status.
- `[Complete]`: implemented, tested, documented, and accepted for its stated
  scope.
- `[Deferred]`: intentionally removed from the current sequence, with its
  reason and restart condition recorded.

Phases and sub-steps are sequential. A phase is complete only when all required
sub-steps and validation gates in that phase are complete.

## Current behavior and gap

The backend already supports `chip_request_from_aoi()`, but the active path in
`notebooks/chip_example.ipynb` still:

- requires `REFERENCE_CHIP` during path validation;
- extracts the exact target grid from that TIFF;
- derives the request sample ID and product selector from its filename; and
- expects a label whose identity, dimensions, and optional grid exactly match
  the final chip.

The current label pipeline accepts semantic `.npy` and instance `.npz` files.
`model/chip_labels.py` rejects any mask whose shape differs from the target
grid, and optional label-grid metadata must match the target exactly.
`model/chip_publication.py` then byte-copies that source label into the dataset.
These are deliberate safeguards, but they prevent reuse of a georeferenced
label covering a full source scene or other larger parent AOI.

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

- The public user AOI is geographic IAU:30100 and remains the final output goal.
  One example AOI and iterables of batch AOIs use the same contract.
  Pixel sizing and dynamic output CRS follow the selected native grid below.
  A request must materialize one
  complete `TargetGrid`: lunar-compatible CRS, finite rectangular bounds,
  invertible affine transform, positive width, and positive height.
- The final chip and final label must have exactly that grid and shape. LTM tile
  boundaries remain intermediate acquisition geometry and must not replace the
  target extent.
- An AOI geometry alone is not a raster grid. The notebook accepts geographic
  bounds with fixed IAU:30100 CRS and derives width/height from the selected
  modality's native resolution. Do not infer a different study area from a projected
  bounding envelope. Freeze longitude-seam handling alongside grid derivation.
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

Crater geometry crossing the final raster boundary is clipped at that boundary
under the user's decision. That rule governs label geometry; an integer
pixel-window convention is still needed to construct the raster. Specify and
test its treatment of partial edge pixels without shifting or rescaling the
native lattice silently. Record both requested geographic bounds and realized
raster footprint. A projected envelope of a geographic rectangle may also
include area outside the requested rectangle; settle that footprint handling
explicitly rather than treating crater clipping as a solution to it.

A0 must still define the following implementation details:

- Use the established 100 m WAC VIS reference resolution for the static-only
  fallback without opening or acquiring WAC imagery. Its standalone CRS and
  pixel-lattice selection are addressed below.
- Selection among multiple candidate rasters within the winning modality and
  the fallback for multiple custom dynamic modalities with no WAC configured;
  do not add WAC imagery or choose by file iteration order implicitly.
- Static-only output CRS and pixel lattice: WAC-equivalent resolution alone
  does not supply these, and static-only must not require intersecting WAC data.
- Integer source-window convention, transformed geographic footprint handling,
  and numerical tolerances. Craters clip to the resulting raster edges;
  native-grid alignment and the requested study extent must be accounted for.

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
- For raster/array labels, the source grid must cover the target; array shape
  alone cannot locate labels. For GeoPackages, assume the scientist supplied
  full-raster labels without requiring reviewed-coverage metadata. Coordinate
  source-footprint representation in the conversion interface if containment is
  needed; do not use crater feature bounds as the raster footprint.
- Do not validate label filenames, sample IDs, or product IDs against imagery.
  Preserve technical file, CRS, geometry, and final target-grid validation.
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
- `AOI_BOUNDS`: geographic `(west, south, east, north)` in repository IAU:30100;
  map explicitly to tiling's named corner arguments.
- AOI CRS is loaded from repository IAU:30100 rather than user-selected.
- Output resolution: derived from selected dynamic imagery, or the WAC default
  for static-only operation. Dimensions are derived, not required example
  inputs. Dynamic output CRS follows the selected raster (WAC precedence);
  static-only grid definition and integer edge-window rules remain to finalize.
- `LABEL_PATH`: explicit source-label association.
- `LABEL_SOURCE_GRID`: optional structured source grid when the label does not
  embed one. This may be read from a sidecar or derived explicitly from the
  corresponding full-scene TIFF without making that TIFF the chip target.
- `SPLIT_GROUP_KEY`: normally the WAC/NAC product ID or another scientifically
  meaningful leakage group.

Adapt `chip_request_from_aoi()` to the agreed geographic-input/output-grid
contract once sizing and CRS are frozen. Validate that the grid represents the
requested study area, including longitude wrapping. The active notebook no
longer requires `REFERENCE_DIR` or `REFERENCE_CHIP`.

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

The caller explicitly supplies the label path for each request, including
exact labels. Do not require filename, sample-ID, product-ID, or source-scene
equality with the chip. One label may feed multiple AOIs, including imagery
from another modality. Batch input must provide these associations explicitly;
do not introduce automatic label matching. Existing directory-discovery
compatibility needs an explicit migration decision, not a new validation gate.
Imagery selectors and unique output sample IDs remain independent requirements.

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

### Semantic-label output

- Read only the needed source window when the storage format permits it.
- Produce one 2D integer mask with shape `(AOI_HEIGHT, AOI_WIDTH)`.
- Preserve class IDs. Map declared source NoData only according to an explicit
  label NoData policy; do not infer invalid pixels from magnitude.
- Validate the derived mask against the target grid and publish it as
  `<sample-id>_label.npy`.

### Instance-label output

The following raster-archive algorithm is a proposal to freeze during A0.
This plan owns the clipping, ID, box, overlap, and occlusion semantics, including
GeoPackage conversion. Use the interface in the linked GeoPackage contract to
produce a consistent target-sized mask/boxes/count artifact and local fixtures.

Proposed raster instance clipping updates mask, boxes, and count together:

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

## Phase A0 — Freeze AOI and label-clipping contracts `[Not Started]`

- `[Not Started]` **A0.1** Confirm the initial input formats: semantic `.npy`,
  instance `.npz`, single-band integer semantic GeoTIFF, and crater GeoPackage.
  Explicitly defer instance GeoTIFFs without box/count metadata.
- `[Not Started]` **A0.2** Freeze the notebook's AOI inputs, output-grid
  derivation, sample/product identity rules, and rectangular-only scope.
- `[Not Started]` **A0.3** Specify explicit label-path association for exact and
  full-scene inputs without filename/product matching; permit parent-label reuse
  across AOIs. Define migration of legacy directory-based discovery.
- `[Not Started]` **A0.4** Freeze exact, aligned-window, and nearest-warp
  relations; full-coverage requirements; categorical NoData behavior; and
  numerical tolerances.
- `[Not Started]` **A0.5** Define our label-conversion interface: GeoPackage
  path, layer, exact target grid, conversion options, validated instance result,
  and structured diagnostics. Freeze clipping/ID/box/overlap/subpixel/occlusion
  semantics using the current producer code and synthetic acceptance fixtures;
  a real notebook run is not a prerequisite.
- `[Not Started]` **A0.6** Add small committed fixtures or fixture builders for
  exact, larger aligned, differently projected, partial-coverage, semantic,
  overlapping-instance, and the linked GeoPackage acceptance cases before
  implementation.
- `[Not Started]` **A0.7** Incorporate the accepted GeoPackage full-raster
  assumption, no readiness state, no label matching, partial imagery NoData
  warnings, and no-imagery failures. Define geometric metadata and conversion
  within this plan and carry integration/test work into phases A1–A8.

Exit gate: the accepted contract and fixtures make every expected output,
warning, and failure deterministic without relying on a real Explore dataset.

## Phase A1 — Extend request and label types `[Not Started]`

- `[Not Started]` **A1.1** Add immutable `LabelInput`,
  `LabelPreparationPlan`, and `PreparedLabelArtifact` records in the chip type
  layer with path, kind, grid, relation, hash, and diagnostic
  validation.
- `[Not Started]` **A1.2** Extend `ChipRequest` compatibly so legacy
  `label_path`/`label_grid` requests normalize to exact mode while AOI callers
  can explicitly request clipping.
- `[Not Started]` **A1.3** Keep `chip_request_from_aoi()` as the canonical
  constructor and derive width/height using the accepted native-dynamic/WAC
  static-only resolution policy and A0 grid rules. Resolve needed source
  metadata before label preparation, without running tiling. Preserve explicit
  reference-grid compatibility and reject ambiguous grid-reference choices.
- `[Not Started]` **A1.4** Extend result, diagnostic, progress-stage, and
  manifest schemas with label-preparation provenance without placing arrays in
  serializable request objects.
- `[Not Started]` **A1.5** Add dictionary/config round-trip tests and
  backward-compatibility tests for existing exact-label callers.

Exit gate: old requests behave identically; new requests can describe a
full-scene label and exact target grid without a reference TIFF.

## Phase A2 — Resolve source labels and plan clipping `[Not Started]`

- `[Not Started]` **A2.1** Refactor label validation into source-structure,
  source-grid/relation, and final-target validation rather than applying the
  target shape check while opening the source.
- `[Not Started]` **A2.2** Support explicit array, GeoTIFF, and GeoPackage
  label paths and sidecar georeferencing. Implement the agreed migration for
  directory-based callers without imposing identity checks on supplied labels.
- `[Not Started]` **A2.3** Verify explicitly associated labels are readable and
  structurally/geospatially usable; do not compare their names or products with
  the request's imagery selectors.
- `[Not Started]` **A2.4** Compute exact/aligned/warp relations, source windows,
  densified coverage, and output encoding without writing files.
- `[Not Started]` **A2.5** Return typed per-sample failures for missing grid
  metadata, incompatible CRS, incomplete coverage, ambiguous sidecars,
  malformed source contents, and unsupported formats.
- `[Not Started]` **A2.6** Prove preflight remains read-only and never invokes
  tiling for a rejected label.

Exit gate: preflight deterministically accepts or rejects every A0 fixture and
produces no dataset/intermediate output.

## Phase A3 — Materialize semantic labels `[Not Started]`

- `[Not Started]` **A3.1** Implement the exact no-copy plan and integer-window
  slicing for aligned `.npy`, `.npz` masks, and semantic GeoTIFFs.
- `[Not Started]` **A3.2** Implement nearest-neighbor warp to the exact target
  transform, CRS, width, and height for non-aligned semantic labels.
- `[Not Started]` **A3.3** Preserve integer class IDs and apply only declared
  NoData/background rules; reject values or conversions that cannot be
  represented safely.
- `[Not Started]` **A3.4** Write derived semantic labels to deterministic
  per-sample staging paths, validate their shape/content/grid provenance, and
  compute hashes.
- `[Not Started]` **A3.5** Add window-versus-warp equivalence, CRS, rotated
  grid, edge, empty-background, partial-coverage, dtype, and NoData tests.

Exit gate: every accepted semantic source produces one target-sized integer
`.npy` label before tiling, with exact-grid inputs still byte-preservable.

## Phase A4 — Materialize instance labels `[Not Started]`

Implement the interface and semantics frozen in A0 for raster archives and
GeoPackage input. The converter remains usable independently of the notebook
and imagery acquisition.

- `[Not Started]` **A4.1** Implement aligned-window and nearest-neighbor mask
  preparation for `.npz` archives.
- `[Not Started]` **A4.2** Transform and clip COCO boxes into target pixel
  coordinates, including rotated/different-CRS grids.
- `[Not Started]` **A4.3** Drop fully outside annotations, remap retained IDs
  stably, and update every mask pixel, box row, and `num_craters` together.
- `[Not Started]` **A4.4** Reapply the established overlap/occlusion heuristic
  after clipping; distinguish valid occlusion from disappearance caused by AOI
  exclusion or resampling.
- `[Not Started]` **A4.5** Validate and hash the target archive, including the
  valid empty-label case.
- `[Not Started]` **A4.6** Add focused tests for partial boxes, fully outside
  instances, ID gaps, overlapping/occluded instances, subpixel instances,
  empty AOIs, malformed archives, and deterministic output bytes.
- `[Not Started]` **A4.7** Implement GeoPackage layer reading, CRS transformation,
  clipping, and rasterization through the A0 interface. Return mask, boxes,
  count, ID mapping, and diagnostics; the worker adapter stages and validates
  the resulting NPZ before tiling. Keep source files read-only.
- `[Not Started]` **A4.8** Test synthetic GeoPackages matching the current
  labeling export, including source-grid differences, empty intersections,
  overlap, edge clipping, and invalid geometry. Later validate one real export.

Exit gate: every accepted instance source publishes a self-consistent
target-sized archive whose IDs and boxes are valid in target pixel space.

## Phase A5 — Integrate orchestration and publication `[Not Started]`

- `[Not Started]` **A5.1** Insert `label/clip` materialization after successful
  preflight and before `acquire_prepared_request()` in both serial and process
  worker paths.
- `[Not Started]` **A5.2** Ensure materialization failure records a failed
  sample, emits structured diagnostics, starts no tiling, and lets later batch
  samples continue.
- `[Not Started]` **A5.3** Pass the validated label artifact explicitly to
  publication rather than recovering it indirectly from the source-label path.
- `[Not Started]` **A5.4** Preserve byte-identical exact-label publication and
  add rollback-safe publication of derived labels.
- `[Not Started]` **A5.5** Integrate intermediate cleanup/retention and protect
  source labels, shared indexes, and unrelated sample intermediates from
  mutation.
- `[Not Started]` **A5.6** Extend dataset-manifest configuration and sample
  documents with source/derived label provenance and stable configuration IDs.
- `[Not Started]` **A5.7** Add serial/parallel equivalence, overwrite,
  source-changed-during-run, publication rollback, failure isolation, and
  manifest validation tests.
- `[Not Started]` **A5.8** Report final-grid partial NoData counts/percentages
  as warnings while allowing usable chips; fail samples with no imagery or
  unusable required imagery and preserve later batch processing.

Exit gate: exact and clipped labels both participate in the same atomic
chip-label publication contract under serial and multiprocessing execution.

## Phase A6 — Make the notebook AOI-first `[Not Started]`

- `[Not Started]` **A6.1** Replace `REFERENCE_DIR` and `REFERENCE_CHIP` in the
  active configuration with sample ID, IAU:30100 geographic bounds, agreed
  output-grid sizing, label
  path, and label source-grid/provenance inputs.
- `[Not Started]` **A6.2** Keep source directories, split behavior, chip worker
  count, index worker count, output root, and index ownership in the same
  user/derived separation established by the tiling notebook and handoff.
- `[Not Started]` **A6.3** Construct the active request only through
  `chip_request_from_aoi()` and display the materialized target grid and
  transformed geographic query AOI before execution.
- `[Not Started]` **A6.4** Remove reference-TIFF validation from the active
  path. Keep the reference-directory API only as a clearly labeled optional
  compatibility/batch example if it still provides instructional value.
- `[Not Started]` **A6.5** Update the commented full workflow to accept a
  deterministic iterable of AOI requests (including GeoDataFrame-derived
  rectangular requests constructed outside the backend) rather than scanning
  a reference directory.
- `[Not Started]` **A6.6** Update visualization for a request with no reference
  image: show generated chip, clipped label, overlay, and concise source-label
  clipping provenance without leaving a blank reference panel.
- `[Not Started]` **A6.7** State prominently that arbitrary polygons are not
  silently converted to bounding boxes and that larger array labels require
  geospatial source-grid metadata.
- `[Not Started]` **A6.8** Preserve one-time coordinator index preparation via
  `resolve_notebook_source_index()` and `ensure_vector_index()` before chip
  workers start.

Exit gate: a clean-kernel **Run All** creates and visualizes one WAC-plus-static
chip from explicit AOI inputs and a larger source label without reading a
reference chip.

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
| Static only | WAC resolution without requiring intersecting WAC imagery |
| No imagery or unusable required imagery | Clear per-sample failure; no pair; later samples continue |
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
| Raster label grid does not cover the complete AOI | Densified footprint containment; fail before tiling and never pad truth; coordinate vector footprint handling with labeling owner |
| Supplied labels cannot be transformed to the target grid | Validate georeferencing and output alignment; trust the scientist's explicit association without filename/product checks |
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
