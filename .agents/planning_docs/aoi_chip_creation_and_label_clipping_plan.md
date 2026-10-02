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
2. a single-band, integer, georeferenced `.tif`/`.tiff` semantic mask.

A standalone instance GeoTIFF is not sufficient because it does not contain the
required COCO boxes and annotation count. Instance input remains `.npz` unless
a future, separately specified metadata format is added.

## Required invariants

### Tiling boundary

- Do not change tiling's AOI, routing, tile-size, ordering, resampling, source
  selection, index ownership, or structured-result contracts.
- Continue calling strict `create_tiles_for_aoi()` from chip acquisition with
  an explicit product selector for every `product_id` source.
- Continue splitting antimeridian-crossing geographic queries in chip
  acquisition and deduplicating `TileCubeRecord` objects by structured fields.
- Keep all label work downstream of the target request and upstream of tiling.
  Never add label cropping to `model/tiling.py`.
- Preserve the current numbered-LTM chip restriction. Upstream polar tiling
  support does not enable polar chip creation.

### Target grid and AOI

- The user AOI remains the final output goal. A request must materialize one
  complete `TargetGrid`: lunar-compatible CRS, finite rectangular bounds,
  invertible affine transform, positive width, and positive height.
- The final chip and final label must have exactly that grid and shape. LTM tile
  boundaries remain intermediate acquisition geometry and must not replace the
  target extent.
- An AOI geometry alone is not a raster grid. The notebook must require bounds,
  CRS, and either width/height or another explicit resolution contract from
  which width and height are deterministically derived.
- Initial scope remains rectangular AOIs. Do not silently replace an arbitrary
  polygon with its envelope. Polygon masking is a separate feature.

### Label safety

- A label larger than the target is accepted only when its source grid is
  independently known. Array shape alone cannot locate the chip AOI within a
  full-scene mask.
- The source label footprint must completely cover the target grid. Do not pad
  missing label coverage with background, because that would create unverified
  training truth.
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
- `AOI_CRS_WKT`: target/output CRS.
- `AOI_BOUNDS`: `(left, bottom, right, top)` in that CRS.
- `AOI_WIDTH` and `AOI_HEIGHT`: exact output columns and rows.
- `LABEL_PATH`: explicit source-label association.
- `LABEL_SOURCE_GRID`: optional structured source grid when the label does not
  embed one. This may be read from a sidecar or derived explicitly from the
  corresponding full-scene TIFF without making that TIFF the chip target.
- `SPLIT_GROUP_KEY`: normally the WAC/NAC product ID or another scientifically
  meaningful leakage group.

`chip_request_from_aoi()` continues to derive a north-up affine transform when
one is not supplied and validates that the resulting raster bounds equal the
input AOI. The notebook no longer requires `REFERENCE_DIR` or
`REFERENCE_CHIP` for the active example.

### Label input and identity

Add a typed label-input contract rather than overloading a bare path. The
recommended record contains:

- source path;
- semantic or instance kind;
- source sample/scene identity;
- source `TargetGrid` when not embedded in the file;
- requested relation: `exact` or `clip_to_target`; and
- optional sidecar/provenance path.

Keep `ChipRequest.label_path` and `label_grid` as backward-compatible inputs,
normalizing them into this record. New AOI callers should use the typed form.

The existing exact-label path continues to require the label's normalized
sample ID to equal the output sample ID. A full-scene label may legitimately
feed multiple AOI samples, so clip mode instead requires an explicit source
scene identity. For WAC/NAC, that source identity must agree with the product
prefix used by the request or an explicit selector. Merely passing a
differently named file must not disable identity validation.

For `.npy` and `.npz`, accept source-grid metadata from either:

1. the typed request association; or
2. one unambiguous JSON sidecar containing `source_grid` and source identity.

Continue reading legacy sidecar `target_grid` as an exact-grid association.
Do not reinterpret it as a larger source grid silently.

For a semantic label GeoTIFF, read its CRS, affine transform, bounds, width,
height, band count, dtype, and NoData directly. Require exactly one integer
band. The canonical published training label remains `.npy`, so existing
dataset loaders do not acquire a GeoTIFF dependency.

### Spatial-relation plan

Preflight classifies one of three relations:

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

The parent-process preflight remains non-writing. It validates identity,
source structure, grid metadata, complete coverage, and produces a compact
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
  instance `.npz`, and single-band integer semantic GeoTIFF. Explicitly defer
  instance GeoTIFFs without box/count metadata.
- `[Not Started]` **A0.2** Freeze the notebook's AOI inputs, output-grid
  derivation, sample/product identity rules, and rectangular-only scope.
- `[Not Started]` **A0.3** Freeze label identity rules for exact versus
  full-scene inputs, including reuse of one parent label by multiple AOIs.
- `[Not Started]` **A0.4** Freeze exact, aligned-window, and nearest-warp
  relations; full-coverage requirements; categorical NoData behavior; and
  numerical tolerances.
- `[Not Started]` **A0.5** Freeze instance box clipping, stable ID remapping,
  occlusion, empty-AOI, and out-of-AOI removal semantics.
- `[Not Started]` **A0.6** Add small committed fixtures or fixture builders for
  exact, larger aligned, differently projected, partial-coverage, semantic,
  and overlapping-instance cases before implementation.

Exit gate: the accepted contract and fixtures make every expected output,
warning, and failure deterministic without relying on a real Explore dataset.

## Phase A1 — Extend request and label types `[Not Started]`

- `[Not Started]` **A1.1** Add immutable `LabelInput`,
  `LabelPreparationPlan`, and `PreparedLabelArtifact` records in the chip type
  layer with strict path, identity, kind, grid, relation, hash, and diagnostic
  validation.
- `[Not Started]` **A1.2** Extend `ChipRequest` compatibly so legacy
  `label_path`/`label_grid` requests normalize to exact mode while AOI callers
  can explicitly request clipping.
- `[Not Started]` **A1.3** Keep `chip_request_from_aoi()` as the canonical
  constructor and add only small conveniences needed to materialize width and
  height from an explicitly chosen resolution. Reject ambiguous combinations.
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
- `[Not Started]` **A2.2** Extend resolution to supported GeoTIFF labels and
  the new sidecar schema while retaining exact full-sample-ID lookup for legacy
  directory-based labels.
- `[Not Started]` **A2.3** Require explicit association for a differently
  named full-scene label and verify its source identity against the request's
  product/selector identity.
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

Exit gate: exact and clipped labels both participate in the same atomic
chip-label publication contract under serial and multiprocessing execution.

## Phase A6 — Make the notebook AOI-first `[Not Started]`

- `[Not Started]` **A6.1** Replace `REFERENCE_DIR` and `REFERENCE_CHIP` in the
  active configuration with sample ID, AOI CRS, bounds, dimensions, label
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
| AOI outside or partly outside label | Per-sample failure before tiling; no pair |
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
| Parent label does not cover the complete AOI | Densified footprint containment; fail before tiling and never pad truth |
| Differently named labels bypass identity checks | Require typed source-scene identity and validate it against product/selector identity |
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
