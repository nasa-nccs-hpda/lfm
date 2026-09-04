# Chip Creation Modernization Plan

## Objective

Replace the training-GeoDataFrame-driven, WAC/static-specific chip workflow
with a target-grid-driven workflow configured by `ChipConfig`. Each
`ChipRequest` defines one authoritative output AOI and raster grid, either
directly or through a reference TIFF; one or more composed `TileConfig`
acquisition groups supply the LTM datacubes; and the chip layer merges,
reprojects, orders, combines, validates, splits, and writes model-ready
samples. One request produces one final chip matching that request's target
grid. LTM zones and tiles remain intermediate acquisition units rather than
final chip boundaries.

## Starting point and prerequisite

The tiling modernization plan is `[Complete]` as of 2026-09-03. Its
configuration-driven backend was regression-tested through **Phase T6**, and
the supported notebook and migration closeout were completed in Phases T7 and
T8. The stable T0–T6 contract is the integration prerequisite for this plan,
and chip modernization may start now.

The earlier dependency on tiling **T8.5** was resolved by assigning migration
of the legacy chip acquisition path to this plan. It will consume the stable
`TileConfig`, public `create_tiles_for_*` API, and structured `TileCubeRecord`
contract rather than maintaining a second tiler.

The source of truth for the tiling handoff is the **Stable tiling contract for
chip creation** section in
[`tiling_modernization_plan.md`](tiling_modernization_plan.md). The Armstrong
scheme, IAU:30100 CRS, LTM zone/zoom/tile address, and polar limitation are
documented in [`TMS/README.md`](../TMS/README.md).

## Status and sequencing

- `[Planned]`: not started.
- `[In-P]`: currently in progress.
- `[Complete]`: implemented, tested, and documented for the stated scope.

Phases and sub-steps are strictly sequential. Start a sub-step only after the
preceding sub-step is `[Complete]`, and start a phase only after every sub-step
in the preceding phase is `[Complete]`. Only one sub-step should be `[In-P]` at
a time. A phase becomes `[Complete]` when all of its sub-steps are complete.

## Tiling backend handoff (2026-09-03)

### Stable implementation map

| Concern | Existing implementation to consume |
|---|---|
| Public configuration | `model/tiling_config.py`: `TileConfig`, `TileSourceConfig`, `BandNoDataOverride`, `tile_config_from_dict` |
| Public acquisition API | `model/tiling.py`: `create_tiles_for_aoi`, `create_tiles_for_point`, `create_tiles_for_index` |
| Orchestration | `model/configured_tiler.py`; do not call `ConfiguredTiler` directly from chip code |
| Zone and tile geometry | `model/TmsIntersector.py`, `model/TmsZoneDef.py`, `model/TmsTileDef.py` |
| Lunar geographic CRS | `model/lunar_crs.py` loading repository `TMS/IAU_30100_2015.wkt` |
| Read-only source indexes | `model/vector_index.py` supporting `.shp` and `.gpkg` |
| Selection and NoData policy | `model/tiling_policy.py` |
| Warp and cube writing | `model/raster_cube.py` |
| Structured outputs and errors | `model/tiling_results.py`: `TileCubeRecord`, `TileSourceError`, `MissingRequiredSourceError` |
| Canonical static bands | `model/static_band_contract.py` and `lfm/all_models/all_tasks/tiling_utils.py::make_static_source` |
| Worked example | `notebooks/tiling_example.ipynb` |

### Contract chip code must preserve

- Import the public exports from `lfm.model` in repository scripts that put the
  repository parent on `sys.path`. Do not import private acquisition helpers or
  instantiate the deprecated `lfm.model.Pipeline.Pipeline`.
- Supply AOIs as `ul_lat`, `ul_lon`, `lr_lat`, and `lr_lon` in IAU:30100. Use
  `load_lunar_geographic_wkt()`; do not add another inline lunar WKT string.
- Treat every source index as a read-only input. The configured location field
  can resolve absolute paths or paths relative to that source's `data_dir`.
  Chip creation must not create an index as a side effect of tiling.
- Pass product selectors as a mapping keyed by source name. A source configured
  as `product_id` requires a selector; `all_intersecting` rejects one.
- Group records by structured fields:
  `(zone, zoom_level, tile_x, tile_y, source_name)`. Filenames are for human
  inspection only and are not an orchestration interface.
- Expect partial record sets. Required missing coverage raises
  `MissingRequiredSourceError`; optional sparse coverage can produce no record
  for one source while other cubes have already been written. Record and clean
  up intermediate outputs according to an explicit chip-level status policy.
- Tiling always uses bilinear resampling and writes 512×512 LTM GeoTIFFs. Any
  chip-stage resampling policy is separate and must not be implemented by
  weakening the tiling contract.
- The canonical static source requests exactly 63 bands in contract order.
  Successful static cubes write all 63 bands using `-32768` as the common
  output NoData value. The two Mini-RF bands declare
  `-3.4028230607370965e38` as a source-only sentinel, which is converted to
  `-32768` during tiling. The notebook's WAC and NAC source configurations set
  `preserve_source_nodata=True`, so their output NoData values follow the
  source metadata rather than being inferred from the modality names.
- A `TileConfig` has exactly one zoom shared by all sources in that acquisition
  group. The current notebook uses WAC plus static at zoom 5 and 1 m NAC plus
  static at zoom 11. `ChipConfig` must not assume all sensor families should
  share the repository's zoom-5 default.
- Use a separate intermediate output directory for each target sample and
  acquisition group. Static filenames omit product IDs, so sharing one output
  directory across samples can cause valid tile-address collisions.
- Only the numbered LTM path is currently supported. LPN/LPS polar metadata is
  present but was not integrated because its geometry differs from the LTM
  longitude-band grids. Detect and report unsupported polar target extents instead
  of routing them through LTM code.

### Validation evidence already completed

- 42 modern backend contract tests and 25 selected legacy geometry/pipeline
  regressions passed in the Explore GDAL environment.
- Product-scoped WAC and NAC AOI acquisition passed with CRS, transform,
  512×512 shape, bands, and NoData metadata verified against written files.
- Mixed WAC/static acquisition passed with the canonical 63 static bands and
  uniform static output NoData.
- Representative modern and legacy WAC/static cubes match in grid, band order,
  masks, and valid pixel values. The intentional static difference is modern
  `float64` output for safely combining mixed source dtypes versus legacy
  first-band-selected `float32`.
- Repeated acquisition produced identical record ordering, metadata, and file
  SHA-256 values, while declared source indexes remained unchanged.

These checks establish the cube acquisition boundary. They do not validate the
legacy chip merge/reprojection code; that work belongs to Phases C0–C8 below.

### Existing chip code is migration input, not the target API

The next agent should inventory `model/chip_making/chip_utils.py`, the legacy
`notebooks/toy_model/chip_example.ipynb`, and current task wrappers such as
`scripts/python/instance_seg/create_instance_wac_static_chips.py`. They contain
useful merge, reprojection, label-copying, and dataset-layout behavior, but also
contain the coupling this plan is intended to remove:

- construction of the deprecated `Pipeline`;
- one hard-coded dynamic tile database plus implicit static behavior;
- WAC/static-specific parameter names and dictionaries;
- filename regexes used to recover LTM zone and tile identity;
- training-GeoDataFrame-driven AOIs; and
- broad tuples/status strings instead of typed sample, grid, and result
  records.

Preserve observable output behavior only after it is written into the C0
contract and covered by a test. Do not preserve an internal legacy technique
merely because it is already present. Conversely, do not change the stable
tiling backend to accommodate old chip assumptions unless a reproducible
tiling defect is demonstrated and added to the tiling regression suite.

## Phase C0 — Define the target-chip contract `[Complete]`

- `[Complete]` **C0.1** Define which explicit AOI requests and TIFFs qualify as
  target-grid inputs and require complete metadata before preflight can pass:
  CRS, affine transform, rectangular bounds, width, height, and stable sample
  identity.
- `[Complete]` **C0.2** Define how sample IDs and optional source product IDs are
  supplied explicitly or extracted from a reference filename without coupling
  the workflow to one sensor's full filename.
- `[Complete]` **C0.3** Define the final chip directory layout, filename suffixes,
  band metadata, image-label pairing conventions, and reproducible split
  manifest in accordance with `docs/dataset_contribution.md`.
- `[Complete]` **C0.4** Define acceptance examples for WAC-only, NAC-only,
  WAC-plus-static, and NAC-plus-static target-grid-driven chip creation,
  including at least one request that crosses an LTM tile boundary. Keep the
  concrete filenames, AOIs, selectors, grids, and expected results in
  `docs/chip_creation_acceptance_examples.txt` for future local and HPC tests.
- `[Complete]` **C0.5** Define acquisition groups and their zoom policy. Sources
  within one group share one `TileConfig` zoom. Default every acquisition group
  to zoom 5, except the built-in NAC acquisition default of zoom 11; allow an
  explicit positive zoom override for any user modality. Treat each group as a
  named shared acquisition grid rather than a modality: output modalities
  identify both their acquisition group and source so a source such as static
  can appear unambiguously in more than one group.
- `[Complete]` **C0.6** Define clear rejection behavior for target extents
  outside the currently supported numbered LTM path, including LPN/LPS polar
  coverage.
- `[Complete]` **C0.7** Define label preflight as a prerequisite for acquisition
  and final-chip writing. A sample must resolve exactly one label whose stable
  identity, file type, archive contract, spatial shape, and available
  geospatial metadata agree with the sample's authoritative target grid.
  Define a typed label-mismatch error that fails only the current sample; a
  mismatched sample writes neither a final chip nor a copied output label.
- `[Complete]` **C0.8** Define explicit dataset-creation split policies. Support
  caller-assigned membership plus deterministic percentage, mixed fixed-count
  and percentage, and fixed-count policies. The default tries to assign 100
  test samples first, then assigns 90 percent of the remaining samples to train
  and 10 percent to validation. All policies use a dataset-creation seed and a
  leakage-prevention group key; fixed targets are best effort and report
  warnings rather than failing the batch when they cannot be met.

### Target-chip contract established by C0

#### C0.1 — Requests and authoritative target grids

- One `ChipRequest` represents one intended final chip. Numbered LTM zones,
  zoom grids, and tiles are intermediate acquisition details and never replace
  or partition the requested final grid.
- A request has a stable sample ID, a target grid, a label association, a
  split-group key, optional caller-assigned split, and optional product
  selectors. A target grid must contain a CRS WKT, affine transform,
  rectangular bounds, positive integer width and height, and its geographic
  query AOI before the request may pass preflight; none of these fields is
  optional at that boundary.
- A reference TIFF is a convenience for constructing a request. It qualifies
  only when it is an existing `.tif` or `.tiff` readable by GDAL, has at least
  one band, positive dimensions, a nonempty CRS, a finite and invertible affine
  transform, and finite nondegenerate bounds. Its exact CRS, transform, width,
  and height are authoritative; its pixel values are not implicitly included
  as an output modality.
- An explicit AOI request qualifies only when its normalized target-grid record
  supplies a lunar-compatible CRS, finite nondegenerate rectangular bounds, a
  finite invertible affine transform, and positive integer raster width and
  height. A convenience constructor may derive a north-up affine transform
  from supplied bounds and dimensions, but that transform must be materialized
  and validated before preflight. Thus the request, rather than an LTM tile or
  model default, supplies output width and height, and the resulting raster
  bounds exactly match the input AOI.
- A bare vector geometry is not a complete raster request because it does not
  define pixel dimensions or alignment. Callers iterating over a GeoDataFrame
  construct plain `ChipRequest` objects outside the backend. The initial
  contract accepts rectangular target extents; it must not silently replace an
  arbitrary nonrectangular study polygon with its envelope.
- Request iterables are materialized and processed deterministically. A
  directory convenience discovers `.tif` and `.tiff` references in sorted
  order, and callers may instead pass an explicitly globbed path iterable.

#### C0.2 — Sample identity, product selectors, and source selection

- An explicit nonempty sample ID is authoritative when supplied. A
  filename-derived sample ID preserves the complete normalized stem, including
  AOI offsets such as `_r12750_c1500`, and preserves its original case for
  output naming. A case-folded comparison key is used for uniqueness and label
  pairing.
- Filename-derived sample IDs strip only documented terminal role suffixes such as
  `_input_nac_chip`, `_input_wac_chip`, `_input_wac_static_chip`, `_label`,
  `_mask`, `_mask_orig`, `_img`, and `_chip`. Role words in the middle of a
  stem remain part of the ID. Empty, unsafe, or case-insensitively duplicate
  IDs fail preflight.
- For built-in WAC and NAC sources, the product ID is the filename component
  before the first underscore. This selector is intentionally distinct from
  the full sample ID because one product may provide multiple AOIs at different
  row/column offsets. An explicit per-request selector keyed by acquisition
  group and source may override the derived selector and is retained in
  diagnostics.
- Multiple requests may therefore share one WAC or NAC product selector. They
  remain distinct requests, label matches, output files, diagnostics, and
  manifest rows as long as their full normalized sample IDs differ; duplicate
  detection must never reject them merely because their product IDs match.
- Other modalities default to `all_intersecting`: the declared source index
  selects every raster intersecting the request AOI without product filtering.
  This does not scan or include unrelated rasters outside the AOI. A new
  modality that needs product scoping must declare `product_id` and provide an
  explicit source-specific selector rule rather than inheriting WAC/NAC
  filename semantics.
- Every `product_id` source requires exactly one selector and every
  `all_intersecting` source rejects one. The effective selector mapping is
  retained in the sample diagnostics.

#### C0.3 — Dataset layout, names, metadata, and pairing

```text
dataset_root/
  dataset_manifest.json
  diagnostics/
  train/
    chips/
    labels/
  val/
    chips/
    labels/
  test/
    chips/
    labels/
```

- Successful chips use `<sample-id><terminal-input-suffix>.tif`; the default
  suffix is derived from configured output modality aliases, for example
  `_input_wac_chip`, `_input_nac_chip`, or `_input_wac_static_chip`. Labels use
  `<sample-id>_label.npy` or `<sample-id>_label.npz`. Both names normalize to
  the same case-insensitive sample key used by the shared dataset loaders.
- The chip is an LZW-compressed GeoTIFF with the target grid's exact CRS,
  transform, width, height, and bounds; a configured dataset-wide NoData value;
  a deterministic dtype; and an ordered band description and `Name` item for
  every band. Output band names must be explicit and unique. Cube band names
  may be retained when unique, while collisions require configured aliases.
- Semantic `.npy` files and instance `.npz` archives are copied without
  rewriting their arrays. Only a successfully validated and written pair may
  appear in a split directory. Bounded staging and cleanup prevent an orphan
  final chip or label when publication fails.
- `dataset_manifest.json` is versioned, deterministically ordered by sample
  identity, and records configuration identity, target grid/AOI, split policy,
  split group and assignment, label source, selectors, output paths, final
  status, and diagnostic location. Failed requests remain in the manifest with
  no newly published output paths. Per-sample diagnostics live outside any
  intermediate directory that retention policy may remove.

#### C0.4 — Acceptance examples

Concrete seed values for these scenarios are maintained in
`docs/chip_creation_acceptance_examples.txt`. They are repository reference
data for constructing local fixtures and locating representative Explore data;
the file records whether each path is illustrative or verified rather than
implying that external rasters and labels are checked into this repository.

| Scenario | Acquisition group | Required final behavior |
|---|---|---|
| WAC only | WAC at zoom 5 | One seven-band chip on the request grid with explicitly configured VIS-then-UV order where legacy compatibility is requested. |
| NAC only | NAC at zoom 11 for the known 1 m example | One configured NAC-band chip on the same request/label grid, with no WAC alias or implicit static source. |
| WAC plus static | WAC and canonical static at zoom 5 | One 70-band chip in configured modality order: seven WAC bands followed by 63 canonical static bands. |
| NAC plus static | NAC and canonical static at zoom 11 | One 64-band chip in configured modality order for the one-band NAC acceptance source and 63 canonical static bands. |

- Each positive example resolves and validates its label before acquisition,
  may span more than one LTM tile, and produces exactly one final chip whose
  shape and geospatial grid equal the request and whose spatial shape equals
  the label mask.
- Zone-boundary and antimeridian examples acquire and composite multiple LTM
  groups without changing the final request grid. An antimeridian request uses
  two non-wrapping public tiling queries and deduplicates their structured
  records.
- Negative examples include missing and malformed labels, identity or shape
  mismatches, missing required source coverage, ambiguous longitude spans, and
  unsupported polar coverage. Each produces a structured per-sample failure,
  no new final chip or output label, and does not stop later batch samples.
- Repeating the same requests and split configuration yields identical
  assignments and manifest ordering regardless of request iteration order.

#### C0.5 — Acquisition groups and zoom policy

- An acquisition group is a unique, filesystem-safe name paired with one
  composed `TileConfig`. It represents sources acquired together on one LTM
  zoom grid, not a data modality, product, split, zone, or tile.
- All sources in a group share its resolved `TileConfig.zoom_level`. The
  general default is zoom 5 and the built-in 1 m NAC default is zoom 11. Static
  adopts the zoom of the group in which it appears. Users may explicitly
  override any group with a positive zoom appropriate to their source
  resolution. A group mixing source families with conflicting defaults must
  declare its zoom explicitly.
- An output modality reference identifies an acquisition group, one source in
  that group, a unique output alias, an ordered band selection, and a
  chip-stage continuous or categorical resampling policy. This qualification
  makes `wac_grid/static` distinct from `nac_grid/static`.
- Each request replaces a group's configured output directory with the flat,
  isolated
  `<intermediate-root>/<sample-id>/<acquisition-group>/` directory. Cube files
  remain flat within it and are orchestrated only through structured records.
- A final chip may combine modalities from different groups because each is
  independently reprojected to the one authoritative target grid. No source
  mosaics combine records from different zones or zooms before reprojection.

#### C0.6 — Geographic coverage and rejection

- The supported production path is the 90 numbered LTM zones from 82 degrees
  south through 82 degrees north. A densified target footprint extending
  beyond that range fails preflight with an `unsupported_polar_coverage`
  status before tiling. LPN/LPS metadata is not routed through LTM code.
- The 80-to-82-degree LTM/polar overlap remains valid numbered-LTM coverage,
  but it requires a focused regression proving that public AOI acquisition
  excludes the unsupported polar definitions. A demonstrated failure is fixed
  narrowly in the tiler and covered there before chip code relies on it.
- A target crossing an ordinary LTM tile or zone boundary is supported. A
  longitude envelope crossing the antimeridian is retained as one logical AOI
  for the final grid but split into `[west, 180]` and `[-180, east]` acquisition
  queries. A footprint with a longitude span of 180 degrees or more is
  ambiguous for this contract and fails before acquisition.
- Failure to transform every densified perimeter point to repository
  IAU:30100, non-finite results, or failure of the transformed footprint to
  cover the target perimeter is invalid target-grid metadata rather than
  missing source coverage.

#### C0.7 — Label preflight and failure isolation

- Split assignment and label preflight occur after request/grid validation and
  before any tiling call. The request resolves either an explicit label path or
  exactly one full normalized-sample-ID match in its configured label source.
  For WAC and NAC, the row/column-qualified sample ID—not the shorter product
  ID—is the label key. Other dynamic modalities likewise use their normalized
  filename stem. Missing, duplicate, unsupported, or misidentified labels
  raise a typed `LabelMismatchError` for the current sample.
- A semantic label is a two-dimensional integer `.npy` mask with spatial shape
  `(target_height, target_width)`. An instance label is an `.npz` archive with
  a two-dimensional integer `mask`, `bboxes` shaped `(N, 4)`, and scalar
  `num_craters == N`; positive mask instance IDs follow the documented `1..N`
  convention. Boxes use the repository's existing COCO
  `(x, y, width, height)` representation and must be finite, positive-area,
  and within the target pixel extent.
- Label grid metadata supplied by a manifest or sidecar is compared with the
  target CRS, affine transform, dimensions, and footprint.
  Plain `.npy` and `.npz` labels do not independently prove a CRS; for them the
  structured request/manifest association, normalized identity, and exact
  spatial shape are the verifiable boundary and this limitation is recorded.
- The single-request operation raises the typed preflight error. Batch
  orchestration catches it, emits a failed result and diagnostic, and continues
  with the next request. It does not invoke tiling and does not create, replace,
  or copy a final dataset chip or label for that failed request.
- Immediately before publication, the assembled in-memory chip is checked
  again against the already validated target grid and label shape. A mismatch
  is handled with the same per-sample failure and no final pair publication.

#### C0.8 — Deterministic dataset splitting

- Split configuration is a validated union of `SimpleSplitConfig`,
  `MixedPercentageNumberSplitConfig`, and `NumberSplitConfig`. A caller may
  still preassign `train`, `val`, or `test` on a request; consistent explicit
  assignments are honored and removed from the pool before configured
  assignment.
- `SimpleSplitConfig` supplies train/validation/test percentages summing to one
  and assigns every remaining group by stable seeded hash thresholds.
- `MixedPercentageNumberSplitConfig` supplies best-effort fixed sample-count
  targets for one or more splits, an explicit priority order for evaluating
  those targets, and percentage ratios summing to one for all remaining
  groups. The repository default first tries to assign 100 samples to `test`,
  then assigns 90 percent of the remaining data to `train` and 10 percent to
  `val`.
- `NumberSplitConfig` supplies best-effort sample-count targets and a priority
  ordering that determines which split claims groups first, for example
  `train`, then `test`, then `val`. When targets cover fewer samples than are
  available, remaining requests are recorded as unassigned by policy and are
  not acquired or published unless an explicit remainder split is configured.
- All requests sharing a group key receive the same split. Explicit assignments
  that place one group in multiple splits are invalid. Grouping should occur at
  the product, study-site, campaign, parent-AOI, or another scientifically
  justified level rather than defaulting silently to individual chip IDs.
- Every automatic policy uses a versioned stable digest of the configuration
  seed and normalized group key. Percentage assignment uses the digest value
  with cumulative ratio thresholds. Number assignment uses a separate
  versioned digest namespace to rank remaining groups deterministically, then
  assigns whole groups in configured split-priority order as close as practical
  to each requested sample count. Python `hash()` and discovery-order shuffling
  are not part of the contract.
- Leakage-prevention groups are atomic. A fixed target may therefore be
  underfilled or exceeded when that is the closest deterministic assignment of
  whole groups. Failure to meet a number target emits a warning containing the
  split, requested count, realized count, and reason; it does not fail other
  samples or the batch.
- Assignment happens before label preflight so every structurally valid request
  retains an auditable planned split even if its label or later processing
  fails. Percentage-only assignments remain stable when unrelated groups are
  added. A fixed-number policy is reproducible for the same input inventory;
  preserving assignments as that inventory grows requires supplying the prior
  manifest as locked assignments before filling any remaining target.
- The manifest records the concrete split-config type, percentages, number
  targets, priority, remainder behavior, seed, algorithm/version, normalized
  group key, assigned or unassigned status, warnings, and final processing
  status. A model-training seed remains independent and may still affect
  initialization, augmentation, shuffling, and optimization, but never dataset
  membership.

## Phase C1 — Introduce chip configuration objects `[Complete]`

- `[Complete]` **C1.1** Add a `ChipConfig` dataclass containing the output dataset
  root, one or more named composed `TileConfig` acquisition groups, and ordered
  output-modality references qualified by acquisition-group name and source
  name. Keep per-sample requests separate so callers may supply an iterable of
  explicit AOIs, reference TIFFs, or requests derived from a GeoDataFrame
  without making the chip backend depend on GeoPandas. Resolve an omitted group
  zoom to 5, except for the built-in NAC group default of 11, and allow an
  explicit positive override for every group.
- `[Complete]` **C1.2** Add final output naming, common NoData, label source,
  optional sample limits, intermediate-cube retention settings, and a
  chip-stage resampling policy distinct from tiling's fixed bilinear policy.
- `[Complete]` **C1.3** Add small structured `ChipRequest`, target-grid,
  geographic-AOI, reference-sample, and result records instead of passing
  unrelated tuples through helper functions. Include the resolved label,
  split-group key, optional caller-assigned split, preflight status, and typed
  label-validation diagnostics in these contracts.
- `[Complete]` **C1.4** Add a split-config protocol and validated
  `SimpleSplitConfig`, `MixedPercentageNumberSplitConfig`, and
  `NumberSplitConfig` dataclasses. Support explicit assignments, percentage
  ratios, best-effort number targets, fixed-target priority, optional remainder
  handling, a stable group-key policy, versioned deterministic hashing, prior
  manifest locks, warnings for unrealized number targets, and the default of
  100 test samples followed by a 90/10 train/validation percentage split.
- `[Complete]` **C1.5** Add a dictionary-to-config constructor consistent with the
  tiling configuration interface.
- `[Complete]` **C1.6** Add focused config tests, including invalid composed
  tiling configurations, duplicate or missing output modalities, invalid split
  ratios, counts, priorities, or names, missing split-group keys, conflicting
  assignments for one split group, default and overridden zooms, and every
  split-config variant.

### C1 implementation evidence

- `model/chip_config.py` contains the immutable acquisition-group,
  output-modality, chip, and three split-configuration contracts plus strict
  dictionary constructors. Omitted zoom resolves to 5 generally and 11 for a
  built-in NAC-plus-static group; a mixed NAC/default-5 group requires an
  explicit override.
- `model/chip_types.py` contains complete target-grid, geographic-AOI,
  reference-sample, source-selector, chip-request, preflight, result,
  label-diagnostic, and typed label-mismatch contracts. Cross-request
  validation permits multiple row/column-qualified samples to share a WAC/NAC
  product selector while rejecting duplicate full sample IDs and conflicting
  explicit assignments within one split group.
- The focused C1 tests pass locally. The modern C1/tiling regression command
  also passed all 67 tests in the project's GDAL-enabled HPC container on
  2026-09-03, with no failures or skips reported.

## Phase C2 — Build requests, AOIs, and target grids `[Complete]`

- `[Complete]` **C2.1** Materialize explicit `ChipRequest` iterables
  deterministically. Also provide reference-TIFF directory and path-iterable
  conveniences, including caller-side glob results, without loading a training
  GeoDataFrame in the backend.
- `[Complete]` **C2.2** Validate explicit AOI requests and reference TIFFs. An
  explicit AOI supplies a CRS, rectangular bounds, width, and height and may
  supply an exact affine transform; otherwise derive its north-up transform so
  the raster bounds exactly equal the AOI. A reference TIFF supplies its exact
  CRS, affine transform, bounds, width, height, and band metadata. Reject
  missing, non-finite, non-invertible, empty, or inconsistent grid metadata
  before creating intermediate cubes.
- `[Complete]` **C2.3** Transform the densified target-grid perimeter to lunar geographic
  `IAU:30100` coordinates using the repository WKT loader and express them in
  the tiling API's upper-left and lower-right AOI convention. Follow the
  densified-perimeter behavior already exercised by
  `scripts/python/all_tasks/extract_datacube_aoi.py`; transforming only two
  corners is not sufficient for every projected raster. Retain one logical
  geographic AOI for diagnostics, but represent an antimeridian-crossing AOI
  as two non-wrapping tiling-query parts split at +180/-180 degrees. Reject
  ambiguous footprints spanning 180 degrees or more.
- `[Complete]` **C2.4** Retain the request's target grid as the exact destination
  grid for final chip creation. Preserve a reference TIFF's original grid; for
  an explicit AOI request use its validated or derived grid. Do not substitute
  LTM tile boundaries for the requested final extent.
- `[Complete]` **C2.5** Verify that each geographic AOI falls within supported
  numbered LTM coverage before acquisition and return a specific unsupported
  status for polar-only requests.
- `[Complete]` **C2.6** Assign every request with a valid identity, target grid,
  and split-group key from an explicit or prior-manifest assignment, or apply
  the configured simple-percentage, mixed-number/percentage, or number-only
  policy. Use versioned stable digest namespaces for ratio thresholds and
  deterministic number-target ranking. Apply whole groups in configured target
  priority, record warnings for every unrealized number target, and mark any
  number-policy remainder without a configured destination as unassigned and
  ineligible for acquisition. Retain every planned assignment in diagnostics
  even if label preflight, acquisition, or writing later fails.
- `[Complete]` **C2.7** Resolve exactly one label from the configured label source
  before acquiring any cubes. Validate its normalized sample identity and
  spatial array shape against the authoritative target-grid width and height;
  preserve semantic `.npy` arrays and require instance `.npz` archives to
  contain the documented `mask`, `bboxes`, and `num_craters` fields with
  mutually consistent dimensions and counts. When label geospatial metadata
  is available from a structured association or sidecar, also compare its CRS,
  transform, and footprint to the target grid. For non-georeferenced `.npy` or
  `.npz` labels, treat the structured sample/manifest association plus exact
  identity and shape as the verifiable boundary, without claiming an
  independent CRS match.
- `[Complete]` **C2.8** Add tests with projected lunar raster fixtures to verify
  geographic AOIs, round-trip coverage of the target extent, axis order,
  zone-boundary behavior, antimeridian query splitting, and polar rejection.
  Add label-preflight tests for missing, duplicate, misidentified, malformed,
  shape-mismatched, and geospatially mismatched labels; assert that rejected
  samples do not invoke tiling or create final dataset files. Add split tests
  covering all three config variants, default behavior, fixed seeds, changed
  seeds, input reordering, unrelated additions, prior-manifest locks, atomic
  groups, target priorities, underfilled and exceeded best-effort counts,
  unassigned number-policy remainders, warnings, and co-location of every
  shared group key. Include two row/column-qualified WAC or NAC requests from
  the same product and verify that each resolves its own full-ID label and
  produces its own result while both use the shared product selector.

### C2 implementation evidence

- `model/chip_requests.py` implements deterministic request and reference-TIFF
  discovery, exact target grids, north-up affine derivation, densified GDAL
  transformation with round-trip checks, antimeridian query splitting, and
  typed ambiguous-longitude and unsupported-polar failures.
- `model/chip_splits.py` implements stable BLAKE2b percentage thresholds,
  prioritized best-effort fixed sample counts, atomic leakage groups,
  caller/prior-manifest locks, structured and emitted target warnings, and
  explicit unassigned remainder handling.
- `model/chip_labels.py` validates full offset-qualified label identity,
  semantic arrays, instance archives using the existing COCO
  `(x, y, width, height)` box convention, and optional structured or sidecar
  grid metadata. `model/chip_preflight.py` isolates invalid grids, unsupported
  geography, unassigned splits, and label mismatches without importing or
  invoking the tiler and without creating output directories.
- The expanded modern suite passes 97 tests locally, with 17 tests skipped
  because the lightweight environment lacks GDAL and NumPy. On 2026-09-03, all
  30 focused C2 request, split, and label tests passed in the project's HPC
  container with GDAL and NumPy available. Combined with the previously passed
  67-test C1/tiling suite, those runs cover the complete 97-test C2 inventory.

## Phase C3 — Acquire cubes through the tiling API `[Complete]`

- `[Complete]` **C3.1** Derive per-source query selectors, including product IDs,
  only after the request and its label pass preflight. Derive them from each
  request and `ChipConfig`. Built-in WAC and NAC sources derive the product ID
  from the first filename component before an underscore while preserving the
  full row/column-qualified stem as the sample ID. Other modalities default to
  `all_intersecting`, as do static/context sources; a new product-scoped
  modality must declare its own selector rule.
- `[Complete]` **C3.2** Call the public AOI tiling function with the composed
  `TileConfig`; do not instantiate or reach into legacy pipeline internals.
  Derive a unique sample/acquisition-group output directory and create a
  sample-specific config, for example with `dataclasses.replace`, before each
  call. Use a flat
  `<intermediate-root>/<sample-id>/<acquisition-group>/` directory for the
  group's cube files; do not add product, zone, or tile subdirectories. For an
  antimeridian-crossing logical AOI, call the public function once per query
  part, then combine and deduplicate records by their structured identity.
- `[Complete]` **C3.3** Group returned `TileCubeRecord` objects by source, zone,
  zoom, and tile coordinates without parsing filenames. Keep records from
  different acquisition groups separate when their zooms differ.
- `[Complete]` **C3.4** Define explicit behavior for absent required modalities,
  optional modalities, multiple LTM zones, incomplete tile coverage, and
  `completed_records` attached to a source error. `run_aoi()` continues to stop
  at the first source failure, but add a focused tiling regression and narrowly
  update its exception propagation so `completed_records` includes records
  from earlier successful tiles as well as sources completed within the
  failing tile. Tiles after the failure remain unattempted.
- `[Complete]` **C3.5** Return and retain structured acquisition diagnostics,
  including selectors, AOI, acquisition group, zoom, records, and any partial
  files, so a batch failure can be reproduced without parsing log text.
  Inventory the current sample/acquisition-group directory after an exception
  to account for files whose writes failed before a `TileCubeRecord` could be
  produced. Use this inventory only for diagnostics and bounded cleanup, never
  to reconstruct structured tile identity by parsing filenames.
- `[Complete]` **C3.6** Add orchestration tests using synthetic structured tiling
  results, including missing required WAC, optional sparse NAC, colliding tile
  addresses in separate sample directories, and multiple acquisition zooms.

### C3 implementation evidence

- `model/chip_acquisition.py` derives effective product selectors only for
  requests eligible after label preflight, uses the public
  `create_tiles_for_aoi()` API with a replaced sample/group output directory,
  splits antimeridian queries, and deduplicates returned records by structured
  acquisition-group, source, zone, zoom, and tile identity.
- Acquisition results retain the logical AOI, query parts, effective selectors,
  records, structured record groups, partial-file inventory, and typed
  diagnostics. Required-source failures stop later groups for that sample;
  absent or tile-sparse optional sources produce structured warnings without
  failing acquisition.
- `ConfiguredTiler.run_aoi()` now augments a `TileSourceError` with records from
  earlier successful tiles while retaining records completed within the
  failing tile. It still stops at the first failed source and leaves later
  tiles unattempted.
- The 110-test modern suite passes locally with 18 GDAL/NumPy-dependent tests
  skipped. The twelve synthetic C3 acquisition tests pass locally. On
  2026-09-03, all 14 focused acquisition and configured-tiler tests passed in
  the project's fully enabled HPC container, including the new `run_aoi()`
  exception-propagation regression.

## Phase C4 — Generalize merge and reprojection `[Complete]`

- `[Complete]` **C4.1** Replace fixed `wac_files` and `static_files` parameters
  with ordered source-to-cube mappings.
- `[Complete]` **C4.2** Merge adjacent LTM cubes independently for each configured
  source modality, zone, and zoom. Never place cubes from different LTM CRSs or
  zoom grids into one source-space mosaic merely because their filenames share
  tile numbers.
- `[Complete]` **C4.3** Normalize or preserve source NoData according to each
  `TileSourceConfig`, `TileCubeRecord.nodata_values`, and reopened raster
  metadata before interpolation. Do not infer NoData from pixel magnitude.
- `[Complete]` **C4.4** Reproject each merged modality onto the request's
  exact CRS, transform, width, and height. Keep this chip-stage resampling
  choice separate from the tiler's fixed bilinear warp: use bilinear for
  continuous image/context data and nearest-neighbor for categorical data.
- `[Complete]` **C4.5** Verify spatial shape, transform, CRS, and extent against the
  target grid before assembly.
- `[Complete]` **C4.6** Add tests for continuous and categorical resampling,
  modality-specific NoData, and multi-tile target extents.
- `[Complete]` **C4.7** Add a request crossing an LTM zone boundary and verify
  that independently reprojected zone groups composite correctly on the one
  authoritative target grid.

### C4 implementation evidence

- `model/chip_reprojection.py` maps configured output modalities to structured
  acquisition-group/source records in configuration order and partitions each
  source by LTM zone and zoom before any mosaic is built.
- Every reopened cube is checked against its `TileCubeRecord` band, CRS, grid,
  and NoData metadata and against the configured source NoData policy. Declared
  sentinels, nonfinite values, and GDAL validity masks are normalized before
  interpolation; pixel-magnitude thresholds are not used.
- Each zone mosaic is warped independently into an existing in-memory dataset
  with the target grid's exact CRS, affine, width, and height. Continuous and
  categorical modalities use their configured bilinear or nearest-neighbor
  chip-stage resampling, and zone results are composited deterministically on
  that one grid.
- The 120-test modern suite passes locally with 26 GDAL/NumPy-dependent tests
  skipped. Both dependency-free C4 mapping tests pass. On 2026-09-03, all ten
  focused C4 tests passed in the project's fully enabled HPC container,
  including the multi-zone lunar reprojection regression.

## Phase C5 — Assemble and write model-ready chips `[Complete]`

- `[Complete]` **C5.1** Select and order bands within each modality from explicit
  configuration or cube metadata.
- `[Complete]` **C5.2** Concatenate modalities in `ChipConfig` order and construct
  the final ordered band-name list.
- `[Complete]` **C5.3** Write a compressed GeoTIFF with the target grid, common
  output NoData, deterministic dtype, and per-band descriptions. Convert every
  input sentinel to the chosen common value before writing because GeoTIFF's
  persisted `TIFFTAG_GDAL_NODATA` is dataset-wide. Make any downcast from the
  modern static cube's safe `float64` dtype explicit and range-checked. Write a
  final dataset chip only after label preflight and split assignment have
  succeeded.
- `[Complete]` **C5.4** Use dataset-compatible sample IDs and terminal filename
  suffixes such as `_input_wac_static_chip.tif`.
- `[Complete]` **C5.5** Reopen each written chip and verify channel count, shape,
  CRS, transform, dataset-wide NoData, band descriptions, per-band masks, and
  finite-data coverage.
- `[Complete]` **C5.6** Add regression tests for band order, including the legacy
  WAC VIS-then-UV ordering where required.

### C5 implementation evidence

- `model/chip_assembly.py` selects bands by configured names or one-based
  indices, concatenates modalities in configuration order, and preserves cube
  metadata order when no narrower selection is configured. Automatic names
  that collide across modalities are qualified with their unique modality
  aliases; explicit output names remain authoritative and must be unique.
- Missing optional modalities become stable all-NoData placeholder channels
  only when configuration or cube metadata defines their band contract. A
  missing required modality or an unknowable optional channel contract is a
  typed per-sample failure.
- Final arrays convert every invalid pixel to the configured common NoData and
  use an explicit range-checked output cast. Integer output additionally
  rejects fractional values, and all dtypes reject a NoData value that cannot
  be represented exactly or valid data that would collide with it.
- Compressed, tiled GeoTIFFs are written to deterministic per-sample staging
  paths under the intermediate root only for requests whose label preflight
  passed and whose split was assigned. C6 remains responsible for publishing
  the validated chip and preserved label together into `train/val/test`.
- Every staged chip is reopened before it is accepted. Validation covers band
  count, dimensions, semantic CRS, affine and extent, dtype, LZW compression,
  dataset-wide NoData, ordered descriptions and `Name` metadata, per-band
  validity masks, finite valid pixels, and nonempty coverage for required
  bands. A failed write or validation removes its temporary GeoTIFF.
- The 130-test modern suite passes locally with 34 GDAL/NumPy-dependent tests
  skipped. Both dependency-free C5 path tests pass. On 2026-09-03, all ten
  focused C5 tests passed in the project's fully enabled HPC container,
  including GeoTIFF write/reopen validation and legacy WAC VIS-then-UV order.

## Phase C6 — Publish validated pairs and dataset splits `[In Progress]`

- `[Complete]` **C6.1** Preserve each preflight-valid semantic `.npy` label or
  instance `.npz` archive without rewriting its contents or assuming the
  legacy `data` key.
- `[Complete]` **C6.2** Publish each validated image-label pair into its assigned
  `train/val/test/{chips,labels}` directories. Use bounded staging and cleanup
  so a per-sample publication error does not leave an orphan final chip or
  label, and never publish either file for a label-preflight failure.
- `[Complete]` **C6.3** Write a deterministic split manifest containing at least
  the sample ID, split-group key, assigned or unassigned split status, concrete
  split-config type, percentages, fixed targets and their priority, requested
  and realized counts, warnings, stable hash algorithm/version and
  dataset-creation seed, AOI or target-grid identity, source label path, output
  paths, and processing status. Keep failed sample assignments in the manifest
  for auditability.
- `[Complete]` **C6.4** Validate the produced directory membership and manifest:
  every successful sample appears exactly once, samples sharing a group key do
  not cross splits, and no failed or mismatched sample has a final dataset
  artifact.
- `[In Progress]` **C6.5** Confirm that the shared semantic and instance datasets can
  discover and load representative generated pairs from every populated split.

### C6 implementation evidence

- `model/chip_publication.py` revalidates the source label and staged chip
  immediately before publication, records their validated SHA-256 identities,
  and copies them into bounded per-destination staging files. It publishes both
  through no-overwrite hard links and rolls back the first artifact if the
  second publication fails. Semantic and instance labels retain byte-identical
  `.npy` or `.npz` contents and receive canonical `<sample-id>_label` names.
- Successful pairs are placed only in their assigned
  `train/val/test/{chips,labels}` directories. Existing outputs are never
  replaced implicitly, and a non-success result is invalid if it exposes final
  output paths.
- `dataset_manifest.json` has a versioned, deterministic JSON schema and no
  volatile timestamp. It records a SHA-256 configuration identity, complete
  chip and split configuration, stable split hash contract and seed, requested
  and realized split counts, fixed-target warnings, target grid and geographic
  AOI, assignment source, effective product selectors, source label and final
  paths, preflight diagnostics, processing status, and diagnostic path. Rows
  are case-insensitively ordered by full sample ID, including failed and
  unassigned samples, and the output can be reused as a prior split manifest.
- Publication validation requires exact on-disk membership: every successful
  sample has exactly one chip and label in one split, leakage groups do not
  cross splits, no extra split artifacts exist, and failed or mismatched
  samples have no final artifact. An optional manifest is parsed and compared
  with the complete expected document.
- `ChipResult` now retains the effective acquisition selectors separately from
  caller-provided request overrides so the manifest can audit derived WAC/NAC
  product IDs as well as explicit selectors.
- The 139-test modern suite passes locally with 37 GDAL/NumPy-dependent tests
  skipped. All six dependency-free C6 manifest tests pass; three
  raster-and-loader-backed C6 tests await the fully enabled HPC environment
  before C6.5 is completed.

## Phase C7 — Add sequential batch orchestration `[Planned]`

- `[Planned]` **C7.1** Add a single-request `create_chip` operation returning a
  structured result with paths, assigned split, status, timing, cube records,
  and diagnostics. Let its label preflight raise a typed label-mismatch error
  before acquisition or final writing.
- `[Planned]` **C7.2** Add deterministic iterable-level `create_chips`
  orchestration plus a reference-directory convenience that processes TIFFs in
  sorted order. Catch typed per-sample label errors, record the failed result
  and planned split, and continue with the remaining samples.
- `[Planned]` **C7.3** Define clear skipped, partial, and failed statuses and
  ensure resources are closed on every path. Preserve tiling error context and
  account for all prior-tile and failing-tile `completed_records`, files
  discovered in the isolated acquisition-group directory, and query parts or
  later tiles that were not attempted after a failure. Persist diagnostics
  outside an intermediate directory when that directory may be cleaned up.
- `[Planned]` **C7.4** Add safe overwrite and intermediate-cube retention
  behavior without deleting output implicitly. Limit cleanup to the current
  sample's explicitly resolved intermediate directory.
- `[Planned]` **C7.5** Validate the serial workflow before introducing any
  optional multiprocessing adapter.
- `[Planned]` **C7.6** Prove that repeated serial runs with the same inputs,
  split configuration, and seed produce byte-identical manifests and identical
  directory membership regardless of discovery order for all three split
  configs. Prove that a different dataset-creation seed may change automatic
  membership without affecting caller- or prior-manifest-assigned splits, and
  that unmet number targets warn without stopping later samples.
- `[Planned]` **C7.7** If still needed after profiling, add multiprocessing using
  the same single-request operation and result contract.

## Phase C8 — Validate complete datasets on HPC `[Planned]`

- `[Planned]` **C8.1** Run a small WAC request set, including explicit AOIs and a
  reference-directory convenience, and inspect every generated image-label
  pair.
- `[Planned]` **C8.2** Run a NAC reference directory through the same API and
  record its product selectors, native NoData configuration, and acquisition
  zoom. Use zoom 11 for the known 1 m processed NAC example unless the selected
  dataset establishes another resolution contract.
- `[Planned]` **C8.3** Run WAC-plus-static creation with the intended full static
  band list and confirm final band order.
- `[Planned]` **C8.4** Compare representative outputs with the legacy chip
  workflow numerically and visually, documenting intentional differences.
- `[Planned]` **C8.5** Run the pre-training diagnostics from
  `docs/dataset_contribution.md` on the resulting dataset layout.
- `[Planned]` **C8.6** Repeat dataset creation with the same split configuration
  and confirm identical manifest assignments, group isolation, and
  `train/val/test` directory membership. Report requested versus realized
  ratios and counts, including warnings and any expected deviation caused by
  grouped assignment, insufficient samples, or failed samples.

## Phase C9 — Modernize the chip example notebook `[Planned]`

- `[Planned]` **C9.1** Create `notebooks/chip_example.ipynb` while retaining the
  legacy notebook until migration is complete.
- `[Planned]` **C9.2** Use the repository-root convention from
  `notebooks/instance_ibm_train.ipynb` and derive repository-owned paths from
  `repo_root`.
- `[Planned]` **C9.3** Put the explicit request or reference-directory inputs,
  label source, output root, product selectors, split mode, split ratios,
  fixed-count targets and priority, dataset-creation seed, split-group policy,
  and other true user inputs in one visible configuration section. Follow the
  tiling notebook with separate derived-variable and path-resolution sections
  for `TileConfig`, `ChipConfig`, indexes, run output, and validation.
- `[Planned]` **C9.4** Demonstrate AOI extraction from one reference TIFF before
  running the configured single-chip workflow.
- `[Planned]` **C9.5** Display structured tiling and chip results rather than
  reconstructing state through filename parsing.
- `[Planned]` **C9.6** Visualize selected output bands, the paired label, NoData
  coverage, and key grid metadata.
- `[Planned]` **C9.7** Execute the notebook top-to-bottom on the HPC system and
  clear stale outputs before committing the modern copy.

## Phase C10 — Complete the chip migration `[Planned]`

- `[Planned]` **C10.1** Update backend docstrings and user documentation with the
  final `ChipConfig` and `SplitConfig` schemas, label-preflight behavior,
  reproducible split manifest, explicit-AOI workflow, and reference-TIFF
  convenience.
- `[Planned]` **C10.2** Update repository scripts and callers to use the new chip
  API or an intentional compatibility wrapper. Completing this step also
  satisfies the chip-caller portion intentionally excluded from tiling T8.2.
- `[Planned]` **C10.3** Deprecate or remove the training-GeoDataFrame dependency
  after all callers migrate.
- `[Planned]` **C10.4** Run formatting, unit tests, dataset loading smoke tests,
  and the final HPC notebook validation.
- `[Planned]` **C10.5** Mark this plan complete and archive the legacy notebook
  workflow according to the repository's chosen deprecation policy. Confirm
  that the completed chip migration still consumes the stable, completed
  tiling contract.
