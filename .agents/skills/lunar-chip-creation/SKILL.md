---
name: lunar-chip-creation
description: Develop, diagnose, document, or validate LFM chip creation and its standard or polar notebooks, including AOI/native target grids, label conversion, band assembly, splits, multiprocessing, and publication. Use for downstream consumption of tiling results, not standalone tiling, crater annotation UI, or model training.
---

# LFM Lunar Chip Creation

Inspect current code before changing behavior. Paths below are relative to the
repository root. Public imports are `lfm.data_processing.chip`; historical
`lfm.model` imports and parent-of-repository test commands predate migration.

## Read according to the task

- Architecture: `.agents/chip_creation_backend_structure.md` and the relevant
  modules under `lfm/data_processing/chip/`.
- AOI, labels, or polar changes:
  `.agents/planning_docs/aoi_chip_creation_and_label_clipping_plan.md`, focusing
  on the relevant phase and its follow-ups. Distinguish planned behavior,
  implementation, unit-test evidence, and real-data acceptance.
- Label conversion: `.agents/crater_labeling_to_chip_creation_contract.md`;
  also read A6T in the AOI plan for TIFF-specific rules. Inspect the exporter in
  `lfm/data_processing/labeling/craters.py` when interpreting actual GPKGs.
- Legacy reference-chip, split, and publication behavior:
  `docs/chip_creation_modernization_plan.md`.
- Acquisition: `.agents/tiling_to_chip_creation_handoff.md` and
  `lfm/data_processing/tiling/tiling_results.py`. When changing tiling itself,
  also use the available `lunar-tiling` skill.
- Notebook work: `notebooks/chip_example.ipynb`,
  `notebooks/chip_polar_example.ipynb`, and
  `lfm/data_processing/chip/chip_notebook_utils.py`. Follow
  `notebooks/tiling_example.ipynb` for import/configuration style.

Older handoffs saying all polar chips are unsupported are historical. Check
current code, tests and accepted user decisions before applying old statements;
flag unresolved contract conflicts rather than silently choosing a policy.

## Keep stage boundaries

```text
Reference TIFF or AOI -> ChipRequest
  -> deterministic preflight/splits -> label validation and preparation plan
  -> per-chip worker: materialize labels -> acquire tiles
     -> mosaic/reproject/clip -> assemble/stage -> publish pair
  -> per-sample diagnostics and coordinator-owned dataset manifest
```

- `chip_types.py` and `chip_config.py` define contracts. `chip_requests.py`
  derives identities, query parts and grids. `chip_preflight.py` and
  `chip_splits.py` make non-writing batch-wide decisions.
- `chip_labels.py` validates final labels; `chip_label_planning.py` validates
  sources and plans conversion. `chip_label_materialization.py` and
  `chip_instance_labels.py` prepare semantic and instance artifacts.
- `chip_acquisition.py` consumes tiling records; `chip_reprojection.py` aligns
  imagery to the target; `chip_assembly.py` selects bands and stages TIFFs.
- `chip_publication.py` owns pair publication and manifest construction;
  `chip_creation.py` coordinates APIs, workers, progress and cleanup.

Use `create_chip`, `create_chips`, and `create_chips_from_reference_directory`
instead of notebook-local pipeline copies. Workers consume prepared requests;
do not independently replan splits per worker. Prepare labels before tiling.

## Grid and acquisition invariants

- AOIs use lunar IAU:30100 north, west, south, east order. Load repository
  definitions from `TMS/`, not terrestrial EPSG:4326.
- Final dynamic output inherits the original **pre-tiling** raster CRS,
  affine lattice and resolution. WAC VIS takes precedence among mixed dynamic
  modalities, with a warning. Conflicting candidate grids need an explicit
  reference choice, not arbitrary file order.
- Densify transformed AOI edges and round the native pixel window outward.
  Preserve requested AOI and realized footprint separately. Clip labels to the
  realized grid; never shrink the request to available imagery.
- Static-only uses a zero-anchored 100 m grid in the selected LTM or polar CRS,
  without requiring WAC coverage. Acquisition zoom is not output resolution.
  Use `default_chip_zoom`: WAC/static LTM 5 and polar 4; NAC LTM 11 and polar 10.
  Preserve explicit overrides.
- Polar support includes north/south AOIs, small antimeridian wraps and ±82°
  seams. Seam acquisition uses single-family parts with optional group-level
  `ltm_zoom_level`/`polar_zoom_level`; omitted overrides preserve the scalar
  TileConfig zoom. Composite by target pixel-center latitude: polar preferred
  at |latitude| >=82°, LTM below, with per-band valid-data fallback. Static-only
  seam grids follow the AOI-center rule. Pole-touching/containing targets,
  spans >=180°, and both-polar-region requests remain gated. Check requested
  and realized coverage; consult the plan for pending HPC/real-data acceptance.
- Acquisition groups include configuration/zoom, not only modality. Preserve
  source selectors and PID grouping. Reference sample IDs retain row/column
  offsets because one product can supply multiple chips/labels.
- Use structured `TileCubeRecord` metadata, not parsed filenames. Split wrapped
  queries and deduplicate records. Preserve completed records and diagnostics
  on later tile failure; partial acquisition is not automatic sample success.

## Label invariants

- Explicit AOI label paths are scientist-selected: no filename/PID/source-scene
  identity gate. Technical format, CRS, coverage and shape checks still apply.
  Legacy directory label resolution has separate matching rules.
- Treat GPKGs as finished full-scene annotations, usually in `craters`.
  Annotation provenance does not require its original raster path to exist.
- Auto-detected TIFF labels are semantic. Instance TIFFs require
  `LabelInput(kind="raster_instance")`. Require single-band integers and
  embedded georeferencing; derived raster labels must cover the realized grid.
  Unknown label pixels/NoData are not automatically background.
- Semantic output is NPY; sampling preserves integer classes using nearest
  mapping, never bilinear interpolation.
- Vector instances use clipped outlines, pixel-center inclusion, highest
  original ID winning overlap and ascending compact IDs. COCO boxes follow
  clipped outlines independently of visible pixels. Keep supported fully
  occluded instances with evidence; warn and omit subpixel outlines without
  pixel support. Read the contract for boundary ties, holes and NPZ heuristics.
- Instance TIFFs have no outlines: boxes follow retained raster support; do not
  invent fully occluded objects. Output NPZ contains `mask`, `bboxes`, and
  scalar `num_craters`. Keep source-to-output ID mappings in provenance.
- All-background labels are valid. Outside craters may be omitted; empty-mask
  success does not prove the intended AOI was selected. Keep inputs read-only.

## Imagery, publication and parallelism

- Canonical seven-band WAC order is VIS 0–4, UV 5–6, then configured static
  bands. Polar monochrome WAC is a separate single-band layout. Reuse band
  constants and existing escaped suffix matching for product-qualified names;
  reject ambiguous matches.
- Default common imagery NoData is **-32768**, configurable. Retain known
  uncovered bands and warn with per-band counts/percentages plus spatial-union
  statistics. Even wholly NoData output is permitted when records/schema exist.
  Empty acquisition, unknown schemas, unreadable sources and processing errors
  are not coverage-fill cases. Do not infer validity from pixel magnitude or
  apply imagery NoData policy to labels.
- Publish verified pairs with no-clobber and rollback protection. Invalid
  labels/failed samples must not leave published chips. Preserve exact-label
  bytes and never clean up shared inputs or indexes.
- Splits are deterministic and group-atomic; fixed-count shortfalls warn.
  `NoSplitConfig` writes directly to output-root `chips/` and `labels/`.
  Preserve preplanned assignments through worker execution/publication.
- Parallelism is across chips; each worker's stages are sequential. Keep GDAL
  handles worker-local, progress and final manifest writes coordinator-owned,
  and serial/spawn results equivalent. Prepare indexes before chip workers;
  workers must not race to rebuild shared indexes.

## Notebook and validation practice

Keep concise per-variable explanations and inline comments, a single-AOI
example, and commented batch/split configuration. Reusable helpers belong in
`chip_notebook_utils.py`. Preserve repository-path normalization, user-edited
paths and saved outputs. Run outputs belong under `notebooks/outputs/` in
distinct directories. Latest-GPKG discovery is not spatial matching: inspect
real label extents before proposing an AOI.

Validate notebook JSON, unique cell IDs and code syntax, accounting for magics.
For backend changes, add focused tests under `lfm/data_processing/tests/chip/`.
Run discovery from the **repository root** in the supported GDAL environment:

```bash
python -m unittest discover -s lfm/data_processing/tests/chip -t . -p 'test_chip*.py' -v
```

For real-data checks, inspect existing wrappers in `scripts/shell/all_tasks/`:
`sbatch_validate_aoi_chip_creation.sh`, `sbatch_validate_wac_chip_creation.sh`,
and `sbatch_profile_chip_creation_parallelism.sh`. Reuse their Explore bind
mapping, grace partition and user-selected Apptainer image. Export sbatch
inputs or use command-prefix assignments.

Distinguish dependency-skipped local tests, GDAL-backed HPC tests and actual
overlay review. Reopen outputs to verify CRS with the shared lunar equivalence
helper, affine/dimensions, band order, NoData, categorical values, boxes and
provenance. Parsing is not notebook Run All; passing unit tests does not close
a phase whose real-data acceptance remains outstanding.
