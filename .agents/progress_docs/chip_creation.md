# Chip Creation Progress Log

Execution evidence and follow-ups; the [AOI plan](../planning_docs/aoi_chip_creation_and_label_clipping_plan.md)
and [label contract](../crater_labeling_to_chip_creation_contract.md) remain
the implementation references. See also the [chip skill](../skills/lunar-chip-creation/SKILL.md).

## 2026-10-07 — dataset coverage, labels and notebook readiness

### Completed and accepted behavior

- Three notebooks separate the concise single-AOI example, full-raster dataset
  workflow, and polar examples. Full workflow uses nonoverlapping native
  256×256 windows, drops incomplete edges, and assigns seeded 80/10/10 splits
  using count-aware placement of intact spatial blocks (default 4×4 chips).
  No clustering/merging or exposed overlap setting. Chip/index workers use
  the allocation; creation runs directly, with an aggregate worker-aware bar
  and no stage heartbeat messages. Rerun separately for each label file.
- Added source-validity filtering before splits: all pixels in every selected
  source-raster band must be unmasked and finite. Windowed reads avoid loading
  full NAC scenes. An optional same-shape Boolean mask further restricts
  coverage. Record rejected window counts without shifting IDs/block anchors.
  WAC planning checks VIS only, not UV/static. The underlying single-AOI
  publication policy still permits NoData.
- Full-workflow visualization requires all dynamic output bands valid (seven
  WAC or one NAC), but permits static gaps. WAC/NAC band 0 is displayed in
  grayscale alongside labels/overlay. All-band NoData totals remain available;
  unknown validity is not treated as valid. Plot selection does not prioritize
  crater-containing chips.
- Nonpolar examples now use `M1412665711CE.prj.vis.mos_label_craters.gpkg`
  under `PROJECT_DATA_DIR/example_notebook_inputs`, paired with
  `processed_data/Lunar/LRO_WAC_Pho_Sites/M1412665711CE.prj.vis.mos.tif`.
  Local download suffixes `(1)`/`(2)` are not HPC defaults; the updated label
  contents must be copied to the configured HPC path.
- Reorganized polar markdown/configuration into matching ordered groups:
  study setup, imagery, labels, AOIs, execution/output. Defaults and assignment
  expressions are unchanged; polar input paths were not replaced by LTM data.
- Moved `make_static_source` to `lfm.data_processing.tiling.static_source`,
  exported through `lfm.data_processing.tiling`. All three chip notebooks and
  the AOI validation script use the new import; the old model helper re-exports
  it for compatibility. See the [tiling log](tiling.md) for import-failure evidence.

### Real-data findings

Inspected `test_outputs/wac_chip_debug.zip` locally with rasterio, NumPy,
SQLite, Shapely and pyproj. Inputs were read-only.

- The original WAC VIS raster is 1798×13815, 100 m/pixel, with about 53–54%
  intrinsic NoData per band. Its source sentinel is approximately
  `-3.4028226550889045e38`, not the output sentinel -32768.
- Across 369 successful outputs, exact source windows reproduced all five VIS
  counts of windows containing NoData / wholly empty: 254/142, 255/143,
  255/143, 255/143, 256/143. No wholly empty output VIS band had valid pixels
  in its corresponding source window. Source UV already contains ~55–56%
  NoData. These observations do not establish pixel-value identity after warping.
- Included chip `M1412665711CE_r9984_c512` has all seven dynamic bands fully
  valid. Static CPR/S1 each have ~56% NoData, H parameter ~3.8%, LDRM ~0.9%.
  Saved masks agree with manifest statistics. Requiring every output band
  valid had excluded useful imagery, not demonstrated a dynamic failure.
- Source-mask filtering retains **113 of 371** complete native windows and
  rejects 258. This is local planning evidence, not 113 proven successful
  final publications or a guarantee of UV coverage. User subsequently reported
  the filtered workflow and visualization were much better.
- Original archive GPKG had three outlines. Download `(1).gpkg` had 157 but
  none intersected the three displayed windows. Download `(2).gpkg` has 244,
  matches the source CRS, and contains supported, visible craters as follows:

| Window | Craters with pixel support | Clipped at window edges |
|---|---:|---:|
| `r10240_c256` | 10 | 2 |
| `r10240_c512` | 6 | 2 |
| `r1024_c1024` | 6 | 2 |

These counts are local geometry/pixel-center checks including highest-ID
overlap handling, not a production label-conversion or notebook Run All test.
Matching the source filename/CRS does not prove full-scene annotation completeness.

### Validation and remaining work

- Coverage/inspection changes: 24 focused tests ran, 21 passed, 3 dependency
  skips. Static helper/import checks plus notebook helpers: 15 ran, 13 passed,
  2 skipped. Fresh-process test rejects attempted model/PyTorch/Lightning imports.
- Notebook JSON, cell-ID uniqueness and code syntax checked; polar config
  expressions verified unchanged. No new full HPC test-suite acceptance claimed.
- Two failures in the archived original run remain unresolved:
  `r10240_c0`: `inconsistent_cube_bands` in 25S;
  `r9984_c0`: `inconsistent_zone_bands`.
  They are separate from expected source/static NoData. Later filtering does
  not establish that the underlying metadata mismatch was fixed.
- Coworker should pull the import fix, restart the kernel, then rerun polar
  setup. Updated `(2)` labels require regeneration of outputs and overlay review.
- Real polar NAC, pole-containing outputs and the broader edge-case matrix
  remain pending where no explicit HPC/visual acceptance exists. Prior north
  seam visual acceptance is not blanket polar acceptance.
