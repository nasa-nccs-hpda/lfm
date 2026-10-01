# Lunar Tiling Progress Log

This document tracks day-by-day work on the LFM lunar tiling backend,
validation, documentation, and public notebook. It is an execution record, not
the authoritative API specification.

Canonical references:

- [Tiling modernization plan](../../docs/tiling_modernization_plan.md)
- [Armstrong tiling scheme](../../TMS/README.md)
- [Supported tiling notebook](../../notebooks/tiling_example.ipynb)
- [Lunar tiling agent skill](../skills/lunar-tiling/SKILL.md)

## Current status

- Core tiling modernization: **Complete**
- Supported production geometry: numbered LTM zones
- Public examples: WAC/static at zoom 5 and NAC/static at zoom 11
- Tiling resampling: bilinear
- Canonical static cube: 63 bands with `-32768` output NoData
- Downstream chip modernization: tracked separately

Known boundaries and follow-ups:

- LPN/LPS production tiling is not implemented.
- Automatic discovery followed by one output cube per PID is not implemented.
  An `all_intersecting` WAC/NAC query instead stacks all intersecting raster
  bands into one source cube per LTM tile with `product_id=None`.
- The completed tiling plan records a maintainer-accepted exception: a separate
  top-to-bottom execution of the final `.ipynb` after the last NAC AOI and
  differential-zoom edits was not recorded.
- The no-PID alternate AOI query added on 2026-09-30 has passed notebook JSON
  and Python-cell syntax validation but has not yet been run against Explore
  data.

## Daily entries

### 2026-10-01

Goal:

- Establish a persistent daily progress record for tiling work.

Completed:

- Created `.agents/progress_docs/tiling.md`.
- Seeded the log with the completed modernization milestones and current
  validation boundaries.

Validation:

- Passed whitespace validation and confirmed that every canonical reference
  resolves to an existing repository file.

Next:

- Append a dated entry whenever tiling code, configuration, notebook behavior,
  validation evidence, or public documentation changes.

### 2026-09-30

Completed:

- Reviewed whether the supported notebook could run without PID filtering.
- Confirmed that `selection_mode="product_id"` requires a nonempty selector.
- Added an opt-in WAC AOI query under `RUN_ALTERNATE_QUERIES` that uses
  `selection_mode="all_intersecting"` and passes no `selectors` argument.
- Documented that this no-PID query produces one stacked WAC cube per LTM tile,
  not one cube per discovered PID.
- Created `.agents/skills/lunar-tiling/SKILL.md` to preserve the modern tiling
  contract and operational conventions for future agent work.

Validation:

- Confirmed that `notebooks/tiling_example.ipynb` remains valid JSON.
- Confirmed Python syntax for every code cell after excluding Jupyter magic
  lines.
- Validated the `lunar-tiling` skill with the official skill validator.
- The new alternate query was not submitted to Explore on this date.

### 2026-09-03

Completed:

- Finalized the user-facing structure and explanatory configuration comments in
  `notebooks/tiling_example.ipynb`.
- Kept user-edited paths, WAC/NAC PIDs, and AOIs separate from derived indexes,
  zoom levels, timestamped outputs, display settings, and path validation.
- Confirmed WAC/static at zoom 5 and moved 1 m NAC/static to zoom 11.
- Added NaN-masked plotting and a maximum display of four tile pairs per AOI.
- Documented the Armstrong scheme, IAU:30100, LTM zones, zoom matrices, tile
  addressing, and the unsupported LPN/LPS production path in `TMS/README.md`.
- Updated the tiling and chip modernization plans with the stable handoff
  contract.
- Marked tiling Phases T0-T8 complete by maintainer signoff.
- Clarified the chip plan's static NoData language: successful canonical static
  cubes contain all 63 ordered bands with `-32768` output NoData; the two
  Mini-RF bands use `-3.4028230607370965e38` only as a source sentinel; WAC and
  NAC preservation is explicitly configured.

Validation:

- Recorded the lack of a separate final `.ipynb` execution as an acceptance
  exception rather than claiming that run occurred.
- Passed notebook structure, JSON, and Python syntax checks available locally.

### 2026-09-01

Completed:

- Replaced the WAC-specific, hard-coded tiling interface with the
  configuration-driven `TileConfig`, `TileSourceConfig`, and
  `BandNoDataOverride` contract.
- Added public AOI, point, and explicit tile-index entry points returning
  structured `TileCubeRecord` results.
- Moved the shared lunar geographic CRS to
  `TMS/IAU_30100_2015.wkt` and loaded it from repository data.
- Generalized source indexes to existing `.shp` or `.gpkg` files queried
  read-only.
- Standardized all tiling warps on bilinear resampling.
- Established the canonical 63-band static order and standardized every static
  output band on `-32768` NoData.
- Preserved the two Mini-RF `-3.4028230607370965e38` source sentinels so they
  are masked before interpolation and converted on output.
- Preserved native WAC and NAC source NoData through explicit configuration.
- Added diagnostics for source ranges, static sentinels, written cube NoData,
  modern-versus-legacy comparison, and display-only overflow investigation.
- Standardized new Slurm wrappers on the `grace` partition and the
  `lfm-container-ipyleaflet` Apptainer image.

Validation:

- WAC validation passed.
- NAC validation passed.
- Mixed WAC/static validation passed after static NoData standardization.
- Modern/legacy regression comparisons passed for covered products; products
  without selected WAC coverage produced an intentional explicit modern error
  instead of a legacy static-only result.
- Determinism validation passed for record order, record metadata, output
  SHA-256 hashes, declared-index immutability, visible index inventory, and
  absence of newly created output indexes.

## Entry template

### YYYY-MM-DD

Goal:

- What this day's work intends to accomplish.

Completed:

- Code, notebook, configuration, diagnostic, or documentation changes.

Validation:

- Commands or jobs run, artifacts inspected, and pass/fail results.
- State explicitly when a change received only static validation.

Decisions:

- Contract decisions, tradeoffs, and boundaries established that day.

Next:

- Remaining work or the next safe step.
