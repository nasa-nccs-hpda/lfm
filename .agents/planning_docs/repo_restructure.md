# Data Processing Package Restructure

Created: 2026-10-06. Status: backend migration implemented and locally validated;
container/HPC and interactive notebook validation remain outstanding.

## Objective and scope

Move the repository-root `model/` implementation into
`lfm/data_processing/`, separating chip creation, lunar tiling, and clustering.
Move `lfm/labeling/` into the same package. Update repository callers, tests,
resource lookup, execution wrappers, and documentation together.

The user requested `chip/`, `tiling/`, and `labeling/`, and accepted a separate
`clustering/` directory for the existing clustering utilities. This document
tracks the migration. The user subsequently authorized implementation, including
consistent canonical imports throughout the repository.

Keep processing behavior, module filenames, public function signatures, output
schemas, numerical policies, and data locations stable. Training/inference
packages and the vendored `graha-lunar-fm/` tree remain in their current
locations. Do not combine the migration with algorithm changes, legacy-code
removal, notebook renames, or a new packaging/build system.

## Current scope and migration risks

The inventory at planning time contains:

- 50 Python implementation/package files in `model/`, including its facade,
  15 top-level `chip_*.py` modules, two legacy `chip_making/` files, and five
  clustering files.
- 43 test modules plus `model/tests/__init__.py`; static inspection counted
  495 test methods. This is an inventory, not a test execution result.
- Two Python files in `lfm/labeling/`.
- Eight notebooks with relevant imports, including archived toy-model examples.
- Additional Python utilities, shell/Slurm wrappers, documentation, and agent
  guidance that refer to the existing package paths.

Recheck this inventory before implementation because other work may add files.
The main risks are import resolution, resource paths, and test discovery:

1. Callers currently use both `model.*` and `lfm.model.*`. The latter often
   works by placing the checkout's parent on `sys.path`, treating the checkout
   itself as `lfm`; the new package belongs to the inner `lfm/` directory.
2. Several modules and tests derive repository paths from fixed `__file__`
   parent depths. Moving files breaks those paths unless updated.
3. The current `model/__init__.py` exports chip and tiling APIs together.
   Splitting it requires an explicit export map and careful import ordering.
4. Tests patch modules by string name and import fixtures from other tests.
   Moving files without updating these references can invalidate coverage.
5. Importing through the inner `lfm/` executes its existing geospatial
   environment initialization. Verify PROJ/GDAL behavior in the container.
6. Clustering helpers currently perform working-directory-based notebook
   discovery at import time. Their relocation must preserve the archived
   notebook entry point without requiring that working directory for imports.

## Target layout

```text
lfm/
  data_processing/
    __init__.py
    _paths.py
    chip/
      __init__.py
      chip_*.py
      chip_making/
        __init__.py
        chip_constants.py
        chip_utils.py
    tiling/
      __init__.py
      ...existing tiling, indexing, grid, and band-contract modules...
    labeling/
      __init__.py
      craters.py
    clustering/
      __init__.py
      ...existing five clustering modules...
    tests/
      __init__.py
      chip/
        __init__.py
        test_chip_*.py
      tiling/
        __init__.py
        ...existing tiling and legacy regression tests...
      labeling/
        __init__.py
      clustering/
        __init__.py
```

The labeling and clustering test directories are destinations for relevant
existing tests found during the full audit or focused migration checks; do not
invent broad new feature suites merely to populate them.

### Module ownership

| Current source | Destination | Notes |
| --- | --- | --- |
| `model/chip_*.py` | `lfm/data_processing/chip/` | Includes chip label planning, conversion, reprojection, publication, and notebook helpers. |
| `model/chip_making/` | `lfm/data_processing/chip/chip_making/` | Preserve the legacy chip workflow and its consumers. |
| `model/clustering/` | `lfm/data_processing/clustering/` | Preserve all five existing modules. |
| `lfm/labeling/` | `lfm/data_processing/labeling/` | Interactive crater-labeling application; keep chip-stage label conversion in `chip/`. |
| Remaining top-level implementation modules in `model/` | `lfm/data_processing/tiling/` | Enumerated below; exclude `__init__.py`. |
| `model/__init__.py` | Split between `chip/__init__.py` and `tiling/__init__.py` | Audit each existing export; keep the new parent facade minimal. |
| `model/tests/test_chip_*.py` | `lfm/data_processing/tests/chip/` | Update fixture imports and mock paths. |
| Remaining `model/tests/test_*.py` | `lfm/data_processing/tests/tiling/` | Includes visualization/helper and legacy regression tests. |

The tiling implementation group is:

```text
Pipeline.py              TmsIntersector.py       TmsTileDef.py
TmsZoneDef.py            configured_tiler.py     create_gpkg.py
grid_registry.py        grid_router.py          grid_tile_def.py
lunar_crs.py             notebook_indexes.py     parallel_quadtree.py
product_ids.py           product_tiling.py       raster_cube.py
source_modes.py          static_band_contract.py tile_matrix.py
tiling.py                tiling_config.py        tiling_policy.py
tiling_preparation.py    tiling_results.py       tiling_workflow.py
vector_index.py          vector_index_builder.py wac_band_contract.py
```

Keep shared CRS, product identity, and band contracts in `tiling/` initially;
chip code already depends on that layer. A separate shared-utilities package
is unnecessary for this move. Retain `Pipeline.py` as a deprecated regression
adapter, not the basis for new behavior.

### Import and compatibility policy

Use one canonical package prefix, `lfm.data_processing`, with the checkout root
on `sys.path`. Example intended public imports:

```python
from lfm.data_processing.chip import ChipConfig, create_chips
from lfm.data_processing.tiling import TileConfig, TileSourceConfig
from lfm.data_processing.tiling import create_tiles_for_aoi
from lfm.data_processing.labeling.craters import CraterLabeler
```

Use relative imports within each package and explicit sibling imports for
chip-to-tiling dependencies. Tiling must not depend on chip creation or the
interactive labeling UI. Importing tiling must not eagerly import clustering
or widget-based labeling modules.

The planned default is a coordinated repository migration with no permanent
`model`, `lfm.model`, or `lfm.labeling` compatibility packages. This breaks old
imports in external notebooks/scripts; document the replacement paths. If
external callers are identified during the audit, record a bounded transition
policy before adding shims. Avoid loading the same implementation under two
module names, which can break class identity and test patching.

## Implementation phases

Use `[Not Started]`, `[In Progress]`, `[Implemented]`, and `[Complete]`.
`[Implemented]` means edits and local checks are done but required container
validation remains. `[Complete]` requires the phase's recorded evidence.

### R0 — Baseline and caller inventory `[Implemented]`

- Record the starting revision and preserve unrelated working-tree changes.
- Inventory every source file and assign its destination using the mapping
  above; reconcile any new or uncategorized modules.
- Find absolute/relative imports, dynamic import strings, mock targets,
  subprocess module names, test commands, resource paths, and notebook root
  checks. Search shell scripts, hidden agent files, and notebook code/Markdown.
- Distinguish references to this package from neural-network variables named
  `model`, third-party modules, and unrelated saved-model paths.
- Capture baseline test results in the available local environment and the
  supported container, with passed/failed/skipped counts and dependency gaps.
- Identify existing external consumers if known; record compatibility scope.

Acceptance: complete file/caller mapping and an explicit baseline, including
any pre-existing failures. No claim of baseline success from static counts.

### R1 — Relocate implementation and establish imports `[Complete]`

- Move files with content-preserving operations so Git can detect renames.
- Add package initializers; distribute the old facade exports by ownership.
- Update internal imports, including legacy chip-to-`Pipeline` references and
  clustering imports. Preserve function/class names and signatures.
- Introduce `data_processing/_paths.py` as the common repository-root helper;
  it must not import processing modules or introduce circular dependencies.
- Use it for repository-owned TMS/CRS metadata and default notebook output
  locations. Keep `TMS/`, datasets, indexes, and notebook outputs in place.
- Audit `lunar_crs.py`, `grid_registry.py`, `TmsTileDef.py`,
  `chip_notebook_utils.py`, and clustering discovery logic in particular.
- Account for geospatial environment initialization through `lfm/__init__.py`.

Acceptance: canonical imports resolve to the moved files, cross-package
imports are acyclic, and resource lookup works independently of the caller's
working directory. No change to scientific behavior or data locations.

### R2 — Update callers and execution wrappers `[Complete]`

- Update active and archived notebook imports and root-discovery guards that
  currently require a root `model/` directory. Preserve their configurations,
  cell identities, and unrelated content; do not execute them just to rewrite
  imports or introduce new saved outputs.
- Update `lfm/all_models/all_tasks/tiling_utils.py` and any additional internal
  consumers found by the audit. Keep this helper's location for this migration.
- Update scripts that insert the repository parent; use the repository root
  for the canonical inner `lfm` package. Audit process-pool import behavior.
- Update `scripts/python/test_container_dependencies.py` to test canonical
  chip, tiling, labeling, and applicable clustering imports.
- Update shell/Slurm test module names, Python working directories, and bind
  assumptions, especially `sbatch_tiling_modernization_tests.sh`.
- Preserve the recently relocated container-test entry paths under
  `scripts/python/` and `scripts/shell/`.

Acceptance: repository entry points resolve the new package without depending
on the checkout directory being named `lfm` or importing the outer checkout as
a package. Existing CLI arguments and container bind destinations still work.

### R3 — Migrate tests and validate behavior `[Implemented]`

- Move existing tests by ownership, update mock strings, shared-fixture
  imports, and repository-relative fixture/script lookup.
- Preserve the existing fixtures and assertions. Add focused checks only for
  migration risks not already covered: canonical import identity, root/resource
  discovery, and wrapper execution from supported directories.
- Run Python syntax checks, shell syntax checks, and notebook JSON/code checks
  accounting for Jupyter magics. Verify cell IDs remain valid and unique.
- Run the complete migrated test suite and compare its inventory/results to
  R0; explain any difference in test counts or skips. Intended discovery from
  the repository root is:

  ```bash
  python -m unittest discover -s lfm/data_processing/tests -t . -v
  ```

- Run container dependency checks and the migrated Slurm validation wrapper.
  Validate PROJ/TMS resource access and process workers in that environment.
- Exercise representative tiling, AOI chip creation, crater labeling, and
  archived clustering workflows. Use temporary/per-run outputs and preserve
  shared source indexes and datasets.
- Compare relevant raster metadata, masks/values, band order, NoData,
  chip/label alignment, and output schemas with baseline evidence. Verify the
  labeling UI still loads imagery and exports the expected GeoPackage format.

Acceptance: no unexplained new failures, skips, missing tests, import errors,
resource-path failures, or output-contract differences. Record local checks
separately from container/HPC and interactive notebook evidence.

### R4 — Documentation and retirement of old paths `[Implemented]`

- Update README, `TMS/README.md`, current documentation under `docs/`, and
  executable examples to show the new layout and imports.
- Update active agent guidance and contracts, especially the
  [lunar-tiling skill](../skills/lunar-tiling/SKILL.md),
  [tiling/chip handoff](../tiling_to_chip_creation_handoff.md),
  [chip backend structure](../chip_creation_backend_structure.md), and the
  [AOI chip plan](aoi_chip_creation_and_label_clipping_plan.md).
- Preserve historical evidence as history; annotate old paths where useful
  instead of rewriting old execution records as though they used the new tree.
- Remove emptied legacy directories after all consumers are migrated. Audit
  the empty root `__init__.py`; remove it only after confirming no supported
  entry point needs the outer-checkout package convention.
- Search for remaining `model.*`, `lfm.model.*`, `lfm.labeling.*`, filesystem
  paths, and old test commands. Classify residual matches rather than globally
  replacing the word `model`.
- Record final test evidence, migration examples, external compatibility
  implications, and any remaining validation limitations in this plan.

Acceptance: no executable repository references to retired import paths or
directories, documentation matches the implemented tree, and all required R3
validation evidence is recorded.

## Delivery and completion

Keep this a dedicated structural refactor. R1/R2 and the test-path portion of R3
are coupled: intermediate moves may be unimportable and should not be released
as independently usable changes. Review the completed diff for accidental
algorithm changes and large notebook/line-ending churn.

The initial effort estimate is 1–2 developer days for migration and local
validation, plus container/HPC and interactive checks. This is an estimate,
not a completion commitment; environment access and baseline failures may
affect it.

Completion requires all phases above, preserved processing contracts, a
documented canonical import convention, and no unexplained validation
regressions. A rollback should revert the coordinated migration as a unit;
it should not require relocating datasets, labels, indexes, or checkpoints.

## Evidence log

- 2026-10-06: Created this plan after inspecting the current package layout,
  import conventions, resource lookup, notebook entry points, and Slurm test
  wrapper. No implementation moves or runtime validation performed.
- 2026-10-06: Implemented the authorized migration from baseline revision
  `bfdd494aa8147806723affd277e1e1b93fe523cb`. The starting worktree was clean.
  Moved all implementation files and the existing 43 test modules according to
  the mapping above. Split all 199 original public exports between the chip
  and tiling facades. Preserved all original test methods.
- Added `lfm/data_processing/_paths.py` for repository-owned resources. Removed
  the retired `model/`, `lfm/labeling/`, and empty outer `__init__.py`. No
  compatibility aliases remain. Relative imports stay within the canonical
  package; external callers use `lfm.data_processing.*` with the checkout root
  on `sys.path`.
- Updated eight notebooks, the shared tiling helper, Python entry points,
  container smoke imports, and affected Slurm wrappers. Verified 21 Python
  entry-point bootstraps and all eight notebook bootstrap cells resolve the
  checkout root from their supported locations. Archived notebooks correctly
  account for their extra `toy_model/` directory level.
- Preserved notebook outputs, execution counts, cell metadata, and existing
  cell IDs. Changed source cells parse after accounting for notebook magics;
  changed notebook JSON is valid and existing cell IDs are unique. Historical
  output tracebacks may still contain the old paths, intentionally.
- Parsed 267 repository Python files outside the vendored Graha tree and
  verified all explicit local backend import targets exist. Changed shell
  wrappers pass `bash -n`. Mocked Apptainer/Slurm execution verified canonical
  test module names, working directories, bind/entry paths, and flag forwarding.
- Baseline local discovery from the checkout parent ran 465 test cases:
  301 passed, 160 skipped, and four test-module import errors caused by missing
  `osgeo` (`test_Pipeline`, `test_TmsIntersector`, `test_TmsTileDef`, and
  `test_TmsZoneDef`). The migrated original suite produced the same counts.
- Added five package-layout regression tests covering public export/class
  identity, tiling import isolation from chip/widget packages, resource lookup
  from another working directory, a differently named checkout path, and
  type identity through a spawned worker. Final local discovery ran 470 cases:
  306 passed, 160 skipped, and the same four missing-GDAL import errors.
  No additional failures or skips were introduced.
- Current README, TMS documentation, backend structure/handoff references, and
  the lunar-tiling skill use the new package paths. Historical plans have a
  migration note instead of rewritten execution evidence. Labeling/clustering
  test directories were not created empty; future tests can use those planned
  destinations.
- R0/R3/R4 retain `[Implemented]` status because this environment has neither
  Apptainer nor Slurm and lacks scientific dependencies needed for full runtime
  validation. No container/GPU job, real-data raster comparison, or interactive
  labeling/clustering session was run. Those checks remain necessary for full
  environment acceptance; the backend and repository caller migration itself
  is implemented.

### Remaining environment validation

From the checkout root in the supported container, run:

```bash
python -m unittest discover -s lfm/data_processing/tests -t . -v
python scripts/python/test_container_dependencies.py --skip-gpu
```

For the existing GPU/container and tiling Slurm entry points, submit from the
checkout root:

```bash
sbatch scripts/shell/test_container.sbatch /absolute/path/to/container
sbatch scripts/shell/all_tasks/sbatch_tiling_modernization_tests.sh
```

Then exercise the tiling/chip notebook workflows and interactive labeling and
archived clustering workflows, recording dependency versions and any output
differences against existing acceptance evidence. Retain the existing shared
source-index protections and per-run output directories.
