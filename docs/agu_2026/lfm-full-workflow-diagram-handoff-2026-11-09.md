# Handoff: design a full lunar chip-creation workflow diagram

## Next-session objective

Help the user design a **full workflow diagram** for a scientific poster. The
earlier work produced one illustrative AOI example, not a complete workflow
diagram. Use that work as verified scientific context and reusable visual assets.
Develop the new diagram's structure around the full workflow and its intended
message, rather than expanding the previous four-step composition.

The user explicitly wants the next agent to have design freedom. Layout,
orientation, visual hierarchy, grouping, level of abstraction, and use of example
images remain open. A dataset overview with an example inset, parallel data
paths, or another composition may fit; choose after understanding the story.
There is no requirement to reuse the previous diagram's geometry or code.

All repository paths below are relative to the `lfm` repository root.

## Establish the scientific scope first

Start with `notebooks/chip_full_workflow.ipynb` to understand dataset-level
planning and execution. Consult `notebooks/crater_labeling.ipynb` for annotation
and `notebooks/chip_example.ipynb` for single-AOI processing. Determine how much
of acquisition, annotation, sample planning/filtering/splits, per-chip processing,
and dataset publication belongs in the new figure before deciding its topology.

Read `.agents/skills/lunar-chip-creation/SKILL.md` for the relevant implementation
and contract pointers. In particular:

- For stage responsibilities and the actual processing sequence, use
  `.agents/chip_creation_backend_structure.md` and
  `lfm/data_processing/chip/chip_creation.py`.
- For label preparation, clipping, instance identities, and output-grid rules,
  use `.agents/crater_labeling_to_chip_creation_contract.md`.
- For intermediate tiling geometry and addressing, use `TMS/README.md` and the
  selected `TMS/RG/` definition. Read the tiling skill if this becomes a major
  part of the diagram.

Important distinction when reusing the example: its AOI is **300 × 300 native
WAC pixels**. The full-workflow notebook's automatic planning uses **256 × 256
native-pixel windows**. An example's dimensions and manual AOI-selection step
must not silently become the description of the batch-planning algorithm.

Keep labels and imagery conceptually distinct: they become aligned outputs on
the target grid; annotations do not generate imagery. The overlay is inspection
evidence. Verify grid inheritance, label-preparation order, filtering, and split
behavior against the sources above rather than inferring them from screenshots.

## User context and design preferences

The poster is **44 × 44 inches**, and this will coexist with other diagrams.
The user prefers a mostly visual explanation with very few words. Scientific
imagery should remain real data; remove plot headers, color bars, and whitespace
when these do not serve the explanation.

The earlier figure uses a transparent background, Helvetica, dark-blue text,
black arrows, and red unfilled AOI rectangles. Exact colors, export settings,
and the font-substitution limitation are in
`test_outputs/workflow_codex_figures/figure_sources.json`. These are useful
poster-style preferences to carry forward unless revised. The earlier roughly
10-inch square size, three-tier arrangement, four headings, stacked tile
presentation, and output triptych are **choices made for that example**, not a
template or required structure for the new full-workflow diagram.

## Reusable assets and their authority

Use `test_outputs/workflow_codex_figures/figure_sources.json` as the existing
example's artifact index and provenance. Use
`test_outputs/workflow_codex_figures/annotation_context_geometry.json` for exact
window coordinates, geometry, source identities, and display limits; do not copy
those tables or coordinates from memory.

- `test_outputs/workflow_codex_figures/annotation_context.svg` is a standalone
  regenerated annotation visualization with real raster imagery and all manual
  crater outlines. `annotation_context.png` is its **background only**, without
  crater outlines; the accompanying geometry JSON supplies the vector shapes.
- `test_outputs/M1097880217CE.prj.vis.mos_label_craters.gpkg` is the authoritative
  full annotation layer. `test_outputs/M1097880217CE_300px.gpkg` contains the
  clipped AOI features and the clipping footprint. Their feature sets differ.
- `test_outputs/M1097880217CE_300px.svg` was rebuilt directly from the clipped
  GeoPackage. Use it as an AOI-only geometry asset, not as the full annotation
  context.
- `test_outputs/workflow_png/tiling_inspection.png` and `chip_inspection.png`
  contain intermediate tile views and final chip/mask/overlay views. The tile
  image shows a subset of acquired addresses, not the complete acquisition.
- `test_outputs/workflow_codex_figures/chip_creation_workflow.svg`, `.pdf`, and
  `.png` are the previous single-example outputs. Consult them for provenance
  or reusable pieces; begin the new full-workflow design from its own purpose.

Reproducible helpers are `build_annotation_context.py`, `build_aoi_svg.py`, and
`build_workflow_figure.py` under `test_outputs/workflow_codex_figures/`. Inspect
them only when reusing or regenerating their assets. Their current composition
is not a specification for the next diagram. The `test_outputs/` directory is
gitignored; existing files are local artifacts, not necessarily tracked files.

## Constraints and lessons that still apply

**Raster access:** the user explicitly requires windowed reads based on crater
extent. The TIFF is `test_outputs/M1097880217CE.prj.vis.mos.tif`. Open its metadata
read-only when needed and read only bounded windows/bands. Reuse the existing
context crop when sufficient. Avoid full-raster pixel reads, whole-TIFF hashing,
and opening the full TIFF in an image viewer. The context helper records the
previous bounded read and how it was derived.

**Geometry:** draw actual GeoPackage polygons and preserve feature IDs, rings,
and clipped shapes. Do not infer crater positions or fit replacement circles
from screenshots. Earlier independent review found the old contextual SVG had
full-scene geometries rather than the actual clipped feature set; it did not
establish that the original polygon shapes had been invented. The repaired
assets have been checked vertex by vertex against their authoritative layers.

**Selection context:** the user's latest correction requires all manually
annotated crater outlines to remain visible outside the red rectangle in the
"Select AOI" view. The current example therefore shows the same full annotation
context in its first two panels. Keep this distinction between choosing a region
and displaying already-clipped labels when repurposing those assets.

**Verification:** metadata, scientific input geometry, and display pixels are
different kinds of evidence. Validate footprint placement with CRS/transforms
and feature correspondence, then inspect the rendered result. Preserve the
scientific inputs. The user previously requested a smaller implementation agent
and an independent **GPT-6.1 Sol reviewer at high reasoning**; that strategy
worked well for scientific correctness. A new review should also assess full
workflow coverage and diagram clarity, without enforcing the old layout.

## Suggested skills

Call the Skill tool for these when applicable; if that interface is unavailable,
read the corresponding local `SKILL.md` directly:

- **`lunar-chip-creation`** — establish the full workflow, stage boundaries,
  target grids, labels, planning, splits, and publication semantics.
- **`lunar-tiling`** — when explaining source discovery, intermediate tiles,
  modalities, lunar projections, or tile-address relationships.
- **`grill-me`**, which invokes **`grilling`** — if the user wants an interview
  to settle the new diagram's message and design. Explore the new diagram's
  choices rather than rerunning the old example's layout decisions.

## Current state and a useful starting point

The prior example and its exports are complete. No full-workflow composition has
been approved. Read the full-workflow notebook, identify the story the poster
needs to communicate, and resolve the new figure's scope and hierarchy with the
user. Reuse the verified assets only where they support that story, then build
and validate an editable diagram suited to the complete workflow.
