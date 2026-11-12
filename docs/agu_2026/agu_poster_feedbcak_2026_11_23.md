# AGU 2026 Poster Feedback

Poster reviewed: [agu2026_kerr_poster_lfm_v2.pdf](agu2026_kerr_poster_lfm_v2.pdf)

Title: *A Reproducible Lunar Preprocessing Pipeline for Vision Foundation Model Evaluation*

This review considers the poster, the scientist-facing workflows in `notebooks/`, and relevant backend code. Suggested measurements below are proposals, not results established during this review.

## Central recommendation

The problem → objectives → implementation framing is a good fit. **The biggest opportunity is to make reproducible dataset construction the central contribution.** The current draft gives substantial attention to lunar tiling and future model comparisons, while several of the strongest informatics contributions—explicit data contracts, deterministic splits, provenance, and failure handling—are barely visible.

## Sharpen the problem statement

“The Moon needs to be studied” supplies context, but it does not yet explain why this software needed to exist. “All-encompassing” also sets a broader expectation than the demonstrated workflow supports.

Suggested problem statement:

> Lunar model evaluation requires combining imagery, contextual measurements, and annotations that differ in projection, resolution, coverage, and format. Reconstructing these preprocessing decisions for each experiment complicates reproducibility and controlled comparison.

Suggested contribution statement:

> We developed a configurable, notebook-driven pipeline that produces spatially aligned image–label datasets with explicit band layouts, reproducible split assignments, and recorded processing provenance.

That connects directly to the title. Keep lunar science motivation, but let the preprocessing problem carry the argument. The Artemis logos currently occupy valuable space without explaining that problem; one small contextual image would be sufficient.

## Make the objectives explicit

Currently, “Motivation” blends motivation and requirements, while “Dataset Creation” lists capabilities. Four numbered objectives would give the audience a clearer structure:

| Objective | How the workflow addresses it | Evidence to show |
|---|---|---|
| **1. Harmonize heterogeneous inputs** | Spatial indexing, lunar tiling, target-grid alignment, explicit band ordering | Different source layers becoming one aligned chip |
| **2. Make datasets reproducible and auditable** | Saved window plans, seeded group assignments, configuration identifiers, manifests and diagnostics | A compact provenance example and a replay comparison |
| **3. Support practical dataset creation** | Notebook configuration, annotation tools, batch planning, parallel chip processing | A small configuration excerpt plus measured throughput |
| **4. Enable controlled model evaluation** | Published image–label pairs, explicit modalities, reusable split membership | A clearly defined downstream comparison |

Repeat those numbers beside the corresponding figures. Each implementation panel can then answer “How did we meet objective 2?” instead of merely naming a subsystem.

## Show the complete dataset workflow

The poster currently shows substantially the same chip-creation diagram twice: once under Dataset Creation and again inside Finetuning FMs. Consolidating them would free considerable space and let the surviving diagram become readable from farther away.

Suggested structure:

**Choose imagery and labels → plan windows and splits → create aligned chips → inspect and publish → evaluate models**

Show imagery and annotations as separate inputs that converge on the same target grid. Under the processing stages, add a narrow strip showing the records produced: **window plan, split assignments, per-sample diagnostics, dataset manifest**. This makes reproducibility visible.

Two details from the notebooks deserve emphasis:

- The [full workflow](notebooks/chip_full_workflow.ipynb) plans nonoverlapping 256 × 256 native-pixel windows, filters invalid source windows, and assigns spatial blocks before processing. “Repeat single-sample steps” understates this dataset-level functionality.
- Intermediate tiles and final chips serve different purposes. Explain that tiling supplies standardized spatial access, while final dynamic chips align to the original source lattice. Avoid implying that acquisition zoom determines final chip resolution.

The notebooks themselves are also part of the contribution. A small “Scientist chooses / Pipeline handles” comparison could communicate the interface:

| Scientist chooses | Pipeline handles |
|---|---|
| Imagery, labels, study area, modalities, split policy | Source discovery, grid alignment, label conversion, band assembly, validation and output records |

This would be more informative than a large notebook screenshot.

## Give the results section evidence about the pipeline

At present, the only planned quantitative result is DINOv3 versus LFM across band counts. That can demonstrate the pipeline’s usefulness, but it does not directly establish reproducibility or efficiency.

Prioritize three compact pieces of evidence:

1. **Correctness:** one enlarged chip, label mask, and overlay, accompanied by verified grid, dimensions, and band layout.
2. **Reproducibility:** repeat a fixed workload with serial and parallel execution; report agreement in sample membership, split assignments, label arrays, raster values, and relevant metadata. Distinguish content equivalence from byte-identical files.
3. **Efficiency:** chips per minute versus worker count for a fixed workload, with hardware, band count, and index/cache conditions stated.

These are proposed measurements. Even a modest measured workload would substantiate “quickly” and “efficient” better than those adjectives alone.

For the model comparison, use named input sets—such as VIS, VIS+UV, and VIS+UV+static—alongside band counts. A count alone does not tell the reader what information was added. Keep split membership and evaluation coverage consistent. The repository’s [experiment planning document](docs/instance_band_experiment.md) records unresolved comparability issues, so present this as a planned demonstration until the protocol and runs are settled.

## Simplify the visual hierarchy

The dark teal title and section bars work well, and the real lunar imagery gives the poster useful specificity. However:

- The saturated blue panels, pale blue frames, and borders compete with the figures. White or very pale backgrounds would give the imagery and plots more emphasis.
- Several figure labels and captions are much smaller than the surrounding bullets. Enlarge the main workflow and remove secondary diagram detail.
- The title leaves “Evaluation” alone on its second line. An intentional break before “for Vision Foundation Model Evaluation” would read more naturally.
- “Finetuning FMs” contains much of the overall workflow. Rename that area around its purpose, such as “Controlled Model Evaluation,” and move preprocessing into the central workflow.

A workable layout would be a short problem-and-objectives band across the top, one large workflow in the middle, and objective-linked evidence below. Keep Armstrong tiling as an inset explaining spatial addressing; its zone counts and zoom ranges need less prominence than its role in interoperability.

## Specific corrections

- **Static bands:** the current canonical collection has **63**, yielding **70 total WAC+static bands**, rather than 68 static bands.
- **Tiling wording:** replace “TM and Polar zones are equally sized/spaced.” The repository describes 90 LTM zones and two differently structured polar grids; emphasize stable multiresolution addresses.
- **Splits:** say “reproducible spatial-block splits.” The notebook explicitly notes that neighboring chips across block boundaries can enter different splits.
- **Terminology:** expand WAC, NAC, IMP, and AOI on first use; change “Lunar Reconnaissance Orbit” to “Lunar Reconnaissance Orbiter.”
- **Footer:** the references and acknowledgements appear to contain template material concerning terrestrial imagery, Maxar licensing, and land-cover grants. Replace these with the sources and support applicable to this work.
- **Export size:** this PDF is **40 × 40 inches**. The earlier repository handoff mentions 44 × 44, so reconcile the intended size before final typography and export.

The highest-priority revision is to **replace the duplicated workflow with one larger diagram that visibly connects each objective to an output and a piece of evidence**. That would make the computer-science contribution much easier to recognize.
