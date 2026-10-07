# Dataset Contribution Guide

This guide describes how to format a new dataset so it can be used by the LFM semantic and instance segmentation workflows.

## Directory Layout

Use split folders with the same structure for every dataset:

```text
dataset_root/
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

## Chip creation split configuration

Use [the full-raster workflow](../notebooks/chip_full_workflow.ipynb) for one
finished crater label GPKG and one WAC/NAC source raster. It creates non-overlapping
256×256 native-pixel chips across the **entire raster**, not just the crater
extent. Incomplete right/bottom windows are dropped and counted. Windows with
no rasterizable craters warn but are retained as background samples. This
assumes the entire raster was annotated; missing annotations are not reliable
negative labels. Run the notebook separately for each label file.

The notebook defaults to **80% training, 10% validation, 10% test**, seed 42.
Assignments apply to whole spatial blocks (default 4×4 chips) anchored at
source pixel (0, 0). Every chip in a block stays in the same split. Percentages
use `assignment_method="count_aware"`: balance chip totals toward the requested
ratios without splitting blocks. Every positive split receives a block when
enough unlocked blocks exist; otherwise a warning explains the empty split.
Largest-first placement and bounded whole-block moves minimize count error,
but do not guarantee an optimal partition or exact quotas. Review the preview
and published counts. Failed samples can further change proportions.

Block grouping reduces local leakage but is not a buffer: neighboring chips
and craters across block boundaries can still belong to different splits.
Select block size based on scientific independence and crater sizes. Changing
product identity, seed, chip size, block size or window inventory can change assignments. Source
IDs use `<product>_r<row>_c<column>` with zero-based native-pixel offsets.

Available configuration classes from `lfm.data_processing.chip`:

| Configuration | Behavior |
|---|---|
| `SimpleSplitConfig` | Seeded percentage assignment of entire groups. `count_aware` balances sample totals; legacy/default `hash` independently assigns groups and can leave splits empty. |
| `MixedPercentageNumberSplitConfig` | Fill fixed sample-count targets in priority order, then apply percentages to remaining groups. |
| `NumberSplitConfig` | Fixed sample-count targets in priority order; configure whether remaining samples are unassigned or sent to a remainder split. |
| `NoSplitConfig` | No partitioning; write directly to dataset-root `chips/` and `labels/`. Quick-example default. |

```python
from lfm.data_processing.chip import SimpleSplitConfig, SplitPercentages

split_config = SimpleSplitConfig(
    percentages=SplitPercentages(train=0.8, val=0.1, test=0.1),
    seed=42,
    group_key_policy="request",
    assignment_method="count_aware",
)
```

Each request's `split_group_key` must identify its spatial block. The policy
string does not construct blocks automatically. The full-workflow planner
creates these keys; using one product-only key would place the entire raster
in one split. Other workflows can deliberately use product-level grouping.

For mixed/count configurations, counts are **samples**, but groups remain
indivisible. Unattainable targets issue warnings rather than splitting groups
or failing the entire pipeline. The legacy mixed default tries to reserve
100 test samples, then uses 90% train / 10% validation for the remainder; it
is **not** the full-raster notebook default. Existing dataset membership can
be retained using `prior_manifest_path`; preserve compatible split settings.
Explicit assignments and prior-manifest locks always take precedence over
balancing, even if they make targets or nonempty splits impossible. The legacy
`hash` mode remains stable when unrelated groups are added; `count_aware`
needs a saved manifest to preserve assignments as the inventory changes.

## File Naming

Prefer identical sample stems for chips and labels:

```text
chips/M123.tif
labels/M123.npy
```

or:

```text
chips/M123.tif
labels/M123.npz
```

If suffixes are needed, keep them terminal and consistent:

```text
chips/M123_input_nac_chip.tif
labels/M123_label.npy
```

The backend can infer common terminal suffixes such as `_input_nac_chip`, `_input_wac_chip`, `_input_wac_static_chip`, `_label`, `_mask`, `_mask_orig`, `_img`, and `_chip`. Avoid filenames where role words such as `label`, `mask`, `chip`, `input`, or `img` appear in the middle of the true sample ID.

## Supported Files

Images:

- `.tif` is preferred for geospatial chips.
- `.npy`, `.npz`, and `.nc` may be supported by specific datasets or datamodules, but should be verified before training.

Semantic labels:

- Prefer `.npy`.
- Store one 2D integer mask per chip.
- Use `0` for background and positive integer class IDs for foreground classes.

Instance labels:

- Prefer `.npz`.
- Required arrays:
  - `mask`: 2D instance mask with `0` as background and visible instance IDs
    drawn from `1..N`.
  - `bboxes`: shape `(N, 4)`; row `i - 1` describes instance ID `i`.
  - `num_craters`: scalar count `N` of annotated crater boxes. Mask
    rasterization may omit an ID when its valid bounding-box region contains
    pixels assigned to another instance; otherwise every annotated ID should
    remain visible.

## Band Layout

Document the stored chip band order clearly. Examples:

```text
WAC chips:
  band 0-4: VIS
  band 5-6: UV

NAC PHO+DTM chips:
  band 0: PHO
  band 1: DTM
```

The notebook `DATA_DICT` should match that stored layout:

```python
DATA_DICT = {
    "dataset_name": "my_dataset",
    "data_dir": "/explore/nobackup/path/to/dataset",
    "dataset_modality": "nac_dtm",
    "image_glob": "*.tif",
    "label_glob": "*.npz",
    "band_filters": {
        "pho": [0],
        "dtm": [0],
    },
    "normalization_modality": "nac",
    "graha_input_modalities": ["nac", "dtm"],
}
```

`band_filters` are modality-local indices. For example, `{"dtm": [0]}` means select the first DTM band within the DTM modality, not necessarily absolute stored chip band `0`.

## Normalization

The default training behavior uses TerraMind/Graha pretraining normalization. If a dataset uses a new sensor, altered scaling, or different preprocessing, document the source of the mean/std values and verify whether `modality_info.yaml` needs a new entry or custom values.

Use `normalization_modality` to choose the pretraining normalization family:

- `vis_uv` for WAC VIS+UV data.
- `nac` for NAC PHO-only, NAC PHO+DTM, and NAC-like single-band data.

Only override `normalization_source` when intentionally running an experiment that compares pretraining stats against finetune-dataset stats.

## Pre-Training Checks

Before training, run basic diagnostics:

- Count matched image-label pairs per split.
- Print image shape and label shape.
- Print image per-band min, max, mean, and std.
- Print unique label values.
- Plot several random chip/label overlays.
- Check nodata count and percentage.
- Confirm labels align visually with image features.

## Rule Of Thumb

If a user cannot explain the dataset layout, band order, label format, and normalization in a few minutes, the dataset documentation and `DATA_DICT` are not clear enough yet.
