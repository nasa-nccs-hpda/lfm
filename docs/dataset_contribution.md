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
  - `mask`: 2D instance mask with `0` as background and `1..N` as instances.
  - `bboxes`: shape `(N, 4)`.
  - `num_craters`: scalar count.

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
