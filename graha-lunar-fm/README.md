# Graha Lunar-FM

Finetuning and inference distribution of the Graha Lunar-FM foundation model. This repo ships:

- `terramind/` — the model package (backbone, tokenizers, data utilities).
- `terratorch_integration/` — TerraTorch-compatible datamodules, tasks, and configs for lunar downstream tasks (crater detection/segmentation/classification, impact-melt-pond segmentation).

Pretraining code is not included.

## Install

```bash
conda create -n graha-lunar-fm python=3.12
conda activate graha-lunar-fm
pip install -e .
```

## Weights

Weights are distributed out-of-band as a tarball. Unpack it so that the layout is:

```
weights/
  backbone/
    checkpoint_weights_final.pt
    full_config.yaml
  modality_info.yaml
```

The configs reference these paths relative to the repo root (`weights/backbone/…`, `weights/modality_info.yaml`). If you want the bundle to live elsewhere, symlink `./weights` to that location.

## Data

Downstream configs reference data under `./data/…` — e.g. `data/NAC/…`, `data/IMP_segmentation_dataset/…`. Point `./data` at wherever your dataset root actually lives (symlink or copy). Each config lists the exact subpaths it expects at the top of its `data.init_args` block.

## Finetuning

Every config under `terratorch_integration/configs/` is a runnable `terratorch fit` target. Example:

```bash
PYTHONPATH=/where/is/this/folder/graha-lunar-fm:$PYTHONPATH terratorch fit -c terratorch_integration/configs/crater_detection_nac_dtm_meta.yaml
```

Common overrides:

```bash
# 1-step smoke test with no data workers
PYTHONPATH=/where/is/this/folder/graha-lunar-fm:$PYTHONPATH \
terratorch fit \
  -c terratorch_integration/configs/crater_detection_nac_dtm_meta.yaml \
  --trainer.max_steps 1 --data.num_workers 0

# Point at a specific data root without editing the yaml
PYTHONPATH=/where/is/this/folder/graha-lunar-fm:$PYTHONPATH \
terratorch fit -c <config>.yaml \
  --data.data_dir /path/to/data \
  --data.metadata_file /path/to/metadata.parquet \
  --data.annotations_file /path/to/annotations.json
```

## Testing
```bash
PYTHONPATH=/where/is/this/folder/graha-lunar-fm:$PYTHONPATH terratorch test --config config_from_finetuning.yaml --ckpt_path finetuned_model.pt

```



Available example for Graha Lunar FM and Resnet50:
 * crater detection (NAC dataset)
 * ice prospectivity
 * IMP segmentation

Full documentation of terratorch is [here](https://torchgeo.org/terratorch/quick_start/).

## Generation / inference

Deferred — a Jupyter notebook covering the generation flow will be added when it lands. Until then this repo covers finetuning only.

## Repo layout

```
Graha-lunar-fm/
├── terramind/                    # model package (backbone, tokenizers, data utils)
├── terratorch_integration/       # TerraTorch datamodules + tasks + configs
│   ├── configs/                  # runnable `terratorch fit` configs
│   ├── data_adapter.py           # LunarNACDTMDataModule (craters)
│   ├── data_adapter_imp.py       # LunarImpSegDataModule
│   ├── lunar_backbone.py         # TerraTorch backbone wrapper
│   ├── lunar_object_detection_task.py
│   ├── lunar_segmentation_task.py
│   └── lunar_register.py         # registers variants + factories
├── weights/                      # (git-ignored) unpacked weights tarball
├── data/                         # (git-ignored) datasets — symlink your data root here
├── pyproject.toml
└── requirements.txt
```
