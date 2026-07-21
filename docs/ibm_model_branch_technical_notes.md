# ibm_model Branch Technical Notes

This document summarizes the main technical changes made on the `ibm_model`
branch as the full-model/Graha workflows were brought into the LFM repo. It is
intended as branch-history context, not as a per-experiment results log.

## Starting Point

- We kept the phase 1 (toy model) instance and semantic segmentation code.
- Graha/Lunar-FM and TerraTorch/TerraMind code was brought into the repo from the IBM team.
- Early work was notebook-heavy and mirrored the other team's TerraTorch-style setup.
- We wanted to develop fine-tuning workflows (training, testing) from notebooks and scripts directly, not through the `terratorch fit` CLI workflow.

## Direct Python Workflow

- Added direct Python construction of Graha/TerraTorch models, datamodules, tasks, losses, callbacks, and trainers.
- Created notebook workflows for interactive development.
- Created matching `.py` scripts and sbatch wrappers for HPC runs.
- Added train-then-checkpoint-sweep pipeline scripts so training and evaluation can run as one background job.
- Moved most executable Python/shell scripts out of notebooks into `scripts/python` and `scripts/shell`.

## Data Layout

The current full-model workflows expect split directories:

```text
data_root/
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

Semantic segmentation:

- Inputs are 7-band WAC `.tif` chips.
- Labels are `.npy` masks.
- Labels are binary crater/background masks.

Instance segmentation:

- Inputs are 7-band WAC `.tif` chips.
- Labels are `.npz` archives.
- Expected label keys are `mask`, `bboxes`, and `num_craters`.
- Instance masks use `0` as background and positive integer IDs for crater instances.
- Bounding boxes come from the preprocessing/tiling flow and are treated as cropped `xyxy` boxes for object-detection-style training.

## Spatial Handling

- The Graha workflows use cropped inputs, not resizing, to align with the 256x256 model setup.
- The toy semantic workflow was updated to support the same split directory structure and crop behavior for closer comparisons. This was mainly done through the `scratch.ipynb` notebook.
- Instance segmentation datamodules preserve instance masks and boxes through cropping.

## Normalization

- Full-model/Graha workflows calculate per-band mean/std from the finetuning training split.
- Toy workflows can optionally use the same style of z-score normalization.
- Earlier experiments showed toy model instability with some normalization settings, so normalization remains configurable.

## Graha WAC Modality Handling

This became an important technical correction.

Initial approach:

- Registered a synthetic `wac` image modality with 7 channels.
- This worked mechanically, but `wac` was not present in the pretrained `modality_info.yaml`.
- Because of that, new WAC modality embeddings were randomly initialized.

Current preferred approach:

- Use the pretrained modalities already present in `modality_info.yaml`.
- Treat the first five WAC bands as `vis`.
- Treat the last two WAC bands as `uv`.
- Use:

```bash
--graha-wac-mode vis-uv
--graha-vis-uv-merge-method mean
```

This causes the Graha backbone to use:

```python
backbone_modalities = ["vis", "uv"]
backbone_new_modalities = None
backbone_merge_method = "mean"
```

Important future lesson:

- Before adding a new modality, check `modality_info.yaml` and the pretraining config.
- If the data can be represented using pretrained modalities, prefer that over randomly initializing new modality embeddings.

## Semantic Segmentation Workflow

Added or updated:

- Toy DINO semantic comparison workflow.
- Graha semantic finetuning workflow.
- Shared comparison script and notebook.
- Semantic checkpoint sweep script and sbatch wrapper.
- Train-then-checkpoint-sweep pipeline.

Current semantic sweep outputs per sample:

- input tensor
- label mask
- hard class prediction
- explicit class prediction alias
- logits
- per-sample metrics

Current semantic aggregate metrics include:

- pixel accuracy
- foreground precision
- foreground recall
- foreground F1
- IoU
- foreground average precision
- background average precision
- mean average precision

## Instance Segmentation Workflow

Added or updated:

- Object-detection-style instance datamodule.
- Graha/Lunar-FM Mask R-CNN instance finetuning workflow.
- Toy instance comparison workflow.
- DINO Mask R-CNN toy path for a closer decoder/head comparison than Mask2Former.
- Instance checkpoint sweep script and sbatch wrapper.
- Train-then-checkpoint-sweep pipeline.

Current instance sweep outputs per sample:

- input tensor
- ground-truth instance mask
- predicted instance mask
- predicted classes
- predicted score logits
- ground-truth boxes
- predicted boxes
- predicted scores
- per-sample metrics

Current instance aggregate metrics include:

- semantic foreground F1 from union masks
- instance precision
- instance recall
- instance F1
- instance mean IoU
- AP50
- mAP across IoU thresholds 0.50:0.05:0.95

## Checkpointing And Output Structure

Training outputs are written under timestamped output directories.

Typical structure:

```text
output_root/
  date_YYYY_MM_DD-time_HH_MM_SS/
    checkpoints/
      toy_model/
      full_model/
    plots/
      single_model/
        toy_model/
        full_model/
      comparison/
    prediction_cache/
    checkpoint_sweep/
    config.json
    timing_summary.json
```

Checkpoints use separate model folders:

- `checkpoints/toy_model`
- `checkpoints/full_model`

Checkpoint filenames were adjusted to avoid path separators from metric names.

## Prediction Cache

Prediction cache files are used as an intermediate normalized prediction format.

Why they exist:

- Toy and Graha models expose different raw output formats.
- The cache gives plotting and comparison code a shared format.
- It avoids holding both large models in memory for side-by-side plots.
- It makes failed plotting/metrics steps easier to debug without rerunning model inference.

Tradeoff:

- During checkpoint sweeps, cache files duplicate some information that is later written into final per-sample output folders.
- This is convenient but creates extra disk I/O and redundant intermediate files.

Potential future refactor:

- Split model-output normalization from cache writing.
- Let sweeps write final outputs directly from normalized in-memory samples.

## Logging

- Instance and semantic scripts now use similar sbatch-friendly progress logs.
- Logs include epoch start/end, periodic train-batch progress, and validation start/end.
- `--progress-log-every-n-batches` controls logging frequency.

Example:

```text
[DINO] train epoch 1/100 started
[DINO] epoch 1 train batch 20/62
[DINO] validation started
[DINO] validation finished
[DINO] train epoch 1 finished in 123.4s
```

## HPC And Runtime Notes

- Current training/inference is single-GPU from the Lightning perspective.
- Requesting more GPUs in sbatch does not speed up these scripts unless the training code is changed to use multiple devices.
- Increasing CPU allocation can help dataloading up to the point where workers saturate available I/O.
- Graha Mask R-CNN can OOM on 32 GB V100s with large batch sizes.
- `--graha-batch-size 16` caused OOM in instance segmentation.
- `--graha-batch-size 8` is a more reasonable first retry; lower to `4` if needed.

## Known Technical Lessons

- Keep internal `Namespace` objects synchronized with script config builders. Several sweep bugs came from new config fields being added to comparison scripts without updating the sweep helper namespaces.
- Prefer `getattr(args, "field", default)` in config builders when notebooks or internal helper namespaces may lag behind CLI changes.
- Be careful with metric names in checkpoint filenames. Lightning can interpret metric names containing `/` as path separators unless filenames are sanitized or `auto_insert_metric_name=False` is used.
- Avoid registering new Graha modalities unless the pretrained config actually supports them or random modality embeddings are intentional.
- Treat validation prediction plots and checkpoint-sweep metrics as different outputs:
  - validation plots are quick qualitative checks during/after training
  - checkpoint sweeps are the quantitative test-set evaluation path

## Current Preferred Commands

Semantic full train plus sweep should use:

```bash
--graha-wac-mode vis-uv
--graha-vis-uv-merge-method mean
--spatial-transform crop
--target-size 256
```

Instance full train plus sweep should use:

```bash
--toy-architecture dino-mask-rcnn
--graha-wac-mode vis-uv
--graha-vis-uv-merge-method mean
--target-size 256
```

Batch sizes should be chosen based on the GPU:

- Semantic can usually tolerate larger batch sizes.
- Instance Graha Mask R-CNN is memory-heavy and may need smaller Graha micro-batches.
