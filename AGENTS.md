# AGENTS.md

## Repo Purpose

This branch develops full-model Lunar-FM/Graha workflows and comparison experiments against toy DINO models for lunar crater semantic and instance segmentation.

## Local Development Environment

- Codex/dev micromamba environment: `lfm-codex-dev`
- Environment path on this Windows machine: `C:\Users\ajkerr1\micromamba\envs\lfm-codex-dev`
- Activate locally with:

```powershell
micromamba activate lfm-codex-dev
```

## Current Layout

- `lfm/full_model/all_tasks`: shared full-model utilities, plotting helpers, and datamodule helpers.
- `lfm/full_model/sem_seg`: Graha/full semantic segmentation implementation.
- `lfm/full_model/inst_seg`: Graha/full instance segmentation implementation.
- `lfm/toy_model/sem_seg`: toy DINO semantic segmentation model, datasets, and Lightning wrappers.
- `lfm/toy_model/inst_seg`: toy instance segmentation models, including DINO Mask R-CNN and Lightning wrappers.
- `scripts/python/all_tasks`: shared script workflows, including train-then-sweep orchestration.
- `scripts/python/semantic_seg`: semantic segmentation scripts.
- `scripts/python/instance_seg`: instance segmentation scripts.
- `scripts/shell/semantic_seg`: semantic segmentation sbatch wrappers.
- `scripts/shell/instance_seg`: instance segmentation sbatch wrappers.
- `notebooks/full_model`: main full-model notebooks.
- `docs`: refactor plans, branch notes, and commit/change history.

## Important Workflows

### Full Comparison Training Pipeline

These run both toy DINO and Graha/full model training, then optionally run checkpoint sweeps.

- Semantic train plus checkpoint sweep:
  `scripts/shell/semantic_seg/sbatch_semantic_train_then_checkpoint_sweep.sh`
- Instance train plus checkpoint sweep:
  `scripts/shell/instance_seg/sbatch_instance_train_then_checkpoint_sweep.sh`
- Four semantic experiment launcher:
  `scripts/shell/semantic_seg/sbatch_4_sem_seg_exp.sh`

### Graha-Only Fine-Tuning

These run only the Graha/full model path.

- Semantic Graha fine-tuning:
  `scripts/shell/semantic_seg/sbatch_semantic_seg_finetuning.sh`
- Instance Graha fine-tuning:
  `scripts/shell/instance_seg/sbatch_instance_seg_finetuning.sh`

### DINO/Toy-Only Fine-Tuning

There is not a separate dedicated toy-only sbatch wrapper for every task. Use the comparison scripts with Graha skipped where available, or keep toy-only work in the notebooks/wrappers until a dedicated launcher is added.

- Semantic toy/comparison script:
  `scripts/python/semantic_seg/semantic_seg_comparison.py`
- Instance toy/comparison script:
  `scripts/python/instance_seg/instance_seg_comparison.py`

Typical semantic toy-only pattern:

```bash
python -u scripts/python/semantic_seg/semantic_seg_comparison.py --skip-graha-fit ...
```

Typical instance toy-only pattern:

```bash
python -u scripts/python/instance_seg/instance_seg_comparison.py --skip-graha-fit ...
```

### Comparison Without Full Training

These workflows compare existing outputs/checkpoints.

- Semantic checkpoint sweep:
  `scripts/shell/semantic_seg/sbatch_semantic_checkpoint_sweep.sh`
- Instance checkpoint sweep:
  `scripts/shell/instance_seg/sbatch_instance_checkpoint_sweep.sh`
- Semantic metrics notebook:
  `notebooks/full_model/sem_seg_test_metrics.ipynb`
- Semantic comparison notebook:
  `notebooks/full_model/semantic_seg_comparison.ipynb`
- Instance comparison notebook:
  `notebooks/full_model/instance_seg_comparison.ipynb`

### Publishing Outputs

- Semantic experiment publishing helper:
  `scripts/shell/semantic_seg/sbatch_publish_sem_seg_experiment.sh`

This copies a selected experiment output directory to the project experiment space and applies recursive permissions/group changes.

## HPC Notes

- Shell scripts must use LF endings. `.gitattributes` enforces `*.sh text eol=lf`.
- Prefer output directories under `/explore/nobackup/...`, not home.
- The `/explore/nobackup/...` path may resolve to `/panfs/ccds02/nobackup/...` in logs.
- DINO/Torch cache may use `~/.cache/torch`; Hugging Face cache may use `~/.cache/huggingface`.
- Comparison checkpoints should use `save_weights_only=True` and `save_last=False` to avoid huge Lightning checkpoints.
- Do not commit generated outputs, checkpoints, logs, copied datasets, or experiment result folders.

## Experiment Defaults

- Semantic comparison can use instance-derived semantic labels with `--semantic-label-source instance`.
- WAC inputs generally use the first 7 bands.
- Graha WAC mode should usually be `--graha-wac-mode vis-uv --graha-vis-uv-merge-method mean`.
- TerraMind/Graha pretraining normalization uses `--normalization-source pretrain`.
- DINO/train-split normalization uses `--normalization-source finetune`.
- Spatial semantic loss is additive, not a replacement for Dice:
  `total_loss = dice_loss + weight * spatial_shape_loss`.
- Current spatial-loss comparison default uses weight `0.05` and pad fraction `0.3`.

## Docs To Read

- `docs/ibm_model_commit_change_log.md`: detailed commit-by-commit branch history.
- `docs/ibm_model_branch_technical_notes.md`: technical context and implementation notes.
- `docs/model_comparison_plan.txt`: semantic comparison plan and status.
- `docs/instance_seg_plan.txt`: instance segmentation plan and status.
- `docs/repo_cleanup_plan.txt`: repo organization and cleanup plan.
- `docs/full_model_complexity_refactor_plan.txt`: complexity-reduction plan for large files/workflows.

## Working Rules

- Do not generate code unless the user prompts with "start the refactor", "generate code for x", or something similar.
- Preserve notebook organization when changing notebook behavior.
- Keep semantic and instance workflows aligned where practical.
- Prefer task-specific modules/scripts over adding more logic directly into notebooks.
- Use `scripts/python/...` for reusable workflow logic and `scripts/shell/...` for sbatch wrappers.
- Keep shell scripts Linux-safe and verify new/edited `.sh` files do not contain CRLF line endings.
- For diagnostic/audit tasks over many files or samples, use progress bars and multiprocessing where appropriate. Prefer `ProcessPoolExecutor` for CPU-bound/file-heavy per-sample work, `ThreadPoolExecutor` for lightweight I/O-bound checks, and plain `tqdm` rather than widget-backed progress bars in HPC/JupyterHub notebooks.
- Use focused commits and avoid mixing generated experiment artifacts with code/doc changes.
