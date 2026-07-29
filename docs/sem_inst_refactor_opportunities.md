# Semantic/Instance Refactor Opportunities

Purpose
-------

This document records code that looks messy, duplicated, or unnecessarily
inconsistent between the semantic and instance segmentation workflows.

Scope inspected:

- `scripts/python/semantic_seg/semantic_seg_comparison.py`
- `scripts/python/instance_seg/instance_seg_comparison.py`
- `scripts/python/semantic_seg/semantic_checkpoint_sweep.py`
- `scripts/python/instance_seg/instance_checkpoint_sweep.py`
- `lfm/full_model/sem_seg/semantic_seg_finetuning.py`
- `lfm/full_model/inst_seg/instance_seg_finetuning.py`
- `lfm/full_model/all_tasks/utils/plot_utils.py`
- `lfm/full_model/all_tasks/datamodules/*`
- `lfm/full_model/sem_seg/*`
- `lfm/full_model/inst_seg/*`
- `lfm/toy_model/sem_seg/lightning_wrappers/*`
- `lfm/toy_model/inst_seg/lightning_wrappers/*`


Highest-Priority Findings
-------------------------

### 1. Comparison scripts are still workflow implementations

Files:

- `scripts/python/semantic_seg/semantic_seg_comparison.py`
- `scripts/python/instance_seg/instance_seg_comparison.py`

Why they fit:

- Messy/complex: both scripts contain configuration dataclasses, path
  validation, checkpoint loading, trainer construction, callbacks, model setup,
  datamodule setup, training, cache generation, plotting, and final comparison.
- Duplicated/OOP candidate: both scripts have `FitProgressLogger`,
  `save_config`, `load_lightning_checkpoint_state`, toy/Graha runner functions,
  and similar main orchestration.
- Sem/inst consistency issue: semantic comparison uses timing rows and a
  `save_timing_summary`; instance comparison does not have an equivalent
  timing summary even though the workflow shape is the same. Semantic
  comparison makes prediction caching optional with `--cache-predictions`;
  instance comparison appears to always create caches as part of its plotting
  path. That may be intentional for instance plots, but the CLI contract should
  be explicit and aligned.

Suggested extraction:

- Keep scripts as `parse_args()`, `build_config()`, and `main()`.
- Move reusable behavior into:
  - `lfm/full_model/sem_seg/comparison.py`
  - `lfm/full_model/inst_seg/comparison.py`
  - `lfm/full_model/all_tasks/workflows.py`

Possible OOP shape:

```python
class ComparisonWorkflow(ABC):
    config: BaseComparisonConfig
    def validate(self) -> None: ...
    def run_toy(self) -> PredictionCache | None: ...
    def run_graha(self) -> PredictionCache | None: ...
    def compare(self, caches: dict[str, PredictionCache]) -> None: ...
```

`SemanticComparisonWorkflow` and `InstanceComparisonWorkflow` should override
only the task-specific setup and plotting/cache codec behavior.


### 2. Fine-tuning modules are aligned at the boundary but duplicated inside

Files:

- `lfm/full_model/sem_seg/semantic_seg_finetuning.py`
- `lfm/full_model/inst_seg/instance_seg_finetuning.py`

Why they fit:

- Messy/complex: both modules are roughly 600 lines and still mix environment
  setup, config defaults, dependency import timing, datamodule setup, train-stat
  calculation, Graha modality construction, task creation, trainer creation,
  checkpoint loading, plotting/cache behavior, and CLI parsing.
- Duplicated/OOP candidate: both define similar versions of
  `FitProgressLogger`, `configure_proj_environment`,
  `configure_python_paths`, `validate_required_paths`,
  `common_datamodule_args`, `calculate_train_stats`,
  `get_normalization_stats`, `_graha_modality_args`, `create_trainer`, and
  `load_lightning_checkpoint_state`.
- Sem/inst consistency issue: instance fine-tuning writes `config.json` and
  `timing_summary.txt`; semantic fine-tuning has no equivalent direct
  `save_config`/`save_timing` in the module. Semantic fine-tuning uses
  `cache_predictions`, while instance fine-tuning uses `plot_predictions`.
  The output products differ, but the user-facing lifecycle should be parallel:
  save config, train or load checkpoint, produce optional predictions/cache,
  save timing.

Suggested extraction:

- `lfm/full_model/all_tasks/environment.py`
  - PROJ/GDAL setup
  - Python path setup for Graha code
- `lfm/full_model/all_tasks/config_io.py`
  - dataclass-to-JSON encoding
  - timing summaries
- `lfm/full_model/all_tasks/graha.py`
  - WAC modality argument construction
  - normalization-source dispatch
  - checkpoint loading
- `lfm/full_model/all_tasks/training.py`
  - shared progress callback
  - ModelCheckpoint factory
  - Trainer factory

Possible OOP shape:

```python
class GrahaFineTuningWorkflow(ABC):
    config: BaseFineTuningConfig
    datamodule_cls: type
    task_cls: type

    def run(self) -> Path: ...
    def create_datamodule(self): ...
    def create_task(self, sample_batch): ...
    def create_trainer(self, output_dir: Path): ...
    def write_predictions(self, output_dir: Path) -> None: ...
```

The base class should own the lifecycle. Semantic and instance subclasses
should own the different datamodule/task/prediction details.


### 3. Checkpoint sweeps duplicate framework mechanics

Files:

- `scripts/python/semantic_seg/semantic_checkpoint_sweep.py`
- `scripts/python/instance_seg/instance_checkpoint_sweep.py`

Why they fit:

- Messy/complex: both scripts are large single-file workflows containing CLI
  config, checkpoint discovery, checkpoint loading, metric computation, summary
  writing, toy setup, Graha setup, and sweep orchestration.
- Duplicated/OOP candidate: both define `CheckpointRecord`,
  `discover_checkpoints`, `_parse_epoch`, `_checkpoint_output_name`,
  `_load_lightning_checkpoint_state`, `_metrics_to_array`, `_write_metrics`,
  `_write_model_summary`, `run_toy_sweep`, `run_graha_sweep`, and `run_sweep`.
- Sem/inst consistency issue: semantic sweep has batch preloading helpers and
  computes semantic AP/F1-style metrics directly in the sweep script. Instance
  sweep computes instance AP and also derives semantic-style binary F1 from
  instance masks. The metrics are task-specific, but checkpoint discovery,
  output layout, summary writing, and model iteration should not diverge.

Suggested extraction:

- `lfm/full_model/all_tasks/checkpoints.py`
  - `CheckpointRecord`
  - discovery
  - epoch parsing
  - Lightning state loading
- `lfm/full_model/all_tasks/sweeps.py`
  - `CheckpointSweepRunner`
  - model loop
  - summary writing
- `lfm/full_model/sem_seg/checkpoint_sweep.py`
  - semantic prediction/metric evaluator
- `lfm/full_model/inst_seg/checkpoint_sweep.py`
  - instance prediction/metric evaluator

Possible OOP shape:

```python
class CheckpointEvaluator(Protocol):
    def setup_model(self, model_name: str): ...
    def evaluate_checkpoint(self, checkpoint: CheckpointRecord) -> dict[str, float]: ...

class CheckpointSweepRunner:
    def run_model(self, model_name: str, checkpoint_dir: Path) -> list[dict]: ...
```


Medium-Priority Findings
------------------------

### 4. Toy and full-model datamodules share ideas but not a common contract

Files:

- `lfm/full_model/all_tasks/datamodules/lunar_segmentation_dataset.py`
- `lfm/full_model/all_tasks/datamodules/lunar_segmentation_datamodule.py`
- `lfm/toy_model/sem_seg/lightning_wrappers/toy_sem_seg_datamodule.py`
- `lfm/toy_model/sem_seg/lightning_wrappers/toy_sem_seg_from_instance_datamodule.py`
- `lfm/toy_model/inst_seg/lightning_wrappers/toy_instance_seg_datamodule.py`
- `lfm/toy_model/inst_seg/lightning_wrappers/toy_dino_mask_rcnn_datamodule.py`

Why they fit:

- Duplicated/OOP candidate: the toy and full-model paths repeat split dataset
  construction, pair-record discovery, optional sample limits, normalization,
  center crop, mask shifting, train-stat calculation, and dataloader setup.
- Sem/inst consistency issue: full-model semantic and instance datamodules are
  now subclasses of a shared `LunarSegmentationDatamodule`, but toy semantic and
  toy instance datamodules still implement their own parallel versions. Some of
  that is architectural, but pair matching and preprocessing should be common.

Suggested extraction:

- A shared `LunarSegmentationDataModule` base with hooks:
  - `dataset_cls`
  - `collate_fn`
  - `make_targets(sample)`
  - `stats_dataset(split)`
- A shared paired-chip dataset base that can be reused by toy and full-model
  tasks, with task-specific target encoders:
  - `SemanticTargetEncoder`
  - `Mask2FormerInstanceTargetEncoder`
  - `ObjectDetectionTargetEncoder`

This should reduce drift in normalization, crop behavior, mask shifting, and
sample limiting.


### 5. Callback classes are duplicated and unevenly located

Files:

- `scripts/python/semantic_seg/semantic_seg_comparison.py`
- `scripts/python/instance_seg/instance_seg_comparison.py`
- `lfm/full_model/sem_seg/semantic_seg_finetuning.py`
- `lfm/full_model/inst_seg/instance_seg_finetuning.py`
- `lfm/full_model/all_tasks/utils/plot_utils.py`

Why it fits:

- Duplicated/OOP candidate: `FitProgressLogger` is defined in multiple files.
  Semantic and instance comparison also define task-specific epoch test-suite
  callbacks in scripts.
- Sem/inst consistency issue: progress logging should behave identically
  across tasks. Plot/test-suite callbacks can be task-specific but should share
  scheduling, output naming, and cache writing contracts.

Suggested extraction:

- `lfm/full_model/all_tasks/callbacks.py`
  - `FitProgressLogger`
  - `PeriodicPredictionCallback`
  - `PeriodicCheckpointEvaluatorCallback`

Task-specific callbacks should pass task adapters instead of reimplementing the
Lightning hook lifecycle.


Lower-Priority Findings
-----------------------

### 6. Legacy toy drivers remain large and parallel

Files:

- `lfm/toy_model/sem_seg/sseg_driver.py`
- `lfm/toy_model/inst_seg/iseg_driver.py`
- `lfm/toy_model/sem_seg/data_cube_inference.py`
- `lfm/toy_model/inst_seg/iseg_dataset.py`

Why it fits:

- Messy/complex: these files are large older task-specific drivers/datasets.
- Duplicated/OOP candidate: some plotting, metrics, prediction, and dataset
  behavior overlaps with active full-model comparison code.
- Sem/inst consistency issue: because these are older drivers, they may drift
  from the newer Lightning-wrapper workflows.

Suggested approach:

- Do not refactor first unless they are still active entrypoints.
- Mark them as legacy/reference or extract only the pieces that active
  workflows still need.


Recommended Refactor Order
--------------------------

1. Create shared all-task modules for callbacks, checkpoint helpers, config I/O,
   timing, and Graha modality/environment setup.
2. Move semantic and instance comparison implementations from scripts into
   package workflow classes.
3. Move checkpoint sweep mechanics into a shared sweep runner with
   task-specific evaluator classes.
4. Normalize fine-tuning lifecycle behavior so both semantic and instance
   save config, save timing, and expose prediction/cache options consistently.
5. Revisit toy/full datamodule inheritance after workflow extraction, because
   datamodule changes have a larger blast radius.


Consistency Rules To Preserve
-----------------------------

- Semantic and instance task differences should live in target encoders,
  prediction adapters, metric evaluators, and plotting adapters.
- Environment setup, path validation, checkpoint discovery/loading, Trainer
  factories, output directories, config serialization, timing summaries, and
  cache manifests should be shared.
- CLI wrappers should remain thin and should not contain task lifecycle logic.
- Existing public imports should be preserved through compatibility re-exports
  during the refactor.
