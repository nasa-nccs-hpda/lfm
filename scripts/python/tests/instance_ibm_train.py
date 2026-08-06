import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

import sys

from functools import partialmethod
from glob import glob
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch
from lightning.pytorch import seed_everything
from tqdm import tqdm

tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)

repo_root = Path.cwd().parent
NOTEBOOK_DIR = repo_root / "notebooks"

if not (repo_root / "lfm").exists():
  raise FileNotFoundError(
      "Cannot find lfm/ directory. Run this notebook from "
      "lfm/notebooks/full_model or update repo_root."
  )

sys.path.insert(0, str(repo_root))

from lfm.all_models.all_tasks.utils import (
  create_timestamped_output_dir,
  plot_instance_cache_predictions,
  save_graha_instance_prediction_cache,
)
from lfm.all_models.inst_seg import build_graha_notebook_configs
from lfm.full_model.inst_seg import instance_graha_components

print("Successfully imported LFM modules")


BASE_OUTPUT_DIR = NOTEBOOK_DIR / "outputs" / "instance_seg_finetuning"  # Base output directory for finetune plots etc.
PRETRAIN_DIR = "/explore/nobackup/projects/lfm/ibm_model_pretrain_dir"  # Where to load Graha configuration/checkpoint from
LIGHTNING_CHECKPOINT = None  # Fine-tuned checkpoint to resume from (fresh starts should use 'None')

# Data dictionary; see README under "Dataset Specifications" for examples
DATA_DICT = {
    "dataset_name": "wac_craters",  # Human-readable dataset name (has no functional effect)
    "data_dir": "/explore/nobackup/projects/lfm/model_inputs/300_300_inputs/full_model_inst_seg_v2",  # Data directory on /explore
    "dataset_modality": "wac",  # Dataset modality
    "image_glob": "*.tif",  # Image (chip) filename pattern the dataset will look for
    "label_glob": "*_label.npz",  # Label (mask) filename pattern the dataset will look for
    "band_filters": {  # Band filters for each modality
        "vis": [0, 1, 2, 3, 4],  # Vis channels to use (0-indexed)
        "uv": [0, 1],  # UV channels to use (0-indexed)
    },
}

# Upper limit for sample counts in train/val/test datasets; if less samples are found, that amount will be used instead
MAX_TRAIN_SAMPLES = 500
MAX_VAL_SAMPLES = 500
MAX_TEST_SAMPLES = 500

BATCH_SIZE = 8  # Number of inputs fed into the model per iteration; often a multiple of 8 for parallelization purposes
MAX_EPOCHS = 1  # Maximum number of epochs, default to 1 for demo purposes. Finetuning typically uses 50-100 epochs, but sometimes less can work.

GRAHA_BACKBONE_LR = 5.0e-5  # Learning rate for Graha FM backbone (lower than head typically)
GRAHA_HEAD_LR = 2.0e-4  # Learning rate for task decoder (higher than backbone typically)
GRAHA_LAYER_DECAY = 0.75  # Decays learning rate further toward the backbone; allows for more gentle tuning of Graha backbone weights
GRAHA_WEIGHT_DECAY = 0.05  # Penalizes large model weights during training, helping model generalize to non-training data
GRAHA_WARMUP_STEPS = 500  # Number of optimizer steps before LR scheduler warmup ends
GRAHA_FREEZE_BACKBONE = False  # Whether to freeze Graha backbone (can be a useful tool during training to change to True/False)

OUTPUT_DIR = create_timestamped_output_dir(BASE_OUTPUT_DIR)

notebook_configs = build_graha_notebook_configs(
    output_dir=OUTPUT_DIR,
    base_output_dir=OUTPUT_DIR,
    graha_pretrain_dir=PRETRAIN_DIR,
    graha_lightning_checkpoint=LIGHTNING_CHECKPOINT,
    data_dict=DATA_DICT,
    max_epochs=MAX_EPOCHS,
    graha_batch_size=BATCH_SIZE,
    max_train_samples=MAX_TRAIN_SAMPLES,
    max_val_samples=MAX_VAL_SAMPLES,
    max_test_samples=MAX_TEST_SAMPLES,
    graha_backbone_lr=GRAHA_BACKBONE_LR,
    graha_head_lr=GRAHA_HEAD_LR,
    graha_layer_decay=GRAHA_LAYER_DECAY,
    graha_weight_decay=GRAHA_WEIGHT_DECAY,
    graha_warmup_steps=GRAHA_WARMUP_STEPS,
    graha_freeze_backbone=GRAHA_FREEZE_BACKBONE,
)

config = notebook_configs.experiment_config
graha_config = notebook_configs.graha_config
deps = notebook_configs.dependencies

seed_everything(config.seed)
instance_graha_components.save_config(graha_config, OUTPUT_DIR)

print("Config created successfully")
print(f"Data dir: {config.data_root}")
print(f"Output dir: {OUTPUT_DIR}")
print(f"Graha modality mode: {config.graha_input_modality_mode}")
print(f"Normalization modality: {config.normalization_modality}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print(f"Notebook output directory: {OUTPUT_DIR}")

# STEP 1: pretraining stats
print("\nSTEP 1: Loading pretraining stats...")
print("="*60)

datamodule_cls = deps["GrahaObjectDetectionInstanceDataModule"]
means, stds = instance_graha_components.get_normalization_stats(
    graha_config,
    datamodule_cls,
)

print("Done.")

print("\nSTEP 2: Creating datamodule and inspecting one training batch...")
print("=" * 60)

graha_datamodule = instance_graha_components.create_datamodule(
    graha_config,
    datamodule_cls,
    means,
    stds,
)
graha_sample_batch = instance_graha_components.inspect_batch(graha_datamodule)

print("Done.")


task_cls = instance_graha_components.make_downstream_object_detection_task_class(
    deps["LunarObjectDetectionTask"]
)

graha_task = instance_graha_components.create_task(
    graha_config,
    task_cls,
    graha_sample_batch,
)
instance_graha_components.run_loss_smoke(graha_task, graha_sample_batch)


trainer = instance_graha_components.create_trainer(graha_config, OUTPUT_DIR)


print("\n" + "=" * 60)
print("Starting training.")
print("=" * 60)

ckpt_path = (
    str(graha_config.lightning_checkpoint)
    if graha_config.lightning_checkpoint is not None
    else None
)
trainer.fit(
    graha_task,
    datamodule=graha_datamodule,
    ckpt_path=ckpt_path,
)

print("Finished training.")


prediction_cache = save_graha_instance_prediction_cache(
    task=graha_task,
    datamodule=graha_datamodule,
    output_dir=OUTPUT_DIR,
    model_name="graha",
    split=config.prediction_split,
    n_samples=config.prediction_n_samples,
    score_threshold=config.prediction_score_threshold,
)

prediction_plot = plot_instance_cache_predictions(
    prediction_cache,
    OUTPUT_DIR / "plots" / "single_model" / "graha_model",
    model_name="graha",
    n_samples=config.prediction_n_samples,
    filename=f"{config.prediction_split}_instance_predictions.png",
)
print(f"Saved prediction plot: {prediction_plot}")

img = mpimg.imread(prediction_plot)
plt.figure(figsize=(16, 14))
plt.imshow(img)
plt.axis("off")
plt.show()

del graha_task, graha_datamodule, trainer
if torch.cuda.is_available():
    torch.cuda.empty_cache()

