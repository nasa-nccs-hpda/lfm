"""Create instance comparison plots from final Toy/Graha checkpoints.

This post-training entrypoint loads two or three completed model checkpoints,
writes prediction caches on a shared split, and creates one all-model plot plus
all pairwise comparison plots. It is intended to run after parallel or serial
fine-tuning jobs have completed.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from lightning.pytorch import seed_everything

LFM_ROOT = Path(__file__).resolve().parents[3]
if str(LFM_ROOT) not in sys.path:
    sys.path.insert(0, str(LFM_ROOT))

import instance_seg_comparison as comparison_workflow
from lfm.all_models.all_tasks import (
    CheckpointRecord,
    discover_checkpoints,
    load_lightning_checkpoint_state,
)
from lfm.all_models.inst_seg.config import build_config_from_args
from lfm.all_models.all_tasks.cli_args import (
    parse_instance_checkpoint_comparison_plot_args,
)
from lfm.all_models.all_tasks.utils import (
    plot_instance_cache_comparison,
    save_graha_instance_prediction_cache,
    save_toy_instance_prediction_cache,
)


@dataclass(frozen=True)
class ModelPlotSpec:
    key: str
    display_name: str
    model_family: str
    toy_architecture: str | None
    checkpoint_path: Path


def _final_checkpoint_from_dir(checkpoint_dir: Path) -> CheckpointRecord:
    checkpoints = discover_checkpoints(checkpoint_dir)
    return checkpoints[-1]


def _resolve_checkpoint(
    *,
    checkpoint_path: Path | None,
    checkpoint_dir: Path | None,
    label: str,
) -> Path | None:
    if checkpoint_path is not None and checkpoint_dir is not None:
        raise ValueError(
            f"Pass either --{label}-checkpoint or --{label}-checkpoint-dir, not both."
        )
    if checkpoint_path is not None:
        checkpoint_path = checkpoint_path.resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"{label} checkpoint does not exist: {checkpoint_path}"
            )
        return checkpoint_path
    if checkpoint_dir is None:
        return None
    record = _final_checkpoint_from_dir(checkpoint_dir)
    print(f"[{label}] selected final checkpoint: {record.path}", flush=True)
    return record.path


def _apply_run_root_defaults(args: argparse.Namespace) -> None:
    if args.run_root is None:
        return
    run_root = args.run_root.resolve()
    if args.mask2former_checkpoint_dir is None and args.mask2former_checkpoint is None:
        args.mask2former_checkpoint_dir = (
            run_root / "toy_mask2former" / "checkpoints" / "toy_model"
        )
    if (
        args.toy_terratorch_checkpoint_dir is None
        and args.toy_terratorch_checkpoint is None
    ):
        args.toy_terratorch_checkpoint_dir = (
            run_root / "toy_dino_terratorch_mask_rcnn" / "checkpoints" / "toy_model"
        )
    if args.graha_checkpoint_dir is None and args.graha_checkpoint is None:
        args.graha_checkpoint_dir = (
            run_root / "graha_mask_rcnn" / "checkpoints" / "full_model"
        )


def build_model_specs(args: argparse.Namespace) -> list[ModelPlotSpec]:
    _apply_run_root_defaults(args)
    specs = []
    mask2former = _resolve_checkpoint(
        checkpoint_path=args.mask2former_checkpoint,
        checkpoint_dir=args.mask2former_checkpoint_dir,
        label="mask2former",
    )
    if mask2former is not None:
        specs.append(
            ModelPlotSpec(
                key="toy_mask2former",
                display_name="Toy Mask2Former",
                model_family="toy",
                toy_architecture="mask2former",
                checkpoint_path=mask2former,
            )
        )

    toy_terratorch = _resolve_checkpoint(
        checkpoint_path=args.toy_terratorch_checkpoint,
        checkpoint_dir=args.toy_terratorch_checkpoint_dir,
        label="toy-terratorch",
    )
    if toy_terratorch is not None:
        specs.append(
            ModelPlotSpec(
                key="toy_dino_terratorch_mask_rcnn",
                display_name="Toy DINO TerraTorch Mask R-CNN",
                model_family="toy",
                toy_architecture="dino-terratorch-mask-rcnn",
                checkpoint_path=toy_terratorch,
            )
        )

    graha = _resolve_checkpoint(
        checkpoint_path=args.graha_checkpoint,
        checkpoint_dir=args.graha_checkpoint_dir,
        label="graha",
    )
    if graha is not None:
        specs.append(
            ModelPlotSpec(
                key="graha_mask_rcnn",
                display_name="Graha Mask R-CNN",
                model_family="graha",
                toy_architecture=None,
                checkpoint_path=graha,
            )
        )

    if len(specs) < 2:
        raise ValueError("At least two model checkpoints are required for comparison.")
    return specs


def _comparison_namespace(
    args: argparse.Namespace, toy_architecture: str
) -> argparse.Namespace:
    max_train = args.max_samples if args.prediction_split == "train" else None
    max_val = args.max_samples if args.prediction_split == "val" else None
    max_test = args.max_samples if args.prediction_split == "test" else None
    return argparse.Namespace(
        data_root=str(args.data_root),
        base_output_dir=str(args.output_dir / "_setup"),
        dino_checkpoint=str(args.dino_checkpoint) if args.dino_checkpoint else None,
        toy_lightning_checkpoint=None,
        graha_pretrain_dir=(
            str(args.graha_pretrain_dir) if args.graha_pretrain_dir else None
        ),
        graha_lightning_checkpoint=None,
        graha_input_modality_mode=args.graha_input_modality_mode,
        graha_vis_uv_merge_method=args.graha_vis_uv_merge_method,
        normalization_source=args.normalization_source,
        normalization_modality=args.normalization_modality,
        image_glob=args.image_glob,
        label_glob=args.label_glob,
        image_suffix=args.image_suffix,
        label_suffix=args.label_suffix,
        toy_architecture=toy_architecture,
        target_size=args.target_size,
        band_filter=args.band_filter,
        max_train_samples=max_train,
        max_val_samples=max_val,
        max_test_samples=max_test,
        toy_batch_size=args.batch_size,
        toy_num_workers=args.num_workers,
        graha_stats_batch_size=args.graha_stats_batch_size,
        graha_batch_size=args.graha_batch_size,
        graha_num_workers=args.graha_num_workers,
        max_epochs=1,
        toy_learning_rate=5.0e-5,
        toy_weight_decay=1.0e-3,
        toy_freeze_backbone=False,
        toy_normalize_inputs=args.normalize_inputs,
        toy_gradient_clip_val=1.0,
        disable_toy_gradient_clipping=True,
        graha_backbone_lr=args.graha_backbone_lr,
        graha_head_lr=args.graha_head_lr,
        graha_layer_decay=args.graha_layer_decay,
        graha_weight_decay=args.graha_weight_decay,
        graha_warmup_steps=args.graha_warmup_steps,
        graha_anchor_sizes=args.graha_anchor_sizes,
        graha_anchor_aspect_ratios=args.graha_anchor_aspect_ratios,
        graha_score_threshold=args.graha_score_threshold,
        plot_every_n_epochs=0,
        plot_n_samples=args.n_samples,
        progress_log_every_n_batches=10**9,
        prediction_split=args.prediction_split,
        prediction_n_samples=args.n_samples,
        prediction_score_threshold=args.score_threshold,
        mask_shift=tuple(args.mask_shift),
        ignore_nodata_in_loss=args.ignore_nodata_in_loss,
        nodata_ignore_index=args.nodata_ignore_index,
        skip_toy_fit=True,
        skip_graha_fit=True,
        no_fit=True,
        run_epoch_test_suite=False,
        epoch_test_split=args.prediction_split,
        epoch_test_n_samples=args.n_samples,
        epoch_test_every_n_epochs=1,
        seed=args.seed,
    )


def _setup_toy(
    args: argparse.Namespace,
    spec: ModelPlotSpec,
):
    from lfm.toy_model.inst_seg.instance_model_adapter import ToyInstanceModelAdapter

    toy_adapter = ToyInstanceModelAdapter()
    comparison_config = build_config_from_args(
        _comparison_namespace(args, spec.toy_architecture or "mask2former")
    )
    datamodule = toy_adapter.create_datamodule(
        comparison_config,
        normalization_modality_info=comparison_workflow.get_toy_normalization_modality_info(
            comparison_config
        ),
    )
    datamodule.setup(args.prediction_split)
    task = toy_adapter.create_model_or_task(comparison_config, datamodule)
    image_processor = toy_adapter.create_image_processor(comparison_config)
    load_lightning_checkpoint_state(task, spec.checkpoint_path)
    task.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return datamodule, task, image_processor


def _setup_graha(args: argparse.Namespace):
    from lfm.full_model.inst_seg.instance_model_adapter import GrahaInstanceModelAdapter

    graha_adapter = GrahaInstanceModelAdapter()
    comparison_config = build_config_from_args(
        _comparison_namespace(args, "mask2former")
    )
    graha_config = graha_adapter.build_comparison_config(
        comparison_config,
        args.output_dir / "_graha_setup",
    )
    graha_adapter.configure_environment()
    graha_adapter.configure_python_paths(graha_config)
    graha_adapter.validate_required_paths(graha_config)
    deps = graha_adapter.import_project_dependencies()
    datamodule_cls = deps["GrahaObjectDetectionInstanceDataModule"]
    task_cls = graha_adapter.make_task_class(deps["LunarObjectDetectionTask"])
    means, stds = graha_adapter.get_normalization_stats(graha_config, datamodule_cls)
    datamodule = graha_adapter.create_datamodule(
        graha_config,
        datamodule_cls,
        means,
        stds,
    )
    datamodule.setup(args.prediction_split)
    task = graha_adapter.create_model_or_task(graha_config, datamodule, task_cls)
    load_lightning_checkpoint_state(task, args._active_checkpoint_path)
    task.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return datamodule, task, None


def write_prediction_cache(
    args: argparse.Namespace,
    spec: ModelPlotSpec,
) -> Path:
    print(f"[{spec.key}] loading checkpoint: {spec.checkpoint_path}", flush=True)
    if spec.model_family == "toy":
        datamodule, task, image_processor = _setup_toy(args, spec)
    elif spec.model_family == "graha":
        args._active_checkpoint_path = spec.checkpoint_path
        datamodule, task, image_processor = _setup_graha(args)
    else:
        raise ValueError(f"Unknown model family: {spec.model_family}")

    cache_output_dir = args.output_dir / "cache_sources" / spec.key
    if image_processor is None:
        cache_dir = save_graha_instance_prediction_cache(
            task=task,
            datamodule=datamodule,
            output_dir=cache_output_dir,
            model_name=spec.key,
            split=args.prediction_split,
            n_samples=args.n_samples,
            score_threshold=args.score_threshold,
        )
    else:
        cache_dir = save_toy_instance_prediction_cache(
            task=task,
            datamodule=datamodule,
            output_dir=cache_output_dir,
            image_processor=image_processor,
            model_name=spec.key,
            split=args.prediction_split,
            n_samples=args.n_samples,
            score_threshold=args.score_threshold,
        )

    del task, datamodule
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cache_dir


def create_comparison_plots(
    *,
    cache_dirs: dict[str, Path],
    output_dir: Path,
    n_samples: int,
) -> dict[str, str]:
    plots_dir = output_dir / "plots"
    outputs = {}
    all_plot = plot_instance_cache_comparison(
        cache_dirs,
        plots_dir,
        n_samples=n_samples,
        filename="all_models_instance_predictions.png",
    )
    outputs["all_models"] = str(all_plot)

    for left, right in itertools.combinations(cache_dirs, 2):
        filename = f"{left}_vs_{right}_instance_predictions.png"
        path = plot_instance_cache_comparison(
            {left: cache_dirs[left], right: cache_dirs[right]},
            plots_dir / "pairwise",
            n_samples=n_samples,
            filename=filename,
        )
        outputs[f"{left}_vs_{right}"] = str(path)
    return outputs


def parse_args() -> argparse.Namespace:
    return parse_instance_checkpoint_comparison_plot_args(description=__doc__)


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    specs = build_model_specs(args)
    cache_dirs = {spec.key: write_prediction_cache(args, spec) for spec in specs}
    plots = create_comparison_plots(
        cache_dirs=cache_dirs,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
    )
    manifest = {
        "models": [
            {
                "key": spec.key,
                "display_name": spec.display_name,
                "model_family": spec.model_family,
                "toy_architecture": spec.toy_architecture,
                "checkpoint_path": str(spec.checkpoint_path),
                "cache_dir": str(cache_dirs[spec.key]),
            }
            for spec in specs
        ],
        "plots": plots,
    }
    with (args.output_dir / "comparison_plot_manifest.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(manifest, f, indent=2)
    print(
        f"Wrote comparison plot manifest to {args.output_dir / 'comparison_plot_manifest.json'}"
    )


if __name__ == "__main__":
    main()
