"""Create instance comparison plots from final Toy/Graha/GFFT checkpoints.

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
from pathlib import Path

import torch
from lightning.pytorch import seed_everything

LFM_ROOT = Path(__file__).resolve().parents[3]
if str(LFM_ROOT) not in sys.path:
    sys.path.insert(0, str(LFM_ROOT))

import instance_seg_comparison as comparison_workflow
from lfm.all_models.all_tasks import (
    load_lightning_checkpoint_state,
)
from lfm.all_models.all_tasks.metrics_comparison import (
    FinalEpochMetricSpec,
    write_final_epoch_metric_comparisons,
)
from lfm.all_models.inst_seg.config import build_config_from_args
from lfm.all_models.inst_seg.plot_config import (
    InstanceCheckpointComparisonPlotConfig,
    ModelPlotSpec,
    build_checkpoint_comparison_plot_config_from_args,
)
from lfm.all_models.all_tasks.cli_args import (
    parse_instance_checkpoint_comparison_plot_args,
)
from lfm.all_models.all_tasks.utils import (
    plot_instance_cache_comparison,
    save_graha_instance_prediction_cache,
    save_toy_instance_prediction_cache,
)


def _comparison_namespace(
    args: InstanceCheckpointComparisonPlotConfig, toy_architecture: str
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
        gfft_config_path=(
            str(args.gfft_config_path) if args.gfft_config_path is not None else None
        ),
        gfft_backbone_checkpoint=(
            str(args.gfft_backbone_checkpoint)
            if args.gfft_backbone_checkpoint is not None
            else None
        ),
        dataset_modality=args.dataset_modality,
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
        excluded_nodata_values=args.excluded_nodata_values,
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
    args: InstanceCheckpointComparisonPlotConfig,
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


def _setup_graha(
    args: InstanceCheckpointComparisonPlotConfig,
    spec: ModelPlotSpec,
):
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
    load_lightning_checkpoint_state(task, spec.checkpoint_path)
    task.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return datamodule, task, None


def _setup_gfft(
    args: InstanceCheckpointComparisonPlotConfig,
    spec: ModelPlotSpec,
):
    from lfm.full_model.inst_seg.instance_gfft_model_adapter import (
        GfftInstanceModelAdapter,
    )

    gfft_adapter = GfftInstanceModelAdapter()
    comparison_config = build_config_from_args(
        _comparison_namespace(args, "mask2former")
    )
    gfft_config = gfft_adapter.build_finetuning_config(
        comparison_config,
        args.output_dir / "_gfft_setup",
    )
    gfft_adapter.configure_environment()
    gfft_adapter.configure_python_paths(gfft_config)
    gfft_adapter.validate_required_paths(gfft_config)
    deps = gfft_adapter.import_project_dependencies()
    datamodule_cls = deps["GrahaObjectDetectionInstanceDataModule"]
    task_cls = gfft_adapter.make_task_class(deps["LunarObjectDetectionTask"])
    means, stds = gfft_adapter.get_normalization_stats(gfft_config, datamodule_cls)
    datamodule = gfft_adapter.create_datamodule(
        gfft_config,
        datamodule_cls,
        means,
        stds,
    )
    datamodule.setup(args.prediction_split)
    task = gfft_adapter.create_model_or_task(gfft_config, datamodule, task_cls)
    load_lightning_checkpoint_state(task, spec.checkpoint_path)
    task.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return datamodule, task, None


def write_prediction_cache(
    args: InstanceCheckpointComparisonPlotConfig,
    spec: ModelPlotSpec,
) -> Path:
    print(f"[{spec.key}] loading checkpoint: {spec.checkpoint_path}", flush=True)
    if spec.model_family == "toy":
        datamodule, task, image_processor = _setup_toy(args, spec)
    elif spec.model_family == "graha":
        datamodule, task, image_processor = _setup_graha(args, spec)
    elif spec.model_family == "gfft":
        datamodule, task, image_processor = _setup_gfft(args, spec)
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


def _test_suite_model_names(spec: ModelPlotSpec) -> tuple[str, ...]:
    if spec.model_family == "toy":
        return ("toy_model", "toy")
    if spec.model_family == "graha":
        return ("full_model", "graha")
    if spec.model_family == "gfft":
        return ("gfft",)
    raise ValueError(f"Unknown model family: {spec.model_family}")


def create_metric_comparisons(
    config: InstanceCheckpointComparisonPlotConfig,
) -> dict[str, object]:
    return write_final_epoch_metric_comparisons(
        [
            FinalEpochMetricSpec(
                key=spec.key,
                display_name=spec.display_name,
                checkpoint_path=spec.checkpoint_path,
                test_suite_model_names=_test_suite_model_names(spec),
            )
            for spec in config.model_specs
        ],
        config.output_dir,
    )


def parse_args() -> argparse.Namespace:
    return parse_instance_checkpoint_comparison_plot_args(description=__doc__)


def main() -> None:
    args = parse_args()
    config = build_checkpoint_comparison_plot_config_from_args(args)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    cache_dirs = {
        spec.key: write_prediction_cache(config, spec) for spec in config.model_specs
    }
    plots = create_comparison_plots(
        cache_dirs=cache_dirs,
        output_dir=config.output_dir,
        n_samples=config.n_samples,
    )
    metric_comparisons = create_metric_comparisons(config)
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
            for spec in config.model_specs
        ],
        "plots": plots,
        "metric_comparisons": metric_comparisons,
    }
    with (config.output_dir / "comparison_plot_manifest.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(manifest, f, indent=2)
    print(
        "Wrote comparison plot manifest to "
        f"{config.output_dir / 'comparison_plot_manifest.json'}"
    )


if __name__ == "__main__":
    main()
