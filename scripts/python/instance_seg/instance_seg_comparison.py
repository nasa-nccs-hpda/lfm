"""Run a true instance segmentation comparison between Toy and Graha.

Both models use the same split instance dataset rooted at
``train/val/test/{chips,labels}``. Toy uses a DINOv3 backbone with Mask2Former
or Mask R-CNN; Graha uses Lunar-FM + TerraTorch Mask R-CNN.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LFM_ROOT = Path(__file__).resolve().parents[3]
if str(LFM_ROOT) not in sys.path:
    sys.path.insert(0, str(LFM_ROOT))

from lfm.all_models.all_tasks import ComparisonExperiment, save_config_json
from lfm.all_models.all_tasks.cli_args import parse_instance_comparison_args
from lfm.all_models.inst_seg.testing.instance_test_suite_callback import (
    GrahaInstancePlotCallback,
    InstanceEpochTestSuiteCallback,
)
from lfm.full_model.inst_seg.instance_model_adapter import GrahaInstanceModelAdapter
from lfm.all_models.all_tasks.utils import (
    create_timestamped_output_dir,
    plot_instance_cache_comparison,
)
from lfm.all_models.all_tasks.utils.utils import ensure_data_symlink

GRAHA_ADAPTER = GrahaInstanceModelAdapter()


@dataclass(frozen=True)
class InstanceComparisonConfig:
    notebook_dir: Path
    lfm_root: Path
    data_root: Path
    base_output_dir: Path
    dino_checkpoint: Path | None
    toy_lightning_checkpoint: Path | None
    graha_pretrain_dir: Path | None
    graha_lightning_checkpoint: Path | None
    graha_input_modality_mode: str
    graha_vis_uv_merge_method: str
    normalization_source: str
    normalization_modality: str
    image_glob: str
    label_glob: str
    image_suffix: str
    label_suffix: str
    toy_architecture: str
    target_size: int
    band_filter: list[int]
    max_train_samples: int | None
    max_val_samples: int | None
    max_test_samples: int | None
    toy_batch_size: int
    toy_num_workers: int
    graha_stats_batch_size: int
    graha_batch_size: int
    graha_num_workers: int
    max_epochs: int
    toy_learning_rate: float
    toy_weight_decay: float
    toy_freeze_backbone: bool
    toy_normalize_inputs: bool
    toy_gradient_clip_val: float | None
    graha_backbone_lr: float
    graha_head_lr: float
    graha_layer_decay: float
    graha_weight_decay: float
    graha_warmup_steps: int
    graha_anchor_sizes: list[list[int]]
    graha_anchor_aspect_ratios: list[float]
    graha_score_threshold: float
    plot_every_n_epochs: int
    plot_n_samples: int
    progress_log_every_n_batches: int
    prediction_split: str
    prediction_n_samples: int
    prediction_score_threshold: float
    mask_shift: tuple[int, int]
    ignore_nodata_in_loss: bool
    nodata_ignore_index: int
    skip_toy_fit: bool
    skip_graha_fit: bool
    run_epoch_test_suite: bool
    epoch_test_split: str
    epoch_test_n_samples: int
    epoch_test_every_n_epochs: int
    seed: int


def _resolve_toy_lightning_checkpoint(args: argparse.Namespace) -> Path | None:
    return (
        Path(args.toy_lightning_checkpoint).resolve()
        if args.toy_lightning_checkpoint
        else None
    )


def build_config(args: argparse.Namespace) -> InstanceComparisonConfig:
    script_dir = Path(__file__).resolve().parent
    lfm_root = script_dir.parents[2]
    notebook_dir = lfm_root / "notebooks" / "full_model"
    scripts_output_dir = lfm_root / "scripts" / "outputs"
    return InstanceComparisonConfig(
        notebook_dir=notebook_dir,
        lfm_root=lfm_root,
        data_root=(
            Path(args.data_root).resolve() if args.data_root else notebook_dir / "data"
        ),
        base_output_dir=(
            Path(args.base_output_dir).resolve()
            if args.base_output_dir
            else scripts_output_dir / "instance_seg_comparison"
        ),
        dino_checkpoint=(
            Path(args.dino_checkpoint).resolve() if args.dino_checkpoint else None
        ),
        toy_lightning_checkpoint=_resolve_toy_lightning_checkpoint(args),
        graha_pretrain_dir=(
            Path(args.graha_pretrain_dir).resolve() if args.graha_pretrain_dir else None
        ),
        graha_lightning_checkpoint=(
            Path(args.graha_lightning_checkpoint).resolve()
            if args.graha_lightning_checkpoint
            else None
        ),
        graha_input_modality_mode=args.graha_input_modality_mode,
        graha_vis_uv_merge_method=args.graha_vis_uv_merge_method,
        normalization_source=getattr(args, "normalization_source", "pretrain"),
        normalization_modality=getattr(args, "normalization_modality", "vis_uv"),
        image_glob=args.image_glob,
        label_glob=args.label_glob,
        image_suffix=args.image_suffix,
        label_suffix=args.label_suffix,
        toy_architecture=args.toy_architecture,
        target_size=args.target_size,
        band_filter=args.band_filter,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        toy_batch_size=args.toy_batch_size,
        toy_num_workers=args.toy_num_workers,
        graha_stats_batch_size=args.graha_stats_batch_size,
        graha_batch_size=args.graha_batch_size,
        graha_num_workers=args.graha_num_workers,
        max_epochs=args.max_epochs,
        toy_learning_rate=args.toy_learning_rate,
        toy_weight_decay=args.toy_weight_decay,
        toy_freeze_backbone=args.toy_freeze_backbone,
        toy_normalize_inputs=args.toy_normalize_inputs,
        toy_gradient_clip_val=(
            None if args.disable_toy_gradient_clipping else args.toy_gradient_clip_val
        ),
        graha_backbone_lr=args.graha_backbone_lr,
        graha_head_lr=args.graha_head_lr,
        graha_layer_decay=args.graha_layer_decay,
        graha_weight_decay=args.graha_weight_decay,
        graha_warmup_steps=args.graha_warmup_steps,
        graha_anchor_sizes=args.graha_anchor_sizes,
        graha_anchor_aspect_ratios=args.graha_anchor_aspect_ratios,
        graha_score_threshold=args.graha_score_threshold,
        plot_every_n_epochs=args.plot_every_n_epochs,
        plot_n_samples=args.plot_n_samples,
        progress_log_every_n_batches=args.progress_log_every_n_batches,
        prediction_split=args.prediction_split,
        prediction_n_samples=args.prediction_n_samples,
        prediction_score_threshold=args.prediction_score_threshold,
        mask_shift=tuple(args.mask_shift),
        ignore_nodata_in_loss=getattr(args, "ignore_nodata_in_loss", False),
        nodata_ignore_index=getattr(args, "nodata_ignore_index", -1),
        skip_toy_fit=args.no_fit or args.skip_toy_fit,
        skip_graha_fit=args.no_fit or args.skip_graha_fit,
        run_epoch_test_suite=args.run_epoch_test_suite,
        epoch_test_split=args.epoch_test_split,
        epoch_test_n_samples=args.epoch_test_n_samples,
        epoch_test_every_n_epochs=args.epoch_test_every_n_epochs,
        seed=args.seed,
    )


def validate_paths(config: InstanceComparisonConfig) -> None:
    required = []
    for split in ("train", "val", "test"):
        required.extend(
            [
                config.data_root / split / "chips",
                config.data_root / split / "labels",
            ]
        )
    for path in [
        config.dino_checkpoint,
        config.toy_lightning_checkpoint,
        config.graha_pretrain_dir,
        config.graha_lightning_checkpoint,
    ]:
        if path is not None:
            required.append(path)
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required paths:\n" + "\n".join(str(path) for path in missing)
        )


def save_config(config: InstanceComparisonConfig, output_dir: Path) -> None:
    save_config_json(config, output_dir / "config.json")


def get_toy_normalization_modality_info(
    config: InstanceComparisonConfig,
) -> Path | None:
    if not config.toy_normalize_inputs or config.normalization_source != "pretrain":
        return None
    normalization_config = GRAHA_ADAPTER.build_comparison_config(
        config,
        config.base_output_dir,
    )
    return normalization_config.modality_info


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return parse_instance_comparison_args(argv, description=__doc__)


def main() -> None:
    args = parse_args()
    notebook_dir = Path(__file__).resolve().parents[2] / "notebooks" / "full_model"
    ensure_data_symlink(args.simlink_dest, notebook_dir / "data")
    config = build_config(args)
    validate_paths(config)
    output_dir = create_timestamped_output_dir(config.base_output_dir)

    def run_toy():
        from lfm.all_models.inst_seg.workflows import instance_toy_workflow

        return instance_toy_workflow.run_toy_workflow(
            config,
            output_dir,
            normalization_modality_info=get_toy_normalization_modality_info(config),
            epoch_test_suite_callback_cls=InstanceEpochTestSuiteCallback,
        )

    def run_graha():
        from lfm.all_models.inst_seg.workflows import instance_graha_workflow

        return instance_graha_workflow.run_graha_workflow(
            config,
            output_dir,
            validation_plot_callback_cls=GrahaInstancePlotCallback,
            epoch_test_suite_callback_cls=InstanceEpochTestSuiteCallback,
        )

    def on_complete(started_at: float, results: dict[str, Any]) -> None:
        toy_prediction_cache = results["toy"]
        graha_prediction_cache = results["graha"]
        if toy_prediction_cache is not None and graha_prediction_cache is not None:
            plot_instance_cache_comparison(
                {
                    "toy": toy_prediction_cache,
                    "graha": graha_prediction_cache,
                },
                output_dir / "plots" / "comparison",
                n_samples=config.prediction_n_samples,
            )

        elapsed = time.perf_counter() - started_at
        with (output_dir / "timing_summary.json").open("w", encoding="utf-8") as f:
            json.dump({"seconds": round(elapsed, 3)}, f, indent=2)
        print(f"Comparison elapsed seconds: {elapsed:.3f}", flush=True)

    ComparisonExperiment(
        config=config,
        output_dir=output_dir,
        checkpoint_subdirs=[
            Path("checkpoints") / "toy_model",
            Path("checkpoints") / "full_model",
        ],
        run_toy=run_toy,
        run_graha=run_graha,
        on_complete=on_complete,
    ).run()


if __name__ == "__main__":
    main()
