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
from pathlib import Path
from typing import Any

LFM_ROOT = Path(__file__).resolve().parents[3]
if str(LFM_ROOT) not in sys.path:
    sys.path.insert(0, str(LFM_ROOT))

from lfm.all_models.all_tasks import ComparisonExperiment
from lfm.all_models.all_tasks.cli_args import parse_instance_comparison_args
from lfm.all_models.inst_seg.config import (
    InstanceSegmentationExperimentConfig,
    build_config_from_args as _build_config_from_args,
)
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


def validate_paths(config: InstanceSegmentationExperimentConfig) -> None:
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


def get_toy_normalization_modality_info(
    config: InstanceSegmentationExperimentConfig,
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
    config = _build_config_from_args(args)
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
