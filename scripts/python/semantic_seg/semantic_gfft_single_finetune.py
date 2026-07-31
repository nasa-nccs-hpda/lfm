"""Run one GFFT semantic-segmentation model without comparison plotting."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

LFM_ROOT = Path(__file__).resolve().parents[3]
if str(LFM_ROOT) not in sys.path:
    sys.path.insert(0, str(LFM_ROOT))

from lfm.all_models.all_tasks import SingleModelExperiment
from lfm.all_models.all_tasks.cli_args import parse_semantic_experiment_args
from lfm.all_models.all_tasks.utils import ensure_data_symlink
from lfm.all_models.sem_seg.config import (
    SemanticSegmentationExperimentConfig,
    build_config_from_args,
)
from lfm.all_models.sem_seg.testing.semantic_test_suite_callback import (
    SemanticEpochTestSuiteCallback,
)
from lfm.all_models.sem_seg.workflows import semantic_gfft_workflow
from scripts.python.semantic_seg import semantic_seg_comparison as comparison


def parse_args() -> argparse.Namespace:
    return parse_semantic_experiment_args(description=__doc__)


def validate_gfft_paths(config: SemanticSegmentationExperimentConfig) -> None:
    comparison.validate_data_paths(config)
    if config.gfft_config_path is None and config.gfft_backbone_checkpoint is None:
        raise ValueError(
            "GFFT semantic fine-tuning requires --gfft-config-path or "
            "--gfft-backbone-checkpoint."
        )
    if config.normalization_source == "pretrain" and config.gfft_config_path is None:
        raise ValueError(
            "GFFT pretrain normalization requires --gfft-config-path so the "
            "workflow can load YAML normalization stats."
        )
    required = [config.gfft_config_path, config.gfft_backbone_checkpoint]
    missing = [path for path in required if path is not None and not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required GFFT paths:\n" + "\n".join(str(path) for path in missing)
        )


def main() -> None:
    args = parse_args()
    notebook_dir = LFM_ROOT / "notebooks" / "full_model"
    ensure_data_symlink(args.simlink_dest, notebook_dir / "data")
    config = build_config_from_args(args)
    validate_gfft_paths(config)
    output_dir = config.base_output_dir
    timing_rows: list[dict[str, Any]] = []

    def on_complete(started_at: float, result: Any) -> None:
        comparison.record_timing(
            timing_rows,
            model="GFFT",
            stage="single_model_total",
            started_at=started_at,
        )
        comparison.save_timing_summary(timing_rows, output_dir)

    SingleModelExperiment(
        model="gfft",
        config=config,
        output_dir=output_dir,
        checkpoint_subdirs=[Path("checkpoints") / "gfft_model"],
        run_model=lambda: semantic_gfft_workflow.run_gfft_workflow(
            config,
            no_fit=config.skip_graha_fit,
            output_dir=output_dir,
            epoch_test_suite_callback_cls=SemanticEpochTestSuiteCallback,
            timing_rows=timing_rows,
            record_timing=comparison.record_timing,
        ),
        on_complete=on_complete,
    ).run()


if __name__ == "__main__":
    main()
