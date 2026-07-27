"""Run one semantic-segmentation model without final comparison plotting."""

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
from lfm.full_model.all_tasks.utils import ensure_data_symlink
from lfm.full_model.sem_seg.semantic_model_adapter import (
    GrahaSemanticModelAdapter,
)
from lfm.toy_model.sem_seg.semantic_model_adapter import (
    ToySemanticModelAdapter,
)
from scripts.python.semantic_seg import (
    semantic_seg_comparison as comparison,
)

TOY_ADAPTER = ToySemanticModelAdapter()
GRAHA_ADAPTER = GrahaSemanticModelAdapter()


def parse_args() -> tuple[str, argparse.Namespace]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", choices=["toy", "graha"], required=True)
    model_args, remaining = parser.parse_known_args()
    return model_args.model, comparison.parse_args(remaining)


def main() -> None:
    model, args = parse_args()
    notebook_dir = LFM_ROOT / "notebooks" / "full_model"
    ensure_data_symlink(args.simlink_dest, notebook_dir / "data")
    config = comparison.build_config(args)
    comparison.validate_data_paths(config)

    output_dir = config.base_output_dir
    timing_rows: list[dict[str, Any]] = []

    def run_model():
        if model == "toy":
            return TOY_ADAPTER.run_workflow(
                config,
                output_dir=output_dir,
                normalization_modality_info=(
                    comparison.get_toy_normalization_modality_info(config)
                ),
                epoch_test_suite_callback_cls=comparison.SemanticEpochTestSuiteCallback,
                timing_rows=timing_rows,
                record_timing=comparison.record_timing,
            )
        return GRAHA_ADAPTER.run_workflow(
            config,
            no_fit=config.skip_graha_fit,
            comparison_output_dir=output_dir,
            epoch_test_suite_callback_cls=comparison.SemanticEpochTestSuiteCallback,
            timing_rows=timing_rows,
            record_timing=comparison.record_timing,
        )

    def on_complete(started_at: float, result) -> None:
        comparison.record_timing(
            timing_rows,
            model=model.title(),
            stage="single_model_total",
            started_at=started_at,
        )
        comparison.save_timing_summary(timing_rows, output_dir)

    SingleModelExperiment(
        model=model,
        config=config,
        output_dir=output_dir,
        run_model=run_model,
        on_complete=on_complete,
    ).run()


if __name__ == "__main__":
    main()
