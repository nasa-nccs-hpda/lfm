"""Run one instance-segmentation model without final comparison plotting."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

LFM_ROOT = Path(__file__).resolve().parents[3]
if str(LFM_ROOT) not in sys.path:
    sys.path.insert(0, str(LFM_ROOT))

from lfm.all_models.all_tasks import SingleModelExperiment
from lfm.full_model.all_tasks.utils import ensure_data_symlink
from lfm.full_model.inst_seg.instance_model_adapter import GrahaInstanceModelAdapter
from lfm.toy_model.inst_seg.instance_model_adapter import ToyInstanceModelAdapter
from scripts.python.instance_seg import instance_seg_comparison as comparison

TOY_ADAPTER = ToyInstanceModelAdapter()
GRAHA_ADAPTER = GrahaInstanceModelAdapter()


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
    comparison.validate_paths(config)

    output_dir = config.base_output_dir

    def run_model():
        if model == "toy":
            return TOY_ADAPTER.run_workflow(
                config,
                output_dir,
                normalization_modality_info=(
                    comparison.get_toy_normalization_modality_info(config)
                ),
                epoch_test_suite_callback_cls=comparison.InstanceEpochTestSuiteCallback,
            )
        return GRAHA_ADAPTER.run_workflow(
            config,
            output_dir,
            validation_plot_callback_cls=comparison.GrahaInstancePlotCallback,
            epoch_test_suite_callback_cls=comparison.InstanceEpochTestSuiteCallback,
        )

    SingleModelExperiment(
        model=model,
        config=config,
        output_dir=output_dir,
        checkpoint_subdirs=[
            Path("checkpoints") / "toy_model",
            Path("checkpoints") / "full_model",
        ],
        run_model=run_model,
    ).run()


if __name__ == "__main__":
    main()
