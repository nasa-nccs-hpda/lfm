"""Run one instance-segmentation model without final comparison plotting."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

LFM_ROOT = Path(__file__).resolve().parents[3]
if str(LFM_ROOT) not in sys.path:
    sys.path.insert(0, str(LFM_ROOT))

from lfm.full_model.all_tasks.utils import ensure_data_symlink
from lfm.full_model.inst_seg.instance_model_adapter import GrahaInstanceModelAdapter
from lfm.toy_model.inst_seg.instance_model_adapter import ToyInstanceModelAdapter
from scripts.python.instance_seg import instance_seg_comparison as comparison

TOY_ADAPTER = ToyInstanceModelAdapter()
GRAHA_ADAPTER = GrahaInstanceModelAdapter()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    return value


def parse_args() -> tuple[str, argparse.Namespace]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", choices=["toy", "graha"], required=True)
    model_args, remaining = parser.parse_known_args()
    return model_args.model, comparison.parse_args(remaining)


def save_single_config(model: str, config: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"config_{model}_model.json").open("w", encoding="utf-8") as f:
        json.dump(_json_ready(asdict(config)), f, indent=2)


def main() -> None:
    started_at = time.perf_counter()
    model, args = parse_args()
    notebook_dir = LFM_ROOT / "notebooks" / "full_model"
    ensure_data_symlink(args.simlink_dest, notebook_dir / "data")
    config = comparison.build_config(args)
    comparison.validate_paths(config)

    output_dir = config.base_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints" / "toy_model").mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints" / "full_model").mkdir(parents=True, exist_ok=True)
    save_single_config(model, config, output_dir)

    if model == "toy":
        TOY_ADAPTER.run_workflow(
            config,
            output_dir,
            normalization_modality_info=comparison.get_toy_normalization_modality_info(
                config
            ),
            epoch_test_suite_callback_cls=comparison.InstanceEpochTestSuiteCallback,
        )
    else:
        GRAHA_ADAPTER.run_workflow(
            config,
            output_dir,
            validation_plot_callback_cls=comparison.GrahaInstancePlotCallback,
            epoch_test_suite_callback_cls=comparison.InstanceEpochTestSuiteCallback,
        )

    elapsed = time.perf_counter() - started_at
    timing_path = output_dir / f"timing_summary_{model}_model.json"
    with timing_path.open("w", encoding="utf-8") as f:
        json.dump({"model": model, "seconds": round(elapsed, 3)}, f, indent=2)
    print(f"{model.title()} elapsed seconds: {elapsed:.3f}", flush=True)


if __name__ == "__main__":
    main()
