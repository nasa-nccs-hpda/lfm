"""Train Toy and Graha semantic segmentation models on split full-model data."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

LFM_ROOT = Path(__file__).resolve().parents[3]
if str(LFM_ROOT) not in sys.path:
    sys.path.insert(0, str(LFM_ROOT))

from lfm.all_models.all_tasks import ComparisonExperiment
from lfm.all_models.all_tasks.cli_args import parse_semantic_comparison_args
from lfm.all_models.all_tasks.utils import (
    create_timestamped_output_dir,
    evaluate_prediction_caches,
    plot_prediction_cache_comparison,
)
from lfm.all_models.all_tasks.utils.utils import ensure_data_symlink
from lfm.full_model.sem_seg.semantic_model_adapter import GrahaSemanticModelAdapter
from lfm.all_models.sem_seg.testing.semantic_test_suite_callback import (
    SemanticEpochTestSuiteCallback,
)
from lfm.all_models.sem_seg.config import (
    SemanticSegmentationExperimentConfig,
    build_config_from_args as _build_config_from_args,
)
from lfm.all_models.sem_seg.workflows import (
    semantic_graha_workflow,
    semantic_toy_workflow,
)

GRAHA_ADAPTER = GrahaSemanticModelAdapter()


def validate_data_paths(config: SemanticSegmentationExperimentConfig) -> None:
    required = []
    for split in ["train", "val", "test"]:
        required.extend(
            [
                config.data_root / split / "chips",
                config.data_root / split / "labels",
            ]
        )
    for checkpoint_path in [
        config.dino_checkpoint,
        config.toy_lightning_checkpoint,
        config.graha_pretrain_dir,
        config.graha_lightning_checkpoint,
    ]:
        if checkpoint_path is not None:
            required.append(checkpoint_path)
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required split data paths:\n"
            + "\n".join(str(path) for path in missing)
        )


def _format_seconds(seconds: float) -> str:
    whole_seconds = int(round(seconds))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def save_timing_summary(timing_rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not timing_rows:
        return

    json_path = output_dir / "timing_summary.json"
    csv_path = output_dir / "timing_summary.csv"
    fieldnames = ["model", "stage", "seconds", "elapsed_hms"]

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(timing_rows, f, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(timing_rows)

    print(f"Saved timing summary to {csv_path}")


def record_timing(
    timing_rows: list[dict[str, Any]],
    *,
    model: str,
    stage: str,
    started_at: float,
) -> None:
    elapsed = time.perf_counter() - started_at
    timing_rows.append(
        {
            "model": model,
            "stage": stage,
            "seconds": round(elapsed, 3),
            "elapsed_hms": _format_seconds(elapsed),
        }
    )
    print(
        f"[timing] {model} {stage}: {_format_seconds(elapsed)} ({elapsed:.3f}s)",
        flush=True,
    )


def get_toy_normalization_modality_info(
    config: SemanticSegmentationExperimentConfig,
) -> Path | None:
    if not config.normalize_inputs or config.normalization_source != "pretrain":
        return None
    normalization_config = GRAHA_ADAPTER.build_comparison_config(
        config,
        config.graha_base_output_dir,
    )
    return normalization_config.modality_info


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return parse_semantic_comparison_args(argv, description=__doc__)


def main() -> None:
    args = parse_args()
    notebook_dir = Path(__file__).resolve().parents[2] / "notebooks" / "full_model"
    ensure_data_symlink(args.simlink_dest, notebook_dir / "data")
    config = _build_config_from_args(args)
    validate_data_paths(config)
    output_dir = create_timestamped_output_dir(config.base_output_dir)
    timing_rows: list[dict[str, Any]] = []

    def run_toy():
        return semantic_toy_workflow.run_toy_workflow(
            config,
            output_dir=output_dir,
            normalization_modality_info=get_toy_normalization_modality_info(config),
            epoch_test_suite_callback_cls=SemanticEpochTestSuiteCallback,
            timing_rows=timing_rows,
            record_timing=record_timing,
        )

    def run_graha():
        _, graha_prediction_cache = semantic_graha_workflow.run_graha_workflow(
            config,
            no_fit=config.skip_graha_fit,
            comparison_output_dir=output_dir,
            epoch_test_suite_callback_cls=SemanticEpochTestSuiteCallback,
            timing_rows=timing_rows,
            record_timing=record_timing,
        )
        return graha_prediction_cache

    def on_complete(started_at: float, results: dict[str, Any]) -> None:
        toy_prediction_cache = results["toy"]
        graha_prediction_cache = results["graha"]
        if config.cache_predictions and toy_prediction_cache and graha_prediction_cache:
            comparison_started_at = time.perf_counter()
            comparison_caches = {
                "toy": toy_prediction_cache,
                "graha": graha_prediction_cache,
            }
            plot_prediction_cache_comparison(
                comparison_caches,
                output_dir / "plots" / "comparison",
                n_samples=min(5, config.prediction_n_samples),
            )
            _, metric_summary = evaluate_prediction_caches(
                comparison_caches,
                output_dir / "comparison_metrics",
            )
            print("Comparison metric summary:")
            for row in metric_summary:
                print("  " + json.dumps(row, sort_keys=True))
            record_timing(
                timing_rows,
                model="Comparison",
                stage="plots_and_metrics",
                started_at=comparison_started_at,
            )
        record_timing(
            timing_rows,
            model="Comparison",
            stage="total",
            started_at=started_at,
        )
        save_timing_summary(timing_rows, output_dir)

    ComparisonExperiment(
        config=config,
        output_dir=output_dir,
        run_toy=run_toy,
        run_graha=run_graha,
        on_complete=on_complete,
    ).run()


if __name__ == "__main__":
    main()
