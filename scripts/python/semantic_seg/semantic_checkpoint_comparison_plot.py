"""Create semantic comparison plots from final Toy/Graha/GFFT checkpoints."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

from lightning.pytorch import seed_everything

LFM_ROOT = Path(__file__).resolve().parents[3]
if str(LFM_ROOT) not in sys.path:
    sys.path.insert(0, str(LFM_ROOT))

from lfm.all_models.all_tasks.cli_args import (
    parse_semantic_checkpoint_comparison_plot_args,
)
from lfm.all_models.sem_seg.plot_config import (
    SemanticCheckpointComparisonPlotConfig,
    SemanticModelPlotSpec,
    build_checkpoint_comparison_plot_config_from_args,
)
from lfm.all_models.sem_seg.plotting import plot_prediction_cache_comparison
from lfm.all_models.sem_seg.workflows import (
    semantic_gfft_workflow,
    semantic_graha_workflow,
    semantic_toy_workflow,
)
from scripts.python.semantic_seg import semantic_seg_comparison as comparison


def write_prediction_cache(
    config: SemanticCheckpointComparisonPlotConfig,
    spec: SemanticModelPlotSpec,
) -> Path:
    print(f"[{spec.key}] loading checkpoint: {spec.checkpoint_path}", flush=True)
    cache_output_dir = config.output_dir / "cache_sources" / spec.key
    if spec.model_family == "toy":
        run_config = replace(
            config.experiment_config,
            toy_lightning_checkpoint=spec.checkpoint_path,
        )
        cache_dir = semantic_toy_workflow.run_toy_workflow(
            run_config,
            output_dir=cache_output_dir,
            normalization_modality_info=(
                comparison.get_toy_normalization_modality_info(run_config)
            ),
        )
    elif spec.model_family == "graha":
        run_config = replace(
            config.experiment_config,
            graha_lightning_checkpoint=spec.checkpoint_path,
        )
        _, cache_dir = semantic_graha_workflow.run_graha_workflow(
            run_config,
            no_fit=True,
            comparison_output_dir=cache_output_dir,
        )
    elif spec.model_family == "gfft":
        run_config = replace(
            config.experiment_config,
            graha_lightning_checkpoint=spec.checkpoint_path,
        )
        _, cache_dir = semantic_gfft_workflow.run_gfft_workflow(
            run_config,
            no_fit=True,
            output_dir=cache_output_dir,
        )
    else:
        raise ValueError(f"Unknown model family: {spec.model_family}")

    if cache_dir is None:
        raise RuntimeError(f"{spec.display_name} did not create a prediction cache.")
    return cache_dir


def create_comparison_plots(
    *,
    cache_dirs: dict[str, Path],
    output_dir: Path,
    n_samples: int,
) -> dict[str, str]:
    plots_dir = output_dir / "plots"
    outputs = {}
    all_plot = plot_prediction_cache_comparison(
        cache_dirs,
        plots_dir,
        n_samples=n_samples,
        filename="all_models_semantic_predictions.png",
    )
    outputs["all_models"] = str(all_plot)

    for left, right in itertools.combinations(cache_dirs, 2):
        filename = f"{left}_vs_{right}_semantic_predictions.png"
        path = plot_prediction_cache_comparison(
            {left: cache_dirs[left], right: cache_dirs[right]},
            plots_dir / "pairwise",
            n_samples=n_samples,
            filename=filename,
        )
        outputs[f"{left}_vs_{right}"] = str(path)
    return outputs


def parse_args() -> argparse.Namespace:
    return parse_semantic_checkpoint_comparison_plot_args(description=__doc__)


def main() -> None:
    args = parse_args()
    config = build_checkpoint_comparison_plot_config_from_args(args)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.experiment_config.seed)

    cache_dirs = {
        spec.key: write_prediction_cache(config, spec) for spec in config.model_specs
    }
    plots = create_comparison_plots(
        cache_dirs=cache_dirs,
        output_dir=config.output_dir,
        n_samples=config.n_samples,
    )
    manifest = {
        "models": [
            {
                "key": spec.key,
                "display_name": spec.display_name,
                "model_family": spec.model_family,
                "checkpoint_path": str(spec.checkpoint_path),
                "cache_dir": str(cache_dirs[spec.key]),
            }
            for spec in config.model_specs
        ],
        "plots": plots,
    }
    manifest_path = config.output_dir / "comparison_plot_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote comparison plot manifest to {manifest_path}")


if __name__ == "__main__":
    main()
