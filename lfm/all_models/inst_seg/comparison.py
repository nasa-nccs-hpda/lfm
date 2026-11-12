"""Fresh best-validation checkpoint inference and held-out paired comparison."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from lfm.all_models.inst_seg.mask_ap import (
    MASK_AP_PROTOCOL,
    InstanceMaskAP,
    mask_ap_inference,
)
from lfm.all_models.inst_seg.paired_experiment import positive_integer
from lfm.all_models.inst_seg.prediction.instance_prediction_cache import (
    _to_cpu_prediction_list,
)
from lfm.all_models.inst_seg.prediction_export import (
    DISPLAY_BANDS,
    MODELS,
    DisplayOptions,
    PairedPredictionExport,
)
from lfm.all_models.inst_seg.training import load_completed_pair, prepare_for_test
from lfm.all_models.instance_comparison_plot import plot_instance_comparison
from lfm.artifacts import write_json as _write_json


def _evaluate_model(result, preparation, output, batch_size, device, exports):
    model = result.run.spec.model
    evaluator = InstanceMaskAP()
    with prepare_for_test(
        result,
        preparation,
        batch_size=batch_size,
        eligibility_dir=output / model / "eligibility",
        device=device,
    ) as prepared:
        print(
            f"{model} test inference started; membership={len(preparation.membership.splits['test'])} chips; batch_size={batch_size}",
            flush=True,
        )
        last_report = time.monotonic()
        with torch.inference_mode(), mask_ap_inference(prepared.task.model):
            for batch_idx, batch in enumerate(
                exports.batches(model, prepared.datamodule.test_dataloader())
            ):
                predictions = _to_cpu_prediction_list(
                    prepared.task(batch["image"].to(device))
                )
                evaluator.update(predictions, batch)
                exports.write(model, batch, predictions)
                now = time.monotonic()
                if (
                    batch_idx == 0
                    or (batch_idx + 1) % 50 == 0
                    or now - last_report >= 60
                ):
                    print(
                        f"{model} test inference; batches={batch_idx + 1}; eligible_chips={exports.count(model)}",
                        flush=True,
                    )
                    last_report = now
                del predictions
    print(
        f"{model} full test inference finished; computing mask AP for {exports.count(model)} eligible chips",
        flush=True,
    )
    metrics = {
        "test_mask_ap": float(evaluator.compute()),
        "eligible_chips": evaluator.num_chips,
    }
    _write_json(output / model / "metrics.json", metrics)
    print(
        f"{model} test evaluation complete; mask_AP={metrics['test_mask_ap']:.6g}; metrics={output / model / 'metrics.json'}",
        flush=True,
    )
    return metrics


def compare_checkpoints(
    toy_result: str | Path,
    graha_result: str | Path,
    output_dir: str | Path,
    *,
    batch_size: int = 32,
    n_samples: int = 5,
    display_score_threshold: float = 0.5,
) -> Path:
    """Evaluate two freshly loaded checkpoints; publish success after the figure."""
    if any(type(value) is not int or value < 1 for value in (batch_size, n_samples)):
        raise ValueError("batch_size and n_samples must be positive integers.")
    if (
        not math.isfinite(display_score_threshold)
        or not 0 <= display_score_threshold <= 1
    ):
        raise ValueError("Display score threshold must be in [0, 1].")
    print(
        "Loading and validating Toy/GRAHA result manifests and selected checkpoints",
        flush=True,
    )
    pair = load_completed_pair(toy_result, graha_result)
    results = {"toy": pair.toy, "graha": pair.graha}
    toy = pair.toy
    output = Path(output_dir).expanduser().resolve()
    # A fresh destination makes failed publication unambiguous and reruns safe.
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exports = PairedPredictionExport(
        output,
        pair.preparation["toy"].membership,
        display=DisplayOptions(n_samples, display_score_threshold),
        width=toy.run.configuration.target_size,
    )
    metrics = {}
    for model in MODELS:
        print(
            f"Loading {model} best validation checkpoint: {results[model].payload['best_checkpoint']['path']}",
            flush=True,
        )
        metrics[model] = _evaluate_model(
            results[model], pair.preparation[model], output, batch_size, device, exports
        )
    reports = {model: output / model / "eligibility/test.json" for model in MODELS}
    saved = exports.finish(reports)
    chips = saved.chips
    labels = {
        model: f"{'Toy' if model == 'toy' else 'GRAHA'} prediction\n{Path(results[model].payload['best_checkpoint']['path']).name} · epoch {results[model].payload['best_checkpoint']['epoch']}"
        for model in MODELS
    }
    print(f"Rendering four-row test comparison for {len(chips)} chips", flush=True)
    figure = plot_instance_comparison(
        chips,
        output / "test_comparison.png",
        model_labels=labels,
        title="Test split · best full-validation mask AP checkpoints",
    )
    print(f"Comparison figure saved: {figure}", flush=True)
    _write_json(
        output / "metrics.json",
        {"split": "test", "models": metrics, "protocol": MASK_AP_PROTOCOL},
    )
    manifest_path = output / "comparison.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "complete",
            "split": "test",
            "config_id": toy.payload["config_id"],
            "split_manifest_id": toy.payload["split_manifest_id"],
            "models": {
                model: {"result_manifest": str(result.path), **result.payload}
                for model, result in results.items()
            },
            "inference": {
                "batch_size": batch_size,
                "device": str(device),
                "mode": "eval_inference_mode",
                "protocol": MASK_AP_PROTOCOL,
            },
            "display": {
                "requested_count": n_samples,
                "chip_ids": saved.selected_ids,
                "selection": "sorted_chip_id_first_eligible_before_predictions",
                "stored_bands": DISPLAY_BANDS,
                "stretch": "per_channel_minmax_raw_crop_common_valid_pixels",
                "score_threshold": display_score_threshold,
            },
            "prediction_encoding": saved.encoding,
            "prediction_indexes": saved.indexes,
            "eligibility_reports": reports,
            "visualization_caches": saved.visualization_caches,
            "metrics": metrics,
            "figure": str(figure),
            "epoch_indexing": "zero_based",
        },
    )
    print(
        f"Evaluation and plot pipeline complete; Toy mask_AP={metrics['toy']['test_mask_ap']:.6g}; GRAHA mask_AP={metrics['graha']['test_mask_ap']:.6g}; manifest={manifest_path}",
        flush=True,
    )
    return manifest_path


def create_comparison_parser():
    parser = argparse.ArgumentParser(
        description="Reload paired best-validation checkpoints and compare the full test split."
    )
    parser.add_argument("--toy-result", type=Path, required=True)
    parser.add_argument("--graha-result", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Fresh comparison directory (rerun in a new directory).",
    )
    parser.add_argument("--batch-size", type=positive_integer, default=32)
    parser.add_argument("--n-samples", type=positive_integer, default=5)
    parser.add_argument("--display-score-threshold", type=float, default=0.5)
    return parser
