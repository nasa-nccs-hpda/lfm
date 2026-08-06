"""Compare final-epoch test-suite metrics across model checkpoints."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class FinalEpochMetricSpec:
    key: str
    display_name: str
    checkpoint_path: Path
    test_suite_model_names: tuple[str, ...]


@dataclass(frozen=True)
class LoadedFinalEpochMetrics:
    key: str
    display_name: str
    output_dir: Path
    suite_dir: Path
    metrics_path: Path
    epoch: int
    metrics: dict[str, float]


def _checkpoint_output_dir(checkpoint_path: Path) -> Path:
    checkpoint_path = Path(checkpoint_path).resolve()
    for parent in checkpoint_path.parents:
        if parent.name == "checkpoints":
            return parent.parent
    raise ValueError(
        f"Could not infer model output directory from checkpoint path: {checkpoint_path}"
    )


def _epoch_from_name(path: Path) -> int:
    match = re.search(r"epoch[_=-](\d+)", path.name)
    return int(match.group(1)) if match else -1


def _load_metric_array(path: Path) -> dict[str, float]:
    arr = np.load(path, allow_pickle=False)
    if arr.dtype.names is None:
        raise ValueError(f"Metric array is not structured: {path}")
    row = arr.reshape(-1)[0]
    return {name: float(row[name]) for name in arr.dtype.names}


def _find_final_metric_path(
    output_dir: Path,
    model_names: tuple[str, ...],
) -> tuple[Path, int]:
    candidates = []
    for model_name in model_names:
        suite_root = output_dir / "test_suite" / model_name
        if not suite_root.exists():
            continue
        for metrics_path in suite_root.glob("epoch_*/metrics.npy"):
            candidates.append((_epoch_from_name(metrics_path.parent), metrics_path))
    if not candidates:
        names = ", ".join(model_names)
        raise FileNotFoundError(
            f"No final-epoch metrics.npy found under {output_dir / 'test_suite'} "
            f"for model name(s): {names}"
        )
    epoch, metrics_path = max(candidates, key=lambda item: (item[0], str(item[1])))
    return metrics_path, epoch


def load_final_epoch_metrics(
    specs: list[FinalEpochMetricSpec],
) -> list[LoadedFinalEpochMetrics]:
    loaded = []
    for spec in specs:
        output_dir = _checkpoint_output_dir(spec.checkpoint_path)
        metrics_path, epoch = _find_final_metric_path(
            output_dir,
            spec.test_suite_model_names,
        )
        loaded.append(
            LoadedFinalEpochMetrics(
                key=spec.key,
                display_name=spec.display_name,
                output_dir=output_dir,
                suite_dir=metrics_path.parent,
                metrics_path=metrics_path,
                epoch=epoch,
                metrics=_load_metric_array(metrics_path),
            )
        )
        print(
            f"[{spec.key}] loaded final-epoch metrics: {metrics_path}",
            flush=True,
        )
    return loaded


def _metric_names(rows: list[LoadedFinalEpochMetrics]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        names.update(row.metrics)
    return sorted(names)


def _score_metric_names(
    rows: list[LoadedFinalEpochMetrics],
    metric_names: list[str],
) -> list[str]:
    score_metrics = []
    for metric in metric_names:
        values = [row.metrics.get(metric, np.nan) for row in rows]
        finite = [value for value in values if np.isfinite(value)]
        if finite and max(abs(value) for value in finite) <= 1.5:
            score_metrics.append(metric)
    return score_metrics or metric_names


def _write_text_summary(
    *,
    loaded: list[LoadedFinalEpochMetrics],
    metric_names: list[str],
    pairwise_rows: list[dict[str, object]],
    three_way_rows: list[dict[str, object]],
    output_dir: Path,
) -> Path:
    path = output_dir / "final_epoch_metric_comparison.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write("Final epoch metric comparison\n")
        f.write("=============================\n\n")
        f.write("Model metrics\n")
        f.write("-------------\n")
        header = ["model", "epoch", "metrics_path", *metric_names]
        f.write("\t".join(header) + "\n")
        for row in loaded:
            values = [
                row.display_name,
                str(row.epoch),
                str(row.metrics_path),
                *[
                    (f"{row.metrics[metric]:.8f}" if metric in row.metrics else "")
                    for metric in metric_names
                ],
            ]
            f.write("\t".join(values) + "\n")

        f.write("\nPairwise comparisons\n")
        f.write("--------------------\n")
        f.write("left_model\tright_model\tmetric\tleft_value\tright_value\tdelta\n")
        for row in pairwise_rows:
            f.write(
                "\t".join(
                    [
                        str(row["left_model"]),
                        str(row["right_model"]),
                        str(row["metric"]),
                        f"{float(row['left_value']):.8f}",
                        f"{float(row['right_value']):.8f}",
                        f"{float(row['delta']):.8f}",
                    ]
                )
                + "\n"
            )

        if three_way_rows:
            f.write("\nThree-way comparisons\n")
            f.write("---------------------\n")
            f.write(
                "metric\tbest_model\tbest_value\tworst_model\tworst_value\tspread\n"
            )
            for row in three_way_rows:
                f.write(
                    "\t".join(
                        [
                            str(row["metric"]),
                            str(row["best_model"]),
                            f"{float(row['best_value']):.8f}",
                            str(row["worst_model"]),
                            f"{float(row['worst_value']):.8f}",
                            f"{float(row['spread']):.8f}",
                        ]
                    )
                    + "\n"
                )
    return path


def _save_model_metric_array(
    loaded: list[LoadedFinalEpochMetrics],
    metric_names: list[str],
    output_dir: Path,
) -> Path:
    dtype = [
        ("model_key", "U64"),
        ("display_name", "U128"),
        ("epoch", "i8"),
        ("output_dir", "U1024"),
        ("metrics_path", "U1024"),
        *[(metric, "f8") for metric in metric_names],
    ]
    arr = np.zeros(len(loaded), dtype=dtype)
    for index, row in enumerate(loaded):
        arr[index]["model_key"] = row.key
        arr[index]["display_name"] = row.display_name
        arr[index]["epoch"] = row.epoch
        arr[index]["output_dir"] = str(row.output_dir)
        arr[index]["metrics_path"] = str(row.metrics_path)
        for metric in metric_names:
            arr[index][metric] = row.metrics.get(metric, np.nan)
    path = output_dir / "final_epoch_model_metrics.npy"
    np.save(path, arr)
    return path


def _save_pairwise_array(
    pairwise_rows: list[dict[str, object]],
    output_dir: Path,
) -> Path:
    dtype = [
        ("left_key", "U64"),
        ("right_key", "U64"),
        ("left_model", "U128"),
        ("right_model", "U128"),
        ("metric", "U128"),
        ("left_value", "f8"),
        ("right_value", "f8"),
        ("delta", "f8"),
    ]
    arr = np.zeros(len(pairwise_rows), dtype=dtype)
    for index, row in enumerate(pairwise_rows):
        for name in arr.dtype.names or ():
            arr[index][name] = row[name]
    path = output_dir / "final_epoch_pairwise_metric_comparison.npy"
    np.save(path, arr)
    return path


def _save_three_way_array(
    three_way_rows: list[dict[str, object]],
    output_dir: Path,
) -> Path | None:
    if not three_way_rows:
        return None
    dtype = [
        ("metric", "U128"),
        ("best_key", "U64"),
        ("best_model", "U128"),
        ("best_value", "f8"),
        ("worst_key", "U64"),
        ("worst_model", "U128"),
        ("worst_value", "f8"),
        ("spread", "f8"),
        ("ranked_models", "U512"),
    ]
    arr = np.zeros(len(three_way_rows), dtype=dtype)
    for index, row in enumerate(three_way_rows):
        for name in arr.dtype.names or ():
            arr[index][name] = row[name]
    path = output_dir / "final_epoch_three_way_metric_comparison.npy"
    np.save(path, arr)
    return path


def _build_pairwise_rows(
    loaded: list[LoadedFinalEpochMetrics],
    metric_names: list[str],
) -> list[dict[str, object]]:
    rows = []
    for left, right in itertools.combinations(loaded, 2):
        for metric in metric_names:
            left_value = left.metrics.get(metric, np.nan)
            right_value = right.metrics.get(metric, np.nan)
            rows.append(
                {
                    "left_key": left.key,
                    "right_key": right.key,
                    "left_model": left.display_name,
                    "right_model": right.display_name,
                    "metric": metric,
                    "left_value": left_value,
                    "right_value": right_value,
                    "delta": right_value - left_value,
                }
            )
    return rows


def _build_three_way_rows(
    loaded: list[LoadedFinalEpochMetrics],
    metric_names: list[str],
) -> list[dict[str, object]]:
    if len(loaded) < 3:
        return []
    rows = []
    for metric in metric_names:
        values = [
            (row.metrics.get(metric, np.nan), row)
            for row in loaded
            if np.isfinite(row.metrics.get(metric, np.nan))
        ]
        if len(values) < 3:
            continue
        ranked = sorted(values, key=lambda item: item[0], reverse=True)
        best_value, best = ranked[0]
        worst_value, worst = ranked[-1]
        rows.append(
            {
                "metric": metric,
                "best_key": best.key,
                "best_model": best.display_name,
                "best_value": best_value,
                "worst_key": worst.key,
                "worst_model": worst.display_name,
                "worst_value": worst_value,
                "spread": best_value - worst_value,
                "ranked_models": ", ".join(row.display_name for _, row in ranked),
            }
        )
    return rows


def _plot_grouped_metrics(
    loaded: list[LoadedFinalEpochMetrics],
    metric_names: list[str],
    path: Path,
    *,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    finite_values = [
        row.metrics.get(metric, np.nan)
        for row in loaded
        for metric in metric_names
        if np.isfinite(row.metrics.get(metric, np.nan))
    ]
    y_max = max(1.0, max(finite_values, default=1.0) * 1.1)
    width = max(900, 170 * max(len(metric_names), 1))
    height = 560
    margin_left, margin_right, margin_top, margin_bottom = 80, 30, 70, 150
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = ["#2F6FED", "#D55E00", "#009E73", "#CC79A7", "#0072B2"]

    draw.text((margin_left, 20), title, fill="black", font=font)
    draw.line(
        [
            (margin_left, margin_top),
            (margin_left, margin_top + plot_height),
            (margin_left + plot_width, margin_top + plot_height),
        ],
        fill="black",
        width=1,
    )
    for tick in range(6):
        value = y_max * tick / 5.0
        y = margin_top + plot_height - int(plot_height * value / y_max)
        draw.line(
            [(margin_left, y), (margin_left + plot_width, y)],
            fill="#dddddd",
            width=1,
        )
        draw.text((10, y - 6), f"{value:.2f}", fill="black", font=font)

    group_width = plot_width / max(len(metric_names), 1)
    bar_width = max(8, int(group_width * 0.7 / max(len(loaded), 1)))
    for metric_index, metric in enumerate(metric_names):
        group_left = margin_left + metric_index * group_width
        for model_index, row in enumerate(loaded):
            value = row.metrics.get(metric, np.nan)
            if not np.isfinite(value):
                continue
            bar_height = int(plot_height * value / y_max)
            x0 = int(group_left + group_width * 0.15 + model_index * bar_width)
            x1 = x0 + bar_width - 2
            y0 = margin_top + plot_height - bar_height
            y1 = margin_top + plot_height
            draw.rectangle(
                [(x0, y0), (x1, y1)],
                fill=colors[model_index % len(colors)],
                outline="black",
            )
        draw.text(
            (int(group_left + 4), margin_top + plot_height + 10),
            metric,
            fill="black",
            font=font,
        )

    legend_x = margin_left
    legend_y = height - 35
    for index, row in enumerate(loaded):
        color = colors[index % len(colors)]
        draw.rectangle(
            [(legend_x, legend_y), (legend_x + 12, legend_y + 12)],
            fill=color,
            outline="black",
        )
        draw.text((legend_x + 18, legend_y), row.display_name, fill="black", font=font)
        legend_x += 150

    image.save(path)


def _plot_pairwise_delta(
    rows: list[dict[str, object]],
    metric_names: list[str],
    path: Path,
    *,
    title: str,
) -> None:
    values_by_metric = {str(row["metric"]): float(row["delta"]) for row in rows}
    values = [values_by_metric.get(metric, np.nan) for metric in metric_names]
    path.parent.mkdir(parents=True, exist_ok=True)
    finite_values = [abs(value) for value in values if np.isfinite(value)]
    y_abs_max = max(0.05, max(finite_values, default=0.05) * 1.15)
    width = max(900, 170 * max(len(metric_names), 1))
    height = 540
    margin_left, margin_right, margin_top, margin_bottom = 80, 30, 70, 145
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    zero_y = margin_top + plot_height // 2
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.text((margin_left, 20), title, fill="black", font=font)
    draw.line(
        [
            (margin_left, margin_top),
            (margin_left, margin_top + plot_height),
            (margin_left + plot_width, margin_top + plot_height),
        ],
        fill="black",
        width=1,
    )
    draw.line(
        [(margin_left, zero_y), (margin_left + plot_width, zero_y)],
        fill="black",
        width=1,
    )
    for tick in [-1.0, -0.5, 0.5, 1.0]:
        value = y_abs_max * tick
        y = zero_y - int((plot_height / 2.0) * value / y_abs_max)
        draw.line(
            [(margin_left, y), (margin_left + plot_width, y)],
            fill="#dddddd",
            width=1,
        )
        draw.text((10, y - 6), f"{value:.2f}", fill="black", font=font)

    group_width = plot_width / max(len(metric_names), 1)
    bar_width = max(12, int(group_width * 0.45))
    for index, metric in enumerate(metric_names):
        value = values_by_metric.get(metric, np.nan)
        if np.isfinite(value):
            x0 = int(margin_left + index * group_width + group_width * 0.25)
            x1 = x0 + bar_width
            y = zero_y - int((plot_height / 2.0) * value / y_abs_max)
            color = "#2f7d32" if value >= 0 else "#b3261e"
            draw.rectangle(
                [(x0, min(y, zero_y)), (x1, max(y, zero_y))],
                fill=color,
                outline="black",
            )
        draw.text(
            (int(margin_left + index * group_width + 4), margin_top + plot_height + 10),
            metric,
            fill="black",
            font=font,
        )

    image.save(path)


def write_final_epoch_metric_comparisons(
    specs: list[FinalEpochMetricSpec],
    output_dir: str | Path,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metric_comparison"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_final_epoch_metrics(specs)
    metric_names = _metric_names(loaded)
    score_metrics = _score_metric_names(loaded, metric_names)
    pairwise_rows = _build_pairwise_rows(loaded, metric_names)
    three_way_rows = _build_three_way_rows(loaded, metric_names)

    text_path = _write_text_summary(
        loaded=loaded,
        metric_names=metric_names,
        pairwise_rows=pairwise_rows,
        three_way_rows=three_way_rows,
        output_dir=metrics_dir,
    )
    model_metrics_path = _save_model_metric_array(loaded, metric_names, metrics_dir)
    pairwise_path = _save_pairwise_array(pairwise_rows, metrics_dir)
    three_way_path = _save_three_way_array(three_way_rows, metrics_dir)

    all_plot_path = metrics_dir / "all_models_final_epoch_metrics.png"
    _plot_grouped_metrics(
        loaded,
        score_metrics,
        all_plot_path,
        title="Final Epoch Metrics: All Models",
    )
    three_way_plot_path = None
    if len(loaded) >= 3:
        three_way_plot_path = metrics_dir / "three_way_final_epoch_metrics.png"
        _plot_grouped_metrics(
            loaded[:3],
            score_metrics,
            three_way_plot_path,
            title="Final Epoch Metrics: Three-Way Comparison",
        )

    pairwise_plot_paths: dict[str, str] = {}
    pairwise_dir = metrics_dir / "pairwise"
    for left, right in itertools.combinations(loaded, 2):
        key = f"{left.key}_vs_{right.key}"
        rows = [
            row
            for row in pairwise_rows
            if row["left_key"] == left.key and row["right_key"] == right.key
        ]
        path = pairwise_dir / f"{key}_final_epoch_metric_deltas.png"
        _plot_pairwise_delta(
            rows,
            score_metrics,
            path,
            title=f"Final Epoch Metric Deltas: {right.display_name} - {left.display_name}",
        )
        pairwise_plot_paths[key] = str(path)

    outputs: dict[str, object] = {
        "text": str(text_path),
        "model_metrics_npy": str(model_metrics_path),
        "pairwise_npy": str(pairwise_path),
        "three_way_npy": str(three_way_path) if three_way_path is not None else None,
        "all_models_plot": str(all_plot_path),
        "three_way_plot": (
            str(three_way_plot_path) if three_way_plot_path is not None else None
        ),
        "pairwise_plots": pairwise_plot_paths,
        "sources": [
            {
                "key": row.key,
                "display_name": row.display_name,
                "epoch": row.epoch,
                "metrics_path": str(row.metrics_path),
                "output_dir": str(row.output_dir),
            }
            for row in loaded
        ],
    }
    print(f"Wrote final-epoch metric comparison to {metrics_dir}", flush=True)
    return outputs
