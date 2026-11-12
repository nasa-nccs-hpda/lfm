"""Render the shared four-row comparison with synthetic data and no ML runtime.

Example from the repository root:
    python scripts/python/instance_seg/preview_instance_comparison.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from lfm.all_models.instance_comparison_plot import (
    InstanceComparisonChip,
    plot_instance_comparison,
)


def make_dummy_chips(
    *,
    n_samples: int = 3,
    chip_size: int = 128,
    seed: int = 42,
) -> list[InstanceComparisonChip]:
    """Generate reproducible 70-band inputs and illustrative detection errors."""
    if n_samples < 1 or chip_size < 32:
        raise ValueError("Use at least one sample and chip_size >= 32.")
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:chip_size, :chip_size] / chip_size
    chips = []
    for index in range(n_samples):
        case = index % 3
        circles = (
            [(0.27, 0.32, 0.12), (0.65, 0.34, 0.16), (0.49, 0.72, 0.09)]
            if case == 0
            else [(0.3, 0.4, 0.15), (0.7, 0.68, 0.1)]
            if case == 1
            else []
        )
        gt = np.zeros((chip_size, chip_size), dtype=np.int32)
        terrain = 0.35 + 0.12 * xx + 0.07 * np.sin(9 * xx + 5 * yy)
        terrain += rng.normal(0, 0.025, terrain.shape)
        for identity, (cx, cy, radius) in enumerate(circles, start=1):
            distance = np.hypot(xx - cx, yy - cy)
            gt[distance < radius] = identity
            terrain -= 0.16 * np.exp(-((distance / radius) ** 4))
            terrain += 0.28 * np.exp(-(((distance - radius) / 0.015) ** 2))
        bands = np.stack(
            [
                np.clip(terrain + 0.015 * np.sin((band + 1) * xx + 3 * yy), 0, 1)
                for band in range(70)
            ]
        ).astype(np.float32)
        valid = ~((xx > 0.78) & (yy > 0.78))
        bands[:, ~valid] = np.nan
        image = bands[[3, 1, 0]].transpose(1, 2, 0)
        toy, graha = gt.copy(), gt.copy()
        if case == 0:
            toy[toy == 3] = 0  # Missed crater.
            graha = np.roll(graha, 1, axis=1)  # Slight boundary error.
        elif case == 1:
            toy[np.hypot(xx - 0.7, yy - 0.27) < 0.07] = 3  # False positive.
            graha[graha == 2] = 0  # Missed crater.
        else:
            toy[np.hypot(xx - 0.5, yy - 0.5) < 0.1] = 1  # Empty GT case.
        gt[~valid] = -1
        toy[~valid] = 0
        graha[~valid] = 0
        case_name = [
            "missed crater",
            "false positive / missed crater",
            "empty ground truth",
        ][case]
        chips.append(
            InstanceComparisonChip(
                key=f"fake_chip_{index + 1:02d}",
                display_image=image,
                ground_truth=gt,
                predictions={"toy": toy, "graha": graha},
                valid_mask=valid,
                display_note=f"{case_name}\n70 bands · RGB from 3,1,0",
            )
        )
    return chips


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scripts/outputs/instance_comparison_preview"),
    )
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--chip-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Overlay instances on the input without adding rows.",
    )
    args = parser.parse_args(argv)
    if args.n_samples < 1 or args.chip_size < 32 or args.dpi < 1:
        parser.error(
            "--n-samples and --dpi must be positive; --chip-size must be >= 32"
        )
    chips = make_dummy_chips(
        n_samples=args.n_samples, chip_size=args.chip_size, seed=args.seed
    )
    figure = plot_instance_comparison(
        chips,
        args.output_dir / "test_chip_comparison_dummy.png",
        model_labels={
            "toy": "Toy prediction",
            "graha": "Full model (GRAHA) prediction",
        },
        title="Input / Toy / full model / ground truth",
        synthetic=True,
        overlay=args.overlay,
        dpi=args.dpi,
    )
    manifest = {
        "synthetic": True,
        "training_run": False,
        "performance_metrics": None,
        "figure": figure.name,
        "seed": args.seed,
        "input_bands": 70,
        "display_bands": [3, 1, 0],
        "chip_size": args.chip_size,
        "rows": ["input", "toy_prediction", "full_model_prediction", "ground_truth"],
        "columns": [chip.key for chip in chips],
        "overlay": args.overlay,
    }
    (args.output_dir / "preview_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Synthetic comparison preview: {figure.resolve()}")
    return figure


if __name__ == "__main__":
    main()
