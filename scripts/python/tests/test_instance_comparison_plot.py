"""Check the local figure slice without importing the training runtime."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lfm.all_models.instance_comparison_plot import (
    InstanceComparisonChip,
    instance_partitions_match,
    plot_instance_comparison,
    validate_instance_comparison_sources,
)


def test_dummy_cli_renders_four_rows_without_loading_ml_packages(tmp_path):
    root = Path(__file__).resolve().parents[3]
    script = root / "scripts/python/instance_seg/preview_instance_comparison.py"
    command = (
        "import runpy, sys; "
        "runpy.run_path(sys.argv.pop(1), run_name='__main__'); "
        "assert not {'torch', 'lightning', 'transformers'} & sys.modules.keys()"
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            command,
            str(script),
            "--output-dir",
            str(tmp_path),
            "--chip-size",
            "32",
            "--dpi",
            "40",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((tmp_path / "preview_manifest.json").read_text())
    assert manifest["rows"] == [
        "input",
        "toy_prediction",
        "full_model_prediction",
        "ground_truth",
    ]
    assert len(manifest["columns"]) == 3
    assert manifest["synthetic"] is True
    assert manifest["performance_metrics"] is None
    with Image.open(tmp_path / manifest["figure"]) as image:
        image.verify()


def test_instance_alignment_accepts_renumbering_but_rejects_merged_objects():
    first = np.array([[0, 1, 1], [0, 2, 2]])
    renumbered = np.array([[0, 8, 8], [0, 3, 3]])
    merged = np.array([[0, 8, 8], [0, 8, 8]])
    assert instance_partitions_match(first, renumbered)
    assert not instance_partitions_match(first, merged)
    assert not instance_partitions_match(first, np.roll(renumbered, 1, axis=1))


def test_cached_sources_allow_model_specific_normalization_and_check_targets():
    mask = np.array([[0, 1], [0, 1]])
    toy = {"image": np.zeros((70, 2, 2)), "gt_mask": mask, "pred_mask": mask}
    graha = {"image": np.ones((70, 2, 2)), "gt_mask": mask * 3, "pred_mask": mask}
    caches = {"toy": {"chip": toy}, "graha": {"chip": graha}}
    validate_instance_comparison_sources(caches, ["chip"])
    graha["gt_mask"] = np.roll(mask, 1, axis=1)
    with pytest.raises(ValueError, match="ground truth differs"):
        validate_instance_comparison_sources(caches, ["chip"])


def test_figure_rejects_predictions_for_a_different_chip_geometry(tmp_path):
    chip = InstanceComparisonChip(
        key="mismatched",
        display_image=np.zeros((8, 8, 3)),
        ground_truth=np.zeros((8, 8), dtype=int),
        predictions={"toy": np.zeros((4, 4), dtype=int)},
    )
    with pytest.raises(ValueError, match="prediction and ground truth must align"):
        plot_instance_comparison(
            [chip], tmp_path / "bad.png", model_labels={"toy": "Toy"}
        )


def test_one_chip_empty_predictions_and_nodata_render(tmp_path):
    mask = np.zeros((16, 16), dtype=int)
    mask[0, :] = -1
    image = np.zeros((16, 16, 3))
    image[0, :] = np.nan
    chip = InstanceComparisonChip(
        key="empty",
        display_image=image,
        ground_truth=mask,
        predictions={"toy": np.zeros_like(mask), "graha": np.zeros_like(mask)},
    )
    path = plot_instance_comparison(
        [chip],
        tmp_path / "empty.png",
        model_labels={"toy": "Toy", "graha": "GRAHA"},
        dpi=40,
    )
    with Image.open(path) as rendered:
        pixels = np.asarray(rendered.convert("RGB"))
    assert np.any(np.all(pixels == [77, 77, 77], axis=-1))
