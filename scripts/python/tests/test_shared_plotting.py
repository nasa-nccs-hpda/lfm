"""Shared policies preserve each task's display behavior and figure lifetime."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from lfm.all_models.all_tasks.utils.display import (
    create_colored_overlay_image,
    prepare_image_for_display,
)
from lfm.all_models.plotting import (
    DisplayPolicy,
    blend_overlay,
    draw_boxes,
    figure_grid,
    prepare_display,
)
from lfm.all_models.sem_seg.plotting.semantic_plotting import (
    plot_prediction_cache_comparison,
)


def test_legacy_global_stretch_and_valid_channel_stretch_remain_distinct():
    image = np.arange(5 * 3 * 4, dtype=np.float32).reshape(3, 4, 5)
    displayed, note = prepare_image_for_display(image)
    selected = image[:, :, [3, 1, 0]]
    np.testing.assert_allclose(
        displayed,
        (selected - selected.min()) / (selected.max() - selected.min() + 1e-8),
    )
    assert note == "5ch (RGB from bands 3,1,0)"
    valid = np.ones((3, 4), dtype=bool)
    valid[0] = False
    image[~valid] = np.nan
    stretched = prepare_display(
        image, valid=valid, policy=DisplayPolicy((3, 1, 0), "channel_minmax")
    )
    assert not stretched[~valid].any()
    np.testing.assert_allclose(stretched[valid].min(axis=0), 0)
    np.testing.assert_allclose(stretched[valid].max(axis=0), 1)


def test_explicit_foreground_retains_semantic_and_instance_predicates():
    image = np.zeros((2, 2))
    mask = np.array([[0, 1], [2, -1]])
    semantic = create_colored_overlay_image(image, mask, color=(1, 0, 0))
    instance = blend_overlay(image, (1, 0, 0), foreground=mask > 0, alpha=0.8)
    np.testing.assert_array_equal(semantic[:, :, 0], [[0, 0.5], [0, 0]])
    np.testing.assert_array_equal(instance[:, :, 0], [[0, 0.8], [0.8, 0]])


def test_single_column_boxes_and_cleanup_on_save_failure(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    before = plt.get_fignums()
    with figure_grid(2, 1, figsize=(3, 5)) as grid:
        assert grid.axes.shape == (2, 1)
        draw_boxes(
            grid.axes[0, 0],
            np.array([[1, 2, 4, 6], [3, 3, 1, 1]]),
            scores=np.array([0.75, 1]),
        )
        patch = grid.axes[0, 0].patches[0]
        assert (patch.get_width(), patch.get_height()) == (3, 4)
        assert grid.axes[0, 0].texts[0].get_text() == "0.75"
    assert plt.get_fignums() == before

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Figure, "savefig", fail)
    with (
        pytest.raises(OSError, match="disk full"),
        figure_grid(1, 1, figsize=(3, 3)) as grid,
    ):
        grid.save(tmp_path / "failure.png")
    assert plt.get_fignums() == before


def test_real_semantic_plotter_one_column_and_cleanup_on_render_error(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    sample = {
        "sample_key": "chip",
        "image_path": "chip.tif",
        "label_path": "chip_label.npz",
        "image": np.ones((5, 8, 8)),
        "label": np.zeros((8, 8)),
        "pred": np.zeros((8, 8)),
        "prob": np.zeros((8, 8)),
    }
    np.savez_compressed(cache / "one.npz", **sample)
    (cache / "manifest.json").write_text(
        json.dumps([{"file": "one.npz", "sample_key": "chip"}])
    )
    before = plt.get_fignums()
    path = plot_prediction_cache_comparison(
        {"toy": cache}, tmp_path / "plots", n_samples=1, dpi=30
    )
    assert path.is_file()
    assert plt.get_fignums() == before
    other = tmp_path / "other-cache"
    other.mkdir()
    np.savez_compressed(other / "one.npz", **{**sample, "image_path": "other-chip.tif"})
    with pytest.raises(ValueError, match="image_path"):
        plot_prediction_cache_comparison(
            {"toy": cache, "graha": other}, tmp_path / "mismatched", n_samples=1, dpi=30
        )
    assert plt.get_fignums() == before
    np.savez_compressed(cache / "one.npz", **{**sample, "image": np.ones((5, 8))})
    with pytest.raises(ValueError):
        plot_prediction_cache_comparison(
            {"toy": cache}, tmp_path / "bad", n_samples=1, dpi=30
        )
    assert plt.get_fignums() == before


def test_real_instance_plotters_share_single_column_figure_lifetime(tmp_path):
    from lfm.all_models.inst_seg.plotting.instance_plotting import (
        plot_instance_cache_comparison,
        plot_instance_cache_predictions,
    )

    cache = tmp_path / "instances"
    cache.mkdir()
    mask = np.zeros((8, 8), dtype=np.int32)
    mask[2:5, 2:5] = 1
    np.savez_compressed(
        cache / "one.npz",
        sample_key="chip",
        image_path="chip.tif",
        image=np.ones((5, 8, 8)),
        gt_mask=mask,
        pred_mask=mask,
        gt_boxes=np.array([[2, 2, 5, 5]]),
        pred_boxes=np.array([[2, 2, 5, 5]]),
        pred_scores=np.array([0.8]),
    )
    before = plt.get_fignums()
    one = plot_instance_cache_predictions(
        cache, tmp_path / "one", model_name="toy", n_samples=1, dpi=30
    )
    detailed = plot_instance_cache_comparison(
        {"toy": cache, "graha": cache},
        tmp_path / "both",
        layout="detailed",
        n_samples=1,
        dpi=30,
    )
    assert one.is_file() and detailed.is_file()
    assert plt.get_fignums() == before
