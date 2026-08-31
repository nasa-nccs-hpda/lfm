"""Parity checks for the remote Graha inference preprocessing path."""

import numpy as np
import torch
from torch import nn

from lfm.all_models.all_tasks.data.base_dataset import LunarSegmentationDataset
from lfm.all_models.all_tasks.data.normalization import ZScoreNormalization
from lfm.all_models.sem_seg.data_cube_inference import preprocess_datacubes
from lfm.all_models.sem_seg.graha_inference import GrahaLogitModel


def test_preprocess_matches_training_minmax_then_zscore():
    """Inference preprocessing must equal the training dataset pipeline."""
    image_chw = np.array(
        [
            [[1.0, 2.0], [3.0, 5.0]],
            [[10.0, 20.0], [30.0, 50.0]],
            [[-5.0, 0.0], [5.0, 15.0]],
        ],
        dtype=np.float32,
    )
    means = [0.25, 0.5, 0.75]
    stds = [0.5, 0.25, 0.125]

    image_hwc = np.moveaxis(image_chw, 0, -1)
    training_scaled = LunarSegmentationDataset.min_max_scale_bands(image_hwc)
    expected = ZScoreNormalization(
        np.asarray(means), np.asarray(stds)
    ).apply(training_scaled, band_filter=[0, 1, 2])
    actual = preprocess_datacubes(
        image_chw[None], means=means, stds=stds
    )[0]

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_preprocess_excludes_nodata_from_scaling_and_preserves_it():
    image_chw = np.array(
        [[[[-32768.0, 1.0], [3.0, 5.0]]]], dtype=np.float32
    )
    nodata_mask = np.array(
        [[[[True, False], [False, False]]]], dtype=bool
    )
    actual = preprocess_datacubes(
        image_chw,
        means=[0.0],
        stds=[1.0],
        nodata_masks=nodata_mask,
    )[0]

    assert actual[0, 0, 0] == -32768.0
    np.testing.assert_allclose(
        actual[0, 1, 0], 0.0, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        actual[1, :, 0], [0.5, 1.0], rtol=1e-6, atol=1e-6
    )


class _TensorOutputTask(nn.Module):
    def forward(self, image):
        return image + 1.0


def test_graha_logit_model_forwards_task_output():
    image = torch.zeros(1, 1, 2, 2)
    output = GrahaLogitModel(_TensorOutputTask())(image)
    torch.testing.assert_close(output, torch.ones_like(image))
