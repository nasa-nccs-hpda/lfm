"""Parity checks for the remote Graha inference preprocessing path."""

import numpy as np
import torch
from torch import nn

from lfm.all_models.all_tasks.data.normalization import ZScoreNormalization
from lfm.all_models.sem_seg.data_cube_inference import preprocess_datacubes
from lfm.all_models.sem_seg.graha_inference import GrahaLogitModel


def test_preprocess_matches_graha_training_zscore_pipeline():
    """Inference preprocessing must match Graha semantic training."""
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
    expected = ZScoreNormalization(
        np.asarray(means), np.asarray(stds)
    ).apply(image_hwc, band_filter=[0, 1, 2])
    actual = preprocess_datacubes(
        image_chw[None], means=means, stds=stds
    )[0]

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_preprocess_excludes_nodata_from_scaling_and_fills_it_for_model():
    image_chw = np.array(
        [[[[-32768.0, 1.0], [3.0, 5.0]]]], dtype=np.float32
    )
    nodata_mask = np.array(
        [[[[True, False], [False, False]]]], dtype=bool
    )
    actual = preprocess_datacubes(
        image_chw,
        means=[0.25],
        stds=[0.5],
        nodata_masks=nodata_mask,
    )[0]

    # The invalid pixel is mean-imputed before z-scoring, so it reaches the
    # model as zero rather than the source -32768 sentinel.
    assert actual[0, 0, 0] == 0.0
    np.testing.assert_allclose(
        actual[0, 0, 1], 1.5, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        actual[1, :, 0], [5.5, 9.5], rtol=1e-6, atol=1e-6
    )


class _TensorOutputTask(nn.Module):
    def forward(self, image):
        return image + 1.0


def test_graha_logit_model_forwards_task_output():
    image = torch.zeros(1, 1, 2, 2)
    output = GrahaLogitModel(_TensorOutputTask())(image)
    torch.testing.assert_close(output, torch.ones_like(image))
