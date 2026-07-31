"""Pixel-wise regression task with the shared LLRD + split-group optimiser
recipe (:class:`_LunarLLRDMixin`).

``terratorch.tasks.PixelwiseRegressionTask`` accepts a single flat ``lr`` and
forwards it to every parameter in the model. For a Lunar-FM fine-tune with
*new* modalities, that's the worst of both worlds:

* The **pretrained encoder** wants a small LR so we don't wash out pretrained
  features.
* The **randomly-initialised new-modality embedders** live *inside* the
  encoder but were just born, so they need a large LR — same order as a fresh
  decoder / head.
* The **decoder + necks + regression head** are fresh weights and also want a
  large LR.

LLRD solves the "deep vs. shallow encoder" part; the shared new-modality rule
solves the "random vs. pretrained inside the encoder" part.

See :mod:`terratorch_integration.lunar_llrd_mixin` for the full recipe.
"""

from __future__ import annotations

from terratorch.tasks import PixelwiseRegressionTask

from .lunar_llrd_mixin import _LunarLLRDMixin


class LunarPixelwiseRegressionTask(_LunarLLRDMixin, PixelwiseRegressionTask):
    """Pixel-wise regression with LLRD, split head/backbone LR, and a
    dedicated LR group for randomly-initialised new-modality embedders.

    All optimiser knobs (``backbone_lr``, ``head_lr``, ``layer_decay``,
    ``weight_decay``, ``head_weight_decay``, ``warmup_steps``, ``cosine_t_max``,
    ``eta_min``, ``betas``) come from :class:`_LunarLLRDMixin`. Any other
    keyword argument is forwarded to
    :class:`~terratorch.tasks.PixelwiseRegressionTask`.
    """
