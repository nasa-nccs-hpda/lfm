"""Vendored Fourier-VQ MultiMAE model from origin/old-main.

Kept self-contained inside terratorch_integration so nothing at the repo root
needs to change. See ../fvqmultimae_backbone.py for the TerraTorch wrapper.
"""

from .fourier_vq_vae import FourierVQMultiMAE
from .input_adapter import LunarFourierInputAdapter, LunarInputAdapter
from .output_adapter import SpatialOutputAdapter

__all__ = [
    "FourierVQMultiMAE",
    "LunarFourierInputAdapter",
    "LunarInputAdapter",
    "SpatialOutputAdapter",
]
