from .vqvae import VQ, VQVAE, DiVAE
from .scheduling import *

# Phaedra tokenizer wrapper (optional dependency)
try:
    from .phaedra_wrapper import PhaedraWrapper
    PHAEDRA_AVAILABLE = True
except ImportError:
    PHAEDRA_AVAILABLE = False
    PhaedraWrapper = None

__all__ = ['VQ', 'VQVAE', 'DiVAE', 'PhaedraWrapper', 'PHAEDRA_AVAILABLE']
