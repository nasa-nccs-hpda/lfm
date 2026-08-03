import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, pack, unpack


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def round_ste(x):
    """Round with straight through gradients."""
    xhat = x.round()
    return x + (xhat - x).detach()


class FiniteScalarQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size: str,  # example: "8-8-8-6-5"
    ):
        super().__init__()

        levels = [int(level) for level in codebook_size.split("-")]

        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", _levels, persistent=False)
        _basis = torch.cumprod(
            torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32
        )
        self.register_buffer("_basis", _basis, persistent=False)

        # initialize codebook
        self.codebook_size = self._levels.prod().item()
        codebook = self.indice_to_code(torch.arange(self.codebook_size))
        self.register_buffer("codebook", codebook, persistent=False)

    def latent_to_code_and_indice(self, latent):
        # Reshape latent to match number of quantization levels
        original_shape = latent.shape
        
        # Flatten all dimensions except the last one
        flat_shape = (-1, original_shape[-1])
        latent_flat = latent.reshape(flat_shape)
        
        if latent_flat.shape[-1] % len(self._levels) == 0:
            # Reshape to group channels into chunks matching number of levels
            group_size = latent_flat.shape[-1] // len(self._levels)
            latent_grouped = latent_flat.reshape(latent_flat.shape[0], len(self._levels), group_size)
            latent_grouped = latent_grouped.mean(dim=-1)  # Average each group
            
            d = self._levels - 1
            number = round_ste(F.sigmoid(latent_grouped) * d)
            code = number / d
            indice = (number * self._basis).sum(dim=-1).to(torch.int32)
            
            # Reshape code back to original dimensions
            code = code.unsqueeze(-1).repeat_interleave(group_size, dim=-1)
            code = code.reshape(original_shape)
            
            return code, indice
        else:
            raise ValueError(f"Input dimension {latent_flat.shape[-1]} must be divisible by number of levels {len(self._levels)}")

    def indice_to_code(self, indice):
        # (..., d)
        code = (indice.unsqueeze(-1) // self._basis) % self._levels
        # convert to [0, 1]
        code = code / (self._levels - 1)
        return code

    def indices_to_embedding(self, indices):
        B, H, W = indices.shape
        indices = rearrange(indices, "b h w -> b (h w)")
        embeddings = self.indice_to_code(indices)
        embeddings = rearrange(embeddings, "b (h w) c -> b c h w", h=H)
        return embeddings

    def forward(self, x):
        quantize, embed_ind = self.latent_to_code_and_indice(x)
        
        # no auxiliary losses needed for FSQ
        loss = torch.tensor([0.0], device=x.device, requires_grad=self.training)
        
        return quantize, loss, None