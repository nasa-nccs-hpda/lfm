# Copyright 2024 EPFL and Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Tuple, Dict, Optional, Union, Any
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from .quantizers import VectorQuantizerLucid, Memcodes, FiniteScalarQuantizer


class VQVAE(nn.Module, PyTorchModelHubMixin):
    """Vector Quantized model that combines an encoder and decoder with a discrete bottleneck.

    Args:
        encoder: The encoder module that converts input to latent representations
        decoder: The decoder module that reconstructs input from latent representations
        latent_dim: Dimensionality of the latent code
        codebook_size: Number of codebook entries
        num_codebooks: Number of parallel codebooks to use
        norm_codes: Whether to normalize the codebook entries to the unit sphere
        norm_latents: Whether to normalize the latent codes for computing commitment loss
        sync_codebook: Enable for multi-GPU training, disable for single GPU
        ema_decay: Decay rate for the exponential moving average of codebook entries
        threshold_ema_dead_code: Threshold for replacing stale codes
        code_replacement_policy: Policy for replacing stale codes ('batch_random' or 'linde_buzo_gray')
        commitment_weight: Weight for the quantizer commitment loss
        kmeans_init: Whether to initialize codebook entries with k-means clustering
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int = 32,
        quant_type: str = "lucid",
        codebook_size: int = 16384,
        num_codebooks: int = 1,
        norm_codes: bool = True,
        norm_latents: bool = False,
        sync_codebook: bool = True,
        ema_decay: float = 0.99,
        threshold_ema_dead_code: float = 0.25,
        code_replacement_policy: str = "batch_random",
        commitment_weight: float = 1.0,
        kmeans_init: bool = False,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.norm_codes = norm_codes
        self.norm_latents = norm_latents
        self.sync_codebook = sync_codebook
        self.ema_decay = ema_decay
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.code_replacement_policy = code_replacement_policy
        self.commitment_weight = commitment_weight
        self.kmeans_init = kmeans_init

        # Init quantizer
        if quant_type == "lucid":
            self.quantize = VectorQuantizerLucid(
                dim=latent_dim,
                codebook_size=codebook_size,
                codebook_dim=latent_dim,
                heads=num_codebooks,
                use_cosine_sim=norm_codes,
                threshold_ema_dead_code=threshold_ema_dead_code,
                code_replacement_policy=code_replacement_policy,
                sync_codebook=sync_codebook,
                decay=ema_decay,
                commitment_weight=self.commitment_weight,
                norm_latents=norm_latents,
                kmeans_init=kmeans_init,
            )
        elif quant_type == "memcodes":
            self.quantize = Memcodes(
                dim=latent_dim,
                codebook_size=codebook_size,
                heads=num_codebooks,
                temperature=1.0,
            )
        elif quant_type == "fsq":
            self.quantize = FiniteScalarQuantizer(codebook_size=codebook_size)
        else:
            raise ValueError(f"{quant_type} not a valid quant_type.")

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        """Encodes an input tensor and quantizes the latent code.

        Args:
            x: Input tensor

        Returns:
            quant: Quantized latent code
            code_loss: Codebook loss
            tokens: Quantized indices
        """
        h = self.encoder(x)
        quant, code_loss, tokens = self.quantize(h)
        return quant, code_loss, tokens

    def decode(self, quant: torch.Tensor) -> torch.Tensor:
        """Decodes quantized latent codes back to the input space.

        Args:
            quant: Quantized latent code

        Returns:
            Decoded tensor
        """
        return self.decoder(quant)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder, quantizer, and decoder.

        Args:
            x: Input tensor

        Returns:
            dec: Decoded tensor
            code_loss: Codebook loss
        """
        quant, code_loss, _ = self.encode(x)
        dec = self.decode(quant)
        return dec, code_loss

    def tokens_to_embedding(self, tokens: torch.LongTensor) -> torch.Tensor:
        """Convert token indices to embeddings.

        Args:
            tokens: Quantized indices of shape B H_Q W_Q

        Returns:
            Quantized latent code of shape B D_Q H_Q W_Q
        """
        return self.quantize.indices_to_embedding(tokens)

    def freeze(self, modules: Union[str, List[str]]) -> None:
        """Freeze model parameters containing the given module name(s).

        Args:
            modules: String or list of strings. Parameters containing these strings
                    in their names will be frozen.
        """
        if isinstance(modules, str):
            modules = [modules]

        for module in modules:
            if "encoder" in module:
                for name, param in self.encoder.named_parameters():
                    param.requires_grad = False
            elif "decoder" in module:
                for name, param in self.decoder.named_parameters():
                    param.requires_grad = False
            else:
                # For other modules, use the original name-based freezing
                for name, param in self.named_parameters():
                    if module in name:
                        param.requires_grad = False
