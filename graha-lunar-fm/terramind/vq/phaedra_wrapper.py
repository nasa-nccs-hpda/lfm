# Copyright 2025 IBM Corp.
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

"""Phaedra Tokenizer Wrapper for TerraMind compatibility.

This wrapper adapts the Phaedra tokenizer to be compatible with the
TerraMind VQVAE interface, enabling seamless integration with the
generation pipeline.
"""

from contextlib import nullcontext
from pathlib import Path

import torch

from omegaconf import OmegaConf


try:
    from phaedra import PhaedraModel, PhaedraSystem
    PHAEDRA_AVAILABLE = True
except ImportError:
    PHAEDRA_AVAILABLE = False
    PhaedraSystem = None
    PhaedraModel = None


class PhaedraWrapper(torch.nn.Module):
    """Wrapper around Phaedra tokenizer to make it compatible with TerraMind's VQVAE interface.

    Phaedra uses a hybrid tokenization approach:
    - Morphological tokens: Structural patterns (FSQ)
    - Amplitude tokens: Continuous values (high-resolution FSQ)

    This wrapper adapts Phaedra's interface to match the expected VQVAE methods:
    - encode() -> (quant, code_loss, tokens)
    - tokenize() -> tokens
    - tokens_to_embedding() -> quant
    - decode_tokens() -> reconstruction
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str | None = None,
        image_size: int | None = None,
        n_channels: int | None = None,
        ckpt_path: str | None = None,
        **kwargs,
    ):
        """Initialize Phaedra wrapper.

        Args:
            config_path: Path to Phaedra config YAML file
            checkpoint_path: Path to trained Phaedra checkpoint directory
            image_size: Input image size (optional, extracted from config if not provided)
            n_channels: Number of input channels (optional, extracted from config if not provided)
            ckpt_path: Path to checkpoint file (alternative to checkpoint_path, for VQVAE compatibility)
            **kwargs: Additional arguments (for compatibility)
        """
        super().__init__()

        if not PHAEDRA_AVAILABLE:
            raise ImportError("Phaedra is not installed.")

        self.config_path = config_path
        self.checkpoint_path = checkpoint_path or ckpt_path

        # Load configuration
        self.config = OmegaConf.load(config_path)
        # Extract image_size and n_channels from config if not provided
        if image_size is None:
            # Use the resolution from data config, or input_h from vae config
            if "data" in self.config and "resolutions" in self.config.data:
                self.image_size = self.config.data.resolutions[0]
            else:
                self.image_size = self.config.tokenizer_hyperparameters.vae_hyperparameters.input_h
        else:
            self.image_size = image_size

        if n_channels is None:
            self.n_channels = self.config.tokenizer_hyperparameters.vae_hyperparameters.input_channels
        else:
            self.n_channels = n_channels

        # Initialize Phaedra system
        self.system = PhaedraSystem(self.config)

        # Load checkpoint if provided
        if self.checkpoint_path is not None:
            self._load_checkpoint(self.checkpoint_path)

        # Store the underlying model for direct access
        self.model = self.system.model

        # Calculate spatial downsampling factor for token count calculation
        # Phaedra's encoder downsamples by 2^(number of encoder levels)
        encoder_mult = self.config.tokenizer_hyperparameters.vae_hyperparameters.encoder_channel_mult
        self.downsample_factor = 2 ** (len(encoder_mult) - 1)

        # Compatibility shims with TerraMind's VQ interface
        self.patch_size = self.downsample_factor
        self.num_codebooks = 2

    def _load_checkpoint(self, checkpoint_path: Path):
        """Load Phaedra checkpoint, prioritizing EMA weights."""
        checkpoint_path = Path(checkpoint_path)

        # Priority 1: Load EMA weights
        ema_path = checkpoint_path / "ema.pt"
        if ema_path.exists():
            try:
                from torch_ema import ExponentialMovingAverage
                ema = ExponentialMovingAverage(self.system.parameters(), decay=0.999)
                ema.load_state_dict(torch.load(ema_path, map_location="cpu", weights_only=False))
                self.system.ema = ema
                print(f"Loaded EMA weights from {ema_path}")
            except ImportError:
                print("torch_ema not available, falling back to full checkpoint")
            else:
                return

        # Fallback to full model checkpoint
        print("Loading full model checkpoint (EMA not available)...")
        if checkpoint_path.is_dir():
            safetensors_path = checkpoint_path / "model.safetensors"

            if safetensors_path.exists():
                self._load_state_file(safetensors_path)
                return

        elif checkpoint_path.is_file():
            self._load_state_file(checkpoint_path)
            return

        print(f"Could not load checkpoint from {checkpoint_path}")

    def _load_state_file(self, path: Path):
        """Load a state dict from a .safetensors file."""
        assert path.suffix == ".safetensors", f"File must be of safetensors, but found {path.suffix}"
        from safetensors.torch import load_file
        state_dict = load_file(str(path), device="cpu")

        try:
            self.system.load_state_dict(state_dict)
            print(f"Loaded Phaedra system state from {path}")
        except RuntimeError:
            # Keys might be rooted at inner model
            self.system.model.load_state_dict(state_dict)
            print(f"Loaded Phaedra inner-model state from {path}")

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input to quantized representation and tokens.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            quant: Quantized latent code [B, D, H_Q, W_Q]
            code_loss: Quantization loss (scalar tensor)
            tokens: Token tensor [B, L, C] where L=H_Q*W_Q, C=2 [morph_token, amp_token]
        """
        quant, emb_loss, tokens_hierarchy, _ = self.model.encode(x)

        # tokens_hierarchy is [morph_tokens_list, amp_tokens]
        # morph_tokens_list is a list of tensors (hierarchical levels)
        # amp_tokens is a tensor [B, H, W]
        morph_tokens_list, amp_tokens = tokens_hierarchy

        # Use the finest morphological level (last in list)
        if isinstance(morph_tokens_list, list):
            morph_tokens = morph_tokens_list[-1]  # [B, H, W]
        else:
            morph_tokens = morph_tokens_list

        # Stack tokens: [B, H, W, 2] where last dim is [morph, amp]
        B, H, W = amp_tokens.shape
        tokens_spatial = torch.stack([morph_tokens, amp_tokens], dim=-1)  # [B, H, W, 2]

        # Reshape to [B, L, C] format to match other tokenizers with multiple codebooks
        # This gives L=H*W spatial locations, each with C=2 codebook indices
        tokens = tokens_spatial.reshape(B, H * W, 2)  # [B, L, 2]

        return quant, emb_loss, tokens

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize input image.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            tokens: Token tensor [B, L, 2] where L=H_Q*W_Q, [morph_token, amp_token]
        """
        # produce_tokens expects a batch dict, but we can call encode directly
        _, _, tokens = self.encode(x)
        return tokens

    def tokens_to_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert tokens back to quantized embeddings.

        Args:
            tokens: Token tensor [B, L, C] where L=H_Q*W_Q and C=2 [morph_token, amp_token]

        Returns:
            quant: Quantized latent code [B, D, H_Q, W_Q]
        """
        # Ensure tokens are in [B, L, C] format
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens with shape [B, L, C], got shape {tokens.shape}")

        B, L, C = tokens.shape
        if C != 2:
            raise ValueError(f"Expected C=2 codebooks for Phaedra, got C={C}")

        # Reshape from [B, L, 2] to [B, H, W, 2]
        H = W = int(L ** 0.5)
        if H * W != L:
            raise ValueError(f"Token sequence length {L} must be a perfect square")

        tokens_spatial = tokens.reshape(B, H, W, C)  # [B, H, W, 2]

        # Split tokens into morphological and amplitude
        morph_tokens = tokens_spatial[..., 0]  # [B, H, W]
        amp_tokens = tokens_spatial[..., 1]  # [B, H, W]

        # Get embeddings from codebooks
        morph_embeddings = self.model.quantizer.get_codebook_entry(morph_tokens)  # [B, D-1, H, W]
        amp_embeddings = self.model.approximate_continuous.get_codebook_entry(amp_tokens)  # [B, 1, H, W]

        # Concatenate to form full quantized representation
        quant = torch.cat([morph_embeddings, amp_embeddings], dim=1)  # [B, D, H, W]

        return quant

    def decode_tokens(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode tokens directly to reconstruction.

        Args:
            tokens: Token tensor [B, H_Q, W_Q, C] where C=2 [morph_token, amp_token]

        Returns:
            reconstruction: Decoded image [B, C, H, W]
        """
        # Extract morphological and amplitude tokens
        morph_tokens = tokens[..., 0]  # [B, H, W]
        amp_tokens = tokens[..., 1]  # [B, H, W]

        # Use EMA weights if available (critical for quality reconstruction)
        if hasattr(self.system, "ema") and self.system.ema is not None:
            context_manager = self.system.ema.average_parameters()
        else:
            context_manager = nullcontext()

        # Phaedra's predict_from_tokens expects [morph_tokens, amp_tokens] order
        with context_manager:
            reconstruction = self.system.predict_from_tokens([morph_tokens, amp_tokens])

        return reconstruction

    def decode_quant(self, quant: torch.Tensor) -> torch.Tensor:
        """Decode quantized representation to reconstruction.

        Args:
            quant: Quantized latent code [B, D, H_Q, W_Q]

        Returns:
            reconstruction: Decoded image [B, C, H, W]
        """
        return self.model.decode(quant)

    def autoencode(self, x: torch.Tensor) -> torch.Tensor:
        """Full autoencoding: encode -> quantize -> decode.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            reconstruction: Decoded image [B, C, H, W]
        """
        quant, _, _ = self.encode(x)
        reconstruction = self.decode_quant(quant)
        return reconstruction

    def forward(self, x: torch.Tensor, mode: str = "default"):
        """Forward pass with different modes.

        Args:
            x: Input tensor
            mode: One of "default", "encode", "decode"

        Returns:
            Depends on mode
        """
        if mode == "encode":
            return self.encode(x)
        elif mode == "decode":
            return self.decode_quant(x)
        else:
            # Default: full forward pass
            return self.model(x, mode=mode)
