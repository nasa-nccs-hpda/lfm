"""Vector Quantization module for HelioFM."""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import torch.nn.functional as F

from .quantizers.quantize_lucid import VectorQuantize as VectorQuantizerLucid
from .quantizers.quantize_memcodes import Memcodes
from .quantizers.quantize_finite_scalar import FiniteScalarQuantizer

class VQ2(nn.Module):
    """A standalone Vector Quantization module that can be plugged into MultiMAE.

    This class handles the quantization of encoder tokens, preserving global tokens if present.
    It supports different quantizer types and can be easily integrated into existing architectures.

    Args:
        dim: Dimensionality of input tokens
        codebook_size: Number of codebook entries
        codebook_dim: Dimensionality of codebook entries (defaults to dim if None)
        heads: Number of parallel codebooks to use
        decay: Decay rate for the exponential moving average of codebook entries
        eps: Small epsilon value for numerical stability
        kmeans_init: Whether to initialize codebook entries with k-means clustering
        kmeans_iters: Number of k-means iterations for initialization
        use_cosine_sim: Whether to use cosine similarity instead of L2 distance
        threshold_ema_dead_code: Threshold for replacing stale codes
        code_replacement_policy: Policy for replacing stale codes ('batch_random' or 'linde_buzo_gray')
        channel_last: Whether input is in channel-last format
        accept_image_fmap: Whether to accept image feature maps
        commitment_weight: Weight for the quantizer commitment loss
        orthogonal_reg_weight: Weight for orthogonal regularization loss
        orthogonal_reg_active_codes_only: Whether to apply orthogonal reg only to active codes
        orthogonal_reg_max_codes: Maximum number of codes for orthogonal regularization
        sample_codebook_temp: Temperature for codebook sampling
        sync_codebook: Enable for multi-GPU training, disable for single GPU
        norm_latents: Whether to normalize the latent codes for computing commitment loss
        quant_type: Type of quantizer to use ('lucid', 'memcodes', or 'fsq')
        num_tokens_to_preserve: Number of tokens to preserve without quantization (e.g., global tokens)
        dtype: Data type for the quantizer
    """
    def __init__(
        self,
        dim,
        codebook_size=8192,
        codebook_dim=None,
        heads=1,
        decay=0.99,
        eps=1e-5,
        kmeans_init=True,
        kmeans_iters=50,
        use_cosine_sim=True,
        threshold_ema_dead_code=0.2,
        code_replacement_policy="batch_random",  # batch_random or linde_buzo_gray
        channel_last=False,
        accept_image_fmap=False,
        commitment_weight=1.0,
        orthogonal_reg_weight=0.05,
        orthogonal_reg_active_codes_only=True,
        orthogonal_reg_max_codes=1000,
        sample_codebook_temp=0.5,
        sync_codebook=True,
        norm_latents=True,
        quant_type: str = 'lucid',
        num_tokens_to_preserve: int = 0,
        dtype=torch.bfloat16
    ):
        super().__init__()
        
        self.dim = dim
        self.num_tokens_to_preserve = num_tokens_to_preserve
        self.commitment_weight = commitment_weight

        # Set default codebook_dim if not provided
        if codebook_dim is None:
            codebook_dim = dim

        # Initialize the vector quantizer
        if quant_type == 'lucid':
            self.quantizer = VectorQuantizerLucid(
                dim=dim,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                heads=heads,
                decay=decay,
                eps=eps,
                kmeans_init=kmeans_init,
                kmeans_iters=kmeans_iters,
                use_cosine_sim=use_cosine_sim,
                threshold_ema_dead_code=threshold_ema_dead_code,
                code_replacement_policy=code_replacement_policy,
                channel_last=channel_last,
                accept_image_fmap=accept_image_fmap,
                commitment_weight=commitment_weight,
                orthogonal_reg_weight=orthogonal_reg_weight,
                orthogonal_reg_active_codes_only=orthogonal_reg_active_codes_only,
                orthogonal_reg_max_codes=orthogonal_reg_max_codes,
                sample_codebook_temp=sample_codebook_temp,
                sync_codebook=sync_codebook,
                norm_latents=norm_latents,
            )
        elif quant_type == 'memcodes':
            self.quantizer = Memcodes(
                dim=dim,
                codebook_size=codebook_size,
                heads=heads,
                temperature=1.,
            )
        elif quant_type == 'fsq':
            # codebook_size = "-".join(["8"] * (dim // 8)) 
            codebook_size = "8-8-8-8" # TODO: change to config or calculation
            self.quantizer = FiniteScalarQuantizer(codebook_size=codebook_size)
        else:
            raise ValueError(f'{quant_type} not a valid quant_type.')

        self.quantizer = self.quantizer.to(dtype)

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the vector quantizer.
        
        Args:
            tokens: Input tokens of shape (B, N, D) where:
                   B is batch size
                   N is number of tokens
                   D is token dimension
        
        Returns:
            Tuple containing:
            - Quantized tokens with same shape as input (B, N, D)
            - Codebook loss
        """
        B, N, D = tokens.shape
        
        # Split tokens into preserved and quantized parts if needed
        if self.num_tokens_to_preserve > 0:
            preserved_tokens = tokens[:, :self.num_tokens_to_preserve]
            tokens_to_quantize = tokens[:, self.num_tokens_to_preserve:]
        else:
            tokens_to_quantize = tokens
            
        # Apply vector quantization
        quantized, vq_loss, _ = self.quantizer(tokens_to_quantize)
        
        # Recombine with preserved tokens if needed
        if self.num_tokens_to_preserve > 0:
            quantized = torch.cat([preserved_tokens, quantized], dim=1)
            
        return quantized, vq_loss

    def get_codebook_entry(self, indices: torch.LongTensor, shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        """Convert indices to their corresponding codebook entries.
        
        Args:
            indices: Tensor of codebook indices
            shape: Optional shape to reshape the output to
        
        Returns:
            Tensor of codebook entries
        """
        return self.quantizer.indices_to_embedding(indices)


class VQ(nn.Module):
    """A standalone Vector Quantization module that can be plugged into MultiMAE.
    
    This class handles the quantization of encoder tokens, preserving global tokens if present.
    It supports different quantizer types and can be easily integrated into existing architectures.
    
    Args:
        dim: Dimensionality of input tokens
        codebook_size: Number of codebook entries
        num_codebooks: Number of parallel codebooks to use
        quant_type: Type of quantizer to use ('lucid', 'memcodes', or 'fsq')
        num_tokens_to_preserve: Number of tokens to preserve without quantization (e.g., global tokens)
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
        dim: int,
        codebook_size: int = 16384,
        num_codebooks: int = 1,
        quant_type: str = 'lucid',
        num_tokens_to_preserve: int = 0,
        norm_codes: bool = True,
        norm_latents: bool = False,
        sync_codebook: bool = False,
        ema_decay: float = 0.99,
        threshold_ema_dead_code: float = 0.25,
        code_replacement_policy: str = 'batch_random',
        commitment_weight: float = 1.0,
        kmeans_init: bool = False,
        dtype=torch.bfloat16
    ):
        super().__init__()
        
        self.dim = dim
        self.num_tokens_to_preserve = num_tokens_to_preserve
        self.commitment_weight = commitment_weight

        # Initialize the vector quantizer
        if quant_type == 'lucid':
            self.quantizer = VectorQuantizerLucid(
                dim=dim,
                codebook_size=codebook_size,
                codebook_dim=dim,
                heads=num_codebooks,
                use_cosine_sim=norm_codes,
                threshold_ema_dead_code=threshold_ema_dead_code,
                code_replacement_policy=code_replacement_policy,
                sync_codebook=sync_codebook,
                decay=ema_decay,
                commitment_weight=commitment_weight,
                norm_latents=norm_latents,
                kmeans_init=kmeans_init,
            )
        elif quant_type == 'memcodes':
            self.quantizer = Memcodes(
                dim=dim,
                codebook_size=codebook_size,
                heads=num_codebooks,
                temperature=1.,
            )
        elif quant_type == 'fsq':
            self.quantizer = FiniteScalarQuantizer(codebook_size=codebook_size)
        else:
            raise ValueError(f'{quant_type} not a valid quant_type.')

        self.quantizer = self.quantizer.to(dtype)

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the vector quantizer.
        
        Args:
            tokens: Input tokens of shape (B, N, D) where:
                   B is batch size
                   N is number of tokens
                   D is token dimension
        
        Returns:
            Tuple containing:
            - Quantized tokens with same shape as input (B, N, D)
            - Codebook loss
        """
        B, N, D = tokens.shape
        
        # Split tokens into preserved and quantized parts if needed
        if self.num_tokens_to_preserve > 0:
            preserved_tokens = tokens[:, :self.num_tokens_to_preserve]
            tokens_to_quantize = tokens[:, self.num_tokens_to_preserve:]
        else:
            tokens_to_quantize = tokens
            
        # Apply vector quantization
        quantized, vq_loss, _ = self.quantizer(tokens_to_quantize)
        
        # Recombine with preserved tokens if needed
        if self.num_tokens_to_preserve > 0:
            quantized = torch.cat([preserved_tokens, quantized], dim=1)
            
        return quantized, vq_loss

    def get_codebook_entry(self, indices: torch.LongTensor, shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        """Convert indices to their corresponding codebook entries.
        
        Args:
            indices: Tensor of codebook indices
            shape: Optional shape to reshape the output to
        
        Returns:
            Tensor of codebook entries
        """
        return self.quantizer.indices_to_embedding(indices) 
    


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25, 
                 decay: float = 0.99, epsilon: float = 1e-5, temperature: float = 1.0,
                 diversity_reg: float = 0.2, chunk_size: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.temperature = temperature
        self.diversity_reg = diversity_reg
        self.chunk_size = chunk_size
        
        # Initialize embeddings - use smaller range for better stability and convergence
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-0.5 / num_embeddings, 0.5 / num_embeddings)
        
        # Initialize EMA variables
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())
        self.register_buffer('usage_count', torch.zeros(num_embeddings))
        
        # Track training steps for temperature annealing
        self.register_buffer('steps', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def reset_dead_vectors(self, encodings: torch.Tensor):
        """Reset unused vectors to active vectors plus noise"""
        if self.training:
            # Update usage counts with EMA
            usage = torch.sum(encodings, dim=0)
            self.usage_count = self.usage_count * self.decay + (1 - self.decay) * usage
            
            # Find dead vectors (usage below threshold)
            dead_indices = torch.where(self.usage_count < self.epsilon)[0]
            
            if len(dead_indices) > 0:
                # Find most used vectors
                _, most_used = torch.topk(self.usage_count, k=min(len(dead_indices), 10))
                
                # For each dead vector, copy a random most used one and add noise
                for i, dead_idx in enumerate(dead_indices):
                    # Use random selection from top used vectors for more diversity
                    used_idx = most_used[i % len(most_used)]
                    # Use smaller noise to keep closer to working embeddings
                    noise = torch.randn_like(self.embedding.weight[dead_idx]) * 0.05
                    self.embedding.weight.data[dead_idx] = self.embedding.weight[used_idx] + noise
                    self.ema_w[dead_idx] = self.ema_w[used_idx] + noise
                    # Reset the usage count with a small non-zero value to give it a chance
                    self.usage_count[dead_idx] = self.epsilon * 2

    def compute_distances(self, flat_input: torch.Tensor) -> torch.Tensor:
        """Compute distances in chunks to save memory"""
        num_inputs = flat_input.shape[0]
        distances = []
        
        # Get current temperature - anneal from initial value to 0.1 over training
        if self.training:
            self.steps += 1
            # Anneal temperature from initial value to 0.1 over 10000 steps
            current_temp = max(0.1, self.temperature * (1.0 - min(1.0, self.steps.item() / 10000)))
        else:
            current_temp = 0.1  # Use low temperature for inference
            
        for i in range(0, num_inputs, self.chunk_size):
            chunk = flat_input[i:i + self.chunk_size]
            chunk_dist = (
                torch.sum(chunk**2, dim=1, keepdim=True) 
                + torch.sum(self.embedding.weight**2, dim=1)
                - 2 * torch.matmul(chunk, self.embedding.weight.t())
            )
            distances.append(chunk_dist)
            
        return torch.cat(distances, dim=0) / current_temp

    @torch.no_grad()
    def update_embeddings(self, encodings: torch.Tensor, flat_input: torch.Tensor):
        """Update embeddings using EMA in a memory efficient way"""
        # Calculate new cluster size
        cluster_size = encodings.sum(0)
        
        # Apply Exponential Moving Average to cluster_size
        self.ema_cluster_size = self.ema_cluster_size * self.decay + (1 - self.decay) * cluster_size
        
        # Add small value to avoid division by zero
        n = cluster_size.sum()
        
        # Laplace smoothing of the cluster size
        smoothed_cluster_size = ((self.ema_cluster_size + self.epsilon) / 
                               (n + self.num_embeddings * self.epsilon) * n)
        
        # Update embeddings in chunks
        dw = torch.zeros_like(self.ema_w)
        for i in range(0, encodings.shape[0], self.chunk_size):
            chunk_encodings = encodings[i:i + self.chunk_size]
            chunk_inputs = flat_input[i:i + self.chunk_size]
            dw += torch.matmul(chunk_encodings.t(), chunk_inputs)
            
        # Apply EMA to embeddings
        self.ema_w = self.ema_w * self.decay + (1 - self.decay) * dw
        
        # Normalize embeddings 
        # self.embedding.weight.data = self.ema_w / smoothed_cluster_size.unsqueeze(1)
        normalized_embeddings = F.normalize(self.ema_w / smoothed_cluster_size.unsqueeze(1), p=2, dim=-1)
        self.embedding.weight.data = normalized_embeddings

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Save original shape
        input_shape = inputs.shape
        
        # Flatten batch and sequence dimensions: [B, N, D] -> [BxN, D]
        flat_input = inputs.reshape(-1, self.embedding_dim)
        
        # Normalize input
        flat_input = F.normalize(flat_input, p=2, dim=-1)
        
        # Calculate distances
        distances = self.compute_distances(flat_input)
            
        # Soft encoding with Gumbel-Softmax during training
        if self.training:
            encodings = F.gumbel_softmax(-distances, tau=self.temperature, dim=-1, hard=True)
        else:
            # Hard encoding during inference
            encoding_indices = torch.argmin(distances, dim=-1)
            encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight)
        
        # Update embeddings and reset dead vectors
        if self.training:
            self.update_embeddings(encodings, flat_input)
            self.reset_dead_vectors(encodings)
        
        # Commitment loss - use cosine similarity instead of MSE for normalized vectors
        e_latent_loss = 1 - F.cosine_similarity(quantized.detach(), flat_input, dim=-1).mean()
        
        # Efficient diversity regularization
        if self.training and self.diversity_reg > 0:
            # Get counts of used vectors in this batch
            used_counts = encodings.sum(0)
            active_indices = torch.where(used_counts > 0)[0]
            
            if len(active_indices) > 1:
                active_embeddings = self.embedding.weight[active_indices]
                # Calculate cosine similarity between all pairs of active embeddings
                similarity = torch.matmul(active_embeddings, active_embeddings.t())
                # Remove diagonal (self-similarity)
                similarity = similarity - torch.eye(len(active_indices), device=similarity.device)
                # Calculate diversity loss - higher similarity means lower diversity
                diversity_loss = similarity.abs().mean()
            else:
                diversity_loss = torch.tensor(0.0, device=e_latent_loss.device)
        else:
            diversity_loss = torch.tensor(0.0, device=e_latent_loss.device)
        
        # Total loss
        loss = self.commitment_cost * e_latent_loss + self.diversity_reg * diversity_loss
        
        # Straight through estimator
        quantized = flat_input + (quantized - flat_input).detach()
        
        # Reshape back to original shape
        quantized = quantized.view(input_shape)
        
        return quantized, loss


class RevivalVQ(VQ):
    """A variant of VQ that implements an aggressive dead token revival scheme.
    
    This class extends the base VQ class with additional mechanisms to:
    1. Track token usage over time with exponential moving average
    2. Identify dead tokens more aggressively
    3. Revive dead tokens using a combination of:
       - Splitting highly used tokens
       - Interpolating between active tokens
       - Adding controlled noise to existing tokens
    
    Args:
        revival_threshold: Usage threshold below which a token is considered dead (default: 0.1)
        revival_frequency: How often to check and revive tokens (in steps) (default: 100)
        revival_window: Number of steps to average usage over (default: 1000)
        revival_temperature: Temperature for token splitting (default: 0.1)
        **kwargs: Arguments passed to parent VQ class
    """
    def __init__(
        self,
        revival_threshold: float = 0.1,
        revival_frequency: int = 100,
        revival_window: int = 1000,
        revival_temperature: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.revival_threshold = revival_threshold
        self.revival_frequency = revival_frequency
        self.revival_window = revival_window
        self.revival_temperature = revival_temperature
        
        # Initialize tracking buffers
        self.register_buffer('steps', torch.zeros(1, dtype=torch.long))
        self.register_buffer('token_usage', torch.zeros(kwargs.get('codebook_size', 8192)))
        self.register_buffer('token_usage_ema', torch.zeros(kwargs.get('codebook_size', 8192)))
        
    def _update_usage_stats(self, indices: torch.Tensor):
        """Update token usage statistics using EMA."""
        if not self.training:
            return
            
        # Get current batch usage
        unique, counts = torch.unique(indices, return_counts=True)
        current_usage = torch.zeros_like(self.token_usage)
        current_usage[unique] = counts.float()
        
        # Update EMA of usage
        decay = 0.99  # High decay for stable estimates
        self.token_usage = decay * self.token_usage + (1 - decay) * current_usage
        self.token_usage_ema = decay * self.token_usage_ema + (1 - decay) * self.token_usage
        
    def _revive_dead_tokens(self):
        """Revive dead tokens using various strategies."""
        if not self.training:
            return
            
        # Normalize usage to [0,1] range
        usage = self.token_usage_ema / self.token_usage_ema.sum()
        
        # Find dead and active tokens
        dead_tokens = torch.where(usage < self.revival_threshold)[0]
        active_tokens = torch.where(usage >= self.revival_threshold)[0]
        
        if len(dead_tokens) == 0 or len(active_tokens) == 0:
            return
            
        # Get the top-k most used tokens
        k = min(len(dead_tokens), len(active_tokens))
        _, top_k_indices = torch.topk(usage[active_tokens], k=k)
        top_k_tokens = active_tokens[top_k_indices]
        
        # For each dead token, apply one of three revival strategies
        for i, dead_idx in enumerate(dead_tokens):
            strategy = i % 3  # Rotate through strategies
            active_idx = top_k_tokens[i % len(top_k_tokens)]
            
            if strategy == 0:
                # Strategy 1: Split a highly used token with noise
                noise = torch.randn_like(self.quantizer.codebook[active_idx]) * self.revival_temperature
                self.quantizer.codebook[dead_idx] = self.quantizer.codebook[active_idx] + noise
                
            elif strategy == 1:
                # Strategy 2: Interpolate between two active tokens
                other_idx = top_k_tokens[(i + 1) % len(top_k_tokens)]
                alpha = torch.rand(1).item()
                self.quantizer.codebook[dead_idx] = (
                    alpha * self.quantizer.codebook[active_idx] +
                    (1 - alpha) * self.quantizer.codebook[other_idx]
                )
                
            else:
                # Strategy 3: Add larger controlled noise to an active token
                noise = torch.randn_like(self.quantizer.codebook[active_idx]) * self.revival_temperature * 2.0
                self.quantizer.codebook[dead_idx] = self.quantizer.codebook[active_idx] + noise
            
            # Normalize the new embedding
            with torch.no_grad():
                self.quantizer.codebook[dead_idx] = F.normalize(self.quantizer.codebook[dead_idx], dim=-1)
    
    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with dead token revival."""
        # Regular VQ forward pass
        quantized, vq_loss, indices = self.quantizer(tokens)
        
        # Update usage statistics
        self._update_usage_stats(indices)
        
        # Periodically revive dead tokens
        self.steps += 1
        if self.training and (self.steps % self.revival_frequency == 0):
            self._revive_dead_tokens()
            
        return quantized, vq_loss