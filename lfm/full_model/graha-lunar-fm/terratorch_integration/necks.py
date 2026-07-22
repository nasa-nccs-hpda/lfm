"""Custom neck module for learned token projection."""

import torch
from torch import nn

from terratorch.models.necks import Neck
from terratorch.registry import TERRATORCH_NECK_REGISTRY


@TERRATORCH_NECK_REGISTRY.register
class LearnedTokenProjection(Neck):
    """Apply learned linear projection to transform the last dimension of features from C to D.

    Dynamically creates projection layers on first forward pass based on actual input dimensions.
    Each feature layer gets its own learned projection from its input channel dimension C_i to output dimension D.
    Works on any tensor shape by projecting only the last dimension.
    Useful for standardizing feature dimensions across layers or adapting to decoder requirements.

    Example usage in config:
        neck:
          - name: LearnedTokenProjection
            output_dim: 768
    """

    def __init__(self, channel_list: list[int], output_dim: int):
        """Initialize the neck. Projection layers are created on first forward pass.

        Args:
            channel_list (list[int]): List of input channel dimensions (may be incorrect if previous neck doesn't update properly)
            output_dim (int): Target output dimension D for all layers
        """
        super().__init__(channel_list)
        self.output_dim = output_dim
        self.num_features = len(channel_list)
        # Initialize with empty ModuleList for checkpoint compatibility
        self.projections = nn.ModuleList()
        self._initialized = False

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Custom state dict loading to handle dynamically created projections."""
        # Check if projections exist in state_dict
        projection_keys = [
            k for k in state_dict.keys() if k.startswith(f"{prefix}projections.")
        ]

        if projection_keys and len(self.projections) == 0:
            # Infer number of projections and their dimensions from state_dict
            num_projections = len([k for k in projection_keys if k.endswith(".weight")])

            # Create placeholder projections with correct dimensions
            for i in range(num_projections):
                weight_key = f"{prefix}projections.{i}.weight"
                if weight_key in state_dict:
                    in_features = state_dict[weight_key].shape[
                        1
                    ]  # weight shape is (out_features, in_features)
                    self.projections.append(nn.Linear(in_features, self.output_dim))

            self._initialized = True
            print(
                f"LearnedTokenProjection: Created {num_projections} projections from checkpoint"
            )

        # Call parent's load method
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, channel_list: list[torch.Tensor], **kwargs) -> list[torch.Tensor]:
        """Apply learned projection to the last dimension of each feature.

        Args:
            channel_list (list[torch.Tensor]): List of feature tensors with shape (..., C_i)

        Returns:
            list[torch.Tensor]: List of projected features with shape (..., D)
        """
        # Initialize projections on first forward pass with actual input dimensions
        # Skip if already loaded from checkpoint (projections not empty)
        if not self._initialized and len(self.projections) == 0:
            actual_dims = [feat.shape[-1] for feat in channel_list]
            self.projections = nn.ModuleList(
                [
                    nn.Linear(in_dim, self.output_dim).to(channel_list[0].device)
                    for in_dim in actual_dims
                ]
            )
            self._initialized = True
            print(
                f"LearnedTokenProjection initialized with input dims: {actual_dims} -> output dim: {self.output_dim}"
            )
        elif len(self.projections) > 0 and not self._initialized:
            # Loaded from checkpoint
            self._initialized = True
            print(
                f"LearnedTokenProjection loaded from checkpoint with {len(self.projections)} projections"
            )

        projected_features = []
        for feat, projection in zip(channel_list, self.projections, strict=True):
            # Apply learned linear projection to the last dimension
            feat_projected = projection(feat)
            projected_features.append(feat_projected)

        return projected_features

    def process_channel_list(self, channel_list: list[int]) -> list[int]:
        """Return the new channel list after projection.

        Args:
            channel_list (list[int]): Input channel dimensions

        Returns:
            list[int]: Output channel dimensions (all equal to output_dim)
        """
        return [self.output_dim] * len(channel_list)
