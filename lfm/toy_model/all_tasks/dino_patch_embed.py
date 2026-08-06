"""DINO patch-embedding helpers shared by toy model tasks."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def flexible_dino_patch_weights(
    original_weights: torch.Tensor,
    weight_assignments: Sequence[str],
) -> torch.Tensor:
    """Map DINO RGB patch weights to an arbitrary input-channel layout.

    For a single grayscale/NAC channel, use the sum of RGB kernels. This is
    exactly equivalent to feeding the same one-channel image into all three
    original RGB channels:

    ``W_R*x + W_G*x + W_B*x = (W_R + W_G + W_B)*x``.
    """
    if original_weights.ndim != 4 or original_weights.shape[1] != 3:
        raise ValueError(
            "Expected original DINO patch weights with shape (out, 3, kH, kW), "
            f"got {tuple(original_weights.shape)}"
        )
    if not weight_assignments:
        raise ValueError("weight_assignments must contain at least one channel.")

    red_weights = original_weights[:, 0, :, :]
    green_weights = original_weights[:, 1, :, :]
    blue_weights = original_weights[:, 2, :, :]

    new_weights = torch.zeros(
        original_weights.shape[0],
        len(weight_assignments),
        original_weights.shape[2],
        original_weights.shape[3],
        device=original_weights.device,
        dtype=original_weights.dtype,
    )

    if len(weight_assignments) == 1:
        new_weights[:, 0, :, :] = red_weights + green_weights + blue_weights
        return new_weights

    for index, assignment in enumerate(weight_assignments):
        if assignment == "blue":
            new_weights[:, index, :, :] = blue_weights
        elif assignment == "green":
            new_weights[:, index, :, :] = green_weights
        elif assignment == "red":
            new_weights[:, index, :, :] = red_weights
        elif assignment == "0.95*red":
            new_weights[:, index, :, :] = red_weights
        elif assignment == "0.7*red+0.3*green":
            new_weights[:, index, :, :] = 0.7 * red_weights + 0.3 * green_weights
        elif assignment == "rgb_sum":
            new_weights[:, index, :, :] = (
                red_weights + green_weights + blue_weights
            )
        elif assignment == "rgb_mean":
            new_weights[:, index, :, :] = (
                red_weights + green_weights + blue_weights
            ) / 3.0
        else:
            print(
                f"Warning: Unknown weight assignment '{assignment}' for band "
                f"{index}; using red weights.",
                flush=True,
            )
            new_weights[:, index, :, :] = red_weights

    return new_weights
