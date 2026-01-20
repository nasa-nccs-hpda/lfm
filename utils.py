"""
utils.py
Loss functions for crater segmentation training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focuses on hard-to-classify examples (small craters).

    Args:
        alpha (float): Weighting factor for class balance
        gamma (float): Focusing parameter for modulating loss
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    Better for handling imbalanced classes and boundary detection.

    Args:
        smooth (float): Smoothing factor to avoid division by zero
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Get probabilities for crater class (class 1)
        probs = torch.softmax(logits, dim=1)[:, 1]  # (B, H, W)
        targets_one_hot = (targets == 1).float()  # (B, H, W)

        # Compute dice coefficient
        intersection = (probs * targets_one_hot).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + targets_one_hot.sum(dim=(1, 2))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BoundaryLoss(nn.Module):
    """
    Boundary Loss to penalize blob predictions.
    Encourages discrete crater boundaries by comparing edge gradients.

    Args:
        weight (float): Weight for boundary loss component
    """

    def __init__(self, weight=0.2):
        super().__init__()
        self.weight = weight

        # Sobel filters for edge detection
        # Register_buffer auto-moves to device
        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
            ).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
            ).view(1, 1, 3, 3),
        )

    def forward(self, logits, targets):
        # Get predicted crater mask
        probs = torch.softmax(logits, dim=1)[:, 1:2]  # (B, 1, H, W)
        targets_mask = (targets == 1).unsqueeze(1).float()  # (B, 1, H, W)

        # Compute edges for predictions
        pred_edges_x = F.conv2d(probs, self.sobel_x, padding=1)
        pred_edges_y = F.conv2d(probs, self.sobel_y, padding=1)
        pred_edges = torch.sqrt(pred_edges_x**2 + pred_edges_y**2)

        # Compute edges for ground truth
        gt_edges_x = F.conv2d(targets_mask, self.sobel_x, padding=1)
        gt_edges_y = F.conv2d(targets_mask, self.sobel_y, padding=1)
        gt_edges = torch.sqrt(gt_edges_x**2 + gt_edges_y**2)

        # Penalize difference in boundary strength
        boundary_loss = F.mse_loss(pred_edges, gt_edges)
        return self.weight * boundary_loss


class CombinedLoss(nn.Module):
    """
    Combined Loss: CrossEntropy + Dice Loss.
    Balances pixel-wise classification with region overlap.

    Args:
        ce_weight (float): Weight for cross-entropy component
        dice_weight (float): Weight for dice loss component
    """

    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


class FullLoss(nn.Module):
    """
    Full Loss: CrossEntropy + Dice + Boundary Loss.
    Most comprehensive loss for crater detection with discrete boundaries.

    Args:
        ce_weight (float): Weight for cross-entropy
        dice_weight (float): Weight for dice loss
        boundary_weight (float): Weight for boundary loss
    """

    def __init__(self, ce_weight=0.4, dice_weight=0.4, boundary_weight=0.2):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss(weight=1.0)  # Weight handled externally

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        boundary_loss = self.boundary(logits, targets)

        total_loss = (
            self.ce_weight * ce_loss
            + self.dice_weight * dice_loss
            + self.boundary_weight * boundary_loss
        )
        return total_loss


def get_loss_function(loss_type="cross_entropy"):
    """
    Factory function to get loss function by name.

    Args:
        loss_type (str): Type of loss function. Options:
            - 'cross_entropy': Standard CrossEntropyLoss
            - 'focal': Focal Loss for class imbalance
            - 'dice': Dice Loss for segmentation
            - 'combined': CrossEntropy + Dice
            - 'full': CrossEntropy + Dice + Boundary (recommended for craters)

    Returns:
        nn.Module: Loss function module
    """
    loss_functions = {
        "cross_entropy": nn.CrossEntropyLoss(),
        "focal": FocalLoss(alpha=0.25, gamma=2.0),
        "dice": DiceLoss(smooth=1.0),
        "combined": CombinedLoss(ce_weight=0.5, dice_weight=0.5),
        "full": FullLoss(ce_weight=0.4, dice_weight=0.4, boundary_weight=0.2),
    }

    if loss_type not in loss_functions:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Available options: {list(loss_functions.keys())}"
        )

    return loss_functions[loss_type]
