"""
utils.py
Loss functions for crater segmentation training.
Includes discriminative loss for instance segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
import subprocess


# ============================================================================
# DISCRIMINATIVE LOSS FOR INSTANCE SEGMENTATION
# ============================================================================

class DiscriminativeLoss(nn.Module):
    """
    Discriminative Loss for instance segmentation embeddings.

    Based on: "Semantic Instance Segmentation with a Discriminative Loss Function"
    (De Brabandere et al., 2017)

    Three components:
    1. Variance (Pull) Loss: Pull embeddings within same instance toward mean
    2. Distance (Push) Loss: Push mean embeddings of different instances apart
    3. Regularization Loss: Keep embeddings bounded

    Args:
        delta_v (float): Variance loss margin (pull threshold)
        delta_d (float): Distance loss margin (push threshold)
        alpha (float): Weight for variance loss
        beta (float): Weight for distance loss
        gamma (float): Weight for regularization loss
    """

    def __init__(self, delta_v=0.5, delta_d=1.5, alpha=1.0, beta=1.0, gamma=0.001):
        super().__init__()
        self.delta_v = delta_v  # Pull margin (smaller = tighter clusters)
        self.delta_d = delta_d  # Push margin (larger = more separation)
        self.alpha = alpha      # Variance loss weight
        self.beta = beta        # Distance loss weight
        self.gamma = gamma      # Regularization loss weight

    def forward(self, embeddings, instance_labels):
        """
        Compute discriminative loss for instance embeddings.

        Args:
            embeddings: (B, C, H, W) - Predicted embeddings (L2-normalized)
            instance_labels: (B, H, W) - Instance masks with unique IDs
                             0 = background, 1+ = instance IDs

        Returns:
            loss_dict: Dictionary with total loss and components
        """
        batch_size = embeddings.shape[0]

        total_var_loss = 0.0
        total_dist_loss = 0.0
        total_reg_loss = 0.0

        num_valid_images = 0

        for b in range(batch_size):
            emb = embeddings[b]  # (C, H, W)
            inst = instance_labels[b]  # (H, W)

            # Get unique instance IDs (excluding background=0)
            unique_instances = torch.unique(inst)
            unique_instances = unique_instances[unique_instances != 0]

            num_instances = len(unique_instances)

            # Skip if no instances
            if num_instances == 0:
                continue

            num_valid_images += 1

            # Compute mean embeddings for each instance
            means = []
            for instance_id in unique_instances:
                mask = (inst == instance_id)  # (H, W)
                instance_embeddings = emb[:, mask]  # (C, N_pixels)

                if instance_embeddings.shape[1] == 0:
                    continue

                mean_emb = instance_embeddings.mean(dim=1)  # (C,)
                means.append(mean_emb)

            if len(means) == 0:
                continue

            means = torch.stack(means)  # (N_instances, C)

            # ================================================================
            # 1. VARIANCE (PULL) LOSS
            # Pull embeddings toward their instance mean
            # ================================================================
            var_loss = 0.0
            for idx, instance_id in enumerate(unique_instances):
                mask = (inst == instance_id)
                instance_embeddings = emb[:, mask]  # (C, N_pixels)

                if instance_embeddings.shape[1] == 0:
                    continue

                mean_emb = means[idx].unsqueeze(1)  # (C, 1)

                # Distance from mean
                distances = torch.norm(instance_embeddings - mean_emb, dim=0)  # (N_pixels,)

                # Hinge loss: max(0, distance - delta_v)^2
                hinge = torch.clamp(distances - self.delta_v, min=0.0)
                var_loss += torch.mean(hinge ** 2)

            var_loss /= num_instances

            # ================================================================
            # 2. DISTANCE (PUSH) LOSS
            # Push different instance means apart
            # ================================================================
            dist_loss = 0.0
            if num_instances > 1:
                # Compute pairwise distances between instance means
                for i in range(num_instances):
                    for j in range(i + 1, num_instances):
                        mean_i = means[i]  # (C,)
                        mean_j = means[j]  # (C,)

                        distance = torch.norm(mean_i - mean_j)

                        # Hinge loss: max(0, 2*delta_d - distance)^2
                        hinge = torch.clamp(2 * self.delta_d - distance, min=0.0)
                        dist_loss += hinge ** 2

                # Normalize by number of pairs
                num_pairs = (num_instances * (num_instances - 1)) / 2
                dist_loss /= num_pairs

            # ================================================================
            # 3. REGULARIZATION LOSS
            # Keep embeddings bounded (prevent unbounded growth)
            # ================================================================
            reg_loss = torch.mean(torch.norm(means, dim=1))

            # Accumulate losses
            total_var_loss += var_loss
            total_dist_loss += dist_loss
            total_reg_loss += reg_loss

        # Average over valid images in batch
        if num_valid_images > 0:
            total_var_loss /= num_valid_images
            total_dist_loss /= num_valid_images
            total_reg_loss /= num_valid_images

        # Weighted combination
        total_loss = (
            self.alpha * total_var_loss +
            self.beta * total_dist_loss +
            self.gamma * total_reg_loss
        )

        return {
            'total': total_loss,
            'variance': total_var_loss,
            'distance': total_dist_loss,
            'regularization': total_reg_loss,
        }


# ============================================================================
# COMBINED INSTANCE SEGMENTATION LOSS
# ============================================================================

class InstanceSegmentationLoss(nn.Module):
    """
    Combined loss for instance segmentation.

    Combines:
    1. Semantic segmentation loss (CrossEntropy)
    2. Discriminative loss for instance embeddings

    Args:
        semantic_weight (float): Weight for semantic loss
        variance_weight (float): Weight for variance (pull) loss
        distance_weight (float): Weight for distance (push) loss
        regularization_weight (float): Weight for regularization loss
        delta_v (float): Variance loss margin
        delta_d (float): Distance loss margin
    """

    def __init__(
        self,
        semantic_weight=1.0,
        variance_weight=1.0,
        distance_weight=1.0,
        regularization_weight=0.001,
        delta_v=0.5,
        delta_d=1.5,
    ):
        super().__init__()

        self.semantic_weight = semantic_weight

        # Semantic segmentation loss
        self.semantic_loss = nn.CrossEntropyLoss()

        # Discriminative loss for embeddings
        self.discriminative_loss = DiscriminativeLoss(
            delta_v=delta_v,
            delta_d=delta_d,
            alpha=variance_weight,
            beta=distance_weight,
            gamma=regularization_weight,
        )

    def forward(self, outputs, labels):
        """
        Compute combined loss for instance segmentation.

        Args:
            outputs: dict with keys:
                - 'semantic': (B, num_classes, H, W) - Semantic logits
                - 'embeddings': (B, embedding_dim, H, W) - Instance embeddings
            labels: (B, H, W) - Instance masks with unique IDs
                    0 = background, 1+ = instance IDs

        Returns:
            total_loss: Combined weighted loss (scalar)
        """
        semantic_logits = outputs['semantic']
        embeddings = outputs['embeddings']

        # ================================================================
        # 1. SEMANTIC LOSS (Binary: crater vs background)
        # ================================================================
        # Convert instance labels to binary semantic labels
        semantic_labels = (labels > 0).long()  # 0=background, 1=any crater
        semantic_loss = self.semantic_loss(semantic_logits, semantic_labels)

        # ================================================================
        # 2. DISCRIMINATIVE LOSS (Separate instances)
        # ================================================================
        disc_loss_dict = self.discriminative_loss(embeddings, labels)

        # ================================================================
        # COMBINE LOSSES
        # ================================================================
        total_loss = (
            self.semantic_weight * semantic_loss +
            disc_loss_dict['total']
        )

        # Store individual components for logging
        self.last_losses = {
            'total': total_loss.item(),
            'semantic': semantic_loss.item(),
            'variance': disc_loss_dict['variance'].item(),
            'distance': disc_loss_dict['distance'].item(),
            'regularization': disc_loss_dict['regularization'].item(),
        }

        return total_loss

    def get_last_losses(self):
        """Get detailed loss components from last forward pass."""
        return getattr(self, 'last_losses', {})


# ============================================================================
# SEMANTIC SEGMENTATION LOSSES (Keep existing ones)
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""

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
    """Dice Loss for segmentation."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)[:, 1]
        targets_one_hot = (targets == 1).float()

        intersection = (probs * targets_one_hot).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + targets_one_hot.sum(dim=(1, 2))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BoundaryLoss(nn.Module):
    """Boundary Loss to penalize blob predictions."""

    def __init__(self, weight=0.2):
        super().__init__()
        self.weight = weight

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
        probs = torch.softmax(logits, dim=1)[:, 1:2]
        targets_mask = (targets == 1).unsqueeze(1).float()

        pred_edges_x = F.conv2d(probs, self.sobel_x, padding=1)
        pred_edges_y = F.conv2d(probs, self.sobel_y, padding=1)
        pred_edges = torch.sqrt(pred_edges_x**2 + pred_edges_y**2)

        gt_edges_x = F.conv2d(targets_mask, self.sobel_x, padding=1)
        gt_edges_y = F.conv2d(targets_mask, self.sobel_y, padding=1)
        gt_edges = torch.sqrt(gt_edges_x**2 + gt_edges_y**2)

        boundary_loss = F.mse_loss(pred_edges, gt_edges)
        return self.weight * boundary_loss


class CombinedLoss(nn.Module):
    """Combined Loss: CrossEntropy + Dice Loss."""

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
    """Full Loss: CrossEntropy + Dice + Boundary Loss."""

    def __init__(self, ce_weight=0.4, dice_weight=0.4, boundary_weight=0.2):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss(weight=1.0)

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


# ============================================================================
# LOSS FACTORY FUNCTION
# ============================================================================

def get_loss_function(loss_type="instance_combined"):
    """
    Factory function to get loss function by name.

    Args:
        loss_type (str): Type of loss function. Options:

            INSTANCE SEGMENTATION:
            - 'instance_combined': Semantic + Discriminative (RECOMMENDED)
            - 'instance_strong_push': More separation between instances
            - 'instance_tight_clusters': Tighter instance clusters

            SEMANTIC SEGMENTATION (backward compatibility):
            - 'cross_entropy': Standard CrossEntropyLoss
            - 'focal': Focal Loss for class imbalance
            - 'dice': Dice Loss for segmentation
            - 'combined': CrossEntropy + Dice
            - 'full': CrossEntropy + Dice + Boundary

    Returns:
        nn.Module: Loss function module
    """
    loss_functions = {
        # Instance segmentation losses
        'instance_combined': InstanceSegmentationLoss(
            semantic_weight=1.0,
            variance_weight=1.0,
            distance_weight=1.0,
            regularization_weight=0.001,
            delta_v=0.5,   # Pull margin
            delta_d=1.5,   # Push margin
        ),
        'instance_strong_push': InstanceSegmentationLoss(
            semantic_weight=1.0,
            variance_weight=1.0,
            distance_weight=2.0,  # Stronger push
            regularization_weight=0.001,
            delta_v=0.5,
            delta_d=2.0,  # Larger margin
        ),
        'instance_tight_clusters': InstanceSegmentationLoss(
            semantic_weight=1.0,
            variance_weight=2.0,  # Stronger pull
            distance_weight=1.0,
            regularization_weight=0.001,
            delta_v=0.3,  # Smaller pull margin
            delta_d=1.5,
        ),

        # Semantic segmentation losses (backward compatibility)
        'cross_entropy': nn.CrossEntropyLoss(),
        'focal': FocalLoss(alpha=0.25, gamma=2.0),
        'dice': DiceLoss(smooth=1.0),
        'combined': CombinedLoss(ce_weight=0.5, dice_weight=0.5),
        'full': FullLoss(ce_weight=0.4, dice_weight=0.4, boundary_weight=0.2),
    }

    if loss_type not in loss_functions:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Available options: {list(loss_functions.keys())}"
        )

    return loss_functions[loss_type]