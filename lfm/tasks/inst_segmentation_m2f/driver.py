"""
driver.py
Training driver for DINO instance segmentation.
Supports training, evaluation, and checkpoint resumption.
Handles images with any number of channels and instance-level predictions.
"""

import os
import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from tqdm import tqdm

from .utils import get_loss_function


# ============================================================================
# METRICS
# ============================================================================


def calculate_instance_metrics(pred_mask, gt_mask, iou_threshold=0.5):
    """
    Calculate instance-level metrics using IoU matching.

    Args:
        pred_mask: Predicted instance mask (H, W) with unique IDs
        gt_mask: Ground truth instance mask (H, W) with unique IDs
        iou_threshold: IoU threshold for considering a match

    Returns:
        dict with precision, recall, F1, and per-instance IoUs
    """
    # Get unique instance IDs (excluding background=0)
    pred_ids = np.unique(pred_mask)
    pred_ids = pred_ids[pred_ids != 0]

    gt_ids = np.unique(gt_mask)
    gt_ids = gt_ids[gt_ids != 0]

    num_pred = len(pred_ids)
    num_gt = len(gt_ids)

    if num_gt == 0 and num_pred == 0:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "num_pred": 0,
            "num_gt": 0,
            "num_matched": 0,
            "mean_iou": 0.0,
        }

    if num_gt == 0:
        return {
            "precision": 0.0,
            "recall": 1.0,
            "f1": 0.0,
            "num_pred": num_pred,
            "num_gt": 0,
            "num_matched": 0,
            "mean_iou": 0.0,
        }

    if num_pred == 0:
        return {
            "precision": 1.0,
            "recall": 0.0,
            "f1": 0.0,
            "num_pred": 0,
            "num_gt": num_gt,
            "num_matched": 0,
            "mean_iou": 0.0,
        }

    # Compute IoU matrix
    iou_matrix = np.zeros((num_pred, num_gt))

    for i, pred_id in enumerate(pred_ids):
        pred_pixels = pred_mask == pred_id

        for j, gt_id in enumerate(gt_ids):
            gt_pixels = gt_mask == gt_id

            # Compute IoU
            intersection = np.logical_and(pred_pixels, gt_pixels).sum()
            union = np.logical_or(pred_pixels, gt_pixels).sum()

            if union > 0:
                iou_matrix[i, j] = intersection / union

    # Match predictions to ground truth (greedy matching by highest IoU)
    matched_pred = set()
    matched_gt = set()
    ious = []

    # Sort all IoUs in descending order
    iou_pairs = []
    for i in range(num_pred):
        for j in range(num_gt):
            if iou_matrix[i, j] >= iou_threshold:
                iou_pairs.append((iou_matrix[i, j], i, j))

    iou_pairs.sort(reverse=True)

    # Greedy matching
    for iou, i, j in iou_pairs:
        if i not in matched_pred and j not in matched_gt:
            matched_pred.add(i)
            matched_gt.add(j)
            ious.append(iou)

    num_matched = len(matched_pred)

    # Calculate metrics
    precision = num_matched / num_pred if num_pred > 0 else 0.0
    recall = num_matched / num_gt if num_gt > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    mean_iou = np.mean(ious) if len(ious) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_pred": num_pred,
        "num_gt": num_gt,
        "num_matched": num_matched,
        "mean_iou": mean_iou,
    }


def calculate_semantic_f1(pred_mask, gt_mask):
    """
    Calculate binary F1 score for semantic segmentation.
    Treats all instances as single class.

    Args:
        pred_mask: Predicted mask (any non-zero = positive)
        gt_mask: Ground truth mask (any non-zero = positive)

    Returns:
        f1: F1 score
    """
    pred_binary = (pred_mask > 0).astype(int).flatten()
    gt_binary = (gt_mask > 0).astype(int).flatten()

    tp = np.sum((pred_binary == 1) & (gt_binary == 1))
    fp = np.sum((pred_binary == 1) & (gt_binary == 0))
    fn = np.sum((pred_binary == 0) & (gt_binary == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return f1


# ============================================================================
# VISUALIZATION
# ============================================================================


def prepare_image_for_display(img):
    """
    Prepare multi-channel image for matplotlib display.

    Args:
        img: Image array with shape (H, W, C) where C can be any number

    Returns:
        img_vis: Image ready for display (H, W) or (H, W, 3)
        display_note: String describing how the image was prepared
    """
    num_channels = img.shape[2]

    # Denormalize image for visualization
    img_normalized = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_normalized = np.clip(img_normalized, 0, 1)

    if num_channels == 1:
        img_vis = img_normalized[:, :, 0]
        display_note = "Grayscale"
    elif num_channels == 3:
        img_vis = img_normalized
        display_note = "RGB"
    else:
        img_vis = img_normalized[:, :, :3]
        display_note = f"{num_channels}ch (showing first 3)"

    return img_vis, display_note


def create_instance_overlay(img_vis, instance_mask, alpha=0.5, colormap="hsv"):
    """
    Create overlay of instance mask on image with unique colors per instance.

    Args:
        img_vis: Base image (H, W) for grayscale or (H, W, 3) for color
        instance_mask: Instance mask (H, W) with unique IDs per instance
        alpha: Transparency for overlay
        colormap: Matplotlib colormap for instance colors

    Returns:
        blended: Blended image with colored instance overlays
    """
    # Ensure img_vis is 3-channel
    if img_vis.ndim == 2:
        img_vis_rgb = np.stack([img_vis] * 3, axis=2)
    else:
        img_vis_rgb = img_vis

    # Create colored instance mask
    unique_instances = np.unique(instance_mask)
    unique_instances = unique_instances[
        unique_instances != 0
    ]  # Exclude background

    if len(unique_instances) == 0:
        return img_vis_rgb

    # Get colormap
    cmap = plt.get_cmap(colormap)

    # Create overlay
    overlay = img_vis_rgb.copy()

    for idx, instance_id in enumerate(unique_instances):
        mask = instance_mask == instance_id
        color = np.array(
            cmap(idx / max(len(unique_instances), 1))[:3]
        )  # RGB only

        # Blend color into overlay where mask is True
        overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha

    return np.clip(overlay, 0, 1)


def visualize_predictions(
    model,
    dataloader,
    image_processor,
    device,
    output_dir,
    epoch,
    n_samples=5,
    threshold=0.5,
):
    """
    Visualize Mask2Former instance segmentation predictions.

    Args:
        model: Trained Mask2Former model
        dataloader: DataLoader to sample from
        image_processor: Image processor for post-processing
        device: torch device
        output_dir: Directory to save plots
        epoch: Current epoch number or label (e.g., "EVAL")
        n_samples: Number of samples to visualize
        threshold: Confidence threshold for predictions
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    # Collect predictions
    images_list = []
    labels_list = []
    preds_list = []

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            mask_labels = batch["mask_labels"]  # Keep on CPU for now

            # Get target sizes - use the actual resized size that model sees
            # The model processes at (304, 304) size
            batch_size = pixel_values.shape[0]
            target_sizes = [
                (pixel_values.shape[2], pixel_values.shape[3])
            ] * batch_size

            # Forward pass (inference only - no labels)
            outputs = model(pixel_values=pixel_values)

            # Post-process to get instance masks
            try:
                post_processed = (
                    image_processor.post_process_instance_segmentation(
                        outputs,
                        threshold=threshold,
                        target_sizes=target_sizes,
                        return_binary_maps=True,
                    )
                )
            except Exception as e:
                print(f"Warning: Post-processing failed: {e}")
                print("Attempting alternative post-processing...")
                # Fallback: try without return_binary_maps
                post_processed = (
                    image_processor.post_process_instance_segmentation(
                        outputs,
                        threshold=threshold,
                        target_sizes=target_sizes,
                    )
                )

            # Convert to instance masks (numpy arrays)
            for idx, result in enumerate(post_processed):
                # Check what keys are available
                if "segmentation" in result:
                    # Standard output format
                    instance_mask = result["segmentation"]
                    if isinstance(instance_mask, torch.Tensor):
                        instance_mask = instance_mask.cpu().numpy()
                    preds_list.append(instance_mask)
                else:
                    # Alternative format - create mask from segments_info
                    print(
                        f"Warning: 'segmentation' not in result. Keys: {result.keys()}"
                    )

                    # Create instance mask from segments_info
                    height, width = target_sizes[idx]
                    instance_mask = np.zeros((height, width), dtype=np.int32)

                    if "segments_info" in result:
                        for segment_idx, segment in enumerate(
                            result["segments_info"]
                        ):
                            # This is a fallback - might not work perfectly
                            print(f"  Segment {segment_idx}: {segment}")

                    preds_list.append(instance_mask)

            # Store images and labels
            images_list.append(pixel_values.cpu())

            # Convert mask_labels from (num_instances, H, W) to (H, W) with instance IDs
            for label_tensor in mask_labels:
                label_numpy = label_tensor.cpu().numpy()

                if label_numpy.ndim == 3:
                    # Convert from (num_instances, H, W) to (H, W)
                    h, w = label_numpy.shape[1], label_numpy.shape[2]
                    label_2d = np.zeros((h, w), dtype=np.int32)
                    for inst_id in range(label_numpy.shape[0]):
                        label_2d[label_numpy[inst_id] > 0] = inst_id + 1
                    labels_list.append(label_2d)
                else:
                    # Already in correct format
                    labels_list.append(label_numpy)

            if len(preds_list) >= n_samples:
                break

    # Concatenate and limit to n_samples
    all_images = torch.cat(images_list, dim=0)[:n_samples]
    all_labels = labels_list[:n_samples]
    all_preds = preds_list[:n_samples]

    num_channels = all_images.shape[1]
    batch_size = min(n_samples, len(all_preds))

    # Verify shapes before visualization
    print(f"\nDebug - Shapes for visualization:")
    for i in range(min(3, batch_size)):
        print(f"  Sample {i}:")
        print(f"    Image: {all_images[i].shape}")
        print(f"    GT Label: {all_labels[i].shape}")
        print(f"    Prediction: {all_preds[i].shape}")

    # Create 5-row visualization
    fig, axes = plt.subplots(5, batch_size, figsize=(4 * batch_size, 20))

    if batch_size == 1:
        axes = axes.reshape(-1, 1)

    # Calculate metrics
    instance_metrics = []
    semantic_f1s = []

    for i in tqdm(range(batch_size), desc="Plotting predictions"):
        img = all_images[i].numpy().transpose(1, 2, 0)  # (H, W, C)
        gt_mask = all_labels[i]  # Should be (H, W)
        pred_mask = all_preds[i]  # Should be (H, W)

        # Verify shapes
        if gt_mask.ndim != 2 or pred_mask.ndim != 2:
            print(f"Warning: Incorrect mask dimensions at sample {i}")
            print(
                f"  GT shape: {gt_mask.shape}, Pred shape: {pred_mask.shape}"
            )
            continue

        # Resize masks to match if needed
        if gt_mask.shape != pred_mask.shape:
            from skimage.transform import resize

            print(f"Warning: Shape mismatch at sample {i}. Resizing...")
            print(f"  GT: {gt_mask.shape}, Pred: {pred_mask.shape}")
            # Resize pred to match gt
            pred_mask = resize(
                pred_mask,
                gt_mask.shape,
                order=0,  # Nearest neighbor
                preserve_range=True,
                anti_aliasing=False,
            ).astype(np.int32)

        # Calculate metrics
        inst_metrics = calculate_instance_metrics(pred_mask, gt_mask)
        sem_f1 = calculate_semantic_f1(pred_mask, gt_mask)

        instance_metrics.append(inst_metrics)
        semantic_f1s.append(sem_f1)

        # Prepare image for display
        img_vis, display_note = prepare_image_for_display(img)
        cmap_image = "gray" if img_vis.ndim == 2 else None

        # Row 0: Original image with metrics
        axes[0, i].imshow(img_vis, cmap=cmap_image)
        axes[0, i].set_title(
            f"Image {i}\n"
            f"Sem F1: {sem_f1:.3f}\n"
            f"Inst F1: {inst_metrics['f1']:.3f}\n"
            f"Pred: {inst_metrics['num_pred']} | GT: {inst_metrics['num_gt']}",
            fontsize=12,
            fontweight="bold",
        )
        axes[0, i].axis("off")

        # Row 1: Predicted instances (colored by ID)
        num_pred = len(np.unique(pred_mask)) - 1  # Exclude background
        pred_colored = create_instance_overlay(
            np.ones_like(img_vis) * 0.2,  # Dark background
            pred_mask,
            alpha=1.0,
        )
        axes[1, i].imshow(pred_colored)
        axes[1, i].set_title(
            f"Predicted ({num_pred} instances)\n"
            f"Precision: {inst_metrics['precision']:.3f}",
            fontsize=11,
        )
        axes[1, i].axis("off")

        # Row 2: Prediction overlay on image
        pred_overlay = create_instance_overlay(img_vis, pred_mask, alpha=0.5)
        axes[2, i].imshow(pred_overlay)
        axes[2, i].set_title(
            f"Prediction Overlay\n"
            f"Mean IoU: {inst_metrics['mean_iou']:.3f}",
            fontsize=11,
        )
        axes[2, i].axis("off")

        # Row 3: Ground truth instances (colored by ID)
        num_gt = len(np.unique(gt_mask)) - 1  # Exclude background
        gt_colored = create_instance_overlay(
            np.ones_like(img_vis) * 0.2, gt_mask, alpha=1.0
        )
        axes[3, i].imshow(gt_colored)
        axes[3, i].set_title(
            f"Ground Truth ({num_gt} instances)\n"
            f"Recall: {inst_metrics['recall']:.3f}",
            fontsize=11,
        )
        axes[3, i].axis("off")

        # Row 4: GT overlay on image
        gt_overlay = create_instance_overlay(img_vis, gt_mask, alpha=0.5)
        axes[4, i].imshow(gt_overlay)
        axes[4, i].set_title(
            f"Ground Truth Overlay\n"
            f"Matched: {inst_metrics['num_matched']}",
            fontsize=11,
        )
        axes[4, i].axis("off")

    # Calculate average metrics
    if len(instance_metrics) > 0:
        avg_inst_f1 = np.mean([m["f1"] for m in instance_metrics])
        avg_sem_f1 = np.mean(semantic_f1s)
        avg_precision = np.mean([m["precision"] for m in instance_metrics])
        avg_recall = np.mean([m["recall"] for m in instance_metrics])
        avg_iou = np.mean([m["mean_iou"] for m in instance_metrics])
    else:
        avg_inst_f1 = avg_sem_f1 = avg_precision = avg_recall = avg_iou = 0.0

    # Add overall metrics as figure title
    fig.suptitle(
        f"Epoch {epoch} - Instance Segmentation Results\n"
        f"Semantic F1: {avg_sem_f1:.3f} | Instance F1: {avg_inst_f1:.3f} | "
        f"Precision: {avg_precision:.3f} | Recall: {avg_recall:.3f} | "
        f"Mean IoU: {avg_iou:.3f}\n"
        f"Input: {num_channels}ch",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()

    # Save figure
    if isinstance(epoch, int):
        epoch_str = f"{epoch:03d}"
    else:
        epoch_str = str(epoch)

    save_path = os.path.join(output_dir, f"predictions_epoch_{epoch_str}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization to {save_path}")
    if len(instance_metrics) > 0:
        print(f"Average Semantic F1: {avg_sem_f1:.3f}")
        print(f"Average Instance F1: {avg_inst_f1:.3f}")
        print(f"Average Precision: {avg_precision:.3f}")
        print(f"Average Recall: {avg_recall:.3f}")
        print(f"Average IoU: {avg_iou:.3f}")

    model.train()


# ============================================================================
# TRAINING
# ============================================================================


def train_epoch(model, dataloader, optimizer, device, desc="Training"):
    """
    Train for one epoch using Mask2Former (no criterion needed).

    Args:
        model: Mask2Former model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        device: torch device
        desc: Description for progress bar

    Returns:
        avg_loss: Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    progress_bar = tqdm(dataloader, desc=desc)

    for batch in progress_bar:
        # Move batch to device
        pixel_values = batch["pixel_values"].to(device)
        mask_labels = [labels.to(device) for labels in batch["mask_labels"]]
        class_labels = [labels.to(device) for labels in batch["class_labels"]]

        # Forward pass - Mask2Former computes loss internally
        optimizer.zero_grad()
        outputs = model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )

        # Loss is part of outputs
        loss = outputs.loss

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        n_batches += 1

        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / n_batches
    return avg_loss


def validate_epoch(model, dataloader, device, desc="Validation"):
    """
    Validate for one epoch using Mask2Former (no criterion needed).

    Args:
        model: Mask2Former model to validate
        dataloader: Validation dataloader
        device: torch device
        desc: Description for progress bar

    Returns:
        avg_loss: Average validation loss
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    progress_bar = tqdm(dataloader, desc=desc)

    with torch.no_grad():
        for batch in progress_bar:
            # Move batch to device
            pixel_values = batch["pixel_values"].to(device)
            mask_labels = [
                labels.to(device) for labels in batch["mask_labels"]
            ]
            class_labels = [
                labels.to(device) for labels in batch["class_labels"]
            ]

            # Forward pass
            outputs = model(
                pixel_values=pixel_values,
                mask_labels=mask_labels,
                class_labels=class_labels,
            )

            loss = outputs.loss

            # Track metrics
            total_loss += loss.item()
            n_batches += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / n_batches
    return avg_loss


# ============================================================================
# UTILITIES
# ============================================================================


def print_model_summary(model):
    """
    Print model parameter summary.

    Note: May not work correctly with Mask2Former's structure.
    Use try/except when calling this function.
    """
    try:
        # Try to access encoder/decoder (might not exist in Mask2Former)
        if hasattr(model, "encoder") and hasattr(model, "decoder"):
            encoder_trainable = sum(
                p.numel()
                for p in model.encoder.parameters()
                if p.requires_grad
            )
            encoder_total = sum(p.numel() for p in model.encoder.parameters())

            decoder_trainable = sum(
                p.numel()
                for p in model.decoder.parameters()
                if p.requires_grad
            )
            decoder_total = sum(p.numel() for p in model.decoder.parameters())

            print(f"\n{'='*60}")
            print("MODEL PARAMETER SUMMARY")
            print(f"{'='*60}")
            print("Encoder:")
            print(
                f"  Trainable: {encoder_trainable:,} / {encoder_total:,} "
                f"({100*encoder_trainable/encoder_total:.2f}%)"
            )
            print("\nDecoder:")
            print(
                f"  Trainable: {decoder_trainable:,} / {decoder_total:,} "
                f"({100*decoder_trainable/decoder_total:.2f}%)"
            )

        # Total parameters (always works)
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in model.parameters())

        print("\nTotal Model:")
        print(
            f"  Trainable: {trainable_params:,} / {total_params:,} "
            f"({100*trainable_params/total_params:.2f}%)"
        )
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"Could not generate detailed model summary: {e}")
        print("Printing basic parameter count only...")

        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in model.parameters())

        print(f"\n{'='*60}")
        print("MODEL PARAMETER SUMMARY")
        print(f"{'='*60}")
        print(
            f"Trainable: {trainable_params:,} / {total_params:,} "
            f"({100*trainable_params/total_params:.2f}%)"
        )
        print(f"{'='*60}\n")


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    train_losses,
    val_losses,
    checkpoint_path,
):
    """Save full checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to: {checkpoint_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """Load checkpoint and restore training state."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint["val_losses"]

    print(f"Resumed from epoch {checkpoint['epoch']}")
    print(f"Training history loaded: {len(train_losses)} epochs")

    return start_epoch, train_losses, val_losses


def load_model_weights(model, checkpoint_path, device):
    """Load model weights from checkpoint."""
    print(f"Loading model weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", None)
        print(f"Loaded model from checkpoint (new format)")
        if epoch is not None:
            print(f"Checkpoint was from epoch: {epoch}")
        return epoch
    else:
        try:
            model.load_state_dict(checkpoint)
            print(f"Loaded model from checkpoint (old format - state_dict)")
        except:
            if hasattr(model, "load_parameters"):
                model.load_parameters(checkpoint_path)
                print(f"Loaded model using model.load_parameters() method")
            else:
                raise ValueError(
                    "Unable to load checkpoint. Format not recognized."
                )
        return None


def evaluate_model(model, val_loader, image_processor, output_dir, device):
    """
    Evaluate Mask2Former model and generate visualizations.

    Args:
        model: Trained Mask2Former model
        val_loader: Validation dataloader
        image_processor: Image processor for post-processing
        output_dir: Directory to save outputs
        device: torch device
    """
    print(f"\n{'='*60}")
    print("EVALUATION MODE")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)
    visualization_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(visualization_dir, exist_ok=True)

    print("Generating evaluation visualizations...")
    visualize_predictions(
        model,
        val_loader,
        image_processor,
        device,
        visualization_dir,
        epoch="EVAL",
        n_samples=5,
    )

    print(f"\n{'='*60}")
    print("Evaluation completed!")
    print(f"Visualizations saved to: {visualization_dir}")
    print(f"{'='*60}\n")


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================


def train_model(
    model,
    train_loader,
    val_loader,
    image_processor,
    output_dir,
    num_epochs=50,
    learning_rate=1e-4,
    weight_decay=1e-4,
    checkpoint_every=10,
    visualize_every=10,
    device=None,
    mode="both",
    checkpoint_path=None,
):
    """
    Main training/evaluation loop for Mask2Former instance segmentation.

    Args:
        model: Mask2Former instance segmentation model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        image_processor: Image processor for post-processing
        output_dir: Directory to save checkpoints and visualizations
        num_epochs: Number of epochs to train
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        checkpoint_every: Save checkpoint every N epochs
        visualize_every: Visualize predictions every N epochs
        device: torch device (if None, will use cuda if available)
        mode: Operation mode - 'train', 'eval', or 'both'
        checkpoint_path: Path to checkpoint file for loading/resuming

    Returns:
        train_losses: Training losses per epoch (None if mode='eval')
        val_losses: Validation losses per epoch (None if mode='eval')
    """

    # Validate arguments
    if mode not in ["train", "eval", "both"]:
        raise ValueError(
            f"Invalid mode: {mode}. Must be 'train', 'eval', or 'both'"
        )

    if mode == "eval" and checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided when mode='eval'")

    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    model = model.to(device)

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    visualization_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)

    # EVALUATION ONLY MODE
    if mode == "eval":
        load_model_weights(model, checkpoint_path, device)
        evaluate_model(model, val_loader, image_processor, output_dir, device)
        return None, None

    # TRAINING MODE

    # NO CRITERION - Mask2Former computes loss internally!
    print(
        "Using Mask2Former's built-in loss (Hungarian matching + combined losses)"
    )

    # Optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    start_epoch = 1

    # Load checkpoint if resuming
    if checkpoint_path is not None:
        start_epoch, train_losses, val_losses = load_checkpoint(
            model, optimizer, scheduler, checkpoint_path, device
        )
        if len(val_losses) > 0:
            best_val_loss = min(val_losses)
            print(f"Best validation loss from checkpoint: {best_val_loss:.4f}")

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Starting from epoch: {start_epoch}")
    print(f"Checkpoints will be saved every {checkpoint_every} epochs")
    print(f"Visualizations will be saved every {visualize_every} epochs")

    # Note: print_model_summary might not work correctly with Mask2Former
    # Comment out if it causes errors
    # try:
    #     print_model_summary(model)
    # except:
    #     print(
    #         "(Skipping model summary - not compatible with Mask2Former structure)"
    #     )

    # Start timing
    training_start_time = time.time()
    epoch_times = []

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")

        # Train (no criterion)
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            desc=f"Epoch {epoch}/{num_epochs} [Train]",
        )
        train_losses.append(train_loss)

        # Validate (no criterion)
        val_loss = validate_epoch(
            model,
            val_loader,
            device,
            desc=f"Epoch {epoch}/{num_epochs} [Val]",
        )
        val_losses.append(val_loss)

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR:         {current_lr:.6f}")
        print(f"  Time:       {epoch_time:.2f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                train_losses,
                val_losses,
                best_path,
            )
            print(f"\n  ✅ Saved best model (val_loss: {val_loss:.4f})")

        # Save checkpoint periodically
        if epoch % checkpoint_every == 0:
            checkpoint_path_epoch = os.path.join(
                checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pt"
            )
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                train_losses,
                val_losses,
                checkpoint_path_epoch,
            )

        # Visualize predictions periodically
        if epoch % visualize_every == 0:
            print("\n  Generating visualizations...")
            visualize_predictions(
                model,
                val_loader,
                image_processor,
                device,
                visualization_dir,
                epoch,
            )

    # Calculate total training time
    total_training_time = time.time() - training_start_time
    avg_epoch_time = np.mean(epoch_times)

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(
        f"Total training time: {total_training_time:.2f}s "
        f"({total_training_time/60:.2f}m)"
    )
    print(f"Average time per epoch: {avg_epoch_time:.2f}s")
    print(f"{'='*60}\n")

    # Save final model
    final_path = os.path.join(checkpoint_dir, "final_model.pt")
    save_checkpoint(
        model,
        optimizer,
        scheduler,
        num_epochs,
        train_losses,
        val_losses,
        final_path,
    )
    print(f"Saved final model to: {final_path}")

    # Print training summary
    if len(train_losses) > 0 and len(val_losses) > 0:
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Final train loss: {train_losses[-1]:.4f}")
        print(f"Final val loss: {val_losses[-1]:.4f}")
        print(f"Best val loss: {min(val_losses):.4f}")
        print(
            f"Total training time: {total_training_time:.2f}s "
            f"({total_training_time/60:.2f}m)"
        )
        print(f"Average time per epoch: {avg_epoch_time:.2f}s")
        print(f"\nOutputs saved to: {output_dir}")
        print(f"  - Checkpoints: {checkpoint_dir}")
        print(f"  - Visualizations: {visualization_dir}")
        print("=" * 60)

    return train_losses, val_losses


if __name__ == "__main__":
    print(
        "Import this module and call train_model() with your model and dataloaders."
    )
