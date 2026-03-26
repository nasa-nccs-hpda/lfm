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
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
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


def instance_colormap(instance_mask):
    """
    Generate a deterministic, high-contrast color mapping for instance segmentation.
    0 values always map to black; other values map to a consistent, high-contrast colormap.

    Args:
        instance_mask: numpy array or torch tensor with instance IDs (0 = background)

    Returns:
        RGB image array (H, W, 3) with values in [0, 1]
    """
    # Convert to numpy if torch tensor
    if isinstance(instance_mask, torch.Tensor):
        instance_mask = instance_mask.cpu().numpy()

    instance_mask = instance_mask.astype(np.int32)

    # Create output RGB image
    h, w = instance_mask.shape[-2:]
    rgb_image = np.zeros((h, w, 3), dtype=np.float32)

    # Get unique instance IDs
    unique_ids = np.unique(instance_mask)

    # Function to generate deterministic color from instance ID
    def id_to_color(instance_id):
        """Generate a deterministic, high-contrast color for a given instance ID."""
        if instance_id == 0:
            return np.array([0.0, 0.0, 0.0])  # Black for background

        # Use golden ratio for well-distributed hues
        golden_ratio = 0.618033988749895
        hue = (instance_id * golden_ratio) % 1.0

        # Alternate saturation and value for more contrast
        saturation = 0.9 if instance_id % 2 == 0 else 0.7
        value = 0.95 if (instance_id // 2) % 2 == 0 else 0.75

        # Convert HSV to RGB
        rgb = mcolors.hsv_to_rgb([hue, saturation, value])
        return rgb

    # Apply colors to each instance
    for instance_id in unique_ids:
        mask = instance_mask == instance_id
        rgb_image[mask] = id_to_color(instance_id)

    return rgb_image


def prepare_image_for_display(
    img, fix_rgb_order=True, method="percentile", clip_percentile=2, std_clip=3
):
    """
    Prepare multi-channel image for matplotlib display.
    Unified version that handles various band configurations and normalization methods.

    Band Configuration Support:
    - 1-band: Grayscale
    - 3-band: RGB or BGR (controlled by fix_rgb_order)
    - 5-band: B, G, _, R, _ → extracts [3, 1, 0] for RGB display
    - 7-band: B, G, _, R, _, UV, UV → extracts [3, 1, 0] for RGB display
    - Other: Shows first 3 bands (optionally BGR→RGB if fix_rgb_order=True)

    Args:
        img: Image array with shape (H, W, C) where C is number of channels
        fix_rgb_order: If True, reorder BGR→RGB for 3-band or when showing first 3 bands
                       Has no effect for 5/7-band (RGB extraction already in correct order)
        method: Normalization method for display
                - 'percentile': Clips based on percentiles (robust to outliers) - RECOMMENDED
                - 'std_clip': Clips based on standard deviations (for z-score normalized data)
                - 'minmax': Simple min-max (fastest, but sensitive to outliers)
        clip_percentile: Percentile for clipping when method='percentile' (default: 2 = 2nd-98th)
        std_clip: Number of standard deviations to clip when method='std_clip' (default: 3)

    Returns:
        img_vis: Image ready for display - shape (H, W) for grayscale or (H, W, 3) for color
        display_note: String describing how the image was prepared
    """
    num_channels = img.shape[2]

    # Step 1: Apply normalization/clipping based on method
    if method == "percentile":
        # Percentile-based clipping (per band) - robust to outliers
        img_clipped = np.zeros_like(img)
        for c in range(num_channels):
            band = img[:, :, c]
            p_low, p_high = np.percentile(
                band, [clip_percentile, 100 - clip_percentile]
            )
            band_clipped = np.clip(band, p_low, p_high)
            if p_high > p_low:
                img_clipped[:, :, c] = (band_clipped - p_low) / (
                    p_high - p_low
                )
            else:
                img_clipped[:, :, c] = 0.5  # Constant band
        img_normalized = np.clip(img_clipped, 0, 1)

    elif method == "std_clip":
        # Standard deviation clipping - good for z-score normalized data
        img_clipped = np.clip(img, -std_clip, std_clip)
        # Rescale to [0, 1]
        img_normalized = (img_clipped + std_clip) / (2 * std_clip)

    elif method == "minmax":
        # Simple min-max normalization - sensitive to outliers but fast
        img_normalized = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img_normalized = np.clip(img_normalized, 0, 1)

    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'percentile', 'std_clip', or 'minmax'"
        )

    # Step 2: Select and arrange bands for display
    if num_channels == 1:
        # Grayscale
        img_vis = img_normalized[:, :, 0]
        display_note = "Grayscale"

    elif num_channels == 3:
        # 3-band: RGB or BGR
        img_vis = img_normalized
        if fix_rgb_order:
            img_vis = img_vis[..., [2, 1, 0]]  # BGR → RGB
            display_note = "RGB (BGR→RGB)"
        else:
            display_note = "RGB"

    elif num_channels == 5:
        # 5-band: B, G, _, R, _
        # Band indices: [0:B, 1:G, 2:?, 3:R, 4:?]
        # Extract R, G, B → indices [3, 1, 0] (already in RGB order)
        img_vis = img_normalized[:, :, [3, 1, 0]]
        display_note = "5ch (RGB from bands 3,1,0)"
        # fix_rgb_order has no effect here - already in correct order

    elif num_channels == 7:
        # 7-band: B, G, _, R, _, UV, UV
        # Band indices: [0:B, 1:G, 2:?, 3:R, 4:?, 5:UV, 6:UV]
        # Extract R, G, B → indices [3, 1, 0] (already in RGB order)
        img_vis = img_normalized[:, :, [3, 1, 0]]
        display_note = "7ch (RGB from bands 3,1,0)"
        # fix_rgb_order has no effect here - already in correct order

    else:
        # Other multi-band: show first 3 channels
        img_vis = img_normalized[:, :, :3]
        if fix_rgb_order:
            img_vis = img_vis[..., [2, 1, 0]]  # Assume BGR → RGB
            display_note = f"{num_channels}ch (first 3, BGR→RGB)"
        else:
            display_note = f"{num_channels}ch (first 3)"

    return img_vis, display_note


def convert_binary_masks_to_instance_map(binary_masks, class_labels=None):
    """
    Convert binary masks to instance segmentation map.
    Uses class_labels to identify background (class 0) vs instances (class 1).

    Args:
        binary_masks: (num_masks, H, W) binary mask array
        class_labels: (num_masks,) array where 0=background, 1=instance
    """
    if isinstance(binary_masks, torch.Tensor):
        binary_masks = binary_masks.cpu().numpy()
    if isinstance(class_labels, torch.Tensor):
        class_labels = class_labels.cpu().numpy()

    if binary_masks.ndim == 2:
        return binary_masks

    num_masks, h, w = binary_masks.shape
    instance_map = np.zeros((h, w), dtype=np.int32)

    instance_counter = 1
    for mask_id in range(num_masks):
        # Skip background masks (class_label == 0)
        if class_labels is not None and class_labels[mask_id] == 0:
            continue

        mask = binary_masks[mask_id] > 0.5
        instance_map[mask] = instance_counter
        instance_counter += 1

    return instance_map


def create_instance_overlay(img_vis, instance_mask, alpha=0.5):
    """
    Create overlay of instance mask on image.

    Args:
        img_vis: Base image (H, W) or (H, W, 3)
        instance_mask: Instance mask (H, W) with 0 = background
        alpha: Transparency for instances (0 = show image, 1 = show mask)

    Returns:
        overlay: (H, W, 3) RGB array
    """
    # Ensure img_vis is 3-channel
    if img_vis.ndim == 2:
        img_vis_rgb = np.stack([img_vis] * 3, axis=2)
    else:
        img_vis_rgb = img_vis.copy()

    # Colorize the mask (0 -> black, others -> distinct colors)
    colored_mask = instance_colormap(instance_mask)

    # Create alpha mask (0 for background, alpha for instances)
    alpha_mask = (instance_mask != 0).astype(np.float32) * alpha
    alpha_mask = alpha_mask[:, :, np.newaxis]  # (H, W, 1)

    # Blend: where mask is 0 (background), show original image
    # where mask is nonzero, blend mask with image
    overlay = img_vis_rgb * (1 - alpha_mask) + colored_mask * alpha_mask

    return np.clip(overlay, 0, 1)


def get_instance_bboxes(instance_mask):
    """
    Extract bounding boxes for each instance.

    Args:
        instance_mask: (H, W) array with unique IDs per instance (0 = background)

    Returns:
        bboxes: List of (x_min, y_min, width, height) for each instance
    """
    unique_instances = np.unique(instance_mask)
    unique_instances = unique_instances[unique_instances != 0]

    bboxes = []
    for instance_id in unique_instances:
        mask = instance_mask == instance_id
        rows, cols = np.where(mask)

        if len(rows) > 0:
            y_min, y_max = rows.min(), rows.max()
            x_min, x_max = cols.min(), cols.max()
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            bboxes.append((x_min, y_min, width, height))

    return bboxes


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
        epoch: Current epoch number or label
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
            mask_labels = batch["mask_labels"]

            batch_size = pixel_values.shape[0]
            target_sizes = [
                (pixel_values.shape[2], pixel_values.shape[3])
            ] * batch_size

            # Forward pass
            outputs = model(pixel_values=pixel_values)

            # Post-process
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
                post_processed = (
                    image_processor.post_process_instance_segmentation(
                        outputs,
                        threshold=threshold,
                        target_sizes=target_sizes,
                    )
                )

            # Convert to instance masks
            for idx, result in enumerate(post_processed):
                if "segmentation" in result:
                    instance_mask = result["segmentation"]
                    if isinstance(instance_mask, torch.Tensor):
                        instance_mask = instance_mask.cpu().numpy()
                    # Pass class labels if available
                    class_labels = result.get("segments_info", None)
                    if class_labels and "label_id" in class_labels[0]:
                        labels = np.array(
                            [seg["label_id"] for seg in class_labels]
                        )
                    else:
                        labels = None
                    instance_mask = convert_binary_masks_to_instance_map(
                        instance_mask, labels
                    )
                    preds_list.append(instance_mask)

            # Store images
            images_list.append(pixel_values.cpu())

            for idx, label_tensor in enumerate(mask_labels):
                label_numpy = label_tensor.cpu().numpy()
                class_label_numpy = (
                    batch["class_labels"][idx].cpu().numpy()
                )  # Get class labels!
                if label_numpy.ndim == 3:
                    label_2d = convert_binary_masks_to_instance_map(
                        label_numpy, class_label_numpy
                    )
                    labels_list.append(label_2d)
                else:
                    labels_list.append(label_numpy)

            if len(preds_list) >= n_samples:
                break

    # Limit to n_samples
    all_images = torch.cat(images_list, dim=0)[:n_samples]
    all_labels = labels_list[:n_samples]
    all_preds = preds_list[:n_samples]

    num_channels = all_images.shape[1]
    batch_size = min(n_samples, len(all_preds))

    # Create 5-row visualization (maintaining original 4 rows + 1 header = 5 rows)
    fig, axes = plt.subplots(5, batch_size, figsize=(4 * batch_size, 20))

    if batch_size == 1:
        axes = axes.reshape(-1, 1)

    # Calculate metrics
    instance_metrics = []
    semantic_f1s = []

    for i in tqdm(range(batch_size), desc="Plotting predictions"):
        img = all_images[i].numpy().transpose(1, 2, 0)  # (H, W, C)
        gt_mask = all_labels[i]
        pred_mask = all_preds[i]

        # Resize if needed
        if gt_mask.shape != pred_mask.shape:
            from skimage.transform import resize

            print(f"Warning: Shape mismatch at sample {i}. Resizing...")
            pred_mask = resize(
                pred_mask,
                gt_mask.shape,
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            ).astype(np.int32)

        # Calculate metrics
        inst_metrics = calculate_instance_metrics(pred_mask, gt_mask)
        sem_f1 = calculate_semantic_f1(pred_mask, gt_mask)

        instance_metrics.append(inst_metrics)
        semantic_f1s.append(sem_f1)

        # Prepare image for display
        # Just use the defaults - works exactly like your old version but more robust
        img_vis, note = prepare_image_for_display(
            img, fix_rgb_order=False, method="std_clip"
        )
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

        # Row 1: Predicted instances (pure mask, no overlay) with RED bounding boxes
        unique_pred = np.unique(pred_mask)
        num_pred = len(unique_pred[unique_pred != 0])
        pred_colored = instance_colormap(pred_mask)
        axes[1, i].imshow(pred_colored, vmin=0, vmax=1)

        # Draw red bounding boxes around predictions
        pred_bboxes = get_instance_bboxes(pred_mask)
        for bbox in pred_bboxes:
            x_min, y_min, width, height = bbox
            rect = patches.Rectangle(
                (x_min, y_min),
                width,
                height,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            axes[1, i].add_patch(rect)

        axes[1, i].set_title(
            f"Predicted ({num_pred} instances)\n"
            f"Precision: {inst_metrics['precision']:.3f}",
            fontsize=11,
        )
        axes[1, i].axis("off")

        # Row 2: Prediction overlay on image with RED bounding boxes
        pred_overlay = create_instance_overlay(img_vis, pred_mask, alpha=0.5)
        axes[2, i].imshow(pred_overlay, vmin=0, vmax=1)

        # Draw red bounding boxes around predictions
        for bbox in pred_bboxes:  # Reuse pred_bboxes from Row 1
            x_min, y_min, width, height = bbox
            rect = patches.Rectangle(
                (x_min, y_min),
                width,
                height,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            axes[2, i].add_patch(rect)

        axes[2, i].set_title(
            f"Prediction Overlay\n"
            f"Mean IoU: {inst_metrics['mean_iou']:.3f}",
            fontsize=11,
        )
        axes[2, i].axis("off")

        # Row 3: Ground truth instances (pure mask, no overlay) with RED bounding boxes
        unique_gt = np.unique(gt_mask)
        num_gt = len(unique_gt[unique_gt != 0])
        gt_colored = instance_colormap(gt_mask)
        axes[3, i].imshow(gt_colored, vmin=0, vmax=1)

        # Draw RED bounding boxes around ground truth (matching your requirement)
        gt_bboxes = get_instance_bboxes(gt_mask)
        for bbox in gt_bboxes:
            x_min, y_min, width, height = bbox
            rect = patches.Rectangle(
                (x_min, y_min),
                width,
                height,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            axes[3, i].add_patch(rect)

        axes[3, i].set_title(
            f"Ground Truth ({num_gt} instances)\n"
            f"Recall: {inst_metrics['recall']:.3f}",
            fontsize=11,
        )
        axes[3, i].axis("off")

        # Row 4: GT overlay on image with RED bounding boxes
        gt_overlay = create_instance_overlay(img_vis, gt_mask, alpha=0.5)
        axes[4, i].imshow(gt_overlay, vmin=0, vmax=1)

        # Draw red bounding boxes around ground truth
        for bbox in gt_bboxes:  # Reuse gt_bboxes from Row 3
            x_min, y_min, width, height = bbox
            rect = patches.Rectangle(
                (x_min, y_min),
                width,
                height,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            axes[4, i].add_patch(rect)

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


def train_epoch(
    model, dataloader, optimizer, device, desc="Training", max_grad_norm=None
):
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

        # Add gradient clipping
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

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
    max_grad_norm=1.0,
    early_stopping_patience=None,
    warmup_epochs=None,
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

    # Learning rate scheduler with optional warmup
    if warmup_epochs is None or warmup_epochs <= 0:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif warmup_epochs > num_epochs:
        raise ValueError(
            "Number of warmup epochs must be less than or equal to total epochs."
        )
    else:
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-7
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    start_epoch = 1

    # Early stopping
    patience_counter = 0

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
            max_grad_norm=max_grad_norm,
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

        # Save best model & check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset counter
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
        else:
            patience_counter += 1

        # Early stopping check
        if (
            early_stopping_patience
            and patience_counter >= early_stopping_patience
        ):
            print(f"\n{'='*60}")
            print(f"⚠️  Early stopping triggered after {epoch} epochs")
            print(f"No improvement for {early_stopping_patience} epochs")
            print(f"{'='*60}")

            # Load best model for final evaluation
            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            print(f"\n📊 Loading best model for final evaluation...")
            print(f"Best model path: {best_model_path}")
            load_model_weights(model, best_model_path, device)

            # Run evaluation on best model
            print(
                f"\n🔍 Running evaluation on best model (val_loss: {best_val_loss:.4f})..."
            )
            evaluate_model(
                model, val_loader, image_processor, output_dir, device
            )

            # Generate final visualizations
            print(f"\n📸 Generating final visualizations...")
            visualize_predictions(
                model,
                val_loader,
                image_processor,
                device,
                visualization_dir,
                epoch=f"early_stop_epoch_{epoch}",  # Mark as early stopped
            )

            print(f"\n✅ Early stopping evaluation complete")
            print(f"{'='*60}\n")
            break

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
