"""
driver.py
Training driver for DINO segmentation with configurable loss functions.
Supports training, evaluation, and checkpoint resumption.
Handles images with any number of channels (grayscale, RGB, multispectral, etc.).
"""

import os
import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from .sseg_utils import get_loss_function


def calculate_f1_score(pred, label):
    """
    Calculate F1 score for binary segmentation.

    Args:
        pred: Predicted mask (numpy array)
        label: Ground truth mask (numpy array)

    Returns:
        f1: F1 score
    """
    pred = pred.flatten()
    label = label.flatten()

    tp = np.sum((pred == 1) & (label == 1))
    fp = np.sum((pred == 1) & (label == 0))
    fn = np.sum((pred == 0) & (label == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return f1


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
                - None: no normalization for display
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

    elif method == None:
        img_normalized = img

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


def create_overlay_image(img_vis, pred_mask):
    """
    Create overlay of prediction mask on image.
    Handles grayscale and color images.

    Args:
        img_vis: Base image (H, W) for grayscale or (H, W, 3) for color
        pred_mask: Binary prediction mask (H, W)

    Returns:
        blended: Blended image with yellow overlay on predictions
    """
    # Ensure img_vis is 3-channel for overlay
    if img_vis.ndim == 2:
        # Convert grayscale to RGB
        img_vis_rgb = np.stack([img_vis] * 3, axis=2)
    else:
        img_vis_rgb = img_vis

    # Create RGBA image for proper alpha blending
    combined = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 4))
    combined[:, :, :3] = img_vis_rgb
    combined[:, :, 3] = 1.0  # Full opacity for base

    # Create overlay image with yellow predictions
    overlay = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 4))
    overlay[:, :, :3] = img_vis_rgb  # Start with image

    # Where prediction is 1, set to yellow with transparency
    mask_bool = pred_mask == 1
    overlay[mask_bool, 0] = 1.0  # Red channel
    overlay[mask_bool, 1] = 1.0  # Green channel
    overlay[mask_bool, 2] = 0.0  # Blue channel (0 = yellow)
    overlay[:, :, 3] = np.where(
        mask_bool, 0.5, 1.0
    )  # 50% transparent where pred=1

    # Blend images
    alpha = overlay[:, :, 3:4]
    blended = overlay[:, :, :3] * alpha + combined[:, :, :3] * (1 - alpha)

    return np.clip(blended, 0, 1)


def visualize_predictions(
    model, dataloader, device, output_dir, epoch, n_samples=5, dpi=300
):
    """
    Visualize model predictions and save to output directory.
    Handles images with any number of channels.

    Args:
        model: Trained model
        dataloader: DataLoader to sample from
        device: torch device
        output_dir (str): Directory to save plots
        epoch (int or str): Current epoch number or epoch label (e.g., "EVAL")
        n_samples (int): Number of samples to visualize
    """
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    # Get a batch of data
    images_list = []
    labels_list = []
    preds_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(images)

            # Convert to probabilities and threshold
            probs = torch.sigmoid(logits)  # [B, 2, H, W] or [B, 1, H, W]

            # Get crater class only (class 1) if 2 channels
            if probs.shape[1] == 2:
                probs = probs[
                    :, 1:2
                ]  # Select class 1, keep channel dim [B, 1, H, W]

            preds = (
                (probs > 0.5).float().squeeze(1)
            )  # [B, H, W] - squeeze channel dim

            # Move to CPU for visualization
            images_list.append(images.cpu())
            labels_list.append(labels.cpu())
            preds_list.append(preds.cpu())

            if sum(len(x) for x in images_list) >= n_samples:
                break

    # Concatenate batches
    all_images = torch.cat(images_list, dim=0)[:n_samples]
    all_labels = torch.cat(labels_list, dim=0)[:n_samples]
    all_preds = torch.cat(preds_list, dim=0)[:n_samples]

    # Get number of channels from first image
    num_channels = all_images.shape[1]

    # Create 4 row viz: img, pred, img/pred composite, ground truth
    batch_size = min(n_samples, len(all_images))
    fig, axes = plt.subplots(4, batch_size, figsize=(4 * batch_size, 16))

    # Handle single sample case
    if batch_size == 1:
        axes = axes.reshape(-1, 1)

    cmap_black_yellow = ListedColormap(["black", "yellow"])
    cmap_black_red = ListedColormap(["black", "red"])

    # Calculate F1 scores for each sample
    f1_scores = []

    # Track display mode for figure title
    display_mode = None

    for i in tqdm(range(batch_size), desc="Plotting predictions"):
        img = all_images[i].numpy().transpose(1, 2, 0)  # (H, W, C)
        label = all_labels[i].numpy()
        pred = all_preds[i].numpy()

        # Calculate F1 score for this sample
        f1 = calculate_f1_score(pred, label)
        f1_scores.append(f1)

        # Prepare image for display (handles any number of channels)
        img_vis, display_note = prepare_image_for_display(
            img,
            fix_rgb_order=False,  # Not needed for 5/7-band RGB extraction
            method="minmax",  # Fixes stripe artifacts
        )
        if display_mode is None:
            display_mode = display_note

        # Determine colormap for grayscale images
        cmap_image = "gray" if img_vis.ndim == 2 else None

        # Row 0: Original image with F1 score
        axes[0, i].imshow(img_vis, cmap=cmap_image)
        axes[0, i].set_title(
            f"Image {i}\nF1: {f1:.3f}", fontsize=16, fontweight="bold"
        )
        axes[0, i].axis("off")

        # Row 1: Predicted mask in black and yellow
        axes[1, i].imshow(pred, cmap=cmap_black_yellow, vmin=0, vmax=1)
        axes[1, i].set_title(f"Prediction {i}", fontsize=14)
        axes[1, i].axis("off")

        # Row 2: Combined overlay - image with yellow pred overlay
        blended = create_overlay_image(img_vis, pred)
        axes[2, i].imshow(blended)
        axes[2, i].set_title(f"Prediction Overlay {i}", fontsize=14)
        axes[2, i].axis("off")

        # Row 3: Ground truth mask in black and red
        axes[3, i].imshow(label, cmap=cmap_black_red, vmin=0, vmax=1)
        axes[3, i].set_title(f"Ground Truth {i}", fontsize=14)
        axes[3, i].axis("off")

    # Add overall F1 score and channel info as figure title
    mean_f1 = np.mean(f1_scores)
    fig.suptitle(
        f"Epoch {epoch} - Mean F1 Score: {mean_f1:.3f} | "
        f"Input: {num_channels}ch ({display_mode})",
        fontsize=20,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()

    # Handle epoch formatting (can be "eval" or a number)
    if isinstance(epoch, int):
        epoch_str = f"{epoch:03d}"
    else:
        epoch_str = str(epoch)

    save_path = os.path.join(output_dir, f"predictions_epoch_{epoch_str}.png")
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to {save_path}")
    print(f"Mean F1 Score: {mean_f1:.3f}")

    model.train()


def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    desc="Training",
    max_grad_norm=1.0,
):
    """
    Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: torch device
        desc (str): Description for progress bar

    Returns:
        avg_loss (float): Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    progress_bar = tqdm(dataloader, desc=desc)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(images)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_grad_norm
        )  # Optimizer step
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        n_batches += 1

        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / n_batches
    return avg_loss


def validate_epoch(model, dataloader, criterion, device, desc="Validation"):
    """
    Validate for one epoch.

    Args:
        model: Model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: torch device
        desc (str): Description for progress bar

    Returns:
        avg_loss (float): Average validation loss
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    progress_bar = tqdm(dataloader, desc=desc)

    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(images)

            # Compute loss
            loss = criterion(logits, labels)

            # Track metrics
            total_loss += loss.item()
            n_batches += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / n_batches
    return avg_loss


def print_model_summary(model):
    encoder_trainable = sum(
        p.numel() for p in model.encoder.parameters() if p.requires_grad
    )
    encoder_total = sum(p.numel() for p in model.encoder.parameters())

    # Calculate trainable parameters for decoder
    decoder_trainable = sum(
        p.numel() for p in model.decoder.parameters() if p.requires_grad
    )
    decoder_total = sum(p.numel() for p in model.decoder.parameters())

    # Calculate trainable parameters for combined model
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())

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
    print("\nCombined Model:")
    print(
        f"  Trainable: {trainable_params:,} / {total_params:,} "
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
    """
    Save full checkpoint including optimizer and scheduler state.

    Args:
        model: Model to save
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        train_losses: List of training losses
        val_losses: List of validation losses
        checkpoint_path: Path to save checkpoint
    """
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
    """
    Load checkpoint and restore training state.

    Args:
        model: Model to load weights into
        optimizer: Optimizer to restore state
        scheduler: Scheduler to restore state
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on

    Returns:
        start_epoch: Epoch to resume from
        train_losses: Training loss history
        val_losses: Validation loss history
    """
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
    """
    Load model weights from checkpoint, handling both old and new formats.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on

    Returns:
        epoch: Epoch number if available, else None
    """
    print(f"Loading model weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Check if it's the new format (dict with 'model_state_dict')
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", None)
        print(f"Loaded model from checkpoint (new format)")
        if epoch is not None:
            print(f"Checkpoint was from epoch: {epoch}")
        return epoch
    else:
        # Old format - directly load state dict or use model's load method
        try:
            # Try loading as state_dict directly
            model.load_state_dict(checkpoint)
            print(f"Loaded model from checkpoint (old format - state_dict)")
        except:
            # If model has custom load_parameters method (like DINOEncoderLoRA)
            if hasattr(model, "load_parameters"):
                model.load_parameters(checkpoint_path)
                print(f"Loaded model using model.load_parameters() method")
            else:
                raise ValueError(
                    "Unable to load checkpoint. Format not recognized and "
                    "model doesn't have load_parameters() method."
                )
        return None


def evaluate_model(model, val_loader, output_dir, device):
    """
    Evaluate model and generate visualizations.

    Args:
        model: Trained model
        val_loader: Validation dataloader
        output_dir: Directory to save visualizations
        device: Device to run evaluation on
    """
    print(f"\n{'='*60}")
    print("EVALUATION MODE")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)
    visualization_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(visualization_dir, exist_ok=True)

    print("Generating evaluation visualizations...")
    visualize_predictions(
        model, val_loader, device, visualization_dir, epoch="EVAL", n_samples=5
    )

    print(f"\n{'='*60}")
    print("Evaluation completed!")
    print(f"Visualizations saved to: {visualization_dir}")
    print(f"{'='*60}\n")


def train_model(
    model,
    train_loader,
    val_loader,
    output_dir,
    num_epochs=50,
    learning_rate=1e-4,
    weight_decay=1e-4,
    checkpoint_every=10,
    visualize_every=10,
    loss_type="cross_entropy",
    device=None,
    mode="both",
    checkpoint_path=None,
    max_grad_norm=1.0,
    early_stopping_patience=None,
    warmup_epochs=None,
):
    """
    Main training/evaluation loop for DINO segmentation.
    Supports images with any number of channels.

    Args:
        model: Segmentation model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        output_dir (str): Directory to save checkpoints and visualizations
        num_epochs (int): Number of epochs to train
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay for optimizer
        checkpoint_every (int): Save checkpoint every N epochs
        visualize_every (int): Visualize predictions every N epochs
        loss_type (str): Type of loss function. Options:
            - 'cross_entropy': Standard CrossEntropyLoss
            - 'focal': Focal Loss for small craters
            - 'dice': Dice Loss for segmentation
            - 'combined': CrossEntropy + Dice (recommended)
            - 'full': CrossEntropy + Dice + Boundary (unstable)
        device: torch device (if None, will use cuda if available)
        mode (str): Operation mode - 'train', 'eval', or 'both' (default both)
        checkpoint_path (str): Path to checkpoint file for loading/resuming

    Returns:
        train_losses (list): Training losses per epoch (None if mode='eval')
        val_losses (list): Validation losses per epoch (None if mode='eval')
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

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    visualization_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)

    # EVALUATION ONLY MODE
    if mode == "eval":
        # Load model weights (handles both old and new formats)
        load_model_weights(model, checkpoint_path, device)

        # Run evaluation
        evaluate_model(model, val_loader, output_dir, device)
        return None, None

    # TRAINING MODE (train or both)

    # Loss function - get from utils based on loss_type
    criterion = get_loss_function(loss_type)
    criterion = criterion.to(device)
    print(f"Using loss function: {loss_type}")
    print(f"Loss function: {criterion.__class__.__name__}")

    # Optimizer: only affect unfrozen model parameters
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # # Learning rate scheduler
    if warmup_epochs is None or warmup_epochs < 0:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif warmup_epochs > num_epochs:
        raise ValueError(
            "Number of warmup epochs must be less than or equal to total epochs."
        )
    elif warmup_epochs > 0.5 * num_epochs:
        print("Warning: warmup epochs is greater than 1/2 of total epochs.")
    else:
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=1e-7,  # Lower minimum
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

    # Load checkpoint if resuming training
    if checkpoint_path is not None:
        start_epoch, train_losses, val_losses = load_checkpoint(
            model, optimizer, scheduler, checkpoint_path, device
        )
        if len(val_losses) > 0:
            best_val_loss = min(val_losses)
            print(f"Best validation loss from checkpoint: {best_val_loss:.4f}")

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Starting from epoch: {start_epoch}")
    print(
        f"Checkpoints will be saved every {checkpoint_every} epochs to: "
        f"{checkpoint_dir}"
    )
    print(
        f"Visualizations will be saved every {visualize_every} epochs to: "
        f"{visualization_dir}"
    )

    print_model_summary(model)

    # Early stopping
    patience_counter = 0

    # Start timing
    training_start_time = time.time()
    epoch_times = []

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")

        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            desc=f"Epoch {epoch}/{num_epochs} [Train]",
            max_grad_norm=max_grad_norm,
        )
        train_losses.append(train_loss)

        # Validate
        val_loss = validate_epoch(
            model,
            val_loader,
            criterion,
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
                model, val_loader, device, visualization_dir, epoch
            )

        # Save best model
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
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
            print(f"  Saved best model (val_loss: {val_loss:.4f})")
            # Save best model...
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
            evaluate_model(model, val_loader, output_dir, device)

            # Generate final visualizations
            print(f"\n📸 Generating final visualizations...")
            visualize_predictions(
                model,
                val_loader,
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

    # Print training summary (only if we have losses)
    if len(train_losses) > 0 and len(val_losses) > 0:
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Final train loss: {train_losses[-1]:.4f}")
        print(f"Final val loss: {val_losses[-1]:.4f}")
        print(f"Best val loss: {min(val_losses):.4f}")
        print(
            f"Total training time: {total_training_time:.2f}s ({total_training_time/60:.2f}m)"
        )
        print(f"Average time per epoch: {avg_epoch_time:.2f}s")
        print(f"\nOutputs saved to: {output_dir}")
        print(f"  - Checkpoints: {checkpoint_dir}")
        print(f"  - Visualizations: {visualization_dir}")
        print("=" * 60)

    return train_losses, val_losses


if __name__ == "__main__":
    # This allows the driver to be run as a script or imported
    print(
        "Import this module and call train_model() with your model and dataloaders."
    )
