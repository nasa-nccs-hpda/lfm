"""Data pipeline adapters for lunar datasets.

This module provides TerraTorch-compatible wrappers for lunar-fm's
WebDataset-based data loading infrastructure and crater detection datasets.
"""

import json

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from lightning.pytorch import LightningDataModule
from matplotlib import patches
from matplotlib.figure import Figure
from PIL import Image
from skimage import measure
from torch import nn
from torch.utils.data import DataLoader, Dataset

from terratorch_integration.data_utils import (
    load_nc_band as _load_nc_band,
)  # noqa: F401  (re-exported)


class Normalize:
    """Normalize images with mean and std.

    Similar to mVHR10 normalization but adapted for lunar images.
    """

    def __init__(
        self,
        means: list[float],
        stds: list[float],
        max_pixel_value: float | None = None,
    ):
        """Initialize normalizer.

        Args:
            means: Mean values for each channel
            stds: Standard deviation values for each channel
            max_pixel_value: Optional max pixel value for scaling
        """
        super().__init__()
        self.means = means
        self.stds = stds
        self.max_pixel_value = max_pixel_value

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Apply normalization to batch.

        Args:
            batch: Batch dict with "image" key

        Returns:
            Normalized batch
        """
        batch["image"] = torch.stack(tuple(batch["image"]))
        image = (
            batch["image"] / self.max_pixel_value
            if self.max_pixel_value is not None
            else batch["image"]
        )

        # Handle different tensor shapes
        if len(image.shape) == 5:
            # Shape: (batch, channels, depth, height, width)
            means = torch.tensor(self.means, device=image.device).view(1, -1, 1, 1, 1)
            stds = torch.tensor(self.stds, device=image.device).view(1, -1, 1, 1, 1)
        elif len(image.shape) == 4:
            # Shape: (batch, channels, height, width)
            means = torch.tensor(self.means, device=image.device).view(1, -1, 1, 1)
            stds = torch.tensor(self.stds, device=image.device).view(1, -1, 1, 1)
        else:
            msg = (
                f"Expected batch to have 5 or 4 dimensions, but got {len(image.shape)}"
            )
            raise ValueError(msg)

        batch["image"] = (image - means) / stds
        return batch


class IdentityTransform(nn.Module):
    """Identity transform that returns input unchanged."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class LunarCraterDataset(Dataset):
    """Lunar crater detection dataset in COCO format.

    Similar to mVHR10 but adapted for lunar crater detection from LRO images.
    Supports bounding boxes and segmentation masks for crater detection.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms: Callable | None = None,
        boxes_output_tag: str = "boxes",
        labels_output_tag: str = "labels",
        masks_output_tag: str = "masks",
        scores_output_tag: str = "scores",
        image_size: int | None = None,
    ):
        """Initialize crater dataset.

        Args:
            root: Root directory containing train2017/val2017 and annotations
            split: "train" or "val"
            transforms: Optional transform function
            boxes_output_tag: Key for bounding boxes in output
            labels_output_tag: Key for labels in output
            masks_output_tag: Key for masks in output
            scores_output_tag: Key for scores in output
            image_size: Target image size for resizing (optional)
        """
        self.root = Path(root)
        self.split = split
        self.transforms = transforms
        self.image_size = image_size
        self.boxes_output_tag = boxes_output_tag
        self.labels_output_tag = labels_output_tag
        self.masks_output_tag = masks_output_tag
        self.scores_output_tag = scores_output_tag

        # Load COCO annotations
        split_name = f"{split}2017"
        self.img_dir = self.root / split_name
        ann_file = self.root / "annotations" / f"instances_{split_name}.json"

        if not ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")

        with open(ann_file, "r") as f:
            self.coco_data = json.load(f)

        # Build image id to annotations mapping
        self.img_id_to_anns = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)

        # Build image list
        self.images = self.coco_data["images"]
        self.categories = {
            cat["id"]: cat["name"] for cat in self.coco_data["categories"]
        }

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        img_info = self.images[index]
        img_id = img_info["id"]

        # Load image as grayscale (single channel)
        img_path = self.img_dir / img_info["file_name"]
        image = Image.open(img_path)

        # Ensure grayscale format
        if image.mode != "L":
            image = image.convert("L")

        # Get original size for bbox scaling
        orig_w, orig_h = image.size

        # Resize if image_size is specified
        if self.image_size is not None:
            image = image.resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            scale_x = self.image_size / orig_w
            scale_y = self.image_size / orig_h
        else:
            scale_x = 1.0
            scale_y = 1.0

        # Convert to tensor: (1, H, W) for single-channel grayscale
        image = torch.from_numpy(np.array(image)).unsqueeze(0).float()

        # Get annotations for this image
        anns = self.img_id_to_anns.get(img_id, [])

        # Extract boxes, labels, and masks
        boxes = []
        labels = []
        masks = []

        for ann in anns:
            # Bounding box in COCO format [x, y, width, height]
            bbox = ann["bbox"]
            # Convert to [x1, y1, x2, y2] and scale to resized image
            x1, y1, w, h = bbox
            x1_scaled = x1 * scale_x
            y1_scaled = y1 * scale_y
            x2_scaled = (x1 + w) * scale_x
            y2_scaled = (y1 + h) * scale_y
            boxes.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled])

            # Label mapping for single-class crater detection.
            # Faster R-CNN expects foreground classes in [1, num_classes - 1],
            # with 0 reserved for background. The COCO annotations may use
            # arbitrary category IDs, so remap every crater annotation to 1.
            labels.append(1)

            # TerraTorch's detection task expects a non-empty mask list per sample
            # because it concatenates masks during batch reformatting. We therefore
            # attach a placeholder mask for every annotation, even though Faster R-CNN
            # only uses boxes and labels here.
            masks.append(torch.zeros((0, 0), dtype=torch.uint8))

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            masks_list = masks
        else:
            # Negative samples with no objects still need one placeholder mask so
            # TerraTorch's torch.cat() over the per-sample mask list does not fail.
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            masks_list = [torch.zeros((0, 0), dtype=torch.uint8)]

        sample = {
            "image": image,
            self.boxes_output_tag: boxes_tensor,
            self.labels_output_tag: labels_tensor,
            self.masks_output_tag: masks_list,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def plot(
        self,
        sample: dict[str, torch.Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
        show_feats: str = "both",
        box_alpha: float = 0.7,
        mask_alpha: float = 0.7,
        confidence_score: float = 0.5,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: A sample returned by __getitem__
            show_titles: Flag indicating whether to show titles above each panel
            suptitle: Optional string to use as a suptitle
            show_feats: Features to show: "boxes", "masks", or "both"
            box_alpha: Alpha value for bounding boxes
            mask_alpha: Alpha value for masks
            confidence_score: Minimum confidence score for predictions

        Returns:
            A matplotlib Figure with the rendered sample

        Example:
            >>> dataset = LunarCraterDataset(root="data/LRO_craters", split="train")
            >>> sample = dataset[0]
            >>> fig = dataset.plot(sample, suptitle="Lunar Crater Detection")
            >>> plt.show()
        """
        assert show_feats in {
            "boxes",
            "masks",
            "both",
        }, f"show_feats must be 'boxes', 'masks', or 'both', got {show_feats}"

        # Normalize image for display (handle single-channel grayscale)
        image = (
            sample["image"].squeeze().cpu().numpy()
        )  # Remove channel dim for grayscale

        # Normalize to [0, 1] for display
        if image.max() > 1.0:
            image = image / 255.0

        # Get ground truth annotations
        boxes = sample[self.boxes_output_tag].cpu().numpy()
        labels = sample[self.labels_output_tag].cpu().numpy()

        # Check if we have masks
        has_masks = (
            self.masks_output_tag in sample
            and len(sample[self.masks_output_tag]) > 0
            and sample[self.masks_output_tag][0].numel() > 0
        )

        if has_masks:
            masks = [
                mask.squeeze().cpu().numpy() for mask in sample[self.masks_output_tag]
            ]

        n_gt = len(boxes)

        # Check if we have predictions
        show_predictions = f"prediction_{self.labels_output_tag}" in sample
        ncols = 2 if show_predictions else 1

        if show_predictions:
            prediction_labels = (
                sample[f"prediction_{self.labels_output_tag}"].cpu().numpy()
            )
            prediction_scores = (
                sample[f"prediction_{self.scores_output_tag}"].cpu().numpy()
            )
            show_pred_boxes = f"prediction_{self.boxes_output_tag}" in sample
            show_pred_masks = f"prediction_{self.masks_output_tag}" in sample

            if show_pred_boxes:
                prediction_boxes = (
                    sample[f"prediction_{self.boxes_output_tag}"].cpu().numpy()
                )
            if show_pred_masks:
                prediction_masks = [
                    mask.squeeze().cpu().numpy()
                    for mask in sample[f"prediction_{self.masks_output_tag}"]
                ]

            n_pred = len(prediction_labels)

        # Create figure
        fig, axs = plt.subplots(ncols=ncols, squeeze=False, figsize=(ncols * 10, 10))

        # Plot ground truth
        axs[0, 0].imshow(image, cmap="gray")
        axs[0, 0].axis("off")

        # Use colormap for different classes (though craters are single class)
        cm = plt.get_cmap("gist_rainbow")
        num_classes = len(self.categories)

        for i in range(n_gt):
            class_num = labels[i]
            color = cm(class_num / max(num_classes, 2))  # Avoid division by zero

            # Add bounding boxes
            if show_feats in {"boxes", "both"}:
                x1, y1, x2, y2 = boxes[i]
                r = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    alpha=box_alpha,
                    linestyle="dashed",
                    edgecolor=color,
                    facecolor="none",
                )
                axs[0, 0].add_patch(r)

                # Add label
                label_name = self.categories.get(class_num, f"Class {class_num}")
                axs[0, 0].text(
                    x1,
                    y1 - 8,
                    label_name,
                    color="white",
                    size=11,
                    backgroundcolor="black",
                    alpha=0.7,
                )

            # Add masks (if available and requested)
            if show_feats in {"masks", "both"} and has_masks:
                mask = masks[i]
                if mask.size > 0:
                    contours = measure.find_contours(mask, 0.5)
                    for verts in contours:
                        verts = np.fliplr(verts)
                        p = patches.Polygon(
                            verts, facecolor=color, alpha=mask_alpha, edgecolor="white"
                        )
                        axs[0, 0].add_patch(p)

        if show_titles:
            axs[0, 0].set_title("Ground Truth", fontsize=14, fontweight="bold")

        # Plot predictions (if available)
        if show_predictions:
            axs[0, 1].imshow(image, cmap="gray")
            axs[0, 1].axis("off")

            for i in range(n_pred):
                score = prediction_scores[i]
                if score < confidence_score:
                    continue

                class_num = prediction_labels[i]
                color = cm(class_num / max(num_classes, 2))

                if show_pred_boxes:
                    # Add bounding boxes
                    x1, y1, x2, y2 = prediction_boxes[i]
                    r = patches.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=2,
                        alpha=box_alpha,
                        linestyle="dashed",
                        edgecolor=color,
                        facecolor="none",
                    )
                    axs[0, 1].add_patch(r)

                    # Add label with confidence score
                    label_name = self.categories.get(class_num, f"Class {class_num}")
                    caption = f"{label_name} {score:.3f}"
                    axs[0, 1].text(
                        x1,
                        y1 - 8,
                        caption,
                        color="white",
                        size=11,
                        backgroundcolor="black",
                        alpha=0.7,
                    )

                # Add prediction masks (if available)
                if show_pred_masks:
                    mask = prediction_masks[i]
                    if mask.size > 0:
                        contours = measure.find_contours(mask, 0.5)
                        for verts in contours:
                            verts = np.fliplr(verts)
                            p = patches.Polygon(
                                verts,
                                facecolor=color,
                                alpha=mask_alpha,
                                edgecolor="white",
                            )
                            axs[0, 1].add_patch(p)

            if show_titles:
                axs[0, 1].set_title("Prediction", fontsize=14, fontweight="bold")

        if suptitle is not None:
            plt.suptitle(suptitle, fontsize=16, fontweight="bold")

        plt.tight_layout()

        return fig


class LunarCraterDataModule(LightningDataModule):
    """TerraTorch-compatible data module for lunar crater detection.

    Similar to mVHR10DataModule but adapted for lunar crater detection.
    Includes normalization and transform support.
    """

    def __init__(
        self,
        root: str = "../data/LRO_craters",
        batch_size: int = 4,
        num_workers: int = 0,
        image_size: int = 512,
        pad: bool = True,
        collate_fn: Callable | None = None,
        boxes_output_tag: str = "boxes",
        labels_output_tag: str = "labels",
        masks_output_tag: str = "masks",
        scores_output_tag: str = "scores",
        apply_norm_in_datamodule: bool = True,
        norm_means: list[float] | None = None,
        norm_stds: list[float] | None = None,
        max_pixel_value: float = 255.0,
        train_transforms: Callable | None = None,
        val_transforms: Callable | None = None,
        test_transforms: Callable | None = None,
    ):
        """Initialize crater data module.

        Args:
            root: Root directory containing train2017/val2017 and annotations
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes
            image_size: Target image size
            pad: Whether to pad images
            collate_fn: Optional custom collate function
            boxes_output_tag: Key for bounding boxes
            labels_output_tag: Key for labels
            masks_output_tag: Key for masks
            scores_output_tag: Key for scores
            apply_norm_in_datamodule: Whether to apply normalization
            norm_means: Mean values for normalization (default: ImageNet means)
            norm_stds: Std values for normalization (default: ImageNet stds)
            max_pixel_value: Max pixel value for scaling
            train_transforms: Optional training transforms
            val_transforms: Optional validation transforms
            test_transforms: Optional test transforms
        """
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.pad = pad
        self.boxes_output_tag = boxes_output_tag
        self.labels_output_tag = labels_output_tag
        self.masks_output_tag = masks_output_tag
        self.scores_output_tag = scores_output_tag
        self.apply_norm_in_datamodule = apply_norm_in_datamodule

        # Set up normalization for single-channel grayscale
        if apply_norm_in_datamodule:
            # Use grayscale normalization (single channel)
            # Default to generic grayscale values if not specified
            means = norm_means or [0.5]  # Single channel mean
            stds = norm_stds or [0.25]  # Single channel std
            self.aug = Normalize(means, stds, max_pixel_value=max_pixel_value)
        else:
            # No normalization - just scale to [0, 1]
            self.aug = Normalize([0], [1], max_pixel_value=max_pixel_value)

        # Set up collate function
        if collate_fn is None:
            self.collate_fn = lambda batch: collate_fn_detection(
                batch,
                boxes_tag=boxes_output_tag,
                labels_tag=labels_output_tag,
                masks_tag=masks_output_tag,
            )
        else:
            self.collate_fn = collate_fn

        # Store transforms
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms

        # Datasets will be created in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str) -> None:
        """Setup datasets for the given stage.

        Args:
            stage: "fit", "validate", or "test"
        """
        if stage in ["fit"]:
            self.train_dataset = LunarCraterDataset(
                root=self.root,
                split="train",
                transforms=self.train_transforms,
                boxes_output_tag=self.boxes_output_tag,
                labels_output_tag=self.labels_output_tag,
                masks_output_tag=self.masks_output_tag,
                scores_output_tag=self.scores_output_tag,
                image_size=self.image_size,
            )

        if stage in ["fit", "validate"]:
            self.val_dataset = LunarCraterDataset(
                root=self.root,
                split="val",
                transforms=self.val_transforms,
                boxes_output_tag=self.boxes_output_tag,
                labels_output_tag=self.labels_output_tag,
                masks_output_tag=self.masks_output_tag,
                scores_output_tag=self.scores_output_tag,
                image_size=self.image_size,
            )

        if stage in ["test"]:
            self.test_dataset = LunarCraterDataset(
                root=self.root,
                split="val",  # Use val as test if no separate test set
                transforms=self.test_transforms,
                boxes_output_tag=self.boxes_output_tag,
                labels_output_tag=self.labels_output_tag,
                masks_output_tag=self.masks_output_tag,
                scores_output_tag=self.scores_output_tag,
                image_size=self.image_size,
            )

    def _dataloader_factory(self, split: str) -> DataLoader:
        """Create dataloader for given split.

        Args:
            split: "train", "val", or "test"

        Returns:
            Configured DataLoader
        """
        dataset_attr = f"{split}_dataset"
        if not hasattr(self, dataset_attr) or getattr(self, dataset_attr) is None:
            stage = "fit" if split == "train" else split
            self.setup(stage)

        dataset = getattr(self, dataset_attr)

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=(split == "train"),
            num_workers=self.num_workers,
            collate_fn=lambda batch: self.aug(self.collate_fn(batch)),
        )

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return self._dataloader_factory("train")

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return self._dataloader_factory("val")

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return self._dataloader_factory("test")


def collate_fn_detection(
    batch: list[dict[str, Any]],
    boxes_tag: str = "boxes",
    labels_tag: str = "labels",
    masks_tag: str = "masks",
    include_metadata: bool = False,
) -> dict[str, Any]:
    """Collate function for detection datasets.

    Args:
        batch: list of samples
        boxes_tag: Key for bounding boxes
        labels_tag: Key for labels
        masks_tag: Key for masks
        include_metadata: If ``True``, stack the ``"metadata"`` tensors into
                          a ``(B, _METADATA_MAX_TOKENS)`` LongTensor.

    Returns:
        Collated batch
    """
    new_batch = {
        "image": torch.stack([item["image"] for item in batch]),
        boxes_tag: [item[boxes_tag] for item in batch],
        labels_tag: [item[labels_tag] for item in batch],
        masks_tag: [item[masks_tag] for item in batch],
    }
    if include_metadata:
        new_batch["metadata"] = torch.stack([item["metadata"] for item in batch])
    return new_batch


# ---------------------------------------------------------------------------
# NAC + DTM multimodal dataset
# ---------------------------------------------------------------------------

_NACDTM_OUTPUT_MODES = ("stack", "separate", "packed")

# Default sequence length reserved per sequence modality (e.g. metadata) in
# the "packed" output mode. Must match LunarBackbone's modality_info
# max_tokens default so unpacking aligns.
_PACKED_SEQ_LEN = 12

# Sentinel used to pad sequence-modality channels in "packed" mode.
_PACKED_SEQ_PAD = -1

# Mapping from parquet column names to the variable names used by MetadataTransform
# and seen by the model during pretraining.
_PARQUET_TO_METADATA_VAR: dict[str, str] = {
    "EMISSION_ANGLE": "EM_ANG",
    "INCIDENCE_ANGLE": "INC_ANG",
    "PHASE_ANGLE": "PHASE_ANG",
    "SUB_SOLAR_GROUND_AZIMUTH": "SS_GROUND_AZIMUTH",
    "SUB_SOLAR_LATITUDE": "SS_LAT",
    "SUB_SOLAR_LONGITUDE": "SS_LON",
    "UPPER_LEFT_LONGITUDE": "UL_LON",
    "UPPER_LEFT_LATITUDE": "UL_LAT",
    "LOWER_RIGHT_LONGITUDE": "LR_LON",
    "LOWER_RIGHT_LATITUDE": "LR_LAT",
}

# Number of metadata tokens (one per variable, no EOS — that is added by the masking layer)
_METADATA_MAX_TOKENS = 10


class MetadataEncoder:
    """Convert a parquet metadata row to a token-ID tensor compatible with the model.

    Replicates the exact preprocessing pipeline used during pretraining:

    1. Rename columns to the canonical ``MetadataTransform.METADATA_VARS`` names.
    2. Build the newline-separated ``KEY=value`` string the transform expects.
    3. Run :class:`~terramind.data.modality_transforms.MetadataTransform` to
       produce binned range-token strings (``"EM_ANG=1.00-->1.50"`` etc.).
    4. Encode each token string with the WordPiece text tokenizer → one integer
       ID per variable.
    5. Return a ``torch.LongTensor`` of shape ``(_METADATA_MAX_TOKENS,)``.

    Args:
        tokenizer_path: Path to the ``*.json`` HuggingFace tokenizer file.
        binning_config_path: Path to the binning CSV used by
            :class:`~terramind.data.modality_transforms.MetadataTransform`.
            Defaults to the bundled file in ``terramind/utils/tokenizer/``.
        shuffle: Whether to shuffle variable order (match training behaviour).
            Defaults to ``False`` for deterministic inference.
    """

    _DEFAULT_TOKENIZER = (
        "terramind/utils/tokenizer/trained/"
        "text_tokenizer_terramind_wordpiece_30k_binned_xyxy_reindexed.json"
    )
    _DEFAULT_BINNING = "terramind/utils/tokenizer/trained/metadata_binning_config.csv"

    def __init__(
        self,
        tokenizer_path: str = _DEFAULT_TOKENIZER,
        binning_config_path: str = _DEFAULT_BINNING,
        shuffle: bool = False,
    ):
        # Lazy-import to avoid hard dependency when metadata is not used
        try:
            from tokenizers import Tokenizer as _HFTokenizer
            from terramind.data.modality_transforms import MetadataTransform
            from terramind.utils.tokenizer.text_tokenizer import encode_sequence
        except ImportError as e:
            raise ImportError(
                "MetadataEncoder requires 'tokenizers' and the terramind package. "
                f"Original error: {e}"
            ) from e

        self._encode_sequence = encode_sequence
        self._tokenizer = _HFTokenizer.from_file(tokenizer_path)
        self._transform = MetadataTransform(
            binning_config_path=binning_config_path,
            shuffle=shuffle,
        )

    def encode(self, row: "pd.Series") -> torch.Tensor:
        """Encode one metadata row to a ``(10,)`` LongTensor.

        Args:
            row: A pandas Series (one row of the metadata parquet) containing
                 the columns defined in ``_PARQUET_TO_METADATA_VAR``.

        Returns:
            ``torch.LongTensor`` of shape ``(_METADATA_MAX_TOKENS,)`` with one
            token ID per metadata variable in the canonical order.
        """
        # Build the canonical text representation
        text = "\n".join(
            f"{var_name}={row[col]}"
            for col, var_name in _PARQUET_TO_METADATA_VAR.items()
        )
        # Bin → list of range-token strings, one per variable
        binned: list[str] = self._transform.postprocess(text)
        # Tokenize: returns list-of-lists (one sub-list per token chunk)
        ids_nested = self._encode_sequence(
            binned, self._tokenizer, max_tokens=_METADATA_MAX_TOKENS
        )
        flat = [i for chunk in ids_nested for i in chunk]

        return torch.tensor(flat, dtype=torch.long)


class LocalNormalizeDTM:
    """Normalize the DTM channel using per-image mean and a global std.

    Formula: ``dtm_norm = (dtm - image_mean) / global_std``

    Works for both output modes:

    * ``"stack"``    – operates on ``batch["image"][:, 1:2, :, :]``
    * ``"separate"`` – operates on ``batch["dtm_3m"]``  (shape ``B, 1, H, W``)
    """

    def __init__(
        self, dtm_global_std: float, output_mode: str = "stack", channel: int | None = 1
    ):
        """Initialize.

        Args:
            dtm_global_std: Pre-computed global standard deviation for DTM.
                            Each image is centred on its own spatial mean
                            then divided by this value.
            output_mode: ``"stack"`` or ``"separate"``.
            channel: In ``"stack"`` mode, which channel index holds DTM.
                     Defaults to 1 (NAC=0, DTM=1). When only DTM is used the
                     data module passes 0.
        """
        self.dtm_global_std = dtm_global_std
        self.output_mode = output_mode
        self.channel = channel if channel is not None else 1

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Apply local DTM normalisation in-place.

        Args:
            batch: Collated batch dict.

        Returns:
            Batch with DTM normalised.
        """
        if self.output_mode == "separate":
            dtm = batch["dtm_3m"]  # (B, 1, H, W)
            dtm_mean = dtm.mean(dim=(-2, -1), keepdim=True)
            batch["dtm_3m"] = (dtm - dtm_mean) / self.dtm_global_std
        else:
            image = batch["image"]  # (B, C, H, W)
            c = self.channel
            dtm = image[:, c : c + 1, :, :]
            dtm_mean = dtm.mean(dim=(-2, -1), keepdim=True)
            image[:, c : c + 1, :, :] = (dtm - dtm_mean) / self.dtm_global_std
            batch["image"] = image
        return batch


class LunarNACDTMDataset(Dataset):
    """Lunar crater detection dataset over NAC and/or DTM imagery.

    Image paths are loaded from a ``metadata.parquet`` file that contains
    ``PHO_TILE`` and ``DTM_TILE`` columns with paths relative to ``data_dir``,
    and a ``dataset`` column (``"train"`` / ``"val"`` / ``"test"``) that
    determines the split.

    Which modalities are loaded is controlled by ``modalities`` (default
    ``["nac", "dtm"]``).  Pass ``["nac"]`` or ``["dtm"]`` to use only one.

    Invalid pixels — ``NaN`` values and the ``-9999.0`` no-data sentinel — are
    imputed with ``0.0`` after loading.  Any annotation whose bounding box
    touches or overlaps at least one invalid pixel in *any* loaded modality is
    removed from that sample before the imputation is applied.

    Annotations follow the COCO format stored in a separate JSON.

    **Output modes** (controlled by ``output_mode``):

    * ``"stack"`` *(default)* – ``"image"`` key holds a ``(2, H, W)`` tensor
      with NAC on channel 0 and DTM on channel 1.  Compatible with any
      single-image backbone that accepts 2-channel input.
    * ``"separate"`` – separate ``"nac"`` (``1, H, W``) and ``"dtm_3m"``
      (``1, H, W``) keys.  Compatible with the multi-modal ``LunarBackbone``
      that expects one tensor per modality key.
    * ``"packed"`` – single ``"image"`` ``(C, H, W)`` tensor that stacks NAC
      (1 ch) + DTM (1 ch) plus one extra channel per sequence modality
      (currently metadata).  The extra channel's flattened spatial layout
      holds the token IDs in the first ``_PACKED_SEQ_LEN`` positions;
      remaining positions are filled with ``_PACKED_SEQ_PAD`` (= -1).
      :class:`LunarBackbone` unpacks this tensor back into a per-modality
      dict in its ``forward()``.  In this mode all normalization (NAC
      min-max + global mean/std, DTM mask + local norm) must be done by
      the dataset itself — the data module's batch-level transforms are
      bypassed because the extra metadata channel would otherwise be
      polluted by image-only normalisers.
    """

    #: sentinel value treated as no-data in every loaded modality
    NODATA_VALUE: float = -9999.0

    def __init__(
        self,
        data_dir: str,
        metadata_file: str,
        annotations_file: str,
        modalities: list[str] | None = None,
        used_for_labeling_only: bool = False,
        drop_no_crater_images: bool = False,
        split: str = "train",
        transforms: Callable | None = None,
        boxes_output_tag: str = "boxes",
        labels_output_tag: str = "labels",
        masks_output_tag: str = "masks",
        scores_output_tag: str = "scores",
        image_size: int | None = None,
        min_diameter_m: float = 5.0,
        output_mode: str = "stack",
        task_mode: str = "detection",
        circle_diameter_mode: str = "min",
        keep_boxes: bool = False,
        nac_scale_factor: float = 1.0,
        metadata_tokenizer_path: str = MetadataEncoder._DEFAULT_TOKENIZER,
        metadata_binning_config_path: str = MetadataEncoder._DEFAULT_BINNING,
        # Normalisation parameters — applied at sample level for all output modes.
        nac_valid_min: float | None = None,
        nac_valid_max: float | None = None,
        nac_mask_threshold: float | None = None,
        nac_mask_fill_value: float = 0.0,
        nac_mask_eps: float = 1e-6,
        nac_norm_mean: float | None = None,
        nac_norm_std: float | None = None,
        dtm_global_std: float | None = None,
        dtm_mask_threshold: float | None = None,
        dtm_mask_fill_value: float = 0.0,
    ):
        """Initialize the dataset.

        Args:
            data_dir: Root directory that ``PHO_TILE`` / ``DTM_TILE`` paths
                      from the parquet are resolved against.
            metadata_file: Path to ``metadata.parquet``.  Must contain columns
                           ``PHO_TILE``, ``DTM_TILE``, ``dataset``
                           (``"train"`` / ``"val"`` / ``"test"``), and
                           ``used_for_labeling`` (``"yes"`` / ``"no"``).
                           Used as the sole source of splits and image paths.
                           Also provides metadata tokens for the model.
            annotations_file: Path to the COCO-format annotations JSON.
            modalities: List of modalities to load.  Supported values are
                        ``"nac"`` and ``"dtm"``.  Defaults to
                        ``["nac", "dtm"]``.  Pass ``["nac"]`` or ``["dtm"]``
                        for single-modality use.
            used_for_labeling_only: When ``True``, restrict the dataset to rows
                                    where ``used_for_labeling == "yes"``.
            drop_no_crater_images: When ``True``, remove from the split any
                                   sample that has no surviving annotations
                                   after both the ``min_diameter_m`` filter
                                   *and* the nodata-overlap check (i.e. boxes
                                   touching NaN / ``-9999`` pixels are also
                                   excluded).  Tiles with no COCO entry at all
                                   are also dropped.  Note: this reads every
                                   tile file once at init time.
            split: Which split to load (``"train"``, ``"val"``, or ``"test"``).
            transforms: Optional per-sample transform callable.
            boxes_output_tag: Output key for bounding boxes.
            labels_output_tag: Output key for labels.
            masks_output_tag: Output key for masks.
            scores_output_tag: Output key for scores.
            image_size: If set, resize all modalities to this square size.
            min_diameter_m: Minimum crater diameter in pixels to keep.
                             Craters whose bounding box satisfies
                             ``min(bbox_w, bbox_h) < min_diameter_m`` are
                             discarded.  Set to 0 to disable filtering.
            output_mode: ``"stack"``, ``"separate"``, or ``"packed"``.
                         See class docstring.
            nac_scale_factor: Scalar applied to raw NAC pixel values before
                              any batch-level normalisation.  The handlabeled
                              ``.nc`` files store raw DN values (1–253) while
                              the pretraining shards store values in ``[0, 1]``.
                              The default ``1/255`` brings the DN range into
                              ``[0, 1]`` so that pretraining mean/std stats
                              (``mean=0.0566, std=0.0395``) apply correctly.
                              Set to ``1.0`` to disable scaling.
            metadata_tokenizer_path: Path to the HuggingFace ``*.json``
                                     tokenizer file.  Defaults to the bundled
                                     reindexed WordPiece tokenizer.
            metadata_binning_config_path: Path to the binning CSV consumed by
                                          ``MetadataTransform``.
        """
        if output_mode not in _NACDTM_OUTPUT_MODES:
            raise ValueError(
                f"output_mode must be one of {_NACDTM_OUTPUT_MODES}, got '{output_mode}'"
            )
        if task_mode not in {"detection", "segmentation"}:
            raise ValueError(
                f"task_mode must be 'detection' or 'segmentation', got '{task_mode}'"
            )
        if circle_diameter_mode not in {"min", "max", "mean"}:
            raise ValueError(
                f"circle_diameter_mode must be 'min', 'max', or 'mean', got '{circle_diameter_mode}'"
            )
        if (nac_norm_mean is None) != (nac_norm_std is None):
            raise ValueError(
                "nac_norm_mean and nac_norm_std must both be set or both be None."
            )

        _modalities = modalities if modalities is not None else ["nac", "dtm"]
        for m in _modalities:
            if m not in ("nac", "dtm", "metadata"):
                raise ValueError(
                    f"modalities entries must be 'nac', 'dtm', or 'metadata', got '{m}'"
                )
        if not _modalities:
            raise ValueError("modalities must contain at least one entry.")

        self.task_mode = task_mode
        self.circle_diameter_mode = circle_diameter_mode
        self.keep_boxes = keep_boxes
        self.nac_valid_min = nac_valid_min
        self.nac_valid_max = nac_valid_max
        self.nac_mask_threshold = nac_mask_threshold
        self.nac_mask_fill_value = nac_mask_fill_value
        self.nac_mask_eps = nac_mask_eps
        self.nac_norm_mean = nac_norm_mean
        self.nac_norm_std = nac_norm_std
        self.dtm_global_std = dtm_global_std
        self.dtm_mask_threshold = dtm_mask_threshold
        self.dtm_mask_fill_value = dtm_mask_fill_value
        self.data_dir = Path(data_dir)
        self.use_nac = "nac" in _modalities
        self.use_dtm = "dtm" in _modalities
        self.use_metadata = "metadata" in _modalities
        self.split = split
        self.transforms = transforms
        self.image_size = image_size
        self.min_diameter_m = min_diameter_m
        self.output_mode = output_mode
        self.nac_scale_factor = nac_scale_factor
        self.boxes_output_tag = boxes_output_tag
        self.labels_output_tag = labels_output_tag
        self.masks_output_tag = masks_output_tag
        self.scores_output_tag = scores_output_tag
        self._metadata_encoder = MetadataEncoder(
            tokenizer_path=metadata_tokenizer_path,
            binning_config_path=metadata_binning_config_path,
        )

        # ------------------------------------------------------------------
        # Load metadata parquet — source of splits, image paths, and metadata
        # ------------------------------------------------------------------
        meta_df = pd.read_parquet(metadata_file)

        if used_for_labeling_only:
            meta_df = meta_df[meta_df["used_for_labeling"] == "yes"]

        split_df = meta_df[meta_df["dataset"] == split].reset_index(drop=True)
        if split_df.empty:
            raise KeyError(
                f"Split '{split}' not found in {metadata_file}. "
                f"Available splits: {list(meta_df['dataset'].unique())}"
            )

        # Build per-sample path records and metadata lookup indexed by pho filename
        self._samples: list[dict[str, Any]] = []
        self._metadata_lookup: dict[str, Any] = {}
        for _, row in split_df.iterrows():
            pho_name = Path(row["PHO_TILE"]).name
            self._samples.append(
                {
                    "pho_name": pho_name,
                    "pho_path": self.data_dir / row["PHO_TILE"],
                    "dtm_path": self.data_dir / row["DTM_TILE"],
                }
            )
            self._metadata_lookup[pho_name] = row

        # ------------------------------------------------------------------
        # Load annotations and intersect with the metadata-derived sample list
        # ------------------------------------------------------------------
        with open(annotations_file, "r") as f:
            coco_data = json.load(f)

        self.categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

        split_pho_names: set[str] = {s["pho_name"] for s in self._samples}

        # Mapping from pho_name → COCO image_id for annotation lookup
        self._pho_name_to_img_id: dict[str, int] = {
            img["file_name"]: img["id"]
            for img in coco_data["images"]
            if img["file_name"] in split_pho_names
        }

        # Build image_id → list[annotation] mapping with diameter filtering
        self.img_id_to_anns: dict[int, list[dict]] = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            _bx, _by, bw, bh = ann["bbox"]
            if min(bw, bh) >= self.min_diameter_m:
                self.img_id_to_anns[img_id].append(ann)

        # ------------------------------------------------------------------
        # Optionally drop tiles that have no annotations after diameter filter
        # AND after nodata-overlap filtering (mirrors __getitem__ logic).
        # ------------------------------------------------------------------
        if drop_no_crater_images:
            kept = []
            for s in self._samples:
                img_id = self._pho_name_to_img_id.get(s["pho_name"])
                raw_anns = self.img_id_to_anns.get(img_id, [])
                if not raw_anns:
                    continue  # no COCO entry or all below diameter threshold

                # Load the actual files to build the nodata mask and test each bbox
                _nac = _load_nc_band(s["pho_path"]) if self.use_nac else None
                _dtm = _load_nc_band(s["dtm_path"]) if self.use_dtm else None
                ref = _nac if _nac is not None else _dtm
                _h, _w = ref.shape
                inv = np.zeros((_h, _w), dtype=bool)
                if _nac is not None:
                    inv |= np.isnan(_nac) | (_nac == self.NODATA_VALUE)
                if _dtm is not None:
                    inv |= np.isnan(_dtm) | (_dtm == self.NODATA_VALUE)

                surviving = 0
                for ann in raw_anns:
                    _bx, _by, bw, bh = ann["bbox"]
                    x1i = int(max(0, _bx))
                    y1i = int(max(0, _by))
                    x2i = int(min(_w, _bx + bw + 1))
                    y2i = int(min(_h, _by + bh + 1))
                    if not inv[y1i:y2i, x1i:x2i].any():
                        surviving += 1

                if surviving > 0:
                    kept.append(s)
            self._samples = kept

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return the sample at *index*.

        Returns:
            dict with keys ``"image"`` (C×H×W float32 tensor),
            plus bounding-box / label / mask keys.
        """
        sample_info = self._samples[index]
        pho_name = sample_info["pho_name"]
        img_id = self._pho_name_to_img_id.get(pho_name)

        # ------------------------------------------------------------------
        # Load NAC (PHO) and/or DTM — each shape (H, W).
        # ------------------------------------------------------------------
        nac_np: np.ndarray | None = None
        dtm_np: np.ndarray | None = None
        orig_h = orig_w = 0
        if self.use_nac:
            nac_np = _load_nc_band(sample_info["pho_path"])
            orig_h, orig_w = nac_np.shape
        if self.use_dtm:
            dtm_np = _load_nc_band(sample_info["dtm_path"])
            if nac_np is None:
                orig_h, orig_w = dtm_np.shape

        # ------------------------------------------------------------------
        # Invalid-pixel mask (NaN or -9999 nodata) — computed before any
        # resize so annotation filtering works in original pixel coords.
        # A pixel is invalid if it is NaN *or* equals NODATA_VALUE in any
        # loaded modality.  Shape: (H, W) bool, True = invalid.
        # ------------------------------------------------------------------
        invalid_mask = np.zeros((orig_h, orig_w), dtype=bool)
        if nac_np is not None:
            invalid_mask |= np.isnan(nac_np) | (nac_np == self.NODATA_VALUE)
        if dtm_np is not None:
            invalid_mask |= np.isnan(dtm_np) | (dtm_np == self.NODATA_VALUE)

        # ------------------------------------------------------------------
        # Impute: replace NaN and -9999 sentinel with 0
        # ------------------------------------------------------------------
        if nac_np is not None:
            nac_np = np.where(
                np.isnan(nac_np) | (nac_np == self.NODATA_VALUE), 0.0, nac_np
            )
        if dtm_np is not None:
            dtm_np = np.where(
                np.isnan(dtm_np) | (dtm_np == self.NODATA_VALUE), 0.0, dtm_np
            )

        # ------------------------------------------------------------------
        # Optional resize (PIL bilinear for whichever modalities are present)
        # ------------------------------------------------------------------
        if self.image_size is not None:
            if nac_np is not None:
                nac_img = Image.fromarray(nac_np, mode="F").resize(
                    (self.image_size, self.image_size), Image.Resampling.BILINEAR
                )
                nac_np = np.asarray(nac_img, dtype=np.float32)
            if dtm_np is not None:
                dtm_img = Image.fromarray(dtm_np, mode="F").resize(
                    (self.image_size, self.image_size), Image.Resampling.BILINEAR
                )
                dtm_np = np.asarray(dtm_img, dtype=np.float32)
            scale_x = self.image_size / orig_w
            scale_y = self.image_size / orig_h
        else:
            scale_x = 1.0
            scale_y = 1.0

        nac_t = (
            torch.from_numpy(nac_np.copy()).unsqueeze(0) * self.nac_scale_factor
            if nac_np is not None
            else None
        )
        dtm_t = (
            torch.from_numpy(dtm_np.copy()).unsqueeze(0) if dtm_np is not None else None
        )

        # ------------------------------------------------------------------
        # Sample-level normalisation — applied for all output modes.
        # ------------------------------------------------------------------
        if nac_t is not None:
            if (
                self.nac_valid_min is not None
                or self.nac_valid_max is not None
                or self.nac_mask_threshold is not None
            ):
                nac_t = _minmax_scale_tensor(
                    nac_t[0],
                    self.nac_valid_min,
                    self.nac_valid_max,
                    self.nac_mask_threshold,
                    self.nac_mask_fill_value,
                    self.nac_mask_eps,
                ).unsqueeze(0)
            if self.nac_norm_mean is not None:
                nac_t = (nac_t - self.nac_norm_mean) / self.nac_norm_std

        if dtm_t is not None:
            if self.dtm_mask_threshold is not None:
                dtm_t = dtm_t.clone()
                dtm_t[dtm_t <= self.dtm_mask_threshold] = self.dtm_mask_fill_value
            if self.dtm_global_std is not None:
                dtm_mean = dtm_t.mean(dim=(-2, -1), keepdim=True)
                dtm_t = (dtm_t - dtm_mean) / self.dtm_global_std

        # ------------------------------------------------------------------
        # Annotations — filter out any bbox that touches an invalid pixel
        # (checked in original image coordinates before scaling).
        # ------------------------------------------------------------------
        raw_anns = self.img_id_to_anns.get(img_id, []) if img_id is not None else []

        boxes: list[list[float]] = []
        labels: list[int] = []
        masks: list[torch.Tensor] = []

        for ann in raw_anns:
            x1, y1, w, h = ann["bbox"]
            # Check the bounding-box region (clamped to image bounds) for
            # invalid pixels in the original-resolution mask.
            x1i = int(max(0, x1))
            y1i = int(max(0, y1))
            x2i = int(min(orig_w, x1 + w + 1))
            y2i = int(min(orig_h, y1 + h + 1))
            if invalid_mask[y1i:y2i, x1i:x2i].any():
                continue  # drop annotation that overlaps invalid data
            boxes.append(
                [
                    x1 * scale_x,
                    y1 * scale_y,
                    (x1 + w) * scale_x,
                    (y1 + h) * scale_y,
                ]
            )
            # Remap all crater annotations to foreground class 1
            labels.append(1)
            masks.append(torch.zeros((0, 0), dtype=torch.uint8))

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            masks_list = masks
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            masks_list = [torch.zeros((0, 0), dtype=torch.uint8)]

        # ------------------------------------------------------------------
        # Segmentation mask: rasterise a filled circle per bbox in the
        # (possibly resized) image frame.  Circle centre = bbox centre,
        # diameter chosen per `circle_diameter_mode` from the scaled bbox
        # side lengths.  Overlapping craters merge via binary OR.
        # ------------------------------------------------------------------
        # Use whichever modality is present as the spatial reference (both
        # share H, W after resize).
        ref_t = nac_t if nac_t is not None else dtm_t
        assert ref_t is not None  # guaranteed by __init__ validation

        seg_mask: torch.Tensor | None = None
        if self.task_mode == "segmentation":
            h_mask = int(ref_t.shape[-2])
            w_mask = int(ref_t.shape[-1])
            seg = torch.zeros((h_mask, w_mask), dtype=torch.long)
            if boxes:
                yy, xx = torch.meshgrid(
                    torch.arange(h_mask, dtype=torch.float32),
                    torch.arange(w_mask, dtype=torch.float32),
                    indexing="ij",
                )
                for x1, y1, x2, y2 in boxes:
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    bw = x2 - x1
                    bh = y2 - y1
                    if self.circle_diameter_mode == "min":
                        d = min(bw, bh)
                    elif self.circle_diameter_mode == "max":
                        d = max(bw, bh)
                    else:
                        d = (bw + bh) / 2.0
                    r = d / 2.0
                    if r <= 0:
                        continue
                    seg[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 1
            seg_mask = seg

        # ------------------------------------------------------------------
        # Metadata token tensor (resolved up-front so packed mode can fold it
        # into the stacked image tensor).  _metadata_lookup is always populated
        # (keyed by pho_name from the parquet split rows).
        # ------------------------------------------------------------------
        meta_row = self._metadata_lookup.get(pho_name)
        metadata_t: torch.Tensor = (
            self._metadata_encoder.encode(meta_row)
            if meta_row is not None
            else torch.zeros(_METADATA_MAX_TOKENS, dtype=torch.long)
        )

        if self.output_mode == "separate":
            sample: dict[str, Any] = {
                "image": ref_t,
                self.boxes_output_tag: boxes_tensor,
                self.labels_output_tag: labels_tensor,
                self.masks_output_tag: masks_list,
            }
            if nac_t is not None:
                sample["nac"] = nac_t
            if dtm_t is not None:
                sample["dtm_3m"] = dtm_t
            sample["metadata"] = metadata_t
        elif self.output_mode == "packed":
            channels = [t for t in (nac_t, dtm_t) if t is not None]
            _, h, w = channels[0].shape
            if self.use_metadata:
                flat_len = h * w
                if flat_len < _PACKED_SEQ_LEN:
                    raise ValueError(
                        f"image spatial dim ({h}x{w}={flat_len}) is smaller than "
                        f"the reserved sequence length {_PACKED_SEQ_LEN}; "
                        f"increase image_size or lower _PACKED_SEQ_LEN."
                    )
                meta_channel = torch.full(
                    (flat_len,),
                    _PACKED_SEQ_PAD,
                    dtype=torch.float32,
                )
                meta_channel[: metadata_t.numel()] = metadata_t.to(torch.float32)
                channels.append(meta_channel.view(1, h, w))

            sample = {
                "image": torch.cat(channels, dim=0),
                self.boxes_output_tag: boxes_tensor,
                self.labels_output_tag: labels_tensor,
                self.masks_output_tag: masks_list,
            }
        else:
            image_channels = [t for t in (nac_t, dtm_t) if t is not None]
            sample = {
                "image": torch.cat(image_channels, dim=0),  # (C, H, W), C in {1, 2}
                self.boxes_output_tag: boxes_tensor,
                self.labels_output_tag: labels_tensor,
                self.masks_output_tag: masks_list,
                "metadata": metadata_t,
            }

        if self.task_mode == "segmentation" and seg_mask is not None:
            # Swap detection targets for a dense per-pixel mask.  Optionally
            # keep the per-crater bboxes under a distinct key so downstream
            # tasks (e.g. shape-loss regularisation) can reason per-crater.
            popped_boxes = sample.pop(self.boxes_output_tag, None)
            sample.pop(self.labels_output_tag, None)
            sample.pop(self.masks_output_tag, None)
            sample["mask"] = seg_mask
            if self.keep_boxes and popped_boxes is not None:
                sample["crater_boxes"] = popped_boxes

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def plot(
        self,
        sample: dict[str, torch.Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
        box_alpha: float = 0.7,
        confidence_score: float = 0.5,
    ) -> Figure:
        """Plot a sample with bounding boxes overlaid on NAC and DTM.

        Each ground-truth panel occupies one column; if predictions are
        present two extra columns are appended (NAC pred, DTM pred).

        Works for both output modes:

        * ``"stack"`` – reads channel 0 (NAC) and channel 1 (DTM) from
          ``sample["image"]``.
        * ``"separate"`` – reads ``sample["nac"]`` and ``sample["dtm_3m"]``.

        The DTM panel uses ``"terrain"`` colormap so elevation gradients are
        visible.  When DTM is not in the sample (e.g. the dataset was
        constructed without ``dtm_dir``), only the NAC panel is shown.

        Args:
            sample: A sample returned by :meth:`__getitem__`.
            show_titles: Whether to show column titles.
            suptitle: Optional figure suptitle.
            box_alpha: Alpha for bounding-box rectangle edges.
            confidence_score: Minimum score threshold for prediction boxes.

        Returns:
            A matplotlib :class:`~matplotlib.figure.Figure`.

        Example:
            >>> ds = LunarNACDTMDataset(...)
            >>> fig = ds.plot(ds[0], suptitle="Sample 0")
            >>> plt.show()
        """
        # ------------------------------------------------------------------
        # Extract NAC and DTM arrays for display
        # ------------------------------------------------------------------
        if "nac" in sample or "dtm_3m" in sample:
            has_nac = "nac" in sample
            has_dtm = "dtm_3m" in sample
            nac_arr = sample["nac"].squeeze(0).cpu().numpy() if has_nac else None
            dtm_arr = sample["dtm_3m"].squeeze(0).cpu().numpy() if has_dtm else None
        else:
            img = sample["image"].cpu().numpy()  # (C, H, W)
            has_nac = self.use_nac
            has_dtm = self.use_dtm
            nac_arr = img[0] if has_nac else None
            dtm_arr = img[1 if has_nac else 0] if has_dtm else None

        def _norm01(arr):
            lo, hi = arr.min(), arr.max()
            return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)

        # Normalise NAC / DTM to [0, 1] for display (each may be absent).
        nac_display = _norm01(nac_arr) if nac_arr is not None else None
        dtm_display = _norm01(dtm_arr) if dtm_arr is not None else None

        # ------------------------------------------------------------------
        # Segmentation branch: overlay mask directly, no bbox path.
        # ------------------------------------------------------------------
        if self.task_mode == "segmentation":
            gt_mask = sample.get("mask")
            pred_mask = sample.get("prediction")
            modalities: list[tuple[str, Any]] = []
            if nac_display is not None:
                modalities.append(("NAC", nac_display))
            if dtm_display is not None:
                modalities.append(("DTM", dtm_display))
            ncols = len(modalities) * (2 if pred_mask is not None else 1)
            fig, axs = plt.subplots(1, ncols, squeeze=False, figsize=(ncols * 6, 6))

            def _overlay(ax, base, mask_arr, title):
                ax.imshow(base, cmap="gray", vmin=0, vmax=1)
                if mask_arr is not None:
                    m = (
                        mask_arr.cpu().numpy()
                        if hasattr(mask_arr, "cpu")
                        else np.asarray(mask_arr)
                    )
                    ax.imshow(np.ma.masked_where(m == 0, m), cmap="autumn", alpha=0.5)
                ax.axis("off")
                if show_titles:
                    ax.set_title(title, fontsize=12, fontweight="bold")

            col = 0
            for name, display in modalities:
                _overlay(axs[0, col], display, gt_mask, f"{name} — GT mask")
                col += 1
            if pred_mask is not None:
                for name, display in modalities:
                    _overlay(axs[0, col], display, pred_mask, f"{name} — pred mask")
                    col += 1
            if suptitle is not None:
                fig.suptitle(suptitle)
            fig.tight_layout()
            return fig

        # ------------------------------------------------------------------
        # Ground-truth annotations
        # ------------------------------------------------------------------
        boxes = sample[self.boxes_output_tag].cpu().numpy()
        labels = sample[self.labels_output_tag].cpu().numpy()
        n_gt = len(boxes)

        # ------------------------------------------------------------------
        # Predictions (optional)
        # ------------------------------------------------------------------
        pred_key = f"prediction_{self.labels_output_tag}"
        show_predictions = pred_key in sample
        if show_predictions:
            pred_labels = sample[pred_key].cpu().numpy()
            pred_scores = sample[f"prediction_{self.scores_output_tag}"].cpu().numpy()
            pred_boxes = (
                sample[f"prediction_{self.boxes_output_tag}"].cpu().numpy()
                if f"prediction_{self.boxes_output_tag}" in sample
                else None
            )
            n_pred = len(pred_labels)

        # ------------------------------------------------------------------
        # Figure layout: one column per available modality for GT, then again
        # for predictions when present.
        # ------------------------------------------------------------------
        det_modalities: list[tuple[str, Any, str]] = []
        if nac_display is not None:
            det_modalities.append(("NAC", nac_display, "gray"))
        if dtm_display is not None:
            det_modalities.append(("DTM", dtm_display, "terrain"))
        gt_cols = len(det_modalities)
        pred_cols = gt_cols if show_predictions else 0
        ncols = gt_cols + pred_cols

        fig, axs = plt.subplots(1, ncols, squeeze=False, figsize=(ncols * 6, 6))

        cm = plt.get_cmap("gist_rainbow")
        num_classes = max(len(self.categories), 2)

        def _draw_boxes(ax, box_array, lbl_array, score_array=None):
            for i in range(len(box_array)):
                if score_array is not None and score_array[i] < confidence_score:
                    continue
                color = cm(int(lbl_array[i]) / num_classes)
                x1, y1, x2, y2 = box_array[i]
                ax.add_patch(
                    patches.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=1.5,
                        alpha=box_alpha,
                        linestyle="dashed",
                        edgecolor=color,
                        facecolor="none",
                    )
                )
                caption = self.categories.get(int(lbl_array[i]), f"cls {lbl_array[i]}")
                if score_array is not None:
                    caption += f" {score_array[i]:.2f}"
                # ax.text(x1, y1 - 4, caption,
                #         color="white", size=9, backgroundcolor="black", alpha=0.75)

        # --- Ground-truth columns ---
        for i, (name, display, cmap) in enumerate(det_modalities):
            axs[0, i].imshow(display, cmap=cmap, vmin=0, vmax=1)
            axs[0, i].axis("off")
            _draw_boxes(axs[0, i], boxes, labels)
            if show_titles:
                axs[0, i].set_title(
                    f"{name} — Ground Truth", fontsize=12, fontweight="bold"
                )

        # --- Prediction columns (mirroring the GT layout) ---
        if show_predictions:
            for i, (name, display, cmap) in enumerate(det_modalities):
                col = gt_cols + i
                axs[0, col].imshow(display, cmap=cmap, vmin=0, vmax=1)
                axs[0, col].axis("off")
                if pred_boxes is not None:
                    _draw_boxes(axs[0, col], pred_boxes, pred_labels, pred_scores)
                if show_titles:
                    axs[0, col].set_title(
                        f"{name} — Prediction", fontsize=12, fontweight="bold"
                    )

        if suptitle is not None:
            plt.suptitle(suptitle, fontsize=14, fontweight="bold")

        plt.tight_layout()
        return fig


class LunarNACDTMDataModule(LightningDataModule):
    """TerraTorch-compatible data module for NAC and/or DTM crater detection.

    Image paths and splits are driven entirely by ``metadata_file`` (a parquet
    with ``PHO_TILE``, ``DTM_TILE``, and ``dataset`` columns).  ``data_dir`` is
    the root that those relative paths are joined against.

    Which modalities to load is controlled by ``modalities`` (default
    ``["nac", "dtm"]``).  Pass ``["nac"]`` or ``["dtm"]`` for single-modality.

    Preprocessing pipeline (applied in order at batch level):

    1. NAC: optional per-image min-max scaling to ``[eps, 1]`` (enabled when
       *nac_valid_min* or *nac_valid_max* is set, or *nac_mask_threshold* is
       set). Omit all three to skip this step entirely.
    2. NAC: optional global normalisation ``(nac - mean) / std`` when
       *nac_norm_mean* / *nac_norm_std* are provided.
    3. DTM: optional sentinel fill, then local normalisation
       ``(dtm - image_mean) / dtm_global_std`` when *dtm_global_std* is set.

    **Output modes** (controlled by ``output_mode``):

    * ``"stack"`` *(default)* – single ``"image"`` tensor ``(B, C, H, W)``.
    * ``"separate"`` – separate ``"nac"`` and ``"dtm_3m"`` tensors
      ``(B, 1, H, W)`` each, matching the key convention expected by
      ``LunarBackbone`` in multi-modal mode.
    """

    def __init__(
        self,
        data_dir: str,
        metadata_file: str,
        annotations_file: str,
        modalities: list[str] | None = None,
        used_for_labeling_only: bool = False,
        drop_no_crater_images: bool = False,
        batch_size: int = 4,
        num_workers: int = 0,
        image_size: int = 256,
        output_mode: str = "stack",
        task_mode: str = "detection",
        circle_diameter_mode: str = "min",
        keep_boxes: bool = False,
        collate_fn: Callable | None = None,
        boxes_output_tag: str = "boxes",
        labels_output_tag: str = "labels",
        masks_output_tag: str = "masks",
        scores_output_tag: str = "scores",
        train_transforms: Callable | None = None,
        val_transforms: Callable | None = None,
        test_transforms: Callable | None = None,
        # NAC preprocessing
        nac_scale_factor: float = 1.0,
        nac_valid_min: float | None = None,
        nac_valid_max: float | None = None,
        nac_mask_threshold: float | None = None,
        nac_mask_fill_value: float = 0.0,
        nac_mask_eps: float = 1e-6,
        nac_norm_mean: float | None = None,
        nac_norm_std: float | None = None,
        # DTM preprocessing
        dtm_global_std: float | None = None,
        dtm_mask_threshold: float | None = None,
        dtm_mask_fill_value: float = 0.0,
        # Crater diameter filter
        min_diameter_m: float = 5.0,
        # Metadata tokenizer
        metadata_tokenizer_path: str = MetadataEncoder._DEFAULT_TOKENIZER,
        metadata_binning_config_path: str = MetadataEncoder._DEFAULT_BINNING,
    ):
        """Initialize the data module.

        Args:
            data_dir: Root directory that ``PHO_TILE`` / ``DTM_TILE`` paths
                      from the parquet are resolved against.
            metadata_file: Path to ``metadata.parquet``.  Used as the sole
                           source of splits and image paths.
            annotations_file: Path to the COCO-format annotations JSON.
            modalities: List of modalities to load (``"nac"`` and/or
                        ``"dtm"``).  Defaults to ``["nac", "dtm"]``.
            used_for_labeling_only: Restrict to rows where
                                    ``used_for_labeling == "yes"``.
            drop_no_crater_images: When ``True``, remove from each split any
                                   sample that has no surviving annotations
                                   after both the ``min_diameter_m`` filter
                                   and the nodata-overlap check.  Tiles with
                                   no COCO entry are also dropped.  Reads
                                   every tile once at init time.
            batch_size: Samples per batch.
            num_workers: DataLoader worker processes.
            image_size: Resize images to this square size.
            output_mode: ``"stack"``, ``"separate"``, or ``"packed"``.
                         See class docstring.
            collate_fn: Optional custom collate function (overrides default).
            boxes_output_tag: Output key for bounding boxes.
            labels_output_tag: Output key for labels.
            masks_output_tag: Output key for masks.
            scores_output_tag: Output key for scores.
            train_transforms: Per-sample transforms for training split.
            val_transforms: Per-sample transforms for validation split.
            test_transforms: Per-sample transforms for test split.
            nac_scale_factor: Scalar applied to raw NAC DN values at load time.
                              Default ``1/255`` brings raw DN (1–253) into
                              ``[0, 1]``.  Set to ``1.0`` if already in
                              ``[0, 1]``.
            nac_valid_min: Lower clip value for NAC min-max scaling.
            nac_valid_max: Upper clip value for NAC min-max scaling.
            nac_mask_threshold: Pixels <= this value in the NAC channel are
                                treated as invalid.  ``None`` disables masking.
            nac_mask_fill_value: Replacement value for invalid NAC pixels.
            nac_mask_eps: Epsilon added to the scaled NAC range.
            nac_norm_mean: Global mean for NAC normalisation (with
                           *nac_norm_std*).  Both must be set or neither.
            nac_norm_std: Global std for NAC normalisation.
            dtm_global_std: Global std for DTM local normalisation.
                            ``None`` leaves DTM unscaled.
            dtm_mask_threshold: Pixels <= this value in the DTM channel are
                                treated as invalid.  ``None`` disables masking.
            dtm_mask_fill_value: Replacement value for invalid DTM pixels.
            min_diameter_m: Minimum crater bounding-box side length in pixels.
                             Default 5.  Set to 0 to disable.
            metadata_tokenizer_path: Path to the HuggingFace ``*.json``
                                     tokenizer file.
            metadata_binning_config_path: Path to the binning CSV.
        """
        if output_mode not in _NACDTM_OUTPUT_MODES:
            raise ValueError(
                f"output_mode must be one of {_NACDTM_OUTPUT_MODES}, got '{output_mode}'"
            )
        if task_mode not in {"detection", "segmentation"}:
            raise ValueError(
                f"task_mode must be 'detection' or 'segmentation', got '{task_mode}'"
            )
        if (nac_norm_mean is None) != (nac_norm_std is None):
            raise ValueError(
                "nac_norm_mean and nac_norm_std must both be set or both be None."
            )
        super().__init__()

        _modalities = modalities if modalities is not None else ["nac", "dtm"]

        self.task_mode = task_mode
        self.circle_diameter_mode = circle_diameter_mode
        self.keep_boxes = keep_boxes
        self.data_dir = data_dir
        self.use_nac = "nac" in _modalities
        self.use_dtm = "dtm" in _modalities
        self.modalities = _modalities
        self.used_for_labeling_only = used_for_labeling_only
        self.drop_no_crater_images = drop_no_crater_images
        self.annotations_file = annotations_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.output_mode = output_mode
        self.boxes_output_tag = boxes_output_tag
        self.labels_output_tag = labels_output_tag
        self.masks_output_tag = masks_output_tag
        self.scores_output_tag = scores_output_tag
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.min_diameter_m = min_diameter_m
        self.nac_scale_factor = nac_scale_factor
        self.metadata_file = metadata_file
        self.metadata_tokenizer_path = metadata_tokenizer_path
        self.metadata_binning_config_path = metadata_binning_config_path
        # Normalisation params forwarded to the dataset (normalisation happens
        # at sample level inside the dataset for all output modes).
        self._nac_valid_min = nac_valid_min
        self._nac_valid_max = nac_valid_max
        self._nac_mask_threshold = nac_mask_threshold
        self._nac_mask_fill_value = nac_mask_fill_value
        self._nac_mask_eps = nac_mask_eps
        self._nac_norm_mean = nac_norm_mean
        self._nac_norm_std = nac_norm_std
        self._dtm_global_std = dtm_global_std
        self._dtm_mask_threshold = dtm_mask_threshold
        self._dtm_mask_fill_value = dtm_mask_fill_value

        # ------------------------------------------------------------------
        # Collate function — metadata is always present now
        # ------------------------------------------------------------------
        _collate_meta = output_mode != "packed"
        if collate_fn is None:
            if task_mode == "segmentation":
                self.collate_fn = lambda batch: collate_fn_segmentation(
                    batch,
                    output_mode=output_mode,
                    include_metadata=_collate_meta,
                )
            elif output_mode == "separate":
                self.collate_fn = lambda batch: collate_fn_detection_multimodal(
                    batch,
                    boxes_tag=boxes_output_tag,
                    labels_tag=labels_output_tag,
                    masks_tag=masks_output_tag,
                    include_metadata=_collate_meta,
                )
            elif output_mode == "packed":
                self.collate_fn = lambda batch: collate_fn_detection_stacked(
                    batch,
                    boxes_tag=boxes_output_tag,
                    labels_tag=labels_output_tag,
                    masks_tag=masks_output_tag,
                )
            else:
                self.collate_fn = lambda batch: collate_fn_detection(
                    batch,
                    boxes_tag=boxes_output_tag,
                    labels_tag=labels_output_tag,
                    masks_tag=masks_output_tag,
                    include_metadata=_collate_meta,
                )
        else:
            self.collate_fn = collate_fn

        self.train_dataset: LunarNACDTMDataset | None = None
        self.val_dataset: LunarNACDTMDataset | None = None
        self.test_dataset: LunarNACDTMDataset | None = None

    def _make_dataset(
        self, split: str, transforms: Callable | None
    ) -> LunarNACDTMDataset:
        return LunarNACDTMDataset(
            data_dir=self.data_dir,
            metadata_file=self.metadata_file,
            annotations_file=self.annotations_file,
            modalities=self.modalities,
            used_for_labeling_only=self.used_for_labeling_only,
            drop_no_crater_images=self.drop_no_crater_images,
            split=split,
            transforms=transforms,
            boxes_output_tag=self.boxes_output_tag,
            labels_output_tag=self.labels_output_tag,
            masks_output_tag=self.masks_output_tag,
            scores_output_tag=self.scores_output_tag,
            image_size=self.image_size,
            min_diameter_m=self.min_diameter_m,
            output_mode=self.output_mode,
            task_mode=self.task_mode,
            circle_diameter_mode=self.circle_diameter_mode,
            keep_boxes=self.keep_boxes,
            nac_scale_factor=self.nac_scale_factor,
            metadata_tokenizer_path=self.metadata_tokenizer_path,
            metadata_binning_config_path=self.metadata_binning_config_path,
            nac_valid_min=self._nac_valid_min,
            nac_valid_max=self._nac_valid_max,
            nac_mask_threshold=self._nac_mask_threshold,
            nac_mask_fill_value=self._nac_mask_fill_value,
            nac_mask_eps=self._nac_mask_eps,
            nac_norm_mean=self._nac_norm_mean,
            nac_norm_std=self._nac_norm_std,
            dtm_global_std=self._dtm_global_std,
            dtm_mask_threshold=self._dtm_mask_threshold,
            dtm_mask_fill_value=self._dtm_mask_fill_value,
        )

    def setup(self, stage: str) -> None:
        """Create datasets for the requested stage.

        Args:
            stage: One of ``"fit"``, ``"validate"``, or ``"test"``.
        """
        if stage == "fit":
            self.train_dataset = self._make_dataset("train", self.train_transforms)
            self.val_dataset = self._make_dataset("val", self.val_transforms)
        elif stage == "validate":
            self.val_dataset = self._make_dataset("val", self.val_transforms)
        elif stage == "test":
            self.test_dataset = self._make_dataset("test", self.test_transforms)

    def _dataloader_factory(self, split: str) -> DataLoader:
        dataset_attr = f"{split}_dataset"
        if getattr(self, dataset_attr, None) is None:
            stage = "fit" if split == "train" else split
            self.setup(stage)
        dataset = getattr(self, dataset_attr)
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=(split == "train"),
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return self._dataloader_factory("train")

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return self._dataloader_factory("val")

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return self._dataloader_factory("test")

    def plot(self, sample, **kwargs):
        # TerraTorch's ObjectDetectionTask gates val image logging on
        # hasattr(datamodule, 'plot'); delegate to the dataset's plot.
        dataset = self.val_dataset or self.train_dataset or self.test_dataset
        if dataset is None:
            raise RuntimeError("No dataset available to plot; call setup() first.")
        return dataset.plot(sample, **kwargs)


# ---------------------------------------------------------------------------
# Private batch transforms for NAC+DTM preprocessing
# ---------------------------------------------------------------------------


def _minmax_scale_tensor(
    nac: torch.Tensor, valid_min, valid_max, mask_threshold, mask_fill_value, mask_eps
) -> torch.Tensor:
    """Apply per-image min-max scaling to a (H, W) NAC tensor."""
    if mask_threshold is not None:
        valid_mask = nac > mask_threshold
        if valid_mask.any():
            valid = nac[valid_mask]
            vmin = valid_min if valid_min is not None else float(valid.min())
            vmax = valid_max if valid_max is not None else float(valid.max())
            if vmax > vmin:
                nac = nac.clone()
                nac[valid_mask] = (valid - vmin) / (vmax - vmin) * (
                    1.0 - mask_eps
                ) + mask_eps
            else:
                nac = nac.clone()
                nac[valid_mask] = 1.0
        nac = nac.clone()
        nac[~valid_mask] = mask_fill_value
    else:
        vmin = valid_min if valid_min is not None else float(nac.min())
        vmax = valid_max if valid_max is not None else float(nac.max())
        if vmax > vmin:
            nac = (nac - vmin) / (vmax - vmin) * (1.0 - mask_eps) + mask_eps
        else:
            nac = torch.ones_like(nac)
    return nac


class _NACMinMaxScale:
    """Min-max scale the NAC channel, operating per-image.

    Handles both output modes:

    * ``"stack"``    – reads/writes ``batch["image"][:, 0, :, :]``
    * ``"separate"`` – reads/writes ``batch["nac"]``  (shape ``B, 1, H, W``)
    """

    def __init__(
        self,
        valid_min: float | None = None,
        valid_max: float | None = None,
        mask_threshold: float | None = None,
        mask_fill_value: float = 0.0,
        mask_eps: float = 1e-6,
        output_mode: str = "stack",
        channel: int | None = 0,
    ):
        self.valid_min = valid_min
        self.valid_max = valid_max
        self.mask_threshold = mask_threshold
        self.mask_fill_value = mask_fill_value
        self.mask_eps = mask_eps
        self.output_mode = output_mode
        self.channel = channel if channel is not None else 0

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        if self.output_mode == "separate":
            nac_batch = batch["nac"]  # (B, 1, H, W)
            processed = [
                _minmax_scale_tensor(
                    img[0],
                    self.valid_min,
                    self.valid_max,
                    self.mask_threshold,
                    self.mask_fill_value,
                    self.mask_eps,
                ).unsqueeze(0)
                for img in nac_batch
            ]
            batch["nac"] = torch.stack(processed)
        else:
            image = batch["image"]  # (B, C, H, W)
            c = self.channel
            processed_imgs = []
            for img in image:
                nac = _minmax_scale_tensor(
                    img[c],
                    self.valid_min,
                    self.valid_max,
                    self.mask_threshold,
                    self.mask_fill_value,
                    self.mask_eps,
                )
                img = img.clone()
                img[c] = nac
                processed_imgs.append(img)
            batch["image"] = torch.stack(processed_imgs)
        return batch


class _DTMMaskFill:
    """Fill invalid pixels in the DTM channel with a fixed value.

    Handles both output modes:

    * ``"stack"``    – operates on ``batch["image"][:, 1, :, :]``
    * ``"separate"`` – operates on ``batch["dtm_3m"]``
    """

    def __init__(
        self,
        mask_threshold: float = -1e30,
        mask_fill_value: float = 0.0,
        output_mode: str = "stack",
        channel: int | None = 1,
    ):
        self.mask_threshold = mask_threshold
        self.mask_fill_value = mask_fill_value
        self.output_mode = output_mode
        self.channel = channel if channel is not None else 1

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        if self.output_mode == "separate":
            dtm = batch["dtm_3m"]  # (B, 1, H, W)
            dtm[dtm <= self.mask_threshold] = self.mask_fill_value
            batch["dtm_3m"] = dtm
        else:
            image = batch["image"]  # (B, C, H, W)
            c = self.channel
            image[:, c, :, :][
                image[:, c, :, :] <= self.mask_threshold
            ] = self.mask_fill_value
            batch["image"] = image
        return batch


class _NACChannelNormalize:
    """Global mean/std normalisation for the NAC channel, applied after min-max scaling.

    Formula: ``nac = (nac - mean) / std``

    Handles both output modes:

    * ``"stack"``    – operates on ``batch["image"][:, 0:1, :, :]``
    * ``"separate"`` – operates on ``batch["nac"]``  (shape ``B, 1, H, W``)
    """

    def __init__(
        self,
        mean: float,
        std: float,
        output_mode: str = "stack",
        channel: int | None = 0,
    ):
        self.mean = mean
        self.std = std
        self.output_mode = output_mode
        self.channel = channel if channel is not None else 0

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        if self.output_mode == "separate":
            batch["nac"] = (batch["nac"] - self.mean) / self.std
        else:
            image = batch["image"]  # (B, C, H, W)
            c = self.channel
            image[:, c : c + 1, :, :] = (
                image[:, c : c + 1, :, :] - self.mean
            ) / self.std
            batch["image"] = image
        return batch


def collate_fn_detection_stacked(
    batch: list[dict[str, Any]],
    boxes_tag: str = "boxes",
    labels_tag: str = "labels",
    masks_tag: str = "masks",
) -> dict[str, Any]:
    """Collate function for the ``"packed"`` output mode.

    Each sample's ``"image"`` is already a normalised ``(C, H, W)`` tensor,
    so we stack into ``(B, C, H, W)`` rather than leaving it as a list.

    Args:
        batch: List of per-sample dicts from :class:`LunarNACDTMDataset`.
        boxes_tag: Key for bounding boxes.
        labels_tag: Key for labels.
        masks_tag: Key for masks.

    Returns:
        Collated batch with ``"image"`` ``(B, C, H, W)`` and detection
        target lists.
    """
    return {
        "image": torch.stack([item["image"] for item in batch]),
        boxes_tag: [item[boxes_tag] for item in batch],
        labels_tag: [item[labels_tag] for item in batch],
        masks_tag: [item[masks_tag] for item in batch],
    }


def collate_fn_detection_multimodal(
    batch: list[dict[str, Any]],
    boxes_tag: str = "boxes",
    labels_tag: str = "labels",
    masks_tag: str = "masks",
    include_metadata: bool = False,
) -> dict[str, Any]:
    """Collate function for the ``"separate"`` output mode.

    Stacks ``"nac"`` and ``"dtm_3m"`` independently and collects detection
    targets the same way as :func:`collate_fn_detection`.

    Args:
        batch: List of per-sample dicts from :class:`LunarNACDTMDataset`.
        boxes_tag: Key for bounding boxes.
        labels_tag: Key for labels.
        masks_tag: Key for masks.
        include_metadata: If ``True``, stack the ``"metadata"`` tensors into
                          a ``(B, _METADATA_MAX_TOKENS)`` LongTensor.

    Returns:
        Collated batch with ``"nac"`` ``(B, 1, H, W)``, ``"dtm_3m"``
        ``(B, 1, H, W)``, detection target lists, and optionally
        ``"metadata"`` ``(B, _METADATA_MAX_TOKENS)``.
    """
    result = {
        "image": torch.stack([item["image"] for item in batch]),
        boxes_tag: [item[boxes_tag] for item in batch],
        labels_tag: [item[labels_tag] for item in batch],
        masks_tag: [item[masks_tag] for item in batch],
    }
    if "nac" in batch[0]:
        result["nac"] = torch.stack([item["nac"] for item in batch])
    if "dtm_3m" in batch[0]:
        result["dtm_3m"] = torch.stack([item["dtm_3m"] for item in batch])
    if include_metadata:
        result["metadata"] = torch.stack([item["metadata"] for item in batch])
    return result


def collate_fn_segmentation(
    batch: list[dict[str, Any]],
    output_mode: str = "stack",
    include_metadata: bool = False,
) -> dict[str, Any]:
    """Collate function for semantic-segmentation mode.

    Stacks ``"image"`` and ``"mask"`` into dense batch tensors.  For the
    ``"separate"`` output mode also stacks the per-modality tensors.

    Args:
        batch: List of per-sample dicts from :class:`LunarNACDTMDataset` with
               ``task_mode="segmentation"``.
        output_mode: Matches the dataset's ``output_mode``.
        include_metadata: If ``True``, stack the ``"metadata"`` tensors.

    Returns:
        Collated batch with ``"image"`` ``(B, C, H, W)`` and ``"mask"``
        ``(B, H, W)`` LongTensor.
    """
    result: dict[str, Any] = {
        "image": torch.stack([item["image"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
    }
    if output_mode == "separate":
        if "nac" in batch[0]:
            result["nac"] = torch.stack([item["nac"] for item in batch])
        if "dtm_3m" in batch[0]:
            result["dtm_3m"] = torch.stack([item["dtm_3m"] for item in batch])
    if include_metadata:
        result["metadata"] = torch.stack([item["metadata"] for item in batch])
    if "crater_boxes" in batch[0]:
        # Variable-length per sample — keep as a list to avoid ragged stacks.
        result["crater_boxes"] = [item["crater_boxes"] for item in batch]
    return result
