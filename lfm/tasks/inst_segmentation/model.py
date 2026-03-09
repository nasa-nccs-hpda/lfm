"""
model.py
DINO encoder with UNet decoder for instance segmentation.
Uses embedding-based approach with discriminative loss for instance separation.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InstanceUNetDecoder(nn.Module):
    """
    UNet-style decoder for instance segmentation.

    Outputs two branches:
    - Semantic segmentation (2 classes: background, crater)
    - Instance embeddings (N-dimensional vectors for instance discrimination)
    """

    def __init__(self, in_channels, embedding_dim=32, num_classes=2):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # Progressive upsampling (shared backbone)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Semantic segmentation head
        self.semantic_head = nn.Conv2d(64, num_classes, kernel_size=1)

        # Instance embedding head
        # Embeddings will be L2-normalized, pixels from same instance should be close
        self.embedding_head = nn.Sequential(
            nn.Conv2d(64, embedding_dim, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: Patch embeddings (batch, num_patches, embed_dim)

        Returns:
            dict with:
                - semantic: (B, num_classes, H, W)
                - embeddings: (B, embedding_dim, H, W)
        """
        batch_size, num_patches, embed_dim = x.shape

        # Calculate spatial dimensions
        patch_h = patch_w = int(num_patches**0.5)

        # Reshape to spatial: (batch, embed_dim, patch_h, patch_w)
        x = x.transpose(1, 2).reshape(batch_size, embed_dim, patch_h, patch_w)

        # Progressive upsampling
        x = self.up1(x)  # 2x
        x = self.up2(x)  # 4x
        x = self.up3(x)  # 8x
        x = self.up4(x)  # 16x

        # Generate outputs
        semantic_logits = self.semantic_head(x)  # (B, num_classes, H, W)
        embeddings = self.embedding_head(x)      # (B, embedding_dim, H, W)

        # L2 normalize embeddings for discriminative loss
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return {
            'semantic': semantic_logits,
            'embeddings': embeddings
        }


class DINOInstanceSegmentation(nn.Module):
    """
    DINO encoder with UNet decoder for instance segmentation.

    Uses embedding-based instance discrimination:
    - Semantic branch identifies crater pixels
    - Embedding branch produces feature vectors for instance separation
    - At inference, clustering groups pixels into instances
    """

    def __init__(self, encoder, embedding_dim=32, num_classes=2, img_size=(304, 304)):
        super().__init__()
        self.encoder = encoder
        self.img_size = img_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # Get embedding dimension from encoder
        self.embed_dim = encoder.embed_dim  # 1024 for vitl16

        # Instance segmentation decoder
        self.decoder = InstanceUNetDecoder(
            self.embed_dim,
            embedding_dim=embedding_dim,
            num_classes=num_classes
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input images (B, C, H, W)

        Returns:
            dict with:
                - semantic: Semantic segmentation logits (B, num_classes, H, W)
                - embeddings: Instance embeddings (B, embedding_dim, H, W)
        """
        # Get patch embeddings from encoder
        features_dict = self.encoder.forward_features(x)
        patch_embeddings = features_dict['x_norm_patchtokens']

        # Decode to segmentation outputs
        outputs = self.decoder(patch_embeddings)

        semantic_logits = outputs['semantic']
        embeddings = outputs['embeddings']

        # Interpolate to exact target size if needed
        if semantic_logits.shape[2:] != self.img_size:
            semantic_logits = F.interpolate(
                semantic_logits,
                size=self.img_size,
                mode='bilinear',
                align_corners=False,
            )
            embeddings = F.interpolate(
                embeddings,
                size=self.img_size,
                mode='bilinear',
                align_corners=False,
            )

        return {
            'semantic': semantic_logits,
            'embeddings': embeddings
        }

    def predict_instances(self, x, semantic_threshold=0.5, min_pixels=10,
                         distance_threshold=0.5, use_morphology=True):
        """
        Full inference pipeline for instance segmentation.

        Args:
            x: Input images (B, C, H, W)
            semantic_threshold: Threshold for crater classification
            min_pixels: Minimum pixels per instance
            distance_threshold: Embedding distance threshold for clustering
            use_morphology: Apply morphological operations to clean masks

        Returns:
            List of instance masks, one per image (each is H x W numpy array)
            where each instance has a unique integer ID (0 = background)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            semantic_logits = outputs['semantic']
            embeddings = outputs['embeddings']

            # Get semantic predictions
            semantic_probs = torch.softmax(semantic_logits, dim=1)
            crater_masks = (semantic_probs[:, 1] > semantic_threshold).cpu().numpy()

            # Get embeddings
            embeddings_np = embeddings.cpu().numpy()

            # Process each image in batch
            batch_size = x.shape[0]
            instance_masks = []

            for i in range(batch_size):
                instance_mask = self._cluster_embeddings(
                    embeddings_np[i],
                    crater_masks[i],
                    min_pixels,
                    distance_threshold,
                    use_morphology
                )
                instance_masks.append(instance_mask)

            return instance_masks

    def _cluster_embeddings(self, embeddings, crater_mask, min_pixels,
                           distance_threshold, use_morphology):
        """
        Cluster embeddings to separate instances.

        Args:
            embeddings: (embedding_dim, H, W)
            crater_mask: Binary mask (H, W)
            min_pixels: Minimum pixels per instance
            distance_threshold: Distance threshold for merging
            use_morphology: Clean up with morphological operations

        Returns:
            instance_mask: (H, W) with unique instance IDs
        """
        from scipy.ndimage import label as connected_components
        from scipy.spatial.distance import cdist
        from skimage.morphology import remove_small_objects

        H, W = crater_mask.shape
        instance_mask = np.zeros((H, W), dtype=np.int32)

        # No craters detected
        if crater_mask.sum() == 0:
            return instance_mask

        # Apply morphological cleaning if requested
        if use_morphology:
            crater_mask = remove_small_objects(crater_mask, min_size=min_pixels)

        # Get connected components as initial instances
        labeled_mask, num_components = connected_components(crater_mask)

        if num_components == 0:
            return instance_mask

        # Compute mean embedding for each component
        component_embeddings = []
        component_ids = []

        for comp_id in range(1, num_components + 1):
            comp_pixels = labeled_mask == comp_id
            if comp_pixels.sum() < min_pixels:
                continue

            # Mean embedding for this component
            comp_embedding = embeddings[:, comp_pixels].mean(axis=1)
            component_embeddings.append(comp_embedding)
            component_ids.append(comp_id)

        if len(component_embeddings) == 0:
            return instance_mask

        component_embeddings = np.array(component_embeddings)  # (N, embedding_dim)

        # Merge components with similar embeddings
        distances = cdist(component_embeddings, component_embeddings, metric='euclidean')

        # Build merge groups
        merged_groups = {}
        for i, comp_id in enumerate(component_ids):
            merged_groups[comp_id] = comp_id

        for i in range(len(component_ids)):
            for j in range(i + 1, len(component_ids)):
                if distances[i, j] < distance_threshold:
                    # Merge component j into component i's group
                    root_i = merged_groups[component_ids[i]]
                    merged_groups[component_ids[j]] = root_i

        # Create final instance mask
        instance_id = 1
        assigned_roots = {}

        for comp_id in component_ids:
            root = merged_groups[comp_id]

            # Assign new instance ID for this root
            if root not in assigned_roots:
                assigned_roots[root] = instance_id
                instance_id += 1

            # Set pixels in instance mask
            comp_pixels = labeled_mask == comp_id
            instance_mask[comp_pixels] = assigned_roots[root]

        return instance_mask

    def save_parameters(self, filename):
        """Save model state (encoder + decoder)."""
        torch.save(self.state_dict(), filename)

    def load_parameters(self, filename):
        """Load model state (encoder + decoder)."""
        self.load_state_dict(torch.load(filename))


def load_dinov3_encoder(weights_local_checkpoint, device, model='dinov3_vitl16'):
    """Load DINOv3 encoder from local checkpoint or torch hub."""
    from urllib.error import HTTPError

    if os.path.exists(weights_local_checkpoint):
        print(f'Loading model from {weights_local_checkpoint}')
        encoder = torch.hub.load(
            repo_or_dir='facebookresearch/dinov3',
            model=model,
            source='github',
            weights=weights_local_checkpoint
        ).to(device)
        print("Encoder loaded with pretrained weights.")
        return encoder
    else:
        raise FileNotFoundError(
            f"Checkpoint not found at {weights_local_checkpoint}. "
            "Please download the DINOv3 checkpoint or provide a valid weights_URL."
        )