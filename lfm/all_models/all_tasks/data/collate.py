"""Collate functions for lunar segmentation dataloaders."""

from __future__ import annotations

import torch


def collate_semantic_segmentation(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "filename": [item["filename"] for item in batch],
    }


def collate_instance_segmentation(batch: list[dict]) -> dict:
    result = collate_semantic_segmentation(batch)
    if "crater_boxes" in batch[0]:
        result["crater_boxes"] = [item["crater_boxes"] for item in batch]
    if "num_craters" in batch[0]:
        result["num_craters"] = torch.stack([item["num_craters"] for item in batch])
    return result


def collate_mask2former_instance_segmentation(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "mask_labels": [item["mask_labels"] for item in batch],
        "class_labels": [item["class_labels"] for item in batch],
        "instance_mask": torch.stack([item["instance_mask"] for item in batch]),
        "filename": [item["filename"] for item in batch],
        "original_size": [item["original_size"] for item in batch],
    }


def collate_object_detection_instance_segmentation(batch: list[dict]) -> dict:
    result = {
        "image": torch.stack([item["image"] for item in batch]),
        "boxes": [item["boxes"] for item in batch],
        "labels": [item["labels"] for item in batch],
        "masks": [item["masks"] for item in batch],
        "filename": [item["filename"] for item in batch],
    }
    if "mask" in batch[0]:
        result["mask"] = torch.stack([item["mask"] for item in batch])
    if "crater_boxes" in batch[0]:
        result["crater_boxes"] = [item["crater_boxes"] for item in batch]
    if "num_craters" in batch[0]:
        result["num_craters"] = torch.stack([item["num_craters"] for item in batch])
    return result
