"""Custom prediction writer for object detection tasks.

This module provides a custom PredictionWriter callback that handles
object detection predictions (lists of dicts with boxes, labels, scores).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from lightning.pytorch.callbacks import BasePredictionWriter


class ObjectDetectionPredictionWriter(BasePredictionWriter):
    """Custom prediction writer for object detection tasks.
    
    Handles predictions that are lists of dictionaries containing
    'boxes', 'labels', and 'scores' for each image in the batch.
    
    This replaces terratorch's default prediction writer which doesn't
    handle list outputs from object detection models.
    """
    
    def __init__(
        self,
        output_dir: str = "predictions",
        boxes_key: str = "boxes",
        labels_key: str = "labels",
        scores_key: str = "scores",
        save_coco_format: bool = True
    ):
        """Initialize prediction writer.
        
        Args:
            output_dir: Directory to save predictions
            boxes_key: Key for bounding boxes in predictions
            labels_key: Key for labels in predictions
            scores_key: Key for scores in predictions
            save_coco_format: Whether to save aggregated COCO format file
        """
        super().__init__(write_interval="batch")
        # Ensure output_dir is a clean path without subdirectories
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.boxes_key = boxes_key
        self.labels_key = labels_key
        self.scores_key = scores_key
        self.save_coco_format = save_coco_format
        self.batch_counter = 0
        
        # For COCO format aggregation
        self.coco_images: List[Dict] = []
        self.coco_annotations: List[Dict] = []
        self.annotation_id = 1
        
        print(f"ObjectDetectionPredictionWriter initialized. Saving to: {self.output_dir}")
    
    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int
    ):
        """Write predictions at the end of each batch.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module
            prediction: Model predictions (list of dicts or dict)
            batch_indices: Indices of samples in batch
            batch: Input batch
            batch_idx: Batch index
            dataloader_idx: Dataloader index
        """
        # Handle different prediction formats
        if isinstance(prediction, list):
            # List of predictions (one per image) - this is the standard format
            predictions_list = prediction
        elif isinstance(prediction, dict):
            # Single dict - wrap in list
            predictions_list = [prediction]
        else:
            # This is the error case we're fixing
            print(f"Warning: Unexpected prediction type: {type(prediction)}")
            print(f"Prediction content: {prediction}")
            return
        
        # Extract image metadata from batch if available
        image_metadata = self._extract_image_metadata(batch, batch_indices, trainer)
        
        # Save each prediction
        for i, pred in enumerate(predictions_list):
            # Convert tensors to lists for JSON serialization
            pred_dict = self._convert_prediction(pred)
            
            # Determine sample index
            if batch_indices is not None and i < len(batch_indices):
                sample_idx = batch_indices[i]
            else:
                sample_idx = self.batch_counter * len(predictions_list) + i
            
            # Save prediction to JSON file directly in output_dir (not in subdirectories)
            output_file = self.output_dir / f"prediction_{sample_idx:06d}.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(pred_dict, f, indent=2)
            
            # Aggregate for COCO format if enabled
            if self.save_coco_format:
                metadata = image_metadata[i] if i < len(image_metadata) else None
                self._add_to_coco_format(pred_dict, sample_idx, metadata)
        
        self.batch_counter += 1
    
    def _convert_prediction(self, pred: Dict[str, Any]) -> Dict[str, Any]:
        """Convert prediction tensors to JSON-serializable format.
        
        Args:
            pred: Prediction dict with tensors
            
        Returns:
            Prediction dict with lists
        """
        pred_dict = {}
        
        if self.boxes_key in pred:
            boxes = pred[self.boxes_key]
            if isinstance(boxes, torch.Tensor):
                pred_dict[self.boxes_key] = boxes.cpu().tolist()
            else:
                pred_dict[self.boxes_key] = boxes
        
        if self.labels_key in pred:
            labels = pred[self.labels_key]
            if isinstance(labels, torch.Tensor):
                pred_dict[self.labels_key] = labels.cpu().tolist()
            else:
                pred_dict[self.labels_key] = labels
        
        if self.scores_key in pred:
            scores = pred[self.scores_key]
            if isinstance(scores, torch.Tensor):
                pred_dict[self.scores_key] = scores.cpu().tolist()
            else:
                pred_dict[self.scores_key] = scores
        
        return pred_dict
    
    def _extract_image_metadata(
        self,
        batch: Any,
        batch_indices: Optional[Sequence[int]],
        trainer: Any
    ) -> List[Dict[str, Any]]:
        """Extract image metadata from batch.
        
        Args:
            batch: Input batch
            batch_indices: Batch indices
            trainer: PyTorch Lightning trainer
            
        Returns:
            List of metadata dicts for each image in batch
        """
        metadata_list = []
        
        # Try to get dataset from trainer
        dataset = None
        if hasattr(trainer, 'datamodule') and hasattr(trainer.datamodule, 'predict_dataset'):
            dataset = trainer.datamodule.predict_dataset
        elif hasattr(trainer, 'predict_dataloaders'):
            dataloaders = trainer.predict_dataloaders
            if dataloaders and len(dataloaders) > 0:
                dataset = dataloaders[0].dataset
        
        # Extract metadata from dataset if available
        if dataset is not None and batch_indices is not None:
            for idx in batch_indices:
                if hasattr(dataset, 'images') and idx < len(dataset.images):
                    img_info = dataset.images[idx]
                    metadata_list.append({
                        'file_name': img_info.get('file_name', f'image_{idx:06d}'),
                        'width': img_info.get('width', 256),
                        'height': img_info.get('height', 256),
                        'id': img_info.get('id', idx)
                    })
                else:
                    metadata_list.append(None)
        
        # If we couldn't get metadata, create default entries
        if not metadata_list:
            batch_size = len(batch['image']) if isinstance(batch, dict) and 'image' in batch else 1
            for i in range(batch_size):
                idx = batch_indices[i] if batch_indices and i < len(batch_indices) else i
                metadata_list.append({
                    'file_name': f'image_{idx:06d}',
                    'width': 256,
                    'height': 256,
                    'id': idx
                })
        
        return metadata_list
    
    def _add_to_coco_format(
        self,
        pred_dict: Dict[str, Any],
        image_id: int,
        metadata: Optional[Dict[str, Any]]
    ):
        """Add prediction to COCO format aggregation.
        
        Args:
            pred_dict: Converted prediction dictionary
            image_id: Image ID
            metadata: Image metadata dict (file_name, width, height)
        """
        # Use metadata if available, otherwise use defaults
        if metadata:
            image_info = {
                "id": metadata.get('id', image_id),
                "file_name": metadata.get('file_name', f'image_{image_id:06d}'),
                "width": metadata.get('width', 256),
                "height": metadata.get('height', 256)
            }
        else:
            image_info = {
                "id": image_id,
                "file_name": f"image_{image_id:06d}",
                "width": 256,
                "height": 256
            }
        
        self.coco_images.append(image_info)
        
        # Add annotations from predictions
        if self.boxes_key in pred_dict and pred_dict[self.boxes_key]:
            boxes = pred_dict[self.boxes_key]
            labels = pred_dict.get(self.labels_key, [1] * len(boxes))
            scores = pred_dict.get(self.scores_key, [1.0] * len(boxes))
            
            for box, label, score in zip(boxes, labels, scores):
                # Convert box format [x1, y1, x2, y2] to COCO format [x, y, width, height]
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                annotation = {
                    "id": self.annotation_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [float(x1), float(y1), float(width), float(height)],
                    "area": float(area),
                    "iscrowd": 0,
                    "score": float(score)
                }
                self.coco_annotations.append(annotation)
                self.annotation_id += 1
    
    def save_coco_format_file(self):
        """Manually save aggregated COCO format file.
        
        Call this method explicitly if on_predict_end is not triggered.
        """
        if not self.save_coco_format:
            print("COCO format saving is disabled")
            return
        
        if not self.coco_images and not self.coco_annotations:
            print("No predictions to save")
            return
        
        coco_output = {
            "categories": [
                {
                    "id": 1,
                    "name": "IMP",
                    "supercategory": "object"
                }
            ],
            "images": self.coco_images,
            "annotations": self.coco_annotations
        }
        
        output_file = self.output_dir / "predictions_coco_format.json"
        with open(output_file, 'w') as f:
            json.dump(coco_output, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Saved COCO format predictions to: {output_file}")
        print(f"Total images: {len(self.coco_images)}")
        print(f"Total annotations: {len(self.coco_annotations)}")
        print(f"{'='*60}\n")
    
    def on_predict_end(self, trainer, pl_module):
        """Save aggregated COCO format file at the end of prediction.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module
        """
        print("\n[ObjectDetectionPredictionWriter] on_predict_end called")
        self.save_coco_format_file()
    
    def on_predict_epoch_end(self, trainer, pl_module):
        """Alternative hook that may be called instead of on_predict_end.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module
        """
        print("\n[ObjectDetectionPredictionWriter] on_predict_epoch_end called")
        self.save_coco_format_file()


# Made with Bob