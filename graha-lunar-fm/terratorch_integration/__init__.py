"""Custom TerraTorch modules for lunar-fm models.

This package provides TerraTorch-compatible wrappers for the lunar model
architecture, enabling integration without porting the entire codebase.

The integration supports the post-refactoring architecture including:
- Three-tier configuration system (schemas → MODALITY_INFO → Hydra configs)
- Schema-based embedding initialization
- Codebook fusion for multi-codebook tokenizers
- Per-codebook loss tracking

Usage:
    Add this directory to TerraTorch's custom_modules_path:

    terratorch fit --config my_config.yaml --custom_modules_path ./terratorch_integration

    Or in YAML:
    custom_modules_path: ./terratorch_integration
"""

__version__ = "2.0.0"
__all__ = ["lunar_backbone", "lunar_register", "data_adapter"]

# Import registration modules to trigger @register decorators
# This happens when TerraTorch imports the terratorch_integration package
from . import lunar_register  # noqa: F401
from .data_adapter import LunarCraterDataModule
from .data_adapter_imp import LunarImpSegDataset, LunarImpSegDataModule

# Explicitly import the registration functions to ensure they're executed
from .lunar_register import (  # noqa: F401
    lunarmind_v1_tiny,
    lunarmind_v1_base,
    lunarmind_v1_large,
    lunar_fvqmultimae,
)
from .necks import LearnedTokenProjection, SimpleFeaturePyramid, MultilayerSimpleFeaturePyramid  # noqa: F401
from .decoders import SumFuseDeepGNDecoder  # noqa: F401
from .lunar_object_detection_task import LunarObjectDetectionTask  # noqa: F401
from .lunar_segmentation_task import (  # noqa: F401
    LunarSegmentationTask,
    LunarShapeSegmentationTask,
)
from .lunar_regression_task import LunarPixelwiseRegressionTask  # noqa: F401
from .lunar_classification_task import (  # noqa: F401
    LunarClassificationTask,
    LunarScalarRegressionTask,
)