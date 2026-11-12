"""Lightning wrappers for toy instance segmentation experiments."""

from importlib import import_module

__all__ = [
    "ToyDinoMaskRCNNLightningModule",
    "ToyDinoMaskRCNNSplitDataModule",
    "ToyInstanceSegLightningModule",
    "ToyInstanceSegSplitDataModule",
]


def __getattr__(name: str):
    modules = {
        "ToyDinoMaskRCNNSplitDataModule": ".toy_dino_mask_rcnn_datamodule",
        "ToyDinoMaskRCNNLightningModule": ".toy_dino_mask_rcnn_lightning",
        "ToyInstanceSegSplitDataModule": ".toy_instance_seg_datamodule",
        "ToyInstanceSegLightningModule": ".toy_instance_seg_lightning",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
