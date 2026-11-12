"""Shared instance segmentation helpers, loaded without optional inference imports."""

from importlib import import_module

_EXPORT_MODULES = {
    "GrahaInstanceNotebookConfigs": "lfm.all_models.inst_seg.notebook_config",
    "GfftInstanceNotebookConfigs": "lfm.all_models.inst_seg.notebook_config",
    "InstanceCheckpointComparisonPlotConfig": "lfm.all_models.inst_seg.plot_config",
    "InstanceCheckpointSweepConfig": "lfm.all_models.inst_seg.sweep_config",
    "InstanceSegmentationExperimentConfig": "lfm.all_models.inst_seg.config",
    "InstanceMaskSegmentationDataModule": "lfm.all_models.inst_seg.data",
    "InstanceSegmentationDataset": "lfm.all_models.inst_seg.data",
    "InstanceSegmentationDataModule": "lfm.all_models.inst_seg.data",
    "LunarInstanceMaskDataset": "lfm.all_models.inst_seg.data",
    "ModelPlotSpec": "lfm.all_models.inst_seg.plot_config",
    "ObjectDetectionInstanceSegmentationDataModule": "lfm.all_models.inst_seg.data",
    "ObjectDetectionInstanceSegmentationDataset": "lfm.all_models.inst_seg.data",
    "boxes_to_tensor": "lfm.all_models.inst_seg.data",
    "build_config": "lfm.all_models.inst_seg.config",
    "build_checkpoint_comparison_plot_config_from_args": "lfm.all_models.inst_seg.plot_config",
    "build_checkpoint_sweep_config_from_args": "lfm.all_models.inst_seg.sweep_config",
    "build_config_from_args": "lfm.all_models.inst_seg.config",
    "build_graha_notebook_configs": "lfm.all_models.inst_seg.notebook_config",
    "build_gfft_notebook_configs": "lfm.all_models.inst_seg.notebook_config",
    "collate_mask2former_instance_segmentation": "lfm.all_models.inst_seg.data",
    "instance_mask_to_object_detection_targets": "lfm.all_models.inst_seg.data",
    "mask_to_binary_instance_targets": "lfm.all_models.inst_seg.data",
    "minmax_scale_per_band": "lfm.all_models.inst_seg.data",
    "GrahaInstanceModel": "lfm.all_models.all_tasks.graha_inference",
    "load_and_configure_input_data": "lfm.all_models.inst_seg.data_cube_inference",
    "plot_instance_inference_results": "lfm.all_models.inst_seg.data_cube_inference",
    "preprocess_datacubes": "lfm.all_models.inst_seg.data_cube_inference",
    "sliding_window_instance_inference": "lfm.all_models.inst_seg.data_cube_inference",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str):
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORT_MODULES[name]), name)
    globals()[name] = value
    return value
