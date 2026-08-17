"""Instance segmentation checkpoint comparison plot configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from lfm.all_models.all_tasks import CheckpointRecord, discover_checkpoints
from lfm.all_models.all_tasks import config_defaults as defaults


@dataclass(frozen=True)
class ModelPlotSpec:
    key: str
    display_name: str
    model_family: str
    toy_architecture: str | None
    checkpoint_path: Path


@dataclass(frozen=True)
class InstanceCheckpointComparisonPlotConfig:
    data_root: Path
    output_dir: Path
    model_specs: list[ModelPlotSpec]
    dino_checkpoint: Path | None
    graha_pretrain_dir: Path | None
    gfft_config_path: Path | None
    gfft_backbone_checkpoint: Path | None
    dataset_modality: str
    graha_input_modality_mode: str
    graha_vis_uv_merge_method: str
    normalization_source: str
    normalization_modality: str
    target_size: int
    band_filter: list[int]
    image_glob: str
    label_glob: str
    image_suffix: str
    label_suffix: str
    max_samples: int | None
    batch_size: int
    num_workers: int
    normalize_inputs: bool
    graha_stats_batch_size: int
    graha_batch_size: int
    graha_num_workers: int
    graha_backbone_lr: float
    graha_head_lr: float
    graha_layer_decay: float
    graha_weight_decay: float
    graha_warmup_steps: int
    graha_anchor_sizes: list[list[int]]
    graha_anchor_aspect_ratios: list[float]
    graha_score_threshold: float
    prediction_split: str
    n_samples: int
    score_threshold: float
    mask_shift: tuple[int, int]
    ignore_nodata_in_loss: bool
    nodata_ignore_index: int
    excluded_nodata_values: list[float] | None
    image_nodata_policy: str
    seed: int


def _final_checkpoint_from_dir(checkpoint_dir: Path) -> CheckpointRecord:
    try:
        checkpoints = discover_checkpoints(checkpoint_dir)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not resolve a final checkpoint from {checkpoint_dir}. "
            "If this came from a three-model run, inspect the corresponding "
            "model job log and confirm that training wrote at least one .ckpt file."
        ) from exc
    return checkpoints[-1]


def _resolve_checkpoint(
    *,
    checkpoint_path: Path | None,
    checkpoint_dir: Path | None,
    label: str,
) -> Path | None:
    if checkpoint_path is not None and checkpoint_dir is not None:
        raise ValueError(
            f"Pass either --{label}-checkpoint or --{label}-checkpoint-dir, not both."
        )
    if checkpoint_path is not None:
        checkpoint_path = checkpoint_path.resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"{label} checkpoint does not exist: {checkpoint_path}"
            )
        return checkpoint_path
    if checkpoint_dir is None:
        return None
    record = _final_checkpoint_from_dir(checkpoint_dir)
    print(f"[{label}] selected final checkpoint: {record.path}", flush=True)
    return record.path


def _apply_run_root_defaults(args: argparse.Namespace) -> None:
    if args.run_root is None:
        return
    run_root = args.run_root.resolve()
    if args.toy_checkpoint_dir is None and args.toy_checkpoint is None:
        args.toy_checkpoint_dir = run_root / "toy_model" / "checkpoints" / "toy_model"
    if args.mask2former_checkpoint_dir is None and args.mask2former_checkpoint is None:
        args.mask2former_checkpoint_dir = (
            run_root / "toy_mask2former" / "checkpoints" / "toy_model"
        )
    if (
        args.toy_terratorch_checkpoint_dir is None
        and args.toy_terratorch_checkpoint is None
    ):
        args.toy_terratorch_checkpoint_dir = (
            run_root / "toy_dino_terratorch_mask_rcnn" / "checkpoints" / "toy_model"
        )
    if args.graha_checkpoint_dir is None and args.graha_checkpoint is None:
        args.graha_checkpoint_dir = (
            run_root / "graha_mask_rcnn" / "checkpoints" / "full_model"
        )
    if args.gfft_checkpoint_dir is None and args.gfft_checkpoint is None:
        args.gfft_checkpoint_dir = (
            run_root / "gfft_mask_rcnn" / "checkpoints" / "gfft_model"
        )


def _build_model_specs(args: argparse.Namespace) -> list[ModelPlotSpec]:
    _apply_run_root_defaults(args)
    specs = []
    toy = _resolve_checkpoint(
        checkpoint_path=args.toy_checkpoint,
        checkpoint_dir=args.toy_checkpoint_dir,
        label="toy",
    )
    if toy is not None:
        specs.append(
            ModelPlotSpec(
                key="toy_model",
                display_name=f"Toy {args.toy_plot_architecture}",
                model_family="toy",
                toy_architecture=args.toy_plot_architecture,
                checkpoint_path=toy,
            )
        )
    else:
        mask2former = _resolve_checkpoint(
            checkpoint_path=args.mask2former_checkpoint,
            checkpoint_dir=args.mask2former_checkpoint_dir,
            label="mask2former",
        )
        if mask2former is not None:
            specs.append(
                ModelPlotSpec(
                    key="toy_mask2former",
                    display_name="Toy Mask2Former",
                    model_family="toy",
                    toy_architecture="mask2former",
                    checkpoint_path=mask2former,
                )
            )

        toy_terratorch = _resolve_checkpoint(
            checkpoint_path=args.toy_terratorch_checkpoint,
            checkpoint_dir=args.toy_terratorch_checkpoint_dir,
            label="toy-terratorch",
        )
        if toy_terratorch is not None:
            specs.append(
                ModelPlotSpec(
                    key="toy_dino_terratorch_mask_rcnn",
                    display_name="Toy DINO TerraTorch Mask R-CNN",
                    model_family="toy",
                    toy_architecture="dino-terratorch-mask-rcnn",
                    checkpoint_path=toy_terratorch,
                )
            )

    graha = _resolve_checkpoint(
        checkpoint_path=args.graha_checkpoint,
        checkpoint_dir=args.graha_checkpoint_dir,
        label="graha",
    )
    if graha is not None:
        specs.append(
            ModelPlotSpec(
                key="graha_mask_rcnn",
                display_name="Graha Mask R-CNN",
                model_family="graha",
                toy_architecture=None,
                checkpoint_path=graha,
            )
        )

    gfft = _resolve_checkpoint(
        checkpoint_path=args.gfft_checkpoint,
        checkpoint_dir=args.gfft_checkpoint_dir,
        label="gfft",
    )
    if gfft is not None:
        specs.append(
            ModelPlotSpec(
                key="gfft_mask_rcnn",
                display_name="GFFT Mask R-CNN",
                model_family="gfft",
                toy_architecture=None,
                checkpoint_path=gfft,
            )
        )

    if len(specs) < 2:
        raise ValueError("At least two model checkpoints are required for comparison.")
    return specs


def build_checkpoint_comparison_plot_config_from_args(
    args: argparse.Namespace,
) -> InstanceCheckpointComparisonPlotConfig:
    dataset_modality = args.dataset_modality
    band_filter = (
        list(args.band_filter)
        if args.band_filter is not None
        else defaults.default_band_filter_for_dataset(dataset_modality)
    )
    return InstanceCheckpointComparisonPlotConfig(
        data_root=args.data_root.resolve(),
        output_dir=args.output_dir.resolve(),
        model_specs=_build_model_specs(args),
        dino_checkpoint=args.dino_checkpoint,
        graha_pretrain_dir=args.graha_pretrain_dir,
        gfft_config_path=args.gfft_config_path,
        gfft_backbone_checkpoint=args.gfft_backbone_checkpoint,
        dataset_modality=dataset_modality,
        graha_input_modality_mode=defaults.resolve_graha_input_modality_mode(
            dataset_modality=dataset_modality,
            graha_input_modality_mode=args.graha_input_modality_mode,
        ),
        graha_vis_uv_merge_method=args.graha_vis_uv_merge_method,
        normalization_source=args.normalization_source,
        normalization_modality=defaults.resolve_normalization_modality(
            dataset_modality=dataset_modality,
            normalization_modality=args.normalization_modality,
        ),
        target_size=args.target_size,
        band_filter=band_filter,
        image_glob=args.image_glob,
        label_glob=args.label_glob,
        image_suffix=args.image_suffix,
        label_suffix=args.label_suffix,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize_inputs=args.normalize_inputs,
        graha_stats_batch_size=args.graha_stats_batch_size,
        graha_batch_size=args.graha_batch_size,
        graha_num_workers=args.graha_num_workers,
        graha_backbone_lr=args.graha_backbone_lr,
        graha_head_lr=args.graha_head_lr,
        graha_layer_decay=args.graha_layer_decay,
        graha_weight_decay=args.graha_weight_decay,
        graha_warmup_steps=args.graha_warmup_steps,
        graha_anchor_sizes=args.graha_anchor_sizes,
        graha_anchor_aspect_ratios=args.graha_anchor_aspect_ratios,
        graha_score_threshold=args.graha_score_threshold,
        prediction_split=args.prediction_split,
        n_samples=args.n_samples,
        score_threshold=args.score_threshold,
        mask_shift=tuple(args.mask_shift),
        ignore_nodata_in_loss=args.ignore_nodata_in_loss,
        nodata_ignore_index=args.nodata_ignore_index,
        excluded_nodata_values=args.excluded_nodata_values,
        image_nodata_policy=args.image_nodata_policy,
        seed=args.seed,
    )
