"""Central argparse definitions for segmentation experiment entrypoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from lfm.all_models.all_tasks import config_defaults as defaults


def _anchor_sizes(value: str) -> list[list[int]]:
    return [[int(x)] for x in value.split(",")]


def _anchor_aspect_ratios(value: str) -> list[float]:
    return [float(x) for x in value.split(",")]


def _add_symlink_arg(parser: argparse.ArgumentParser, *, semantic_help: bool) -> None:
    kwargs = {
        "dest": "simlink_dest",
        "type": str,
        "default": None,
    }
    if semantic_help:
        kwargs["help"] = (
            "Optional source directory for notebooks/full_model/data. If ./data is already a "
            "symlink, it must point to this same directory."
        )
    parser.add_argument("--simlink-dest", "--symlink-dest", **kwargs)


def _add_model_selection_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(defaults.DEFAULT_MODELS),
        choices=defaults.MODEL_CHOICES,
    )


def _add_dataset_modality_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-modality",
        choices=defaults.DATASET_MODALITY_CHOICES,
        default=defaults.DEFAULT_DATASET_MODALITY,
        help=(
            "Input dataset modality. Defaults drive Graha input modality mode "
            "and pretrain normalization modality unless those lower-level "
            "options are supplied explicitly."
        ),
    )


def _add_data_root_args(
    parser: argparse.ArgumentParser,
    *,
    output_arg: str,
    checkpoint_dirs: bool = False,
    path_type=str,
) -> None:
    parser.add_argument("--data-root", type=path_type, default=None)
    parser.add_argument(output_arg, type=path_type, default=None)
    if checkpoint_dirs:
        parser.add_argument("--toy-checkpoint-dir", type=path_type, default=None)
        parser.add_argument("--graha-checkpoint-dir", type=path_type, default=None)


def _add_graha_input_args(parser: argparse.ArgumentParser, *, path_type=str) -> None:
    parser.add_argument("--graha-pretrain-dir", type=path_type, default=None)
    parser.add_argument(
        "--gfft-config-path",
        type=path_type,
        default=None,
        help="Optional TerraTorch-style GFFT YAML config used by GFFT-only workflows.",
    )
    parser.add_argument(
        "--gfft-backbone-checkpoint",
        type=path_type,
        default=None,
        help="Optional GFFT/Fourier-VQ MultiMAE backbone .pth checkpoint.",
    )
    parser.add_argument(
        "--graha-input-modality-mode",
        choices=defaults.GRAHA_INPUT_MODALITY_CHOICES,
        default=None,
        help=(
            "Low-level Graha modality registration override. If omitted, it is "
            "derived from --dataset-modality."
        ),
    )
    parser.add_argument(
        "--graha-vis-uv-merge-method",
        choices=defaults.GRAHA_VIS_UV_MERGE_CHOICES,
        default=defaults.DEFAULT_GRAHA_VIS_UV_MERGE_METHOD,
    )


def _add_normalization_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--normalization-source",
        choices=defaults.NORMALIZATION_SOURCE_CHOICES,
        default=defaults.DEFAULT_NORMALIZATION_SOURCE,
        help="When normalizing inputs, use TerraMind pretraining stats or finetuning train-split stats.",
    )
    parser.add_argument(
        "--normalization-modality",
        choices=defaults.NORMALIZATION_MODALITY_CHOICES,
        default=None,
        help=(
            "Which modality family to use when --normalization-source=pretrain. "
            "If omitted, it is derived from --dataset-modality."
        ),
    )


def _add_normalization_args_without_help(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--normalization-source",
        choices=defaults.NORMALIZATION_SOURCE_CHOICES,
        default=defaults.DEFAULT_NORMALIZATION_SOURCE,
    )
    parser.add_argument(
        "--normalization-modality",
        choices=defaults.NORMALIZATION_MODALITY_CHOICES,
        default=None,
    )


def _add_band_filter_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--band-filter",
        type=int,
        nargs="+",
        default=list(defaults.DEFAULT_BAND_FILTER),
    )


def _add_semantic_label_source_arg(
    parser: argparse.ArgumentParser,
    *,
    detailed_help: bool,
) -> None:
    help_text = (
        "Use .npy semantic labels or .npz instance labels converted to semantic masks."
        if detailed_help
        else None
    )
    parser.add_argument(
        "--semantic-label-source",
        choices=defaults.SEMANTIC_LABEL_SOURCE_CHOICES,
        default=defaults.DEFAULT_SEMANTIC_LABEL_SOURCE,
        help=help_text,
    )


def _add_matching_args(
    parser: argparse.ArgumentParser,
    *,
    image_suffix_default: str,
    label_glob_default: str = defaults.DEFAULT_LABEL_GLOB,
    image_glob_default: str = defaults.DEFAULT_IMAGE_GLOB,
    path_defaults: bool = False,
) -> None:
    parser.add_argument(
        "--image-glob",
        default=image_glob_default,
        help="Chip filename glob inside each split/chips directory.",
    )
    parser.add_argument(
        "--label-glob",
        default=label_glob_default,
        help="Label filename glob inside each split/labels directory.",
    )
    parser.add_argument(
        "--image-suffix",
        default=image_suffix_default,
        help=(
            "Optional suffix stripped from chip stems before matching labels. "
            "If omitted, matching is inferred from chip/label filename tokens."
            if not path_defaults
            else None
        ),
    )
    parser.add_argument(
        "--label-suffix",
        default=defaults.DEFAULT_LABEL_SUFFIX,
        help=(
            "Optional suffix stripped from label stems before matching chips. "
            "If omitted, matching is inferred from chip/label filename tokens."
            if not path_defaults
            else None
        ),
    )


def _add_sample_limit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)


def _add_nodata_args(
    parser: argparse.ArgumentParser,
    *,
    semantic_help: bool,
) -> None:
    parser.add_argument(
        "--ignore-nodata-in-loss",
        action="store_true",
        help=(
            "Ignore TIFF nodata pixels in semantic segmentation loss and metrics."
            if semantic_help
            else "Thread TIFF nodata pixels through instance target preprocessing."
        ),
    )
    parser.add_argument(
        "--nodata-ignore-index",
        type=int,
        default=defaults.DEFAULT_NODATA_IGNORE_INDEX,
        help="Target label value used for ignored nodata pixels.",
    )


def _add_prediction_args(
    parser: argparse.ArgumentParser,
    *,
    prediction_default: str,
    n_samples_default: int,
    include_score_threshold: bool,
) -> None:
    parser.add_argument(
        "--prediction-split",
        choices=defaults.SPLIT_CHOICES,
        default=prediction_default,
    )
    parser.add_argument("--prediction-n-samples", type=int, default=n_samples_default)
    if include_score_threshold:
        parser.add_argument(
            "--prediction-score-threshold",
            type=float,
            default=defaults.DEFAULT_PREDICTION_SCORE_THRESHOLD,
        )


def _add_epoch_test_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-epoch-test-suite", action="store_true")
    parser.add_argument(
        "--epoch-test-split",
        choices=defaults.SPLIT_CHOICES,
        default=defaults.DEFAULT_EPOCH_TEST_SPLIT,
    )
    parser.add_argument(
        "--epoch-test-n-samples",
        type=int,
        default=defaults.DEFAULT_EPOCH_TEST_N_SAMPLES,
    )
    parser.add_argument(
        "--epoch-test-every-n-epochs",
        type=int,
        default=defaults.DEFAULT_EPOCH_TEST_EVERY_N_EPOCHS,
    )


def _add_graha_anchor_args(
    parser: argparse.ArgumentParser,
    *,
    parsed_defaults: bool,
) -> None:
    if parsed_defaults:
        parser.add_argument(
            "--graha-anchor-sizes",
            type=_anchor_sizes,
            default=[list(size) for size in defaults.DEFAULT_GRAHA_ANCHOR_SIZES],
        )
        parser.add_argument(
            "--graha-anchor-aspect-ratios",
            type=_anchor_aspect_ratios,
            default=list(defaults.DEFAULT_GRAHA_ANCHOR_ASPECT_RATIOS),
        )
    else:
        parser.add_argument(
            "--graha-anchor-sizes",
            type=str,
            default=defaults.DEFAULT_GRAHA_ANCHOR_SIZES_CSV,
        )
        parser.add_argument(
            "--graha-anchor-aspect-ratios",
            type=str,
            default=defaults.DEFAULT_GRAHA_ANCHOR_ASPECT_RATIOS_CSV,
        )


def create_semantic_experiment_parser(
    description: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    _add_symlink_arg(parser, semantic_help=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--base-output-dir", type=str, default=None)
    parser.add_argument(
        "--no-timestamp-output-dir",
        action="store_true",
        help=(
            "Write directly to --base-output-dir instead of creating a "
            "timestamped child directory. Intended for higher-level orchestrators "
            "that already create a timestamped run directory."
        ),
    )
    parser.add_argument("--dino-checkpoint", type=str, default=None)
    _add_dataset_modality_arg(parser)
    parser.add_argument(
        "--toy-lightning-checkpoint",
        type=str,
        default=None,
        help="Optional Toy Lightning .ckpt. Resumes fit, or loads weights when Toy fit is skipped.",
    )
    _add_band_filter_arg(parser)
    parser.add_argument("--target-size", type=int, default=defaults.DEFAULT_TARGET_SIZE)
    _add_semantic_label_source_arg(parser, detailed_help=True)
    _add_matching_args(parser, image_suffix_default=defaults.DEFAULT_IMAGE_SUFFIX)
    _add_sample_limit_args(parser)
    _add_nodata_args(parser, semantic_help=True)
    parser.add_argument(
        "--batch-size", type=int, default=defaults.DEFAULT_SEMANTIC_BATCH_SIZE
    )
    parser.add_argument(
        "--num-workers", type=int, default=defaults.DEFAULT_SEMANTIC_NUM_WORKERS
    )
    parser.add_argument("--max-epochs", type=int, default=defaults.DEFAULT_MAX_EPOCHS)
    parser.add_argument(
        "--learning-rate", type=float, default=defaults.DEFAULT_SEMANTIC_LEARNING_RATE
    )
    parser.add_argument(
        "--weight-decay", type=float, default=defaults.DEFAULT_SEMANTIC_WEIGHT_DECAY
    )
    parser.add_argument(
        "--toy-loss-type",
        type=str,
        default=defaults.DEFAULT_TOY_SEMANTIC_LOSS_TYPE,
        help="Loss function for the Toy semantic model. Graha semantic currently uses its own Dice loss path.",
    )
    parser.add_argument("--use-toy-shape-loss", action="store_true")
    parser.add_argument(
        "--toy-shape-loss-weight",
        type=float,
        default=defaults.DEFAULT_TOY_SHAPE_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--toy-shape-loss-pad-frac",
        type=float,
        default=defaults.DEFAULT_TOY_SHAPE_LOSS_PAD_FRAC,
    )
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument(
        "--normalize-inputs",
        action="store_true",
        help="Enable Toy z-score normalization.",
    )
    _add_normalization_args(parser)
    parser.add_argument(
        "--toy-gradient-clip-val",
        type=float,
        default=defaults.DEFAULT_TOY_GRADIENT_CLIP_VAL,
    )
    parser.add_argument(
        "--disable-toy-gradient-clipping",
        action="store_true",
        help="Disable Toy gradient clipping to match Graha's current trainer path.",
    )
    parser.add_argument(
        "--plot-every-n-epochs",
        type=int,
        default=defaults.DEFAULT_PLOT_EVERY_N_EPOCHS,
    )
    parser.add_argument(
        "--plot-n-samples", type=int, default=defaults.DEFAULT_PLOT_N_SAMPLES
    )
    parser.add_argument("--cache-predictions", action="store_true")
    _add_prediction_args(
        parser,
        prediction_default=defaults.DEFAULT_PREDICTION_SPLIT,
        n_samples_default=defaults.DEFAULT_SEMANTIC_PREDICTION_N_SAMPLES,
        include_score_threshold=False,
    )
    parser.add_argument("--graha-base-output-dir", type=str, default=None)
    parser.add_argument("--graha-pretrain-dir", type=str, default=None)
    parser.add_argument(
        "--gfft-config-path",
        type=str,
        default=None,
        help="Optional TerraTorch-style GFFT YAML config used by GFFT-only workflows.",
    )
    parser.add_argument("--gfft-backbone-checkpoint", type=str, default=None)
    parser.add_argument(
        "--graha-lightning-checkpoint",
        type=str,
        default=None,
        help="Optional Graha Lightning .ckpt. Resumes fit, or loads weights when Graha fit is skipped.",
    )
    parser.add_argument(
        "--graha-input-modality-mode",
        choices=defaults.GRAHA_INPUT_MODALITY_CHOICES,
        default=None,
        help=(
            "Low-level Graha modality registration override. If omitted, it is "
            "derived from --dataset-modality."
        ),
    )
    parser.add_argument(
        "--graha-vis-uv-merge-method",
        choices=defaults.GRAHA_VIS_UV_MERGE_CHOICES,
        default=defaults.DEFAULT_GRAHA_VIS_UV_MERGE_METHOD,
    )
    parser.add_argument(
        "--graha-shape-loss-weight",
        type=float,
        default=defaults.DEFAULT_GRAHA_SHAPE_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--graha-shape-loss-pad-frac",
        type=float,
        default=defaults.DEFAULT_GRAHA_SHAPE_LOSS_PAD_FRAC,
    )
    parser.add_argument(
        "--graha-stats-batch-size",
        type=int,
        default=defaults.DEFAULT_GRAHA_STATS_BATCH_SIZE,
    )
    parser.add_argument(
        "--graha-batch-size",
        type=int,
        default=defaults.DEFAULT_GRAHA_SEMANTIC_BATCH_SIZE,
    )
    parser.add_argument(
        "--graha-num-workers", type=int, default=defaults.DEFAULT_GRAHA_NUM_WORKERS
    )
    parser.add_argument(
        "--progress-log-every-n-batches",
        type=int,
        default=defaults.DEFAULT_PROGRESS_LOG_EVERY_N_BATCHES,
        help="Flush train-batch progress every N batches in sbatch logs.",
    )
    parser.add_argument("--seed", type=int, default=defaults.DEFAULT_SEED)
    parser.add_argument(
        "--no-fit", action="store_true", help="Build data/model/trainer but skip fit."
    )
    parser.add_argument(
        "--skip-toy-fit", action="store_true", help="Skip only Toy fitting."
    )
    parser.add_argument(
        "--skip-graha-fit", action="store_true", help="Skip only Graha fitting."
    )
    _add_epoch_test_args(parser)
    return parser


def parse_semantic_experiment_args(
    argv: list[str] | None = None,
    *,
    description: str | None = None,
) -> argparse.Namespace:
    return create_semantic_experiment_parser(description).parse_args(argv)


def create_instance_experiment_parser(
    description: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    _add_symlink_arg(parser, semantic_help=False)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--base-output-dir", type=str, default=None)
    parser.add_argument(
        "--no-timestamp-output-dir",
        action="store_true",
        help=(
            "Write directly to --base-output-dir instead of creating a "
            "timestamped child directory. Intended for higher-level orchestrators "
            "that already create a timestamped run directory."
        ),
    )
    parser.add_argument("--dino-checkpoint", type=str, default=None)
    _add_dataset_modality_arg(parser)
    parser.add_argument(
        "--toy-lightning-checkpoint",
        type=str,
        default=None,
        help="Optional Toy Lightning .ckpt. Resumes fit, or loads weights when Toy fit is skipped.",
    )
    parser.add_argument("--graha-pretrain-dir", type=str, default=None)
    parser.add_argument(
        "--gfft-config-path",
        type=str,
        default=None,
        help="Optional TerraTorch-style GFFT YAML config used by GFFT-only workflows.",
    )
    parser.add_argument("--gfft-backbone-checkpoint", type=str, default=None)
    parser.add_argument("--graha-lightning-checkpoint", type=str, default=None)
    parser.add_argument(
        "--graha-input-modality-mode",
        choices=defaults.GRAHA_INPUT_MODALITY_CHOICES,
        default=None,
        help=(
            "Low-level Graha modality registration override. If omitted, it is "
            "derived from --dataset-modality."
        ),
    )
    parser.add_argument(
        "--graha-vis-uv-merge-method",
        choices=defaults.GRAHA_VIS_UV_MERGE_CHOICES,
        default=defaults.DEFAULT_GRAHA_VIS_UV_MERGE_METHOD,
    )
    _add_normalization_args(parser)
    _add_matching_args(parser, image_suffix_default=defaults.DEFAULT_IMAGE_SUFFIX)
    parser.add_argument(
        "--toy-architecture",
        choices=defaults.TOY_INSTANCE_ARCHITECTURE_CHOICES,
        default=defaults.DEFAULT_INSTANCE_TOY_ARCHITECTURE,
        help=(
            "Toy instance head to train. Use dino-mask-rcnn for TorchVision "
            "Mask R-CNN or dino-terratorch-mask-rcnn for TerraTorch Mask R-CNN."
        ),
    )
    parser.add_argument("--target-size", type=int, default=defaults.DEFAULT_TARGET_SIZE)
    _add_band_filter_arg(parser)
    _add_sample_limit_args(parser)
    parser.add_argument(
        "--toy-batch-size",
        "--batch-size",
        dest="toy_batch_size",
        type=int,
        default=defaults.DEFAULT_INSTANCE_TOY_BATCH_SIZE,
        help="Toy instance batch size. --batch-size is accepted for parity with semantic scripts.",
    )
    parser.add_argument(
        "--toy-num-workers",
        type=int,
        default=defaults.DEFAULT_INSTANCE_TOY_NUM_WORKERS,
    )
    parser.add_argument(
        "--graha-stats-batch-size",
        type=int,
        default=defaults.DEFAULT_GRAHA_STATS_BATCH_SIZE,
    )
    parser.add_argument(
        "--graha-batch-size",
        type=int,
        default=defaults.DEFAULT_GRAHA_INSTANCE_BATCH_SIZE,
    )
    parser.add_argument(
        "--graha-num-workers", type=int, default=defaults.DEFAULT_GRAHA_NUM_WORKERS
    )
    parser.add_argument("--max-epochs", type=int, default=defaults.DEFAULT_MAX_EPOCHS)
    parser.add_argument(
        "--toy-learning-rate",
        type=float,
        default=defaults.DEFAULT_INSTANCE_LEARNING_RATE,
    )
    parser.add_argument(
        "--toy-weight-decay",
        type=float,
        default=defaults.DEFAULT_INSTANCE_WEIGHT_DECAY,
    )
    parser.add_argument("--toy-freeze-backbone", action="store_true")
    parser.add_argument(
        "--toy-normalize-inputs",
        "--normalize-inputs",
        dest="toy_normalize_inputs",
        action="store_true",
        help="Enable Toy instance z-score normalization. --normalize-inputs is accepted for parity with semantic scripts.",
    )
    parser.add_argument(
        "--toy-gradient-clip-val",
        type=float,
        default=defaults.DEFAULT_TOY_GRADIENT_CLIP_VAL,
    )
    parser.add_argument("--disable-toy-gradient-clipping", action="store_true")
    parser.add_argument(
        "--graha-backbone-lr", type=float, default=defaults.DEFAULT_GRAHA_BACKBONE_LR
    )
    parser.add_argument(
        "--graha-head-lr", type=float, default=defaults.DEFAULT_GRAHA_HEAD_LR
    )
    parser.add_argument(
        "--graha-layer-decay", type=float, default=defaults.DEFAULT_GRAHA_LAYER_DECAY
    )
    parser.add_argument(
        "--graha-weight-decay", type=float, default=defaults.DEFAULT_GRAHA_WEIGHT_DECAY
    )
    parser.add_argument(
        "--graha-warmup-steps", type=int, default=defaults.DEFAULT_GRAHA_WARMUP_STEPS
    )
    _add_graha_anchor_args(parser, parsed_defaults=True)
    parser.add_argument(
        "--graha-score-threshold",
        type=float,
        default=defaults.DEFAULT_GRAHA_SCORE_THRESHOLD,
    )
    parser.add_argument(
        "--plot-every-n-epochs",
        type=int,
        default=defaults.DEFAULT_PLOT_EVERY_N_EPOCHS,
    )
    parser.add_argument(
        "--plot-n-samples", type=int, default=defaults.DEFAULT_PLOT_N_SAMPLES
    )
    parser.add_argument(
        "--progress-log-every-n-batches",
        type=int,
        default=defaults.DEFAULT_PROGRESS_LOG_EVERY_N_BATCHES,
        help="Flush train-batch progress every N batches in sbatch logs.",
    )
    _add_prediction_args(
        parser,
        prediction_default=defaults.DEFAULT_PREDICTION_SPLIT,
        n_samples_default=defaults.DEFAULT_INSTANCE_PREDICTION_N_SAMPLES,
        include_score_threshold=True,
    )
    parser.add_argument(
        "--mask-shift", type=int, nargs=2, default=defaults.DEFAULT_MASK_SHIFT
    )
    _add_nodata_args(parser, semantic_help=False)
    parser.add_argument(
        "--skip-toy-fit", action="store_true", help="Skip only Toy fitting."
    )
    parser.add_argument("--skip-graha-fit", action="store_true")
    parser.add_argument("--no-fit", action="store_true")
    _add_epoch_test_args(parser)
    parser.add_argument("--seed", type=int, default=defaults.DEFAULT_SEED)
    return parser


def parse_instance_experiment_args(
    argv: list[str] | None = None,
    *,
    description: str | None = None,
) -> argparse.Namespace:
    return create_instance_experiment_parser(description).parse_args(argv)


def create_single_model_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", choices=defaults.MODEL_CHOICES, required=True)
    return parser


def parse_single_model_args(
    argv: list[str] | None = None,
) -> tuple[str, list[str]]:
    model_args, remaining = create_single_model_parser().parse_known_args(argv)
    return model_args.model, remaining


def create_checkpoint_pipeline_parser(
    description: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--task", choices=defaults.TASK_CHOICES, required=True)
    _add_symlink_arg(parser, semantic_help=False)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--base-output-dir", type=str, default=None)
    parser.add_argument("--existing-training-output-dir", type=str, default=None)
    parser.add_argument("--sweep-output-root", type=str, default=None)
    parser.add_argument("--skip-sweep", action="store_true")
    _add_model_selection_arg(parser)
    _add_dataset_modality_arg(parser)
    parser.add_argument("--dino-checkpoint", type=str, default=None)
    _add_graha_input_args(parser)
    parser.add_argument("--target-size", type=int, default=defaults.DEFAULT_TARGET_SIZE)
    _add_band_filter_arg(parser)
    _add_semantic_label_source_arg(parser, detailed_help=True)
    _add_matching_args(parser, image_suffix_default=defaults.DEFAULT_IMAGE_SUFFIX)
    _add_sample_limit_args(parser)
    _add_nodata_args(parser, semantic_help=True)
    parser.add_argument("--max-epochs", type=int, default=defaults.DEFAULT_MAX_EPOCHS)
    parser.add_argument(
        "--plot-every-n-epochs",
        type=int,
        default=defaults.DEFAULT_PLOT_EVERY_N_EPOCHS,
    )
    parser.add_argument(
        "--plot-n-samples", type=int, default=defaults.DEFAULT_PLOT_N_SAMPLES
    )
    _add_prediction_args(
        parser,
        prediction_default=defaults.DEFAULT_PREDICTION_SPLIT,
        n_samples_default=defaults.DEFAULT_INSTANCE_PREDICTION_N_SAMPLES,
        include_score_threshold=False,
    )
    _add_epoch_test_args(parser)
    parser.add_argument(
        "--sweep-split",
        choices=defaults.SPLIT_CHOICES,
        default=defaults.DEFAULT_SWEEP_SPLIT,
    )
    parser.add_argument("--sweep-max-samples", type=int, default=None)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--seed", type=int, default=defaults.DEFAULT_SEED)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument(
        "--batch-size", type=int, default=defaults.DEFAULT_SEMANTIC_BATCH_SIZE
    )
    parser.add_argument(
        "--num-workers", type=int, default=defaults.DEFAULT_SEMANTIC_NUM_WORKERS
    )
    parser.add_argument(
        "--learning-rate", type=float, default=defaults.DEFAULT_SEMANTIC_LEARNING_RATE
    )
    parser.add_argument(
        "--weight-decay", type=float, default=defaults.DEFAULT_SEMANTIC_WEIGHT_DECAY
    )
    parser.add_argument(
        "--toy-loss-type", type=str, default=defaults.DEFAULT_TOY_SEMANTIC_LOSS_TYPE
    )
    parser.add_argument("--use-toy-shape-loss", action="store_true")
    parser.add_argument(
        "--toy-shape-loss-weight",
        type=float,
        default=defaults.DEFAULT_TOY_SHAPE_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--toy-shape-loss-pad-frac",
        type=float,
        default=defaults.DEFAULT_TOY_SHAPE_LOSS_PAD_FRAC,
    )
    parser.add_argument(
        "--graha-shape-loss-weight",
        type=float,
        default=defaults.DEFAULT_GRAHA_SHAPE_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--graha-shape-loss-pad-frac",
        type=float,
        default=defaults.DEFAULT_GRAHA_SHAPE_LOSS_PAD_FRAC,
    )
    parser.add_argument("--normalize-inputs", action="store_true")
    _add_normalization_args(parser)

    parser.add_argument(
        "--toy-batch-size",
        type=int,
        default=defaults.DEFAULT_INSTANCE_TOY_BATCH_SIZE,
    )
    parser.add_argument(
        "--toy-num-workers",
        type=int,
        default=defaults.DEFAULT_INSTANCE_TOY_NUM_WORKERS,
    )
    parser.add_argument(
        "--toy-learning-rate",
        type=float,
        default=defaults.DEFAULT_INSTANCE_LEARNING_RATE,
    )
    parser.add_argument(
        "--toy-weight-decay",
        type=float,
        default=defaults.DEFAULT_INSTANCE_WEIGHT_DECAY,
    )
    parser.add_argument("--toy-normalize-inputs", action="store_true")
    parser.add_argument(
        "--toy-architecture",
        choices=defaults.TOY_INSTANCE_ARCHITECTURE_CHOICES,
        default=defaults.DEFAULT_INSTANCE_TOY_ARCHITECTURE,
    )
    parser.add_argument("--disable-toy-gradient-clipping", action="store_true")

    parser.add_argument(
        "--graha-stats-batch-size",
        type=int,
        default=defaults.DEFAULT_GRAHA_STATS_BATCH_SIZE,
    )
    parser.add_argument(
        "--graha-batch-size",
        type=int,
        default=defaults.DEFAULT_GRAHA_INSTANCE_BATCH_SIZE,
    )
    parser.add_argument(
        "--graha-num-workers", type=int, default=defaults.DEFAULT_GRAHA_NUM_WORKERS
    )
    parser.add_argument(
        "--graha-backbone-lr", type=float, default=defaults.DEFAULT_GRAHA_BACKBONE_LR
    )
    parser.add_argument(
        "--graha-head-lr", type=float, default=defaults.DEFAULT_GRAHA_HEAD_LR
    )
    parser.add_argument(
        "--graha-layer-decay", type=float, default=defaults.DEFAULT_GRAHA_LAYER_DECAY
    )
    parser.add_argument(
        "--graha-weight-decay", type=float, default=defaults.DEFAULT_GRAHA_WEIGHT_DECAY
    )
    parser.add_argument(
        "--graha-warmup-steps", type=int, default=defaults.DEFAULT_GRAHA_WARMUP_STEPS
    )
    _add_graha_anchor_args(parser, parsed_defaults=False)
    parser.add_argument(
        "--graha-score-threshold",
        type=float,
        default=defaults.DEFAULT_GRAHA_SCORE_THRESHOLD,
    )
    parser.add_argument(
        "--prediction-score-threshold",
        type=float,
        default=defaults.DEFAULT_PREDICTION_SCORE_THRESHOLD,
    )
    parser.add_argument(
        "--progress-log-every-n-batches",
        type=int,
        default=defaults.DEFAULT_PIPELINE_PROGRESS_LOG_EVERY_N_BATCHES,
    )
    parser.add_argument(
        "--mask-shift", type=int, nargs=2, default=defaults.DEFAULT_MASK_SHIFT
    )

    parser.add_argument(
        "--comparison-extra-arg",
        action="append",
        default=[],
        help="Extra raw argument token for the comparison command. Repeat for multiple tokens.",
    )
    parser.add_argument(
        "--sweep-extra-arg",
        action="append",
        default=[],
        help="Extra raw argument token for the checkpoint sweep command. Repeat for multiple tokens.",
    )
    return parser


def parse_checkpoint_pipeline_args(
    *,
    description: str | None = None,
) -> argparse.Namespace:
    return create_checkpoint_pipeline_parser(description).parse_args()


def create_semantic_checkpoint_sweep_parser(
    description: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    _add_symlink_arg(parser, semantic_help=False)
    _add_data_root_args(parser, output_arg="--output-root", checkpoint_dirs=True)
    _add_model_selection_arg(parser)
    _add_dataset_modality_arg(parser)
    _add_band_filter_arg(parser)
    parser.add_argument("--target-size", type=int, default=defaults.DEFAULT_TARGET_SIZE)
    _add_semantic_label_source_arg(parser, detailed_help=False)
    _add_matching_args(parser, image_suffix_default=defaults.DEFAULT_IMAGE_SUFFIX)
    parser.add_argument(
        "--batch-size", type=int, default=defaults.DEFAULT_SEMANTIC_BATCH_SIZE
    )
    parser.add_argument(
        "--num-workers", type=int, default=defaults.DEFAULT_SEMANTIC_NUM_WORKERS
    )
    parser.add_argument("--normalize-inputs", action="store_true")
    _add_normalization_args_without_help(parser)
    parser.add_argument("--max-test-samples", type=int, default=None)
    _add_nodata_args(parser, semantic_help=True)
    parser.add_argument("--dino-checkpoint", type=str, default=None)
    _add_graha_input_args(parser)
    parser.add_argument(
        "--graha-stats-batch-size",
        type=int,
        default=defaults.DEFAULT_GRAHA_STATS_BATCH_SIZE,
    )
    parser.add_argument(
        "--graha-batch-size",
        type=int,
        default=defaults.DEFAULT_GRAHA_SEMANTIC_BATCH_SIZE,
    )
    parser.add_argument(
        "--graha-num-workers", type=int, default=defaults.DEFAULT_GRAHA_NUM_WORKERS
    )
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--seed", type=int, default=defaults.DEFAULT_SEED)
    parser.add_argument(
        "--verbose", action="store_true", help="Show model/datamodule setup output."
    )
    parser.add_argument(
        "--no-preload-test-batches",
        dest="preload_test_batches",
        action="store_false",
        help="Disable one-time test dataloader preload and iterate the dataloader for every checkpoint.",
    )
    parser.set_defaults(preload_test_batches=True)
    return parser


def parse_semantic_checkpoint_sweep_args(
    *,
    description: str | None = None,
) -> argparse.Namespace:
    return create_semantic_checkpoint_sweep_parser(description).parse_args()


def create_instance_checkpoint_sweep_parser(
    description: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    _add_symlink_arg(parser, semantic_help=False)
    _add_data_root_args(parser, output_arg="--output-root", checkpoint_dirs=True)
    _add_model_selection_arg(parser)
    _add_dataset_modality_arg(parser)
    parser.add_argument("--target-size", type=int, default=defaults.DEFAULT_TARGET_SIZE)
    _add_band_filter_arg(parser)
    _add_matching_args(parser, image_suffix_default=defaults.DEFAULT_IMAGE_SUFFIX)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--toy-batch-size",
        "--batch-size",
        dest="toy_batch_size",
        type=int,
        default=defaults.DEFAULT_INSTANCE_TOY_BATCH_SIZE,
        help="Toy instance batch size. --batch-size is accepted for parity with semantic scripts.",
    )
    parser.add_argument(
        "--toy-num-workers",
        type=int,
        default=defaults.DEFAULT_INSTANCE_SWEEP_NUM_WORKERS,
    )
    parser.add_argument(
        "--toy-normalize-inputs",
        "--normalize-inputs",
        dest="toy_normalize_inputs",
        action="store_true",
        help="Enable Toy instance z-score normalization. --normalize-inputs is accepted for parity with semantic scripts.",
    )
    _add_normalization_args_without_help(parser)
    parser.add_argument(
        "--toy-architecture",
        choices=defaults.TOY_INSTANCE_ARCHITECTURE_CHOICES,
        default=defaults.DEFAULT_INSTANCE_TOY_ARCHITECTURE,
    )
    parser.add_argument("--dino-checkpoint", type=str, default=None)
    _add_graha_input_args(parser)
    parser.add_argument(
        "--graha-stats-batch-size",
        type=int,
        default=defaults.DEFAULT_GRAHA_STATS_BATCH_SIZE,
    )
    parser.add_argument(
        "--graha-batch-size",
        type=int,
        default=defaults.DEFAULT_GRAHA_INSTANCE_BATCH_SIZE,
    )
    parser.add_argument(
        "--graha-num-workers",
        type=int,
        default=defaults.DEFAULT_GRAHA_SWEEP_NUM_WORKERS,
    )
    parser.add_argument(
        "--graha-backbone-lr", type=float, default=defaults.DEFAULT_GRAHA_BACKBONE_LR
    )
    parser.add_argument(
        "--graha-head-lr", type=float, default=defaults.DEFAULT_GRAHA_HEAD_LR
    )
    parser.add_argument(
        "--graha-layer-decay", type=float, default=defaults.DEFAULT_GRAHA_LAYER_DECAY
    )
    parser.add_argument(
        "--graha-weight-decay", type=float, default=defaults.DEFAULT_GRAHA_WEIGHT_DECAY
    )
    parser.add_argument(
        "--graha-warmup-steps", type=int, default=defaults.DEFAULT_GRAHA_WARMUP_STEPS
    )
    _add_graha_anchor_args(parser, parsed_defaults=True)
    parser.add_argument(
        "--graha-score-threshold",
        type=float,
        default=defaults.DEFAULT_GRAHA_SCORE_THRESHOLD,
    )
    parser.add_argument(
        "--prediction-split",
        choices=defaults.SPLIT_CHOICES,
        default=defaults.DEFAULT_SWEEP_SPLIT,
    )
    parser.add_argument(
        "--prediction-score-threshold",
        type=float,
        default=defaults.DEFAULT_PREDICTION_SCORE_THRESHOLD,
    )
    parser.add_argument(
        "--mask-shift", type=int, nargs=2, default=defaults.DEFAULT_MASK_SHIFT
    )
    _add_nodata_args(parser, semantic_help=False)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--seed", type=int, default=defaults.DEFAULT_SEED)
    parser.add_argument("--verbose", action="store_true")
    return parser


def parse_instance_checkpoint_sweep_args(
    *,
    description: str | None = None,
) -> argparse.Namespace:
    return create_instance_checkpoint_sweep_parser(description).parse_args()


def create_instance_checkpoint_comparison_plot_parser(
    description: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--toy-checkpoint", type=Path, default=None)
    parser.add_argument("--toy-checkpoint-dir", type=Path, default=None)
    parser.add_argument(
        "--toy-plot-architecture",
        choices=defaults.TOY_INSTANCE_ARCHITECTURE_CHOICES,
        default=defaults.DEFAULT_INSTANCE_TOY_ARCHITECTURE,
    )
    parser.add_argument("--mask2former-checkpoint", type=Path, default=None)
    parser.add_argument("--mask2former-checkpoint-dir", type=Path, default=None)
    parser.add_argument("--toy-terratorch-checkpoint", type=Path, default=None)
    parser.add_argument("--toy-terratorch-checkpoint-dir", type=Path, default=None)
    parser.add_argument("--graha-checkpoint", type=Path, default=None)
    parser.add_argument("--graha-checkpoint-dir", type=Path, default=None)
    parser.add_argument("--gfft-checkpoint", type=Path, default=None)
    parser.add_argument("--gfft-checkpoint-dir", type=Path, default=None)
    parser.add_argument("--dino-checkpoint", type=Path, default=None)
    _add_dataset_modality_arg(parser)
    _add_graha_input_args(parser, path_type=Path)
    parser.add_argument("--target-size", type=int, default=defaults.DEFAULT_TARGET_SIZE)
    _add_band_filter_arg(parser)
    _add_matching_args(
        parser,
        image_suffix_default=defaults.DEFAULT_WAC_IMAGE_SUFFIX,
        label_glob_default=defaults.DEFAULT_INSTANCE_LABEL_GLOB,
        path_defaults=True,
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--batch-size", type=int, default=defaults.DEFAULT_INSTANCE_TOY_BATCH_SIZE
    )
    parser.add_argument(
        "--num-workers", type=int, default=defaults.DEFAULT_INSTANCE_SWEEP_NUM_WORKERS
    )
    parser.add_argument(
        "--graha-stats-batch-size",
        type=int,
        default=defaults.DEFAULT_GRAHA_STATS_BATCH_SIZE,
    )
    parser.add_argument(
        "--graha-batch-size",
        type=int,
        default=defaults.DEFAULT_GRAHA_INSTANCE_BATCH_SIZE,
    )
    parser.add_argument(
        "--graha-num-workers",
        type=int,
        default=defaults.DEFAULT_GRAHA_SWEEP_NUM_WORKERS,
    )
    parser.add_argument("--normalize-inputs", action="store_true")
    _add_normalization_args_without_help(parser)
    parser.add_argument(
        "--graha-backbone-lr", type=float, default=defaults.DEFAULT_GRAHA_BACKBONE_LR
    )
    parser.add_argument(
        "--graha-head-lr", type=float, default=defaults.DEFAULT_GRAHA_HEAD_LR
    )
    parser.add_argument(
        "--graha-layer-decay", type=float, default=defaults.DEFAULT_GRAHA_LAYER_DECAY
    )
    parser.add_argument(
        "--graha-weight-decay", type=float, default=defaults.DEFAULT_GRAHA_WEIGHT_DECAY
    )
    parser.add_argument(
        "--graha-warmup-steps", type=int, default=defaults.DEFAULT_GRAHA_WARMUP_STEPS
    )
    _add_graha_anchor_args(parser, parsed_defaults=True)
    parser.add_argument(
        "--graha-score-threshold",
        type=float,
        default=defaults.DEFAULT_GRAHA_SCORE_THRESHOLD,
    )
    parser.add_argument(
        "--prediction-split",
        choices=defaults.SPLIT_CHOICES,
        default=defaults.DEFAULT_PREDICTION_SPLIT,
    )
    parser.add_argument(
        "--n-samples", type=int, default=defaults.DEFAULT_INSTANCE_PREDICTION_N_SAMPLES
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=defaults.DEFAULT_PREDICTION_SCORE_THRESHOLD,
    )
    parser.add_argument(
        "--mask-shift", type=int, nargs=2, default=defaults.DEFAULT_MASK_SHIFT
    )
    parser.add_argument("--ignore-nodata-in-loss", action="store_true")
    parser.add_argument(
        "--nodata-ignore-index",
        type=int,
        default=defaults.DEFAULT_NODATA_IGNORE_INDEX,
    )
    parser.add_argument("--seed", type=int, default=defaults.DEFAULT_SEED)
    return parser


def parse_instance_checkpoint_comparison_plot_args(
    *,
    description: str | None = None,
) -> argparse.Namespace:
    return create_instance_checkpoint_comparison_plot_parser(description).parse_args()


def create_semantic_checkpoint_comparison_plot_parser(
    description: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--toy-checkpoint", type=Path, default=None)
    parser.add_argument("--toy-checkpoint-dir", type=Path, default=None)
    parser.add_argument("--graha-checkpoint", type=Path, default=None)
    parser.add_argument("--graha-checkpoint-dir", type=Path, default=None)
    parser.add_argument("--gfft-checkpoint", type=Path, default=None)
    parser.add_argument("--gfft-checkpoint-dir", type=Path, default=None)
    parser.add_argument("--dino-checkpoint", type=Path, default=None)
    _add_dataset_modality_arg(parser)
    _add_graha_input_args(parser, path_type=Path)
    parser.add_argument("--target-size", type=int, default=defaults.DEFAULT_TARGET_SIZE)
    _add_band_filter_arg(parser)
    _add_semantic_label_source_arg(parser, detailed_help=False)
    _add_matching_args(parser, image_suffix_default=defaults.DEFAULT_IMAGE_SUFFIX)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--batch-size", type=int, default=defaults.DEFAULT_SEMANTIC_BATCH_SIZE
    )
    parser.add_argument(
        "--num-workers", type=int, default=defaults.DEFAULT_SEMANTIC_NUM_WORKERS
    )
    parser.add_argument("--normalize-inputs", action="store_true")
    _add_normalization_args_without_help(parser)
    parser.add_argument(
        "--graha-shape-loss-weight",
        type=float,
        default=defaults.DEFAULT_GRAHA_SHAPE_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--graha-shape-loss-pad-frac",
        type=float,
        default=defaults.DEFAULT_GRAHA_SHAPE_LOSS_PAD_FRAC,
    )
    parser.add_argument(
        "--graha-stats-batch-size",
        type=int,
        default=defaults.DEFAULT_GRAHA_STATS_BATCH_SIZE,
    )
    parser.add_argument(
        "--graha-batch-size",
        type=int,
        default=defaults.DEFAULT_GRAHA_SEMANTIC_BATCH_SIZE,
    )
    parser.add_argument(
        "--graha-num-workers", type=int, default=defaults.DEFAULT_GRAHA_NUM_WORKERS
    )
    parser.add_argument(
        "--prediction-split",
        choices=defaults.SPLIT_CHOICES,
        default=defaults.DEFAULT_PREDICTION_SPLIT,
    )
    parser.add_argument(
        "--n-samples", type=int, default=defaults.DEFAULT_SEMANTIC_PREDICTION_N_SAMPLES
    )
    parser.add_argument("--ignore-nodata-in-loss", action="store_true")
    parser.add_argument(
        "--nodata-ignore-index",
        type=int,
        default=defaults.DEFAULT_NODATA_IGNORE_INDEX,
    )
    parser.add_argument("--seed", type=int, default=defaults.DEFAULT_SEED)
    return parser


def parse_semantic_checkpoint_comparison_plot_args(
    *,
    description: str | None = None,
) -> argparse.Namespace:
    return create_semantic_checkpoint_comparison_plot_parser(description).parse_args()
