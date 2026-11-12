"""Standard-library CLI definitions shared by preparation and Grace submission."""

import argparse
from pathlib import Path

DEFAULT_DATA_ROOT = Path(
    "/explore/nobackup/projects/lfm/model_inputs/300_300_inputs/"
    "fm_all_static_all_wac_iseg_v3"
)
DEFAULT_DINO_CHECKPOINT = Path(
    "/explore/nobackup/projects/lfm/model_weights/"
    "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
)
DEFAULT_GRAHA_PRETRAIN_DIR = Path(
    "/explore/nobackup/projects/lfm/ibm_model_pretrain_dir_v2"
)
TRAINING_OVERRIDES = {
    "toy_learning_rate",
    "toy_weight_decay",
    "toy_gradient_clip_val",
    "disable_toy_gradient_clipping",
    "graha_backbone_lr",
    "graha_head_lr",
    "graha_layer_decay",
    "graha_weight_decay",
    "graha_warmup_steps",
    "graha_anchor_sizes",
    "graha_anchor_aspect_ratios",
}


def positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


def create_paired_experiment_parser(
    *, require_run_root: bool = True
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare matched Toy/GRAHA 70-band input manifests (no training)."
    )
    parser.add_argument("--run-root", type=Path, required=require_run_root)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dino-checkpoint", type=Path, default=DEFAULT_DINO_CHECKPOINT)
    parser.add_argument(
        "--graha-pretrain-dir", type=Path, default=DEFAULT_GRAHA_PRETRAIN_DIR
    )
    parser.add_argument("--max-epochs", type=positive_integer, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--toy-batch-size", type=positive_integer, default=32)
    parser.add_argument("--graha-batch-size", type=positive_integer, default=32)
    parser.add_argument("--toy-num-workers", type=int, default=10)
    parser.add_argument("--graha-num-workers", type=int, default=10)
    parser.add_argument("--image-glob", default="*.tif")
    parser.add_argument("--label-glob", default="*_label.npz")
    parser.add_argument("--image-suffix")
    parser.add_argument("--label-suffix")
    parser.add_argument("--excluded-nodata-values", type=float, nargs="+")
    for name in sorted(
        TRAINING_OVERRIDES
        - {
            "disable_toy_gradient_clipping",
            "graha_anchor_sizes",
            "graha_anchor_aspect_ratios",
        }
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=int if name == "graha_warmup_steps" else float,
            default=argparse.SUPPRESS,
        )
    return parser
