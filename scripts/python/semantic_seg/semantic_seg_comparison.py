"""Train Toy and Graha semantic segmentation models on split full-model data."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

from lfm.full_model.all_tasks.utils import (
    create_timestamped_output_dir,
    evaluate_prediction_caches,
    plot_prediction_cache_comparison,
)
from lfm.full_model.all_tasks.utils.utils import ensure_data_symlink
from lfm.full_model.sem_seg.semantic_model_adapter import GrahaSemanticModelAdapter
from lfm.toy_model.sem_seg.semantic_model_adapter import ToySemanticModelAdapter

TOY_ADAPTER = ToySemanticModelAdapter()
GRAHA_ADAPTER = GrahaSemanticModelAdapter()


@dataclass(frozen=True)
class ToyComparisonConfig:
    repo_root: Path
    notebook_dir: Path
    data_root: Path
    base_output_dir: Path
    dino_checkpoint: Path | None
    toy_lightning_checkpoint: Path | None
    band_filter: list[int]
    target_size: tuple[int, int]
    spatial_transform: str
    semantic_label_source: str
    image_glob: str
    label_glob: str
    image_suffix: str
    label_suffix: str
    image_file_type: str
    max_train_samples: int | None
    max_val_samples: int | None
    max_test_samples: int | None
    ignore_nodata_in_loss: bool
    nodata_ignore_index: int
    batch_size: int
    num_workers: int
    max_epochs: int
    learning_rate: float
    weight_decay: float
    loss_type: str
    use_toy_shape_loss: bool
    toy_shape_loss_weight: float
    toy_shape_loss_pad_frac: float
    freeze_encoder: bool
    normalize_inputs: bool
    normalization_source: str
    normalization_modality: str
    toy_gradient_clip_val: float | None
    plot_every_n_epochs: int
    plot_n_samples: int
    cache_predictions: bool
    prediction_split: str
    prediction_n_samples: int
    graha_base_output_dir: Path
    graha_pretrain_dir: Path | None
    graha_lightning_checkpoint: Path | None
    graha_input_modality_mode: str
    graha_vis_uv_merge_method: str
    graha_shape_loss_weight: float
    graha_shape_loss_pad_frac: float
    graha_stats_batch_size: int
    graha_batch_size: int
    graha_num_workers: int
    progress_log_every_n_batches: int
    skip_toy_fit: bool
    skip_graha_fit: bool
    run_epoch_test_suite: bool
    epoch_test_split: str
    epoch_test_n_samples: int
    epoch_test_every_n_epochs: int
    seed: int


def _file_type_from_glob(pattern: str) -> str:
    stripped = pattern.replace("*", "")
    if stripped.startswith(".") and stripped.count(".") == 1:
        return stripped
    suffix = Path(stripped).suffix
    return suffix or ".tif"


SEMANTIC_EPOCH_TEST_METRICS = [
    "pixel_accuracy",
    "foreground_precision",
    "foreground_recall",
    "foreground_f1",
    "iou",
    "predicted_foreground_fraction",
    "ground_truth_foreground_fraction",
]


def _semantic_metric_array(metrics: dict[str, float]) -> np.ndarray:
    row = np.zeros((), dtype=[(name, "f8") for name in SEMANTIC_EPOCH_TEST_METRICS])
    for name in SEMANTIC_EPOCH_TEST_METRICS:
        row[name] = float(metrics[name])
    return row


def _semantic_counts(
    pred: np.ndarray,
    label: np.ndarray,
    *,
    ignore_index: int | None = None,
) -> dict[str, float]:
    pred_flat = pred.reshape(-1)
    label_flat = label.reshape(-1)
    if ignore_index is not None:
        valid = label_flat != int(ignore_index)
        pred_flat = pred_flat[valid]
        label_flat = label_flat[valid]
    pred_bool = pred_flat.astype(bool)
    label_bool = label_flat.astype(bool)
    return {
        "tp": float(np.sum(pred_bool & label_bool)),
        "fp": float(np.sum(pred_bool & ~label_bool)),
        "fn": float(np.sum(~pred_bool & label_bool)),
        "tn": float(np.sum(~pred_bool & ~label_bool)),
        "n": float(pred_bool.size),
        "pred_fg": float(np.sum(pred_bool)),
        "label_fg": float(np.sum(label_bool)),
    }


def _semantic_metrics(counts: dict[str, float]) -> dict[str, float]:
    eps = 1e-8
    tp, fp, fn, tn, n = (
        counts["tp"],
        counts["fp"],
        counts["fn"],
        counts["tn"],
        counts["n"],
    )
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return {
        "pixel_accuracy": float((tp + tn) / (tp + tn + fp + fn + eps)),
        "foreground_precision": float(precision),
        "foreground_recall": float(recall),
        "foreground_f1": float(2 * precision * recall / (precision + recall + eps)),
        "iou": float(tp / (tp + fp + fn + eps)),
        "predicted_foreground_fraction": float(counts["pred_fg"] / (n + eps)),
        "ground_truth_foreground_fraction": float(counts["label_fg"] / (n + eps)),
    }


def _write_semantic_metrics(
    output_dir: Path, metrics: dict[str, float], *, header: str
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "metrics.npy", _semantic_metric_array(metrics))
    with (output_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write(header.rstrip() + "\n")
        for name in SEMANTIC_EPOCH_TEST_METRICS:
            f.write(f"{name}: {metrics[name]:.8f}\n")


def _image_for_plot(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[0] >= 4:
        rgb = np.stack([image[3], image[1], image[0]], axis=-1)
    elif image.ndim == 3:
        rgb = np.moveaxis(image[: min(3, image.shape[0])], 0, -1)
        if rgb.shape[-1] == 1:
            rgb = rgb[..., 0]
    else:
        rgb = image
    arr = rgb.astype(np.float32)
    lo, hi = np.nanpercentile(arr, [2, 98])
    if hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def _plot_semantic_epoch_samples(
    samples: list[dict[str, np.ndarray]], save_path: Path
) -> None:
    if not samples:
        return
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    save_path.parent.mkdir(parents=True, exist_ok=True)
    n_cols = len(samples)
    fig, axes = plt.subplots(4, n_cols, figsize=(4 * n_cols, 14))
    if n_cols == 1:
        axes = axes.reshape(4, 1)
    cmap_pred = ListedColormap(["black", "yellow"])
    cmap_label = ListedColormap(["black", "red"])
    for col, sample in enumerate(samples):
        image = _image_for_plot(sample["image"])
        pred = sample["pred"]
        label = sample["label"]
        overlay = image.copy()
        if overlay.ndim == 2:
            overlay = np.repeat(overlay[..., None], 3, axis=-1)
        overlay[pred > 0] = [1.0, 1.0, 0.0]
        axes[0, col].imshow(image, cmap="gray" if image.ndim == 2 else None)
        axes[0, col].set_title(sample["sample_key"], fontsize=10)
        axes[1, col].imshow(pred, cmap=cmap_pred, vmin=0, vmax=1)
        axes[1, col].set_title("Prediction", fontsize=10)
        axes[2, col].imshow(overlay)
        axes[2, col].set_title("Overlay", fontsize=10)
        axes[3, col].imshow(label, cmap=cmap_label, vmin=0, vmax=1)
        axes[3, col].set_title("Ground Truth", fontsize=10)
        for row in range(4):
            axes[row, col].axis("off")
    fig.suptitle(
        "Epoch Test Suite Semantic Predictions", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved semantic epoch test-suite plot to {save_path}", flush=True)


def _sample_key_from_name(filename: str | None, index: int) -> str:
    if filename:
        return Path(str(filename)).name.split("_input", 1)[0].replace("_label", "")
    return f"sample_{index:04d}"


def _extract_semantic_batch(batch):
    if isinstance(batch, dict):
        return (
            batch["image"],
            batch["mask"],
            batch.get("filename", [None] * batch["image"].shape[0]),
        )
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        filenames = batch[2] if len(batch) > 2 else [None] * batch[0].shape[0]
        return batch[0], batch[1], filenames
    raise TypeError(f"Unsupported semantic batch type: {type(batch)}")


class SemanticEpochTestSuiteCallback(Callback):
    """Run a semantic test suite at epoch end and save arrays/metrics."""

    def __init__(
        self,
        *,
        output_dir: Path,
        model_name: str,
        split: str,
        n_samples: int,
        every_n_epochs: int,
        ignore_index: int | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.split = split
        self.n_samples = n_samples
        self.every_n_epochs = every_n_epochs
        self.ignore_index = ignore_index

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch + 1
        if self.every_n_epochs <= 0 or epoch % self.every_n_epochs != 0:
            return
        datamodule = trainer.datamodule
        datamodule.setup("fit" if self.split in {"train", "val"} else "test")
        dataloader = getattr(datamodule, f"{self.split}_dataloader")()
        device = pl_module.device
        was_training = pl_module.training
        pl_module.eval()
        epoch_dir = (
            self.output_dir / "test_suite" / self.model_name / f"epoch_{epoch:03d}"
        )
        total = {
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "tn": 0.0,
            "n": 0.0,
            "pred_fg": 0.0,
            "label_fg": 0.0,
        }
        saved = 0
        plot_samples = []
        with torch.no_grad():
            for batch in dataloader:
                images, labels, filenames = _extract_semantic_batch(batch)
                images = images.to(device)
                labels = labels.to(device)
                output = pl_module(images)
                logits = output.output if hasattr(output, "output") else output
                preds = (
                    logits.argmax(dim=1).long()
                    if logits.shape[1] > 1
                    else (torch.sigmoid(logits[:, 0]) > 0.5).long()
                )
                images_np = images.detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy()
                preds_np = preds.detach().cpu().numpy()
                for i in range(images_np.shape[0]):
                    if saved >= self.n_samples:
                        break
                    sample_key = _sample_key_from_name(
                        filenames[i] if i < len(filenames) else None, saved
                    )
                    sample_dir = epoch_dir / sample_key
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    np.save(sample_dir / f"{sample_key}_input.npy", images_np[i])
                    np.save(sample_dir / f"{sample_key}_label.npy", labels_np[i])
                    np.save(sample_dir / f"{sample_key}_pred.npy", preds_np[i])
                    if len(plot_samples) < 5:
                        plot_samples.append(
                            {
                                "sample_key": sample_key,
                                "image": images_np[i],
                                "label": labels_np[i],
                                "pred": preds_np[i],
                            }
                        )
                    counts = _semantic_counts(
                        preds_np[i],
                        labels_np[i],
                        ignore_index=self.ignore_index,
                    )
                    for key, value in counts.items():
                        total[key] += value
                    _write_semantic_metrics(
                        sample_dir,
                        _semantic_metrics(counts),
                        header=f"model: {self.model_name}\nepoch: {epoch}\nsample_key: {sample_key}",
                    )
                    saved += 1
                if saved >= self.n_samples:
                    break
        aggregate = _semantic_metrics(total)
        _write_semantic_metrics(
            epoch_dir,
            aggregate,
            header=f"model: {self.model_name}\nepoch: {epoch}\nsplit: {self.split}\nsamples: {saved}",
        )
        _plot_semantic_epoch_samples(
            plot_samples,
            epoch_dir / f"{self.split}_semantic_predictions.png",
        )
        pl_module.train(was_training)
        print(
            f"[{self.model_name}] epoch {epoch:03d} test suite: "
            f"F1={aggregate['foreground_f1']:.4f}, IoU={aggregate['iou']:.4f}, samples={saved}",
            flush=True,
        )


def build_config(args: argparse.Namespace) -> ToyComparisonConfig:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    notebook_dir = repo_root / "notebooks" / "full_model"
    scripts_output_dir = repo_root / "scripts" / "outputs"
    data_root = (
        Path(args.data_root).resolve() if args.data_root else notebook_dir / "data"
    )
    base_output_dir = (
        Path(args.base_output_dir).resolve()
        if args.base_output_dir
        else scripts_output_dir / "semantic_seg_comparison"
    )
    dino_checkpoint = (
        Path(args.dino_checkpoint).resolve() if args.dino_checkpoint else None
    )
    toy_lightning_checkpoint_arg = args.toy_lightning_checkpoint
    toy_lightning_checkpoint = (
        Path(toy_lightning_checkpoint_arg).resolve()
        if toy_lightning_checkpoint_arg
        else None
    )
    graha_base_output_dir = (
        Path(args.graha_base_output_dir).resolve()
        if args.graha_base_output_dir
        else scripts_output_dir / "graha_finetuning"
    )
    graha_pretrain_dir = (
        Path(args.graha_pretrain_dir).resolve() if args.graha_pretrain_dir else None
    )
    graha_lightning_checkpoint = (
        Path(args.graha_lightning_checkpoint).resolve()
        if args.graha_lightning_checkpoint
        else None
    )

    return ToyComparisonConfig(
        repo_root=repo_root,
        notebook_dir=notebook_dir,
        data_root=data_root,
        base_output_dir=base_output_dir,
        dino_checkpoint=dino_checkpoint,
        toy_lightning_checkpoint=toy_lightning_checkpoint,
        band_filter=args.band_filter,
        target_size=(args.target_size, args.target_size),
        spatial_transform=args.spatial_transform,
        semantic_label_source=args.semantic_label_source,
        image_glob=args.image_glob,
        label_glob=args.label_glob,
        image_suffix=args.image_suffix,
        label_suffix=args.label_suffix,
        image_file_type=_file_type_from_glob(args.image_glob),
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        ignore_nodata_in_loss=getattr(args, "ignore_nodata_in_loss", False),
        nodata_ignore_index=getattr(args, "nodata_ignore_index", -1),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss_type=args.loss_type,
        use_toy_shape_loss=args.use_toy_shape_loss,
        toy_shape_loss_weight=args.toy_shape_loss_weight,
        toy_shape_loss_pad_frac=args.toy_shape_loss_pad_frac,
        freeze_encoder=args.freeze_encoder,
        normalize_inputs=args.normalize_inputs,
        normalization_source=getattr(args, "normalization_source", "pretrain"),
        normalization_modality=getattr(args, "normalization_modality", "vis_uv"),
        toy_gradient_clip_val=(
            None if args.disable_toy_gradient_clipping else args.toy_gradient_clip_val
        ),
        plot_every_n_epochs=args.plot_every_n_epochs,
        plot_n_samples=args.plot_n_samples,
        cache_predictions=args.cache_predictions,
        prediction_split=args.prediction_split,
        prediction_n_samples=args.prediction_n_samples,
        graha_base_output_dir=graha_base_output_dir,
        graha_pretrain_dir=graha_pretrain_dir,
        graha_lightning_checkpoint=graha_lightning_checkpoint,
        graha_input_modality_mode=args.graha_input_modality_mode,
        graha_vis_uv_merge_method=args.graha_vis_uv_merge_method,
        graha_shape_loss_weight=getattr(args, "graha_shape_loss_weight", 0.05),
        graha_shape_loss_pad_frac=getattr(args, "graha_shape_loss_pad_frac", 0.3),
        graha_stats_batch_size=args.graha_stats_batch_size,
        graha_batch_size=args.graha_batch_size,
        graha_num_workers=args.graha_num_workers,
        progress_log_every_n_batches=getattr(args, "progress_log_every_n_batches", 25),
        skip_toy_fit=args.no_fit or args.skip_toy_fit,
        skip_graha_fit=args.no_fit or args.skip_graha_fit,
        run_epoch_test_suite=args.run_epoch_test_suite,
        epoch_test_split=args.epoch_test_split,
        epoch_test_n_samples=args.epoch_test_n_samples,
        epoch_test_every_n_epochs=args.epoch_test_every_n_epochs,
        seed=args.seed,
    )


def validate_data_paths(config: ToyComparisonConfig) -> None:
    required = []
    for split in ["train", "val", "test"]:
        required.extend(
            [
                config.data_root / split / "chips",
                config.data_root / split / "labels",
            ]
        )
    for checkpoint_path in [
        config.dino_checkpoint,
        config.toy_lightning_checkpoint,
        config.graha_pretrain_dir,
        config.graha_lightning_checkpoint,
    ]:
        if checkpoint_path is not None:
            required.append(checkpoint_path)
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required split data paths:\n"
            + "\n".join(str(path) for path in missing)
        )


def save_config(config: ToyComparisonConfig, output_dir: Path) -> None:
    def encode(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return list(value)
        return value

    payload = {key: encode(value) for key, value in asdict(config).items()}
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _format_seconds(seconds: float) -> str:
    whole_seconds = int(round(seconds))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def save_timing_summary(timing_rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not timing_rows:
        return

    json_path = output_dir / "timing_summary.json"
    csv_path = output_dir / "timing_summary.csv"
    fieldnames = ["model", "stage", "seconds", "elapsed_hms"]

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(timing_rows, f, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(timing_rows)

    print(f"Saved timing summary to {csv_path}")


def record_timing(
    timing_rows: list[dict[str, Any]],
    *,
    model: str,
    stage: str,
    started_at: float,
) -> None:
    elapsed = time.perf_counter() - started_at
    timing_rows.append(
        {
            "model": model,
            "stage": stage,
            "seconds": round(elapsed, 3),
            "elapsed_hms": _format_seconds(elapsed),
        }
    )
    print(
        f"[timing] {model} {stage}: {_format_seconds(elapsed)} ({elapsed:.3f}s)",
        flush=True,
    )


def get_toy_normalization_modality_info(config: ToyComparisonConfig) -> Path | None:
    if not config.normalize_inputs or config.normalization_source != "pretrain":
        return None
    normalization_config = GRAHA_ADAPTER.build_comparison_config(
        config,
        config.graha_base_output_dir,
    )
    return normalization_config.modality_info


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simlink-dest",
        "--symlink-dest",
        dest="simlink_dest",
        type=str,
        default=None,
        help=(
            "Optional source directory for notebooks/full_model/data. If ./data is already a "
            "symlink, it must point to this same directory."
        ),
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--base-output-dir", type=str, default=None)
    parser.add_argument("--dino-checkpoint", type=str, default=None)
    parser.add_argument(
        "--toy-lightning-checkpoint",
        type=str,
        default=None,
        help="Optional Toy Lightning .ckpt. Resumes fit, or loads weights when Toy fit is skipped.",
    )
    parser.add_argument(
        "--band-filter", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6]
    )
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument(
        "--spatial-transform", choices=["resize", "crop"], default="crop"
    )
    parser.add_argument(
        "--semantic-label-source",
        choices=["semantic", "instance"],
        default="semantic",
        help="Use .npy semantic labels or .npz instance labels converted to semantic masks.",
    )
    parser.add_argument(
        "--image-glob",
        default="*.tif",
        help="Chip filename glob inside each split/chips directory.",
    )
    parser.add_argument(
        "--label-glob",
        default="*_label.*",
        help="Label filename glob inside each split/labels directory.",
    )
    parser.add_argument(
        "--image-suffix",
        default="_input_wac_static_chip",
        help="Suffix stripped from chip stems before matching labels.",
    )
    parser.add_argument(
        "--label-suffix",
        default="_label",
        help="Suffix stripped from label stems before matching chips.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument(
        "--ignore-nodata-in-loss",
        action="store_true",
        help="Ignore TIFF nodata pixels in semantic segmentation loss and metrics.",
    )
    parser.add_argument(
        "--nodata-ignore-index",
        type=int,
        default=-1,
        help="Target label value used for ignored nodata pixels.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--loss-type", type=str, default="focal_dice")
    parser.add_argument("--use-toy-shape-loss", action="store_true")
    parser.add_argument("--toy-shape-loss-weight", type=float, default=0.05)
    parser.add_argument("--toy-shape-loss-pad-frac", type=float, default=0.3)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument(
        "--normalize-inputs",
        action="store_true",
        help="Enable Toy z-score normalization.",
    )
    parser.add_argument(
        "--normalization-source",
        choices=["pretrain", "finetune"],
        default="pretrain",
        help="When normalizing inputs, use TerraMind pretraining stats or finetuning train-split stats.",
    )
    parser.add_argument(
        "--normalization-modality",
        choices=["vis_uv", "nac"],
        default="vis_uv",
        help="Which modality family to use when --normalization-source=pretrain.",
    )
    parser.add_argument("--toy-gradient-clip-val", type=float, default=1.0)
    parser.add_argument(
        "--disable-toy-gradient-clipping",
        action="store_true",
        help="Disable Toy gradient clipping to match Graha's current trainer path.",
    )
    parser.add_argument("--plot-every-n-epochs", type=int, default=1)
    parser.add_argument("--plot-n-samples", type=int, default=5)
    parser.add_argument("--cache-predictions", action="store_true")
    parser.add_argument(
        "--prediction-split", choices=["train", "val", "test"], default="val"
    )
    parser.add_argument("--prediction-n-samples", type=int, default=20)
    parser.add_argument("--graha-base-output-dir", type=str, default=None)
    parser.add_argument("--graha-pretrain-dir", type=str, default=None)
    parser.add_argument(
        "--graha-lightning-checkpoint",
        type=str,
        default=None,
        help="Optional Graha Lightning .ckpt. Resumes fit, or loads weights when Graha fit is skipped.",
    )
    parser.add_argument(
        "--graha-input-modality-mode", choices=["new-wac", "vis-uv"], default="new-wac"
    )
    parser.add_argument(
        "--graha-vis-uv-merge-method", choices=["mean", "max"], default="mean"
    )
    parser.add_argument("--graha-shape-loss-weight", type=float, default=0.05)
    parser.add_argument("--graha-shape-loss-pad-frac", type=float, default=0.3)
    parser.add_argument("--graha-stats-batch-size", type=int, default=16)
    parser.add_argument("--graha-batch-size", type=int, default=16)
    parser.add_argument("--graha-num-workers", type=int, default=10)
    parser.add_argument(
        "--progress-log-every-n-batches",
        type=int,
        default=25,
        help="Flush train-batch progress every N batches in sbatch logs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-fit", action="store_true", help="Build data/model/trainer but skip fit."
    )
    parser.add_argument(
        "--skip-toy-fit", action="store_true", help="Skip only Toy fitting."
    )
    parser.add_argument(
        "--skip-graha-fit", action="store_true", help="Skip only Graha fitting."
    )
    parser.add_argument("--run-epoch-test-suite", action="store_true")
    parser.add_argument(
        "--epoch-test-split", choices=["train", "val", "test"], default="test"
    )
    parser.add_argument("--epoch-test-n-samples", type=int, default=100)
    parser.add_argument("--epoch-test-every-n-epochs", type=int, default=1)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    notebook_dir = Path(__file__).resolve().parents[2] / "notebooks" / "full_model"
    ensure_data_symlink(args.simlink_dest, notebook_dir / "data")
    config = build_config(args)
    validate_data_paths(config)
    output_dir = create_timestamped_output_dir(config.base_output_dir)
    save_config(config, output_dir)
    timing_rows: list[dict[str, Any]] = []

    toy_prediction_cache = TOY_ADAPTER.run_workflow(
        config,
        output_dir=output_dir,
        normalization_modality_info=get_toy_normalization_modality_info(config),
        epoch_test_suite_callback_cls=SemanticEpochTestSuiteCallback,
        timing_rows=timing_rows,
        record_timing=record_timing,
    )

    _, graha_prediction_cache = GRAHA_ADAPTER.run_workflow(
        config,
        no_fit=config.skip_graha_fit,
        comparison_output_dir=output_dir,
        epoch_test_suite_callback_cls=SemanticEpochTestSuiteCallback,
        timing_rows=timing_rows,
        record_timing=record_timing,
    )

    if config.cache_predictions and toy_prediction_cache and graha_prediction_cache:
        comparison_started_at = time.perf_counter()
        comparison_caches = {
            "toy": toy_prediction_cache,
            "graha": graha_prediction_cache,
        }
        plot_prediction_cache_comparison(
            comparison_caches,
            output_dir / "plots" / "comparison",
            n_samples=min(5, config.prediction_n_samples),
        )
        _, metric_summary = evaluate_prediction_caches(
            comparison_caches,
            output_dir / "comparison_metrics",
        )
        print("Comparison metric summary:")
        for row in metric_summary:
            print("  " + json.dumps(row, sort_keys=True))
        record_timing(
            timing_rows,
            model="Comparison",
            stage="plots_and_metrics",
            started_at=comparison_started_at,
        )

    save_timing_summary(timing_rows, output_dir)


if __name__ == "__main__":
    main()
