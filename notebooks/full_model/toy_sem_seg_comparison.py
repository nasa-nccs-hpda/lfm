"""Train the toy DINO semantic segmentation model on split full-model data."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from lfm.toy_model.sem_seg.lightning_wrappers.toy_sem_seg_datamodule import (
    ToySemSegSplitDataModule,
)
from lfm.toy_model.sem_seg.lightning_wrappers.toy_sem_seg_lightning import (
    ToySemSegLightningModule,
)
from lfm.full_model.utils import create_timestamped_output_dir
from lfm.full_model.utils.utils import ensure_data_symlink
from lfm.toy_model.sem_seg.sseg_model import DINOSegmentation, load_dinov3_encoder


@dataclass(frozen=True)
class ToyComparisonConfig:
    repo_root: Path
    notebook_dir: Path
    data_root: Path
    base_output_dir: Path
    dino_checkpoint: Path | None
    band_filter: list[int]
    target_size: tuple[int, int]
    spatial_transform: str
    max_train_samples: int | None
    max_val_samples: int | None
    max_test_samples: int | None
    batch_size: int
    num_workers: int
    max_epochs: int
    learning_rate: float
    weight_decay: float
    loss_type: str
    freeze_encoder: bool
    normalize_inputs: bool
    seed: int


def build_config(args: argparse.Namespace) -> ToyComparisonConfig:
    notebook_dir = Path(__file__).resolve().parent
    repo_root = notebook_dir.parents[1]
    data_root = Path(args.data_root).resolve() if args.data_root else notebook_dir / "data"
    base_output_dir = (
        Path(args.base_output_dir).resolve()
        if args.base_output_dir
        else notebook_dir / "outputs" / "toy_sem_seg_comparison"
    )
    dino_checkpoint = Path(args.dino_checkpoint).resolve() if args.dino_checkpoint else None

    return ToyComparisonConfig(
        repo_root=repo_root,
        notebook_dir=notebook_dir,
        data_root=data_root,
        base_output_dir=base_output_dir,
        dino_checkpoint=dino_checkpoint,
        band_filter=args.band_filter,
        target_size=(args.target_size, args.target_size),
        spatial_transform=args.spatial_transform,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss_type=args.loss_type,
        freeze_encoder=args.freeze_encoder,
        normalize_inputs=False,
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
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required split data paths:\n" + "\n".join(str(path) for path in missing)
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


def create_datamodule(config: ToyComparisonConfig, output_dir: Path) -> ToySemSegSplitDataModule:
    datamodule = ToySemSegSplitDataModule(
        data_root=config.data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        target_size=config.target_size,
        spatial_transform=config.spatial_transform,
        band_filter=config.band_filter,
        normalize_inputs=config.normalize_inputs,
        max_train_samples=config.max_train_samples,
        max_val_samples=config.max_val_samples,
        max_test_samples=config.max_test_samples,
        output_dir=output_dir,
    )
    datamodule.setup("fit")
    print("Data sanity summary:")
    for key, value in datamodule.get_sanity_summary().items():
        print(f"  {key}: {value}")
    return datamodule


def create_model(config: ToyComparisonConfig, weight_assignments: list[str]) -> DINOSegmentation:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.dino_checkpoint is not None:
        encoder = load_dinov3_encoder(
            weights_local_checkpoint=str(config.dino_checkpoint),
            device=device,
        )
    else:
        encoder = load_dinov3_encoder(device=device)

    return DINOSegmentation(
        encoder=encoder,
        num_classes=2,
        img_size=config.target_size,
        freeze_encoder=config.freeze_encoder,
        weight_assignments=weight_assignments,
    )


def create_lightning_module(
    config: ToyComparisonConfig,
    model: DINOSegmentation,
) -> ToySemSegLightningModule:
    return ToySemSegLightningModule(
        model=model,
        loss_type=config.loss_type,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_epochs=config.max_epochs,
    )


def create_trainer(config: ToyComparisonConfig, output_dir: Path) -> Trainer:
    print("Creating Lightning trainer...", flush=True)
    return Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="32",
        max_epochs=config.max_epochs,
        check_val_every_n_epoch=1,
        log_every_n_steps=5,
        logger=TensorBoardLogger(
            save_dir=str(output_dir / "tb_logs"),
            name="toy-sem-seg-comparison",
        ),
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(
                dirpath=str(output_dir / "checkpoints"),
                monitor="val/loss",
                mode="min",
                filename="best-{epoch:02d}-{val/loss:.3f}",
                save_top_k=3,
                save_last=True,
            ),
        ],
    )


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--band-filter", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--spatial-transform", choices=["resize", "crop"], default="crop")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--loss-type", type=str, default="focal_dice")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fit", action="store_true", help="Build data/model/trainer but skip fit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notebook_dir = Path(__file__).resolve().parent
    ensure_data_symlink(args.simlink_dest, notebook_dir / "data")
    config = build_config(args)
    validate_data_paths(config)
    output_dir = create_timestamped_output_dir(config.base_output_dir)
    save_config(config, output_dir)

    seed_everything(config.seed)
    datamodule = create_datamodule(config, output_dir)
    if datamodule.weight_assignments is None:
        raise RuntimeError("DataModule did not create weight assignments.")

    model = create_model(config, datamodule.weight_assignments)
    task = create_lightning_module(config, model)
    trainer = create_trainer(config, output_dir)
    print("Lightning trainer created.", flush=True)

    if args.no_fit:
        print("Skipping trainer.fit() because --no-fit was set.")
        return
    print("Starting trainer.fit()...", flush=True)
    trainer.fit(task, datamodule=datamodule)
    print("trainer.fit() complete. Starting trainer.test()...", flush=True)
    trainer.test(task, datamodule=datamodule, ckpt_path="best")
    print("trainer.test() complete.", flush=True)


if __name__ == "__main__":
    main()
