"""Matched 70-band Toy/GRAHA inputs; no training or scheduler side effects."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lfm.all_models.all_tasks.data.image_io import PairRecord, find_pair_records
from lfm.all_models.all_tasks.experiments import json_ready
from lfm.all_models.inst_seg.config import (
    InstanceSegmentationExperimentConfig,
    build_config,
)
from lfm.all_models.inst_seg.paired_experiment_cli import (
    DEFAULT_DATA_ROOT,
    DEFAULT_DINO_CHECKPOINT,
    DEFAULT_GRAHA_PRETRAIN_DIR,
    TRAINING_OVERRIDES,
)
from lfm.all_models.inst_seg.paired_experiment_cli import (
    create_paired_experiment_parser as create_paired_experiment_parser,  # noqa: PLC0414 -- public re-export
)
from lfm.all_models.inst_seg.paired_experiment_cli import (
    positive_integer as positive_integer,  # noqa: PLC0414 -- public re-export
)

SPLITS = ("train", "val", "test")


def _identity(payload: Any) -> str:
    """Identify small configuration/manifest contents, never raster or weight data."""
    encoded = json.dumps(json_ready(payload), sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def band_layout() -> list[dict[str, Any]]:
    """Explicit stored indices; static identities are positional, not guessed names."""
    return [
        {"stored_index": start + index, "modality": modality, "modality_index": index}
        for modality, start, count in (("vis", 0, 5), ("uv", 5, 2), ("static", 7, 63))
        for index in range(count)
    ]


@dataclass(frozen=True)
class PairedInstanceExperiment:
    toy: InstanceSegmentationExperimentConfig
    graha: InstanceSegmentationExperimentConfig

    def __post_init__(self) -> None:
        for config in (self.toy, self.graha):
            if config.run_epoch_test_suite:
                raise ValueError(
                    "Paired training must not evaluate the held-out test split each epoch."
                )
            if config.band_filter != list(range(70)) or config.target_size != 256:
                raise ValueError(
                    "This baseline requires all 70 ordered bands and 256 crops."
                )
            if (
                config.graha_backend_modalities != ["vis", "uv", "static"]
                or config.graha_input_modality_mode != "vis-uv-static"
            ):
                raise ValueError("GRAHA must retain native vis/uv/static embeddings.")
            if config.toy_architecture != "dino-mask-rcnn":
                raise ValueError("The paired baseline requires Toy DINO Mask R-CNN.")
            if (
                isinstance(config.max_epochs, bool)
                or not isinstance(config.max_epochs, int)
                or config.max_epochs < 1
            ):
                raise ValueError("max_epochs must be a positive integer.")
        common_fields = (
            "data_root",
            "seed",
            "max_epochs",
            "band_filter",
            "target_size",
            "image_glob",
            "label_glob",
            "image_suffix",
            "label_suffix",
            "mask_shift",
            "ignore_nodata_in_loss",
            "nodata_ignore_index",
            "excluded_nodata_values",
            "image_nodata_policy",
            "max_train_samples",
            "max_val_samples",
            "max_test_samples",
        )
        if any(
            getattr(self.toy, key) != getattr(self.graha, key) for key in common_fields
        ):
            raise ValueError(
                "The model configurations must share dataset membership and geometry."
            )
        if self.toy.normalization_source != "finetune" or self.toy.toy_normalize_inputs:
            raise ValueError(
                "Toy must retain per-chip min/max without z-score normalization."
            )
        if (
            self.graha.normalization_source != "pretrain"
            or self.graha.normalization_modality != "vis_uv_static"
        ):
            raise ValueError(
                "GRAHA must use its own VIS/UV/static pretraining statistics."
            )
        if self.toy.dino_checkpoint is None or self.graha.graha_pretrain_dir is None:
            raise ValueError(
                "Both models require explicit pretrained-weight locations."
            )

    def payload(self) -> dict[str, Any]:
        return json_ready(
            {
                "schema_version": 1,
                "band_layout": band_layout(),
                "display_bands": [3, 1, 0],
                "models": {"toy": asdict(self.toy), "graha": asdict(self.graha)},
                "initialization": {
                    "toy": {
                        "checkpoint": self.toy.dino_checkpoint,
                        "method": "DINO spectral kernel mapping from chip band metadata",
                    },
                    "graha": {
                        "checkpoint": self.graha.graha_pretrain_dir
                        / "checkpoints/checkpoint_weights_final.pt",
                        "backbone_config": self.graha.graha_pretrain_dir
                        / "full_config.yaml",
                        "modality_info": self.graha.graha_pretrain_dir
                        / "modality_info.yaml",
                        "backend_modalities": ["vis", "uv", "static"],
                        "new_modalities": None,
                    },
                },
                "normalization": {
                    "toy": {"method": "per_chip_per_band_minmax", "zscore": False},
                    "graha": {
                        "method": "pretraining_zscore",
                        "modality_info": self.graha.graha_pretrain_dir
                        / "modality_info.yaml",
                    },
                },
                "feature_pyramids": {
                    "toy": {
                        "layers": [5, 11, 17, 23],
                        "strides": [8, 16, 32, 64],
                        "method": "DINO projections and interpolation",
                    },
                    "graha": {
                        "layers": [2, 5, 8, 11],
                        "token_grid_height": 32,
                        "method": "LearnedInterpolateToPyramidal + FeaturePyramidNetworkNeck",
                    },
                },
            }
        )

    @property
    def config_id(self) -> str:
        return _identity(self.payload())


def resolve_paired_experiment(
    *,
    run_root: str | Path,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    dino_checkpoint: str | Path = DEFAULT_DINO_CHECKPOINT,
    graha_pretrain_dir: str | Path = DEFAULT_GRAHA_PRETRAIN_DIR,
    max_epochs: int = 50,
    seed: int = 42,
    toy_batch_size: int = 32,
    graha_batch_size: int = 32,
    toy_num_workers: int = 10,
    graha_num_workers: int = 10,
    image_glob: str = "*.tif",
    label_glob: str = "*_label.npz",
    image_suffix: str | None = None,
    label_suffix: str | None = None,
    excluded_nodata_values: list[float] | None = None,
    **training_overrides: Any,
) -> PairedInstanceExperiment:
    """Expand the fixed baseline without changing legacy CLI defaults or reading pixels."""
    unknown = set(training_overrides) - TRAINING_OVERRIDES
    if unknown:
        raise TypeError(
            f"Unsupported baseline override(s): {', '.join(sorted(unknown))}"
        )
    common = dict(
        data_root=data_root,
        dino_checkpoint=dino_checkpoint,
        graha_pretrain_dir=graha_pretrain_dir,
        toy_architecture="dino-mask-rcnn",
        dataset_modality="wac_static",
        band_filter=list(range(70)),
        graha_input_modality_mode="vis-uv-static",
        graha_backend_modalities=["vis", "uv", "static"],
        normalization_modality="vis_uv_static",
        toy_normalize_inputs=False,
        toy_freeze_backbone=False,
        graha_freeze_backbone=False,
        target_size=256,
        toy_batch_size=toy_batch_size,
        graha_batch_size=graha_batch_size,
        toy_num_workers=toy_num_workers,
        graha_num_workers=graha_num_workers,
        max_epochs=max_epochs,
        seed=seed,
        max_train_samples=None,
        max_val_samples=None,
        max_test_samples=None,
        image_glob=image_glob,
        label_glob=label_glob,
        image_suffix=image_suffix,
        label_suffix=label_suffix,
        ignore_nodata_in_loss=True,
        nodata_ignore_index=-1,
        image_nodata_policy="per_band",
        excluded_nodata_values=excluded_nodata_values,
        run_epoch_test_suite=False,
        **training_overrides,
    )
    run_root = Path(run_root).expanduser().resolve()
    return PairedInstanceExperiment(
        toy=build_config(
            base_output_dir=run_root / "toy", normalization_source="finetune", **common
        ),
        graha=build_config(
            base_output_dir=run_root / "graha",
            normalization_source="pretrain",
            **common,
        ),
    )


@dataclass(frozen=True)
class InstanceSplitManifest:
    data_root: Path
    splits: dict[str, list[PairRecord]]

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "data_root": str(self.data_root),
            "splits": {
                split: [
                    {
                        "chip_id": record.chip_id,
                        "image": str(record.image_path.relative_to(self.data_root)),
                        "label": str(record.label_path.relative_to(self.data_root)),
                    }
                    for record in self.splits[split]
                ]
                for split in SPLITS
            },
        }

    @property
    def manifest_id(self) -> str:
        return _identity(self.payload())


def build_split_manifest(experiment: PairedInstanceExperiment) -> InstanceSplitManifest:
    """Pair filenames only; pixel eligibility is handled during later sample loading."""
    config = experiment.toy
    splits = {}
    seen_images: set[Path] = set()
    for split in SPLITS:
        records = find_pair_records(
            config.data_root / split / "chips",
            config.data_root / split / "labels",
            image_glob=config.image_glob,
            label_glob=config.label_glob,
            image_suffix=config.image_suffix,
            label_suffix=config.label_suffix,
            require_all_labels=True,
        )
        splits[split] = []
        for record in records:
            if record.image_path.resolve() in seen_images:
                raise ValueError(
                    f"Repeated image across split membership: {record.image_path}"
                )
            seen_images.add(record.image_path.resolve())
            chip_id = f"{split}/{record.image_path.name}"
            splits[split].append(
                PairRecord(record.image_path, record.label_path, chip_id)
            )
    return InstanceSplitManifest(config.data_root, splits)


def prepare_experiment(
    experiment: PairedInstanceExperiment, run_root: str | Path
) -> None:
    """Persist authoritative small manifests; never load models or scan raster data."""
    root = Path(run_root).expanduser().resolve()
    if (
        experiment.toy.base_output_dir != root / "toy"
        or experiment.graha.base_output_dir != root / "graha"
    ):
        raise ValueError("Output root must match the resolved model configurations.")
    if any(
        (root / name).exists() for name in ("experiment.json", "split_manifest.json")
    ):
        raise FileExistsError(
            "Experiment manifests already exist; load them or choose a fresh run root."
        )
    manifest = build_split_manifest(experiment)
    root.mkdir(parents=True, exist_ok=True)
    payloads = {
        "split_manifest.json": {
            **manifest.payload(),
            "manifest_id": manifest.manifest_id,
        },
        "experiment.json": {
            **experiment.payload(),
            "config_id": experiment.config_id,
            "split_manifest_id": manifest.manifest_id,
        },
    }
    for name, payload in payloads.items():
        with (root / name).open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")


def load_experiment(
    run_root: str | Path,
) -> tuple[PairedInstanceExperiment, InstanceSplitManifest]:
    """Reload persisted membership without rediscovering files in either model job."""
    root = Path(run_root)
    payload = json.loads((root / "experiment.json").read_text())
    membership = json.loads((root / "split_manifest.json").read_text())
    if payload["schema_version"] != 1 or membership["schema_version"] != 1:
        raise ValueError("Unsupported paired experiment schema.")
    path_fields = {
        "notebook_dir",
        "lfm_root",
        "data_root",
        "base_output_dir",
        "dino_checkpoint",
        "toy_lightning_checkpoint",
        "graha_pretrain_dir",
        "graha_lightning_checkpoint",
        "gfft_config_path",
        "gfft_backbone_checkpoint",
    }
    configs = {}
    for model in ("toy", "graha"):
        values = dict(payload["models"][model])
        for name in path_fields:
            if values[name] is not None:
                values[name] = Path(values[name])
        values["mask_shift"] = tuple(values["mask_shift"])
        configs[model] = InstanceSegmentationExperimentConfig(**values)
    experiment = PairedInstanceExperiment(**configs)
    data_root = Path(membership["data_root"])
    manifest = InstanceSplitManifest(
        data_root,
        {
            split: [
                PairRecord(
                    data_root / item["image"],
                    data_root / item["label"],
                    item["chip_id"],
                )
                for item in membership["splits"][split]
            ]
            for split in SPLITS
        },
    )
    if (
        payload["config_id"]
        != _identity(
            {
                key: value
                for key, value in payload.items()
                if key not in {"config_id", "split_manifest_id"}
            }
        )
        or payload["config_id"] != experiment.config_id
        or membership["manifest_id"] != manifest.manifest_id
        or payload["split_manifest_id"] != manifest.manifest_id
        or data_root != experiment.toy.data_root
    ):
        raise ValueError(
            "Experiment configuration and split manifest identities do not match."
        )
    return experiment, manifest


def create_experiment_datamodule(
    experiment: PairedInstanceExperiment,
    manifest: InstanceSplitManifest,
    model: str,
    *,
    include_raw_image: bool = False,
    initial_epoch: int = 0,
    configuration: InstanceSegmentationExperimentConfig | None = None,
    batch_size: int | None = None,
    eligibility_dir: Path | None = None,
):
    """Use the real model-specific data adapters with one saved membership list."""
    if model not in {"toy", "graha"}:
        raise ValueError(f"Unsupported model: {model}")
    config = configuration or getattr(experiment, model)
    if manifest.data_root != config.data_root:
        raise ValueError("Split manifest belongs to another dataset.")
    common = {
        "data_root": config.data_root,
        "image_glob": config.image_glob,
        "label_glob": config.label_glob,
        "image_suffix": config.image_suffix,
        "label_suffix": config.label_suffix,
        "band_filter": config.band_filter,
        "split_records": manifest.splits,
        "include_raw_image": include_raw_image,
        "skip_invalid_chips": True,
        "skip_report_dir": eligibility_dir or config.base_output_dir / "eligibility",
        "no_data_replace": 0.0,
        "ignore_nodata_in_loss": config.ignore_nodata_in_loss,
        "nodata_ignore_index": config.nodata_ignore_index,
        "excluded_nodata_values": config.excluded_nodata_values,
        "image_nodata_policy": config.image_nodata_policy,
        "mask_shift": config.mask_shift,
    }
    if model == "toy":
        from lfm.toy_model.inst_seg.lightning_wrappers.toy_dino_mask_rcnn_datamodule import (
            ToyDinoMaskRCNNSplitDataModule,
        )

        datamodule = ToyDinoMaskRCNNSplitDataModule(
            **common,
            batch_size=batch_size or config.toy_batch_size,
            num_workers=config.toy_num_workers,
            target_size=config.target_size,
            normalize_inputs=False,
            scale_inputs=True,
        )
    else:
        from lfm.all_models.all_tasks.data.normalization import (
            load_terramind_pretraining_stats,
        )
        from lfm.full_model.inst_seg.graha_instance_datamodule import (
            GrahaObjectDetectionInstanceDataModule,
        )

        means, stds = load_terramind_pretraining_stats(
            config.graha_pretrain_dir / "modality_info.yaml",
            normalization_modality=config.normalization_modality,
            band_filter=config.band_filter,
        )
        datamodule = GrahaObjectDetectionInstanceDataModule(
            **common,
            batch_size=batch_size or config.graha_batch_size,
            num_workers=config.graha_num_workers,
            crop_size=config.target_size,
            means=means,
            stds=stds,
        )
    datamodule.paired_sampler_seed = config.seed
    datamodule.paired_sampler_epoch = initial_epoch
    datamodule.setup()
    for split in SPLITS:
        dataset = getattr(datamodule, f"{split}_dataset")
        if dataset.num_channels != 70:
            raise ValueError(
                f"Expected 70 stored bands in {split}, got {dataset.num_channels}."
            )
    return datamodule
