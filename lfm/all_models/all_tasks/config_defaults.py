"""Shared defaults for segmentation experiment configuration and CLIs."""

from __future__ import annotations

MODEL_CHOICES = ["toy", "graha"]
TASK_CHOICES = ["semantic", "instance"]
SPLIT_CHOICES = ["train", "val", "test"]
SEMANTIC_LABEL_SOURCE_CHOICES = ["auto", "semantic", "instance"]

DATASET_MODALITY_CHOICES = ["wac", "wac_static", "nac", "nac_dtm"]
GRAHA_INPUT_MODALITY_CHOICES = [
    "single",
    "vis",
    "uv",
    "static",
    "vis-uv",
    "vis-uv-static",
    "nac",
    "dtm",
    "nac-dtm",
]
GRAHA_BACKEND_MODALITY_CHOICES = ["wac", "nac", "dtm", "vis", "uv", "static"]
GRAHA_VIS_UV_MERGE_CHOICES = ["mean", "max", "concat"]
NORMALIZATION_SOURCE_CHOICES = ["pretrain", "finetune"]
NORMALIZATION_MODALITY_CHOICES = ["vis_uv", "vis_uv_static", "nac"]
NORMALIZATION_MODALITY_CLI_CHOICES = ["vis-uv", "vis-uv-static", "nac"]
IMAGE_NODATA_POLICY_CHOICES = ["union", "per_band", "preserve"]
TOY_INSTANCE_ARCHITECTURE_CHOICES = [
    "mask2former",
    "dino-mask-rcnn",
    "dino-terratorch-mask-rcnn",
]

DEFAULT_MODELS = ["toy", "graha"]
DEFAULT_WAC_BAND_FILTER = [0, 1, 2, 3, 4, 5, 6]
DEFAULT_WAC_STATIC_BAND_FILTER = list(range(70))
DEFAULT_NAC_BAND_FILTER = [0]
DEFAULT_NAC_DTM_BAND_FILTER = [0, 1]
DEFAULT_BAND_FILTER = DEFAULT_WAC_BAND_FILTER
DEFAULT_GRAHA_ANCHOR_SIZES = [[8], [16], [32], [64]]
DEFAULT_GRAHA_ANCHOR_ASPECT_RATIOS = [0.5, 1.0, 2.0]
DEFAULT_GRAHA_ANCHOR_SIZES_CSV = "8,16,32,64"
DEFAULT_GRAHA_ANCHOR_ASPECT_RATIOS_CSV = "0.5,1.0,2.0"

DEFAULT_IMAGE_GLOB = "*chip*.tif"
DEFAULT_LABEL_GLOB = "*label.*"
DEFAULT_INSTANCE_LABEL_GLOB = "*label*.npz"
DEFAULT_IMAGE_SUFFIX = None
DEFAULT_WAC_IMAGE_SUFFIX = None
DEFAULT_LABEL_SUFFIX = None
DEFAULT_SEMANTIC_LABEL_SOURCE = "auto"

DEFAULT_TARGET_SIZE = 256
DEFAULT_MAX_EPOCHS = 100
DEFAULT_SEED = 42
DEFAULT_IGNORE_NODATA_IN_LOSS = False
DEFAULT_NODATA_IGNORE_INDEX = -1
DEFAULT_IMAGE_NODATA_POLICY = "per_band"
DEFAULT_MASK_SHIFT = (0, 0)

DEFAULT_GRAHA_INPUT_MODALITY_MODE = "vis-uv"
DEFAULT_GRAHA_VIS_UV_MERGE_METHOD = "concat"
DEFAULT_DATASET_MODALITY = "wac"
DEFAULT_NORMALIZATION_SOURCE = "pretrain"
DEFAULT_NORMALIZATION_MODALITY = "vis_uv"

GRAHA_FRONTEND_TO_BACKEND_MODALITY = {
    "pho": "nac",
    "nac": "nac",
    "dtm": "dtm",
    "vis": "vis",
    "uv": "uv",
    "static": "static",
    "wac": "wac",
}

GRAHA_BACKEND_MODALITY_CHANNELS = {
    "wac": None,
    "nac": 1,
    "dtm": 1,
    "vis": 5,
    "uv": 2,
    "static": 63,
}

GRAHA_INPUT_MODE_TO_BACKEND_MODALITIES = {
    "vis": ["vis"],
    "uv": ["uv"],
    "static": ["static"],
    "vis-uv": ["vis", "uv"],
    "vis-uv-static": ["vis", "uv", "static"],
    "nac": ["nac"],
    "dtm": ["dtm"],
    "nac-dtm": ["nac", "dtm"],
}

DEFAULT_SEMANTIC_BATCH_SIZE = 16
DEFAULT_SEMANTIC_NUM_WORKERS = 10
DEFAULT_SEMANTIC_LEARNING_RATE = 5.0e-5
DEFAULT_SEMANTIC_WEIGHT_DECAY = 1.0e-3
DEFAULT_TOY_SEMANTIC_LOSS_TYPE = "focal_dice"
DEFAULT_SEMANTIC_PREDICTION_N_SAMPLES = 20

DEFAULT_INSTANCE_TOY_ARCHITECTURE = "mask2former"
DEFAULT_INSTANCE_TOY_BATCH_SIZE = 2
DEFAULT_INSTANCE_TOY_NUM_WORKERS = 10
DEFAULT_INSTANCE_SWEEP_NUM_WORKERS = 4
DEFAULT_INSTANCE_LEARNING_RATE = 5.0e-5
DEFAULT_INSTANCE_WEIGHT_DECAY = 1.0e-3
DEFAULT_INSTANCE_PREDICTION_N_SAMPLES = 5

DEFAULT_TOY_SHAPE_LOSS_WEIGHT = 0.05
DEFAULT_TOY_SHAPE_LOSS_PAD_FRAC = 0.3
DEFAULT_TOY_GRADIENT_CLIP_VAL = 1.0

DEFAULT_GRAHA_SHAPE_LOSS_WEIGHT = 0.05
DEFAULT_GRAHA_SHAPE_LOSS_PAD_FRAC = 0.3
DEFAULT_GRAHA_STATS_BATCH_SIZE = 16
DEFAULT_GRAHA_SEMANTIC_BATCH_SIZE = 16
DEFAULT_GRAHA_INSTANCE_BATCH_SIZE = 2
DEFAULT_GRAHA_NUM_WORKERS = 10
DEFAULT_GRAHA_SWEEP_NUM_WORKERS = 4
DEFAULT_GRAHA_BACKBONE_LR = 5.0e-5
DEFAULT_GRAHA_HEAD_LR = 2.0e-4
DEFAULT_GRAHA_LAYER_DECAY = 0.75
DEFAULT_GRAHA_WEIGHT_DECAY = 0.05
DEFAULT_GRAHA_WARMUP_STEPS = 500
DEFAULT_GRAHA_SCORE_THRESHOLD = 0.5
DEFAULT_GRAHA_FREEZE_BACKBONE = False

DEFAULT_PLOT_EVERY_N_EPOCHS = 1
DEFAULT_PLOT_N_SAMPLES = 5
DEFAULT_PREDICTION_SPLIT = "val"
DEFAULT_SWEEP_SPLIT = "test"
DEFAULT_PREDICTION_SCORE_THRESHOLD = 0.5
DEFAULT_EPOCH_TEST_SPLIT = "test"
DEFAULT_EPOCH_TEST_N_SAMPLES = 100
DEFAULT_EPOCH_TEST_EVERY_N_EPOCHS = 1
DEFAULT_PROGRESS_LOG_EVERY_N_BATCHES = 25
DEFAULT_PIPELINE_PROGRESS_LOG_EVERY_N_BATCHES = 20


def normalization_modality_for_dataset(dataset_modality: str) -> str:
    if dataset_modality == "wac":
        return "vis_uv"
    if dataset_modality == "wac_static":
        return "vis_uv_static"
    if dataset_modality == "nac":
        return "nac"
    if dataset_modality == "nac_dtm":
        return "nac"
    raise ValueError(
        f"dataset_modality must be one of {DATASET_MODALITY_CHOICES}, "
        f"got {dataset_modality!r}."
    )


def normalize_normalization_modality(normalization_modality: str) -> str:
    modality = normalization_modality.replace("-", "_").lower()
    if modality in NORMALIZATION_MODALITY_CHOICES:
        return modality
    raise ValueError(
        "normalization_modality must be one of "
        "{'vis_uv', 'vis_uv_static', 'nac'} internally or "
        "{'vis-uv', 'vis-uv-static', 'nac'} on the CLI, got "
        f"{normalization_modality!r}."
    )


def graha_input_modality_mode_for_dataset(dataset_modality: str) -> str:
    if dataset_modality == "wac":
        return "vis-uv"
    if dataset_modality == "wac_static":
        return "single"
    if dataset_modality == "nac":
        return "single"
    if dataset_modality == "nac_dtm":
        return "nac-dtm"
    raise ValueError(
        f"dataset_modality must be one of {DATASET_MODALITY_CHOICES}, "
        f"got {dataset_modality!r}."
    )


def default_band_filter_for_dataset(dataset_modality: str | None) -> list[int]:
    modality = dataset_modality or DEFAULT_DATASET_MODALITY
    if modality == "wac":
        return list(DEFAULT_WAC_BAND_FILTER)
    if modality == "wac_static":
        return list(DEFAULT_WAC_STATIC_BAND_FILTER)
    if modality == "nac":
        return list(DEFAULT_NAC_BAND_FILTER)
    if modality == "nac_dtm":
        return list(DEFAULT_NAC_DTM_BAND_FILTER)
    raise ValueError(
        f"dataset_modality must be one of {DATASET_MODALITY_CHOICES}, "
        f"got {dataset_modality!r}."
    )


def resolve_normalization_modality(
    *,
    dataset_modality: str | None,
    normalization_modality: str | None,
) -> str:
    if normalization_modality is not None:
        return normalize_normalization_modality(normalization_modality)
    return normalization_modality_for_dataset(
        dataset_modality or DEFAULT_DATASET_MODALITY
    )


def resolve_graha_input_modality_mode(
    *,
    dataset_modality: str | None,
    graha_input_modality_mode: str | None,
) -> str:
    if graha_input_modality_mode is not None:
        normalized = graha_input_modality_mode.replace("_", "-").lower()
        if normalized not in GRAHA_INPUT_MODALITY_CHOICES:
            raise ValueError(
                "graha_input_modality_mode must be one of "
                f"{GRAHA_INPUT_MODALITY_CHOICES}, got {graha_input_modality_mode!r}."
            )
        return normalized
    return graha_input_modality_mode_for_dataset(
        dataset_modality or DEFAULT_DATASET_MODALITY
    )


def graha_backend_modalities_for_input_mode(
    graha_input_modality_mode: str | None,
) -> list[str] | None:
    """Map a Graha input modality mode to backend TerraTorch modalities."""
    if graha_input_modality_mode is None:
        return None
    mode = graha_input_modality_mode.replace("_", "-").lower()
    return GRAHA_INPUT_MODE_TO_BACKEND_MODALITIES.get(mode)


def normalize_graha_backend_modalities(
    graha_backend_modalities: object,
) -> list[str] | None:
    """Normalize explicit TerraTorch backend modality names."""
    if graha_backend_modalities is None:
        return None
    if isinstance(graha_backend_modalities, str):
        raw_modalities = [
            item.strip()
            for item in graha_backend_modalities.replace(",", " ").split()
            if item.strip()
        ]
    else:
        try:
            raw_modalities = [str(item).strip() for item in graha_backend_modalities]
        except TypeError as exc:
            raise TypeError(
                "graha_backend_modalities must be a string or sequence of strings."
            ) from exc
    modalities = [item.replace("-", "_").lower() for item in raw_modalities]
    if not modalities:
        return None
    unknown = sorted(set(modalities) - set(GRAHA_BACKEND_MODALITY_CHOICES))
    if unknown:
        raise ValueError(
            "graha_backend_modalities must contain only "
            f"{GRAHA_BACKEND_MODALITY_CHOICES}, got unknown value(s) {unknown}."
        )
    if "wac" in modalities and len(modalities) > 1:
        raise ValueError(
            "graha_backend_modalities='wac' is a fused synthetic fallback and "
            "cannot be mixed with other backend modalities."
        )
    return modalities


def graha_backend_modalities_for_frontend(
    selected_modalities: object,
) -> list[str] | None:
    """Map data/frontend modality names to TerraTorch backend modality names."""
    if selected_modalities is None:
        return None
    if isinstance(selected_modalities, str):
        raw_modalities = [
            item.strip()
            for item in selected_modalities.replace(",", " ").split()
            if item.strip()
        ]
    else:
        try:
            raw_modalities = [str(item).strip() for item in selected_modalities]
        except TypeError as exc:
            raise TypeError(
                "selected_modalities must be a string or sequence of strings."
            ) from exc
    backend_modalities: list[str] = []
    for modality in raw_modalities:
        key = modality.replace("-", "_").lower()
        if key not in GRAHA_FRONTEND_TO_BACKEND_MODALITY:
            raise ValueError(
                f"Cannot infer Graha backend modality for selected modality {modality!r}. "
                "Pass graha_backend_modalities explicitly."
            )
        backend_modalities.append(GRAHA_FRONTEND_TO_BACKEND_MODALITY[key])
    return backend_modalities or None


def expected_graha_backend_num_channels(
    graha_backend_modalities: list[str] | None,
) -> int | None:
    """Return expected concatenated channel count for known backend modalities."""
    modalities = normalize_graha_backend_modalities(graha_backend_modalities)
    if not modalities or "wac" in modalities:
        return None
    channel_counts = [
        GRAHA_BACKEND_MODALITY_CHANNELS[modality] for modality in modalities
    ]
    if any(count is None for count in channel_counts):
        return None
    return int(sum(count for count in channel_counts if count is not None))


def graha_input_modality_mode_for_backend(
    graha_backend_modalities: list[str] | None,
) -> str | None:
    """Choose the closest legacy Graha mode label for backend modalities."""
    modalities = normalize_graha_backend_modalities(graha_backend_modalities)
    if modalities is None:
        return None
    for mode, mode_modalities in GRAHA_INPUT_MODE_TO_BACKEND_MODALITIES.items():
        if modalities == mode_modalities:
            return mode
    if modalities == ["vis", "uv"]:
        return "vis-uv"
    if modalities == ["nac", "dtm"]:
        return "nac-dtm"
    return "single"


def resolve_semantic_label_source(
    *,
    semantic_label_source: str | None,
    label_glob: str | None,
    label_file_type: str | None = None,
    data_root: object | None = None,
) -> str:
    """Resolve semantic label source from explicit config or label file type.

    ``.npz`` labels are treated as instance archives that should be converted
    to semantic masks. ``.npy`` labels are treated as semantic masks already.
    Ambiguous globs such as ``*label.*`` are resolved from existing split label
    files when possible.
    """
    source = semantic_label_source or DEFAULT_SEMANTIC_LABEL_SOURCE
    if source != "auto":
        if source not in {"semantic", "instance"}:
            raise ValueError(
                "semantic_label_source must be 'auto', 'semantic', or 'instance', "
                f"got {source!r}."
            )
        return source

    candidates = [label_file_type, label_glob]
    for value in candidates:
        lowered = str(value or "").lower()
        if ".npz" in lowered:
            return "instance"
        if ".npy" in lowered:
            return "semantic"

    if data_root is not None and label_glob:
        from pathlib import Path

        root = Path(data_root)
        suffix_counts: dict[str, int] = {}
        for split in ("train", "val", "test"):
            labels_dir = root / split / "labels"
            if not labels_dir.exists():
                continue
            for path in labels_dir.glob(label_glob):
                suffix = path.suffix.lower()
                suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
        if suffix_counts:
            if suffix_counts.get(".npz", 0) > 0:
                return "instance"
            if suffix_counts.get(".npy", 0) > 0:
                return "semantic"

    return "semantic"
