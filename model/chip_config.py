"""Validated configuration objects for target-grid-driven chip creation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
import operator
from pathlib import Path
import re
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

from .tiling_config import TileConfig, tile_config_from_dict


ChipResampling = Literal["bilinear", "nearest"]
IntermediateRetention = Literal["never", "on_failure", "always"]
SplitName = Literal["train", "val", "test"]
AssignmentName = Literal["train", "val", "test", "unsplit"]

CHIP_RESAMPLING_METHODS = ("bilinear", "nearest")
INTERMEDIATE_RETENTION_POLICIES = ("never", "on_failure", "always")
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "val", "test")
ASSIGNMENT_NAMES: tuple[AssignmentName, ...] = (*SPLIT_NAMES, "unsplit")
SPLIT_HASH_VERSION = "blake2b-v1"
DEFAULT_SPLIT_SEED = 0
SUPPORTED_OUTPUT_DTYPES = (
    "uint8",
    "uint16",
    "int16",
    "uint32",
    "int32",
    "float32",
    "float64",
)

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _non_empty_text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty.")
    return text


def _safe_name(value: object, *, field_name: str) -> str:
    text = _non_empty_text(value, field_name=field_name)
    if not _SAFE_NAME.fullmatch(text):
        raise ValueError(
            f"{field_name} must contain only letters, numbers, underscores, "
            "and hyphens, and must start with a letter or number."
        )
    return text


def _path(value: object, *, field_name: str) -> Path:
    if not str(value).strip():
        raise ValueError(f"{field_name} must not be empty.")
    try:
        return Path(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{field_name} must be path-like.") from exc


def _require_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")
    return value


def _reject_unknown_keys(
    values: Mapping[str, Any],
    *,
    allowed: set[str],
    field_name: str,
) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise TypeError(f"Unknown {field_name} option(s): {', '.join(unknown)}")


def _string_tuple(value: object, *, field_name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings or None.")
    result = tuple(
        _non_empty_text(item, field_name=f"{field_name} item") for item in value
    )
    if not result:
        raise ValueError(f"{field_name} must not be empty when provided.")
    if len({item.casefold() for item in result}) != len(result):
        raise ValueError(f"{field_name} must not contain duplicates.")
    return result


def _index_tuple(value: object, *, field_name: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence of 1-based indices.")
    if any(isinstance(item, bool) for item in value):
        raise TypeError(f"{field_name} values must be integers, not booleans.")
    try:
        result = tuple(operator.index(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} values must be integers.") from exc
    if not result:
        raise ValueError(f"{field_name} must not be empty when provided.")
    if any(index < 1 for index in result):
        raise ValueError(f"{field_name} values must be 1-based positive integers.")
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must not contain duplicates.")
    return result


def _split_name(value: object, *, field_name: str) -> SplitName:
    name = str(value).strip().lower()
    if name not in SPLIT_NAMES:
        valid = ", ".join(SPLIT_NAMES)
        raise ValueError(f"{field_name} must be one of {valid}, got {value!r}.")
    return name  # type: ignore[return-value]


def _priority_tuple(value: object, *, field_name: str) -> tuple[SplitName, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence of split names.")
    priority = tuple(
        _split_name(item, field_name=f"{field_name} item") for item in value
    )
    if not priority:
        raise ValueError(f"{field_name} must not be empty.")
    if len(set(priority)) != len(priority):
        raise ValueError(f"{field_name} must not contain duplicates.")
    return priority


@dataclass(frozen=True)
class AcquisitionGroupConfig:
    """Name one composed tiling configuration used as an acquisition grid."""

    name: str
    tile_config: TileConfig

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _safe_name(self.name, field_name="acquisition group name"),
        )
        if not isinstance(self.tile_config, TileConfig):
            raise TypeError("tile_config must be a TileConfig.")


@dataclass(frozen=True)
class OutputModalityConfig:
    """Select ordered output bands from one source in one acquisition group."""

    acquisition_group: str
    source_name: str
    alias: str
    band_names: tuple[str, ...] | None = None
    band_indices: tuple[int, ...] | None = None
    output_band_names: tuple[str, ...] | None = None
    resampling: ChipResampling = "bilinear"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "acquisition_group",
            _safe_name(
                self.acquisition_group,
                field_name="output modality acquisition_group",
            ),
        )
        object.__setattr__(
            self,
            "source_name",
            _non_empty_text(
                self.source_name,
                field_name="output modality source_name",
            ),
        )
        object.__setattr__(
            self,
            "alias",
            _safe_name(self.alias, field_name="output modality alias"),
        )
        object.__setattr__(
            self,
            "band_names",
            _string_tuple(self.band_names, field_name="band_names"),
        )
        object.__setattr__(
            self,
            "band_indices",
            _index_tuple(self.band_indices, field_name="band_indices"),
        )
        object.__setattr__(
            self,
            "output_band_names",
            _string_tuple(
                self.output_band_names,
                field_name="output_band_names",
            ),
        )
        if self.band_names is not None and self.band_indices is not None:
            raise ValueError("Configure band_names or band_indices, not both.")
        selected_count = None
        if self.band_names is not None:
            selected_count = len(self.band_names)
        elif self.band_indices is not None:
            selected_count = len(self.band_indices)
        if (
            selected_count is not None
            and self.output_band_names is not None
            and len(self.output_band_names) != selected_count
        ):
            raise ValueError(
                "output_band_names must match the configured band selection length."
            )
        if self.resampling not in CHIP_RESAMPLING_METHODS:
            valid = ", ".join(CHIP_RESAMPLING_METHODS)
            raise ValueError(
                f"Chip resampling must be one of {valid}, got "
                f"{self.resampling!r}."
            )


@dataclass(frozen=True)
class SplitPercentages:
    """Train/validation/test proportions for a percentage assignment stage."""

    train: float
    val: float
    test: float

    def __post_init__(self) -> None:
        for name in SPLIT_NAMES:
            value = getattr(self, name)
            if isinstance(value, bool):
                raise TypeError(f"{name} percentage must be numeric, not boolean.")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{name} percentage must be numeric.") from exc
            if not math.isfinite(number) or number < 0.0 or number > 1.0:
                raise ValueError(
                    f"{name} percentage must be finite and between 0 and 1."
                )
            object.__setattr__(self, name, number)
        if not math.isclose(
            self.train + self.val + self.test,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Split percentages must sum to 1.")


@dataclass(frozen=True)
class SplitCounts:
    """Best-effort fixed sample targets; ``None`` means no target."""

    train: int | None = None
    val: int | None = None
    test: int | None = None

    def __post_init__(self) -> None:
        for name in SPLIT_NAMES:
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} count must be an integer or None.")
            if value < 1:
                raise ValueError(f"{name} count must be positive when configured.")
        if not self.targeted_splits:
            raise ValueError("At least one positive split count is required.")

    @property
    def targeted_splits(self) -> tuple[SplitName, ...]:
        """Return fixed-target split names in canonical order."""
        return tuple(
            name for name in SPLIT_NAMES if getattr(self, name) is not None
        )  # type: ignore[return-value]


@runtime_checkable
class SplitConfig(Protocol):
    """Shared configuration surface for deterministic dataset splitting."""

    seed: int
    group_key_policy: str
    hash_version: str
    prior_manifest_path: Path | None


def _validate_split_common(config: SplitConfig) -> None:
    if isinstance(config.seed, bool) or not isinstance(config.seed, int):
        raise TypeError("split seed must be an integer.")
    policy = _non_empty_text(
        config.group_key_policy,
        field_name="group_key_policy",
    )
    object.__setattr__(config, "group_key_policy", policy)
    if config.hash_version != SPLIT_HASH_VERSION:
        raise ValueError(
            f"hash_version must be {SPLIT_HASH_VERSION!r} for this contract."
        )
    if config.prior_manifest_path is not None:
        object.__setattr__(
            config,
            "prior_manifest_path",
            _path(
                config.prior_manifest_path,
                field_name="prior_manifest_path",
            ),
        )


def _validate_fixed_priority(
    counts: SplitCounts,
    priority: tuple[SplitName, ...],
) -> None:
    targeted = set(counts.targeted_splits)
    if set(priority) != targeted:
        raise ValueError(
            "fixed_priority must contain every configured fixed-count split "
            "exactly once and no other splits."
        )


@dataclass(frozen=True)
class SimpleSplitConfig:
    """Assign all unassigned groups using deterministic percentage thresholds."""

    percentages: SplitPercentages = field(
        default_factory=lambda: SplitPercentages(0.8, 0.1, 0.1)
    )
    seed: int = DEFAULT_SPLIT_SEED
    group_key_policy: str = "request"
    hash_version: str = SPLIT_HASH_VERSION
    prior_manifest_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.percentages, SplitPercentages):
            raise TypeError("percentages must be a SplitPercentages object.")
        _validate_split_common(self)


@dataclass(frozen=True)
class MixedPercentageNumberSplitConfig:
    """Fill prioritized fixed targets, then split the remaining groups by ratio."""

    fixed_counts: SplitCounts = field(
        default_factory=lambda: SplitCounts(test=100)
    )
    fixed_priority: tuple[SplitName, ...] = ("test",)
    remaining_percentages: SplitPercentages = field(
        default_factory=lambda: SplitPercentages(0.9, 0.1, 0.0)
    )
    seed: int = DEFAULT_SPLIT_SEED
    group_key_policy: str = "request"
    hash_version: str = SPLIT_HASH_VERSION
    prior_manifest_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.fixed_counts, SplitCounts):
            raise TypeError("fixed_counts must be a SplitCounts object.")
        if not isinstance(self.remaining_percentages, SplitPercentages):
            raise TypeError(
                "remaining_percentages must be a SplitPercentages object."
            )
        priority = _priority_tuple(
            self.fixed_priority,
            field_name="fixed_priority",
        )
        object.__setattr__(self, "fixed_priority", priority)
        _validate_fixed_priority(self.fixed_counts, priority)
        _validate_split_common(self)


@dataclass(frozen=True)
class NumberSplitConfig:
    """Assign prioritized best-effort counts, optionally routing the remainder."""

    fixed_counts: SplitCounts
    fixed_priority: tuple[SplitName, ...]
    remainder_split: SplitName | None = None
    seed: int = DEFAULT_SPLIT_SEED
    group_key_policy: str = "request"
    hash_version: str = SPLIT_HASH_VERSION
    prior_manifest_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.fixed_counts, SplitCounts):
            raise TypeError("fixed_counts must be a SplitCounts object.")
        priority = _priority_tuple(
            self.fixed_priority,
            field_name="fixed_priority",
        )
        object.__setattr__(self, "fixed_priority", priority)
        _validate_fixed_priority(self.fixed_counts, priority)
        if self.remainder_split is not None:
            object.__setattr__(
                self,
                "remainder_split",
                _split_name(
                    self.remainder_split,
                    field_name="remainder_split",
                ),
            )
        _validate_split_common(self)


@dataclass(frozen=True)
class NoSplitConfig:
    """Assign every request to the dataset root without creating partitions."""

    seed: int = DEFAULT_SPLIT_SEED
    group_key_policy: str = "request"
    hash_version: str = SPLIT_HASH_VERSION
    prior_manifest_path: Path | None = None

    def __post_init__(self) -> None:
        _validate_split_common(self)
        if self.prior_manifest_path is not None:
            raise ValueError("NoSplitConfig does not accept a prior split manifest.")


SplitConfigType: TypeAlias = (
    SimpleSplitConfig
    | MixedPercentageNumberSplitConfig
    | NumberSplitConfig
    | NoSplitConfig
)


def default_split_config() -> MixedPercentageNumberSplitConfig:
    """Return the repository default: 100 test, then 90/10 train/validation."""
    return MixedPercentageNumberSplitConfig()


@dataclass(frozen=True)
class ChipConfig:
    """Top-level configuration shared by every request in one chip batch."""

    output_root: Path
    label_source: Path
    acquisition_groups: tuple[AcquisitionGroupConfig, ...]
    output_modalities: tuple[OutputModalityConfig, ...]
    split_config: SplitConfigType = field(default_factory=default_split_config)
    intermediate_root: Path | None = None
    output_suffix: str | None = None
    common_nodata: float = -32768.0
    output_dtype: str = "float32"
    sample_limit: int | None = None
    intermediate_retention: IntermediateRetention = "on_failure"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "output_root",
            _path(self.output_root, field_name="output_root"),
        )
        object.__setattr__(
            self,
            "label_source",
            _path(self.label_source, field_name="label_source"),
        )
        intermediate_root = (
            self.output_root / ".intermediate"
            if self.intermediate_root is None
            else _path(self.intermediate_root, field_name="intermediate_root")
        )
        object.__setattr__(self, "intermediate_root", intermediate_root)

        groups = tuple(self.acquisition_groups)
        if not groups:
            raise ValueError("ChipConfig requires at least one acquisition group.")
        if any(not isinstance(group, AcquisitionGroupConfig) for group in groups):
            raise TypeError(
                "acquisition_groups must contain AcquisitionGroupConfig objects."
            )
        group_keys = [group.name.casefold() for group in groups]
        if len(set(group_keys)) != len(group_keys):
            raise ValueError(
                "Acquisition group names must be case-insensitively unique."
            )
        object.__setattr__(self, "acquisition_groups", groups)

        modalities = tuple(self.output_modalities)
        if not modalities:
            raise ValueError("ChipConfig requires at least one output modality.")
        if any(not isinstance(item, OutputModalityConfig) for item in modalities):
            raise TypeError(
                "output_modalities must contain OutputModalityConfig objects."
            )
        aliases = [item.alias.casefold() for item in modalities]
        if len(set(aliases)) != len(aliases):
            raise ValueError(
                "Output modality aliases must be case-insensitively unique."
            )
        references = [
            (item.acquisition_group.casefold(), item.source_name.casefold())
            for item in modalities
        ]
        if len(set(references)) != len(references):
            raise ValueError(
                "Output modalities must not repeat an acquisition-group/source pair."
            )
        groups_by_name = {group.name: group for group in groups}
        for item in modalities:
            if item.acquisition_group not in groups_by_name:
                raise ValueError(
                    f"Output modality {item.alias!r} references unknown acquisition "
                    f"group {item.acquisition_group!r}."
                )
            try:
                groups_by_name[item.acquisition_group].tile_config.source(
                    item.source_name
                )
            except KeyError as exc:
                raise ValueError(
                    f"Output modality {item.alias!r} references unknown source "
                    f"{item.source_name!r} in acquisition group "
                    f"{item.acquisition_group!r}."
                ) from exc
        explicit_band_names = [
            name.casefold()
            for item in modalities
            for name in (item.output_band_names or ())
        ]
        if len(set(explicit_band_names)) != len(explicit_band_names):
            raise ValueError(
                "Explicit output band names must be case-insensitively unique "
                "across modalities."
            )
        object.__setattr__(self, "output_modalities", modalities)

        if not isinstance(
            self.split_config,
            (
                SimpleSplitConfig,
                MixedPercentageNumberSplitConfig,
                NumberSplitConfig,
                NoSplitConfig,
            ),
        ):
            raise TypeError("split_config must be a supported split config object.")
        if self.output_suffix is not None:
            suffix = str(self.output_suffix).strip()
            if (
                Path(suffix).name != suffix
                or not re.fullmatch(r"_input_[A-Za-z0-9_-]+_chip\.tiff?", suffix)
            ):
                raise ValueError(
                    "output_suffix must be a terminal filename suffix such as "
                    "'_input_wac_static_chip.tif'."
                )
            object.__setattr__(self, "output_suffix", suffix)
        if isinstance(self.common_nodata, bool):
            raise TypeError("common_nodata must be numeric, not boolean.")
        try:
            nodata = float(self.common_nodata)
        except (TypeError, ValueError) as exc:
            raise TypeError("common_nodata must be numeric.") from exc
        if not math.isfinite(nodata):
            raise ValueError("common_nodata must be finite.")
        object.__setattr__(self, "common_nodata", nodata)
        dtype = str(self.output_dtype).strip().lower()
        if dtype not in SUPPORTED_OUTPUT_DTYPES:
            valid = ", ".join(SUPPORTED_OUTPUT_DTYPES)
            raise ValueError(f"output_dtype must be one of {valid}.")
        object.__setattr__(self, "output_dtype", dtype)
        if self.sample_limit is not None:
            if isinstance(self.sample_limit, bool) or not isinstance(
                self.sample_limit, int
            ):
                raise TypeError("sample_limit must be an integer or None.")
            if self.sample_limit < 1:
                raise ValueError("sample_limit must be positive when configured.")
        if self.intermediate_retention not in INTERMEDIATE_RETENTION_POLICIES:
            valid = ", ".join(INTERMEDIATE_RETENTION_POLICIES)
            raise ValueError(
                f"intermediate_retention must be one of {valid}, got "
                f"{self.intermediate_retention!r}."
            )

    @property
    def final_output_suffix(self) -> str:
        """Return the configured or deterministically derived chip suffix."""
        if self.output_suffix is not None:
            return self.output_suffix
        aliases = "_".join(item.alias for item in self.output_modalities)
        return f"_input_{aliases}_chip.tif"

    def acquisition_group(self, name: str) -> AcquisitionGroupConfig:
        """Return an acquisition group by its exact configured name."""
        for group in self.acquisition_groups:
            if group.name == name:
                return group
        raise KeyError(f"Unknown acquisition group: {name!r}")


def default_zoom_for_sources(source_names: Sequence[str]) -> int:
    """Resolve the C0 default zoom for a group whose zoom was omitted."""
    normalized = {
        _non_empty_text(name, field_name="source name").casefold()
        for name in source_names
    }
    if not normalized:
        raise ValueError("An acquisition group requires at least one source.")
    if "nac" in normalized:
        conflicting = normalized - {"nac", "static"}
        if conflicting:
            names = ", ".join(sorted(conflicting))
            raise ValueError(
                "An acquisition group combining built-in NAC with sources that "
                f"default to zoom 5 ({names}) must declare zoom_level explicitly."
            )
        return 11
    return 5


def _percentages_from_dict(
    value: object,
    *,
    field_name: str,
) -> SplitPercentages:
    values = _require_mapping(value, field_name=field_name)
    _reject_unknown_keys(values, allowed=set(SPLIT_NAMES), field_name=field_name)
    if not values:
        raise ValueError(f"{field_name} must not be empty.")
    return SplitPercentages(
        train=values.get("train", 0.0),
        val=values.get("val", 0.0),
        test=values.get("test", 0.0),
    )


def _counts_from_dict(value: object, *, field_name: str) -> SplitCounts:
    values = _require_mapping(value, field_name=field_name)
    _reject_unknown_keys(values, allowed=set(SPLIT_NAMES), field_name=field_name)
    if not values:
        raise ValueError(f"{field_name} must not be empty.")
    return SplitCounts(
        train=values.get("train"),
        val=values.get("val"),
        test=values.get("test"),
    )


def split_config_from_dict(value: Mapping[str, Any]) -> SplitConfigType:
    """Build one split configuration from a plain mapping."""
    values = _require_mapping(value, field_name="split config")
    common_keys = {
        "type",
        "seed",
        "group_key_policy",
        "hash_version",
        "prior_manifest_path",
    }
    if "type" not in values:
        raise KeyError("Split config must include 'type'.")
    config_type = str(values["type"]).strip().lower()
    common = {
        "seed": values.get("seed", DEFAULT_SPLIT_SEED),
        "group_key_policy": values.get("group_key_policy", "request"),
        "hash_version": values.get("hash_version", SPLIT_HASH_VERSION),
        "prior_manifest_path": values.get("prior_manifest_path"),
    }
    if config_type == "simple":
        _reject_unknown_keys(
            values,
            allowed=common_keys | {"percentages"},
            field_name="simple split config",
        )
        if "percentages" not in values:
            raise KeyError("Simple split config must include 'percentages'.")
        return SimpleSplitConfig(
            percentages=_percentages_from_dict(
                values["percentages"],
                field_name="split percentages",
            ),
            **common,
        )
    if config_type == "mixed_percentage_number":
        _reject_unknown_keys(
            values,
            allowed=common_keys
            | {"fixed_counts", "fixed_priority", "remaining_percentages"},
            field_name="mixed split config",
        )
        required = ("fixed_counts", "fixed_priority", "remaining_percentages")
        for name in required:
            if name not in values:
                raise KeyError(f"Mixed split config must include {name!r}.")
        return MixedPercentageNumberSplitConfig(
            fixed_counts=_counts_from_dict(
                values["fixed_counts"],
                field_name="fixed_counts",
            ),
            fixed_priority=_priority_tuple(
                values["fixed_priority"],
                field_name="fixed_priority",
            ),
            remaining_percentages=_percentages_from_dict(
                values["remaining_percentages"],
                field_name="remaining_percentages",
            ),
            **common,
        )
    if config_type == "number":
        _reject_unknown_keys(
            values,
            allowed=common_keys
            | {"fixed_counts", "fixed_priority", "remainder_split"},
            field_name="number split config",
        )
        for name in ("fixed_counts", "fixed_priority"):
            if name not in values:
                raise KeyError(f"Number split config must include {name!r}.")
        return NumberSplitConfig(
            fixed_counts=_counts_from_dict(
                values["fixed_counts"],
                field_name="fixed_counts",
            ),
            fixed_priority=_priority_tuple(
                values["fixed_priority"],
                field_name="fixed_priority",
            ),
            remainder_split=values.get("remainder_split"),
            **common,
        )
    if config_type == "no_split":
        _reject_unknown_keys(
            values,
            allowed=common_keys,
            field_name="no-split config",
        )
        return NoSplitConfig(**common)
    raise ValueError(
        "split config type must be 'simple', 'mixed_percentage_number', "
        f"'number', or 'no_split', got {values['type']!r}."
    )


def _output_modality_from_dict(value: object) -> OutputModalityConfig:
    values = _require_mapping(value, field_name="output modality")
    _reject_unknown_keys(
        values,
        allowed={
            "acquisition_group",
            "source",
            "alias",
            "bands",
            "output_band_names",
            "resampling",
        },
        field_name="output modality",
    )
    for required in ("acquisition_group", "source", "alias"):
        if required not in values:
            raise KeyError(f"Output modality must include {required!r}.")
    band_names = None
    band_indices = None
    bands = values.get("bands")
    if bands not in (None, "all"):
        band_values = _require_mapping(bands, field_name="output modality bands")
        _reject_unknown_keys(
            band_values,
            allowed={"names", "indices"},
            field_name="output modality bands",
        )
        band_names = band_values.get("names")
        band_indices = band_values.get("indices")
    return OutputModalityConfig(
        acquisition_group=values["acquisition_group"],
        source_name=values["source"],
        alias=values["alias"],
        band_names=band_names,
        band_indices=band_indices,
        output_band_names=values.get("output_band_names"),
        resampling=values.get("resampling", "bilinear"),
    )


def chip_config_from_dict(value: Mapping[str, Any]) -> ChipConfig:
    """Build a :class:`ChipConfig` from a notebook-friendly mapping."""
    values = _require_mapping(value, field_name="chip config")
    _reject_unknown_keys(
        values,
        allowed={
            "output_root",
            "label_source",
            "intermediate_root",
            "acquisition_groups",
            "output_modalities",
            "split",
            "output_suffix",
            "common_nodata",
            "output_dtype",
            "sample_limit",
            "intermediate_retention",
        },
        field_name="chip config",
    )
    for required in (
        "output_root",
        "label_source",
        "acquisition_groups",
        "output_modalities",
    ):
        if required not in values:
            raise KeyError(f"Chip config must include {required!r}.")

    output_root = _path(values["output_root"], field_name="output_root")
    raw_intermediate_root = values.get("intermediate_root")
    intermediate_root = (
        output_root / ".intermediate"
        if raw_intermediate_root is None
        else _path(raw_intermediate_root, field_name="intermediate_root")
    )
    raw_groups = _require_mapping(
        values["acquisition_groups"],
        field_name="acquisition_groups",
    )
    groups: list[AcquisitionGroupConfig] = []
    for raw_name, raw_group in raw_groups.items():
        name = _safe_name(raw_name, field_name="acquisition group name")
        group_values = dict(
            _require_mapping(
                raw_group,
                field_name=f"acquisition_groups[{name!r}]",
            )
        )
        raw_sources = _require_mapping(
            group_values.get("sources"),
            field_name=f"acquisition_groups[{name!r}].sources",
        )
        if "zoom_level" not in group_values:
            group_values["zoom_level"] = default_zoom_for_sources(
                [str(source_name) for source_name in raw_sources]
            )
        group_values.setdefault("output_dir", intermediate_root / name)
        groups.append(
            AcquisitionGroupConfig(
                name=name,
                tile_config=tile_config_from_dict(group_values),
            )
        )

    raw_modalities = values["output_modalities"]
    if isinstance(raw_modalities, str) or not isinstance(
        raw_modalities, Sequence
    ):
        raise TypeError("output_modalities must be a sequence of mappings.")
    modalities = tuple(
        _output_modality_from_dict(item) for item in raw_modalities
    )
    split = (
        default_split_config()
        if "split" not in values
        else split_config_from_dict(values["split"])
    )
    return ChipConfig(
        output_root=output_root,
        label_source=_path(values["label_source"], field_name="label_source"),
        acquisition_groups=tuple(groups),
        output_modalities=modalities,
        split_config=split,
        intermediate_root=intermediate_root,
        output_suffix=values.get("output_suffix"),
        common_nodata=values.get("common_nodata", -32768.0),
        output_dtype=values.get("output_dtype", "float32"),
        sample_limit=values.get("sample_limit"),
        intermediate_retention=values.get(
            "intermediate_retention",
            "on_failure",
        ),
    )


__all__ = [
    "ASSIGNMENT_NAMES",
    "AcquisitionGroupConfig",
    "AssignmentName",
    "CHIP_RESAMPLING_METHODS",
    "ChipConfig",
    "ChipResampling",
    "DEFAULT_SPLIT_SEED",
    "INTERMEDIATE_RETENTION_POLICIES",
    "IntermediateRetention",
    "MixedPercentageNumberSplitConfig",
    "NoSplitConfig",
    "NumberSplitConfig",
    "OutputModalityConfig",
    "SPLIT_HASH_VERSION",
    "SPLIT_NAMES",
    "SUPPORTED_OUTPUT_DTYPES",
    "SimpleSplitConfig",
    "SplitConfig",
    "SplitConfigType",
    "SplitCounts",
    "SplitName",
    "SplitPercentages",
    "chip_config_from_dict",
    "default_split_config",
    "default_zoom_for_sources",
    "split_config_from_dict",
]
