"""Deterministic, leakage-group-aware split planning for chip requests."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal
import warnings as warnings_module

from .chip_config import (
    MixedPercentageNumberSplitConfig,
    NumberSplitConfig,
    SPLIT_NAMES,
    SimpleSplitConfig,
    SplitConfigType,
    SplitCounts,
    SplitName,
    SplitPercentages,
)
from .chip_requests import materialize_requests
from .chip_types import ChipRequest


AssignmentSource = Literal[
    "explicit",
    "prior_manifest",
    "automatic_fixed",
    "automatic_percentage",
    "automatic_remainder",
    "unassigned",
]


class SplitTargetWarning(UserWarning):
    """A deterministic whole-group assignment could not meet a fixed target."""


@dataclass(frozen=True)
class SplitAssignment:
    """The planned dataset membership for one chip request."""

    sample_id: str
    split_group_key: str
    assigned_split: SplitName | None
    source: AssignmentSource


@dataclass(frozen=True)
class SplitWarning:
    """A best-effort fixed target could not be met exactly."""

    code: str
    message: str
    split: SplitName
    requested_count: int
    realized_count: int
    reason: str


@dataclass(frozen=True)
class SplitPlan:
    """Deterministically ordered assignments and fixed-target warnings."""

    assignments: tuple[SplitAssignment, ...]
    warnings: tuple[SplitWarning, ...] = ()

    def assignment_for(self, sample_id: str) -> SplitAssignment:
        """Return one assignment using case-insensitive sample identity."""
        key = str(sample_id).casefold()
        matches = tuple(
            item for item in self.assignments if item.sample_id.casefold() == key
        )
        if len(matches) != 1:
            raise KeyError(f"Expected one split assignment for {sample_id!r}.")
        return matches[0]

    @property
    def realized_counts(self) -> dict[str, int]:
        """Count samples with a realized split assignment."""
        counts = Counter(
            item.assigned_split
            for item in self.assignments
            if item.assigned_split is not None
        )
        return {name: int(counts.get(name, 0)) for name in SPLIT_NAMES}


@dataclass
class _RequestGroup:
    normalized_key: str
    display_key: str
    requests: tuple[ChipRequest, ...]
    assigned_split: SplitName | None = None
    source: AssignmentSource | None = None

    @property
    def size(self) -> int:
        return len(self.requests)


def _digest_value(
    *,
    config: SplitConfigType,
    namespace: str,
    group_key: str,
) -> int:
    payload = "\0".join(
        (
            config.hash_version,
            namespace,
            str(config.seed),
            config.group_key_policy,
            group_key.casefold(),
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=16).digest(), "big")


def _digest_fraction(
    *,
    config: SplitConfigType,
    namespace: str,
    group_key: str,
) -> float:
    return _digest_value(
        config=config,
        namespace=namespace,
        group_key=group_key,
    ) / float(1 << 128)


def _split_for_fraction(
    fraction: float,
    percentages: SplitPercentages,
) -> SplitName:
    train_end = percentages.train
    val_end = train_end + percentages.val
    if fraction < train_end:
        return "train"
    if fraction < val_end:
        return "val"
    return "test"


def _manifest_group_locks(path: Path | None) -> dict[str, SplitName]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Prior split manifest does not exist: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read prior split manifest: {path}") from exc
    if not isinstance(document, Mapping):
        raise ValueError("Prior split manifest must contain a JSON object.")
    samples = document.get("samples")
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise ValueError("Prior split manifest must contain a 'samples' array.")
    locks: dict[str, SplitName] = {}
    for index, raw_sample in enumerate(samples):
        if not isinstance(raw_sample, Mapping):
            raise ValueError(f"Prior manifest sample {index} must be an object.")
        group_key = raw_sample.get("split_group_key")
        split = raw_sample.get("assigned_split", raw_sample.get("split"))
        if split in (None, "unassigned"):
            continue
        if group_key is None or not str(group_key).strip():
            raise ValueError(
                f"Prior manifest sample {index} has a split but no split_group_key."
            )
        normalized_split = str(split).strip().lower()
        if normalized_split not in SPLIT_NAMES:
            raise ValueError(
                f"Prior manifest sample {index} has unknown split {split!r}."
            )
        normalized_group = str(group_key).strip().casefold()
        previous = locks.setdefault(
            normalized_group,
            normalized_split,  # type: ignore[arg-type]
        )
        if previous != normalized_split:
            raise ValueError(
                f"Prior manifest split group {group_key!r} has conflicting splits."
            )
    return locks


def _request_groups(requests: tuple[ChipRequest, ...]) -> list[_RequestGroup]:
    grouped: dict[str, list[ChipRequest]] = {}
    display_keys: dict[str, str] = {}
    for request in requests:
        key = request.split_group_key.casefold()
        grouped.setdefault(key, []).append(request)
        display_keys.setdefault(key, request.split_group_key)
    return [
        _RequestGroup(
            normalized_key=key,
            display_key=display_keys[key],
            requests=tuple(grouped[key]),
        )
        for key in sorted(grouped)
    ]


def _apply_locked_assignments(
    groups: list[_RequestGroup],
    prior_locks: Mapping[str, SplitName],
) -> None:
    for group in groups:
        explicit = {
            request.assigned_split
            for request in group.requests
            if request.assigned_split is not None
        }
        if len(explicit) > 1:
            raise ValueError(
                f"Split group {group.display_key!r} has conflicting explicit splits."
            )
        explicit_split = next(iter(explicit), None)
        prior_split = prior_locks.get(group.normalized_key)
        if (
            explicit_split is not None
            and prior_split is not None
            and explicit_split != prior_split
        ):
            raise ValueError(
                f"Split group {group.display_key!r} conflicts with its prior "
                f"manifest assignment: {explicit_split!r} versus {prior_split!r}."
            )
        if explicit_split is not None:
            group.assigned_split = explicit_split
            group.source = "explicit"
        elif prior_split is not None:
            group.assigned_split = prior_split
            group.source = "prior_manifest"


def _assign_percentages(
    groups: Sequence[_RequestGroup],
    *,
    percentages: SplitPercentages,
    config: SplitConfigType,
    namespace: str,
) -> None:
    for group in groups:
        if group.source is not None:
            continue
        group.assigned_split = _split_for_fraction(
            _digest_fraction(
                config=config,
                namespace=namespace,
                group_key=group.normalized_key,
            ),
            percentages,
        )
        group.source = "automatic_percentage"


def _assigned_count(groups: Sequence[_RequestGroup], split: SplitName) -> int:
    return sum(group.size for group in groups if group.assigned_split == split)


def _assign_fixed_counts(
    groups: list[_RequestGroup],
    *,
    counts: SplitCounts,
    priority: tuple[SplitName, ...],
    config: SplitConfigType,
) -> list[SplitWarning]:
    warnings: list[SplitWarning] = []
    for split in priority:
        requested_count = getattr(counts, split)
        if requested_count is None:
            continue
        realized_count = _assigned_count(groups, split)
        locked_count = realized_count
        candidates = sorted(
            (group for group in groups if group.source is None),
            key=lambda group: (
                _digest_value(
                    config=config,
                    namespace=f"fixed:{split}",
                    group_key=group.normalized_key,
                ),
                group.normalized_key,
            ),
        )
        available_count = sum(group.size for group in candidates)
        for group in candidates:
            current_distance = abs(requested_count - realized_count)
            proposed_count = realized_count + group.size
            proposed_distance = abs(requested_count - proposed_count)
            if proposed_distance <= current_distance:
                group.assigned_split = split
                group.source = "automatic_fixed"
                realized_count = proposed_count
            if realized_count == requested_count:
                break
        if realized_count != requested_count:
            if locked_count > requested_count:
                reason = "preassigned_samples_exceed_target"
            elif locked_count + available_count < requested_count:
                reason = "insufficient_samples"
            else:
                reason = "atomic_split_groups"
            warnings.append(
                SplitWarning(
                    code="unrealized_fixed_target",
                    message=(
                        f"Split {split!r} requested {requested_count} samples but "
                        f"the deterministic whole-group assignment realized "
                        f"{realized_count}."
                    ),
                    split=split,
                    requested_count=requested_count,
                    realized_count=realized_count,
                    reason=reason,
                )
            )
    return warnings


def _finish_number_remainder(
    groups: Sequence[_RequestGroup],
    remainder_split: SplitName | None,
) -> None:
    for group in groups:
        if group.source is not None:
            continue
        group.assigned_split = remainder_split
        group.source = (
            "automatic_remainder" if remainder_split is not None else "unassigned"
        )


def plan_splits(
    requests: Sequence[ChipRequest],
    config: SplitConfigType,
) -> SplitPlan:
    """Plan deterministic assignments without modifying requests or writing files."""
    if not isinstance(
        config,
        (
            SimpleSplitConfig,
            MixedPercentageNumberSplitConfig,
            NumberSplitConfig,
        ),
    ):
        raise TypeError("config must be a supported split configuration.")
    ordered_requests = materialize_requests(requests)
    groups = _request_groups(ordered_requests)
    _apply_locked_assignments(
        groups,
        _manifest_group_locks(config.prior_manifest_path),
    )
    warnings: list[SplitWarning] = []
    if isinstance(config, SimpleSplitConfig):
        _assign_percentages(
            groups,
            percentages=config.percentages,
            config=config,
            namespace="percentage",
        )
    elif isinstance(config, MixedPercentageNumberSplitConfig):
        warnings.extend(
            _assign_fixed_counts(
                groups,
                counts=config.fixed_counts,
                priority=config.fixed_priority,
                config=config,
            )
        )
        _assign_percentages(
            groups,
            percentages=config.remaining_percentages,
            config=config,
            namespace="percentage:remainder",
        )
    else:
        warnings.extend(
            _assign_fixed_counts(
                groups,
                counts=config.fixed_counts,
                priority=config.fixed_priority,
                config=config,
            )
        )
        _finish_number_remainder(groups, config.remainder_split)

    by_sample: dict[str, SplitAssignment] = {}
    for group in groups:
        if group.source is None:
            raise RuntimeError(f"Split group {group.display_key!r} was not planned.")
        for request in group.requests:
            by_sample[request.sample_id.casefold()] = SplitAssignment(
                sample_id=request.sample_id,
                split_group_key=request.split_group_key,
                assigned_split=group.assigned_split,
                source=group.source,
            )
    assignments = tuple(
        by_sample[request.sample_id.casefold()] for request in ordered_requests
    )
    for warning in warnings:
        warnings_module.warn(
            warning.message,
            SplitTargetWarning,
            stacklevel=2,
        )
    return SplitPlan(assignments=assignments, warnings=tuple(warnings))


__all__ = [
    "AssignmentSource",
    "SplitAssignment",
    "SplitPlan",
    "SplitTargetWarning",
    "SplitWarning",
    "plan_splits",
]
