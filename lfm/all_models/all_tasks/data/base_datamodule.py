"""Shared datamodule layout helpers for split lunar datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SplitFolderDataLayout:
    """Resolve standard ``split/chips`` and ``split/labels`` data directories."""

    data_root: Path
    chips_subdir: str = "chips"
    labels_subdir: str = "labels"

    def split_chips_dir(self, split: str) -> Path:
        return self.data_root / split / self.chips_subdir

    def split_labels_dir(self, split: str) -> Path:
        return self.data_root / split / self.labels_subdir

    def flat_chips_dir(self) -> Path:
        return self.data_root / self.chips_subdir

    def flat_labels_dir(self) -> Path:
        return self.data_root / self.labels_subdir

    def split_dirs(self, split: str) -> tuple[Path, Path]:
        return self.split_chips_dir(split), self.split_labels_dir(split)

    def flat_dirs(self) -> tuple[Path, Path]:
        return self.flat_chips_dir(), self.flat_labels_dir()

    def has_split(self, split: str) -> bool:
        chips_dir, labels_dir = self.split_dirs(split)
        return chips_dir.exists() and labels_dir.exists()

    def missing_split_dirs(self, splits: list[str]) -> list[Path]:
        missing: list[Path] = []
        for split in splits:
            chips_dir, labels_dir = self.split_dirs(split)
            if not chips_dir.exists():
                missing.append(chips_dir)
            if not labels_dir.exists():
                missing.append(labels_dir)
        return missing

    def require_split_dirs(self, splits: list[str]) -> None:
        missing = self.missing_split_dirs(splits)
        if missing:
            raise FileNotFoundError(
                "Missing split data directories:\n"
                + "\n".join(str(path) for path in missing)
            )
