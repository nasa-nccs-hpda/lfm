"""Chip provenance shared by concrete semantic and instance artifacts.

Prediction arrays and target equivalence remain task-specific. This metadata can
be checked without loading a model or importing a task package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class ChipMetadata:
    chip_id: str
    image_path: str
    label_path: str
    valid_mask: np.ndarray
    display_note: str = ""

    def validate_pair(
        self, other: ChipMetadata, *, paths: Literal["exact", "name"] = "exact"
    ) -> None:
        if self.chip_id != other.chip_id:
            raise ValueError(f"{self.chip_id}: model exports differ in chip_id.")
        if self.valid_mask.ndim != 2 or not np.array_equal(
            self.valid_mask, other.valid_mask
        ):
            raise ValueError(
                f"{self.chip_id}: model exports differ in validity or crop geometry."
            )
        for field in ("image_path", "label_path"):
            first, second = getattr(self, field), getattr(other, field)
            # Older display caches can omit provenance under filename matching.
            if paths == "name" and (not first or not second):
                continue
            matches = (
                first == second
                if paths == "exact"
                else Path(first).name == Path(second).name
            )
            if not matches:
                raise ValueError(f"{self.chip_id}: model exports differ in {field}.")
