"""General notebook utilities for Graha/Lunar-FM fine-tuning."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def create_timestamped_output_dir(base_dir: str | Path) -> Path:
    """Create a timestamped subdirectory under ``base_dir``."""
    timestamp = datetime.now().strftime("date_%Y_%m_%d-time_%H_%M_%S")
    output_dir = Path(base_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using output subdir: {output_dir}")
    return output_dir
