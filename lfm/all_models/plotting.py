"""Import-light display policies, figure lifetime, overlays and XYXY boxes.

Uses NumPy/Matplotlib only. Task renderers retain colorization and metrics.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


@dataclass(frozen=True)
class DisplayPolicy:
    bands: tuple[int, ...]
    stretch: (
        Literal["global_minmax", "channel_minmax", "percentile", "std_clip"] | None
    ) = "global_minmax"
    clip_percentile: float = 2.0
    std_clip: float = 3.0
    epsilon: float = 0.0
    validate_finite: bool = True


def prepare_display(
    image: np.ndarray, *, policy: DisplayPolicy, valid: np.ndarray | None = None
) -> np.ndarray:
    """Select HWC bands and stretch using only valid pixels; invalid output is 0."""
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[:, :, None]
    values = image[:, :, policy.bands]
    valid = (
        np.ones(image.shape[:2], dtype=bool)
        if valid is None
        else np.asarray(valid, dtype=bool)
    )
    if valid.shape != image.shape[:2] or not valid.any():
        raise ValueError("Display requires an aligned nonempty valid footprint.")
    if policy.validate_finite and not np.isfinite(values[valid]).all():
        raise ValueError("Non-finite raw display values inside the valid footprint.")
    if policy.stretch is None:
        result = values.copy()
    elif policy.stretch == "std_clip":
        result = (
            np.clip(values, -policy.std_clip, policy.std_clip) + policy.std_clip
        ) / (2 * policy.std_clip)
    elif policy.stretch in ("global_minmax", "channel_minmax", "percentile"):
        result = np.zeros_like(values, dtype=np.float32)
        for index in range(values.shape[-1]):
            band = values[:, :, index]
            selected = (
                values[valid] if policy.stretch == "global_minmax" else band[valid]
            )
            if policy.stretch == "percentile":
                low, high = np.percentile(
                    selected, [policy.clip_percentile, 100 - policy.clip_percentile]
                )
            else:
                low, high = selected.min(), selected.max()
            if high > low or policy.epsilon:
                selected_band = (
                    np.clip(band[valid], low, high)
                    if policy.stretch == "percentile"
                    else band[valid]
                )
                result[:, :, index][valid] = (selected_band - low) / (
                    high - low + policy.epsilon
                )
    else:
        raise ValueError(f"Unknown display method: {policy.stretch}")
    result = np.clip(result, 0, 1)
    result[~valid] = 0
    return result[:, :, 0] if len(policy.bands) == 1 else result


def blend_overlay(
    image: np.ndarray, colors, *, foreground: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    if not 0 <= alpha <= 1:
        raise ValueError("Overlay alpha must be in [0, 1].")
    rgb = np.repeat(image[:, :, None], 3, axis=2) if image.ndim == 2 else image
    foreground = np.asarray(foreground, dtype=bool)
    if foreground.shape != rgb.shape[:2]:
        raise ValueError("Overlay foreground must align with image.")
    return np.clip(
        np.where(
            foreground[:, :, None], (1 - alpha) * rgb + alpha * np.asarray(colors), rgb
        ),
        0,
        1,
    )


def draw_boxes(
    ax,
    boxes: np.ndarray,
    *,
    coordinates: Literal["xyxy", "xywh"] = "xyxy",
    color: str = "red",
    linewidth: float = 1,
    scores: np.ndarray | None = None,
) -> None:
    """Draw pixel extents; XYXY uses exclusive right and bottom coordinates."""
    if coordinates not in ("xyxy", "xywh"):
        raise ValueError("Boxes require explicit xyxy or xywh coordinates.")
    for index, (x1, y1, a, b) in enumerate(np.asarray(boxes).reshape(-1, 4)):
        width, height = (a - x1, b - y1) if coordinates == "xyxy" else (a, b)
        if width <= 0 or height <= 0:
            continue
        ax.add_patch(
            Rectangle(
                (x1, y1),
                width,
                height,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
            )
        )
        if scores is not None and index < len(scores):
            ax.text(
                x1,
                y1,
                f"{float(scores[index]):.2f}",
                color=color,
                fontsize=7,
                bbox={
                    "facecolor": "black",
                    "alpha": 0.45,
                    "pad": 1,
                    "edgecolor": "none",
                },
            )


@dataclass(frozen=True)
class FigureGrid:
    figure: Figure
    axes: np.ndarray

    def save(
        self,
        output: str | Path,
        *,
        dpi: int = 150,
        facecolor: str = "white",
        bbox_inches: str | None = None,
    ) -> Path:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(
            output, dpi=dpi, facecolor=facecolor, bbox_inches=bbox_inches
        )
        return output


@contextmanager
def figure_grid(rows: int, columns: int, *, figsize):
    """Always yield 2-D axes and close the figure, including rendering errors."""
    figure, axes = plt.subplots(rows, columns, figsize=figsize, squeeze=False)
    figure.patch.set_facecolor("white")
    try:
        yield FigureGrid(figure, axes)
    finally:
        plt.close(figure)
