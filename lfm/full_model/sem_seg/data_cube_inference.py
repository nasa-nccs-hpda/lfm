"""Sliding-window inference over paired WAC/static full-scene datacubes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from tiler import Merger, Tiler
import torch

from lfm.all_models.all_tasks.utils.common import (
    _extract_logits,
    _prediction_probabilities,
)
from lfm.toy_model.sem_seg.data_cube_inference import group_cubes_by_tile


WAC_TRAINING_ORDER = [2, 3, 4, 5, 6, 0, 1]


def find_scene_pairs(input_dir: str | Path) -> list[tuple[str, Path, Path]]:
    """Find the WAC/static file pair for every pipeline tile in a directory."""
    input_dir = Path(input_dir)
    paths = sorted(input_dir.rglob("*.tif"))
    if not paths:
        raise FileNotFoundError(f"No .tif files found beneath {input_dir}")
    grouped, _ = group_cubes_by_tile([str(path) for path in paths])
    pairs = []
    for tile_id, files in sorted(grouped.items()):
        if len(files["wac"]) != 1 or len(files["static"]) != 1:
            raise ValueError(
                f"Tile {tile_id} must have exactly one WAC and one static cube; "
                f"found {len(files['wac'])} WAC and {len(files['static'])} static."
            )
        pairs.append((tile_id, Path(files["wac"][0]), Path(files["static"][0])))
    if not pairs:
        raise FileNotFoundError(
            f"No paired WAC/static pipeline tiles were found beneath {input_dir}"
        )
    return pairs


def read_scene(
    wac_path: Path,
    static_path: Path,
    *,
    band_filter: list[int] | None,
    excluded_nodata_values: list[float] | tuple[float, ...] = (),
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read one pair in the same 5 VIS + 2 UV + 63 static order as training."""
    with rasterio.open(wac_path) as src:
        if src.count < 7:
            raise ValueError(f"Expected at least 7 WAC bands in {wac_path}, got {src.count}")
        wac = src.read([index + 1 for index in WAC_TRAINING_ORDER]).astype(np.float32)
        wac_nodata = src.nodata
        profile = src.profile.copy()
    with rasterio.open(static_path) as src:
        static = src.read().astype(np.float32)
        static_nodata = src.nodata

    if wac.shape[1:] != static.shape[1:]:
        raise ValueError(
            f"WAC/static spatial shapes differ: {wac.shape[1:]} vs {static.shape[1:]}"
        )
    image = np.concatenate([wac, static], axis=0)
    if band_filter is not None:
        if max(band_filter) >= image.shape[0]:
            raise IndexError(
                f"Band filter requires channel {max(band_filter)}, but the paired "
                f"scene contains only {image.shape[0]} channels."
            )
        image = image[band_filter]

    invalid = ~np.isfinite(image)
    for value in (wac_nodata, static_nodata, *excluded_nodata_values):
        if value is not None:
            invalid |= image == float(value)
    invalid_pixels = invalid.any(axis=0)
    image[invalid] = 0.0
    return image, invalid_pixels, profile


def sliding_window_predict(
    image: np.ndarray,
    model: torch.nn.Module,
    *,
    means: list[float],
    stds: list[float],
    device: torch.device,
    tile_size: int = 256,
    overlap: float = 0.25,
    threshold: float = 0.5,
    batch_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize, predict overlapping tiles, and blend foreground probabilities."""
    if image.ndim != 3:
        raise ValueError(f"Expected CHW input, got shape {image.shape}")
    means_array = np.asarray(means, dtype=np.float32).reshape(-1, 1, 1)
    stds_array = np.asarray(stds, dtype=np.float32).reshape(-1, 1, 1)
    if means_array.shape[0] != image.shape[0] or stds_array.shape[0] != image.shape[0]:
        raise ValueError(
            f"Normalization has {means_array.shape[0]} channels, but scene has "
            f"{image.shape[0]} selected channels."
        )
    if np.any(stds_array <= 0):
        raise ValueError("All normalization standard deviations must be positive.")
    normalized = ((image - means_array) / stds_array).transpose(1, 2, 0)

    height, width, channels = normalized.shape
    input_tiler = Tiler(
        data_shape=normalized.shape,
        tile_shape=(tile_size, tile_size, channels),
        channel_dimension=-1,
        overlap=overlap,
        mode="reflect",
    )
    output_tiler = Tiler(
        data_shape=(height, width, 1),
        tile_shape=(tile_size, tile_size, 1),
        channel_dimension=-1,
        overlap=overlap,
        mode="reflect",
    )
    merger = Merger(output_tiler, window="triang")
    model.eval()
    with torch.inference_mode():
        for tile_ids, tile_batch in input_tiler(
            normalized, batch_size=batch_size, progress_bar=True
        ):
            inputs = torch.from_numpy(tile_batch).permute(0, 3, 1, 2).float().to(device)
            probabilities = _prediction_probabilities(_extract_logits(model(inputs)))
            probabilities = probabilities.detach().cpu().numpy()[..., None]
            for tile_id, probability in zip(np.atleast_1d(tile_ids), probabilities):
                merger.add(int(tile_id), probability)
    probability = merger.merge(unpad=True).squeeze(-1).astype(np.float32)
    prediction = (probability > threshold).astype(np.uint8)
    return prediction, probability


def write_scene_outputs(
    output_dir: str | Path,
    tile_id: str,
    profile: dict,
    prediction: np.ndarray,
    probability: np.ndarray,
    invalid_pixels: np.ndarray,
) -> tuple[Path, Path]:
    """Write georeferenced binary prediction and probability GeoTIFFs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction = prediction.copy()
    probability = probability.copy()
    prediction[invalid_pixels] = 255
    probability[invalid_pixels] = -9999.0

    mask_path = output_dir / f"Tile-{tile_id}_semantic_mask.tif"
    probability_path = output_dir / f"Tile-{tile_id}_semantic_probability.tif"
    mask_profile = {**profile, "count": 1, "dtype": "uint8", "nodata": 255}
    probability_profile = {
        **profile,
        "count": 1,
        "dtype": "float32",
        "nodata": -9999.0,
    }
    with rasterio.open(mask_path, "w", **mask_profile) as dst:
        dst.write(prediction, 1)
    with rasterio.open(probability_path, "w", **probability_profile) as dst:
        dst.write(probability, 1)
    return mask_path, probability_path


def run_datacube_inference(
    *,
    model: torch.nn.Module,
    device: torch.device,
    input_dir: str | Path,
    output_dir: str | Path,
    band_filter: list[int] | None,
    means: list[float],
    stds: list[float],
    excluded_nodata_values: list[float] | tuple[float, ...] = (),
    tile_size: int = 256,
    overlap: float = 0.25,
    threshold: float = 0.5,
    batch_size: int = 1,
) -> list[dict[str, Path | str]]:
    """Run inference for all paired full-scene datacubes beneath ``input_dir``."""
    results = []
    pairs = find_scene_pairs(input_dir)
    for index, (tile_id, wac_path, static_path) in enumerate(pairs, start=1):
        print(f"Processing scene {index}/{len(pairs)}: Tile-{tile_id}")
        image, invalid_pixels, profile = read_scene(
            wac_path,
            static_path,
            band_filter=band_filter,
            excluded_nodata_values=excluded_nodata_values,
        )
        prediction, probability = sliding_window_predict(
            image,
            model,
            means=means,
            stds=stds,
            device=device,
            tile_size=tile_size,
            overlap=overlap,
            threshold=threshold,
            batch_size=batch_size,
        )
        mask_path, probability_path = write_scene_outputs(
            output_dir,
            tile_id,
            profile,
            prediction,
            probability,
            invalid_pixels,
        )
        results.append(
            {
                "tile_id": tile_id,
                "wac": wac_path,
                "static": static_path,
                "mask": mask_path,
                "probability": probability_path,
            }
        )
        print(f"Saved {mask_path} and {probability_path}")
    return results
