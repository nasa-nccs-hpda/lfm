"""Prediction-cache helpers shared through the all-tasks utility API.

Semantic cache implementation lives here. Instance cache implementation lives
under ``lfm.all_models.inst_seg.instance_prediction_cache`` and is exposed lazily
through this shared utility API.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from lfm.all_models.all_tasks.utils.common import (
    _extract_image_and_mask,
    _extract_logits,
    _extract_paths,
    _get_split_dataloader,
    _move_batch_to_device,
    _prediction_probabilities,
    _sample_key,
)


def save_toy_instance_prediction_cache(*args, **kwargs):
    from lfm.all_models.inst_seg.instance_prediction_cache import (
        save_toy_instance_prediction_cache,
    )

    return save_toy_instance_prediction_cache(*args, **kwargs)


def save_graha_instance_prediction_cache(*args, **kwargs):
    from lfm.all_models.inst_seg.instance_prediction_cache import (
        save_graha_instance_prediction_cache,
    )

    return save_graha_instance_prediction_cache(*args, **kwargs)


def _load_instance_prediction_cache(*args, **kwargs):
    from lfm.all_models.inst_seg.instance_prediction_cache import (
        _load_instance_prediction_cache,
    )

    return _load_instance_prediction_cache(*args, **kwargs)


def save_prediction_cache(
    task,
    datamodule,
    output_dir: str | Path,
    *,
    model_name: str,
    split: str = "val",
    n_samples: int = 20,
    setup_datamodule: bool = True,
) -> Path:
    """Run one model over a split and save lightweight prediction .npz files."""
    cache_dir = Path(output_dir) / "prediction_cache" / model_name / split
    cache_dir.mkdir(parents=True, exist_ok=True)

    if setup_datamodule:
        datamodule.setup("fit" if split in {"train", "val"} else split)
    dataloader = _get_split_dataloader(datamodule, split)
    was_training = task.training

    task.eval()
    manifest = []
    saved = 0
    device = task.device
    with torch.no_grad():

        for batch in dataloader:

            image_paths, label_paths = _extract_paths(batch)

            batch = _move_batch_to_device(batch, device)

            x, y = _extract_image_and_mask(batch)

            output = _extract_logits(task(x))

            probs = _prediction_probabilities(output)

            preds = (probs > 0.5).long()

            images = x.detach().cpu()

            labels = y.detach().cpu()

            probs = probs.detach().cpu()

            preds = preds.detach().cpu()

            for i in range(images.shape[0]):

                if saved >= n_samples:

                    break

                image_path = image_paths[i] if i < len(image_paths) else None

                label_path = label_paths[i] if i < len(label_paths) else None

                sample_key = _sample_key(image_path, saved)

                filename = f"{saved:04d}_{sample_key}.npz"

                save_path = cache_dir / filename

                np.savez_compressed(
                    save_path,
                    image=images[i].numpy(),
                    label=labels[i].numpy(),
                    pred=preds[i].numpy(),
                    prob=probs[i].numpy(),
                    sample_key=sample_key,
                    model_name=model_name,
                    image_path=image_path or "",
                    label_path=label_path or "",
                )

                manifest.append(
                    {
                        "index": saved,
                        "sample_key": sample_key,
                        "file": filename,
                        "image_path": image_path,
                        "label_path": label_path,
                    }
                )

                saved += 1

            if saved >= n_samples:

                break

    task.train(was_training)
    manifest_path = cache_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {saved} prediction cache file(s) to {cache_dir}")
    return cache_dir


def _load_prediction_cache(cache_dir: str | Path) -> dict[str, dict]:
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        files = [cache_dir / item["file"] for item in manifest]
    else:

        files = sorted(cache_dir.glob("*.npz"))
    samples = {}
    for path in files:

        data = np.load(path, allow_pickle=False)

        sample_key = str(data["sample_key"])

        samples[sample_key] = {
            "file": path,
            "image": data["image"],
            "label": data["label"],
            "pred": data["pred"],
            "prob": data["prob"],
            "image_path": str(data["image_path"]),
            "label_path": str(data["label_path"]),
        }
    return samples
