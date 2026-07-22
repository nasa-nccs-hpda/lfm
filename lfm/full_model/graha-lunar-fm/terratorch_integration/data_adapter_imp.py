"""IMP segmentation dataset and datamodule for TerraTorch integration.

Provides :class:`LunarImpSegDataset` and :class:`LunarImpSegDataModule` for
semantic segmentation of Impact Melt Ponds (IMP) from LRO NAC images.

Dataset layout expected on disk::

    <root>/
        instances_train.json
        instances_val.json
        instances_test.json
        train/   *.nc
        val/     *.nc
        test/    *.nc
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from terratorch_integration.data_utils import load_nc_band

# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------


class Normalize:
    """Normalise a batch's ``"image"`` tensor with per-channel mean / std.

    Accepts batches where ``batch["image"]`` is either a list of
    ``(C, H, W)`` tensors (pre-stack) or an already-stacked
    ``(B, C, H, W)`` tensor.

    Args:
        means: Per-channel mean values.  Required unless ``per_image=True``.
        stds: Per-channel std values.  Required unless ``per_image=True``.
        max_pixel_value: If given, images are divided by this value before
            mean/std normalisation.
        per_image: If ``True``, normalise each image by its own mean and std
            rather than using fixed statistics.
    """

    def __init__(
        self,
        means: list[float] | None = None,
        stds: list[float] | None = None,
        max_pixel_value: float | None = None,
        per_image: bool = False,
    ) -> None:
        if not per_image and (means is None or stds is None):
            raise ValueError("means and stds must be provided when per_image=False")
        self.means = means
        self.stds = stds
        self.max_pixel_value = max_pixel_value
        self.per_image = per_image

    def _normalize_per_image(self, img: torch.Tensor) -> torch.Tensor:
        mean = img.mean(dim=(1, 2), keepdim=True)
        std = img.std(dim=(1, 2), keepdim=True)
        std = torch.where(std > 0, std, torch.ones_like(std))
        return (img - mean) / std

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        if self.per_image:
            imgs = []
            for img in batch["image"]:
                if self.max_pixel_value is not None:
                    img = img / self.max_pixel_value
                imgs.append(self._normalize_per_image(img))
            batch["image"] = torch.stack(imgs)
        else:
            batch["image"] = torch.stack(list(batch["image"]))
            image = (
                batch["image"] / self.max_pixel_value
                if self.max_pixel_value is not None
                else batch["image"]
            )
            if image.ndim == 4:
                means = torch.tensor(self.means, device=image.device).view(1, -1, 1, 1)
                stds = torch.tensor(self.stds, device=image.device).view(1, -1, 1, 1)
            elif image.ndim == 5:
                means = torch.tensor(self.means, device=image.device).view(
                    1, -1, 1, 1, 1
                )
                stds = torch.tensor(self.stds, device=image.device).view(1, -1, 1, 1, 1)
            else:
                raise ValueError(f"Expected 4- or 5-D image tensor, got {image.ndim}D")
            batch["image"] = (image - means) / stds
        return batch


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class LunarImpSegDataset(Dataset):
    """Lunar IMP semantic segmentation dataset.

    Converts COCO-format instance annotations (polygons) into a flat
    semantic segmentation mask:

    * ``0`` → background (NO_IMP / negative patch)
    * ``1`` → IMP foreground

    Images are single-channel float32 ``(1, H, W)`` tensors loaded from
    NetCDF (``.nc``) files.  Masks are int64 ``(H, W)`` tensors.

    Args:
        root: Root directory containing ``train/``, ``val/``, ``test/``
            sub-folders and the three ``instances_*.json`` annotation files.
        split: One of ``"train"``, ``"val"``, or ``"test"``.
        image_size: If given, images and masks are resized to this square
            size (images with BILINEAR, masks with NEAREST).
        transforms: Optional callable applied to the returned sample dict
            after all other processing.
        mask_output_tag: Key used for the segmentation mask in the returned
            sample dict.
        band_name: HDF5 dataset key inside each ``.nc`` file that holds the
            image band.  Defaults to ``"data"`` for the IMP tiles.
        fraction: Retain only this fraction of images, e.g. ``0.1`` for
            10 %.  Must be in ``(0, 1]``.  ``None`` keeps all images.
        fraction_seed: RNG seed used when ``fraction`` is set, for
            reproducible subsampling across runs.
    """

    _ANN_FILE: dict[str, str] = {
        "train": "instances_train.json",
        "val": "instances_val.json",
        "test": "instances_test.json",
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int | None = None,
        transforms: Callable | None = None,
        mask_output_tag: str = "mask",
        band_name: str = "data",
        fraction: float | None = None,
        fraction_seed: int = 0,
    ) -> None:
        if split not in self._ANN_FILE:
            raise ValueError(
                f"split must be one of {list(self._ANN_FILE)}, got '{split}'"
            )
        if fraction is not None and not (0 < fraction <= 1):
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")

        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.transforms = transforms
        self.mask_output_tag = mask_output_tag
        self.band_name = band_name
        self.fraction = fraction
        self.fraction_seed = fraction_seed
        self.img_dir = self.root / split

        ann_path = self.root / self._ANN_FILE[split]
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")

        with open(ann_path) as f:
            coco = json.load(f)

        images: list[dict] = coco["images"]

        # Optional reproducible subsampling (training data reduction experiments)
        if fraction is not None and fraction < 1.0:
            rng = np.random.default_rng(fraction_seed)
            n = max(1, round(len(images) * fraction))
            indices = rng.choice(len(images), size=n, replace=False)
            indices.sort()  # preserve original file ordering
            images = [images[i] for i in indices]

        self.images: list[dict] = images

        # Build image-id → list-of-annotations mapping
        self.img_id_to_anns: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            self.img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, Any]:
        img_info = self.images[index]
        img_id = img_info["id"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        # Load image band as float32 (H, W); NetCDF may be (1, H, W) or (H, W).
        image_array = load_nc_band(self.img_dir / img_info["file_name"], self.band_name)

        # Rasterise COCO polygons → binary semantic mask (0 = background, 1 = IMP).
        # Filling with a fixed value of 1 avoids leaking arbitrary COCO
        # category_id values into what must be a two-class target.
        mask_pil = Image.new("L", (img_w, img_h), 0)
        draw = ImageDraw.Draw(mask_pil)
        for ann in self.img_id_to_anns.get(img_id, []):
            for polygon in ann["segmentation"]:
                if len(polygon) < 6:
                    continue
                coords = [
                    (polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)
                ]
                draw.polygon(coords, fill=1)

        # Optional resize.  Use PIL mode 'F' so float32 values pass through
        # without being reinterpreted as uint8 (mode 'L').
        if self.image_size is not None:
            image_array = np.asarray(
                Image.fromarray(image_array, mode="F").resize(
                    (self.image_size, self.image_size), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            )
            mask_pil = mask_pil.resize(
                (self.image_size, self.image_size), Image.Resampling.NEAREST
            )

        # np.array(...) below forces a writable copy so PyTorch does not warn
        # about non-writable NumPy → Tensor conversions (PIL's buffer may be
        # read-only after resize).
        sample: dict[str, Any] = {
            "image": torch.from_numpy(
                np.array(image_array, dtype=np.float32, copy=True)
            ).unsqueeze(0),
            self.mask_output_tag: torch.from_numpy(
                np.array(mask_pil, dtype=np.int64, copy=True)
            ),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: str | None = None,
        mask_alpha: float = 0.45,
    ) -> Figure:
        """Render a sample as a matplotlib Figure for TensorBoard / inspection.

        Produces **1 panel** (GT overlay) when only ground-truth is present,
        or **2 panels** (GT | Prediction) when
        ``"prediction_{mask_output_tag}"`` is also in the sample dict.

        The figure can be logged directly to TensorBoard::

            writer.add_figure("val/sample", fig, global_step)

        Args:
            sample: Dict as returned by :meth:`__getitem__`.  Must contain
                ``"image"`` and ``self.mask_output_tag``.  Optionally also
                ``"prediction_{self.mask_output_tag}"`` for a second panel.
            show_titles: Whether to show panel titles.
            suptitle: Optional overall figure title.
            mask_alpha: Opacity of the mask overlay (0 = transparent, 1 = opaque).

        Returns:
            A :class:`matplotlib.figure.Figure`.
        """
        img = sample["image"]
        if isinstance(img, torch.Tensor):
            img = img.squeeze().cpu().numpy()
        img = img.astype(float)
        lo, hi = img.min(), img.max()
        if hi > lo:
            img = (img - lo) / (hi - lo)

        # 0 → transparent, 1 → IMP orange
        cmap = ListedColormap(["none", "#f97316"])

        # TerraTorch's SemanticSegmentationTask writes predictions under the
        # plain key "prediction"; earlier detection code used
        # "prediction_{mask_output_tag}".  Accept either.
        pred_key: str | None = None
        for k in (f"prediction_{self.mask_output_tag}", "prediction"):
            if k in sample:
                pred_key = k
                break
        has_pred = pred_key is not None
        ncols = 2 if has_pred else 1

        fig, axs = plt.subplots(
            1, ncols, squeeze=False, figsize=(ncols * 6, 6), tight_layout=True
        )

        def _render(ax: Any, mask_tensor: Any, title: str) -> None:
            ax.imshow(img, cmap="gray", interpolation="nearest")
            if isinstance(mask_tensor, torch.Tensor):
                mask_np = mask_tensor.squeeze().cpu().numpy()
            else:
                mask_np = np.asarray(mask_tensor)
            ax.imshow(
                np.clip(mask_np, 0, 1).astype(float),
                cmap=cmap,
                vmin=0,
                vmax=1,
                alpha=mask_alpha,
                interpolation="nearest",
            )
            ax.axis("off")
            if show_titles:
                ax.set_title(title, fontsize=12, fontweight="bold")

        _render(axs[0, 0], sample[self.mask_output_tag], "Ground Truth")
        if has_pred:
            _render(axs[0, 1], sample[pred_key], "Prediction")

        if suptitle:
            fig.suptitle(suptitle, fontsize=13)

        return fig


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class LunarImpSegDataModule(LightningDataModule):
    """Lightning DataModule for the IMP semantic segmentation dataset.

    Wraps :class:`LunarImpSegDataset` for train / val / test splits and
    applies normalisation via :class:`Normalize`.

    Val and test splits are always loaded in full; ``train_fraction`` only
    affects the training split.

    Args:
        root: Root directory of the IMP segmentation dataset.
        batch_size: Samples per batch.
        num_workers: DataLoader worker processes.
        image_size: Optional square resize applied to images and masks.
        mask_output_tag: Key used for the segmentation mask tensor.
        band_name: HDF5 dataset key inside each ``.nc`` file.  Passed through
            to :class:`LunarImpSegDataset`.
        norm_means: Per-channel means.  Defaults to ``[0.5]``.
        norm_stds: Per-channel stds.  Defaults to ``[0.25]``.
        max_pixel_value: Value used to scale images to ``[0, 1]`` before
            mean/std normalisation.  Defaults to ``1.0`` (no rescaling); the
            IMP ``.nc`` tiles are already stored in a normalised range.
        per_image_norm: If ``True``, normalise each image by its own mean
            and std instead of fixed statistics.
        train_transforms: Optional transform for training samples.
        val_transforms: Optional transform for validation samples.
        test_transforms: Optional transform for test samples (falls back to
            ``val_transforms`` if not set).
        train_fraction: Retain only this fraction of training images, e.g.
            ``0.1`` for 10 %.  ``None`` keeps all.
        fraction_seed: RNG seed for reproducible subsampling.
    """

    def __init__(
        self,
        root: str,
        batch_size: int = 8,
        num_workers: int = 4,
        image_size: int | None = None,
        mask_output_tag: str = "mask",
        band_name: str = "data",
        norm_means: list[float] | None = None,
        norm_stds: list[float] | None = None,
        max_pixel_value: float = 1.0,
        per_image_norm: bool = False,
        train_transforms: Callable | None = None,
        val_transforms: Callable | None = None,
        test_transforms: Callable | None = None,
        train_fraction: float | None = None,
        fraction_seed: int = 0,
    ) -> None:
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.mask_output_tag = mask_output_tag
        self.band_name = band_name
        self.train_fraction = train_fraction
        self.fraction_seed = fraction_seed
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms or val_transforms

        if per_image_norm:
            self.aug = Normalize(per_image=True, max_pixel_value=max_pixel_value)
        else:
            self.aug = Normalize(
                means=norm_means or [0.5],
                stds=norm_stds or [0.25],
                max_pixel_value=max_pixel_value,
            )

        self.train_dataset: LunarImpSegDataset | None = None
        self.val_dataset: LunarImpSegDataset | None = None
        self.test_dataset: LunarImpSegDataset | None = None

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = LunarImpSegDataset(
                root=self.root,
                split="train",
                image_size=self.image_size,
                transforms=self.train_transforms,
                mask_output_tag=self.mask_output_tag,
                band_name=self.band_name,
                fraction=self.train_fraction,
                fraction_seed=self.fraction_seed,
            )
            self.val_dataset = LunarImpSegDataset(
                root=self.root,
                split="val",
                image_size=self.image_size,
                transforms=self.val_transforms,
                mask_output_tag=self.mask_output_tag,
                band_name=self.band_name,
            )

        if stage == "validate":
            self.val_dataset = LunarImpSegDataset(
                root=self.root,
                split="val",
                image_size=self.image_size,
                transforms=self.val_transforms,
                mask_output_tag=self.mask_output_tag,
                band_name=self.band_name,
            )

        if stage in ("test", "predict"):
            self.test_dataset = LunarImpSegDataset(
                root=self.root,
                split="test",
                image_size=self.image_size,
                transforms=self.test_transforms,
                mask_output_tag=self.mask_output_tag,
                band_name=self.band_name,
            )

    def _collate(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Stack images and masks, then normalise."""
        collated: dict[str, Any] = {
            "image": [item["image"] for item in batch],
            self.mask_output_tag: torch.stack(
                [item[self.mask_output_tag] for item in batch]
            ),
        }
        return self.aug(collated)

    def _dataloader_factory(self, split: str) -> DataLoader:
        dataset_attr = f"{split}_dataset"
        if getattr(self, dataset_attr, None) is None:
            # Map dataloader split names to Lightning stage names accepted by setup().
            stage = {"train": "fit", "val": "validate", "test": "test"}[split]
            self.setup(stage)
        return DataLoader(
            dataset=getattr(self, dataset_attr),
            batch_size=self.batch_size,
            shuffle=(split == "train"),
            num_workers=self.num_workers,
            collate_fn=self._collate,
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader_factory("train")

    def val_dataloader(self) -> DataLoader:
        return self._dataloader_factory("val")

    def test_dataloader(self) -> DataLoader:
        return self._dataloader_factory("test")

    def plot(self, sample: dict[str, Any], **kwargs: Any) -> Figure:
        """Delegate to :meth:`LunarImpSegDataset.plot`.

        TerraTorch's segmentation task calls ``datamodule.plot(sample)`` when
        logging validation images, so this proxy satisfies that protocol.
        """
        dataset = self.val_dataset or self.train_dataset or self.test_dataset
        if dataset is None:
            raise RuntimeError("No dataset available; call setup() first.")
        return dataset.plot(sample, **kwargs)
