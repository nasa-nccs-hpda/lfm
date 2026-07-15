"""Custom semantic-segmentation tasks.

``LunarSegmentationTask`` — LLRD + split encoder/decoder LR
------------------------------------------------------------
Mirrors the same optimiser recipe used in ``LunarObjectDetectionTask``:

* **backbone_lr** — base LR for the top encoder layer; deeper layers get
  ``backbone_lr * layer_decay ** k`` (LLRD).
* **head_lr** — LR for the decoder, segmentation head, and auxiliary heads.
* **layer_decay** — per-layer decay factor inside the encoder. 1.0 = flat.
* Warmup + cosine annealing schedule.
* Bias / norm / positional params always skip weight decay.

Config note: do **not** set top-level ``optimizer:`` / ``lr_scheduler:`` when
using this task — those monkey-patch ``configure_optimizers`` and would
override the LLRD groups. Pass LR settings as task init args instead.

``LunarShapeSegmentationTask`` — per-crater roundness regulariser
-----------------------------------------------------------------
Inherits the LLRD setup from ``LunarSegmentationTask`` and adds an auxiliary
shape-compactness loss that penalises non-circular predicted crater blobs.
The regulariser operates per bounding-box ROI (bboxes passed through the
batch under the ``"crater_boxes"`` key by the ``LunarNACDTMDataset`` when
constructed with ``keep_boxes=True``), so it is not confused by multiple
craters in the same image.

Shape term (per bbox ROI)::

    perimeter = Σ √(dx² + dy²)                (L2 gradient magnitude)
    compactness = 4π · area / perimeter²      ∈ (0, 1], =1 for a disk
    L_shape     = clamp(1 − compactness, min=0)

The L2 magnitude of the pixel-space gradient integrates to the true
perimeter for a disk, so a filled disk scores ~1 (loss ~0) without any
correction constant.  L2 was chosen over the cheaper L1 TV variant because
axis-aligned rectangles are the L1-optimum (they score >1 and get clamped
to zero loss even though they're not round); the L2 form correctly
penalises them.

TV was chosen over Sobel-magnitude perimeter because ``|dx|+|dy|`` has
well-behaved gradients near zero and requires no epsilon inside a sqrt.

Turn the regulariser off by setting ``shape_loss_weight: 0`` — the class
then behaves exactly like ``LunarSegmentationTask`` (aside from the
``crater_boxes`` pop, which is a no-op when the key is absent).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from terratorch.tasks import SemanticSegmentationTask
from terratorch.tasks.segmentation_tasks import to_segmentation_prediction


# ---------------------------------------------------------------------------
# Shared LLRD helpers (mirrors lunar_object_detection_task.py)
# ---------------------------------------------------------------------------

_NO_DECAY_TOKENS = (
    "bias",
    "norm.weight",
    "layernorm.weight",
    "ln.weight",
    "register_tokens",
    "pos_emb",
    "pos_embed",
    "cls_token",
    "mod_emb",
)


def _no_decay(name: str) -> bool:
    lname = name.lower()
    return any(tok in lname for tok in _NO_DECAY_TOKENS)


def _encoder_block_index(name: str) -> int | None:
    """Return the integer block index if *name* sits inside an encoder block.

    Supports multiple ViT-style layouts:
      - Lunar FM:  ``...model.encoder.<i>.<rest>``
      - timm ViT:  ``...blocks.<i>.<rest>``
      - generic:   ``...layers.<i>.<rest>``
    Returns ``None`` for names that don't sit inside a recognised block.
    """
    for key in (".model.encoder.", ".blocks.", ".layers."):
        idx = name.find(key)
        if idx < 0:
            continue
        tail = name[idx + len(key):]
        first = tail.split(".", 1)[0]
        if first.isdigit():
            return int(first)
    return None


def _is_encoder_param(name: str) -> bool:
    """True for parameters that belong to the backbone/encoder.

    For a terratorch PixelWiseModel the module tree is::

        task.model
          .encoder   ← backbone (ViT, etc.)
          .decoder   ← neck / FPN / decoder head
          .head      ← final segmentation conv
          .aux_heads ← auxiliary decoders

    ``self.model.named_parameters()`` yields names starting with ``encoder.``
    for backbone params (the LunarBackbone wrapper adds its own ``.model.``
    below, so a full name looks like ``encoder.model.encoder.<i>....``).
    Everything else (decoder, head, aux_heads, neck) is head-side.
    """
    return name.startswith("encoder.")


def _is_new_modality_param(name: str, new_modality_names: list[str]) -> bool:
    """True for encoder-embedding params that belong to a *new* (randomly
    initialised) modality, i.e. one that was not present in the pretrained
    checkpoint.

    These live under ``encoder.model.encoder_embeddings.<mod_name>.``
    and must be trained at ``head_lr`` rather than the tiny LLRD layer-0 LR,
    because they start from random init and need to converge at the same rate
    as the freshly initialised decoder/head.

    The modality names come directly from ``LunarBackbone._new_modalities``
    (the same dict that was passed as ``backbone_new_modalities`` in the
    config), so the user never has to repeat them in the task config.
    """
    if not new_modality_names:
        return False
    return any(
        f"encoder_embeddings.{mod}." in name for mod in new_modality_names
    )


# ---------------------------------------------------------------------------
# LunarSegmentationTask
# ---------------------------------------------------------------------------

class LunarSegmentationTask(SemanticSegmentationTask):
    """Segmentation task with LLRD and split encoder/decoder param groups.

    Args:
        backbone_lr: Base LR for the top encoder layer (encoder_norm + last
            block). Deeper blocks get ``backbone_lr * layer_decay ** k``.
        head_lr: LR for decoder, segmentation head, and auxiliary heads.
        layer_decay: Per-layer LR decay factor inside the encoder. 1.0 disables
            LLRD (flat backbone LR).
        weight_decay: Weight decay applied to matrix params in the encoder.
        head_weight_decay: Weight decay for decoder / head params. Defaults to
            the same value as ``weight_decay`` if not set.
        warmup_steps: Linear warmup length in optimiser steps. 0 disables.
        cosine_t_max: Total steps for the cosine phase. If ``None``, falls back
            to ``trainer.estimated_stepping_batches - warmup_steps``.
        eta_min: Final LR floor for cosine annealing.
        betas: AdamW betas.

    All other keyword arguments are forwarded to
    :class:`~terratorch.tasks.SemanticSegmentationTask`.
    """

    def __init__(
        self,
        *args: Any,
        backbone_lr: float = 5.0e-5,
        head_lr: float = 2.0e-4,
        layer_decay: float = 0.75,
        weight_decay: float = 0.05,
        head_weight_decay: float | None = None,
        warmup_steps: int = 500,
        cosine_t_max: int | None = None,
        eta_min: float = 1.0e-6,
        betas: tuple[float, float] = (0.9, 0.98),
        **kwargs: Any,
    ) -> None:
        # Disable base-class optimizer/scheduler — we build them ourselves.
        kwargs.setdefault("optimizer", None)
        kwargs.setdefault("optimizer_hparams", None)
        kwargs.setdefault("scheduler", None)
        kwargs.setdefault("scheduler_hparams", None)
        super().__init__(*args, **kwargs)
        self.backbone_lr = float(backbone_lr)
        self.head_lr = float(head_lr)
        self.layer_decay = float(layer_decay)
        self.weight_decay = float(weight_decay)
        self.head_weight_decay = (
            float(head_weight_decay) if head_weight_decay is not None else float(weight_decay)
        )
        self.warmup_steps = int(warmup_steps)
        self.cosine_t_max = cosine_t_max
        self.eta_min = float(eta_min)
        self.betas = tuple(betas)

    # ------------------------------------------------------------------ utils

    def _get_new_modality_names(self) -> list[str]:
        """Read new-modality names directly from ``LunarBackbone._new_modalities``.

        For a PixelWiseModel the encoder *is* the LunarBackbone, so the path is
        ``task.model.encoder``. Returns an empty list when the encoder has no
        ``_new_modalities`` attribute or it is None/empty.
        """
        enc = self.model.encoder
        new_mods = getattr(enc, "_new_modalities", None) or {}
        return list(new_mods.keys())

    def _num_encoder_blocks(self) -> int:
        """Number of transformer blocks in the encoder.

        Falls back to counting ``model.encoder.blocks`` (timm-style) then
        ``model.encoder.model.encoder`` (Lunar FM style), then 12.
        """
        enc = self.model.encoder
        if hasattr(enc, "blocks"):
            return len(enc.blocks)
        if hasattr(enc, "model") and hasattr(enc.model, "encoder"):
            return len(enc.model.encoder)
        # generic fallback: scan named params for the highest block index
        max_idx = -1
        for name, _ in self.model.named_parameters():
            idx = _encoder_block_index(name)
            if idx is not None and idx > max_idx:
                max_idx = idx
        return max_idx + 1 if max_idx >= 0 else 12

    def _param_layer_id(self, name: str, num_blocks: int) -> int:
        """Integer layer id in ``[0, num_blocks + 1]``.

        0        = embeddings / register tokens / patch embed
        1..N     = encoder blocks (1 = deepest, N = shallowest)
        N+1      = encoder_norm (top of encoder)
        """
        blk = _encoder_block_index(name)
        if blk is not None:
            return blk + 1
        if "encoder_norm" in name:
            return num_blocks + 1
        return 0

    def _make_param_groups(self) -> list[dict[str, Any]]:
        num_blocks = self._num_encoder_blocks()
        top_layer = num_blocks + 1
        new_mod_names = self._get_new_modality_names()
        groups: dict[tuple[str, bool], dict[str, Any]] = {}

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            in_encoder = _is_encoder_param(name)
            nd = _no_decay(name)

            if in_encoder and _is_new_modality_param(name, new_mod_names):
                # Randomly-initialised embedding for a new modality: train at
                # head_lr so it converges at the same rate as the decoder/head.
                group_name = f"new_mod_emb_{'nd' if nd else 'wd'}"
                lr = self.head_lr
                wd = 0.0 if nd else self.head_weight_decay
            elif in_encoder:
                layer_id = self._param_layer_id(name, num_blocks)
                scale = self.layer_decay ** (top_layer - layer_id)
                lr = self.backbone_lr * scale
                wd = 0.0 if nd else self.weight_decay
                group_name = f"encoder_layer_{layer_id}_{'nd' if nd else 'wd'}"
            else:
                lr = self.head_lr
                wd = 0.0 if nd else self.head_weight_decay
                group_name = f"head_{'nd' if nd else 'wd'}"

            g = groups.setdefault(
                group_name,
                {"params": [], "lr": lr, "weight_decay": wd, "name": group_name},
            )
            g["params"].append(p)

        # Stable ordering: encoder first (deepest → shallowest), then head/new_mod.
        def _sort_key(g: dict) -> tuple:
            n = g["name"]
            if n.startswith("encoder_layer_"):
                # sort by layer id numerically
                try:
                    layer_num = int(n.split("_")[2])
                except (IndexError, ValueError):
                    layer_num = -1
                return (0, layer_num, n)
            return (1, 0, n)

        return sorted(groups.values(), key=_sort_key)

    # ---------------------------------------------------------- optim/sched

    def configure_optimizers(self):  # type: ignore[override]
        param_groups = self._make_param_groups()
        n_params = sum(sum(p.numel() for p in g["params"]) for g in param_groups)
        print(
            f"[LunarSegmentationTask] {len(param_groups)} param groups, "
            f"{n_params / 1e6:.1f}M trainable params. "
            f"layer_decay={self.layer_decay} backbone_lr={self.backbone_lr} "
            f"head_lr={self.head_lr}"
        )
        for g in param_groups:
            print(
                f"  {g['name']:<36s} lr={g['lr']:.2e} wd={g['weight_decay']:.2e} "
                f"n={sum(p.numel() for p in g['params']):,}"
            )

        optimizer = torch.optim.AdamW(param_groups, betas=self.betas)

        try:
            total_steps = int(self.trainer.estimated_stepping_batches)
        except Exception:
            total_steps = 0

        t_max = self.cosine_t_max
        if t_max is None:
            t_max = max(1, total_steps - self.warmup_steps) if total_steps else 10_000

        cosine = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=self.eta_min)
        if self.warmup_steps > 0:
            warmup = LinearLR(
                optimizer,
                start_factor=1.0 / max(self.warmup_steps, 1),
                end_factor=1.0,
                total_iters=self.warmup_steps,
            )
            scheduler = SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[self.warmup_steps]
            )
        else:
            scheduler = cosine

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


# ---------------------------------------------------------------------------
# Shape-loss helper
# ---------------------------------------------------------------------------

def crater_shape_loss(
    prob: Tensor,
    boxes_per_image: list[Tensor],
    *,
    pad_frac: float = 0.3,
    min_side: int = 4,
    eps: float = 1e-6,
) -> Tensor:
    """Per-crater compactness regulariser.

    Args:
        prob: ``(B, H, W)`` sigmoid/softmax probability of the crater class.
        boxes_per_image: list of ``(N_i, 4)`` tensors in ``[x1, y1, x2, y2]``
                         image-frame pixel coords (same frame as ``prob``).
        pad_frac: How much to enlarge each bbox before cropping the ROI.
                  Padding lets the perimeter term "see" the crater edge.
        min_side: Skip ROIs smaller than this in either dimension.
        eps: Small constant to avoid divide-by-zero when a crater is entirely
             absent from the prediction.

    Returns:
        Scalar tensor: mean per-crater ``(1 − compactness)``, in ``[0, 1)``.
        Zero when the prediction is a perfect disk; ~1 for very elongated
        blobs.  Returns ``prob.new_tensor(0.0)`` if no valid ROIs exist in the
        batch (e.g. every image has no craters).
    """
    if prob.dim() != 3:
        raise ValueError(f"prob must be (B, H, W), got {tuple(prob.shape)}")
    B, H, W = prob.shape
    if len(boxes_per_image) != B:
        raise ValueError(
            f"boxes_per_image has length {len(boxes_per_image)}, "
            f"but batch dim is {B}"
        )

    total = prob.new_tensor(0.0)
    count = 0
    for b, bxs in enumerate(boxes_per_image):
        if bxs is None or bxs.numel() == 0:
            continue
        for row in bxs.tolist():
            x1, y1, x2, y2 = row[:4]
            cw = max(x2 - x1, 0.0)
            ch = max(y2 - y1, 0.0)
            if cw < 1.0 or ch < 1.0:
                continue
            X1 = max(0, int(x1 - pad_frac * cw))
            Y1 = max(0, int(y1 - pad_frac * ch))
            X2 = min(W, int(x2 + pad_frac * cw))
            Y2 = min(H, int(y2 + pad_frac * ch))
            if (X2 - X1) < min_side or (Y2 - Y1) < min_side:
                continue
            patch = prob[b, Y1:Y2, X1:X2]
            area = patch.sum() + eps
            # Anti-alias the boundary with a light 2×2 mean before the
            # gradient: a plain discrete grid double-counts diagonal edges
            # vs axis-aligned ones (making squares score lower than disks
            # otherwise), and this small blur equalises the two.
            p = patch.unsqueeze(0).unsqueeze(0)
            p = torch.nn.functional.avg_pool2d(p, kernel_size=2, stride=1, padding=0)
            p = torch.nn.functional.pad(p, (0, 1, 0, 1))
            p = p.squeeze(0).squeeze(0)
            dx = p[:-1, 1:] - p[:-1, :-1]
            dy = p[1:, :-1] - p[:-1, :-1]
            perim = torch.sqrt(dx * dx + dy * dy + eps).sum()
            compactness = 4.0 * math.pi * area / (perim * perim + eps)
            total = total + (1.0 - compactness).clamp(min=0.0)
            count += 1

    if count == 0:
        return total  # zero, same device/dtype as prob
    return total / count


# ---------------------------------------------------------------------------
# LunarShapeSegmentationTask
# ---------------------------------------------------------------------------

class LunarShapeSegmentationTask(LunarSegmentationTask):
    """Segmentation task with LLRD **and** a per-crater roundness loss.

    Combines the encoder/decoder LR split from :class:`LunarSegmentationTask`
    with the shape-compactness regulariser originally in
    ``LunarSegmentationTask`` (the old single-class version of this file).

    Extra init args:
        shape_loss_weight: Multiplier for the shape regulariser. Set to 0 to
                           disable (recommended baseline: ``0.05``; scan
                           ``{0.02, 0.05, 0.1, 0.2}`` if needed).
        shape_loss_pad_frac: ROI padding around each bbox, as a fraction of
                             the bbox side.
        crater_boxes_key: Batch key holding the per-image list of bboxes.
                          Must match ``LunarNACDTMDataset(keep_boxes=True)``.
        crater_class_index: Index of the crater class in the softmax output.
                            Defaults to 1 (background=0, crater=1).
    """

    def __init__(
        self,
        *args: Any,
        shape_loss_weight: float = 0.05,
        shape_loss_pad_frac: float = 0.3,
        crater_boxes_key: str = "crater_boxes",
        crater_class_index: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.shape_loss_weight = float(shape_loss_weight)
        self.shape_loss_pad_frac = float(shape_loss_pad_frac)
        self.crater_boxes_key = crater_boxes_key
        self.crater_class_index = int(crater_class_index)

    # ---- helpers ----------------------------------------------------------

    def _pop_crater_boxes(self, batch: dict[str, Any]) -> list[Tensor] | None:
        """Detach crater bboxes from the batch so they're not forwarded into
        the model as kwargs.  Returns the popped value (or ``None``)."""
        return batch.pop(self.crater_boxes_key, None)

    def _add_shape_loss(
        self,
        base_loss: Tensor,
        model_output: Any,
        crater_boxes: list[Tensor] | None,
        stage: str,
    ) -> Tensor:
        """Add the shape regulariser to ``base_loss``. Logs the component
        under ``{stage}/shape_loss``. No-op when the weight is 0 or when
        the batch contains no bboxes."""
        if self.shape_loss_weight == 0.0 or not crater_boxes:
            return base_loss
        logits = model_output.output  # (B, C, H, W)
        prob = logits.softmax(dim=1)[:, self.crater_class_index]  # (B, H, W)
        shape_l = crater_shape_loss(
            prob,
            crater_boxes,
            pad_frac=self.shape_loss_pad_frac,
        )
        self.log(f"{stage}/shape_loss", shape_l, prog_bar=(stage == "train"))
        return base_loss + self.shape_loss_weight * shape_l

    # ---- lightning hooks --------------------------------------------------

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        crater_boxes = self._pop_crater_boxes(batch)
        x = batch["image"]
        y = self.squeeze_ground_truth(batch["mask"])
        other_keys = batch.keys() - {"image", "mask", "filename"}
        rest = {k: batch[k] for k in other_keys}
        model_output = self(x, **rest)
        loss = self.train_loss_handler.compute_loss(
            model_output, y, self.criterion, self.aux_loss
        )
        loss["loss"] = self._add_shape_loss(loss["loss"], model_output, crater_boxes, "train")
        self.train_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=y.shape[0])
        y_hat_hard = to_segmentation_prediction(model_output)
        self.train_metrics.update(y_hat_hard, y)
        return loss["loss"]

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        crater_boxes = self._pop_crater_boxes(batch)
        x = batch["image"]
        y = self.squeeze_ground_truth(batch["mask"])
        other_keys = batch.keys() - {"image", "mask", "filename"}
        rest = {k: batch[k] for k in other_keys}
        model_output = self.handle_full_or_tiled_inference(
            x, self.tiled_inference_on_validation, **rest
        )
        loss = self.val_loss_handler.compute_loss(
            model_output, y, self.criterion, self.aux_loss
        )
        loss["loss"] = self._add_shape_loss(loss["loss"], model_output, crater_boxes, "val")
        self.val_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=y.shape[0])
        y_hat_hard = to_segmentation_prediction(model_output)
        self.val_metrics.update(y_hat_hard, y)
        if self._do_plot_samples(batch_idx):
            batch["prediction"] = y_hat_hard
            self.plot_sample(batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # No shape loss at test time; just make sure the extra key doesn't
        # get forwarded into the model.
        self._pop_crater_boxes(batch)
        return super().test_step(batch, batch_idx, dataloader_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        self._pop_crater_boxes(batch)
        return super().predict_step(batch, batch_idx, dataloader_idx)
