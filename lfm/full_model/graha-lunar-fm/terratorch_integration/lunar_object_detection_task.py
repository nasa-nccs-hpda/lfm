"""ObjectDetectionTask with layer-wise LR decay (LLRD) and split param groups.

Standard fine-tuning uses one LR for everything, which is wrong for a
ViT+detector: the pretrained backbone wants a small LR (to avoid destroying
learned features), while the freshly-initialised RPN and ROI heads want a
larger one. LLRD additionally decays LR by depth inside the backbone, so
bottom encoder layers (generic features) train slower than top ones (task-
specific features). Recipe follows ViTDet / BEiT / MAE-detection.

Config-level notes:
- Do NOT set top-level ``optimizer:`` / ``lr_scheduler:`` in the Lightning
  CLI config. Those monkey-patch ``configure_optimizers`` and would clobber
  the LLRD groups. Configure them via task init args instead.
- Params matching ``no_decay_names`` (biases, norms, register tokens,
  pos embeddings) skip weight decay regardless of their LR group.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from terratorch.tasks import ObjectDetectionTask
from torchvision.models.detection.rpn import AnchorGenerator

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
    """Return the integer block index if ``name`` sits inside an encoder block,
    otherwise ``None``. Matches ``...model.encoder.<i>.<rest>``."""
    key = ".model.encoder."
    idx = name.find(key)
    if idx < 0:
        return None
    tail = name[idx + len(key) :]
    first = tail.split(".", 1)[0]
    if first.isdigit():
        return int(first)
    return None


def _is_backbone_param(name: str) -> bool:
    # Module tree with Faster-R-CNN:
    #   task.model = ObjectDetectionModel
    #     .torchvision_model = FasterRCNN
    #       .backbone = BackboneWrapper
    #         .backbone = LunarBackbone
    #           .model = TerraMind (encoder, embeddings, register_tokens, ...)
    # Everything under `torchvision_model.backbone.backbone.model.` is the
    # pretrained encoder + its embeddings. The FPN/neck sits at
    # `torchvision_model.backbone.necks.*` and is treated as a head param.
    return "torchvision_model.backbone.backbone.model." in name


class LunarObjectDetectionTask(ObjectDetectionTask):
    """Object detection task with LLRD and split backbone/head param groups.

    Args:
        backbone_lr: Base LR for the top backbone layer (encoder_norm + last
            block). Deeper blocks get ``backbone_lr * layer_decay ** k``.
        head_lr: LR for necks, RPN, ROI heads, and anything not inside the
            backbone.
        layer_decay: Per-layer LR decay factor. 1.0 disables LLRD.
        weight_decay: Weight decay applied to matrix params (norms, biases,
            and positional/register tokens are always excluded).
        warmup_steps: Linear warmup length in optimizer steps. 0 disables.
        cosine_t_max: Total steps for the cosine phase. If ``None``, falls
            back to ``trainer.estimated_stepping_batches - warmup_steps``.
        eta_min: Final LR floor for cosine.
        betas: AdamW betas.

    Everything else is forwarded to :class:`ObjectDetectionTask`.
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
        anchor_sizes: list[list[int]] | None = None,
        anchor_aspect_ratios: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        # Force base-class optimizer settings off — we build the optimizer
        # ourselves in configure_optimizers.
        kwargs.setdefault("optimizer", None)
        kwargs.setdefault("optimizer_hparams", None)
        kwargs.setdefault("scheduler", None)
        kwargs.setdefault("scheduler_hparams", None)
        # These must be set BEFORE super().__init__() — torchgeo BaseTask
        # calls self.configure_models() from inside its __init__, and our
        # override reads self.anchor_sizes / self.anchor_aspect_ratios.
        self.anchor_sizes = anchor_sizes
        self.anchor_aspect_ratios = anchor_aspect_ratios
        super().__init__(*args, **kwargs)
        self.backbone_lr = float(backbone_lr)
        self.head_lr = float(head_lr)
        self.layer_decay = float(layer_decay)
        self.weight_decay = float(weight_decay)
        self.head_weight_decay = (
            float(head_weight_decay)
            if head_weight_decay is not None
            else float(weight_decay)
        )
        self.warmup_steps = int(warmup_steps)
        self.cosine_t_max = cosine_t_max
        self.eta_min = float(eta_min)
        self.betas = tuple(betas)

    def configure_models(self) -> None:  # type: ignore[override]
        super().configure_models()
        # Override the factory's hardcoded anchor generator. The upstream
        # ObjectDetectionModelFactory uses (32, 64, 128, 256, 512) — those
        # are too big for lunar craters (min diameter 5 px). Replace the
        # RPN's AnchorGenerator so anchors match the target-object scale.
        if self.anchor_sizes is None:
            return
        aspect = tuple(self.anchor_aspect_ratios or (0.5, 1.0, 2.0))
        sizes = tuple(tuple(int(s) for s in level) for level in self.anchor_sizes)
        aspect_ratios = (aspect,) * len(sizes)
        rpn = self.model.torchvision_model.rpn
        rpn.anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)
        print(
            f"[LunarObjectDetectionTask] overrode RPN anchors: "
            f"sizes={sizes} aspect_ratios={aspect}"
        )

    # ------------------------------------------------------------------ utils
    def _num_encoder_blocks(self) -> int:
        # LunarBackbone stores encoder depth as len(self.out_channels).
        return len(self.model.torchvision_model.backbone.backbone.out_channels)

    def _param_layer_id(self, name: str, num_blocks: int) -> int:
        """Return an integer layer id in [0, num_blocks+1].

        0        = embeddings / register tokens
        1..N     = encoder blocks (1 = deepest / lowest LR, N = shallowest)
        N+1      = encoder_norm (top of backbone)
        """
        blk = _encoder_block_index(name)
        if blk is not None:
            # deeper block index i (larger i) is closer to the head; give it a
            # higher layer id so it gets a larger LR.
            return blk + 1
        if "encoder_norm" in name:
            return num_blocks + 1
        # embeddings, register tokens, anything else in the backbone
        return 0

    def _make_param_groups(self) -> list[dict[str, Any]]:
        num_blocks = self._num_encoder_blocks()
        top_layer = num_blocks + 1
        groups: dict[tuple[int, bool, bool], dict[str, Any]] = {}

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            in_backbone = _is_backbone_param(name)
            nd = _no_decay(name)
            if in_backbone:
                layer_id = self._param_layer_id(name, num_blocks)
                # top_layer gets LR = backbone_lr; each step down multiplies by layer_decay.
                scale = self.layer_decay ** (top_layer - layer_id)
                lr = self.backbone_lr * scale
            else:
                layer_id = -1  # sentinel for "head"
                lr = self.head_lr
            if nd:
                wd = 0.0
            elif in_backbone:
                wd = self.weight_decay
            else:
                wd = self.head_weight_decay
            key = (layer_id, in_backbone, nd)
            g = groups.setdefault(
                key,
                {
                    "params": [],
                    "lr": lr,
                    "weight_decay": wd,
                    "name": (
                        f"backbone_layer_{layer_id}_{'nd' if nd else 'wd'}"
                        if in_backbone
                        else f"head_{'nd' if nd else 'wd'}"
                    ),
                },
            )
            g["params"].append(p)

        # Stable ordering: backbone first (deepest → shallowest), then head.
        ordered = sorted(
            groups.values(),
            key=lambda g: (0 if g["name"].startswith("backbone") else 1, g["name"]),
        )
        return ordered

    # ---------------------------------------------------------- optim/sched
    def configure_optimizers(self):  # type: ignore[override]
        param_groups = self._make_param_groups()
        n_params = sum(sum(p.numel() for p in g["params"]) for g in param_groups)
        print(
            f"[LunarObjectDetectionTask] {len(param_groups)} param groups, "
            f"{n_params/1e6:.1f}M trainable params. "
            f"layer_decay={self.layer_decay} backbone_lr={self.backbone_lr} "
            f"head_lr={self.head_lr}"
        )
        for g in param_groups:
            print(
                f"  {g['name']:<32s} lr={g['lr']:.2e} wd={g['weight_decay']:.2e} "
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
