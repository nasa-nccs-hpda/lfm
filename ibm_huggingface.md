# NASA-IBM Lunar Foundation Model (NASA-IBM LFM)
A multimodal, multi-resolution foundation model for lunar remote sensing.

A ViT-B encoder–decoder trained from scratch on SomBench — ~2 million co-registered lunar tile bundles spanning 11 modalities at two spatial scales (LROC NAC ~1 m/px and LROC WAC ~100 m/px). It adapts the TerraMind masked-token recipe to the Moon with two extensions:

Acquisition geometry as explicit context — per-tile illumination angles, solar-frame anchors, and tile footprint are sequence-tokenized encoder inputs. Lunar surface appearance is governed more by illumination geometry than by intrinsic surface variation, so this hands the model the dominant confound instead of making it recover a quantity already recorded for every tile.
Joint mixed-resolution pretraining — NAC- and WAC-anchored tiles train in a single mixed-batch loop at native resolution, so one set of weights covers both resolution families across a 100× scale gap.
FlexiViT patch embedding lets the checkpoint be fine-tuned at other patch sizes without retraining the backbone; modality-wise tokenization lets you drop or add modalities at fine-tuning time.

📄 Full method, ablations, and per-baseline results: paper · 💻 Fine-tuning code: GitHub

## NASA-IBM LFM architecture overview

Modality-wise tokenization (left/center-left) feeds a TerraMind-style any-to-any masked-token objective (center-right); the pretrained encoder then supports multimodal generation, multimodal fine-tuning, and FlexiViT fine-tuning at new patch sizes (right).

## Model at a glance
Encoder / decoder	ViT-B (768 dim, 12 layers, 12 heads) / 12-layer Transformer, shared width
Input size / patch size	256 × 256 px / 16 × 16 at pretraining (FlexiViT-resizable; results below use ps8)
Modalities	11 — 9 dense image-like, plus per-tile optical metadata and static-map context
Objective	Cross-entropy over discrete token vocabularies at sampled target positions
Tokenizers	9 modality-specific VQ-VAE, FSQ quantization (levels 8,8,8,6,5), DDPM decoder
Pretraining	16 × H100, 150k steps, global batch 1,536, peak LR 1e-4 cosine, bf16, ~1.1k GPU-h
Data	SomBench — 963,609 WAC + 1,000,113 NAC bundles, split geographically by LTM zone / LPS cap
Downstream adaptation	TerraTorch
License	Apache-2.0
Repository contents
backbone/         pretrained checkpoint and config to be used for finetuning/generation
tokenizers/       checkpoints and configs for all tokenizers

## Intended use
A reusable representation for lunar remote sensing: fine-tune or LoRA-adapt the encoder for detection, segmentation, and dense regression on LROC WAC/NAC imagery and co-registered terrain products, with modality subsets and patch grids that differ from pretraining. Any-to-any multimodal generation is supported as a qualitative probe of learned cross-modal structure.

Out of scope. Not a scientific-grade generative product — generated fields are not calibrated predictions and are no substitute for instruments, stereo photogrammetry, or geodetic solutions. The model maintains no geodetic reference frame: it recovers local structure but not absolute values (DTM shape is right at a shifted elevation offset; generated lat/lon can be off by tens of degrees). Not validated for operational decisions such as landing-site certification or hazard clearance. Ice-prospectivity outputs regress a knowledge-driven fuzzy-overlay prospectivity map, not measured ice. Not evaluated beyond the Moon or on products absent from SomBench.

## Modalities
Dense modalities are patch-embedded and tokenized as images. A sample is either NAC- or WAC-centered — the two families are never mixed within a sample.

Family	Modality	Key	Native res.	Ch.	Source
WAC (51.2 km tiles)	Visible reflectance	vis	100 m	5	LROC WAC
Ultraviolet reflectance	uv	500 m	2	LROC WAC
Topography / slope / aspect	dtm, slope, aspect	60 m	1 / 1 / 2	SLDEM2015
NAC (512 m tiles)	Panchromatic imagery	nac	1 m	1	LROC NAC
Topography / slope / aspect	dtm_3m, slope_3m, aspect_3m	3 m	1 / 1 / 2	NAC-stereo DTM
Aspect is stored as a sine–cosine pair to avoid angular wrap-around.

Two sequence-like modalities carry scalar context, binned and stringified before text tokenization (e.g. with 0.25° longitude bins, a tile-center longitude of 87.49° becomes C_LON=87.25-->87.50):

- Optical metadata (8 fields) — solar incidence, emission, phase and azimuth angles; sub-solar and tile-center coordinates; ground sampling distance. Available at inference whenever the image is, so no label leakage.
- Static-map context (28 fields, tile-footprint averages) — Diviner thermophysics (TREG, TBOL, ROCK_ABUND, HPAR, DICE), LOLA products (ROUGHNESS, PSR, AVG_ILLUM, ALBEDO), Mini-RF radar (MINIRF_*), Kaguya mineralogy (MI_*, SW_FE, SP_MINER), LROC WAC (nr643, TIO2, WAC_*), GRAIL (GRAVITY), and Lunar Prospector (HYDROGEN). Coverage varies by instrument footprint, from near-global to polar-only (~6,900 polar tiles per track).

## Usage
Fine-tuning runs through TerraTorch, configured from YAML. The companion repo adds the SomBench benchmark datamodules, backbone wrappers, support for registering new input modalities into the pretrained backbone, and FlexiViT patch-embedding interpolation.

```bash
git clone https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model && cd $_
pip install -e .
terratorch fit --config configs/finetune/ice_prospectivity.yaml
```

Point the backbone checkpoint field at the downloaded checkpoint from this repo. Exact registry names and full hyperparameters are in the GitHub.

Checkpoint can be downloaded using Python:

```python
from huggingface_hub import snapshot_download

# Only download model weights and config
snapshot_download(repo_id="nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model", allow_patterns="backbone/*", local_dir="./")

# Download entire HuggingFace directory
snapshot_download(repo_id="nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model", local_dir="./")
```

## Which adaptation strategy? From our experiments:

LoRA is the sensible default (rank 16, α = 32, on encoder attention + MLP linears). It matches or beats full fine-tuning on crater detection, stays competitive on segmentation, trains a small fraction of the encoder parameters, and shows narrower seed spread.
Full fine-tuning — it keeps an edge on our two smallest benchmarks. Use layer-wise LR decay with a lower encoder LR than head LR, so pretrained features aren't pushed away early.
Frozen encoder only after validating on your task: best on 100-tile IMP segmentation, but below every baseline on crater detection.

## Evaluation
Four SomBench benchmarks via TerraTorch, with loaders, splits, augmentations, loss, and metric held fixed across backbones — only encoder weights and initialization differ. Mean ± std over 5 seeds (3 for the ablation). Baselines are ResNet-50, ViT-B MAE, ConvNeXt-B, ConvNeXt-V2-B, SwinV2-B, DaViT-B (ImageNet-1k/22k), plus DeepLabV3+ and SegFormer on segmentation. Full tables are in the paper.

Benchmark	Metric	Best NASA-IBM LFM	Best baseline	Random-init LFM
Robbins craters, WAC, 50% data	mAP ↑	0.2541 ± 0.0018 (full FT)	0.2313 ± 0.0027 (SwinV2-B)	0.2197 ± 0.0027
Robbins craters, WAC, 100% data	mAP ↑	0.2581 ± 0.0017 (LoRA)	0.2420 ± 0.0047 (SwinV2-B)	0.2289 ± 0.0037
NAC craters, meter scale	mAP ↑	0.1543 ± 0.0098 (LoRA)	0.1552 ± 0.0086 (SwinV2-B)	0.1274 ± 0.0151
Irregular Mare Patches	IoU₁ ↑	0.5709 ± 0.0114 (frozen)	0.5687 ± 0.0181 (ConvNeXtV2-B)	0.3142 ± 0.0746
Polar ice prospectivity	RMSE ↓	0.0293 ± 0.0013 (full FT)	0.0377 ± 0.0004 (SwinV2-B)	0.0397 ± 0.0004

- Label efficiency. On WAC craters the pretrained variants at 50% training data already match or exceed SwinV2-B trained on the full set. The ranking is stable across fractions and the gap exceeds seed spread.

- Widest margin: ice prospectivity, and it decomposes. The LFM gives each modality its own pretrained patch adapter and concatenates tokens along the sequence axis (native token-level late fusion), whereas baselines can only stack the 8 layers as channels at a shared stem. The random-init control shares that tokenization without lunar pretraining and already beats 5 of 6 ImageNet baselines (0.0397); pretraining accounts for the rest. In a modality-count ablation the LFM with only 3 modalities (aspect, slope, DICE) matches ConvNeXt-B using the full 8-layer stack (0.0434 vs 0.0437).

- NAC craters and IMP segmentation: treat leaders as comparable — margins are smaller than the seed spread. On NAC, scores are low across the board (part of the benchmark is annotated at 5 m/px and visibly blurrier) and the frozen variant performs like random init, so some encoder adaptation is needed at meter scale. On IMP the decisive signal is the random-init control collapsing to IoU₁ 0.31 — pretraining is essential in that regime.

- Qualitative generation. Chained generation confirms the intended cross-modal correspondences: slope and aspect from DTM at both scales; reflectance whose illumination pattern follows the supplied metadata (shadows align with the sun direction implied by sub-solar coordinates) even where it doesn't match pixel-for-pixel; and coarse spatial pattern across the full co-registered set. Absolute values drift (see Out of scope). These are qualitative sanity checks, not a systematic evaluation.

Limitations
Ablations not yet isolated. Separating the contribution of geometry tokenization and mixed-resolution training from lunar pretraining as a whole remains to be run; the ice-prospectivity random-init control is the one place architecture and pretraining are teased apart.
NAC pretraining is site-limited by construction — restricted to NAC PHO sites and frames with co-registered 3 m stereo DTMs (1,095 frames): globally distributed, not globally dense.
No geodetic reference frame; ice-prospectivity target is a model rather than a measurement (see Out of scope).