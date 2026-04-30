# DiverseDiT Baseline (JAX / TPU)

**DiverseDiT** baseline for ImageNet 256×256, implemented in JAX/Flax and optimised for Kaggle TPU (v5p-8) training.

This branch extends the vanilla SiT backbone with **block-diversity regularisation** (orthogonality, mutual information proxy, and dispersion losses) and optional **long skip-layer connections** between mirrored encoder/decoder blocks.

## Features

| Feature | Description |
|---------|-------------|
| **Flow Matching** | Velocity prediction with uniform timestep sampling (`τ ∈ [0,1]`) |
| **Block Diversity Loss** | Orthogonality + MI proxy + dispersion regularisation across transformer blocks |
| **Long Skip Connections** | Optional DiverseDiT-style mirrored early/late block residual fusion |
| **Classifier-Free Guidance** | `--cfg-prob` label dropout during training; `--sample-cfg-scale` at inference |
| **Periodic Checkpointing** | `--ckpt-freq` / `--ckpt-keep` with automatic cleanup |
| **Resume Training** | `--resume` restores from latest checkpoint (msgpack or Orbax) |
| **Kaggle TPU Compat** | Shardy dialect disabled (`JAX_USE_SHARDY=0`), Orbax fallback to msgpack |
| **EMA** | Exponential moving average of online params (`--ema-decay 0.9999`) |
| **Integrated Metrics** | FID, sFID, IS, Precision/Recall computed during training |

## Quick Start

### Training on Kaggle TPU

```bash
python train.py \
    --data-path /path/to/imagenet_latents/*.ar \
    --val-data-path /path/to/val_latents/*.ar \
    --model-size XL \
    --batch-size 256 \
    --epochs 100 \
    --steps-per-epoch 1000 \
    --learning-rate 1e-4 \
    --cfg-prob 0.1 \
    --block-diversity-loss \
    --skip-layer-connection \
    --diversity-pair-mode paper \
    --ckpt-dir ./checkpoints/diversedit \
    --ckpt-freq 5000 \
    --ckpt-keep 2 \
    --wandb-project diversedit-jax
```

### Resume from Checkpoint

```bash
python train.py \
    --data-path /path/to/imagenet_latents/*.ar \
    --model-size XL \
    --batch-size 256 \
    --cfg-prob 0.1 \
    --block-diversity-loss \
    --skip-layer-connection \
    --ckpt-dir ./checkpoints/diversedit \
    --resume
```

## Training Arguments

### Core

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-path` | *required* | Path/glob to training ArrayRecord files |
| `--val-data-path` | `None` | Path/glob to validation ArrayRecord files |
| `--model-size` | `XL` | DiT backbone: `S`, `B`, `L`, `XL` |
| `--batch-size` | `256` | Global batch size (divided across devices) |
| `--epochs` | `100` | Number of training epochs |
| `--steps-per-epoch` | `1000` | Steps per epoch |
| `--learning-rate` | `1e-4` | AdamW learning rate |
| `--grad-clip` | `1.0` | Gradient clipping max norm |
| `--ema-decay` | `0.9999` | EMA decay rate |

### Classifier-Free Guidance

| Argument | Default | Description |
|----------|---------|-------------|
| `--cfg-prob` | `0.0` | Label dropout probability for CFG training (e.g. `0.1`) |
| `--sample-cfg-scale` | `1.0` | CFG scale at sampling time |

### Block Diversity

| Argument | Default | Description |
|----------|---------|-------------|
| `--block-diversity-loss` | `false` | Enable diversity regularisation |
| `--skip-layer-connection` | `false` | Enable DiverseDiT long residual connections |
| `--diversity-pair-mode` | `paper` | Pair selection: `mirror`, `all`, `paper`, `first_pairs`, `projection-biased` |
| `--diversity-max-pairs` | `10` | Max number of block pairs for orth/MI losses |
| `--diversity-orth-weight` | `0.33` | Weight for orthogonality loss |
| `--diversity-mi-weight` | `0.33` | Weight for MI proxy loss |
| `--diversity-disp-weight` | `0.33` | Weight for dispersion loss |
| `--diversity-gate-low` | `0.1` | Adaptive gate lower threshold |
| `--diversity-gate-high` | `0.5` | Adaptive gate upper threshold |
| `--diversity-paper-num-layers` | `10` | Number of layers sampled in `paper` mode |

### Checkpointing

| Argument | Default | Description |
|----------|---------|-------------|
| `--ckpt-dir` | `./checkpoints` | Checkpoint save directory |
| `--ckpt-freq` | `5000` | Save every N steps (`0` = end only) |
| `--ckpt-keep` | `1` | Number of recent checkpoints to retain |
| `--resume` | `false` | Resume from latest checkpoint |

### Evaluation

| Argument | Default | Description |
|----------|---------|-------------|
| `--eval-freq` | `500` | Validation loss eval frequency |
| `--sample-freq` | `1000` | Sample generation frequency |
| `--fid-freq` | `10000` | FID computation frequency |
| `--num-fid-samples` | `4000` | Number of samples for FID |
| `--fid-num-steps` | `50` | ODE steps for FID generation |
| `--block-corr-freq` | `0` | Block correlation analysis frequency |

## Project Structure

```
diversedit-sit/
├── train.py                # Training script (JAX/Flax, TPU-optimised)
├── sample.py               # Standalone sampling script
├── prepare_data.py         # Dataset preparation utilities
├── prepare_data_tpu.py     # TPU-specific data preparation
├── smoke_test_diversedit.py # Smoke test for diversity loss
├── smoke_test_metrics.py   # Smoke test for metrics pipeline
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── src/
    ├── model.py            # SelfFlowDiT with skip connections
    ├── sampling.py         # Flow matching ODE/SDE samplers
    ├── metrics.py          # FID, IS, Precision/Recall
    ├── fid_utils.py        # FID statistics computation
    └── utils.py            # Position encoding utilities
```

## Model Architecture

- **Backbone**: SiT-XL/2 (DiT with flow matching)
- **Patch size**: 2 × 2
- **Hidden dim**: 1152 (XL), 28 transformer blocks, 16 heads
- **Conditioning**: Global timestep + class label (adaLN-Zero)
- **DiverseDiT extensions**:
  - Long skip connections: `LongSkipFusion` fuses mirrored early/late blocks via LayerNorm + Linear projection
  - Block diversity: orthogonality, MI proxy, and dispersion losses with adaptive gating
- **Output**: Velocity prediction (learn_sigma enabled)

## Diversity Loss Details

The diversity regularisation encourages distinct feature representations across transformer blocks:

1. **Orthogonality loss** — minimises cosine similarity between mean activations of block pairs
2. **MI proxy loss** — minimises absolute token-wise cosine similarity between block pairs
3. **Dispersion loss** — maximises uniform feature usage across all captured layers (normalised variance)

An **adaptive gate** ramps the diversity loss contribution between `gate_low` and `gate_high` thresholds to prevent overwhelming the generative loss early in training.

## Environment Notes

This branch includes automatic Kaggle TPU compatibility fixes:

- `JAX_USE_SHARDY=0` / `ENABLE_SHARDY=0` — disables SDY dialect
- `JAX_PLATFORMS=tpu,cpu` — explicit platform ordering
- Orbax checkpoint fallback to `flax.serialization` (msgpack) when `flax.training.checkpoints` is unavailable

## Acknowledgments

- [DiverseDiT](https://github.com/Hila/DiverseDiT) — Diversity-Enhanced Diffusion Transformers
- [SiT](https://github.com/willisma/SiT) — Scalable Interpolant Transformers
- [REPA](https://github.com/sihyun-yu/REPA) — Representation Alignment for Generation
- [Self-Flow](https://github.com/thanhlamauto/Self-Flow) — Self-Supervised Flow Matching
