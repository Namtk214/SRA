# SiT-SRA: Scalable Interpolant Transformer with Self-Refining Alignment (JAX/Flax)

A JAX/Flax implementation of **SiT** (Scalable Interpolant Transformers) enhanced with **SRA** (Self-Refining Alignment) for class-conditional image generation on ImageNet 256×256.

Designed for **TPU training** with full data-parallel support and built-in evaluation (FID, sFID, IS, Precision/Recall).

## Overview

**SiT-SRA** combines two key ideas:

- **Flow Matching** (SiT): Trains a Diffusion Transformer (DiT) backbone using the interpolant/flow-matching framework with velocity prediction (`v = x₀ − x₁`).
- **Self-Refining Alignment** (SRA): A self-distillation objective that aligns intermediate features between a student block and an EMA teacher block at a slightly denoised timestep, improving generation quality without external supervision.

## Architecture

| Config | Hidden | Depth | Heads | Params |
|--------|--------|-------|-------|--------|
| SiT-S  | 384    | 12    | 6     | ~33M   |
| SiT-B  | 768    | 12    | 12    | ~130M  |
| SiT-L  | 1024   | 24    | 16    | ~460M  |
| SiT-XL | 1152   | 28    | 16    | ~675M  |

All variants use patch size **2**, latent channels **4**, adaLN-Zero conditioning, and sinusoidal positional embeddings.

## Project Structure

```
SiT-SRA/
├── train.py                 # Main training script (TPU data-parallel)
├── sample.py                # Standalone sampling / FID generation
├── prepare_data.py          # Encode ImageNet → VAE latents (GPU/CPU)
├── prepare_data_tpu.py      # Encode ImageNet → VAE latents (TPU)
├── merge_ar_files.py        # Merge sharded .ar files
├── requirements.txt         # Python dependencies
├── README.md
└── src/
    ├── model.py             # SelfFlowDiT model (Flax)
    ├── sampling.py          # Euler ODE/SDE denoise loops
    ├── fid_utils.py         # InceptionV3 feature extraction (JAX)
    ├── metrics.py           # FID, sFID, Precision/Recall computation
    ├── inception_is_subprocess.py  # Inception Score worker (torchvision)
    └── utils.py             # Positional encoding utilities
```

## Requirements

### TPU (recommended)

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install flax optax orbax-checkpoint
pip install array-record grain
pip install wandb Pillow scipy einops torchvision diffusers transformers
```

### GPU

```bash
pip install jax[cuda12]
pip install flax optax orbax-checkpoint
pip install array-record grain
pip install wandb Pillow scipy einops torchvision diffusers transformers
```

## Data Preparation

### Step 1: Encode ImageNet to VAE latents

The training pipeline reads pre-encoded SD-VAE latents stored in **ArrayRecord** (`.ar`) format. Each record is a pickled dict:

```python
{
    "latent": np.ndarray,   # shape (4, 32, 32), dtype float32
                             # Pre-sampled & scaled (×0.18215) VAE latent
    "label": int             # ImageNet class index (0-999)
}
```

**On TPU:**
```bash
python prepare_data_tpu.py \
    --imagenet-path /path/to/imagenet \
    --output-dir /path/to/output \
    --vae-model stabilityai/sd-vae-ft-ema \
    --split train \
    --num-shards 128
```

**On GPU/CPU:**
```bash
python prepare_data.py \
    --imagenet-path /path/to/imagenet \
    --output-dir /path/to/output \
    --split train
```

### Step 2: Real images for FID evaluation

FID is computed against **original ImageNet validation images** (not VAE reconstructions). The default behavior auto-discovers `imagenet-object-localization-challenge.zip` in the `--vae-model` directory.

## Training

### Basic training (TPU v5p-8)

```bash
python train.py \
    --model-size XL \
    --data-path "/path/to/latents/train*.ar" \
    --val-data-path "/path/to/latents/val*.ar" \
    --batch-size 256 \
    --epochs 400 \
    --steps-per-epoch 5000 \
    --learning-rate 1e-4 \
    --ema-decay 0.9999 \
    --grad-clip 1.0 \
    --cfg-prob 0.1 \
    --block-out-s 8 \
    --block-out-t 20 \
    --t-max 0.2 \
    --align-weight 0.2 \
    --loss-type sml1 \
    --sample-cfg-scale 1.0 \
    --fid-cfg-scale 1.0 \
    --ckpt-dir /path/to/checkpoints \
    --vae-model /path/to/sd-vae \
    --wandb-project selfflow-jax \
    --fid-freq 50000 \
    --num-fid-samples 50000 \
    --fid-num-steps 250
```

### Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-size` | `XL` | Backbone: `S`, `B`, `L`, `XL` |
| `--batch-size` | `256` | Global batch size (split across devices) |
| `--epochs` | `100` | Number of training epochs |
| `--steps-per-epoch` | `1000` | Steps per epoch |
| `--learning-rate` | `1e-4` | AdamW learning rate (weight_decay=0) |
| `--ema-decay` | `0.9999` | EMA decay for teacher network |
| `--grad-clip` | `1.0` | Max gradient norm |
| `--cfg-prob` | `0.0` | Label dropout prob for CFG training (0.1 recommended) |
| `--data-path` | required | Glob pattern to training `.ar` files |
| `--val-data-path` | `None` | Glob pattern to validation `.ar` files |

### SRA Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--block-out-s` | `4` | Student feature extraction block (1-indexed) |
| `--block-out-t` | `8` | Teacher feature extraction block (1-indexed) |
| `--t-max` | `0.2` | Max timestep gap between student and teacher |
| `--align-weight` | `0.2` | Base weight for SRA alignment loss |
| `--loss-type` | `sml1` | Alignment loss: `sml1`, `l2`, or `l1` |
| `--align-decay-start-epoch` | `149` | Epoch to start decaying alignment weight |
| `--align-decay-denom` | `1000.0` | Decay denominator |
| `--align-decay-base` | `0.1` | Decay base |

### Evaluation Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--fid-freq` | `10000` | Compute FID every N steps (0 = disable) |
| `--num-fid-samples` | `4000` | Number of samples for FID (paper: 50000) |
| `--fid-num-steps` | `50` | Denoising steps for FID (paper: 250) |
| `--fid-cfg-scale` | `1.0` | CFG scale for FID generation |
| `--fid-batch-size` | `32` | Generation batch size |
| `--sample-freq` | `1000` | Preview sample frequency |
| `--sample-num-steps` | `50` | Steps for preview samples |
| `--sample-cfg-scale` | `1.0` | CFG scale for previews |
| `--inception-score` | `True` | Enable Inception Score |
| `--precision-recall` | `True` | Enable Precision/Recall |

### VAE & Checkpoint Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--vae-model` | (see code) | Path or HF repo ID for SD-VAE (Flax/PyTorch/ZIP) |
| `--vae-hf-config` | (see code) | Path to VAE config.json |
| `--ckpt-dir` | `./checkpoints` | Checkpoint directory |
| `--log-freq` | `20` | Log metrics every N steps |
| `--eval-freq` | `500` | Validation eval frequency |

## Sampling

### Generate samples from a trained checkpoint

```bash
python sample.py \
    --ckpt /path/to/checkpoint \
    --output-dir ./samples \
    --num-fid-samples 50000 \
    --num-steps 250 \
    --cfg-scale 1.0 \
    --batch-size 64 \
    --mode SDE
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--ckpt` | required | Path to model checkpoint |
| `--output-dir` | `./samples` | Output directory |
| `--num-fid-samples` | `50000` | Number of samples |
| `--num-steps` | `250` | Denoising steps |
| `--mode` | `SDE` | `SDE` or `ODE` |
| `--cfg-scale` | `1.0` | Classifier-free guidance scale |

## Training Details

### Loss

```
L_total = L_gen + α(epoch) × L_align
```

- **L_gen**: MSE velocity prediction loss `||v_θ(x_τ, τ) − (x₀ − x₁)||²`
- **L_align**: Feature alignment (smooth L1) between student block `s` and EMA teacher block `t`
- **α(epoch)**: Decaying schedule — constant until epoch 149, then `0.2 × 0.1^((epoch-149)/1000)`

### Flow Matching Convention

- `τ = 0` → pure noise (`x₁`)
- `τ = 1` → clean data (`x₀`)
- Interpolant: `x_τ = (1 − τ) × x₁ + τ × x₀`
- Target: `v = x₀ − x₁`

### SRA Self-Distillation

```
teacher_τ = clip(τ − Δτ, 0, 1),  Δτ ~ U(0, t_max)
```

The student (online) network at block `s` is trained to match the EMA teacher at block `t`, where the teacher sees a slightly cleaner version of the input.

### VAE Decoding

The training code supports two VAE decode backends:
1. **Flax on TPU** (preferred): Loads `FlaxAutoencoderKL` directly, runs decode on TPU via `pmap`
2. **PyTorch subprocess** (fallback): Spawns an isolated subprocess with `torch` + `diffusers` to avoid JAX/protobuf conflicts

## Monitoring

Training logs to **Weights & Biases**:

| Metric | Description |
|--------|-------------|
| `train/gen_loss` | Generation loss |
| `train/align_loss` | SRA alignment loss |
| `train/loss` | Total loss |
| `train/grad_norm` | Gradient norm |
| `train/param_norm` | Parameter norm |
| `val/gen_loss` | Validation generation loss |
| `FID`, `sFID` | Fréchet Inception Distance |
| `IS` | Inception Score |
| `Precision`, `Recall` | kNN manifold metrics |

## Acknowledgments

This implementation builds upon:
- [SiT](https://github.com/willisma/SiT) — Scalable Interpolant Transformers
- [REPA](https://github.com/sihyun-yu/REPA) — Representation Alignment for Generation
- [DiT](https://github.com/facebookresearch/DiT) — Diffusion Transformers
