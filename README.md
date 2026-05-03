# Self-Flow + REPA / iREPA (JAX / TPU)

**Self-Flow** (Self-Supervised Flow Matching) with **REPA** (Representation Alignment) and **iREPA** (improved REPA) for ImageNet 256×256, implemented in JAX/Flax and optimised for Kaggle TPU (v5p-8) training.

This branch (`feat/irepa-repa-paper`) supports three training modes:

| Mode | Flag | Description |
|------|------|-------------|
| **Vanilla SiT** | _(default)_ | Flow matching only |
| **REPA** | `--encoder-depth N` | + DINOv2 feature alignment (MLP projector) |
| **iREPA** | `--encoder-depth N --irepa` | + Conv projector + spatial normalization |

---

## Quick Start

### 1. Data Preparation

Precompute VAE latents from ImageNet into ArrayRecord format:

```bash
python prepare_data_tpu.py \
    --split train val \
    --data-dir /path/to/ILSVRC/Data/CLS-LOC \
    --output-dir ./imagenet_latents \
    --batch-size 128 \
    --num-shards 1024 \
    --group-size 1
```

> **Note (REPA):** Each record stores `{"latent", "label", "image_path"}`. For REPA training, the original images must be accessible at the saved `image_path` on the training machine.

### 2. Convert DINOv2 Weights (required for REPA/iREPA)

```bash
python convert_dinov2_weights.py --output dinov2_vitb14_flax.pkl
```

### 3. Training — Vanilla SiT

```bash
python train.py \
    --data-path '/path/to/imagenet_latents/train-*.ar' \
    --val-data-path '/path/to/imagenet_latents/val-*.ar' \
    --model-size XL \
    --batch-size 256 \
    --epochs 400 --steps-per-epoch 5000 \
    --learning-rate 1e-4 \
    --ema-decay 0.9999 \
    --grad-clip 1.0 \
    --cfg-prob 0.1 \
    --ckpt-dir ./checkpoints \
    --ckpt-freq 20000 \
    --ckpt-keep 1 \
    --wandb-project selfflow-jax \
    --vae-model /path/to/sdvae-ema \
    --vae-hf-config /path/to/sdvae-ema/config.json \
    --log-freq 20 \
    --eval-freq 500 \
    --eval-batches 4 \
    --sample-freq 5000 \
    --sample-num-steps 250 \
    --sample-cfg-scale 1.0 \
    --fid-freq 50000 \
    --num-fid-samples 50000 \
    --fid-batch-size 32 \
    --fid-eval-local-batch 4 \
    --fid-num-steps 250 \
    --fid-cfg-scale 1.0 \
    --vae-decode-batch-size 8 \
    --inception-score \
    --inception-score-splits 10 \
    --precision-recall \
    --pr-k 3 \
    --pr-max-samples 5000 \
    --preflight-checks
```

### 4. Training — REPA

```bash
python train.py \
    --data-path '/path/to/imagenet_latents/train-*.ar' \
    --val-data-path '/path/to/imagenet_latents/val-*.ar' \
    --model-size XL \
    --batch-size 256 \
    --epochs 400 --steps-per-epoch 5000 \
    --learning-rate 1e-4 \
    --ema-decay 0.9999 \
    --grad-clip 1.0 \
    --cfg-prob 0.1 \
    --encoder-depth 8 \
    --repa-proj-coeff 0.5 \
    --repa-proj-dim 2048 \
    --repa-z-dim 768 \
    --repa-align-tau-min 0.0 \
    --repa-align-tau-max 1.0 \
    --dinov2-weights ./dinov2_vitb14_flax.pkl \
    --ckpt-dir ./checkpoints \
    --ckpt-freq 20000 \
    --ckpt-keep 1 \
    --wandb-project selfflow-jax \
    --vae-model /path/to/sdvae-ema \
    --vae-hf-config /path/to/sdvae-ema/config.json \
    --log-freq 20 \
    --eval-freq 500 \
    --eval-batches 4 \
    --sample-freq 5000 \
    --sample-num-steps 250 \
    --sample-cfg-scale 1.0 \
    --fid-freq 50000 \
    --num-fid-samples 50000 \
    --fid-batch-size 32 \
    --fid-eval-local-batch 4 \
    --fid-num-steps 250 \
    --fid-cfg-scale 1.0 \
    --vae-decode-batch-size 8 \
    --inception-score \
    --inception-score-splits 10 \
    --precision-recall \
    --pr-k 3 \
    --pr-max-samples 5000 \
    --preflight-checks
```

### 5. Training — iREPA

Same as REPA, add `--irepa` or configure conv projector + spatial norm individually:

```bash
python train.py \
    --data-path '/path/to/imagenet_latents/train-*.ar' \
    --val-data-path '/path/to/imagenet_latents/val-*.ar' \
    --model-size XL \
    --batch-size 256 \
    --epochs 400 --steps-per-epoch 5000 \
    --learning-rate 1e-4 \
    --ema-decay 0.9999 \
    --grad-clip 1.0 \
    --cfg-prob 0.1 \
    --encoder-depth 8 \
    --irepa \
    --irepa-spatial-gamma 1.0 \
    --repa-proj-coeff 0.5 \
    --repa-proj-dim 2048 \
    --repa-z-dim 768 \
    --repa-align-tau-min 0.0 \
    --repa-align-tau-max 1.0 \
    --dinov2-weights ./dinov2_vitb14_flax.pkl \
    --ckpt-dir ./checkpoints \
    --ckpt-freq 20000 \
    --ckpt-keep 1 \
    --wandb-project selfflow-jax \
    --vae-model /path/to/sdvae-ema \
    --vae-hf-config /path/to/sdvae-ema/config.json \
    --log-freq 20 \
    --eval-freq 500 \
    --eval-batches 4 \
    --sample-freq 5000 \
    --sample-num-steps 250 \
    --sample-cfg-scale 1.0 \
    --fid-freq 50000 \
    --num-fid-samples 50000 \
    --fid-batch-size 32 \
    --fid-eval-local-batch 4 \
    --fid-num-steps 250 \
    --fid-cfg-scale 1.0 \
    --vae-decode-batch-size 8 \
    --inception-score \
    --inception-score-splits 10 \
    --precision-recall \
    --pr-k 3 \
    --pr-max-samples 5000 \
    --preflight-checks
```

### 6. Resume Training

```bash
python train.py \
    --resume \
    --ckpt-dir ./checkpoints \
    ... (same flags as original run)
```

Resume restores:
- **Training state**: params + optimizer state + step counter
- **EMA params**: from `<ckpt-dir>/ema/`
- **Training loop**: skips to correct epoch/step

Supports both `flax.training.checkpoints` (orbax) and msgpack fallback formats.

### 7. Inference — Generate 50k Samples

```bash
python sample.py \
    --ckpt checkpoints/selfflow_imagenet256.pt \
    --output-dir ./samples \
    --num-fid-samples 50000 \
    --num-steps 250 \
    --mode SDE \
    --cfg-scale 1.0
```

---

## Model Architecture

Based on **SiT-XL/2** (Scalable Interpolant Transformers):

| Variant | Hidden | Depth | Heads | Params |
|---------|--------|-------|-------|--------|
| S       | 384    | 12    | 6     | ~33M   |
| B       | 768    | 12    | 12    | ~130M  |
| L       | 1024   | 24    | 16    | ~460M  |
| **XL**  | 1152   | 28    | 16    | ~675M  |

Key components:
- **adaLN-Zero** conditioning (adaptive layer norm with zero init)
- **Per-token timestep** support (inference) / **global timestep** (training)
- **REPA projector**: 3-layer MLP (2048 hidden → 768 out) or Conv2D 3×3
- **DINOv2 ViT-B/14**: frozen feature extractor (bfloat16 on TPU)

---

## Training Details

### Loss Functions

**Vanilla SiT (Flow Matching):**
```
τ ~ U(0,1)
x_τ = (1-τ)·ε + τ·x₀
loss = ||v_θ(x_τ, τ) - (x₀ - ε)||²
```

**REPA alignment (added when `--encoder-depth > 0`):**
```
align_loss = -cos_sim(projector(hidden[encoder_depth]), DINOv2(image))
total_loss = diff_loss + proj_coeff × align_loss
```

### CFG (Classifier-Free Guidance)

When `--cfg-prob > 0`, the label embedding table grows by 1 entry (null class). All model instances (training, sampling, FID) use the same `cfg_prob` to ensure matching embedding sizes.

### Optimizer
- **AdamW** (weight decay = 0) + gradient clipping (max_norm = 1)
- **EMA decay**: 0.9999 (used for eval/sampling)

---

## Complete Command Line Reference

### Core Training

| Flag | Default | Description |
|------|---------|-------------|
| `--data-path` | _(required)_ | Path/glob to training ArrayRecord files |
| `--val-data-path` | `None` | Path/glob to validation ArrayRecord files |
| `--model-size` | `XL` | Backbone size: `S`, `B`, `L`, `XL` |
| `--batch-size` | `256` | Global batch size (÷ device count) |
| `--epochs` | `100` | Number of epochs |
| `--steps-per-epoch` | `1000` | Steps per epoch |
| `--learning-rate` | `1e-4` | Learning rate |
| `--ema-decay` | `0.9999` | EMA decay for smoothed params (eval only) |
| `--grad-clip` | `1.0` | Gradient clipping max norm |
| `--cfg-prob` | `0.0` | Label dropout for CFG training (use `0.1`) |

### Checkpointing & Resume

| Flag | Default | Description |
|------|---------|-------------|
| `--ckpt-dir` | `./checkpoints` | Checkpoint directory |
| `--ckpt-freq` | `5000` | Save checkpoint every N steps (0 = only at end) |
| `--ckpt-keep` | `1` | Number of recent checkpoints to keep |
| `--resume` | `False` | Resume training from latest checkpoint in `--ckpt-dir` |

### REPA / iREPA

| Flag | Default | Description |
|------|---------|-------------|
| `--encoder-depth` | `0` | Block index for REPA alignment (0 = disabled) |
| `--repa-proj-coeff` | `0.5` | REPA alignment loss coefficient |
| `--dinov2-weights` | `None` | Path to `dinov2_vitb14_flax.pkl` (required when encoder-depth > 0) |
| `--irepa` | `False` | Enable both iREPA changes (conv proj + spatial norm) |
| `--irepa-conv-proj` | `= --irepa` | Use Conv2D 3×3 projector instead of MLP |
| `--irepa-spatial-norm` | `= --irepa` | Spatial normalize DINOv2 tokens before alignment |
| `--irepa-spatial-gamma` | `1.0` | Gamma coefficient for spatial normalization |
| `--repa-align-tau-min` | `0.0` | Lower tau bound for alignment (inclusive) |
| `--repa-align-tau-max` | `1.0` | Upper tau bound for alignment |
| `--repa-proj-dim` | `2048` | Hidden dim of MLP projector |
| `--repa-z-dim` | `768` | Output dim of projector (must match DINOv2) |

### VAE Model

| Flag | Default | Description |
|------|---------|-------------|
| `--vae-model` | _(Kaggle path)_ | Local path or HF repo ID for VAE decode. Accepts: (1) dir with `flax_model.msgpack`, (2) dir with `*.zip`, (3) HF repo ID |
| `--vae-hf-config` | _(Kaggle path)_ | Path to VAE `config.json`. Used when `--vae-model` dir lacks `config.json` |
| `--vae-decode-batch-size` | `8` | Micro-batch for VAE decode (lower if OOM) |

### Logging & Eval

| Flag | Default | Description |
|------|---------|-------------|
| `--wandb-project` | `sit-vanilla-jax` | WandB project name |
| `--no-wandb` | `False` | Disable WandB logging |
| `--log-freq` | `20` | Steps between WandB metric logs |
| `--eval-freq` | `500` | Steps between validation loss eval |
| `--eval-batches` | `4` | Validation batches per eval |

### Sample Preview

| Flag | Default | Description |
|------|---------|-------------|
| `--sample-freq` | `1000` | Steps between sample previews |
| `--sample-num-steps` | `50` | Denoising steps for previews (paper: 250) |
| `--sample-cfg-scale` | `1.0` | CFG scale for previews |

### FID Evaluation

| Flag | Default | Description |
|------|---------|-------------|
| `--fid-freq` | `10000` | Steps between FID evaluation (0 = off) |
| `--num-fid-samples` | `4000` | Samples for FID (paper: 50000) |
| `--fid-batch-size` | `32` | Batch size for FID generation |
| `--fid-eval-local-batch` | `4` | Per-device eval micro-batch for FID metrics |
| `--fid-num-steps` | `50` | Denoising steps for FID (paper: 250) |
| `--fid-cfg-scale` | `1.0` | CFG scale for FID |

### Inception Score

| Flag | Default | Description |
|------|---------|-------------|
| `--inception-score` / `--no-inception-score` | `True` | Enable/disable Inception Score |
| `--inception-score-weights` | `None` | Local path to Inception-v3 weights (skip download) |
| `--inception-score-splits` | `10` | Splits for IS computation |

### Precision / Recall

| Flag | Default | Description |
|------|---------|-------------|
| `--precision-recall` / `--no-precision-recall` | `True` | Enable/disable kNN Precision/Recall |
| `--pr-k` | `3` | k for kNN manifold metrics |
| `--pr-max-samples` | `5000` | Max samples for P/R (subset mode) |
| `--pr-full-mode` | `False` | Use all FID samples for P/R (O(N²) heavy) |

### Linear Probe

| Flag | Default | Description |
|------|---------|-------------|
| `--linear-probe` / `--no-linear-probe` | `False` | Enable linear probe accuracy inference |
| `--probe-save-path` | `None` | Path to `.npz` with probe weights (required) |
| `--probe-layer` | `None` | Layer to probe (default: final block) |
| `--probe-eval-batches` | `4` | Val batches for probe inference |

### Block Correlation Diagnostic

| Flag | Default | Description |
|------|---------|-------------|
| `--block-corr-freq` | `0` | Steps between block correlation heatmap (0 = off) |
| `--block-corr-batches` | `2` | Val batches for block correlation |

### Preflight / Safety

| Flag | Default | Description |
|------|---------|-------------|
| `--preflight-checks` | `False` | Run decode/metric smoke test before training |
| `--preflight-only` | `False` | Run preflight then exit |
| `--preflight-sample-count` | `4` | Samples for preflight decode test |
| `--preflight-fid-samples` | `16` | Fake samples for preflight FID test |
| `--preflight-fid-memory-probe` | `False` | Run one train step + FID batch at startup to catch OOMs |
| `--mock-data` | `False` | Allow random mock batches (NOT for real training) |

---

## Project Structure

```
Self-Flow/
├── train.py                     # Training script (vanilla SiT / REPA / iREPA)
├── sample.py                    # Inference: generate 50k samples for FID
├── prepare_data_tpu.py          # Precompute VAE latents → ArrayRecord
├── convert_dinov2_weights.py    # Convert DINOv2 PyTorch → Flax pickle
├── merge_ar_files.py            # Merge ArrayRecord shards
├── debug_train_startup.py       # Debug training startup
├── smoke_test_metrics.py        # Test metrics pipeline
├── requirements.txt             # Dependencies
└── src/
    ├── model.py                 # SelfFlowDiT (DiT + REPA + iREPA)
    ├── sampling.py              # SDE/ODE sampling loop
    ├── dinov2_flax.py           # DINOv2 ViT-B/14 (Flax)
    ├── metrics.py               # FID, sFID, IS, Precision/Recall (streaming)
    ├── fid_utils.py             # FID computation utilities
    ├── inception_is_subprocess.py # Inception Score subprocess worker
    └── utils.py                 # Positional encoding, token processing
```

## Acknowledgments

- [REPA](https://github.com/sihyun-yu/REPA) — Representation Alignment for Generation
- [SiT](https://github.com/willisma/SiT) — Scalable Interpolant Transformers
- [Self-Flow](https://bfl.ai/research/self-flow) — Self-Supervised Flow Matching
