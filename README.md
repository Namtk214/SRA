# Dispersive Loss SiT Baseline (JAX / TPU)

**Dispersive Loss (DispLoss)** baseline for ImageNet 256×256, implemented in JAX/Flax and optimised for Kaggle TPU (v5p-8) training.

This branch extends the vanilla SiT backbone with a **Dispersive Loss regularisation** that encourages diversity in hidden-layer representations by minimising the log-mean-exp of pairwise negative squared distances.

## Features

| Feature | Description |
|---------|-------------|
| **Flow Matching** | Velocity prediction with uniform timestep sampling (`τ ∈ [0,1]`) |
| **Dispersive Loss** | Batchwise diversity regulariser on a chosen hidden layer (`--disp --disp-layer`) |
| **CFG Training** | Class label dropout (`--cfg-prob 0.1`) for classifier-free guidance |
| **Periodic Checkpointing** | `--ckpt-freq` / `--ckpt-keep` with automatic cleanup |
| **Resume Training** | `--resume` restores full state (params + optimizer + step) |
| **Kaggle TPU Compat** | Shardy dialect disabled, Orbax fallback to msgpack |
| **EMA** | Exponential moving average of online params (`--ema-decay 0.9999`) |
| **Integrated Metrics** | FID, sFID, IS, Precision/Recall computed during training |

## Quick Start

### Training on Kaggle TPU

```bash
python train.py \
    --data-path /path/to/imagenet_latents/*.ar \
    --val-data-path /path/to/val_latents/*.ar \
    --model-size B \
    --batch-size 256 \
    --epochs 100 \
    --steps-per-epoch 1000 \
    --learning-rate 1e-4 \
    --cfg-prob 0.1 \
    --disp \
    --disp-layer 6 \
    --disp-lambda 0.25 \
    --eval-freq 500 \
    --sample-freq 1000 \
    --sample-cfg-scale 1.0 \
    --fid-freq 10000 \
    --vae-model /kaggle/input/models/damtrunghieu/sdvae-ema/flax/default/1 \
    --inception-score-weights /kaggle/input/models/ctlcmleon/inception-v3/pytorch/default/1/inception_v3_google-0cc3c7bd.pth \
    --ckpt-dir ./checkpoints/disp-loss \
    --ckpt-freq 5000 \
    --ckpt-keep 2 \
    --wandb-project disp-loss-jax
```

### Resume from Checkpoint

```bash
python train.py \
    --data-path /path/to/imagenet_latents/*.ar \
    --model-size B \
    --batch-size 256 \
    --cfg-prob 0.1 \
    --disp \
    --disp-layer 6 \
    --disp-lambda 0.25 \
    --vae-model /kaggle/input/models/damtrunghieu/sdvae-ema/flax/default/1 \
    --inception-score-weights /kaggle/input/models/ctlcmleon/inception-v3/pytorch/default/1/inception_v3_google-0cc3c7bd.pth \
    --ckpt-dir ./checkpoints/disp-loss \
    --resume
```

## Training Arguments

### Core

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-path` | *required* | Path/glob to training ArrayRecord files |
| `--val-data-path` | `None` | Path/glob to validation ArrayRecord files |
| `--model-size` | `B` | DiT backbone: `S`, `B`, `L`, `XL` |
| `--batch-size` | `256` | Global batch size (divided across devices) |
| `--epochs` | `100` | Number of training epochs |
| `--steps-per-epoch` | `1000` | Steps per epoch |
| `--learning-rate` | `1e-4` | AdamW learning rate |
| `--grad-clip` | `1.0` | Gradient clipping max norm |
| `--ema-decay` | `0.9999` | EMA decay rate |
| `--cfg-prob` | `0.1` | Class label dropout prob for CFG training (`0` = no CFG) |

### Model Weights (Kaggle)

| Argument | Default | Description |
|----------|---------|-------------|
| `--vae-model` | `stabilityai/sd-vae-ft-ema` | Path to VAE model (local Flax dir or HF repo ID) |
| `--vae-hf-config` | `stabilityai/sd-vae-ft-ema` | HF config ID fallback when VAE dir has no `config.json` |
| `--inception-score-weights` | `None` | Path to local Inception-v3 `.pth` weights for IS/FID |

### Dispersive Loss

| Argument | Default | Description |
|----------|---------|-------------|
| `--disp` | `false` | Enable Dispersive Loss regularisation |
| `--disp-layer` | `6` | Hidden layer index for feature extraction (must be ≤ depth) |
| `--disp-lambda` | `0.25` | Dispersive Loss weight |

### Checkpointing

| Argument | Default | Description |
|----------|---------|-------------|
| `--ckpt-dir` | `./checkpoints` | Checkpoint save directory (auto-converted to absolute path) |
| `--ckpt-freq` | `5000` | Save every N steps (`0` = end only) |
| `--ckpt-keep` | `1` | Number of recent checkpoints to retain |
| `--resume` | `false` | Resume from latest checkpoint (params + optimizer + step) |

### Evaluation & Sampling

| Argument | Default | Description |
|----------|---------|-------------|
| `--log-freq` | `100` | Log training metrics every N steps |
| `--eval-freq` | `500` | Validation loss eval frequency |
| `--eval-batches` | `4` | Number of validation batches per eval |
| `--sample-freq` | `1000` | Sample generation frequency |
| `--sample-num-steps` | `50` | ODE denoising steps for sample previews |
| `--sample-cfg-scale` | `1.0` | CFG scale for sample previews (`1.0` = no guidance) |

### FID & Metrics

| Argument | Default | Description |
|----------|---------|-------------|
| `--fid-freq` | `10000` | FID computation frequency (`0` = disable) |
| `--num-fid-samples` | `4000` | Number of samples for FID |
| `--fid-num-steps` | `50` | ODE steps for FID generation |
| `--fid-cfg-scale` | `1.0` | CFG scale for FID generation |
| `--fid-batch-size` | `256` | Total batch size for FID generation |
| `--fid-eval-local-batch` | `32` | Per-device batch for FID generation |
| `--inception-score` | `true` | Enable Inception Score computation |
| `--inception-score-splits` | `10` | Number of splits for IS |
| `--precision-recall` | `true` | Enable Precision/Recall (kNN-manifold) |
| `--pr-k` | `3` | k for kNN in Precision/Recall |
| `--pr-max-samples` | `5000` | Sample cap for PR (may be O(N²)) |

### Advanced / Preflight

| Argument | Default | Description |
|----------|---------|-------------|
| `--preflight-checks` | `false` | Run memory/shape validation before training |
| `--preflight-only` | `false` | Exit after preflight checks |
| `--linear-probe` | `false` | Enable linear probe accuracy on val features |
| `--block-corr-freq` | `0` | Block feature correlation frequency |
| `--no-wandb` | `false` | Disable Weights & Biases logging |
| `--mock-data` | `false` | Use synthetic data for testing |

## Dispersive Loss Details

The Dispersive Loss encourages diversity in hidden representations:

1. Extract features from a specified transformer block (`--disp-layer`)
2. Flatten features per sample → compute pairwise squared L2 distances (normalised by feature dim)
3. Zero out the diagonal (self-pairs)
4. Compute `log(mean(exp(-dist²)))` — penalises collapsed representations
5. Scale by `--disp-lambda` and add to the generative MSE loss

The loss is minimised when representations are maximally dispersed (far apart), preventing mode collapse in the feature space.

## CFG (Classifier-Free Guidance) Training

With `--cfg-prob 0.1`, during training 10% of labels are randomly replaced with a null class token. This enables classifier-free guidance at inference time:

- **Training**: `--cfg-prob 0.1` (drops 10% of labels)
- **Sampling**: `--sample-cfg-scale 1.5` (guidance strength, `1.0` = no guidance)
- **FID**: `--fid-cfg-scale 1.5` (for FID with guidance)

## Project Structure

```
disp-loss-sit/
├── train.py                # Training script (JAX/Flax, TPU-optimised)
├── sample.py               # Standalone sampling script
├── prepare_data.py         # Dataset preparation utilities
├── prepare_data_tpu.py     # TPU-specific data preparation
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── src/
    ├── model.py            # SelfFlowDiT backbone (configurable class_dropout_prob)
    ├── jax_compat.py       # replicate/unreplicate helpers
    ├── sampling.py         # Flow matching ODE/SDE samplers
    ├── metrics.py          # FID, IS, Precision/Recall
    ├── fid_utils.py        # FID statistics computation
    └── utils.py            # Position encoding utilities
```

## Environment Notes

This branch includes automatic Kaggle TPU compatibility fixes:

- `JAX_USE_SHARDY=0` / `ENABLE_SHARDY=0` — disables SDY dialect
- `JAX_PLATFORMS=tpu,cpu` — explicit platform ordering
- Orbax checkpoint fallback to `flax.serialization` (msgpack) when `flax.training.checkpoints` is unavailable

## Acknowledgments

- [SiT](https://github.com/willisma/SiT) — Scalable Interpolant Transformers
- [REPA](https://github.com/sihyun-yu/REPA) — Representation Alignment for Generation
- [Self-Flow](https://github.com/thanhlamauto/Self-Flow) — Self-Supervised Flow Matching
