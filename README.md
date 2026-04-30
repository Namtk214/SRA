# LayerSync SiT Baseline (JAX / TPU)

**LayerSync** baseline for ImageNet 256×256, implemented in JAX/Flax and optimised for Kaggle TPU (v5p-8) training.

This branch extends the vanilla SiT backbone with a **LayerSync regularisation** loss that encourages alignment between weak (early) and strong (late) block representations via cosine similarity, guided by a tunable lambda weight.

## Features

| Feature | Description |
|---------|-------------|
| **Flow Matching** | Velocity prediction with uniform timestep sampling (`τ ∈ [0,1]`) |
| **LayerSync Loss** | Cosine alignment between weak/strong block features (`--layersync-lambda`) |
| **Classifier-Free Guidance** | `--cfg-dropout-rate` label dropout during training; `--sample-cfg-scale` at inference |
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
    --cfg-dropout-rate 0.1 \
    --layersync-lambda 1.0 \
    --layersync-weak-layer 8 \
    --layersync-strong-layer 16 \
    --vae-model /kaggle/input/models/damtrunghieu/sdvae-ema/flax/default/1 \
    --inception-score-weights /kaggle/input/models/ctlcmleon/inception-v3/pytorch/default/1/inception_v3_google-0cc3c7bd.pth \
    --ckpt-dir ./checkpoints/layersync \
    --ckpt-freq 5000 \
    --ckpt-keep 2 \
    --wandb-project layersync-jax
```

### Resume from Checkpoint

```bash
python train.py \
    --data-path /path/to/imagenet_latents/*.ar \
    --model-size XL \
    --batch-size 256 \
    --cfg-dropout-rate 0.1 \
    --layersync-lambda 1.0 \
    --vae-model /kaggle/input/models/damtrunghieu/sdvae-ema/flax/default/1 \
    --inception-score-weights /kaggle/input/models/ctlcmleon/inception-v3/pytorch/default/1/inception_v3_google-0cc3c7bd.pth \
    --ckpt-dir ./checkpoints/layersync \
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

### Model Weights (Kaggle)

| Argument | Default | Description |
|----------|---------|-------------|
| `--vae-model` | `stabilityai/sd-vae-ft-ema` | Path to VAE model (local Flax dir or HF repo ID) |
| `--vae-hf-config` | `stabilityai/sd-vae-ft-ema` | HF config ID fallback when VAE dir has no `config.json` |
| `--inception-score-weights` | `None` | Path to local Inception-v3 `.pth` weights for IS/FID |

### Classifier-Free Guidance

| Argument | Default | Description |
|----------|---------|-------------|
| `--cfg-dropout-rate` | `0.1` | Label dropout probability for CFG training |
| `--sample-cfg-scale` | `1.0` | CFG scale at sampling time |

### LayerSync

| Argument | Default | Description |
|----------|---------|-------------|
| `--layersync-lambda` | `0.0` | LayerSync loss weight (`0.0` = disabled) |
| `--layersync-weak-layer` | `8` | Early (weak) block index for feature extraction |
| `--layersync-strong-layer` | `16` | Late (strong) block index for feature extraction |

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

## Project Structure

```
layersync-sit/
├── train.py                # Training script (JAX/Flax, TPU-optimised)
├── sample.py               # Standalone sampling script
├── prepare_data.py         # Dataset preparation utilities
├── prepare_data_tpu.py     # TPU-specific data preparation
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── src/
    ├── model.py            # SelfFlowDiT with class_dropout_prob
    ├── jax_compat.py       # replicate_tree / unreplicate_tree helpers
    ├── sampling.py         # Flow matching ODE/SDE samplers
    ├── metrics.py          # FID, IS, Precision/Recall
    ├── fid_utils.py        # FID statistics computation
    └── utils.py            # Position encoding utilities
```

## LayerSync Loss Details

LayerSync regularisation aligns representations from two transformer blocks:

1. Extract features from the **weak layer** (early block, e.g. layer 8) and **strong layer** (late block, e.g. layer 16)
2. Stop gradients on the strong layer features (treat as target)
3. Compute **negative mean cosine similarity** between L2-normalised token features
4. Scale by `--layersync-lambda` and add to the generative MSE loss

The intuition is that encouraging early blocks to produce representations similar to late blocks improves feature reuse and training efficiency.

## Environment Notes

This branch includes automatic Kaggle TPU compatibility fixes:

- `JAX_USE_SHARDY=0` / `ENABLE_SHARDY=0` — disables SDY dialect
- `JAX_PLATFORMS=tpu,cpu` — explicit platform ordering
- Orbax checkpoint fallback to `flax.serialization` (msgpack) when `flax.training.checkpoints` is unavailable

## Acknowledgments

- [SiT](https://github.com/willisma/SiT) — Scalable Interpolant Transformers
- [REPA](https://github.com/sihyun-yu/REPA) — Representation Alignment for Generation
- [Self-Flow](https://github.com/thanhlamauto/Self-Flow) — Self-Supervised Flow Matching
