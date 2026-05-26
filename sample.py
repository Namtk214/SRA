#!/usr/bin/env python3
"""
Sample images from a trained SiT diffusion model (JAX/Flax).

Usage:
    # Using EMA params (recommended):
    python sample.py --ckpt-dir ./checkpoints/sit-xl --ema --output-dir ./samples

    # Using online params:
    python sample.py --ckpt-dir ./checkpoints/sit-xl --output-dir ./samples

    # Direct checkpoint path:
    python sample.py --ckpt ./checkpoints/sit-xl/checkpoint_1325000 --output-dir ./samples

This script generates images for FID evaluation, outputting an NPZ file
compatible with the ADM evaluation suite.
"""

import os
import sys
import math
import glob
import json
import argparse
import functools
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image
from tqdm import tqdm
from einops import rearrange
import collections.abc

# Import from local src/ folder
from src.model import SelfFlowDiT
from src.sampling import denoise_loop


DIT_VARIANTS = {
    "S":  {"hidden_size": 384,  "depth": 12, "num_heads": 6},
    "B":  {"hidden_size": 768,  "depth": 12, "num_heads": 12},
    "L":  {"hidden_size": 1024, "depth": 24, "num_heads": 16},
    "XL": {"hidden_size": 1152, "depth": 28, "num_heads": 16},
}


def _model_config_for_size(model_size, num_classes=1001):
    """Return the full model-init config dict for a DiT variant name (S/B/L/XL)."""
    variant = DIT_VARIANTS[model_size.upper()]
    return dict(
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=variant["hidden_size"],
        depth=variant["depth"],
        num_heads=variant["num_heads"],
        mlp_ratio=4.0,
        num_classes=num_classes,
        learn_sigma=True,
        compatibility_mode=True,
    )


def create_npz_from_samples(samples, output_path):
    """Save samples to NPZ file for ADM evaluation."""
    samples = np.stack(samples, axis=0)
    np.savez(output_path, arr_0=samples)
    print(f"Saved {len(samples)} samples to {output_path}")


def _find_latest_orbax_ckpt(ckpt_dir):
    """Find the latest checkpoint_NNNNN directory inside ckpt_dir."""
    candidates = sorted([
        d for d in os.listdir(ckpt_dir)
        if os.path.isdir(os.path.join(ckpt_dir, d))
        and d.startswith("checkpoint_")
        and os.path.isfile(os.path.join(ckpt_dir, d, "_METADATA"))
    ])
    if candidates:
        return os.path.join(ckpt_dir, candidates[-1])
    return None


def _is_orbax_dir(path):
    """Check if a directory is an Orbax checkpoint (has _METADATA file)."""
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "_METADATA"))


def _load_orbax_pytree(ckpt_path, template=None):
    """Load a pytree from an Orbax checkpoint directory."""
    import orbax.checkpoint as ocp
    checkpointer = ocp.PyTreeCheckpointer()
    if template is not None:
        return checkpointer.restore(ckpt_path, item=template)
    return checkpointer.restore(ckpt_path)


def _load_flax_ckpt(ckpt_path, template=None):
    """Load checkpoint using flax.training.checkpoints (handles both formats)."""
    from flax.training import checkpoints as flax_ckpt
    return flax_ckpt.restore_checkpoint(ckpt_path, template)


def load_model(ckpt_path=None, ckpt_dir=None, model_size="XL", use_ema=False):
    """Load the DiT backbone from checkpoint.

    Supports multiple checkpoint formats:
      1. Orbax directory with {params, opt_state, step} (current format)
      2. Orbax EMA directory (flat pytree)
      3. Legacy flax.training.checkpoints (msgpack)
      4. Flat/nested param dicts

    Args:
        ckpt_path: Direct path to a checkpoint directory or file.
        ckpt_dir: Checkpoint root directory (auto-finds latest).
        model_size: DiT variant name (S/B/L/XL).
        use_ema: If True, load EMA params from ema/ subfolder.

    Returns:
        (model, params, config) tuple.
    """
    # Resolve checkpoint path
    resolved_path = None
    ema_path = None

    if ckpt_path and os.path.exists(ckpt_path):
        resolved_path = ckpt_path
        # EMA is sibling: ../ema/checkpoint_NNNNN
        parent = os.path.dirname(ckpt_path)
        ema_parent = os.path.join(parent, "ema")
        if os.path.isdir(ema_parent):
            ema_ckpt = _find_latest_orbax_ckpt(ema_parent)
            if ema_ckpt:
                ema_path = ema_ckpt
    elif ckpt_dir and os.path.isdir(ckpt_dir):
        # Auto-find latest checkpoint in directory
        resolved_path = _find_latest_orbax_ckpt(ckpt_dir)
        if resolved_path is None:
            # Maybe ckpt_dir itself is an orbax checkpoint
            if _is_orbax_dir(ckpt_dir):
                resolved_path = ckpt_dir
        # EMA subfolder
        ema_parent = os.path.join(ckpt_dir, "ema")
        if os.path.isdir(ema_parent):
            ema_ckpt = _find_latest_orbax_ckpt(ema_parent)
            if ema_ckpt:
                ema_path = ema_ckpt

    if resolved_path is None:
        print(f"WARNING: No checkpoint found at ckpt_path={ckpt_path}, ckpt_dir={ckpt_dir}")
        print("Using randomly initialized parameters.")
        config = _model_config_for_size(model_size)
        model = SelfFlowDiT(**config, per_token=False)
        key = jax.random.PRNGKey(0)
        patch_dim = config["in_channels"] * config["patch_size"] ** 2
        n_patches = (config["input_size"] // config["patch_size"]) ** 2
        dummy_x = jnp.ones((1, n_patches, patch_dim))
        dummy_t = jnp.ones((1,))
        dummy_vec = jnp.ones((1,), dtype=jnp.int32)
        variables = model.init(key, dummy_x, timesteps=dummy_t, vector=dummy_vec, deterministic=True)
        return model, variables["params"], config

    print(f"Loading checkpoint from: {resolved_path}")
    if use_ema and ema_path:
        print(f"Will use EMA params from: {ema_path}")
    elif use_ema:
        print("WARNING: --ema requested but no EMA checkpoint found. Using online params.")
        use_ema = False

    # ── Load main checkpoint to detect num_classes ──
    params = None

    if _is_orbax_dir(resolved_path):
        # Current Orbax format: {params, opt_state, step}
        print(f"Detected Orbax checkpoint directory")
        raw = _load_orbax_pytree(resolved_path)

        if isinstance(raw, dict) and "params" in raw:
            params = raw["params"]
            step = raw.get("step", "?")
            print(f"Loaded main checkpoint at step {step}")
        else:
            # Flat pytree (e.g. EMA checkpoint directly)
            params = raw
            print(f"Loaded flat Orbax checkpoint")
    else:
        # Legacy flax checkpoint or msgpack
        print(f"Trying legacy flax checkpoint restore...")
        raw = _load_flax_ckpt(resolved_path, template=None)
        if raw is not None:
            if isinstance(raw, collections.abc.Mapping) and "params" in raw:
                params = raw["params"]
            elif isinstance(raw, collections.abc.Mapping) and "backbone" in raw:
                params = dict(raw["backbone"])
            else:
                params = raw

    if params is None:
        raise RuntimeError(f"Failed to load checkpoint from {resolved_path}")

    # ── Auto-detect num_classes from embedding shape ──
    num_classes = 1001  # default
    try:
        emb = params['LabelEmbedder_0']['Embed_0']['embedding']
        num_classes = emb.shape[0]
        print(f"Auto-detected num_classes={num_classes} from embedding shape {emb.shape}")
    except (KeyError, TypeError, AttributeError):
        print(f"Could not detect num_classes, using default={num_classes}")

    # ── Build model with correct num_classes ──
    config = _model_config_for_size(model_size, num_classes=num_classes)
    model = SelfFlowDiT(**config, per_token=False)

    # Initialize to get correct parameter structure (for key alignment)
    key = jax.random.PRNGKey(0)
    patch_dim = config["in_channels"] * config["patch_size"] ** 2
    n_patches = (config["input_size"] // config["patch_size"]) ** 2
    dummy_x = jnp.ones((1, n_patches, patch_dim))
    dummy_t = jnp.ones((1,))
    dummy_vec = jnp.ones((1,), dtype=jnp.int32)
    variables = model.init(key, dummy_x, timesteps=dummy_t, vector=dummy_vec, deterministic=True)
    model_param_keys = set(variables["params"].keys())

    # ── Align checkpoint keys with model keys ──
    # nn.remat(DiTBlock) → Flax names it "CheckpointDiTBlock_*"
    # Checkpoint may have "DiTBlock_*" or "CheckpointDiTBlock_*"
    def _align_keys(d):
        if not isinstance(d, dict):
            return d
        new_d = {}
        for k, v in d.items():
            new_key = k
            if k.startswith("DiTBlock_") and f"Checkpoint{k}" in model_param_keys:
                new_key = f"Checkpoint{k}"
            elif k.startswith("CheckpointDiTBlock_"):
                plain = k.replace("CheckpointDiTBlock_", "DiTBlock_")
                if plain in model_param_keys:
                    new_key = plain
            new_d[new_key] = _align_keys(v)
        return new_d

    params = _align_keys(params)

    # ── Load EMA params if requested ──
    if use_ema and ema_path:
        print(f"Loading EMA params from: {ema_path}")
        ema_raw = _load_orbax_pytree(ema_path)
        ema_params = _align_keys(ema_raw)
        # Verify EMA has the expected structure
        try:
            ema_emb = ema_params['LabelEmbedder_0']['Embed_0']['embedding']
            if ema_emb.shape[0] != num_classes:
                print(f"EMA embedding has {ema_emb.shape[0]} classes, truncating to {num_classes}")
                ema_params['LabelEmbedder_0']['Embed_0']['embedding'] = ema_emb[:num_classes]
        except (KeyError, TypeError):
            pass
        params = ema_params
        print(f"Using EMA params for sampling.")

    print(f"Model: SiT-{model_size}, num_classes={num_classes}")
    return model, params, config


def load_vae(vae_model="stabilityai/sd-vae-ft-mse", dtype=jnp.bfloat16):
    """Load the SD-VAE for decoding latents to images."""
    from diffusers.models import FlaxAutoencoderKL

    if os.path.isdir(vae_model):
        # Local Flax VAE
        flax_msgpack = os.path.join(vae_model, "flax_model.msgpack")
        if os.path.exists(flax_msgpack):
            vae, vae_params = FlaxAutoencoderKL.from_pretrained(
                vae_model, dtype=dtype,
            )
        else:
            vae, vae_params = FlaxAutoencoderKL.from_pretrained(
                vae_model, from_pt=True, dtype=dtype,
            )
    else:
        # HuggingFace model ID
        vae, vae_params = FlaxAutoencoderKL.from_pretrained(
            vae_model, from_pt=True, dtype=dtype,
        )
    scale_factor = 0.18215
    shift_factor = 0.0
    return vae, vae_params, scale_factor, shift_factor


def build_sample_step(model, vae, scale_factor, shift_factor, num_classes=1001):
    """Build JIT-compiled sampling function."""

    @functools.partial(jax.jit, static_argnames=("batch_size", "num_steps"))
    def sample_batch_jit(
        params,
        vae_params,
        rng,
        class_labels,
        batch_size,
        num_steps,
        cfg_scale,
        guidance_low,
        guidance_high,
    ):
        latent_channels = 4
        latent_size = 32
        patch_size = 2

        rng, noise_rng = jax.random.split(rng)
        noise = jax.random.normal(
            noise_rng,
            (batch_size, latent_channels, latent_size, latent_size),
            dtype=jnp.bfloat16
        )

        # Patchify matching the training dataloader
        x = rearrange(
            noise,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=patch_size, p2=patch_size
        )
        token_h = latent_size // patch_size
        token_w = latent_size // patch_size

        # Enable CFG
        use_cfg = cfg_scale > 1.0
        if use_cfg:
            x = jnp.concatenate([x, x], axis=0)
            null_labels = jnp.full_like(class_labels, num_classes - 1)
            class_labels = jnp.concatenate([null_labels, class_labels], axis=0)

        def model_fn(z_x, t):
            return model.apply(
                {"params": params},
                z_x,
                timesteps=t,
                vector=class_labels,
                deterministic=True
            )

        rng, denoise_rng = jax.random.split(rng)
        samples = denoise_loop(
            model_fn=model_fn,
            x=x,
            rng=denoise_rng,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            guidance_low=guidance_low,
            guidance_high=guidance_high,
            mode="SDE",
            reverse=False,
        )

        if use_cfg:
            samples = samples[batch_size:]

        # Unpatchify → NCHW
        samples = rearrange(
            samples,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=token_h, w=token_w,
            p1=patch_size, p2=patch_size, c=latent_channels
        )

        # Decode latents using VAE
        latents = samples / scale_factor + shift_factor
        latents = jnp.transpose(latents, (0, 2, 3, 1))  # NCHW → NHWC

        images = vae.apply({"params": vae_params}, latents, method=vae.decode).sample
        images = jnp.transpose(images, (0, 2, 3, 1))  # NCHW → NHWC
        images = (images + 1.0) / 2.0
        images = jnp.clip(images, 0.0, 1.0)

        images = (images * 255.0).astype(jnp.uint8)
        return images

    return sample_batch_jit


def main():
    parser = argparse.ArgumentParser(description="Sample images from SiT model (JAX)")
    # Checkpoint args
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Direct path to a checkpoint directory (Orbax) or file (msgpack)")
    parser.add_argument("--ckpt-dir", type=str, default=None,
                        help="Checkpoint root directory (auto-finds latest checkpoint_NNNNN)")
    parser.add_argument("--ema", action="store_true", default=False,
                        help="Use EMA params instead of online params (recommended for eval)")
    # Output args
    parser.add_argument("--output-dir", type=str, default="./samples", help="Output directory")
    parser.add_argument("--num-fid-samples", type=int, default=50000, help="Number of samples to generate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    # Sampling args
    parser.add_argument("--num-steps", type=int, default=250, help="Number of diffusion steps")
    parser.add_argument("--mode", type=str, default="SDE", choices=["SDE"], help="Sampling mode")
    parser.add_argument("--seed", type=int, default=31, help="Random seed")
    parser.add_argument("--cfg-scale", type=float, default=1.0, help="CFG scale (1.0 = no guidance)")
    parser.add_argument("--guidance-low", type=float, default=0.0, help="Lower guidance bound")
    parser.add_argument("--guidance-high", type=float, default=0.7, help="Upper guidance bound")
    # Model/VAE args
    parser.add_argument("--model-size", type=str, default="XL",
                        choices=["S", "B", "L", "XL"], help="DiT backbone size")
    parser.add_argument("--vae-model", type=str, default="stabilityai/sd-vae-ft-mse",
                        help="HuggingFace VAE model ID or local path")
    # Image saving
    parser.add_argument("--save-images", action="store_true", default=False,
                        help="Also save individual PNG images")
    args = parser.parse_args()

    if args.ckpt is None and args.ckpt_dir is None:
        parser.error("Must specify --ckpt or --ckpt-dir")

    print(f"=== SiT-{args.model_size} Sampling ===")
    print(f"Samples: {args.num_fid_samples}, Steps: {args.num_steps}, CFG: {args.cfg_scale}")
    print(f"EMA: {args.ema}, Seed: {args.seed}")

    rng = jax.random.PRNGKey(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        (output_dir / "images").mkdir(exist_ok=True)

    # Load model
    model, params, config = load_model(
        ckpt_path=args.ckpt,
        ckpt_dir=args.ckpt_dir,
        model_size=args.model_size,
        use_ema=args.ema,
    )

    # Load VAE
    print(f"Loading VAE: {args.vae_model}")
    vae, vae_params, scale_factor, shift_factor = load_vae(vae_model=args.vae_model)

    # Build sampling function
    sample_step_fn = build_sample_step(
        model, vae, scale_factor, shift_factor,
        num_classes=config["num_classes"],
    )

    total_samples = args.num_fid_samples
    num_batches = math.ceil(total_samples / args.batch_size)

    all_samples = []

    print(f"Generating {total_samples} samples in {num_batches} batches of {args.batch_size}...")
    for batch_idx in tqdm(range(num_batches), desc="Sampling"):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, total_samples)
        needed = batch_end - batch_start

        rng, class_rng, step_rng = jax.random.split(rng, 3)
        # Static batch size for JIT; slice afterwards
        class_labels = jax.random.randint(class_rng, (args.batch_size,), 0, 1000)

        images = sample_step_fn(
            params=params,
            vae_params=vae_params,
            rng=step_rng,
            class_labels=class_labels,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
        )

        images_np = np.asarray(images)[:needed]
        all_samples.append(images_np)

        if args.save_images:
            for i, img in enumerate(images_np):
                global_idx = batch_start + i
                Image.fromarray(img).save(output_dir / "images" / f"{global_idx:06d}.png")

    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = all_samples[:args.num_fid_samples]

    npz_path = output_dir / f"samples_{len(all_samples)}.npz"
    create_npz_from_samples(list(all_samples), npz_path)

    print(f"\nDone! NPZ saved at: {npz_path}")
    print(f"Shape: {all_samples.shape}, dtype: {all_samples.dtype}")


if __name__ == "__main__":
    main()
