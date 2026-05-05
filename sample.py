#!/usr/bin/env python3
"""
Sample images from a trained Self-Flow diffusion model (JAX/Flax).

Optimized for TPU v5p-8 (8 cores):
  - jax.pmap across all TPU cores for 8x throughput
  - bfloat16 for TPU-native compute
  - Orbax checkpoint loading (TPU-trained checkpoints)
  - Static CFG branching to avoid tracing errors

Usage (Kaggle TPU v5p-8):
    python sample.py --ckpt path/to/ema/checkpoint_420000 \
        --model-size B --num-fid-samples 50000 --batch-size 64

Output: NPZ file compatible with ADM evaluation suite.
"""

import os
import math
import time
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

from src.model import SelfFlowDiT
from src.sampling import denoise_loop


DIT_VARIANTS = {
    "S":  {"hidden_size": 384,  "depth": 12, "num_heads": 6},
    "B":  {"hidden_size": 768,  "depth": 12, "num_heads": 12},
    "L":  {"hidden_size": 1024, "depth": 24, "num_heads": 16},
    "XL": {"hidden_size": 1152, "depth": 28, "num_heads": 16},
}

DEFAULT_CFG_DROPOUT_RATE = 0.1


def _model_config_for_size(model_size, cfg_dropout_rate=DEFAULT_CFG_DROPOUT_RATE):
    """Return the full model-init config dict for a DiT variant name."""
    variant = DIT_VARIANTS[model_size.upper()]
    return dict(
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=variant["hidden_size"],
        depth=variant["depth"],
        num_heads=variant["num_heads"],
        mlp_ratio=4.0,
        num_classes=1000,
        learn_sigma=True,
        compatibility_mode=True,
        class_dropout_prob=cfg_dropout_rate,
    )


def create_npz_from_samples(samples, output_path):
    """Save samples to NPZ file for ADM evaluation."""
    np.savez(output_path, arr_0=samples)
    print(f"Saved {len(samples)} samples to {output_path}")


def load_vae(vae_model="stabilityai/sd-vae-ft-mse", dtype=jnp.bfloat16):
    """Load the SD-VAE for decoding latents to images."""
    from diffusers.models import FlaxAutoencoderKL

    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        vae_model,
        from_pt=True,
        dtype=dtype,
    )
    scale_factor = 0.18215
    shift_factor = 0.0
    return vae, vae_params, scale_factor, shift_factor


def load_model(ckpt_path=None, model_size="B", cfg_dropout_rate=DEFAULT_CFG_DROPOUT_RATE):
    """Load the DiT backbone from an Orbax checkpoint (TPU-trained).

    Handles:
      - TPU sharding → single/multi-device re-sharding
      - DiTBlock_* ↔ CheckpointDiTBlock_* key remapping (nn.remat)
      - Skipping feature_head / SimpleHead (not needed for sampling)
    """
    config = _model_config_for_size(model_size, cfg_dropout_rate=cfg_dropout_rate)
    model = SelfFlowDiT(**config, per_token=False)

    # Initialize parameters with random key
    key = jax.random.PRNGKey(0)
    patch_dim = config["in_channels"] * config["patch_size"] ** 2
    n_patches = (config["input_size"] // config["patch_size"]) ** 2
    dummy_x = jnp.ones((1, n_patches, patch_dim))
    dummy_t = jnp.ones((1,))
    dummy_vec = jnp.ones((1,), dtype=jnp.int32)

    variables = model.init(key, dummy_x, timesteps=dummy_t, vector=dummy_vec, deterministic=True)
    params = variables["params"]

    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")

        import orbax.checkpoint as ocp

        # Build target tree: remap CheckpointDiTBlock → DiTBlock for ckpt
        ckpt_target = {}
        for k, v in params.items():
            if k in ('SimpleHead_0', 'feature_head'):
                continue
            ck = k.replace('CheckpointDiTBlock_', 'DiTBlock_') if k.startswith('CheckpointDiTBlock_') else k
            ckpt_target[ck] = v

        # Use first device for restore, replicate later
        sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        restore_args = jax.tree_util.tree_map(
            lambda x: ocp.ArrayRestoreArgs(
                sharding=sharding, global_shape=x.shape, dtype=x.dtype),
            ckpt_target)

        restored = ocp.PyTreeCheckpointer().restore(
            ckpt_path, item=ckpt_target, restore_args=restore_args)

        # Remap DiTBlock → CheckpointDiTBlock and merge back
        target_keys = set(params.keys())
        for k, val in restored.items():
            new_key = k
            if k.startswith('DiTBlock_') and k not in target_keys:
                new_key = k.replace('DiTBlock_', 'CheckpointDiTBlock_')
            if new_key in target_keys:
                params[new_key] = val

        total = sum(v.size for v in jax.tree_util.tree_leaves(params))
        print(f"Loaded {total:,} parameters")

    return model, params


def build_sample_step_pmap(model, vae, scale_factor, shift_factor, use_cfg=False):
    """Build pmap-compiled sampling function for multi-device TPU.

    Args:
        use_cfg: Static bool — whether to use classifier-free guidance.
                 Must be set at build time to avoid tracer errors.
    """

    @functools.partial(jax.pmap, static_broadcasted_argnums=(4, 5),
                       axis_name="devices")
    def sample_batch_pmap(
        params,
        vae_params,
        rng,
        class_labels,
        # static:
        batch_size_per_device,
        num_steps,
        # traced (per-device):
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
            (batch_size_per_device, latent_channels, latent_size, latent_size),
            dtype=jnp.float32
        )

        x = rearrange(
            noise,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=patch_size, p2=patch_size
        )
        token_h = latent_size // patch_size
        token_w = latent_size // patch_size

        # CFG: static branch (no tracer issue)
        if use_cfg:
            x = jnp.concatenate([x, x], axis=0)
            null_labels = jnp.full_like(class_labels, 1000)
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
        # cfg_scale: pass None when no CFG to avoid tracer bool error in denoise_loop
        effective_cfg = cfg_scale if use_cfg else None
        samples = denoise_loop(
            model_fn=model_fn,
            x=x,
            rng=denoise_rng,
            num_steps=num_steps,
            cfg_scale=effective_cfg,
            guidance_low=guidance_low,
            guidance_high=guidance_high,
            mode="SDE",
            reverse=False,
        )

        if use_cfg:
            samples = samples[batch_size_per_device:]

        # Unpatchify → NCHW latent
        samples = rearrange(
            samples,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=token_h, w=token_w,
            p1=patch_size, p2=patch_size, c=latent_channels
        )

        # VAE decode
        latents = samples / scale_factor + shift_factor
        latents = jnp.transpose(latents, (0, 2, 3, 1))  # NCHW → NHWC

        images = vae.apply({"params": vae_params}, latents, method=vae.decode).sample
        images = jnp.transpose(images, (0, 2, 3, 1))  # NCHW → NHWC
        images = (images + 1.0) / 2.0
        images = jnp.clip(images, 0.0, 1.0)
        images = (images * 255.0).astype(jnp.uint8)

        return images

    return sample_batch_pmap


def replicate(tree, num_devices):
    """Replicate a pytree across devices for pmap."""
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (num_devices,) + x.shape).copy(),
        tree)


def main():
    parser = argparse.ArgumentParser(
        description="Sample images from SiT model (JAX) — TPU v5p-8 optimized")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="./samples", help="Output directory")
    parser.add_argument("--num-fid-samples", type=int, default=50000,
                        help="Number of samples to generate")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size PER DEVICE (total = batch_size × num_devices)")
    parser.add_argument("--num-steps", type=int, default=250, help="Number of diffusion steps")
    parser.add_argument("--mode", type=str, default="SDE", choices=["SDE"], help="Sampling mode")
    parser.add_argument("--seed", type=int, default=31, help="Random seed")
    parser.add_argument("--save-images", action="store_true", default=False,
                        help="Save individual PNG images (slow, disabled by default)")
    parser.add_argument("--model-size", type=str, default="B",
                        choices=["S", "B", "L", "XL"], help="DiT backbone size")
    parser.add_argument("--vae-model", type=str, default="stabilityai/sd-vae-ft-mse",
                        choices=["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"],
                        help="HuggingFace VAE model ID")
    parser.add_argument("--cfg-scale", type=float, default=1.0,
                        help="CFG scale (1.0 = no guidance)")
    parser.add_argument("--cfg-dropout-rate", type=float, default=DEFAULT_CFG_DROPOUT_RATE,
                        help="CFG class dropout rate used during training")
    parser.add_argument("--no-cfg-dropout", dest="cfg_dropout_rate",
                        action="store_const", const=0.0)
    parser.add_argument("--guidance-low", type=float, default=0.0)
    parser.add_argument("--guidance-high", type=float, default=0.7)
    parser.add_argument("--ref-batch", type=str, default=None,
                        help="Path to reference NPZ for automatic FID evaluation")
    args = parser.parse_args()

    if not 0.0 <= args.cfg_dropout_rate < 1.0:
        raise ValueError("--cfg-dropout-rate must be in [0.0, 1.0)")
    if args.cfg_dropout_rate <= 0.0 and args.cfg_scale > 1.0:
        raise ValueError("--cfg-scale > 1 requires --cfg-dropout-rate > 0")

    # ── Device setup ─────────────────────────────────────────────────────
    num_devices = jax.device_count()
    local_devices = jax.local_devices()
    print(f"=== SiT-{args.model_size} Sampler (TPU-optimized pmap) ===")
    print(f"Devices: {num_devices}x {local_devices[0].platform.upper()}")
    print(f"Batch: {args.batch_size}/device × {num_devices} devices "
          f"= {args.batch_size * num_devices} total/iter")
    print(f"Samples: {args.num_fid_samples}, Steps: {args.num_steps}, "
          f"CFG: {args.cfg_scale}")

    # ── Load model & VAE ─────────────────────────────────────────────────
    model, params = load_model(
        args.ckpt, model_size=args.model_size,
        cfg_dropout_rate=args.cfg_dropout_rate)
    vae, vae_params, scale_factor, shift_factor = load_vae(
        vae_model=args.vae_model)

    # Replicate params across all devices
    print(f"Replicating params to {num_devices} devices...")
    params_rep = replicate(params, num_devices)
    vae_params_rep = replicate(vae_params, num_devices)

    # ── Build pmap function ──────────────────────────────────────────────
    use_cfg = args.cfg_scale > 1.0
    sample_fn = build_sample_step_pmap(
        model, vae, scale_factor, shift_factor, use_cfg=use_cfg)

    # ── JIT warmup ───────────────────────────────────────────────────────
    bs_per_device = args.batch_size
    total_per_iter = bs_per_device * num_devices
    total_samples = args.num_fid_samples
    num_batches = math.ceil(total_samples / total_per_iter)

    print(f"\nJIT compiling (first batch)... This may take a few minutes.")
    t_compile = time.time()

    rng = jax.random.PRNGKey(args.seed)
    rng, warmup_rng = jax.random.split(rng)
    # Per-device RNGs
    warmup_rngs = jax.random.split(warmup_rng, num_devices)
    warmup_labels = jax.random.randint(
        jax.random.PRNGKey(0), (num_devices, bs_per_device), 0, 1000)

    # Broadcast scalar args to per-device
    cfg_arr = jnp.full((num_devices,), args.cfg_scale)
    glow_arr = jnp.full((num_devices,), args.guidance_low)
    ghigh_arr = jnp.full((num_devices,), args.guidance_high)

    warmup_images = sample_fn(
        params_rep, vae_params_rep, warmup_rngs, warmup_labels,
        bs_per_device, args.num_steps,
        cfg_arr, glow_arr, ghigh_arr)
    jax.block_until_ready(warmup_images)

    compile_time = time.time() - t_compile
    print(f"  Compiled in {compile_time:.1f}s")

    # ── Sampling loop ────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        (output_dir / "images").mkdir(exist_ok=True)

    all_samples = []
    generated = 0

    # Use warmup batch as first batch
    first_images = np.asarray(warmup_images)  # (num_devices, bs, 256, 256, 3)
    first_images = first_images.reshape(-1, 256, 256, 3)
    needed = min(total_per_iter, total_samples)
    all_samples.append(first_images[:needed])
    generated += needed

    print(f"\nSampling {total_samples} images in {num_batches} iterations "
          f"({total_per_iter} imgs/iter)...")
    t_start = time.time()

    for batch_idx in tqdm(range(1, num_batches), desc="Sampling"):
        needed = min(total_per_iter, total_samples - generated)

        rng, class_rng, step_rng = jax.random.split(rng, 3)
        # Per-device RNGs and labels
        step_rngs = jax.random.split(step_rng, num_devices)
        class_labels = jax.random.randint(
            class_rng, (num_devices, bs_per_device), 0, 1000)

        images = sample_fn(
            params_rep, vae_params_rep, step_rngs, class_labels,
            bs_per_device, args.num_steps,
            cfg_arr, glow_arr, ghigh_arr)

        images_np = np.asarray(images).reshape(-1, 256, 256, 3)[:needed]
        all_samples.append(images_np)
        generated += needed

        if args.save_images:
            offset = generated - needed
            for i, img in enumerate(images_np):
                Image.fromarray(img).save(
                    output_dir / "images" / f"{offset + i:06d}.png")

    elapsed = time.time() - t_start
    total_time = compile_time + elapsed
    print(f"\nDone! Total: {total_time/60:.1f} min "
          f"(compile: {compile_time:.0f}s, sampling: {elapsed/60:.1f} min)")
    if generated > total_per_iter:
        print(f"Speed: {elapsed/(generated - total_per_iter):.4f} s/img (post-compile)")

    all_samples = np.concatenate(all_samples, axis=0)[:total_samples]
    npz_path = output_dir / f"samples_{total_samples}.npz"
    create_npz_from_samples(all_samples, npz_path)
    print(f"Shape: {all_samples.shape}, dtype: {all_samples.dtype}")

    # ── Auto FID evaluation ──────────────────────────────────────────────
    if args.ref_batch and os.path.exists(args.ref_batch):
        print(f"\n{'='*60}")
        print(f"Running ADM FID evaluation...")
        print(f"  Reference: {args.ref_batch}")
        print(f"  Samples:   {npz_path}")
        print(f"{'='*60}")
        import subprocess
        evaluator_candidates = [
            "/workspace/guided-diffusion/evaluations/evaluator.py",
            "./guided-diffusion/evaluations/evaluator.py",
            "../guided-diffusion/evaluations/evaluator.py",
        ]
        evaluator_path = None
        for p in evaluator_candidates:
            if os.path.exists(p):
                evaluator_path = p
                break
        if evaluator_path:
            result = subprocess.run(
                ["python3", evaluator_path, args.ref_batch, str(npz_path)],
                capture_output=True, text=True)
            # Print only metric lines
            for line in (result.stdout + result.stderr).split('\n'):
                if any(k in line for k in
                       ['FID', 'sFID', 'Inception Score', 'Precision', 'Recall']):
                    print(line)
            print(f"\n{'='*60}")
            print("EVALUATION COMPLETE")
            print(f"{'='*60}")
        else:
            print("WARNING: evaluator.py not found. Run FID evaluation manually.")
    elif args.ref_batch:
        print(f"WARNING: --ref-batch file not found: {args.ref_batch}")


if __name__ == "__main__":
    main()
