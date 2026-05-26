#!/usr/bin/env python3
"""
Sample images from a trained Self-Flow / LayerSync model (JAX/Flax).

Supports the current Orbax checkpoint bundle format:
    ckpt_dir/
      checkpoint_<step>/          ← main (params + opt_state + step)
      ema/
        checkpoint_<step>/        ← EMA params (flat dict)

Usage:
    # From bundle dir (auto-finds latest EMA checkpoint):
    python sample.py --ckpt-dir /path/to/layersync-L-v3 \
        --model-size XL --num-fid-samples 50000 --batch-size 64

    # Direct Orbax checkpoint path (backward compat):
    python sample.py --ckpt /path/to/ema/checkpoint_1460000 \
        --model-size XL --num-fid-samples 50000

Output: NPZ file compatible with ADM evaluation suite.
"""

import os
import re
import math
import time
import glob
import argparse
import functools
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image
from tqdm import tqdm
from einops import rearrange

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


# ── Checkpoint loading utilities ──────────────────────────────────────────────

def _align_keys(d, reference_keys):
    """Recursively rename checkpoint keys to match the model's param structure.

    Flax nn.remat wraps DiTBlock → CheckpointDiTBlock. Checkpoints saved
    with/without remat will have mismatched keys.
    """
    if not isinstance(d, dict):
        return d
    new_d = {}
    for k, v in d.items():
        new_key = k
        if k.startswith("CheckpointDiTBlock_"):
            plain_key = k.replace("CheckpointDiTBlock_", "DiTBlock_", 1)
            if plain_key in reference_keys and k not in reference_keys:
                new_key = plain_key
        elif k.startswith("DiTBlock_"):
            remat_key = k.replace("DiTBlock_", "CheckpointDiTBlock_", 1)
            if remat_key in reference_keys and k not in reference_keys:
                new_key = remat_key
        child_ref = reference_keys
        if isinstance(reference_keys, dict) and new_key in reference_keys:
            child_val = reference_keys[new_key]
            if isinstance(child_val, dict):
                child_ref = child_val
        new_d[new_key] = _align_keys(v, child_ref)
    return new_d


def _harmonize_num_classes(ckpt_params, model_params):
    """Auto-detect and fix num_classes mismatch in LabelEmbedder embedding."""
    def _find_label_embedding(p, path=""):
        if not isinstance(p, dict):
            return None, None
        for k, v in p.items():
            full = f"{path}/{k}"
            if k == "LabelEmbedder_0":
                embed = v.get("Embed_0", {}).get("embedding", None)
                if embed is not None:
                    return embed, full + "/Embed_0/embedding"
            result, rpath = _find_label_embedding(v, full)
            if result is not None:
                return result, rpath
        return None, None

    ckpt_emb, ckpt_path = _find_label_embedding(ckpt_params)
    model_emb, _ = _find_label_embedding(model_params)
    if ckpt_emb is None or model_emb is None:
        return ckpt_params
    ckpt_nc, model_nc = ckpt_emb.shape[0], model_emb.shape[0]
    if ckpt_nc == model_nc:
        return ckpt_params

    print(f"[harmonize] num_classes mismatch: ckpt={ckpt_nc} vs model={model_nc}. Adjusting...")
    if ckpt_nc > model_nc:
        new_emb = ckpt_emb[:model_nc]
    else:
        pad = jnp.zeros((model_nc - ckpt_nc, *ckpt_emb.shape[1:]), dtype=ckpt_emb.dtype)
        new_emb = jnp.concatenate([ckpt_emb, pad], axis=0)

    path_parts = [p for p in ckpt_path.split("/") if p]
    def _set_nested(d, keys, val):
        if len(keys) == 1:
            d[keys[0]] = val
            return
        _set_nested(d[keys[0]], keys[1:], val)
    _set_nested(ckpt_params, path_parts, new_emb)
    return ckpt_params


def _find_latest_orbax_dir(base_dir):
    """Find the latest checkpoint_<step> directory in base_dir."""
    dirs = sorted(
        (d for d in glob.glob(os.path.join(base_dir, "checkpoint_*")) if os.path.isdir(d)),
        key=lambda p: int(re.search(r'checkpoint_(\d+)', os.path.basename(p)).group(1))
            if re.search(r'checkpoint_(\d+)', os.path.basename(p)) else 0,
    )
    return dirs[-1] if dirs else None


def _load_orbax_checkpoint(ckpt_path):
    """Load an Orbax PyTree checkpoint and move arrays to CPU."""
    import orbax.checkpoint as ocp
    print(f"Loading Orbax checkpoint: {ckpt_path}")
    checkpointer = ocp.PyTreeCheckpointer()
    raw = checkpointer.restore(ckpt_path)
    cpu_device = jax.devices('cpu')[0]
    raw = jax.tree_util.tree_map(
        lambda x: jax.device_put(x, cpu_device) if hasattr(x, 'shape') else x,
        raw,
    )
    return raw


def resolve_ckpt_path(args):
    """Resolve the EMA checkpoint path from --ckpt-dir or --ckpt.

    Returns: (ema_ckpt_path, step)
    """
    # Direct path: --ckpt points to an Orbax checkpoint directory
    if args.ckpt and os.path.isdir(args.ckpt):
        m = re.search(r'checkpoint_(\d+)', os.path.basename(args.ckpt))
        step = int(m.group(1)) if m else 0
        return args.ckpt, step

    # Bundle dir: --ckpt-dir points to the bundle root
    ckpt_dir = getattr(args, 'ckpt_dir', None) or args.ckpt
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")

    # Try ema/ subdirectory first (preferred for sampling)
    ema_dir = os.path.join(ckpt_dir, "ema")
    if os.path.isdir(ema_dir):
        latest = _find_latest_orbax_dir(ema_dir)
        if latest:
            m = re.search(r'checkpoint_(\d+)', os.path.basename(latest))
            step = int(m.group(1)) if m else 0
            print(f"Found EMA checkpoint: {latest} (step {step})")
            return latest, step

    # Fallback: main checkpoint (extract 'params' key)
    latest = _find_latest_orbax_dir(ckpt_dir)
    if latest:
        m = re.search(r'checkpoint_(\d+)', os.path.basename(latest))
        step = int(m.group(1)) if m else 0
        print(f"Found main checkpoint: {latest} (step {step})")
        return latest, step

    raise FileNotFoundError(f"No checkpoint_* found in {ckpt_dir} or {ema_dir}")


def load_model(ckpt_path, model_size="XL", cfg_dropout_rate=DEFAULT_CFG_DROPOUT_RATE, step=0):
    """Load the DiT backbone from an Orbax checkpoint.

    Handles:
      - EMA checkpoint (flat params dict) or main checkpoint (params in 'params' key)
      - DiTBlock_* ↔ CheckpointDiTBlock_* key remapping (nn.remat)
      - num_classes mismatch auto-fix
    """
    config = _model_config_for_size(model_size, cfg_dropout_rate=cfg_dropout_rate)
    model = SelfFlowDiT(**config, per_token=False)

    # Initialize on CPU
    cpu_device = jax.devices('cpu')[0]
    with jax.default_device(cpu_device):
        key = jax.random.PRNGKey(0)
        patch_dim = config["in_channels"] * config["patch_size"] ** 2
        n_patches = (config["input_size"] // config["patch_size"]) ** 2
        dummy_x = jnp.ones((1, n_patches, patch_dim))
        dummy_t = jnp.ones((1,))
        dummy_vec = jnp.ones((1,), dtype=jnp.int32)
        variables = model.init(key, dummy_x, timesteps=dummy_t, vector=dummy_vec, deterministic=True)
        params = variables["params"]

    if ckpt_path is not None and os.path.exists(ckpt_path):
        raw = _load_orbax_checkpoint(ckpt_path)

        # Determine if this is a main checkpoint (has 'params' key) or EMA (flat dict)
        if isinstance(raw, dict) and 'params' in raw and 'opt_state' in raw:
            print(f"Main checkpoint detected (step={raw.get('step', step)}). Extracting 'params'...")
            ckpt_params = raw['params']
        else:
            print(f"EMA checkpoint detected (flat params dict).")
            ckpt_params = raw

        # Align keys and harmonize
        ckpt_params = _align_keys(ckpt_params, params)
        ckpt_params = _harmonize_num_classes(ckpt_params, params)
        ckpt_params = jax.tree_util.tree_map(jnp.asarray, ckpt_params)

        # Merge into model params
        for k in list(params.keys()):
            if k in ckpt_params:
                params[k] = ckpt_params[k]

        total = sum(v.size for v in jax.tree_util.tree_leaves(params))
        print(f"Loaded {total:,} parameters")
    else:
        print(f"WARNING: No checkpoint loaded — using random init!")

    return model, params


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


def build_sample_step_pmap(model, vae, scale_factor, shift_factor, use_cfg=False):
    """Build pmap-compiled sampling function for multi-device TPU."""

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
        description="Sample images from SiT/LayerSync model (JAX) — TPU optimized")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to a specific Orbax checkpoint dir (e.g. ema/checkpoint_1460000)")
    parser.add_argument("--ckpt-dir", type=str, default=None,
                        help="Path to checkpoint bundle dir (auto-finds latest EMA). "
                             "Priority: --ckpt > --ckpt-dir")
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
    parser.add_argument("--model-size", type=str, default="XL",
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

    if args.ckpt is None and args.ckpt_dir is None:
        parser.error("Specify --ckpt or --ckpt-dir")

    if not 0.0 <= args.cfg_dropout_rate < 1.0:
        raise ValueError("--cfg-dropout-rate must be in [0.0, 1.0)")
    if args.cfg_dropout_rate <= 0.0 and args.cfg_scale > 1.0:
        raise ValueError("--cfg-scale > 1 requires --cfg-dropout-rate > 0")

    # ── Resolve checkpoint ────────────────────────────────────────────────
    ckpt_path, step = resolve_ckpt_path(args)
    print(f"Checkpoint: {ckpt_path} (step {step})")

    # ── Device setup ─────────────────────────────────────────────────────
    num_devices = jax.device_count()
    local_devices = jax.local_devices()
    print(f"\n=== SiT-{args.model_size} Sampler (TPU-optimized pmap) ===")
    print(f"Devices: {num_devices}x {local_devices[0].platform.upper()}")
    print(f"Batch: {args.batch_size}/device × {num_devices} devices "
          f"= {args.batch_size * num_devices} total/iter")
    print(f"Samples: {args.num_fid_samples}, Steps: {args.num_steps}, "
          f"CFG: {args.cfg_scale}")

    # ── Load model & VAE ─────────────────────────────────────────────────
    model, params = load_model(
        ckpt_path, model_size=args.model_size,
        cfg_dropout_rate=args.cfg_dropout_rate,
        step=step)
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
    warmup_rngs = jax.random.split(warmup_rng, num_devices)
    warmup_labels = jax.random.randint(
        jax.random.PRNGKey(0), (num_devices, bs_per_device), 0, 1000)

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
    first_images = np.asarray(warmup_images)
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

    # Output filename includes step number for traceability
    npz_name = f"samples_{total_samples}_step{step}.npz" if step > 0 else f"samples_{total_samples}.npz"
    npz_path = output_dir / npz_name
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
