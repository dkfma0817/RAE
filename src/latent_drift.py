#!/usr/bin/env python3
"""
Measure latent drift under iterative reconstruction:
  x_{k+1} = D(E(x_k))
  z_k = E(x_k)

Reports:
- MSE(z_k, z_0), cosine(z_k, z_0)
- MSE(z_k, z_{k-1})
- ||z_k||, mean/std
Optionally save x_k frames.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs
from stage1 import RAE


DEFAULT_IMAGE = Path("assets/pixabay_cat.png")


def get_device(explicit: str | None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    return transforms.ToTensor()(image).unsqueeze(0)  # (1,3,H,W)


def flatten_latent(z: torch.Tensor) -> torch.Tensor:
    # z can be (B,N,C) or (B,C,H,W) etc. Flatten per sample to (B, D)
    return z.reshape(z.shape[0], -1)


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).pow(2).mean().item()


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    # a,b: (B,D)
    a = a / (a.norm(dim=1, keepdim=True) + eps)
    b = b / (b.norm(dim=1, keepdim=True) + eps)
    return (a * b).sum(dim=1).mean().item()


@torch.no_grad()
def step_T(rae: RAE, x: torch.Tensor, clamp: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One step:
      z = E(x)
      x_next = D(z)
    Returns (z, x_next)
    """
    z = rae.encode(x)
    x_next = rae.decode(z)
    if clamp:
        x_next = x_next.clamp(0.0, 1.0)
    return z, x_next


def main() -> None:
    p = argparse.ArgumentParser(description="Latent drift analysis for iterative RAE reconstruction.")
    p.add_argument("--config", required=True, help="Path to YAML config with stage_1 section.")
    p.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Input image path.")
    p.add_argument("--steps", type=int, default=30, help="Number of iterations.")
    p.add_argument("--device", help="cuda/cpu/cuda:1...")
    p.add_argument("--save_all", action="store_true", help="Save x_k images for every step.")
    p.add_argument("--outdir", type=Path, default=Path("latent_runs"), help="Output directory.")
    p.add_argument("--no_clamp", action="store_true", help="Do not clamp x to [0,1] each step.")
    args = p.parse_args()

    device = get_device(args.device)

    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    rae_config, *_ = parse_configs(args.config)
    if rae_config is None:
        raise ValueError(f"No stage_1 section found in config {args.config}")

    torch.set_grad_enabled(False)
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()

    x0 = load_image(args.image).to(device).clamp(0.0, 1.0)
    x = x0

    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.save_all:
        save_image(x0, args.outdir / "x_000.png")

    clamp_each = not args.no_clamp

    # compute z0 (encoder output on original x0, using the same encode pipeline)
    z0, x1 = step_T(rae, x0, clamp=clamp_each)
    z0f = flatten_latent(z0)

    # We will start logs from k=0 (x0,z0) and then k=1..steps
    print("k\tz_shape\t\t z_mse_to_z0\t z_cos_to_z0\t z_mse_to_prev\t z_norm\t\t z_mean\t\t z_std")
    print("-" * 120)

    z_prev = z0
    z_prevf = z0f

    # Log k=0
    z_norm0 = z0f.norm(dim=1).mean().item()
    z_mean0 = z0f.mean().item()
    z_std0 = z0f.std(unbiased=False).item()
    print(f"{0:03d}\t{tuple(z0.shape)}\t {0.0: .6e}\t {1.0: .6f}\t {'-':>12}\t {z_norm0: .6e}\t {z_mean0: .6e}\t {z_std0: .6e}")

    # Now iterate starting from x1 we already computed (optional: keep consistent with x_{k+1}=T(x_k))
    x = x1
    if args.save_all:
        save_image(x, args.outdir / "x_001.png")

    # Log k=1 based on z1 = E(x1)
    z1 = rae.encode(x)
    z1f = flatten_latent(z1)
    z_mse_z0 = mse(z1f, z0f)
    z_cos_z0 = cosine(z1f, z0f)
    z_mse_prev = mse(z1f, z_prevf)
    z_norm = z1f.norm(dim=1).mean().item()
    z_mean = z1f.mean().item()
    z_std = z1f.std(unbiased=False).item()
    print(f"{1:03d}\t{tuple(z1.shape)}\t {z_mse_z0: .6e}\t {z_cos_z0: .6f}\t {z_mse_prev: .6e}\t {z_norm: .6e}\t {z_mean: .6e}\t {z_std: .6e}")

    z_prevf = z1f

    # Continue k=2..steps
    for k in range(2, args.steps + 1):
        # one full step x <- T(x)
        _, x = step_T(rae, x, clamp=clamp_each)

        if args.save_all:
            save_image(x, args.outdir / f"x_{k:03d}.png")

        # latent at this x
        z = rae.encode(x)
        zf = flatten_latent(z)

        z_mse_z0 = mse(zf, z0f)
        z_cos_z0 = cosine(zf, z0f)
        z_mse_prev = mse(zf, z_prevf)
        z_norm = zf.norm(dim=1).mean().item()
        z_mean = zf.mean().item()
        z_std = zf.std(unbiased=False).item()

        print(f"{k:03d}\t{tuple(z.shape)}\t {z_mse_z0: .6e}\t {z_cos_z0: .6f}\t {z_mse_prev: .6e}\t {z_norm: .6e}\t {z_mean: .6e}\t {z_std: .6e}")

        z_prevf = zf

    print(f"\nSaved frames: {'yes' if args.save_all else 'no'}  |  outdir: {args.outdir.resolve()}")
    print("Interpretation tips:")
    print("- If z_cos_to_z0 stays ~1 and z_mse_to_z0 stays small: encoder representations are stable under drift.")
    print("- If z_cos_to_z0 drops steadily / z_mse_to_z0 grows: re-encoding changes the representation (latent drift).")
    print("- If z_norm/mean/std blow up: normalization mismatch or numerical issues.")


if __name__ == "__main__":
    main()
