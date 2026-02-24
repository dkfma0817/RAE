#!/usr/bin/env python3
"""
Compute idempotence gap for iterative RAE reconstructions.

Given saved images x_k, computes:
  gap(k) = MSE( T(x_k), x_k )   where T(x)=D(E(x))

Also prints min/max to sanity-check clamp/range issues.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms

from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs
from stage1 import RAE


def get_device(explicit: str | None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_png(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    x = transforms.ToTensor()(img).unsqueeze(0)  # (1,C,H,W) in [0,1] if png
    return x


@torch.no_grad()
def T(rae: RAE, x: torch.Tensor, clamp: bool = True) -> torch.Tensor:
    z = rae.encode(x)
    y = rae.decode(z)
    if clamp:
        y = y.clamp(0.0, 1.0)
    return y


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).pow(2).mean().item()


def list_images(run_dir: Path, pattern: str) -> List[Path]:
    files = sorted(run_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern} under {run_dir}")
    return files


def main() -> None:
    p = argparse.ArgumentParser(description="Compute idempotence gap for saved recon iterations.")
    p.add_argument("--config", required=True, help="Stage-1 YAML config (with stage_1 section).")
    p.add_argument("--run_dir", required=True, type=Path, help="Directory containing x_*.png files.")
    p.add_argument("--pattern", default="x_*.png", help="Glob pattern for saved frames (default: x_*.png).")
    p.add_argument("--device", help="Torch device (cuda/cpu/cuda:1...).")
    p.add_argument("--no_clamp", action="store_true", help="Do not clamp T(x) output (for debugging).")
    args = p.parse_args()

    device = get_device(args.device)

    rae_config, *_ = parse_configs(args.config)
    if rae_config is None:
        raise ValueError(f"No stage_1 section found in config {args.config}")

    torch.set_grad_enabled(False)
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()

    frames = list_images(args.run_dir, args.pattern)

    print(f"Found {len(frames)} frames under {args.run_dir.resolve()}")
    print("k\tfile\t\t\tgap_mse\t\t x_min..x_max\t\t T(x)_min..T(x)_max")
    print("-" * 100)

    clamp = not args.no_clamp

    for k, fp in enumerate(frames):
        x = load_png(fp).to(device)

        # range checks
        x_min, x_max = x.min().item(), x.max().item()

        y = T(rae, x, clamp=clamp)
        y_min, y_max = y.min().item(), y.max().item()

        gap = mse(y, x)

        print(f"{k:03d}\t{fp.name:16s}\t{gap: .6e}\t [{x_min:.3f},{x_max:.3f}]\t\t [{y_min:.3f},{y_max:.3f}]")

    print("\nInterpretation:")
    print("- gap ~ 0  : near-idempotent (projection-like)")
    print("- gap stays high: repeated application keeps changing output (drift)")
    print("- gap decreases with k: may converge to a fixed point / attractor")


if __name__ == "__main__":
    main()