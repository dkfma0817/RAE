#!/usr/bin/env python3
"""
Frequency decomposition analysis for iterative RAE recon outputs.

Given saved frames x_k, compute vs x_0:
- full_mse(k)  = MSE(x_k, x_0)
- low_mse(k)   = MSE(G(x_k), G(x_0))      (Gaussian blur low-frequency)
- high_mse(k)  = MSE((x_k-G(x_k)), (x_0-G(x_0)))  (high-frequency residual)
- mean_rgb_drift(k) = ||meanRGB(x_k) - meanRGB(x_0)||_2

No model needed: purely image-space analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def load_png(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    x = transforms.ToTensor()(img).unsqueeze(0)  # (1,3,H,W) in [0,1]
    return x


def list_images(run_dir: Path, pattern: str) -> List[Path]:
    files = sorted(run_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern} under {run_dir}")
    return files


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).pow(2).mean().item()


def gaussian_kernel2d(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    # 1D coords centered at 0
    k = kernel_size
    coords = torch.arange(k, device=device) - (k - 1) / 2
    g1 = torch.exp(-(coords**2) / (2 * sigma**2))
    g1 = g1 / g1.sum()
    g2 = torch.outer(g1, g1)  # (k,k)
    return g2


def gaussian_blur(x: torch.Tensor, kernel_size: int = 21, sigma: float = 5.0) -> torch.Tensor:
    """
    x: (B,C,H,W)
    depthwise conv with gaussian kernel per channel
    """
    assert kernel_size % 2 == 1, "kernel_size should be odd"
    device = x.device
    B, C, H, W = x.shape
    k2 = gaussian_kernel2d(kernel_size, sigma, device=device)  # (k,k)
    kernel = k2.view(1, 1, kernel_size, kernel_size).repeat(C, 1, 1, 1)  # (C,1,k,k)
    pad = kernel_size // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    y = F.conv2d(x_pad, kernel, groups=C)
    return y


def mean_rgb(x: torch.Tensor) -> torch.Tensor:
    # x: (1,3,H,W) -> (3,)
    return x.mean(dim=(0, 2, 3)).squeeze(0)


def main() -> None:
    p = argparse.ArgumentParser(description="Low/High frequency error decomposition for recon iterations.")
    p.add_argument("--run_dir", required=True, type=Path, help="Directory containing x_*.png frames.")
    p.add_argument("--pattern", default="x_*.png", help="Glob pattern (default: x_*.png).")
    p.add_argument("--kernel", type=int, default=21, help="Gaussian blur kernel size (odd). default=21")
    p.add_argument("--sigma", type=float, default=5.0, help="Gaussian blur sigma. default=5.0")
    p.add_argument("--device", default="cpu", help="cpu or cuda (optional). default=cpu")
    args = p.parse_args()

    device = torch.device(args.device)

    frames = list_images(args.run_dir, args.pattern)
    x0 = load_png(frames[0]).to(device)

    low0 = gaussian_blur(x0, kernel_size=args.kernel, sigma=args.sigma)
    high0 = x0 - low0
    rgb0 = mean_rgb(x0)

    print(f"Found {len(frames)} frames under {args.run_dir.resolve()}")
    print(f"Using Gaussian blur: kernel={args.kernel}, sigma={args.sigma}")
    print("k\tfile\t\t full_mse\t low_mse\t high_mse\t mean_rgb_drift(L2)")
    print("-" * 110)

    for k, fp in enumerate(frames):
        x = load_png(fp).to(device)

        low = gaussian_blur(x, kernel_size=args.kernel, sigma=args.sigma)
        high = x - low

        full = mse(x, x0)
        low_m = mse(low, low0)
        high_m = mse(high, high0)

        rgb = mean_rgb(x)
        rgb_drift = torch.norm(rgb - rgb0, p=2).item()

        print(f"{k:03d}\t{fp.name:16s}\t{full: .6e}\t{low_m: .6e}\t{high_m: .6e}\t{rgb_drift: .6e}")

    print("\nNotes:")
    print("- If high_mse grows faster: texture/detail (high-frequency) changes dominate.")
    print("- If low_mse or mean_rgb_drift grows: color/illumination/structure drift dominates.")
    print("- Try a couple (kernel,sigma) settings to test robustness (e.g., sigma=3,5,8).")


if __name__ == "__main__":
    main()
