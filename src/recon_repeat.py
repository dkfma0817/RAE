#!/usr/bin/env python3
"""
Run iterative stage-1 RAE reconstructions from a config file.
x_{k+1} = D(E(x_k))
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Optional

import torch
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
    tensor = transforms.ToTensor()(image).unsqueeze(0)  # (1, C, H, W)
    return tensor


def reconstruct(rae: RAE, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        latent = rae.encode(image)
        recon = rae.decode(latent)
    return latent, recon


@torch.no_grad()
def iterate_recon(
    rae: RAE,
    x0: torch.Tensor,
    steps: int,
    clamp_each_step: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (x_last, latent_last).
    """
    x = x0
    latent = None
    for _ in range(steps):
        latent, x = reconstruct(rae, x)
        if clamp_each_step:
            x = x.clamp(0.0, 1.0)
    return x, latent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Iteratively reconstruct an input image using a Stage-1 RAE loaded from config."
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config with a stage_1 section.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE,
                        help=f"Input image to reconstruct (default: {DEFAULT_IMAGE}).")
    parser.add_argument("--outdir", type=Path, default=Path("recon_runs"),
                        help="Directory to save outputs (default: recon_runs).")
    parser.add_argument("--steps", type=int, default=1,
                        help="Number of reconstruction iterations (default: 1).")
    parser.add_argument("--save_all", action="store_true",
                        help="If set, save every intermediate x_k as PNG.")
    parser.add_argument("--device", help="Torch device to use (e.g. cuda, cuda:1, cpu). Auto-detect if omitted.")
    parser.add_argument("--no_clamp_each_step", action="store_true",
                        help="If set, do NOT clamp after each step (not recommended).")
    args = parser.parse_args()

    device = get_device(args.device)

    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    rae_config, *_ = parse_configs(args.config)
    if rae_config is None:
        raise ValueError(
            f"No stage_1 section found in config {args.config}. "
            "Please supply a config with a stage_1 target."
        )

    torch.set_grad_enabled(False)
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()

    x0 = load_image(args.image).to(device).clamp(0.0, 1.0)

    args.outdir.mkdir(parents=True, exist_ok=True)
    # Save original for reference
    save_image(x0, args.outdir / "x_000.png")

    x = x0
    latent = None
    clamp_each_step = not args.no_clamp_each_step

    for k in range(1, args.steps + 1):
        latent, x = reconstruct(rae, x)
        if clamp_each_step:
            x = x.clamp(0.0, 1.0)

        if args.save_all or k == args.steps:
            save_image(x, args.outdir / f"x_{k:03d}.png")

        # Quick scalar diagnostics (not fancy, but useful)
        l2_prev = (x - (x0 if k == 1 else prev_x)).pow(2).mean().item() if k > 1 else float("nan")
        l2_to_x0 = (x - x0).pow(2).mean().item()
        print(f"[step {k:03d}] latent={tuple(latent.shape)}  mse_to_x0={l2_to_x0:.6e}"
              + (f"  mse_to_prev={l2_prev:.6e}" if k > 1 else ""))

        prev_x = x

    print(f"Saved outputs to {args.outdir.resolve()}")
    print(f"Input shape: {tuple(x0.shape)}, final latent shape: {tuple(latent.shape)}, final recon shape: {tuple(x.shape)}")


if __name__ == "__main__":
    main()
