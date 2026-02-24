#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from diffusers import AutoencoderKL


def load_img(p: Path) -> torch.Tensor:
    img = Image.open(p).convert("RGB")
    x = transforms.ToTensor()(img).unsqueeze(0)  # (1,3,H,W) in [0,1]
    return x


@torch.no_grad()
def encode_decode(vae: AutoencoderKL, x: torch.Tensor, deterministic: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    # diffusers VAE expects pixel_values in [-1, 1]
    x_in = x * 2.0 - 1.0
    x_in = x_in.to(device=next(vae.parameters()).device, dtype=next(vae.parameters()).dtype)


    enc = vae.encode(x_in)  # returns AutoencoderKLOutput(latent_dist=DiagonalGaussianDistribution) :contentReference[oaicite:4]{index=4}
    if deterministic:
        z = enc.latent_dist.mean  # deterministic
        # (mean 대신 mode가 있으면 mode 써도 됨. 버전에 따라 .mode() 제공 여부가 다름)
    else:
        z = enc.latent_dist.sample()

    # SD 계열은 보통 scaling_factor를 곱해서 latent space 스케일 맞춤
    if hasattr(vae.config, "scaling_factor") and vae.config.scaling_factor is not None:
        z_scaled = z * vae.config.scaling_factor
    else:
        z_scaled = z

    # decode: decode expects scaled latents in many pipelines, so reverse scaling
    if hasattr(vae.config, "scaling_factor") and vae.config.scaling_factor is not None:
        z_in = z_scaled / vae.config.scaling_factor
    else:
        z_in = z_scaled

    dec = vae.decode(z_in).sample
    dec = dec.float()
    x_out = (dec + 1.0) / 2.0


    return z, x_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default="stabilityai/sdxl-vae", help="HF repo id for SDXL VAE")
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--outdir", type=Path, default=Path("vae_repeat"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--deterministic", action="store_true", help="Use mean latents (recommended)")
    ap.add_argument("--save_all", action="store_true")
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=dtype).to(args.device)
    vae.eval()

    x0 = load_img(args.image).to(args.device, dtype=torch.float32)  # keep pixels float32
    x = x0

    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.save_all:
        save_image(x, args.outdir / "x_000.png")

    z0, _ = encode_decode(vae, x0, deterministic=args.deterministic)
    z0f = z0.reshape(z0.shape[0], -1)

    print("k\tlatent_shape\tmse_to_z0\tcos_to_z0\tmse_to_prev")
    print("-" * 80)

    z_prevf = z0f
    print(f"{0:03d}\t{tuple(z0.shape)}\t{0.0:.6e}\t{1.0:.6f}\t-")

    for k in range(1, args.steps + 1):
        z, x = encode_decode(vae, x, deterministic=args.deterministic)
        if args.save_all:
            save_image(x, args.outdir / f"x_{k:03d}.png")

        zf = z.reshape(z.shape[0], -1)
        mse_z0 = (zf - z0f).pow(2).mean().item()
        cos = torch.nn.functional.cosine_similarity(zf, z0f, dim=1).mean().item()
        mse_prev = (zf - z_prevf).pow(2).mean().item()

        print(f"{k:03d}\t{tuple(z.shape)}\t{mse_z0:.6e}\t{cos:.6f}\t{mse_prev:.6e}")
        z_prevf = zf

    print(f"done. outdir={args.outdir.resolve()}")


if __name__ == "__main__":
    main()
