#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python src/mask_code.py \
  --config configs/stage1/training/DINOv2-B_decXL.yaml \
  --ckpt pca_results/008-RAE/checkpoints/1280000.pt \
  --img imagenette2/train/n03394916/ILSVRC2012_val_00038137.JPEG \
  --code_id 680 \
  --fill_mode zero \
  --resize_mode letterbox \
  --save mask_680_zero.png
  
  
python src/mask_code.py \
  --config configs/stage1/training/DINOv2-B_decXL.yaml \
  --ckpt pca_results/008-RAE/checkpoints/1280000.pt \
  --img imagenette2/train/n02102040/n02102040_8126.JPEG \
  --code_id 680 \
  --fill_mode zero \
  --resize_mode crop \
  --save mask_680_zero_crop.png

imagenette2/train/n02102040/n02102040_8126.JPEG
imagenette_debug/train/n01440764/n01440764_18.JPEG
imagenette2/train/n03394916/ILSVRC2012_val_00038137.JPEG
"""

import argparse
import os
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf

from utils.model_utils import instantiate_from_config


# -------------------------
# Model load
# -------------------------
def load_model_from_ckpt(config_path: str, ckpt_path: str, device: torch.device):
    conf = OmegaConf.load(config_path)

    if "stage_1" in conf:
        model_conf = conf.stage_1
    elif "model" in conf:
        model_conf = conf.model
    else:
        model_conf = conf

    print(f"🎯 Target class: {model_conf.get('target', 'Not Found')}")
    model = instantiate_from_config(model_conf).to(device).eval()

    print(f"📂 Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # prefer EMA if exists
    if isinstance(checkpoint, dict) and "ema" in checkpoint:
        print("⚡ Using EMA weights")
        state_dict = checkpoint["ema"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # strip DDP prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"✅ Model loaded. Missing keys: {len(msg.missing_keys)} / Unexpected: {len(msg.unexpected_keys)}")
    return model


# -------------------------
# Image preprocess
# -------------------------
def _resize_squash(img: Image.Image, size: int) -> Image.Image:
    # no crop, but aspect ratio is distorted
    return img.resize((size, size), resample=Image.BICUBIC)

def _resize_center_crop(img: Image.Image, size: int) -> Image.Image:
    # imagenet-style: resize shorter side then center crop
    # use 1.14 factor as you used (approx 256 -> crop 224)
    resize_dim = int(round(size * 1.14))
    tfm = T.Compose([
        T.Resize(resize_dim, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
    ])
    return tfm(img)

def _resize_letterbox(img: Image.Image, size: int, pad_mode: str = "reflect") -> Image.Image:
    # keep aspect ratio, pad to square (no crop, no distortion)
    w, h = img.size
    scale = size / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img_r = img.resize((new_w, new_h), resample=Image.BICUBIC)

    # to tensor, pad, back to PIL
    x = T.ToTensor()(img_r)  # (3,new_h,new_w)
    pad_w = size - new_w
    pad_h = size - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    if pad_mode == "reflect":
        x = F.pad(x, (left, right, top, bottom), mode="reflect")
    elif pad_mode == "replicate":
        x = F.pad(x, (left, right, top, bottom), mode="replicate")
    elif pad_mode == "constant":
        x = F.pad(x, (left, right, top, bottom), mode="constant", value=0.0)
    else:
        raise ValueError(f"Unknown pad_mode: {pad_mode}")

    return T.ToPILImage()(x)

def load_image_tensor(path: str, input_size: int, device: torch.device, resize_mode: str) -> Tuple[Image.Image, torch.Tensor]:
    img = Image.open(path).convert("RGB")

    if resize_mode == "squash":
        img_p = _resize_squash(img, input_size)
    elif resize_mode == "crop":
        img_p = _resize_center_crop(img, input_size)
    elif resize_mode == "letterbox":
        img_p = _resize_letterbox(img, input_size, pad_mode="reflect")
    else:
        raise ValueError(f"Unknown resize_mode: {resize_mode}")

    x = T.ToTensor()(img_p).unsqueeze(0).to(device)  # (1,3,H,W) in [0,1]
    return img_p, x


# -------------------------
# Encode -> VQ indices/z_q
# -------------------------
@torch.no_grad()
def encode_to_zq_and_indices(model, x: torch.Tensor):
    """
    Uses the SAME path as RAE._vq_process, but exposes z_q and indices.
    Returns:
      z_q: (B, Cq, H, W)  (quantized in vq_embed_dim space)
      indices: (B, H, W) long
    """
    # model.encode handles: resize to encoder_input_size + normalize by encoder_mean/std
    z = model.encode(x)  # (B, latent_dim, H, W) if reshape_to_2d=True

    if hasattr(model, "pca_reweight") and model.pca_reweight is not None:
        z = model.pca_reweight(z)

    if not (hasattr(model, "vq_pre") and model.vq_pre is not None):
        raise RuntimeError("Model has no vq_pre. Is use_vq enabled in this checkpoint/config?")

    z_small = model.vq_pre(z)

    if hasattr(model, "vq_z_norm") and bool(model.vq_z_norm):
        denom = z_small.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
        z_small = z_small / denom

    out = model.vq_layer(z_small)

    if isinstance(out, (tuple, list)) and len(out) >= 3:
        z_q, vq_loss, indices = out[0], out[1], out[2]
    else:
        raise RuntimeError("vq_layer output doesn't include (z_q, loss, indices).")

    # normalize indices shape to (B,H,W)
    if indices.dim() == 2:
        # (B,N) -> (B,H,W)
        B, N = indices.shape
        H = W = int(round(N ** 0.5))
        indices = indices.view(B, H, W)

    return z_q, indices


# -------------------------
# Masking by code id
# -------------------------
def mask_code_in_zq(
    z_q: torch.Tensor,
    indices: torch.Tensor,
    code_id: int,
    fill_mode: str = "mean",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    z_q: (B,C,H,W)
    indices: (B,H,W)
    fill_mode:
      - mean: fill masked positions with per-sample spatial mean (B,C,1,1)
      - zero: fill with 0
      - global_mean: fill with global mean across batch+space (1,C,1,1) (sometimes more stable)
    Returns:
      z_q_masked, mask (B,1,H,W) float {0,1}
    """
    assert z_q.dim() == 4
    B, C, H, W = z_q.shape
    assert indices.shape == (B, H, W)

    mask = (indices == int(code_id)).float().unsqueeze(1)  # (B,1,H,W)

    if fill_mode == "mean":
        fill = z_q.mean(dim=(2, 3), keepdim=True)  # (B,C,1,1)
    elif fill_mode == "global_mean":
        fill = z_q.mean(dim=(0, 2, 3), keepdim=True)  # (1,C,1,1)
    elif fill_mode == "zero":
        fill = torch.zeros((1, C, 1, 1), device=z_q.device, dtype=z_q.dtype)
    else:
        raise ValueError(f"Unknown fill_mode: {fill_mode}")

    z_q_masked = z_q * (1.0 - mask) + fill * mask
    return z_q_masked, mask


# -------------------------
# Decode from z_q
# -------------------------
@torch.no_grad()
def decode_from_zq(model, z_q: torch.Tensor) -> torch.Tensor:
    """
    z_q (B,Cq,H,W) -> vq_post -> model.decode -> image tensor (B,3,Hout,Wout) in pixel space
    """
    if not (hasattr(model, "vq_post") and model.vq_post is not None):
        raise RuntimeError("Model has no vq_post. Is use_vq enabled?")

    z_out = model.vq_post(z_q)       # (B, latent_dim, H, W)
    x_rec = model.decode(z_out)      # (B,3,*,*) pixel space (already unnormalized in decode)
    x_rec = torch.clamp(x_rec, 0.0, 1.0)
    return x_rec


# -------------------------
# Visualization
# -------------------------
def to_np_img(t: torch.Tensor) -> np.ndarray:
    # (1,3,H,W) -> (H,W,3)
    return t.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()

def mask_to_overlay(mask_1chw: torch.Tensor, out_size: int) -> np.ndarray:
    """
    mask: (1,1,H,W) in {0,1}
    returns: (out_size,out_size) float
    """
    m = torch.nn.functional.interpolate(mask_1chw, size=(out_size, out_size), mode="nearest")
    return m.detach().cpu().squeeze().numpy()

def save_compare_figure(
    save_path: str,
    img_in: Image.Image,
    recon_norm: np.ndarray,
    recon_mask: np.ndarray,
    title_mask: str,
):
    panels = [
        ("Input (preprocessed)", np.array(img_in)),
        ("Recon (normal)", recon_norm),
        (title_mask, recon_mask),
    ]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, (title, im) in zip(axes, panels):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✨ Saved: {save_path}")

def save_overlay_figure(save_path: str, img_in: Image.Image, overlay: np.ndarray, code_id: int):
    img_np = np.array(img_in).astype(np.float32) / 255.0
    heat = overlay
    # simple red overlay
    out = img_np.copy()
    out[..., 0] = np.clip(out[..., 0] + 0.75 * heat, 0.0, 1.0)

    plt.figure(figsize=(6, 6))
    plt.title(f"Mask positions for code_id={code_id}")
    plt.imshow(out)
    plt.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"🔥 Saved overlay: {save_path}")


# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--img", required=True, type=str)

    p.add_argument("--code_id", default=680, type=int, help="VQ code index to mask (default: 680)")
    p.add_argument("--fill_mode", default="mean", choices=["mean", "zero", "global_mean"])

    p.add_argument("--input_size", default=224, type=int)
    p.add_argument("--resize_mode", default="squash", choices=["squash", "crop", "letterbox"],
                   help="squash: no crop but distorts aspect; crop: ImageNet-ish; letterbox: no crop no distortion (pads).")

    p.add_argument("--save", default="mask_code.png", type=str)
    p.add_argument("--save_overlay", default=None, type=str, help="Optional path to save overlay image.")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_from_ckpt(args.config, args.ckpt, device)

    img_pil, x = load_image_tensor(args.img, args.input_size, device, resize_mode=args.resize_mode)

    with torch.no_grad():
        z_q, indices = encode_to_zq_and_indices(model, x)
        # baseline recon
        recon_norm = decode_from_zq(model, z_q)

        # mask specific code id
        z_q_m, mask = mask_code_in_zq(z_q, indices, code_id=args.code_id, fill_mode=args.fill_mode)
        recon_mask = decode_from_zq(model, z_q_m)

    recon_norm_np = to_np_img(recon_norm)
    recon_mask_np = to_np_img(recon_mask)

    title_mask = f"Recon (mask code={args.code_id}, fill={args.fill_mode})"
    save_compare_figure(args.save, img_pil, recon_norm_np, recon_mask_np, title_mask)

    if args.save_overlay is not None:
        # mask is (B,1,H,W) where H=W=16 typically; upscale to input size for visualization
        overlay = mask_to_overlay(mask[:1], out_size=args.input_size)  # (input_size,input_size)
        save_overlay_figure(args.save_overlay, img_pil, overlay, args.code_id)


if __name__ == "__main__":
    main()
