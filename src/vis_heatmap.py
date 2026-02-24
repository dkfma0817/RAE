#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize where specific VQ code indices appear (heatmap overlay).

- Load a single image
- Run RAE encode -> pca_reweight(optional) -> vq_pre -> z_norm(optional) -> vq_layer
- Get indices (B,H,W) usually (1,16,16)
- Build a mask for selected codes (can be multiple codes)
- Upsample mask and overlay on:
  - input(preprocessed) : usually 224x224
  - recon(normal)       : could be 224x224 OR 256x256 depending on decoder_patch_size
  - both

This fixes shape mismatch by generating separate heatmaps for input_size and recon_size.

Usage:
python src/vis_heatmap.py \
  --config configs/stage1/training/DINOv2-B_decXL.yaml \
  --ckpt pca_results/008-RAE/checkpoints/1280000.pt \
  --img imagenette_debug/train/n01440764/n01440764_18.JPEG \
  --codes 680 \
  --overlay_on both \
  --resize_mode squash \
  --save_prefix out/code680

"""

import argparse
import os
from typing import List, Tuple

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

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "ema" in ckpt:
        print("⚡ Using EMA weights")
        sd = ckpt["ema"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt

    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    print(f"✅ Loaded. Missing={len(msg.missing_keys)} Unexpected={len(msg.unexpected_keys)}")
    return model


# -------------------------
# Preprocess
# -------------------------
def _resize_squash(img: Image.Image, size: int) -> Image.Image:
    # no crop, but distorts aspect ratio
    return img.resize((size, size), resample=Image.BICUBIC)

def _resize_center_crop(img: Image.Image, size: int) -> Image.Image:
    # imagenet-ish: enlarge then crop
    resize_dim = int(round(size * 1.14))
    tfm = T.Compose([
        T.Resize(resize_dim, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
    ])
    return tfm(img)

def _resize_letterbox(img: Image.Image, size: int, pad_mode: str = "reflect") -> Image.Image:
    # keep aspect, pad to square
    w, h = img.size
    scale = size / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img_r = img.resize((new_w, new_h), resample=Image.BICUBIC)

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

def load_image_tensor(path: str, input_size: int, device: torch.device, resize_mode: str):
    img = Image.open(path).convert("RGB")
    if resize_mode == "squash":
        img_p = _resize_squash(img, input_size)
    elif resize_mode == "crop":
        img_p = _resize_center_crop(img, input_size)
    elif resize_mode == "letterbox":
        img_p = _resize_letterbox(img, input_size, pad_mode="reflect")
    else:
        raise ValueError(f"Unknown resize_mode: {resize_mode}")

    x = T.ToTensor()(img_p).unsqueeze(0).to(device)  # (1,3,H,W)
    return img_p, x


# -------------------------
# VQ indices + recon
# -------------------------
@torch.no_grad()
def get_indices_and_recon(model, x: torch.Tensor):
    """
    Returns:
      indices: (1,H,W) long on cpu
      recon: (1,3,Hout,Wout) float in [0,1]
    """
    # model.encode includes: resize-to-encoder_input_size + normalize by encoder_mean/std
    z = model.encode(x)  # (1,C,H,W)

    if hasattr(model, "pca_reweight") and model.pca_reweight is not None:
        z = model.pca_reweight(z)

    if not (hasattr(model, "vq_pre") and model.vq_pre is not None):
        raise RuntimeError("No vq_pre found. Is this a VQ-enabled checkpoint/config?")

    z_small = model.vq_pre(z)

    if hasattr(model, "vq_z_norm") and bool(model.vq_z_norm):
        denom = z_small.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
        z_small = z_small / denom

    out = model.vq_layer(z_small)
    if not (isinstance(out, (tuple, list)) and len(out) >= 3):
        raise RuntimeError("vq_layer output does not include indices.")
    z_q, vq_loss, indices = out[0], out[1], out[2]

    # normalize indices to (1,H,W)
    if indices.dim() == 2:
        # (1,N) -> (1,H,W)
        N = indices.shape[1]
        H = W = int(round(N ** 0.5))
        indices = indices.view(1, H, W)
    elif indices.dim() == 3:
        pass
    else:
        raise RuntimeError(f"Unexpected indices shape: {tuple(indices.shape)}")

    indices_cpu = indices.detach().cpu().long()

    # baseline recon (normal): decode quantized z_q
    if not (hasattr(model, "vq_post") and model.vq_post is not None):
        raise RuntimeError("No vq_post found. Is this a VQ-enabled checkpoint/config?")
    z_out = model.vq_post(z_q)
    recon = model.decode(z_out)
    recon = torch.clamp(recon, 0.0, 1.0)

    return indices_cpu, recon


# -------------------------
# Heatmap + overlay
# -------------------------
def parse_codes(codes_str: str) -> List[int]:
    codes = [int(x.strip()) for x in codes_str.split(",") if x.strip()]
    if len(codes) == 0:
        raise ValueError("No codes provided. Example: --codes 680 or --codes 680,681")
    return codes

def codes_to_mask(indices_1hw: torch.Tensor, codes: List[int]) -> np.ndarray:
    """
    indices_1hw: (1,H,W) long
    returns mask (H,W) float: 1 where code matches any in codes else 0
    """
    idx = indices_1hw[0]  # (H,W)
    m = torch.zeros_like(idx, dtype=torch.float32)
    for c in codes:
        m = m + (idx == int(c)).float()
    m = torch.clamp(m, 0, 1)
    return m.numpy()

def upsample_mask(mask_hw: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    mask_hw: (H,W) float
    returns: (out_h,out_w) float
    """
    t = torch.tensor(mask_hw, dtype=torch.float32)[None, None, ...]  # (1,1,H,W)
    up = F.interpolate(t, size=(out_h, out_w), mode="nearest")[0, 0].numpy()
    return up

def normalize_heat(heat: np.ndarray) -> np.ndarray:
    m = float(heat.max())
    if m <= 0:
        return heat
    return heat / (m + 1e-8)

def overlay_on_image(img_rgb01: np.ndarray, heat01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    img_rgb01: (H,W,3) in [0,1]
    heat01: (H,W) in [0,1]
    """
    heat01 = np.clip(heat01, 0, 1)
    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(heat01)[..., :3]  # (H,W,3)
    out = (1 - alpha) * img_rgb01 + alpha * heat_rgb
    return np.clip(out, 0, 1)

def save_image(path: str, img01: np.ndarray, title: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.imshow(img01)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✨ Saved: {path}")


# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--img", required=True, type=str)

    p.add_argument("--codes", required=True, type=str,
                   help="Comma-separated code ids. e.g. '680' or '680,123,9'")
    p.add_argument("--overlay_on", default="both", choices=["input", "recon", "both"])
    p.add_argument("--alpha", default=0.45, type=float)

    p.add_argument("--input_size", default=224, type=int)
    p.add_argument("--resize_mode", default="squash", choices=["squash", "crop", "letterbox"])

    p.add_argument("--save_prefix", default="out/code_heatmap", type=str,
                   help="Prefix for outputs. Saves *_input.png, *_recon.png, *_heat_input.png, *_heat_recon.png")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    codes = parse_codes(args.codes)

    model = load_model_from_ckpt(args.config, args.ckpt, device)
    img_in_pil, x = load_image_tensor(args.img, args.input_size, device, args.resize_mode)

    indices_1hw, recon = get_indices_and_recon(model, x)

    # token-grid mask (e.g., 16x16)
    mask_hw = codes_to_mask(indices_1hw, codes)
    Ht, Wt = mask_hw.shape
    hits = int(mask_hw.sum())
    print(f"📌 Token grid: {Ht}x{Wt} | hits={hits} tokens ({hits/(Ht*Wt)*100:.1f}%) | codes={codes}")

    # heat for input (usually 224x224)
    heat_input = upsample_mask(mask_hw, out_h=args.input_size, out_w=args.input_size)
    heat_input = normalize_heat(heat_input)

    # heat for recon (could be 224 or 256 etc.)
    recon_h = int(recon.shape[2])
    recon_w = int(recon.shape[3])
    heat_recon = upsample_mask(mask_hw, out_h=recon_h, out_w=recon_w)
    heat_recon = normalize_heat(heat_recon)

    # ---------- input overlay ----------
    if args.overlay_on in ["input", "both"]:
        img_in = np.array(img_in_pil).astype(np.float32) / 255.0  # (224,224,3)
        over_in = overlay_on_image(img_in, heat_input, alpha=args.alpha)
        save_image(f"{args.save_prefix}_input.png", over_in, title=f"Codes {codes} on INPUT")
        # heat only (optional)
        heat_in_rgb = plt.get_cmap("jet")(heat_input)[..., :3]
        save_image(f"{args.save_prefix}_heat_input.png", heat_in_rgb, title=f"Heat (input) codes {codes}")

    # ---------- recon overlay ----------
    if args.overlay_on in ["recon", "both"]:
        recon_np = recon.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        recon_np = np.clip(recon_np, 0, 1)
        over_rec = overlay_on_image(recon_np, heat_recon, alpha=args.alpha)
        save_image(f"{args.save_prefix}_recon.png", over_rec, title=f"Codes {codes} on RECON ({recon_h}x{recon_w})")
        # heat only (optional)
        heat_rec_rgb = plt.get_cmap("jet")(heat_recon)[..., :3]
        save_image(f"{args.save_prefix}_heat_recon.png", heat_rec_rgb, title=f"Heat (recon) codes {codes}")

if __name__ == "__main__":
    main()
