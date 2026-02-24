#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Codebook Visualization Tool (patch samples + heatmap overlay)

What it does:
- Browse images under DATA_DIR (recursive)
- Run RAE encode -> (optional) pca_reweight -> vq_pre -> (optional) z_norm -> vq_layer
- Extract VQ indices (token grid, e.g., 16x16)
- Find top-K most frequent codes
- For each top code:
  (A) save patch samples (14x14) from where the code occurs
  (B) save overlay samples: highlight positions of that code on the resized input image
  (C) save aggregate overlay: summed heatmap across images

Notes:
- This script uses "squash" preprocessing: Resize((input_size, input_size)).
  That avoids cropping (no object cut), but distorts aspect ratio slightly.
"""

import os
import glob
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

from utils.model_utils import instantiate_from_config


# ==========================================
# 설정 (원하면 여기만 바꿔도 됨)
# ==========================================
CONFIG_PATH = "configs/stage1/training/DINOv2-B_decXL.yaml"
CKPT_PATH   = "pca_results/008-RAE/checkpoints/1280000.pt"

DATA_DIR    = "imagenette2/train/n02102040"
NUM_IMAGES  = 50          # 너무 많으면 오래 걸림
TOP_K_CODES = 5

INPUT_SIZE  = 224         # token grid가 16x16이면 patch=14
PATCH_SIZE  = 14          # DINOv2 patch=14 기준 (224/16=14)

# 샘플 저장 개수 제한
MAX_PATCH_SAMPLES_PER_CODE   = 12  # code별 patch 이미지 최대 저장 개수
MAX_OVERLAY_SAMPLES_PER_CODE = 6   # code별 overlay 예시 이미지 최대 저장 개수

# overlay 투명도
OVERLAY_ALPHA = 0.45
# ==========================================


# ------------------------------------------
# Model loading
# ------------------------------------------
def load_model(config_path, ckpt_path, device):
    conf = OmegaConf.load(config_path)
    model_conf = conf.stage_1 if "stage_1" in conf else (conf.model if "model" in conf else conf)
    model = instantiate_from_config(model_conf).to(device).eval()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["ema"] if isinstance(ckpt, dict) and "ema" in ckpt else (ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)

    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    print(f"✅ Loaded: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    return model


# ------------------------------------------
# Preprocess + patch extraction
# ------------------------------------------
def get_image_tensor_and_patches(img_path, input_size=224, patch_size=14):
    """
    Returns:
      x: (1,3,input_size,input_size) tensor (squash resize)
      img_np: (H,W,3) resized numpy image for visualization
      patches: list of patch numpy arrays in row-major order (len = grid*grid)
      grid_size: int (e.g., 16)
    """
    img = Image.open(img_path).convert("RGB")

    tfm = T.Compose([
        T.Resize((input_size, input_size)),  # squash (no crop)
        T.ToTensor(),
    ])
    x = tfm(img).unsqueeze(0)  # (1,3,H,W)

    img_np = np.array(img.resize((input_size, input_size), resample=Image.BICUBIC))

    grid_size = input_size // patch_size
    assert grid_size * patch_size == input_size, f"input_size={input_size} not divisible by patch_size={patch_size}"

    patches = []
    for i in range(grid_size):
        for j in range(grid_size):
            h0, w0 = i * patch_size, j * patch_size
            patch = img_np[h0:h0+patch_size, w0:w0+patch_size, :]
            patches.append(patch)

    return x, img_np, patches, grid_size


# ------------------------------------------
# VQ indices extraction (match your RAE path)
# ------------------------------------------
@torch.no_grad()
def extract_vq_indices(model, x: torch.Tensor):
    """
    Runs:
      z = model.encode(x)
      optional pca_reweight
      z = vq_pre(z)
      optional z_norm
      out = vq_layer(z) -> indices

    Returns:
      indices_2d: (H,W) long tensor on CPU
      H,W inferred
    """
    z = model.encode(x)  # (B,C,H,W) if reshape_to_2d=True

    if hasattr(model, "pca_reweight") and model.pca_reweight is not None:
        z = model.pca_reweight(z)

    if hasattr(model, "vq_pre") and model.vq_pre is not None:
        z = model.vq_pre(z)

    if hasattr(model, "vq_z_norm") and bool(model.vq_z_norm):
        denom = z.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
        z = z / denom

    out = model.vq_layer(z)

    indices = None
    if isinstance(out, (tuple, list)) and len(out) >= 3:
        indices = out[2]
    elif hasattr(model, "last_vq_indices") and model.last_vq_indices is not None:
        indices = model.last_vq_indices
    else:
        raise RuntimeError("Could not extract indices: vq_layer output has no indices and model.last_vq_indices is None.")

    # indices shape could be (B,H,W) or (B,N)
    if indices.dim() == 3:
        # (B,H,W)
        indices_2d = indices[0].detach().cpu().long()
        H, W = indices_2d.shape
        return indices_2d, H, W

    if indices.dim() == 2:
        # (B,N) assume square
        flat = indices[0].detach().cpu().long()
        N = flat.numel()
        H = W = int(round(N ** 0.5))
        if H * W != N:
            raise RuntimeError(f"indices length {N} is not a perfect square.")
        indices_2d = flat.view(H, W)
        return indices_2d, H, W

    raise RuntimeError(f"Unexpected indices dim: {indices.dim()} shape={tuple(indices.shape)}")


# ------------------------------------------
# Visualization helpers
# ------------------------------------------
def save_patch_strip(code: int, count: int, patches: list, out_path: str):
    if len(patches) == 0:
        return
    fig, axes = plt.subplots(1, len(patches), figsize=(2 * len(patches), 2))
    fig.suptitle(f"Code {code} | used {count} times | patch samples={len(patches)}", fontsize=14)
    if len(patches) == 1:
        axes = [axes]
    for ax, p in zip(axes, patches):
        ax.imshow(p)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ saved patches: {out_path}")


def overlay_mask_on_image(img_np: np.ndarray, mask_2d: np.ndarray, alpha: float = 0.45):
    """
    img_np: (H,W,3) uint8 or float
    mask_2d: (gridH,gridW) in {0,1} or counts
    Returns: overlay image as float in [0,1]
    """
    img = img_np.astype(np.float32) / 255.0 if img_np.dtype != np.float32 else np.clip(img_np, 0, 1)

    H, W, _ = img.shape
    mh, mw = mask_2d.shape

    # upsample mask to image size using nearest
    mask_t = torch.tensor(mask_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,mh,mw)
    mask_up = torch.nn.functional.interpolate(mask_t, size=(H, W), mode="nearest").squeeze().numpy()

    # normalize for visualization
    if mask_up.max() > 0:
        heat = mask_up / (mask_up.max() + 1e-8)
    else:
        heat = mask_up

    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(heat)[..., :3]  # drop alpha

    # blend
    out = (1 - alpha) * img + alpha * heat_rgb
    out = np.clip(out, 0.0, 1.0)
    return out


def save_overlay_grid(code: int, overlays: list, title: str, out_path: str):
    if len(overlays) == 0:
        return
    fig, axes = plt.subplots(1, len(overlays), figsize=(4 * len(overlays), 4))
    fig.suptitle(title, fontsize=14)
    if len(overlays) == 1:
        axes = [axes]
    for ax, im in zip(axes, overlays):
        ax.imshow(im)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ saved overlay: {out_path}")


# ------------------------------------------
# Main
# ------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CONFIG_PATH, CKPT_PATH, device)

    img_files = glob.glob(os.path.join(DATA_DIR, "**", "*.JPEG"), recursive=True)
    img_files = img_files[:NUM_IMAGES]
    print(f"🔍 Found {len(img_files)} images (using first {len(img_files)}).")

    # --------------------------------------------------
    # Pass 1: count codes (and maybe store a few patches, but topK unknown yet)
    # --------------------------------------------------
    code_counts = defaultdict(int)

    # We'll store per-image indices for second pass to avoid recompute? (optional)
    # For simplicity & robustness, we'll do second pass inference again (NUM_IMAGES small).
    for img_path in tqdm(img_files, desc="Pass1 counting"):
        try:
            x, _, _, _ = get_image_tensor_and_patches(img_path, INPUT_SIZE, PATCH_SIZE)
            x = x.to(device)

            indices_2d, H, W = extract_vq_indices(model, x)
            flat = indices_2d.view(-1).numpy()
            for c in flat:
                code_counts[int(c)] += 1
        except Exception as e:
            print(f"[Pass1] Error {img_path}: {e}")
            continue

    if len(code_counts) == 0:
        print("❌ No codes counted. Check your model/VQ path.")
        return

    sorted_codes = sorted(code_counts.items(), key=lambda x: x[1], reverse=True)[:TOP_K_CODES]
    top_codes = [c for c, _ in sorted_codes]
    print("\n📊 Top codes:")
    for rank, (c, cnt) in enumerate(sorted_codes):
        print(f"  #{rank+1}: code={c} count={cnt}")

    # --------------------------------------------------
    # Pass 2: collect patch samples + overlay samples + aggregate heatmaps for top codes
    # --------------------------------------------------
    code_to_patches = {c: [] for c in top_codes}
    code_to_overlay_samples = {c: [] for c in top_codes}
    code_to_agg_heat = {c: None for c in top_codes}
    code_to_agg_seen = {c: 0 for c in top_codes}

    for img_path in tqdm(img_files, desc="Pass2 collecting"):
        try:
            x, img_np, patches_np, grid_size = get_image_tensor_and_patches(img_path, INPUT_SIZE, PATCH_SIZE)
            x = x.to(device)

            indices_2d, H, W = extract_vq_indices(model, x)
            # safety: ensure token grid matches patch grid
            if H != grid_size or W != grid_size:
                # not fatal, but mapping patches->tokens may be wrong
                print(f"[Warn] grid mismatch: tokens {H}x{W} vs patches {grid_size}x{grid_size} for {img_path}")

            idx_np_2d = indices_2d.numpy()

            for c in top_codes:
                mask = (idx_np_2d == c).astype(np.float32)  # (H,W), 0/1

                # aggregate heat
                if code_to_agg_heat[c] is None:
                    code_to_agg_heat[c] = mask.copy()
                else:
                    code_to_agg_heat[c] += mask
                code_to_agg_seen[c] += 1

                # patch samples: collect patches where code appears (up to MAX per code)
                if len(code_to_patches[c]) < MAX_PATCH_SAMPLES_PER_CODE:
                    # gather positions (row-major)
                    flat = idx_np_2d.reshape(-1)
                    positions = np.where(flat == c)[0]
                    # take a few positions from this image
                    for pos in positions[: max(0, MAX_PATCH_SAMPLES_PER_CODE - len(code_to_patches[c]))]:
                        if pos < len(patches_np):
                            code_to_patches[c].append(patches_np[pos])

                # overlay samples: save a few example overlays per code
                if len(code_to_overlay_samples[c]) < MAX_OVERLAY_SAMPLES_PER_CODE:
                    # only add overlay if code actually appears in this image
                    if mask.sum() > 0:
                        overlay = overlay_mask_on_image(img_np, mask, alpha=OVERLAY_ALPHA)
                        code_to_overlay_samples[c].append(overlay)

        except Exception as e:
            print(f"[Pass2] Error {img_path}: {e}")
            continue

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------
    os.makedirs("code_vis", exist_ok=True)

    for rank, (c, cnt) in enumerate(sorted_codes):
        # (A) patch strip
        patch_out = os.path.join("code_vis", f"vis_code_{c}_rank{rank}_patches.png")
        save_patch_strip(c, cnt, code_to_patches[c], patch_out)

        # (B) overlay samples grid
        ov_out = os.path.join("code_vis", f"vis_code_{c}_rank{rank}_overlay_samples.png")
        title = f"Code {c} overlay samples (alpha={OVERLAY_ALPHA})"
        save_overlay_grid(c, code_to_overlay_samples[c], title, ov_out)

        # (C) aggregate overlay
        agg = code_to_agg_heat[c]
        if agg is None:
            continue
        # normalize by seen images (optional)
        agg_norm = agg / max(1, code_to_agg_seen[c])

        # create overlay on a blank gray background (so you can see hotspots clearly)
        bg = np.ones((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32) * 0.5
        agg_overlay = overlay_mask_on_image((bg * 255).astype(np.uint8), agg_norm, alpha=0.75)

        agg_out = os.path.join("code_vis", f"vis_code_{c}_rank{rank}_aggregate.png")
        plt.figure(figsize=(5, 5))
        plt.title(f"Code {c} aggregate heatmap (avg over {code_to_agg_seen[c]} imgs)")
        plt.imshow(agg_overlay)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(agg_out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   ✅ saved aggregate: {agg_out}")

    print("\n🎉 Done. Outputs are saved under ./code_vis/")
    print("   - *_patches.png : patch samples where the code appears")
    print("   - *_overlay_samples.png : example overlays on real images")
    print("   - *_aggregate.png : average heatmap across images")


if __name__ == "__main__":
    main()
