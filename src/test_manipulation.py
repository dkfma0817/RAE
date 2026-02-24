"""
Usage examples

복원 
python src/test_manipulation.py \
  --config configs/stage1/training/DINOv2-B_decXL.yaml \
  --ckpt pca_results/008-RAE/checkpoints/1280000.pt \
  --img_a imagenette_debug/train/n01440764/n01440764_96.JPEG \
  --task recon \
  --save recon.png

마스킹 
python src/test_manipulation.py \
  --config configs/stage1/training/DINOv2-B_decXL.yaml \
  --ckpt pca_results/008-RAE/checkpoints/1280000.pt \
  --img_a dog.png \
  --task mask \
  --extent 4 \
  --save mask.png
  
imagenette_debug/train/n01440764/n01440764_37.JPEG
imagenette_debug/train/n01440764/n01440764_96.JPEG

2) Swap center region between two images
  python src/test_manipulation.py \
  --config configs/stage1/training/DINOv2-B_decXL.yaml \
  --ckpt pca_results/008-RAE/checkpoints/1280000.pt \
  --img_a dog.png \
  --img_b imagenette_debug/train/n01440764/n01440764_96.JPEG \
  --task swap \
  --extent 4 \
  --save out_swap.png


"""

import argparse
import os
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf

from utils.model_utils import instantiate_from_config


# =========================================================
# Utilities
# =========================================================

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

    # EMA 우선
    if isinstance(checkpoint, dict) and "ema" in checkpoint:
        print("⚡ Loading EMA weights ...")
        state_dict = checkpoint["ema"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # DDP 'module.' 제거
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"✅ Model loaded. Missing keys: {len(msg.missing_keys)} / Unexpected: {len(msg.unexpected_keys)}")
    if len(msg.missing_keys) > 0:
        print("  - missing (first 20):", msg.missing_keys[:20])
    if len(msg.unexpected_keys) > 0:
        print("  - unexpected (first 20):", msg.unexpected_keys[:20])

    return model


def load_image_tensor_center_crop(path: str, input_size: int, device: torch.device) -> Tuple[Image.Image, torch.Tensor]:
    """
    비율 유지 + Resize 조금 크게 + CenterCrop (Script B 스타일)
    주의: model.encode()가 내부에서 다시 224로 interpolate 하긴 함.
    """
    img = Image.open(path).convert("RGB")

    resize_dim = int(round(input_size * 1.14))
    tfm = transforms.Compose([
        transforms.Resize(resize_dim, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ])
    x = tfm(img).unsqueeze(0).to(device)

    vis_tfm = transforms.Compose([
        transforms.Resize(resize_dim, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
    ])
    img_cropped = vis_tfm(img)

    return img_cropped, x


def to_np_img(t: torch.Tensor) -> np.ndarray:
    """(B,3,H,W) -> (H,W,3) in [0,1]"""
    x = t.detach().float().cpu().squeeze(0).permute(1, 2, 0).numpy()
    return np.clip(x, 0.0, 1.0)


def save_grid(save_path: str, panels: List[Tuple[str, np.ndarray]], figsize: Tuple[int, int]):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    print(f"✨ Saved: {save_path}")
    plt.show()


# =========================================================
# RAE-exact VQ pipeline (match rae.py)
# =========================================================

@torch.no_grad()
def encode_to_zq_exact(model, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    rae.py 기준 '정확한' z_q 추출.

    Returns:
      z_q: (B, vq_embed_dim, H, W) e.g. (B,256,16,16)
      indices: (B, H, W) or similar (depends on VectorQuantizer)
      info: dict
    """
    if not getattr(model, "use_vq", False):
        raise RuntimeError("This script expects a VQ-enabled RAE (model.use_vq=True).")

    # 1) encode(): includes (resize->224, normalize by encoder mean/std, reshape_to_2d, optional latent stats)
    z = model.encode(x)  # (B, latent_dim, 16, 16)

    # 2) PCA reweight BEFORE vq_pre (rae.py _vq_process)
    if getattr(model, "pca_reweight", None) is not None:
        z_rw = model.pca_reweight(z)
    else:
        z_rw = z

    # 3) vq_pre
    z_small = model.vq_pre(z_rw)  # (B, vq_embed_dim, 16, 16)

    # 4) optional z-norm
    if getattr(model, "vq_z_norm", False):
        denom = z_small.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
        z_small_n = z_small / denom
    else:
        z_small_n = z_small

    # 5) VQ layer
    out = model.vq_layer(z_small_n)

    # 가장 흔한: (z_q, vq_loss, indices)
    if isinstance(out, (tuple, list)) and len(out) >= 3:
        z_q, vq_loss, indices = out[0], out[1], out[2]
    else:
        # 혹시 구현이 다르면 여기서 터짐
        raise RuntimeError(f"Unexpected vq_layer output type/len: {type(out)} / {getattr(out, '__len__', lambda: 'NA')()}")

    model.last_vq_indices = indices

    info = {
        "z_shape": tuple(z.shape),
        "z_rw_shape": tuple(z_rw.shape),
        "z_small_shape": tuple(z_small.shape),
        "z_small_n_shape": tuple(z_small_n.shape),
        "z_q_shape": tuple(z_q.shape),
        "vq_loss": float(vq_loss.detach().cpu().item()) if torch.is_tensor(vq_loss) else None,
        "indices_shape": tuple(indices.shape) if torch.is_tensor(indices) else None,
    }
    return z_q, indices, info


@torch.no_grad()
def decode_from_zq_exact(model, z_q: torch.Tensor) -> torch.Tensor:
    """
    rae.py 기준 '정확한' 복원:
      z_q -> vq_post -> model.decode()
    """
    if not getattr(model, "use_vq", False):
        raise RuntimeError("This script expects a VQ-enabled RAE (model.use_vq=True).")

    z_latent = model.vq_post(z_q)      # (B, latent_dim, 16, 16)
    x_rec = model.decode(z_latent)     # (B, 3, 256, 256) + denorm 포함
    return torch.clamp(x_rec, 0.0, 1.0)


# =========================================================
# Latent manipulations
# =========================================================

def _center_slice(H: int, W: int, extent: int):
    ch, cw = H // 2, W // 2
    h0, h1 = ch - extent, ch + extent
    w0, w1 = cw - extent, cw + extent
    return h0, h1, w0, w1


def mask_center_fill(z_q: torch.Tensor, extent: int, fill: torch.Tensor) -> torch.Tensor:
    """
    fill: (B,C,1,1) or (B,C,Hm,Wm) broadcastable to region.
    """
    z = z_q.clone()
    B, C, H, W = z.shape
    h0, h1, w0, w1 = _center_slice(H, W, extent)
    z[:, :, h0:h1, w0:w1] = fill
    return z


def mask_center_mean(z_q: torch.Tensor, extent: int) -> torch.Tensor:
    fill = z_q.mean(dim=(2, 3), keepdim=True)
    return mask_center_fill(z_q, extent, fill)


def mask_center_zero(z_q: torch.Tensor, extent: int) -> torch.Tensor:
    B, C, _, _ = z_q.shape
    fill = torch.zeros((B, C, 1, 1), device=z_q.device, dtype=z_q.dtype)
    return mask_center_fill(z_q, extent, fill)



def mask_center_mode_code(model, z_q: torch.Tensor, indices: torch.Tensor, extent: int) -> torch.Tensor:
    """
    현재 이미지에서 가장 많이 쓰인 코드(mode)를 마스크 영역에 채움.
    '없앰' 느낌에 꽤 가까운 안정적인 베이스라인으로 자주 씀.
    """
    vq = model.vq_layer

    # embedding weight 찾기
    emb = None
    for attr in ["embedding", "embeddings", "codebook", "embed"]:
        if hasattr(vq, attr):
            obj = getattr(vq, attr)
            if hasattr(obj, "weight"):
                emb = obj.weight
                break
    if emb is None and hasattr(vq, "embedding_weight"):
        emb = vq.embedding_weight
    if emb is None:
        raise RuntimeError("Cannot find codebook embedding weights in model.vq_layer. Check your VectorQuantizer implementation.")

    # indices shape: 보통 (B, H, W) 또는 (B, HW)
    # 가능한 경우만 지원
    if indices.dim() == 3:
        # (B,H,W)
        flat = indices.view(indices.shape[0], -1)
    elif indices.dim() == 2:
        flat = indices
    else:
        raise RuntimeError(f"Unexpected indices shape: {tuple(indices.shape)}")

    # mode per batch
    fills = []
    for b in range(flat.shape[0]):
        vals, counts = torch.unique(flat[b], return_counts=True)
        mode_idx = vals[counts.argmax()].long()
        fills.append(emb[mode_idx].view(1, -1, 1, 1))  # (1,C,1,1)
    fill = torch.cat(fills, dim=0)  # (B,C,1,1)

    return mask_center_fill(z_q, extent, fill)


def shuffle_positions(z_q: torch.Tensor, mode: str = "global", block: int = 2) -> torch.Tensor:
    z = z_q.clone()
    B, C, H, W = z.shape

    if mode == "global":
        flat = z.view(B, C, -1)
        idx = torch.randperm(flat.shape[2], device=z.device)
        return flat[:, :, idx].view(B, C, H, W)

    if mode == "local":
        assert H % block == 0 and W % block == 0, "H,W must be divisible by block for local shuffle."
        out = z.clone()
        for i in range(0, H, block):
            for j in range(0, W, block):
                patch = out[:, :, i:i+block, j:j+block]
                flat = patch.reshape(B, C, -1)
                idx = torch.randperm(flat.shape[2], device=z.device)
                out[:, :, i:i+block, j:j+block] = flat[:, :, idx].reshape(B, C, block, block)
        return out

    raise ValueError(f"Unknown shuffle mode={mode}")


def swap_center(zq_a: torch.Tensor, zq_b: torch.Tensor, extent: int) -> Tuple[torch.Tensor, torch.Tensor]:
    assert zq_a.shape == zq_b.shape
    B, C, H, W = zq_a.shape
    h0, h1, w0, w1 = _center_slice(H, W, extent)

    a2 = zq_a.clone()
    b2 = zq_b.clone()
    tmp = a2[:, :, h0:h1, w0:w1].clone()
    a2[:, :, h0:h1, w0:w1] = b2[:, :, h0:h1, w0:w1]
    b2[:, :, h0:h1, w0:w1] = tmp
    return a2, b2


# =========================================================
# Tasks
# =========================================================

@torch.no_grad()
def task_recon(model, img_pil, x, save_path: str):
    panels = [("Input", np.array(img_pil))]

    zq, indices, info = encode_to_zq_exact(model, x)
    recon = decode_from_zq_exact(model, zq)

    panels.append(("Recon", to_np_img(recon)))
    print("[recon] info:", info)

    save_grid(save_path, panels, figsize=(10, 5))


@torch.no_grad()
def task_mask(model, img_pil, x, save_path: str, extent: int):
    panels = [("Input", np.array(img_pil))]

    zq, indices, info = encode_to_zq_exact(model, x)
    print("[mask] info:", info)

    # fill variants
    zq_mean = mask_center_mean(zq, extent)
    zq_zero = mask_center_zero(zq, extent)
    zq_mode_code = mask_center_mode_code(model, zq, indices, extent)

    panels.append((f"Mask mean (extent={extent})", to_np_img(decode_from_zq_exact(model, zq_mean))))
    panels.append((f"Mask zero (extent={extent})", to_np_img(decode_from_zq_exact(model, zq_zero))))
    panels.append((f"Mask mode_code", to_np_img(decode_from_zq_exact(model, zq_mode_code))))

    save_grid(save_path, panels, figsize=(5 * len(panels), 5))


@torch.no_grad()
def task_shuffle(model, img_pil, x, save_path: str, shuffle_mode: str):
    panels = [("Input", np.array(img_pil))]

    zq, indices, info = encode_to_zq_exact(model, x)
    print("[shuffle] info:", info)

    zq_s = shuffle_positions(zq, mode=shuffle_mode)
    recon_s = decode_from_zq_exact(model, zq_s)
    panels.append((f"Shuffle ({shuffle_mode})", to_np_img(recon_s)))

    save_grid(save_path, panels, figsize=(10, 5))


@torch.no_grad()
def task_swap(model, img_a_pil, x_a, img_b_pil, x_b, save_path: str, extent: int):
    panels = [("Input A", np.array(img_a_pil)), ("Input B", np.array(img_b_pil))]

    zq_a, idx_a, info_a = encode_to_zq_exact(model, x_a)
    zq_b, idx_b, info_b = encode_to_zq_exact(model, x_b)
    print("[swap A] info:", info_a)
    print("[swap B] info:", info_b)

    zq_a2, zq_b2 = swap_center(zq_a, zq_b, extent)

    panels.append((f"A swap<-B (extent={extent})", to_np_img(decode_from_zq_exact(model, zq_a2))))
    panels.append((f"B swap<-A (extent={extent})", to_np_img(decode_from_zq_exact(model, zq_b2))))

    save_grid(save_path, panels, figsize=(5 * len(panels), 5))


# =========================================================
# CLI
# =========================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--img_a", required=True, type=str)
    p.add_argument("--img_b", default=None, type=str, help="Needed for swap")

    p.add_argument("--task", default="recon", choices=["recon", "mask", "shuffle", "swap"])
    p.add_argument("--input_size", default=256, type=int, help="Input size before model.encode() resizes to 224 internally")

    p.add_argument("--extent", default=4, type=int, help="extent=4 => 8x8 area on 16x16 grid")
    p.add_argument("--shuffle_mode", default="global", choices=["global", "local"])

    p.add_argument("--save", default="out.png", type=str)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_ckpt(args.config, args.ckpt, device)

    if not getattr(model, "use_vq", False):
        raise RuntimeError(
            "Loaded model has use_vq=False. This script is for z_q (VQ) manipulation.\n"
            "Please use a VQ-enabled config/checkpoint."
        )

    img_a_pil, x_a = load_image_tensor_center_crop(args.img_a, args.input_size, device)

    if args.task == "recon":
        task_recon(model, img_a_pil, x_a, args.save)
        return

    if args.task == "mask":
        task_mask(model, img_a_pil, x_a, args.save, extent=args.extent)
        return

    if args.task == "shuffle":
        task_shuffle(model, img_a_pil, x_a, args.save, shuffle_mode=args.shuffle_mode)
        return

    if args.task == "swap":
        if args.img_b is None:
            raise ValueError("--task swap requires --img_b")
        img_b_pil, x_b = load_image_tensor_center_crop(args.img_b, args.input_size, device)
        task_swap(model, img_a_pil, x_a, img_b_pil, x_b, args.save, extent=args.extent)
        return

    raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()
