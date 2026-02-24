#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs
from stage1 import RAE


def get_device(explicit: str | None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    return tfm(img)



def pool_latent(z: torch.Tensor) -> torch.Tensor:
    """
    z expected:
      - (B, C, H, W)  -> mean over H,W => (B,C)
      - (B, N, C)     -> mean over N   => (B,C)
      - (B, C)        -> keep          => (B,C)
    """
    if z.ndim == 4:
        return z.mean(dim=(2, 3))
    if z.ndim == 3:
        return z.mean(dim=1)
    if z.ndim == 2:
        return z
    raise ValueError(f"Unexpected latent shape: {tuple(z.shape)}")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config with stage_1 section.")
    parser.add_argument("--split_json", default="/dataset/flickr30k/coco_fmt/train.json",
                        help="COCO-format json (train/val/test).")
    parser.add_argument("--image_dir", default="/dataset/flickr30k/flickr30k-images",
                        help="Directory containing images.")
    parser.add_argument("--out_pt", default="/dataset/flickr30k/features/features_rae_cont.pt",
                        help="Output .pt path (dict[filename]->tensor[d]).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--normalize_prefix", action="store_true",
                        help="Optional: L2-normalize pooled features (CLIP처럼).")
    args = parser.parse_args()

    device = get_device(args.device)

    # ----- load dataset list -----
    coco = json.load(open(args.split_json, "r"))
    filenames: List[str] = [im["file_name"] for im in coco["images"]]

    # ----- load model -----
    rae_config, *_ = parse_configs(args.config)
    if rae_config is None:
        raise ValueError(f"No stage_1 section found in config {args.config}")

    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()

    print("[INFO] num images:", len(filenames))
    print("[INFO] saving to:", args.out_pt)

    out: Dict[str, torch.Tensor] = {}
    bs = args.batch_size

    # ----- batched extraction -----
    batch_x = []
    batch_f = []

    for fn in tqdm(filenames, desc="extract"):
        img_path = Path(args.image_dir) / fn
        if not img_path.exists():
            # 누락 이미지가 있으면 스킵 (정상적으로는 없어야 함)
            continue

        x = load_image_tensor(img_path)  # (C,H,W)
        batch_x.append(x)
        batch_f.append(fn)

        if len(batch_x) == bs:
            X = torch.stack(batch_x, dim=0).to(device)  # (B,C,H,W)
            z = rae.encode(X)                            # latent
            v = pool_latent(z)                           # (B,d)
            if args.normalize_prefix:
                v = F.normalize(v, dim=-1)

            v = v.detach().cpu().float()
            for i, f in enumerate(batch_f):
                out[f] = v[i]

            batch_x, batch_f = [], []

    # tail
    if batch_x:
        X = torch.stack(batch_x, dim=0).to(device)
        z = rae.encode(X)
        v = pool_latent(z)
        if args.normalize_prefix:
            v = F.normalize(v, dim=-1)
        v = v.detach().cpu().float()
        for i, f in enumerate(batch_f):
            out[f] = v[i]

    Path(args.out_pt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out_pt)

    # sanity
    k = next(iter(out.keys()))
    print("[OK] saved", args.out_pt)
    print("num:", len(out), "sample:", k, "vec shape:", tuple(out[k].shape), "dtype:", out[k].dtype)


if __name__ == "__main__":
    main()
