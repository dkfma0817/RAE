#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_gpt2.py  (FIXED / DROP-IN)

Fixes included (the ones you need right now):
1) ✅ GPT-2 stays in eval() ALWAYS (dropout OFF).  (prefix-only training otherwise often fails)
2) ✅ Flickr30k `raw` caption column that stores a *list string* like ["cap1",...,"cap5"]:
   - parses it via ast.literal_eval
   - trains by sampling ONE caption per __getitem__ (random each time)
3) ✅ Overfit mode no longer breaks when image exists only in val/test:
   - overfit selects rows from ALL splits
   - then makes a tiny train/val split internally (or val=train if too small)
4) ✅ Adds CLI flags for --img_col/--cap_col/--split_col (and they actually work)
5) ✅ Robust val_loss handling when val set is empty (no NaN spam)
6) ✅ Optional small unfreezing knobs (ln_f / lm_head / last_n blocks) if you want them later

Usage examples:
- Full train (DINO):
  CUDA_VISIBLE_DEVICES=0 python src/train_gpt2.py --mode dino --csv ... --imgdir ... --img_col filename --cap_col raw --split_col split

- Overfit a single image:
  CUDA_VISIBLE_DEVICES=0 python src/train_gpt2.py --mode dino --csv ... --imgdir ... --img_col filename --cap_col raw --split_col split \
      --overfit_image 1000092795.jpg --batch 1 --epochs 200 --lr 2e-4 --outdir runs/overfit_1img_dino_fixed
"""

import os
import math
import argparse
import random
import ast
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


# ----------------------------
# Split helpers
# ----------------------------
def normalize_split_value(x: str) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ["train", "training"]:
        return "train"
    if s in ["val", "valid", "validation", "dev"]:
        return "val"
    if s in ["test", "testing"]:
        return "test"
    return None


def stable_split_from_filename(fname: str) -> str:
    # deterministic split if CSV has no split column
    h = abs(hash(fname)) % 1000
    if h < 800:
        return "train"
    elif h < 900:
        return "val"
    else:
        return "test"


def guess_columns(df: pd.DataFrame) -> Tuple[str, str, Optional[str]]:
    img_cands = ["image", "filename", "file_name", "img", "image_name", "photo", "path"]
    cap_cands = ["caption", "sentence", "comment", "text", "description", "raw", "rawcaption"]

    img_col = None
    cap_col = None
    split_col = None

    for cand in img_cands:
        for c in df.columns:
            if cand in c.lower():
                img_col = c
                break
        if img_col:
            break

    for cand in cap_cands:
        for c in df.columns:
            if cand == c.lower() or cand in c.lower():
                cap_col = c
                break
        if cap_col:
            break

    for c in df.columns:
        if "split" in c.lower():
            split_col = c
            break

    if img_col is None or cap_col is None:
        raise RuntimeError(
            f"Could not infer image/caption columns. df.columns={list(df.columns)}\n"
            "Pass --img_col/--cap_col/--split_col explicitly."
        )
    return img_col, cap_col, split_col


# ----------------------------
# Caption parsing
# ----------------------------
def parse_caption_cell(v) -> List[str]:
    """
    Flickr30k CSV can store captions either as:
      - a single string caption
      - a stringified python list: ["cap1", "cap2", ...]
    Return: list of captions (len>=1).
    """
    s = "" if v is None else str(v)
    st = s.strip()
    if len(st) == 0:
        return [""]
    if st.startswith("[") and st.endswith("]"):
        try:
            obj = ast.literal_eval(st)
            if isinstance(obj, (list, tuple)) and len(obj) > 0:
                return [str(x) for x in obj]
        except Exception:
            pass
    return [s]


# ----------------------------
# Dataset
# ----------------------------
class Flickr30kCaptions(Dataset):
    """
    split in {"train","val","test","all"}
    If captions are list-strings, samples 1 caption per item (random each access).
    """
    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        split: str,
        tokenizer: GPT2TokenizerFast,
        image_size: int = 224,
        max_text_len: int = 40,
        img_col: Optional[str] = None,
        cap_col: Optional[str] = None,
        split_col: Optional[str] = None,
        seed: int = 0,
    ):
        assert split in ["train", "val", "test", "all"], f"bad split={split}"
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.split = split
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

        if img_col is None or cap_col is None:
            gi, gc, gs = guess_columns(self.df)
            img_col = img_col or gi
            cap_col = cap_col or gc
            split_col = split_col or gs

        self.img_col = img_col
        self.cap_col = cap_col
        self.split_col = split_col

        # Determine normalized split per row
        if self.split_col is not None and self.split_col in self.df.columns:
            self.df["_norm_split"] = [normalize_split_value(v) for v in self.df[self.split_col].tolist()]
        else:
            self.df["_norm_split"] = [stable_split_from_filename(str(v)) for v in self.df[self.img_col].tolist()]

        # Filter by split unless split="all"
        if split != "all":
            self.df = self.df[self.df["_norm_split"] == split].reset_index(drop=True)

        self.tfm = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.df)

    def _resolve_img_path(self, img_name: str) -> str:
        img_path = img_name
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.image_dir, img_name)
        return img_path

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        img_name = str(row[self.img_col])
        caps = parse_caption_cell(row[self.cap_col])
        cap = self.rng.choice(caps) if len(caps) > 0 else ""

        img_path = self._resolve_img_path(img_name)
        img = Image.open(img_path).convert("RGB")
        x = self.tfm(img)  # (3,H,W)

        enc = self.tokenizer(
            cap,
            truncation=True,
            max_length=self.max_text_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)         # (L,)
        attention_mask = enc["attention_mask"].squeeze(0)

        return {
            "image": x,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


@dataclass
class Collate:
    pad_id: int
    def __call__(self, batch):
        images = torch.stack([b["image"] for b in batch], dim=0)
        input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
        attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
        return {"image": images, "input_ids": input_ids, "attention_mask": attention_mask}


# ----------------------------
# Visual prefix encoders
# ----------------------------
class DinoPrefix(nn.Module):
    """
    Baseline: DINOv2 patch tokens -> LN -> MLP -> GPT2 hidden (+ pos)
    """
    def __init__(self, gpt_hidden: int, prefix_len: int = 256, dino_name: str = "facebook/dinov2-base", use_fast: bool = True):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel

        # Silence that "slow processor" warning by default
        self.processor = AutoImageProcessor.from_pretrained(dino_name, use_fast=use_fast)
        self.dino = AutoModel.from_pretrained(dino_name)
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad = False

        dino_hidden = self.dino.config.hidden_size
        self.ln = nn.LayerNorm(dino_hidden)
        self.mlp = nn.Sequential(
            nn.Linear(dino_hidden, gpt_hidden),
            nn.GELU(),
            nn.Linear(gpt_hidden, gpt_hidden),
        )

        self.prefix_len = prefix_len
        self.pos = nn.Parameter(torch.zeros(1, prefix_len, gpt_hidden))
        nn.init.normal_(self.pos, std=0.02)

    @torch.no_grad()
    def _extract_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        # processor accepts list of uint8 arrays
        imgs = (images.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype(np.uint8)
        inputs = self.processor(images=list(imgs), return_tensors="pt")
        inputs = {k: v.to(images.device) for k, v in inputs.items()}
        out = self.dino(**inputs)
        tokens = out.last_hidden_state[:, 1:, :]  # drop CLS
        return tokens

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self._extract_patch_tokens(images)  # (B,N,D)
        if tokens.size(1) > self.prefix_len:
            tokens = tokens[:, :self.prefix_len, :]
        elif tokens.size(1) < self.prefix_len:
            pad = torch.zeros(tokens.size(0), self.prefix_len - tokens.size(1), tokens.size(2), device=tokens.device)
            tokens = torch.cat([tokens, pad], dim=1)

        x = self.ln(tokens)
        x = self.mlp(x)  # (B,prefix_len,gpt_hidden)
        return x + self.pos


class VQPrefix(nn.Module):
    """
    Proposed: VQ indices -> emb -> MLP -> GPT2 hidden (+ pos)
    """
    def __init__(self, gpt_hidden: int, codebook_size: int, grid_hw: int = 16, emb_dim: int = 256):
        super().__init__()
        self.grid_hw = grid_hw
        self.prefix_len = grid_hw * grid_hw

        self.emb = nn.Embedding(codebook_size, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, gpt_hidden),
            nn.GELU(),
            nn.Linear(gpt_hidden, gpt_hidden),
        )
        self.pos = nn.Parameter(torch.zeros(1, self.prefix_len, gpt_hidden))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, indices_2d: torch.Tensor) -> torch.Tensor:
        B, H, W = indices_2d.shape
        flat = indices_2d.view(B, -1)

        if flat.size(1) != self.prefix_len:
            if flat.size(1) > self.prefix_len:
                flat = flat[:, :self.prefix_len]
            else:
                pad = torch.zeros(B, self.prefix_len - flat.size(1), dtype=flat.dtype, device=flat.device)
                flat = torch.cat([flat, pad], dim=1)

        x = self.emb(flat)  # (B,HW,emb_dim)
        x = self.mlp(x)     # (B,HW,gpt_hidden)
        return x + self.pos


# ----------------------------
# Your VQ indices extractor (RAE)
# ----------------------------
class RAETwitterExtractor(nn.Module):
    """
    Frozen wrapper around your RAE that returns VQ indices (B,H,W) long.
    """
    def __init__(self, config_path: str, ckpt_path: str, device: torch.device):
        super().__init__()
        from omegaconf import OmegaConf
        from utils.model_utils import instantiate_from_config

        conf = OmegaConf.load(config_path)
        model_conf = conf.stage_1 if "stage_1" in conf else (conf.model if "model" in conf else conf)
        model = instantiate_from_config(model_conf).to(device).eval()

        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt["ema"] if isinstance(ckpt, dict) and "ema" in ckpt else (ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        _ = model.load_state_dict(sd, strict=False)

        for p in model.parameters():
            p.requires_grad = False

        self.model = model
        self.device = device

        # try infer codebook size
        if hasattr(self.model, "vq_layer") and hasattr(self.model.vq_layer, "num_embeddings"):
            self.codebook_size = int(self.model.vq_layer.num_embeddings)
        elif hasattr(self.model, "vq_layer") and hasattr(self.model.vq_layer, "embedding"):
            self.codebook_size = int(self.model.vq_layer.embedding.num_embeddings)
        else:
            self.codebook_size = None

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images.to(self.device)
        z = self.model.encode(x)  # (B,C,H,W) expected

        if hasattr(self.model, "pca_reweight") and self.model.pca_reweight is not None:
            z = self.model.pca_reweight(z)
        if hasattr(self.model, "vq_pre") and self.model.vq_pre is not None:
            z = self.model.vq_pre(z)
        if hasattr(self.model, "vq_z_norm") and bool(self.model.vq_z_norm):
            denom = z.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
            z = z / denom

        out = self.model.vq_layer(z)
        indices = None
        if isinstance(out, (tuple, list)) and len(out) >= 3:
            indices = out[2]
        elif hasattr(self.model, "last_vq_indices") and self.model.last_vq_indices is not None:
            indices = self.model.last_vq_indices
        else:
            raise RuntimeError("Could not extract indices from vq_layer output.")

        if indices.dim() == 3:
            return indices.long()
        if indices.dim() == 2:
            B, N = indices.shape
            H = W = int(round(N ** 0.5))
            if H * W != N:
                raise RuntimeError(f"indices length {N} not square.")
            return indices.view(B, H, W).long()
        raise RuntimeError(f"Unexpected indices shape: {tuple(indices.shape)}")


# ----------------------------
# Captioning model: Prefix + GPT-2 (frozen by default)
# ----------------------------
class PrefixGPT2Captioner(nn.Module):
    def __init__(self, gpt2_name: str):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_name)
        # IMPORTANT: keep dropout OFF always
        self.gpt2.eval()

    def freeze_all(self):
        for p in self.gpt2.parameters():
            p.requires_grad = False

    def unfreeze_ln_f(self):
        if hasattr(self.gpt2, "transformer") and hasattr(self.gpt2.transformer, "ln_f"):
            for p in self.gpt2.transformer.ln_f.parameters():
                p.requires_grad = True

    def unfreeze_lm_head(self):
        if hasattr(self.gpt2, "lm_head"):
            for p in self.gpt2.lm_head.parameters():
                p.requires_grad = True

    def unfreeze_last_n_blocks(self, n: int):
        if n <= 0:
            return
        blocks = self.gpt2.transformer.h
        n = min(n, len(blocks))
        for blk in blocks[-n:]:
            for p in blk.parameters():
                p.requires_grad = True

    def forward(self, prefix_embeds: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        prefix_embeds: (B, P, H)
        input_ids: (B, L)
        attention_mask: (B, L)
        """
        B, P, H = prefix_embeds.shape
        tok_emb = self.gpt2.transformer.wte(input_ids)          # (B,L,H)
        inputs_embeds = torch.cat([prefix_embeds, tok_emb], 1)  # (B,P+L,H)

        labels = input_ids.clone()
        labels = torch.cat([torch.full((B, P), -100, device=labels.device, dtype=labels.dtype), labels], 1)

        attn = torch.cat([torch.ones((B, P), device=attention_mask.device, dtype=attention_mask.dtype), attention_mask], 1)

        # Ensure GPT-2 stays eval() (dropout OFF) even during training
        self.gpt2.eval()
        out = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attn, labels=labels)
        return out.loss


# ----------------------------
# Evaluate
# ----------------------------
@torch.no_grad()
def evaluate(model_cap, prefixer, vq_extractor, dl, device, mode: str) -> float:
    model_cap.gpt2.eval()
    prefixer.eval()
    if vq_extractor is not None:
        vq_extractor.eval()

    losses = []
    for batch in tqdm(dl, desc="Eval", leave=False):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)

        if mode == "dino":
            prefix = prefixer(images)
        else:
            idx = vq_extractor(images)
            prefix = prefixer(idx.to(device))

        loss = model_cap(prefix, input_ids, attn)
        losses.append(float(loss.item()))

    if len(losses) == 0:
        # no validation batches -> return +inf so it won't be selected as best
        return float("inf")
    return float(np.mean(losses))


# ----------------------------
# Overfit selection logic
# ----------------------------
def make_overfit_df(df_all: pd.DataFrame, img_col: str, overfit_image: Optional[str], overfit_n: int) -> pd.DataFrame:
    df = df_all.copy()
    if overfit_image is not None:
        df = df[df[img_col].astype(str) == str(overfit_image)].reset_index(drop=True)
    elif overfit_n > 0:
        df = df.head(overfit_n).reset_index(drop=True)
    return df.reset_index(drop=True)


def train_val_split_df(df: pd.DataFrame, val_ratio: float = 0.2, seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) <= 1:
        return df.copy(), df.copy()
    rng = np.random.RandomState(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(df) * val_ratio)))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    if len(tr_idx) == 0:
        tr_idx = val_idx
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", default="/dataset/flickr30k/flickr_annotations_30k.csv")
    ap.add_argument("--imgdir", default="/dataset/flickr30k/flickr30k-images")
    ap.add_argument("--mode", choices=["dino", "vq"], required=True)

    ap.add_argument("--img_col", default=None)
    ap.add_argument("--cap_col", default=None)
    ap.add_argument("--split_col", default=None)

    ap.add_argument("--gpt2", default="gpt2")
    ap.add_argument("--max_text_len", type=int, default=40)
    ap.add_argument("--image_size", type=int, default=224)

    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.01)

    # DINO
    ap.add_argument("--dino_name", default="facebook/dinov2-base")
    ap.add_argument("--use_fast_processor", action="store_true", help="Use fast image processor when available")
    ap.add_argument("--grid_hw", type=int, default=16)

    # VQ / RAE
    ap.add_argument("--rae_config", default="configs/stage1/training/DINOv2-B_decXL.yaml")
    ap.add_argument("--rae_ckpt",   default="pca_results/008-RAE/checkpoints/1280000.pt")
    ap.add_argument("--vq_emb_dim", type=int, default=256)

    # optional small unfreezing
    ap.add_argument("--unfreeze_ln_f", action="store_true")
    ap.add_argument("--unfreeze_lm_head", action="store_true")
    ap.add_argument("--unfreeze_last_n", type=int, default=0)

    # overfit
    ap.add_argument("--overfit_n", type=int, default=0)
    ap.add_argument("--overfit_image", type=str, default=None)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="runs/flickr_prefix_gpt2_fixed")

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    # tokenizer
    tok = GPT2TokenizerFast.from_pretrained(args.gpt2)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Load full df once to know columns + allow overfit from ALL splits
    df_all = pd.read_csv(args.csv)
    if args.img_col is None or args.cap_col is None:
        gi, gc, gs = guess_columns(df_all)
        args.img_col = args.img_col or gi
        args.cap_col = args.cap_col or gc
        args.split_col = args.split_col or gs

    # datasets
    train_ds = Flickr30kCaptions(
        args.csv, args.imgdir, "train", tok,
        image_size=args.image_size, max_text_len=args.max_text_len,
        img_col=args.img_col, cap_col=args.cap_col, split_col=args.split_col,
        seed=args.seed,
    )
    val_ds = Flickr30kCaptions(
        args.csv, args.imgdir, "val", tok,
        image_size=args.image_size, max_text_len=args.max_text_len,
        img_col=args.img_col, cap_col=args.cap_col, split_col=args.split_col,
        seed=args.seed + 1,
    )

    # Overfit mode: select rows from ALL, then create internal train/val split
    if args.overfit_n > 0 or args.overfit_image is not None:
        ds_all = Flickr30kCaptions(
            args.csv, args.imgdir, "all", tok,
            image_size=args.image_size, max_text_len=args.max_text_len,
            img_col=args.img_col, cap_col=args.cap_col, split_col=args.split_col,
            seed=args.seed,
        )
        df_sel = make_overfit_df(ds_all.df, ds_all.img_col, args.overfit_image, args.overfit_n)
        if len(df_sel) == 0:
            raise RuntimeError(f"[Overfit mode] No rows found for overfit_image={args.overfit_image} / overfit_n={args.overfit_n}")

        df_tr, df_va = train_val_split_df(df_sel, val_ratio=0.2, seed=args.seed)
        train_ds = ds_all
        val_ds = ds_all
        train_ds.df = df_tr.reset_index(drop=True)
        val_ds.df = df_va.reset_index(drop=True)
        print(f"[Overfit mode] selected_rows={len(df_sel)} train={len(train_ds)} val={len(val_ds)}")

    collate = Collate(pad_id=tok.pad_token_id)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate)

    # captioner
    cap = PrefixGPT2Captioner(args.gpt2).to(device)
    cap.freeze_all()

    # optional unfreeze knobs (small, safe)
    if args.unfreeze_ln_f:
        cap.unfreeze_ln_f()
    if args.unfreeze_lm_head:
        cap.unfreeze_lm_head()
    if args.unfreeze_last_n > 0:
        cap.unfreeze_last_n_blocks(args.unfreeze_last_n)

    # prefixer + extractor
    vq_extractor = None
    prefix_len = args.grid_hw * args.grid_hw

    if args.mode == "dino":
        prefixer = DinoPrefix(
            gpt_hidden=cap.gpt2.config.n_embd,
            prefix_len=prefix_len,
            dino_name=args.dino_name,
            use_fast=args.use_fast_processor,
        ).to(device)
    else:
        vq_extractor = RAETwitterExtractor(args.rae_config, args.rae_ckpt, device=device).to(device)
        if vq_extractor.codebook_size is None:
            raise RuntimeError("Could not infer codebook size from RAE extractor; hardcode it or expose it in model.")
        prefixer = VQPrefix(
            gpt_hidden=cap.gpt2.config.n_embd,
            codebook_size=vq_extractor.codebook_size,
            grid_hw=args.grid_hw,
            emb_dim=args.vq_emb_dim,
        ).to(device)

    # optimizer: prefixer + any unfrozen gpt2 params
    train_params = list(prefixer.parameters()) + [p for p in cap.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=args.wd)

    best_val = float("inf")

    for ep in range(args.epochs):
        # KEY: keep GPT-2 eval() ALWAYS
        cap.gpt2.eval()
        prefixer.train()
        if vq_extractor is not None:
            vq_extractor.eval()

        pbar = tqdm(train_dl, desc=f"Train ep{ep}")
        running = 0.0

        for it, batch in enumerate(pbar):
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)

            if args.mode == "dino":
                prefix = prefixer(images)
            else:
                with torch.no_grad():
                    idx = vq_extractor(images)  # (B,H,W)
                prefix = prefixer(idx.to(device))

            loss = cap(prefix, input_ids, attn)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(train_params, 1.0)
            opt.step()

            running += float(loss.item())
            pbar.set_postfix(loss=running / (it + 1))

        val_loss = evaluate(cap, prefixer, vq_extractor, val_dl, device, args.mode)
        print(f"[ep {ep}] val_loss = {val_loss:.4f}")

        ckpt = {
            "epoch": ep,
            "mode": args.mode,
            "prefixer": prefixer.state_dict(),
            "val_loss": val_loss,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.outdir, f"{args.mode}_ep{ep}.pt"))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(args.outdir, f"{args.mode}_best.pt"))

    print("Done. Best val:", best_val)


if __name__ == "__main__":
    main()
