#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import random
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from transformers import GPT2LMHeadModel, GPT2TokenizerFast


# ----------------------------
# CSV helpers
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


def read_rows(csv_path: str, img_col: str, cap_col: str, split_col: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    assert img_col in df.columns, f"img_col={img_col} not in columns={list(df.columns)}"
    assert cap_col in df.columns, f"cap_col={cap_col} not in columns={list(df.columns)}"
    assert split_col in df.columns, f"split_col={split_col} not in columns={list(df.columns)}"

    df["_split"] = [normalize_split_value(v) for v in df[split_col].tolist()]
    df = df[df["_split"] == split].reset_index(drop=True)
    return df


# ----------------------------
# Prefix encoders
# ----------------------------
class DinoPrefix(nn.Module):
    def __init__(self, gpt_hidden: int, prefix_len: int = 256, dino_name: str = "facebook/dinov2-base"):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(dino_name)
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
        x = self.mlp(x)
        return x + self.pos


class VQPrefix(nn.Module):
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
        x = self.emb(flat)
        x = self.mlp(x)
        return x + self.pos


class RAETwitterExtractor(nn.Module):
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

        if hasattr(self.model, "vq_layer") and hasattr(self.model.vq_layer, "num_embeddings"):
            self.codebook_size = int(self.model.vq_layer.num_embeddings)
        elif hasattr(self.model, "vq_layer") and hasattr(self.model.vq_layer, "embedding"):
            self.codebook_size = int(self.model.vq_layer.embedding.num_embeddings)
        else:
            self.codebook_size = None

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images.to(self.device)
        z = self.model.encode(x)

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
# CKPT loader
# ----------------------------
def load_prefixer_ckpt(prefixer: nn.Module, ckpt_path: str) -> Tuple[List[str], List[str]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("prefixer", ckpt)
    missing, unexpected = prefixer.load_state_dict(sd, strict=False)
    return list(missing), list(unexpected)


# ----------------------------
# Decoding helpers
# ----------------------------
def _apply_repetition_penalty(logits: torch.Tensor, generated: torch.Tensor, penalty: float) -> torch.Tensor:
    # logits: (B,V), generated: (B,T)
    if penalty is None or penalty <= 1.0:
        return logits
    logits = logits.clone()
    B, V = logits.shape
    for b in range(B):
        prev = generated[b].tolist()
        for t in set(prev):
            if t < 0 or t >= V:
                continue
            if logits[b, t] < 0:
                logits[b, t] *= penalty
            else:
                logits[b, t] /= penalty
    return logits


def _top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p is None or top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)

    # mask tokens with cumulative prob > top_p
    mask = cum > top_p
    # keep at least 1 token
    mask[..., 0] = False

    sorted_logits = sorted_logits.masked_fill(mask, -1e10)
    # scatter back
    out = torch.empty_like(logits)
    out.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
    return out


def _calc_banned_tokens_no_repeat_ngram(prev_tokens: List[int], n: int) -> set:
    # returns a set of token ids that would create a repeated n-gram
    if n is None or n <= 0:
        return set()
    if len(prev_tokens) < n - 1:
        return set()

    # build mapping: prefix (n-1 tokens) -> set(next_token)
    mapping = {}
    for i in range(len(prev_tokens) - n + 1):
        prefix = tuple(prev_tokens[i:i + n - 1])
        nxt = prev_tokens[i + n - 1]
        mapping.setdefault(prefix, set()).add(nxt)

    cur_prefix = tuple(prev_tokens[-(n - 1):])
    return mapping.get(cur_prefix, set())


@torch.no_grad()
def decode_with_prefix(
    gpt2: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    prefix_embeds: torch.Tensor,          # (B,P,H)
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    top_p: float,
    temperature: float,
    num_beams: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> List[str]:
    """
    Supports: sampling OR simple beam search (slow but ok for N<=20).
    Uses full forward each step (no KV cache) for simplicity/robustness.
    """
    gpt2.eval()
    device = prefix_embeds.device
    B, P, H = prefix_embeds.shape

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) == 0:
        prompt_ids = [tokenizer.eos_token_id]

    if num_beams is None or num_beams <= 1:
        # sampling/greedy
        cur = torch.tensor([prompt_ids], device=device, dtype=torch.long).repeat(B, 1)  # (B,T)
        for _ in range(max_new_tokens):
            tok_emb = gpt2.transformer.wte(cur)  # (B,T,H)
            inputs_embeds = torch.cat([prefix_embeds, tok_emb], dim=1)  # (B,P+T,H)
            attn = torch.ones((B, inputs_embeds.size(1)), device=device, dtype=torch.long)

            out = gpt2(inputs_embeds=inputs_embeds, attention_mask=attn)
            logits = out.logits[:, -1, :]  # (B,V)

            logits = _apply_repetition_penalty(logits, cur, repetition_penalty)

            # no repeat ngram
            if no_repeat_ngram_size and no_repeat_ngram_size > 0:
                for b in range(B):
                    banned = _calc_banned_tokens_no_repeat_ngram(cur[b].tolist(), no_repeat_ngram_size)
                    if len(banned) > 0:
                        logits[b, list(banned)] = -1e10

            if temperature and temperature > 0:
                logits = logits / max(1e-6, temperature)

            if do_sample:
                logits = _top_p_filtering(logits, top_p)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)  # (B,1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)

            cur = torch.cat([cur, next_id], dim=1)

            # early stop if all EOS
            if torch.all(next_id.squeeze(1) == tokenizer.eos_token_id):
                break

        # decode (remove prompt)
        gen = cur[:, len(prompt_ids):]
        return [tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen]

    # Beam search (batch=1 recommended; for B>1 we'll do per-sample)
    texts = []
    for b in range(B):
        beams = [(prompt_ids[:], 0.0)]  # (tokens, score)
        for _ in range(max_new_tokens):
            all_cands = []
            for seq, score in beams:
                cur = torch.tensor([seq], device=device, dtype=torch.long)
                tok_emb = gpt2.transformer.wte(cur)
                inputs_embeds = torch.cat([prefix_embeds[b:b+1], tok_emb], dim=1)
                attn = torch.ones((1, inputs_embeds.size(1)), device=device, dtype=torch.long)

                out = gpt2(inputs_embeds=inputs_embeds, attention_mask=attn)
                logits = out.logits[:, -1, :].squeeze(0)  # (V,)

                logits = _apply_repetition_penalty(logits.unsqueeze(0), cur, repetition_penalty).squeeze(0)

                if no_repeat_ngram_size and no_repeat_ngram_size > 0:
                    banned = _calc_banned_tokens_no_repeat_ngram(seq, no_repeat_ngram_size)
                    if len(banned) > 0:
                        logits[list(banned)] = -1e10

                if temperature and temperature > 0:
                    logits = logits / max(1e-6, temperature)

                # expand
                logprobs = F.log_softmax(logits, dim=-1)  # (V,)
                topk = torch.topk(logprobs, k=num_beams)
                for logp, tid in zip(topk.values.tolist(), topk.indices.tolist()):
                    all_cands.append((seq + [tid], score + logp))

            # keep best beams
            all_cands.sort(key=lambda x: x[1], reverse=True)
            beams = all_cands[:num_beams]

            # if all beams ended, stop
            if all(seq[-1] == tokenizer.eos_token_id for seq, _ in beams):
                break

        best_seq = beams[0][0][len(prompt_ids):]
        texts.append(tokenizer.decode(best_seq, skip_special_tokens=True).strip())

    return texts


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", default="/dataset/flickr30k/flickr_annotations_30k.csv")
    ap.add_argument("--imgdir", default="/dataset/flickr30k/flickr30k-images")
    ap.add_argument("--img_col", default="filename")
    ap.add_argument("--cap_col", default="raw")
    ap.add_argument("--split_col", default="split")
    ap.add_argument("--split", default="val")

    ap.add_argument("--dino_ckpt", default="runs/flickr_prefix_gpt2/dino_best.pt")
    ap.add_argument("--vq_ckpt",   default="runs/flickr_prefix_gpt2/vq_best.pt")
    ap.add_argument("--only", choices=["both", "dino", "vq"], default="both")

    ap.add_argument("--rae_config", default="configs/stage1/training/DINOv2-B_decXL.yaml")
    ap.add_argument("--rae_ckpt",   default="pca_results/008-RAE/checkpoints/1280000.pt")

    ap.add_argument("--gpt2", default="gpt2")
    ap.add_argument("--dino_name", default="facebook/dinov2-base")

    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--grid_hw", type=int, default=16)
    ap.add_argument("--vq_emb_dim", type=int, default=256)

    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/flickr_prefix_gpt2/qual_compare.md")

    # decoding controls
    ap.add_argument("--prompt", type=str, default="A photo of ")
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = GPT2TokenizerFast.from_pretrained(args.gpt2)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    gpt2 = GPT2LMHeadModel.from_pretrained(args.gpt2).to(device).eval()
    for p in gpt2.parameters():
        p.requires_grad = False

    df = read_rows(args.csv, args.img_col, args.cap_col, args.split_col, split=args.split)
    if len(df) == 0:
        raise RuntimeError(f"No samples found for split={args.split}")

    # group captions per image
    grouped = df.groupby(args.img_col)[args.cap_col].apply(list).reset_index()
    grouped = grouped.sample(n=min(args.n, len(grouped)), random_state=args.seed).reset_index(drop=True)

    tfm = T.Compose([T.Resize((args.image_size, args.image_size)), T.ToTensor()])

    prefix_len = args.grid_hw * args.grid_hw
    dino_prefixer = None
    vq_prefixer = None
    vq_extractor = None

    if args.only in ["both", "dino"]:
        dino_prefixer = DinoPrefix(gpt_hidden=gpt2.config.n_embd, prefix_len=prefix_len, dino_name=args.dino_name).to(device).eval()
        miss, unexp = load_prefixer_ckpt(dino_prefixer, args.dino_ckpt)
        print(f"[Load DINO ckpt] missing={len(miss)} unexpected={len(unexp)}")

    if args.only in ["both", "vq"]:
        vq_extractor = RAETwitterExtractor(args.rae_config, args.rae_ckpt, device=device).to(device).eval()
        codebook_size = vq_extractor.codebook_size
        if codebook_size is None:
            raise RuntimeError("Could not infer codebook size from RAE extractor.")
        vq_prefixer = VQPrefix(gpt_hidden=gpt2.config.n_embd, codebook_size=codebook_size, grid_hw=args.grid_hw, emb_dim=args.vq_emb_dim).to(device).eval()
        miss, unexp = load_prefixer_ckpt(vq_prefixer, args.vq_ckpt)
        print(f"[Load VQ ckpt] missing={len(miss)} unexpected={len(unexp)}")

    rows_out: List[Dict] = []
    for i in tqdm(range(len(grouped)), desc="Generate"):
        fname = str(grouped.loc[i, args.img_col])
        captions = grouped.loc[i, args.cap_col]
        ref = captions[0] if isinstance(captions, list) and len(captions) > 0 else ""

        img_path = fname if os.path.isabs(fname) else os.path.join(args.imgdir, fname)
        img = Image.open(img_path).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)

        out_row = {"image": fname, "ref": ref}

        if dino_prefixer is not None:
            with torch.no_grad():
                prefix = dino_prefixer(x)
            gen = decode_with_prefix(
                gpt2, tok, prefix,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                top_p=args.top_p,
                temperature=args.temperature,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )[0]
            out_row["dino"] = gen

        if vq_prefixer is not None and vq_extractor is not None:
            with torch.no_grad():
                idx = vq_extractor(x)
                prefix = vq_prefixer(idx.to(device))
            gen = decode_with_prefix(
                gpt2, tok, prefix,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                top_p=args.top_p,
                temperature=args.temperature,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )[0]
            out_row["vq"] = gen

        rows_out.append(out_row)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"# Flickr30k Qualitative Caption Comparison (N={len(rows_out)})\n\n")
        f.write(f"- split: `{args.split}`\n")
        f.write(f"- DINO ckpt: `{args.dino_ckpt}`\n")
        f.write(f"- VQ ckpt: `{args.vq_ckpt}`\n")
        f.write(f"- decoding: prompt=`{args.prompt}` do_sample={args.do_sample} top_p={args.top_p} temp={args.temperature} beams={args.num_beams}\n\n")

        for k, r in enumerate(rows_out):
            f.write(f"## {k+1}. {r['image']}\n\n")
            f.write(f"![Image]({os.path.join(args.imgdir, r['image'])})\n\n")
            f.write(f"**Reference (one of captions):** {r['ref']}\n\n")
            if "dino" in r:
                f.write(f"**DINO:** {r['dino']}\n\n")
            if "vq" in r:
                f.write(f"**VQ:** {r['vq']}\n\n")
            f.write("---\n\n")

    print(f"✅ Saved qualitative comparison to: {args.out}")


if __name__ == "__main__":
    main()
