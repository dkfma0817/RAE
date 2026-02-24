"""
Frozen RAE(+VQ) + Frozen GPT-2 + Trainable Adapter
==================================================
- RAE checkpoint(.pt)에서 VQ indices를 뽑아 prefix embedding으로 만들고
- GPT-2는 고정한 채, Adapter(Embedding(+MLP))만 학습합니다.

개선사항:
- 입력 정규화 일치 (norm_mode)
- 학습 설정 개선 (lr, epochs)
- Overfitting 테스트 모드
- VQ 인덱스 디버깅
- 체크포인트 저장/로드
"""

import os
import ast
import math
import random
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

from omegaconf import OmegaConf
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs


# =========================
# 0) Utils
# =========================
def freeze_module(m: nn.Module):
    m.eval()
    for p in m.parameters():
        p.requires_grad = False


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    cfg_dict: dict,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "adapter": model.adapter.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "cfg": cfg_dict,
        },
        path,
    )


def load_checkpoint(path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
    ckpt = torch.load(path, map_location="cpu")
    model.adapter.load_state_dict(ckpt["adapter"], strict=True)
    if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = int(ckpt.get("epoch", 0))
    start_step = int(ckpt.get("step", 0))
    return start_epoch, start_step


# =========================
# 0.5) Transform builder
# =========================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(image_size: int, norm_mode: str = "imagenet") -> transforms.Compose:
    """
    norm_mode:
      - "none"      : ToTensor() only -> [0,1]
      - "minus1_1"  : ToTensor() then x*2-1 -> [-1,1]
      - "imagenet"  : ToTensor() + Normalize(mean,std)  (DINO/DINOv2에서 흔함)
    """
    t = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]

    norm_mode = (norm_mode or "none").lower()
    if norm_mode == "none":
        return transforms.Compose(t)
    if norm_mode in ["minus1_1", "neg1_1", "[-1,1]"]:
        t.append(transforms.Lambda(lambda x: x * 2.0 - 1.0))
        return transforms.Compose(t)
    if norm_mode in ["imagenet", "in1k", "dino", "dinov2"]:
        t.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
        return transforms.Compose(t)

    raise ValueError(f"Unknown norm_mode='{norm_mode}'. Use: none / minus1_1 / imagenet")


# =========================
# 1) Adapter
# =========================
class VQPrefixAdapter(nn.Module):
    """
    VQ token indices [B, L] -> GPT-2 입력 임베딩 [B, num_prefix, d_model]
    Cross-Attention으로 L개 토큰을 num_prefix개로 압축
    """

    def __init__(self, num_codes: int, d_model: int, num_prefix: int = 16, use_mlp: bool = True):
        super().__init__()
        self.embed = nn.Embedding(num_codes, d_model)
        self.num_prefix = num_prefix
        
        # Learned queries: [num_prefix, d_model]
        self.queries = nn.Parameter(torch.randn(num_prefix, d_model) * 0.02)
        
        # Cross-Attention
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, batch_first=True)
        
        if use_mlp:
            self.proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: [B, L]
        x = self.embed(idx)  # [B, L, D]
        B = x.size(0)
        
        # Expand queries for batch
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, num_prefix, D]
        
        # Cross-Attention: queries attend to VQ tokens
        pooled, _ = self.attn(q, x, x, need_weights=False)  # [B, num_prefix, D]
        
        return self.proj(pooled)


# =========================
# 2) Main: RAE + Adapter + GPT2
# =========================
class VQToGPT2(nn.Module):
    def __init__(
        self,
        rae_ckpt_path: str,
        rae_config_path: str,
        gpt_model_name: str = "gpt2",
        max_text_len: int = 64,
        use_adapter_mlp: bool = True,
        num_prefix: int = 16,  # ✅ NEW: Cross-Attention으로 압축할 토큰 수
        debug_vq_return: bool = False,
    ):
        super().__init__()

        self.max_text_len = max_text_len
        self.debug_vq_return = debug_vq_return
        self.num_prefix = num_prefix

        # ---- A) RAE load & freeze ----
        print(f"[Load] RAE config: {rae_config_path}")
        cfg = OmegaConf.load(rae_config_path)
        rae_config, *_ = parse_configs(rae_config_path)
        self.rae = instantiate_from_config(rae_config).cpu()

        print(f"[Load] RAE ckpt: {rae_ckpt_path}")
        ckpt = torch.load(rae_ckpt_path, map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = self.rae.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[Warn] Missing keys in RAE load_state_dict: {len(missing)}")
        if unexpected:
            print(f"[Warn] Unexpected keys in RAE load_state_dict: {len(unexpected)}")

        freeze_module(self.rae)

        # ---- B) GPT-2 load & freeze ----
        print(f"[Load] GPT-2: {gpt_model_name}")
        self.gpt = GPT2LMHeadModel.from_pretrained(gpt_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        freeze_module(self.gpt)

        # ---- C) Adapter ----
        vq_num_codes = rae_config.get("params", {}).get("vq_config", {}).get("params", {}).get("n_embed", 1024)
        if not isinstance(vq_num_codes, int) or vq_num_codes <= 0:
            print("[Warn] Could not infer codebook size from config. Fallback to 1024.")
            vq_num_codes = 1024

        gpt_dim = self.gpt.config.n_embd
        print(f"[Info] codebook size(K)={vq_num_codes}, GPT2 dim={gpt_dim}, num_prefix={num_prefix}")
        self.adapter = VQPrefixAdapter(vq_num_codes, gpt_dim, num_prefix=num_prefix, use_mlp=use_adapter_mlp)

    @torch.no_grad()
    def get_vq_indices(self, images: torch.Tensor) -> torch.Tensor:
        """
        DINOv2 encoder output(패치 토큰)을 VQ proj Conv2d가 먹는 [B,C,H,W]로 변환 후
        quantizer에서 indices를 뽑아 [B, L]로 반환.
        """
        self.rae.eval()

        enc = self.rae.encoder(images)

        if isinstance(enc, dict):
            for k in ["x_norm_patchtokens", "patch_tokens", "tokens", "x"]:
                if k in enc and torch.is_tensor(enc[k]):
                    enc = enc[k]
                    break
            else:
                for v in enc.values():
                    if torch.is_tensor(v):
                        enc = v
                        break

        if isinstance(enc, (tuple, list)):
            cand = None
            for t in enc:
                if torch.is_tensor(t) and t.dim() >= 3 and t.size(-1) in (768, 1024):
                    cand = t
                    break
            enc = cand if cand is not None else enc[0]

        if not torch.is_tensor(enc):
            raise RuntimeError(f"Encoder output is not a tensor. type={type(enc)}")

        x = enc

        if x.dim() == 4 and x.size(-1) in (768, 1024):
            x = x[:, 0, :, :]

        if x.dim() != 3:
            raise RuntimeError(f"Unexpected patch token tensor shape: {tuple(x.shape)} (need [B,N,C])")

        B, N, C = x.shape
        H = int(math.sqrt(N))
        if H * H != N:
            raise RuntimeError(f"Patch token count N={N} is not a square. Can't reshape to HxW grid.")

        x = x.transpose(1, 2).contiguous().view(B, C, H, H)

        proj = None
        if hasattr(self.rae, "vq_pre") and isinstance(self.rae.vq_pre, nn.Conv2d):
            proj = self.rae.vq_pre
        elif hasattr(self.rae, "vq_layer") and isinstance(self.rae.vq_layer, nn.Conv2d):
            proj = self.rae.vq_layer

        if proj is not None:
            x = proj(x)

        quant = None
        for name in ["quantize", "vq", "vector_quantizer", "codebook", "quantizer", "vq_layer"]:
            if hasattr(self.rae, name) and isinstance(getattr(self.rae, name), nn.Module):
                m = getattr(self.rae, name)
                if not isinstance(m, nn.Conv2d):
                    quant = m
                    break
        if quant is None:
            raise RuntimeError("Could not find quantizer module. Check your RAE attributes (quantize/vq/codebook...).")

        out = quant(x)

        idx = None
        if isinstance(out, (tuple, list)):
            for t in out:
                if torch.is_tensor(t) and t.dtype in (torch.int64, torch.int32):
                    idx = t
                    break
            if idx is None and len(out) >= 3:
                info = out[-1]
                if isinstance(info, (tuple, list)) and len(info) >= 3 and torch.is_tensor(info[2]):
                    idx = info[2]
                elif isinstance(info, dict):
                    for k in ["indices", "index", "codes"]:
                        if k in info and torch.is_tensor(info[k]):
                            idx = info[k]
                            break
        elif isinstance(out, dict):
            for k in ["indices", "index", "codes"]:
                if k in out and torch.is_tensor(out[k]):
                    idx = out[k]
                    break

        if idx is None:
            raise RuntimeError("Quantizer output didn't contain indices. Inspect quant(x) output structure.")

        idx = idx.long()
        if idx.dim() == 3:
            idx = idx.view(idx.size(0), -1)
        else:
            idx = idx.view(idx.size(0), -1)

        return idx

    def forward(self, images: torch.Tensor, captions: List[str]) -> torch.Tensor:
        device = images.device

        with torch.no_grad():
            img_idx = self.get_vq_indices(images).to(device)

        img_embeds = self.adapter(img_idx)

        tok = self.tokenizer(
            captions,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
        )
        input_ids = tok["input_ids"].to(device)
        attn_mask = tok["attention_mask"].to(device)

        with torch.no_grad():
            txt_embeds = self.gpt.transformer.wte(input_ids)

        inputs_embeds = torch.cat([img_embeds, txt_embeds], dim=1)

        B = images.size(0)
        L_img = img_embeds.size(1)

        labels = torch.cat(
            [
                torch.full((B, L_img), -100, device=device, dtype=torch.long),
                input_ids,
            ],
            dim=1,
        )

        img_mask = torch.ones((B, L_img), device=device, dtype=attn_mask.dtype)
        full_mask = torch.cat([img_mask, attn_mask], dim=1)

        outputs = self.gpt(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=labels)
        return outputs.loss

    @torch.no_grad()
    def generate(self, images: torch.Tensor, prompt: str = "", max_new_tokens: int = 30):
        device = images.device
        self.eval()

        B = images.size(0)

        img_idx = self.get_vq_indices(images).to(device)
        img_embeds = self.adapter(img_idx)

        prompt_texts = [prompt] * B if prompt.strip() != "" else [""] * B

        tok = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32,
        ).to(device)

        input_ids = tok["input_ids"]
        attn_mask = tok["attention_mask"]

        txt_embeds = self.gpt.transformer.wte(input_ids)
        inputs_embeds = torch.cat([img_embeds, txt_embeds], dim=1)

        img_mask = torch.ones((B, img_embeds.size(1)), device=device, dtype=attn_mask.dtype)
        full_mask = torch.cat([img_mask, attn_mask], dim=1)

        out_ids = self.gpt.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        return [self.tokenizer.decode(out_ids[i], skip_special_tokens=True) for i in range(B)]


# =========================
# 3) Dataset (Flickr30k CSV)
# =========================
class Flickr30kCSVDataset(Dataset):
    def __init__(
        self,
        root="/dataset/flickr30k",
        split="train",
        image_size=256,
        norm_mode: str = "imagenet",
    ):
        self.root = root
        self.split = split
        self.img_dir = os.path.join(root, "flickr30k-images")
        self.csv_path = os.path.join(root, "flickr_annotations_30k.csv")

        df = pd.read_csv(self.csv_path)
        df = df[df["split"] == split].reset_index(drop=True)

        df["path"] = df["filename"].apply(lambda x: os.path.join(self.img_dir, x))
        df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)

        self.df = df
        self.transform = build_transform(image_size=image_size, norm_mode=norm_mode)

        print(f"[Dataset] split={split} | N={len(self.df)} | norm_mode={norm_mode}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["path"]

        caps = ast.literal_eval(row["raw"])
        caption = random.choice(caps)

        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, caption


# =========================
# 3.5) Tiny Dataset for Overfitting Test
# =========================
class TinyDataset(Dataset):
    """학습이 제대로 되는지 확인용 - 작은 데이터로 overfit 테스트"""
    def __init__(self, full_dataset, n=10):
        self.ds = full_dataset
        self.indices = list(range(min(n, len(full_dataset))))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.ds[self.indices[idx]]


# =========================
# 4) Train Loop
# =========================
@dataclass
class TrainConfig:
    rae_ckpt_path: str
    rae_config_path: str
    gpt_name: str = "gpt2"
    data_root: str = "/dataset/flickr30k"
    image_size: int = 256
    train_split: str = "train"
    val_split: str = "val"
    norm_mode: str = "imagenet"  # ✅ 핵심: RAE가 DINOv2 기반이면 imagenet

    batch_size: int = 8  # ✅ 증가
    lr: float = 1e-4  # ✅ 감소 (더 안정적)
    epochs: int = 5  # ✅ 증가
    max_text_len: int = 32  # ✅ 감소 (짧은 캡션에 집중)
    use_adapter_mlp: bool = True
    num_prefix: int = 16  # ✅ Cross-Attention으로 압축 (324 -> 16 tokens)
    grad_clip: Optional[float] = 1.0

    log_every: int = 20  # ✅ 자주 로그
    gen_every: int = 200  # ✅ 자주 생성 확인
    save_every: int = 1000
    out_dir: str = "adapter_ckpts"
    resume_path: Optional[str] = None

    # ✅ 디버깅 옵션
    debug_vq: bool = False  # VQ 인덱스 출력
    overfit_test: bool = False  # 작은 데이터로 overfit 테스트
    overfit_n: int = 10  # overfit 테스트 샘플 수

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    num_workers: int = 4


def train(cfg: TrainConfig):
    set_seed(cfg.seed)

    print(f"\n{'='*60}")
    print(f"[Config] norm_mode = {cfg.norm_mode}")
    print(f"[Config] lr = {cfg.lr}, epochs = {cfg.epochs}, batch_size = {cfg.batch_size}")
    print(f"[Config] max_text_len = {cfg.max_text_len}")
    print(f"[Config] overfit_test = {cfg.overfit_test}")
    print(f"{'='*60}\n")

    model = VQToGPT2(
        rae_ckpt_path=cfg.rae_ckpt_path,
        rae_config_path=cfg.rae_config_path,
        gpt_model_name=cfg.gpt_name,
        max_text_len=cfg.max_text_len,
        use_adapter_mlp=cfg.use_adapter_mlp,
        num_prefix=cfg.num_prefix,  # ✅ NEW
        debug_vq_return=False,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=cfg.lr)

    start_epoch = 0
    global_step = 0
    if cfg.resume_path is not None and os.path.isfile(cfg.resume_path):
        start_epoch, global_step = load_checkpoint(cfg.resume_path, model, optimizer)
        print(f"[Resume] Loaded {cfg.resume_path} (epoch={start_epoch}, step={global_step})")

    train_ds = Flickr30kCSVDataset(
        root=cfg.data_root,
        split=cfg.train_split,
        image_size=cfg.image_size,
        norm_mode=cfg.norm_mode,
    )

    # ✅ Overfitting 테스트 모드
    if cfg.overfit_test:
        print(f"\n[WARNING] OVERFITTING TEST MODE: Using only {cfg.overfit_n} samples!")
        print(f"[INFO] Loss should drop to < 0.5 if training works correctly.\n")
        train_ds = TinyDataset(train_ds, n=cfg.overfit_n)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    val_ds = Flickr30kCSVDataset(
        root=cfg.data_root,
        split=cfg.val_split,
        image_size=cfg.image_size,
        norm_mode=cfg.norm_mode,
    )
    val_dl = DataLoader(val_ds, batch_size=2, shuffle=True, num_workers=0)
    val_iter = iter(val_dl)

    from collections import deque
    ma = deque(maxlen=100)

    model.train()
    print("[Train] Start adapter training...\n")

    for epoch in range(start_epoch, cfg.epochs):
        running = 0.0

        for imgs, caps in train_dl:
            imgs = imgs.to(cfg.device, non_blocking=True)
            caps = list(caps)

            # ✅ 첫 배치에서 VQ 인덱스 디버깅
            if cfg.debug_vq and global_step == 0:
                with torch.no_grad():
                    img_idx = model.get_vq_indices(imgs)
                    print(f"\n[Debug VQ] shape: {img_idx.shape}")
                    print(f"[Debug VQ] range: [{img_idx.min().item()}, {img_idx.max().item()}]")
                    print(f"[Debug VQ] unique codes: {img_idx.unique().numel()}")
                    print(f"[Debug VQ] example indices: {img_idx[0, :10].tolist()}\n")

            optimizer.zero_grad(set_to_none=True)
            loss = model(imgs, caps)
            loss.backward()

            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), cfg.grad_clip)

            optimizer.step()

            loss_val = float(loss.item())
            running += loss_val
            ma.append(loss_val)

            if cfg.log_every > 0 and global_step % cfg.log_every == 0:
                ma100 = sum(ma) / len(ma) if len(ma) > 0 else loss_val
                print(f"Epoch {epoch} | Step {global_step:5d} | Loss {loss_val:.4f} | MA100 {ma100:.4f}")

            if cfg.gen_every > 0 and global_step % cfg.gen_every == 0 and global_step > 0:
                model.eval()
                try:
                    vimgs, vcaps = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_dl)
                    vimgs, vcaps = next(val_iter)
                
                vimgs = vimgs.to(cfg.device)
                gens = model.generate(vimgs[:2], prompt="A photo of", max_new_tokens=20)
                
                print(f"\n{'='*60}")
                print(f"[Gen @ step {global_step}]")
                print(f"  GT:  {vcaps[0]}")
                print(f"  Gen: {gens[0]}")
                if len(gens) > 1 and len(vcaps) > 1:
                    print(f"  GT:  {vcaps[1]}")
                    print(f"  Gen: {gens[1]}")
                print(f"{'='*60}\n")
                
                model.train()

            if cfg.save_every > 0 and global_step > 0 and global_step % cfg.save_every == 0:
                path = os.path.join(cfg.out_dir, f"adapter_step{global_step:07d}.pt")
                save_checkpoint(path, model, optimizer, epoch, global_step, cfg_dict=cfg.__dict__)
                print(f"[Save] {path}")

            global_step += 1

        epoch_avg = running / max(1, len(train_dl))
        print(f"\n[Epoch {epoch}] avg loss = {epoch_avg:.4f}\n")

        # Epoch 끝날 때마다 저장
        path = os.path.join(cfg.out_dir, f"adapter_epoch{epoch:03d}.pt")
        save_checkpoint(path, model, optimizer, epoch, global_step, cfg_dict=cfg.__dict__)
        print(f"[Save] {path}\n")

    # 최종 저장
    final_path = os.path.join(cfg.out_dir, "adapter_last.pt")
    save_checkpoint(final_path, model, optimizer, cfg.epochs - 1, global_step, cfg_dict=cfg.__dict__)
    print(f"\n[Save] {final_path}")
    print("[Train] Done! 🎉\n")


if __name__ == "__main__":
    # ✅ 환경변수로 설정 가능
    norm_mode = os.environ.get("NORM_MODE", "imagenet")
    overfit_test = os.environ.get("OVERFIT_TEST", "0") == "1"
    out_dir = os.environ.get("OUT_DIR", f"adapter_ckpts/flickr30k_v3_{norm_mode}")

    cfg = TrainConfig(
        rae_ckpt_path="pca_results/008-RAE/checkpoints/1280000.pt",
        rae_config_path="configs/stage1/training/DINOv2-B_decXL.yaml",
        
        # ✅ 핵심 설정
        norm_mode=norm_mode,  # imagenet (DINOv2 사용 시)
        
        # ✅ 학습 설정
        epochs=5,
        batch_size=8,
        lr=1e-4,
        max_text_len=32,
        num_prefix=16,  # ✅ 324 -> 16 tokens (Cross-Attention)
        
        # ✅ 로깅
        log_every=20,
        gen_every=200,
        save_every=1000,
        
        # ✅ 출력
        out_dir=out_dir,
        resume_path=None,
        
        # ✅ 디버깅
        debug_vq=True,  # 첫 배치에서 VQ 인덱스 확인
        overfit_test=overfit_test,  # 작은 데이터로 테스트
        overfit_n=10,
    )
    
    train(cfg)