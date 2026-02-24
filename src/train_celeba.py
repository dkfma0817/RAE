import os, json, math
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from omegaconf import OmegaConf
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 너 repo 구조에 맞춰 import
try:
    from utils.model_utils import instantiate_from_config
except ImportError:
    from src.utils.model_utils import instantiate_from_config

from stage1 import RAE
from models.t2i_model import RAE_T2I_Model

from transformers import AutoTokenizer


# -----------------------------
# Dataset
# -----------------------------
class CelebAJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str, transform=None):
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.items.append(json.loads(line))
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ex = self.items[idx]
        img = Image.open(ex["image"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        prompt = ex["prompt"]
        return img, prompt


def collate_fn(batch):
    imgs, prompts = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    prompts = list(prompts)
    return imgs, prompts


# -----------------------------
# helpers
# -----------------------------
def load_stage1_rae(stage1_yaml: str, device: str) -> RAE:
    cfg = OmegaConf.load(stage1_yaml)
    rae: RAE = instantiate_from_config(cfg.stage_1).to(device)
    rae.eval()
    for p in rae.parameters():
        p.requires_grad = False
    return rae


# @torch.no_grad()
# def encode_text_siglip(tokenizer, text_encoder, projector, prompts: List[str], device: str, max_len: int = 64):
#     """
#     prompts -> token -> SigLIP text encoder -> pooled embedding -> projector to DiT hidden
#     이 함수는 네 RAE_T2I_Model 내부 구조에 따라 약간 달라질 수 있는데,
#     여기서는 'text_encoder + projector'가 모델 안에 있다고 가정하고,
#     text_encoder output에서 pooled vec 하나를 뽑아 projector로 보내는 형태로 작성.
#     """
#     tok = tokenizer(
#         prompts,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=max_len,
#     )
#     tok = {k: v.to(device) for k, v in tok.items()}

#     out = text_encoder(**tok)

#     # HuggingFace SigLIP 텍스트 모델은 보통 out.pooler_output 또는 out.last_hidden_state 사용 가능
#     if hasattr(out, "pooler_output") and out.pooler_output is not None:
#         pooled = out.pooler_output
#     else:
#         # 마지막 토큰 평균 같은 간단 pooling
#         pooled = out.last_hidden_state.mean(dim=1)

#     text_emb = projector(pooled)  # (B, cond_dim==encoder_hidden_size)
#     return text_emb


# -----------------------------
# main train
# -----------------------------
@dataclass
class TrainCfg:
    device: str = "cuda"
    # data
    train_jsonl: str = "celeba_data/t2i_jsonl/train.jsonl"
    image_size: int = 256  # RAE가 기대하는 해상도에 맞추기 (보통 256)
    num_workers: int = 4

    # model
    stage1_yaml: str = "configs/stage1/pretrained/SigLIP2.yaml"  # 너가 쓰는 stage1 config
    siglip_name = "google/siglip2-base-patch16-256"
    train_text_encoder: bool = False  # SigLIP freeze

    # optim
    batch_size: int = 8
    lr: float = 5e-5
    epochs: int = 20
    grad_accum: int = 1
    max_norm: float = 1.0

    # flow
    t_eps: float = 1e-3
    latent_scale: float = 0.5
    text_drop_prob: float = 0.2 # CFG 학습용


def train(cfg: TrainCfg):
    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1) data
    transform = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.CenterCrop(cfg.image_size),
        transforms.ToTensor(),
        # ⚠️ RAE 입력 분포에 따라 [-1,1] 필요하면 아래 주석 해제
        # transforms.Lambda(lambda x: x * 2 - 1),
    ])
    dataset = CelebAJsonlDataset(cfg.train_jsonl, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    print(f"Train samples: {len(dataset):,}")

    # 2) stage1 RAE
    rae = load_stage1_rae(cfg.stage1_yaml, device)
    # latent_dim 얻기 (네 RAE 구현마다 이름 다를 수 있어 방어)
    rae_dim = getattr(rae, "latent_dim", None)
    if rae_dim is None:
        # encode 한 번 해보고 채널 잡기
        with torch.no_grad():
            x0, _ = next(iter(loader))
            x0 = x0.to(device)
            z0 = rae.encode(x0)
            if isinstance(z0, (tuple, list)):
                z0 = z0[0]
            # z0가 (B,C,H,W)면 C를 latent_dim으로, (B,N,C)면 C를 latent_dim으로
            rae_dim = z0.shape[-1] if z0.dim() == 3 else z0.shape[1]
    print(f"RAE latent dim: {rae_dim}")

    # 3) stage2 model (DiT/DDT + text conditioner)
    model = RAE_T2I_Model(
        siglip_name=cfg.siglip_name,
        train_text_encoder=cfg.train_text_encoder,
        # 아래 파라미터들은 네 모델 정의에 맞춰 유지
        input_size=16,            # 🔥 여기 정확히 맞추려면 z0 shape 확인해서 수정해야 함
        in_channels=rae_dim,
        hidden_size=[1152, 2048],
        depth=[28, 2],
        num_heads=[16, 16],
    ).to(device)
    model.train()

    # 4) tokenizer + text encoder handle
    # tokenizer = AutoTokenizer.from_pretrained(cfg.siglip_name)
    # # 모델 내부에 text encoder / projector가 있다고 가정
    # # (네 RAE_T2I_Model 구조에 맞게 attribute 이름만 맞추면 됨)
    # # text_encoder = model.text_conditioner.text_encoder
    # # projector = model.text_conditioner.projector  # text -> cond_dim projection (너가 만들어둔 proj)

    # # freeze 확인
    # if not cfg.train_text_encoder:
    #     text_encoder.eval()
    #     for p in text_encoder.parameters():
    #         p.requires_grad = False

    # 5) optimizer (trainable만)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(params, lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    criterion = nn.MSELoss()

    # 6) train loop
    global_step = 0
    opt.zero_grad(set_to_none=True)

    for ep in range(cfg.epochs):
        pbar = tqdm(loader, desc=f"epoch {ep}")
        running = 0.0
        for step, (images, prompts) in enumerate(pbar):
            images = images.to(device, non_blocking=True)

            # (A) 이미지 -> latent
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    z0 = rae.encode(images)
                    if isinstance(z0, (tuple, list)):
                        z0 = z0[0]
                    z0 = z0 * cfg.latent_scale

            B = z0.shape[0]
            t = cfg.t_eps + (1 - 2 * cfg.t_eps) * torch.rand(B, device=device)
            t_expand = t.view(B, *([1] * (z0.dim() - 1)))

            eps = torch.randn_like(z0)
            zt = (1 - t_expand) * z0 + t_expand * eps
            target_v = eps - z0

            # (B) text dropout (CFG 학습용 uncond 섞기)
            if cfg.text_drop_prob > 0:
                drop = (torch.rand(B, device=device) < cfg.text_drop_prob).tolist()
                prompts = ["" if d else s for d, s in zip(drop, prompts)]

            # (C) text -> text_emb (루프에서 만들고 model에는 임베딩 전달)
            # with torch.no_grad():  # text encoder freeze면 no_grad로 가는 게 효율적
            #     text_emb = encode_text_siglip(tokenizer, text_encoder, projector, prompts, device)

            # (D) forward + loss
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                pred_v = model(zt, t, prompts)
                loss = criterion(pred_v, target_v)

            # 🔥 추가
            if not torch.isfinite(loss):
                print(f"[WARN] Non-finite loss detected. Skipping step. loss={loss.item()}")
                opt.zero_grad(set_to_none=True)
                continue


            loss = loss / cfg.grad_accum
            scaler.scale(loss).backward()

            if (step + 1) % cfg.grad_accum == 0:
                scaler.unscale_(opt)
                if cfg.max_norm and cfg.max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, cfg.max_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                global_step += 1

            running += loss.item() * cfg.grad_accum
            pbar.set_postfix(loss=running / (step + 1))

        # epoch ckpt
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_path = f"checkpoints/please_{ep}.pt"
        torch.save({"model": model.state_dict(), "epoch": ep}, ckpt_path)
        print(f"Saved: {ckpt_path}")

    print("Done.")


if __name__ == "__main__":
    cfg = TrainCfg()
    train(cfg)
