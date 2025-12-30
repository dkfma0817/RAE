import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import webdataset as wds
from torchvision import transforms
from omegaconf import OmegaConf

from models.t2i_model import RAE_T2I_Model

try:
    from src.utils.model_utils import instantiate_from_config
except ImportError:
    from utils.model_utils import instantiate_from_config


def load_stage1_encoder(config_path, device):
    cfg = OmegaConf.load(config_path)
    stage1 = instantiate_from_config(cfg.stage_1).to(device)
    stage1.eval()
    for p in stage1.parameters():
        p.requires_grad = False
    return stage1


def get_cc12m_loader(tar_path_pattern, batch_size=8, num_workers=4):
    # ★ SigLIP patch14라면 224가 가장 깔끔 (16x16=256 토큰)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = (
        wds.WebDataset(tar_path_pattern)
        .shuffle(1000)
        .decode("pil")
        .to_tuple("jpg", "json")
        .map_tuple(transform, lambda x: x["caption"])
    )

    return wds.WebLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 하이퍼파라미터 ----
    batch_size = 32
    lr = 1e-4
    epochs = 100
    accumulation_steps = 4
    text_drop_prob = 0.2            # ★ CFG 학습용 텍스트 드롭아웃
    t_eps = 1e-3                    # ★ t 끝단 살짝 피하기

    tar_pattern = "../dataset/cc12m/cc12m-train-{0000..0100}.tar"
    stage1_yaml = "configs/stage1/pretrained/SigLIP2.yaml"  # 아래 템플릿 참고

    print(f"🚀 Training Start on {device}")

    # 1) stage1 encoder(=SigLIP2wNorm 등) 로드
    # ... (이전 코드) ...

    # 1) stage1 encoder(=RAE) 로드
    enc = load_stage1_encoder(stage1_yaml, device)
    
    # [수정 포인트] enc.hidden_size -> enc.latent_dim 으로 변경
    # 로그를 보면 'latent_dim'이라는 속성이 있습니다. 이게 RAE의 출력 채널 수(예: 768)입니다.
    rae_dim = enc.latent_dim 
    
    print(f"✅ Stage1 encoder loaded. Latent Dim: {rae_dim}")

    # 2) T2I 모델 로드
    model = RAE_T2I_Model(
        siglip_name="google/siglip-so400m-patch14-384", 
        train_text_encoder=False,
        input_size=16,          # 224 / 14 = 16
        in_channels=rae_dim,    # [수정 완료] 여기에 enc.latent_dim 값을 넣습니다.
        hidden_size=[1152, 2048],
        depth=[28, 2],
        num_heads=[16, 16],
        # max_text_len=64,
    ).to(device)
    
    # ... (이후 코드 동일) ...
    model.train()

    # 3) Optimizer / Loss / AMP
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=lr)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()

    # 4) Loader
    loader = get_cc12m_loader(tar_pattern, batch_size=batch_size)

    # 5) Train
    optimizer.zero_grad()
    for ep in range(epochs):
        pbar = tqdm(loader)
        for step, (images, texts) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            texts = list(texts)

            # --- (A) 이미지 → latent z0 (B, N, C) ---
            # --- (A) RAE Latent 추출 ---
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    z0 = enc.encode(images) # [B, 256, 768]
                    if isinstance(z0, tuple): z0 = z0[0]

                    # -------------------------------------------------------
                    # [핵심 수정] Latent Scaling (Normalization)
                    # 원본 값 범위: Max ~258, Std ~3.2 -> 학습 터짐 방지
                    # -------------------------------------------------------
                    scale_factor = 0.1 # 이 값은 나중에 Inference 때도 똑같이 써야 함!
                    z0 = z0 * scale_factor


            # --- (B) Rectified Flow 샘플링 ---
            B = z0.shape[0]
            t = (t_eps + (1 - 2*t_eps) * torch.rand(B, device=device))
            epsilon = torch.randn_like(z0)

            # [수정됨] z0의 차원 수에 맞춰서 t를 늘려줍니다.
            # z0가 (B, C, H, W) 4D라면 -> (B, 1, 1, 1)
            # z0가 (B, N, C) 3D라면    -> (B, 1, 1)
            t_expand = t.view(B, *([1] * (len(z0.shape) - 1)))

            # 이제 차원이 맞으므로 에러가 나지 않습니다.
            zt = (1 - t_expand) * z0 + t_expand * epsilon
            target_v = epsilon - z0

            # --- (C) 텍스트 CFG dropout (학습에서 uncond도 같이 학습) ---
            if text_drop_prob > 0:
                mask = (torch.rand(B, device=device) < text_drop_prob).tolist()
                # null 텍스트: 빈 문자열(간단 버전)
                # 더 안정적으로 하려면 learned null embedding을 쓰는 방식으로 발전시키면 됨
                texts = ["" if m else s for m, s in zip(mask, texts)]

            # --- (D) Forward + Loss ---
            with torch.cuda.amp.autocast():
                pred_v = model(zt, t, texts)  # (B, N, C)
                loss = criterion(pred_v, target_v)

            # --- (E) Update (grad accumulation) ---
            loss_scaled = loss / accumulation_steps
            scaler.scale(loss_scaled).backward()

            if (step + 1) % accumulation_steps == 0:
                # [추가됨] 1. 먼저 스케일링을 풉니다 (Unscale)
                scaler.unscale_(optimizer)
                
                # [추가됨] 2. 그래디언트 클리핑 (보통 1.0 사용) -> 여기서 폭발을 막음!
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 3. 스텝 진행
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if step % 10 == 0:
                pbar.set_description(f"ep{ep} loss={loss.item():.4f}")

        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/t2i_ep{ep}.pt")

    print('학습 끝 ')


if __name__ == "__main__":
    train()
