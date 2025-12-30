import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from torchvision.utils import save_image # ★ 저장 함수 변경

# 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.t2i_model import RAE_T2I_Model
try:
    from src.utils.model_utils import instantiate_from_config
except ImportError:
    from utils.model_utils import instantiate_from_config

# ==========================================
# 1. 설정값
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "checkpoints/t2i_ep19.pt"  # 19에폭 체크포인트
stage1_yaml = "configs/stage1/pretrained/SigLIP2.yaml" 
model_name = "google/siglip-so400m-patch14-384"

# ★ 학습 때 사용한 스케일링 값 (필수!)
scale_factor = 0.1  

# 생성할 프롬프트
prompts = [
    "A cute cat sitting on a sofa",
    "A beautiful landscape of mountains and lake",
    "A red apple on the plate"
]

# ==========================================
# 2. 모델 로드
# ==========================================
def load_models():
    print("🚀 Loading RAE (Decoder)...")
    cfg = OmegaConf.load(stage1_yaml)
    rae = instantiate_from_config(cfg.stage_1).to(device)
    rae.eval()
    
    # RAE Latent Dim 확인
    rae_dim = rae.latent_dim if hasattr(rae, "latent_dim") else 768

    print("🚀 Loading DiT Model...")
    model = RAE_T2I_Model(
        siglip_name=model_name,
        train_text_encoder=False,
        input_size=16,          
        in_channels=rae_dim,
        hidden_size=[1152, 2048],
        depth=[28, 2],
        num_heads=[16, 16],
    ).to(device)
    
    print(f"📥 Loading Weights from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False) 
    model.eval()
    
    return rae, model

# ==========================================
# 3. 오일러 샘플러 (핵심 수정 포함)
# ==========================================
@torch.no_grad()
def sample(model, rae, prompt, steps=50, cfg_scale=4.0):
    # (1) 텍스트 준비
    text_inputs = [prompt]
    uncond_inputs = [""] 
    
    # (2) 초기 노이즈 생성
    # ★ 중요: RAE 테스트 성공 시 256x256이었으므로, 토큰 개수도 맞춰줍니다.
    # SigLIP Patch Size = 16 이라고 가정하면: 256 / 16 = 16 grid
    H = W = 16 
    C = rae.latent_dim 
    z = torch.randn(1, C, H, W).to(device)
    
    print(f"🎨 Generating: '{prompt}' (Shape: {z.shape})")
    
    # (3) Rectified Flow Sampling (Euler)
    dt = 1.0 / steps 
    for i in tqdm(range(steps)):
        t_value = 1.0 - i * dt 
        t = torch.tensor([t_value]).to(device)
        
        # CFG Guidance
        v_cond = model(z, t, text_inputs)
        v_uncond = model(z, t, uncond_inputs)
        v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
        
        # Update (z_next = z - v * dt)
        z = z - v_pred * dt

    # (4) 디코딩 및 후처리 (여기가 제일 중요!)
    # -------------------------------------------------
    # ★ [필수] 학습 때 0.1 곱했으니, 0.1로 나눠서 복구!
    z = z / scale_factor 
    
    with torch.cuda.amp.autocast():
        img_tensor = rae.decode(z) 
        
    return img_tensor

# ==========================================
# 4. 실행 및 저장
# ==========================================
if __name__ == "__main__":
    rae, model = load_models()
    
    os.makedirs("outputs", exist_ok=True)
    
    for prompt in prompts:
        # 순서 주의: (model, rae)
        img_tensor = sample(model, rae, prompt, steps=50, cfg_scale=4.0)
        
        save_name = f"outputs/{prompt.replace(' ', '_')[:20]}.png"
        
        # ★ [핵심 수정] torchvision의 save_image 활용
        # normalize=True : 이미지 값의 최소~최대를 찾아서 0~1로 쫙 펴줍니다. (깨짐 방지)
        # value_range : 명시하지 않으면 자동(min-max)으로 맞춥니다.
        save_image(img_tensor, save_name, normalize=True)
        
        print(f"💾 Saved to {save_name}")