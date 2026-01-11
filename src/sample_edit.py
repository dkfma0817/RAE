import sys
import os
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision import transforms
from torchvision.utils import save_image
from omegaconf import OmegaConf
from tqdm import tqdm

from models.t2i_model import RAE_T2I_Model
try:
    from src.utils.model_utils import instantiate_from_config
except ImportError:
    from utils.model_utils import instantiate_from_config

# ==========================================
# 1. RAE 인코더/디코더 로드
# ==========================================
def load_rae(config_path, device):
    print(f"🧊 Loading RAE from {config_path}")
    config = OmegaConf.load(config_path)
    rae = instantiate_from_config(config.stage_1).to(device)
    rae.eval()
    return rae

# ==========================================
# 2. 이미지 로드 및 전처리 함수
# ==========================================
def load_image(image_path, size=224):
    print(f"🖼️ Loading image: {image_path}")
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        # DINOv2 학습 시 사용된 정규화
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0) # [1, 3, H, W]

# ==========================================
# 3. SDEdit Sampler (핵심!)
# ==========================================
@torch.no_grad()
def sample_sdedit(model, z_source, prompt, strength=0.6, steps=50):
    """
    z_source: 원본 이미지의 Latent
    prompt: 텍스트 조건 (단일 문자열)
    strength: 원본을 얼마나 바꿀지 (0.0 = 안 바꿈, 1.0 = 완전 새로 생성)
    """
    device = z_source.device
    batch_size = z_source.shape[0]
    
    # 1. 시작 시간 계산 (t=strength 시점부터 시작)
    # strength가 0.6이면, 노이즈가 60% 섞인 시점부터 복구 시작
    start_step = int(steps * (1 - strength)) # 역순이라 1-strength
    
    # 2. 노이즈 섞기 (Forward Process)
    # Flow Matching: z_t = (1-t) * z_0 + t * noise
    noise = torch.randn_like(z_source)
    t_start = strength
    z_noisy = (1 - t_start) * z_source + t_start * noise
    
    print(f"🎨 Editing with prompt: '{prompt}' (Strength: {strength})")
    
    z_t = z_noisy
    
    # 3. Denoising Loop (중간부터 시작!)
    # strength 시점부터 0까지 깎아내려감
    total_steps = int(steps * strength)
    
    for i in tqdm(range(total_steps)):
        # 현재 시간 t (strength -> ... -> 0)
        num_t = strength - (i / steps)
        t_input = torch.full((batch_size,), num_t, device=device, dtype=torch.float32)
        
        # 모델 예측
        velocity = model(z_t, t_input, [prompt])
        
        # Euler Step
        dt = 1.0 / steps
        z_t = z_t - velocity * dt
        
    return z_t

# ==========================================
# 4. 실행 함수
# ==========================================
def run_editing():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 설정 (여기를 바꾸세요!) ---
    image_path = "../RL/data/train2014/COCO_train2014_000000579758.jpg" 
    # init_image = load_image(img_url).convert("RGB")
    prompt = "What is the player doing?"  # 바꾸고 싶은 텍스트
    edit_strength = 0.5    # 0.5 ~ 0.8 추천 (클수록 많이 바뀜)
    
    checkpoint_path = "checkpoints/t2i_experiment/epoch_1.pt"
    rae_config_path = "configs/stage1/pretrained/DINOv2-B.yaml"
    
    # 1. 모델 로드
    print("🔥 Loading Models...")
    model = RAE_T2I_Model(llm_name="Qwen/Qwen2.5-1.5B", train_llm=False, input_size=16, in_channels=768).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    rae = load_rae(rae_config_path, device)
    
    # 2. 이미지 로드 -> Latent 변환
    if not os.path.exists(image_path):
        print(f"❌ Error: {image_path} 파일이 없습니다. 강아지 사진을 구해서 넣어주세요!")
        return

    img_tensor = load_image(image_path).to(device)
    with torch.no_grad():
        # RAE 인코더 통과
        z_source = rae.encode(img_tensor)
        if isinstance(z_source, tuple): z_source = z_source[0]
        
        # 차원 변환 (B, L, C) -> (B, C, H, W)
        if len(z_source.shape) == 3:
            B, L, C = z_source.shape
            H = W = int(L ** 0.5)
            z_source = z_source.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    # 3. 편집 (Editing) 수행
    z_edited = sample_sdedit(model, z_source, prompt, strength=edit_strength)
    
    # 4. 결과 복원 및 저장
    print("🖨️ Decoding...")
    with torch.no_grad():
        img_edited = rae.decode(z_edited)
        img_recon = rae.decode(z_source) # 원본 복원 (비교용)
    
    # 원본(복원)과 편집본을 나란히 저장
    final_img = torch.cat([img_recon, img_edited], dim=3) # 옆으로 붙이기
    save_image(final_img, "editing_result.png", normalize=True, value_range=(-1, 1))
    print(f"✅ Result saved to editing_result.png (Left: Original, Right: Edited)")

if __name__ == "__main__":
    run_editing()