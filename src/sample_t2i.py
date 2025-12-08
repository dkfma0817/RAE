import sys
import os

# 경로 설정 (필수)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision.utils import save_image
from omegaconf import OmegaConf
from tqdm import tqdm

# 모델 관련 임포트
from models.t2i_model import RAE_T2I_Model
# try:
from utils.model_utils import instantiate_from_config
# except ImportError:
#     from utils.model_utils import instantiate_from_config

# ==========================================
# 1. RAE (디코더) 로드 함수
# ==========================================
def load_rae(config_path, device):
    print(f"🧊 Loading RAE from {config_path}")
    config = OmegaConf.load(config_path)
    rae = instantiate_from_config(config.stage_1).to(device)
    rae.eval()
    return rae

# ==========================================
# 2. 간단한 Euler Sampler (노이즈 -> 이미지)
# ==========================================
@torch.no_grad()
def sample_euler(model, z, prompts, steps=50):
    """
    Flow Matching을 위한 간단한 Euler Solver
    z: 초기 노이즈 (Latent)
    prompts: 텍스트 리스트
    steps: 노이즈를 깎는 횟수
    """
    dt = 1.0 / steps  # 시간 간격
    z_t = z.clone()   # 현재 Latent
    
    print(f"🎨 Sampling for prompts: {prompts}")
    
    # t=1 (노이즈) 에서 t=0 (깨끗한 이미지)으로 이동
    for i in tqdm(range(steps)):
        # 현재 시간 t 계산 (1.0 -> ... -> 0.0)
        num_t = 1.0 - (i / steps)
        
        # 모델 입력용 t 텐서 만들기
        t_input = torch.full((z.shape[0],), num_t, device=z.device, dtype=torch.float32)
        
        # 모델이 예측한 속도(velocity) v
        # v = model(z_t, t, text)
        velocity = model(z_t, t_input, prompts)
        
        # Euler Step: z_{t-1} = z_t - v * dt
        z_t = z_t - velocity * dt
        
    return z_t # 최종 생성된 Latent (z_0)

# ==========================================
# 3. 실행 함수
# ==========================================
def run_sample():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 설정 ---
    checkpoint_path = "checkpoints/t2i_experiment/epoch_1.pt" # 저장된 파일 경로
    rae_config_path = "configs/stage1/pretrained/DINOv2-B.yaml"
    
    # 테스트할 프롬프트 (원하는 걸로 바꾸세요!)
    prompts = [
        "A photo of a cute dog",
        "A red apple on a table"
    ]
    
    # 1. T2I 모델 로드 & 체크포인트 적용
    print("🔥 Loading Trained T2I Model...")
    model = RAE_T2I_Model(
        llm_name="Qwen/Qwen2.5-1.5B",
        train_llm=False,
        input_size=16,
        in_channels=768
    ).to(device)
    
    # 저장된 가중치 불러오기
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    
    # 2. RAE 로드 (이미지 복원용)
    rae = load_rae(rae_config_path, device)
    
    # 3. 초기 노이즈 생성
    # Latent Shape: [Batch, Channel, H, W]
    batch_size = len(prompts)
    z_noise = torch.randn(batch_size, 768, 16, 16).to(device)
    
    # 4. 샘플링 (DiT가 그림 그리기)
    print("🎨 Generating Latents...")
    generated_latents = sample_euler(model, z_noise, prompts, steps=30)
    
    # 5. 디코딩 (RAE가 이미지로 인쇄)
    print("🖨️ Decoding to Pixels...")
    with torch.no_grad():
        # Latent가 (B, C, H, W)인지 (B, L, C)인지 확인 후 RAE 입력에 맞춤
        # 보통 RAE decode는 (B, C, H, W)를 받거나 알아서 처리함
        images = rae.decode(generated_latents)
        
    # 6. 저장
    save_path = "result_epoch_1.png"
    save_image(images, save_path, nrow=2, normalize=True, value_range=(-1, 1))
    print(f"✅ Image saved to {save_path}")

if __name__ == "__main__":
    run_sample()