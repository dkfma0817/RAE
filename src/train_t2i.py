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

# 1. 모델 임포트 (이제 'models'를 바로 찾을 수 있습니다!)
from models.t2i_model import RAE_T2I_Model

# 2. RAE 관련 모듈 임포트
# (utils가 src 안에 있다면 src.utils, RAE 바로 아래라면 utils로 시작)
# 보통 RAE 구조상 src 안에 utils가 있다면 아래처럼 씁니다.
try:
    from src.utils.model_utils import instantiate_from_config
except ImportError:
    # 만약 실행 위치에 따라 src를 빼야 한다면
    from utils.model_utils import instantiate_from_config

# ==========================================
# 1. RAE 인코더 로드 함수 (추가됨!)
# ==========================================
def load_rae_encoder(config_path, device):
    print(f"🧊 Loading RAE Config from {config_path}")
    config = OmegaConf.load(config_path)
    
    # RAE 모델 생성 (Stage 1)
    rae = instantiate_from_config(config.stage_1).to(device)
    rae.eval() # 무조건 평가 모드 (Frozen)
    
    # 인코더만 분리 (Forward 함수 래핑)
    # RAE 클래스 구조에 따라 rae.encode()가 있을 수도 있고 없을 수도 있음.
    # 보통은 rae.encode(x)가 Latent를 뱉음.
    return rae

# ==========================================
# 2. 데이터 로더 (이미지 전처리 포함)
# ==========================================
def get_cc12m_loader(tar_path_pattern, batch_size=4, num_workers=4):
    # RAE(DINOv2) 입력 크기인 224x224로 맞춤 (설정 파일의 encoder_input_size 참고)
    transform = transforms.Compose([
        transforms.Resize(224), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        # DINOv2는 보통 ImageNet Mean/Std를 씁니다. (0.5 정규화 아님!)
    ])

    dataset = (
        wds.WebDataset(tar_path_pattern)
        .shuffle(1000)
        .decode("pil")
        .to_tuple("jpg", "json")
        .map_tuple(transform, lambda x: x["caption"])
    )

    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    ).with_length(5000)
    
    return loader

# ==========================================
# 3. 메인 학습 함수
# ==========================================
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4 # 메모리 확인하며 조절
    lr = 1e-4
    epochs = 1
    
    # 경로 설정 (사용자 환경에 맞게!)
    tar_pattern = "../dataset/cc12m-train-{0000..0004}.tar"
    rae_config_path = "configs/stage1/pretrained/DINOv2-B.yaml" # 방금 보여주신 설정 파일 경로
    
    print(f"🚀 Training Start on {device}")

    # [1] RAE 인코더 준비 (Real!)
    try:
        rae_model = load_rae_encoder(rae_config_path, device)
        print("✅ RAE Encoder Loaded Successfully!")
    except Exception as e:
        print(f"❌ RAE Load Failed: {e}")
        return

    # [2] T2I 모델 준비 (학습 대상)
    print("🔥 Loading T2I Model...")
    model = RAE_T2I_Model(
        llm_name="Qwen/Qwen2.5-1.5B",
        train_llm=False,
        # RAE Config에서 확인한 값 적용 (DINOv2 Base 기준)
        input_size=16,    # 224 / 14 = 16
        in_channels=768   # DINOv2 Base Hidden Size
    ).to(device)
    model.train()

    # [3] 데이터 로더 & 옵티마이저
    dataloader = get_cc12m_loader(tar_pattern, batch_size=batch_size)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # [4] 학습 루프
    for epoch in range(epochs):
        pbar = tqdm(dataloader)
        for step, (images, texts) in enumerate(pbar):
            images = images.to(device)
            
            # --- [핵심] RAE를 통해 Latent 추출 ---
            with torch.no_grad():
                # rae.encode()가 (Latent, ..) 튜플을 뱉는지 텐서만 뱉는지 확인 필요
                # 보통 RAE 코드는 z = rae.encode(x) 형태임
                z_0 = rae_model.encode(images)
                
                # 만약 z_0가 튜플이면 분리 (예: dist.sample())
                if isinstance(z_0, tuple): z_0 = z_0[0]
                
                # 정규화 (선택): RAE Latent가 너무 크면 스케일링 필요
                # z_0 = z_0 * 0.18215 (SD 스타일) - 일단은 그냥 씀

            # --- Forward Process (노이즈 추가) ---
            t = torch.rand(z_0.shape[0], device=device)
            noise = torch.randn_like(z_0)
            
            t_expand = t.view(-1, 1, 1, 1)
            z_t = (1 - t_expand) * z_0 + t_expand * noise
            target = noise - z_0 

            # --- 모델 예측 및 업데이트 ---
            pred = model(z_t, t, list(texts))
            
            loss = criterion(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_description(f"Loss: {loss.item():.4f}")
            
    os.makedirs("checkpoints/t2i_experiment", exist_ok=True)
    save_path = f"checkpoints/t2i_experiment/epoch_{epochs}.pt"
    print(f"💾 Saving Checkpoint to {save_path}...")
    torch.save(model.state_dict(), save_path)
    print("✅ Done! Training Complete.")

if __name__ == "__main__":
    train()