import sys
import os

# [중요] 프로젝트 루트(RAE 폴더)를 경로에 추가하여 모듈 import 에러 방지
PROJECT_ROOT = "/home01/x3098a02/x3098a02/RAE"
sys.path.append(PROJECT_ROOT)

import torch
import webdataset as wds
from torchvision import transforms
from tqdm import tqdm
from omegaconf import OmegaConf
from src.utils.model_utils import instantiate_from_config
import io

def get_rae_encoder(config_path, device):
    print(f"🧊 Loading RAE Config from: {config_path}")
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.stage_1).to(device)
    model.eval()
    return model

def preprocess():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ==================================================================
    # [설정] 서버 경로에 맞게 수정됨
    # ==================================================================
    
    # 1. 입력 데이터 패턴 (0000 ~ 2175)
    # braceexpand가 설치되어 있어야 {a..b} 문법이 작동합니다. 
    # 혹시 에러가 나면 pip install braceexpand 하세요.
    input_pattern = "/home01/x3098a02/dataset/cc12m/cc12m-train-{0000..2175}.tar"
    
    # 2. 출력 저장 경로 (dataset 폴더 옆에 latents 폴더 생성)
    output_dir = "/home01/x3098a02/dataset/cc12m_latents"
    
    # 3. RAE 설정 파일 경로 (절대 경로로 지정)
    rae_config_path = os.path.join(PROJECT_ROOT, "configs/stage1/pretrained/DINOv2-B.yaml")
    
    # ==================================================================

    os.makedirs(output_dir, exist_ok=True)
    print(f"Input: {input_pattern}")
    print(f"Output: {output_dir}")

    # 1. 모델 로드
    rae = get_rae_encoder(rae_config_path, device)

    # 2. 전처리 파이프라인 (DINOv2/SigLIP용)
    transform = transforms.Compose([
        transforms.Resize(224), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. 데이터셋 로드 (안전장치 추가)
    dataset = (
        wds.WebDataset(input_pattern, handler=wds.warn_and_continue) # 깨진 파일 무시
        .decode("pil", handler=wds.warn_and_continue)                # 깨진 이미지 무시
        .to_tuple("__key__", "jpg", "json")
        # .map_tuple(transform, lambda x: x) # 여기서 변환하면 느릴 수 있어 배치 처리로 넘김
    )
    
    # 서버 사양에 맞춰 workers와 batch를 늘림
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=64,      # GPU 메모리 넉넉하면 64~128 추천
        num_workers=8,      # CPU 코어 수에 따라 조절 (8~16 추천)
        collate_fn=None,
        pin_memory=True
    )

    # 4. 저장용 Writer
    # 1000개씩 묶어서 저장 (파일명: cc12m-latents-00000.tar ...)
    sink = wds.ShardWriter(f"{output_dir}/cc12m-latents-%05d.tar", maxcount=1000)

    print("🚀 Start Extracting Latents... (This will take a while)")
    
    total_processed = 0
    
    # tqdm에 total을 줄 수 없으므로(개수 모름), 그냥 진행합니다.
    for keys, images, jsons in tqdm(dataloader):
        try:
            # 이미지 전처리 & GPU 이동
            img_tensors = torch.stack([transform(img) for img in images]).to(device)
            
            with torch.no_grad():
                # Latent 추출
                latents = rae.encode(img_tensors)
                if isinstance(latents, tuple): latents = latents[0]
                
                # (B, L, C) -> (B, C, H, W) 변환
                # DINOv2: (B, 256, 768) -> (B, 768, 16, 16)
                if len(latents.shape) == 3:
                    B, L, C = latents.shape
                    H = W = int(L**0.5)
                    latents = latents.view(B, H, W, C).permute(0, 3, 1, 2).cpu()
                else:
                    latents = latents.cpu()

            # 저장
            for i in range(len(keys)):
                # Latent를 Bytes로 변환 (.pth 포맷)
                buffer = io.BytesIO()
                torch.save(latents[i].clone(), buffer) # clone으로 메모리 이슈 방지
                
                sample = {
                    "__key__": keys[i],
                    "pth": buffer.getvalue(), 
                    "json": jsons[i]
                }
                sink.write(sample)
            
            total_processed += len(keys)
            
        except Exception as e:
            print(f"Error in batch: {e}")
            continue

    sink.close()
    print(f"Done! Total {total_processed} latents saved to {output_dir}")

if __name__ == "__main__":
    preprocess()