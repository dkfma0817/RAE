import sys
import os

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

# 이미지는 텐서로 쌓고(stack), 나머지(Key, JSON)는 리스트로 둡니다.
def custom_collate(batch):
    keys, imgs, jsons = zip(*batch)
    # 이미지는 이미 텐서로 변환되었으므로 stack 사용
    imgs = torch.stack(imgs)
    return list(keys), imgs, list(jsons)

def preprocess():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 설정
    input_pattern = "/home01/x3098a02/dataset/cc12m/cc12m-train-{0000..2175}.tar"
    output_dir = "/home01/x3098a02/dataset/cc12m_latents"
    rae_config_path = os.path.join(PROJECT_ROOT, "configs/stage1/pretrained/DINOv2-B.yaml")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f" Input: {input_pattern}")
    print(f" Output: {output_dir}")

    # 1. 모델 로드
    rae = get_rae_encoder(rae_config_path, device)

    # 2. 전처리 파이프라인
    transform = transforms.Compose([
        transforms.Resize(224), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # [수정] 데이터 변환 함수 (워커 프로세스에서 실행됨 -> 속도 향상)
    def process_sample(sample):
        key, img, json_data = sample
        # 여기서 미리 텐서로 변환합니다
        return key, transform(img), json_data

    # 3. 데이터셋 로드
    dataset = (
        wds.WebDataset(input_pattern, handler=wds.warn_and_continue)
        .decode("pil", handler=wds.warn_and_continue)
        .to_tuple("__key__", "jpg", "json")
        .map(process_sample) # [중요] 변환을 여기서 수행
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=64,
        num_workers=8,
        collate_fn=custom_collate, # [중요] 커스텀 collate 적용
        pin_memory=True
    )

    # 4. 저장용 Writer
    sink = wds.ShardWriter(f"{output_dir}/cc12m-latents-%05d.tar", maxcount=1000)

    print(" Start Extracting Latents... (This will take a while)")
    
    total_processed = 0
    
    for keys, img_tensors, jsons in tqdm(dataloader):
        try:
            # [수정] 이미 텐서로 변환되어 넘어오므로 바로 GPU로 보냄
            img_tensors = img_tensors.to(device)
            
            with torch.no_grad():
                # Latent 추출
                latents = rae.encode(img_tensors)
                if isinstance(latents, tuple): latents = latents[0]
                
                # (B, L, C) -> (B, C, H, W) 변환
                if len(latents.shape) == 3:
                    B, L, C = latents.shape
                    H = W = int(L**0.5)
                    latents = latents.view(B, H, W, C).permute(0, 3, 1, 2).cpu()
                else:
                    latents = latents.cpu()

            # 저장
            for i in range(len(keys)):
                buffer = io.BytesIO()
                torch.save(latents[i].clone(), buffer)
                
                sample = {
                    "__key__": keys[i],
                    "pth": buffer.getvalue(), 
                    "json": jsons[i]
                }
                sink.write(sample)
            
            total_processed += len(keys)
            
        except Exception as e:
            print(f" Error in batch: {e}")
            continue

    sink.close()
    print(f" Done! Total {total_processed} latents saved to {output_dir}")

if __name__ == "__main__":
    preprocess()