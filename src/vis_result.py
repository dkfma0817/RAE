import torch
import torchvision.transforms as T
from PIL import Image
from omegaconf import OmegaConf
from utils.model_utils import instantiate_from_config
import matplotlib.pyplot as plt
import os
import glob

# ================= 사용자가 수정할 부분 =================
config_path = "configs/stage1/training/DINOv2-B_decXL.yaml"
ckpt_path = "pca_results/008-RAE/checkpoints/1280000.pt"
# ckpt_path = "dual_results/000-RAE/checkpoints/0008000.pt"
# ckpt_path = "results/005-RAE-bf16/checkpoints/0327500.pt"
# ckpt_path = "sub_results3/004-RAE/checkpoints/best.pt"
image_path = "out/dog.png"


# 만약 위 경로에 파일이 없으면 폴더에서 첫 번째 jpg를 자동으로 찾습니다.
if not os.path.exists(image_path):
    print(f"⚠️ 지정한 이미지({image_path})가 없어서 자동으로 검색합니다...")
    found_imgs = glob.glob("imagenette2/train/*/*.JPEG")
    if found_imgs:
        image_path = found_imgs[0]
        print(f"🔎 찾은 이미지: {image_path}")
    else:
        print("❌ 이미지를 찾을 수 없습니다. 경로를 확인해주세요.")
        exit()
# ======================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(config_path, ckpt_path):
    print("🚀 모델 로딩 중...")
    config = OmegaConf.load(config_path)
    
    # 모델 생성 (VQ 설정이 YAML에 있다면 자동으로 적용됨)
    model = instantiate_from_config(config.stage_1).to(device)
    
    # 체크포인트 불러오기
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    # EMA 모델도 있으면 그걸 쓰는게 더 좋음
    if "ema" in ckpt:
        print("📦 EMA 가중치 발견! EMA 모델 사용")
        state_dict = ckpt["ema"]
    else:
        print("📦 일반 모델 가중치 사용")
        state_dict = ckpt["model"]
    
    # DDP로 학습해서 'module.'이 붙어있을 수 있음
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
        
    msg = model.load_state_dict(new_state_dict, strict=False)
    if msg.missing_keys or msg.unexpected_keys:
        print(f"⚠️ Load 메시지: {msg}")
    else:
        print("✅ 체크포인트 완벽히 로드됨")
    
    model.eval()
    return model

# def process_image(img_path, size=256):
#     transform = T.Compose([
#         T.Resize(int(size * 1.14)),  # 약간 크게 리사이즈 후
#         T.CenterCrop(size),          # 중앙 크롭
#         T.ToTensor(),
#     ])
#     img = Image.open(img_path).convert("RGB")
#     return transform(img).unsqueeze(0).to(device)
def process_image(img_path, size=256):
    transform = T.Compose([
        T.Resize(int(size * 1.14), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.CenterCrop(size),
        T.ToTensor(),
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)

def main():
    # 1. 모델 준비
    model = load_model(config_path, ckpt_path)
    
    # VQ 모드 확인 및 정보 출력
    if hasattr(model, 'use_vq') and model.use_vq:
        print("\n✅ VQ 모드 활성화됨")
        print(f"   - 코드북 크기: {model.vq_layer.num_embeddings}")
        print(f"   - 임베딩 차원: {model.vq_layer.embedding_dim}")
        print(f"   - Commitment cost: {model.vq_layer.commitment_cost}")
        
        # 코드북 사용률 체크
        cluster_size = model.vq_layer.cluster_size.cpu()
        used_codes = (cluster_size > 0.1).sum().item()  # 임계값 0.1 이상
        total_codes = model.vq_layer.num_embeddings
        usage_pct = used_codes / total_codes * 100
        
        print(f"   - 사용된 코드: {used_codes}/{total_codes} ({usage_pct:.1f}%)")
        
        if usage_pct < 30:
            print("   ⚠️ 코드북 collapse 가능성! 사용률이 낮습니다.")
        elif usage_pct > 80:
            print("   ✨ 코드북이 골고루 사용되고 있습니다!")
    else:
        print("\n❌ VQ 모드 비활성화 (일반 RAE)")
    
    # 2. 이미지 준비 및 추론
    print(f"\n🖼️ 테스트 이미지: {image_path}")
    img = process_image(image_path)
    print("입력 텐서 mean/std:", img.mean().item(), img.std().item())
    
    with torch.no_grad():
        output = model(img)
        
        if isinstance(output, tuple):
            recon, vq_loss = output
            print(f"\n📊 VQ Loss: {vq_loss.item():.6f}")
        else:
            recon = output
        
        # 사용된 코드 인덱스 확인
        if hasattr(model, 'last_vq_indices') and model.last_vq_indices is not None:
            indices = model.last_vq_indices[0]  # (H, W)
            unique_codes = torch.unique(indices)
            print(f"🎨 이 이미지에 사용된 고유 코드: {len(unique_codes)}개")
            print(f"   인덱스 범위: [{indices.min().item()}, {indices.max().item()}]")

    # 3. 결과 정리
    orig_np = img[0].permute(1, 2, 0).cpu().numpy()
    recon_np = recon[0].permute(1, 2, 0).cpu().numpy()
    
    # 값 범위 확인 (디버깅)
    print(f"\n🔍 원본 픽셀 범위: [{orig_np.min():.3f}, {orig_np.max():.3f}]")
    print(f"🔍 복원 픽셀 범위: [{recon_np.min():.3f}, {recon_np.max():.3f}]")
    
    # 0~1 범위로 클리핑 (왜곡 방지)
    recon_np = recon_np.clip(0, 1)
    
    # 4. MSE/PSNR 계산 (선택)
    mse = ((orig_np - recon_np) ** 2).mean()
    psnr = 10 * torch.log10(torch.tensor(1.0 / mse))
    print(f"📈 MSE: {mse:.6f}, PSNR: {psnr:.2f} dB")

    # 5. 시각화 및 저장
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image", fontsize=14)
    plt.imshow(orig_np)
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    title = f"VQ Reconstruction (Step {ckpt_path.split('/')[-1].split('.')[0]})"
    if hasattr(model, 'use_vq') and model.use_vq:
        title += f"\nPSNR: {psnr:.1f} dB"
    plt.title(title, fontsize=14)
    plt.imshow(recon_np)
    plt.axis("off")
    
    plt.tight_layout()
    # save_path = "vq_result_improved.png"
    save_path = f"pca_result.png"
    # save_path = f"sub_pca_result.png"

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✨ 결과 이미지 저장: {save_path}")



if __name__ == "__main__":
    main()
    
    

    
    
 ##### dual 구조 ####   
# import torch
# import torchvision.transforms as T
# from PIL import Image
# from omegaconf import OmegaConf
# from utils.model_utils import instantiate_from_config
# import matplotlib.pyplot as plt
# import os
# import glob
# import numpy as np

# # ================= 사용자가 수정할 부분 =================
# config_path = "configs/stage1/training/DINOv2-B_decXL.yaml"
# ckpt_path = "dual_results2/001-RAE/checkpoints/0032000.pt"
# image_path = "imagenette2/train/n01440764/n01440764_18.JPEG"
# out_path = "dual_recon.png"
# image_size = 256
# # ======================================================

# device = "cuda" if torch.cuda.is_available() else "cpu"


# def load_model(config_path: str, ckpt_path: str):
#     print("🚀 모델 로딩 중...")
#     cfg = OmegaConf.load(config_path)

#     # 모델 생성 (YAML의 stage_1 설정 그대로)
#     model = instantiate_from_config(cfg.stage_1).to(device)

#     # 체크포인트 불러오기
#     ckpt = torch.load(ckpt_path, map_location="cpu")

#     # EMA 우선
#     if "ema" in ckpt and ckpt["ema"] is not None:
#         print("📦 EMA 가중치 발견! EMA 모델 사용")
#         state_dict = ckpt["ema"]
#     else:
#         print("📦 일반 모델 가중치 사용")
#         state_dict = ckpt["model"]

#     # DDP prefix 제거
#     new_state_dict = {}
#     for k, v in state_dict.items():
#         new_state_dict[k.replace("module.", "")] = v

#     msg = model.load_state_dict(new_state_dict, strict=False)
#     if msg.missing_keys or msg.unexpected_keys:
#         print(f"⚠️ Load 메시지:")
#         if msg.missing_keys:
#             print(f"  - missing_keys({len(msg.missing_keys)}): {msg.missing_keys[:20]}")
#             if len(msg.missing_keys) > 20:
#                 print("    ...")
#         if msg.unexpected_keys:
#             print(f"  - unexpected_keys({len(msg.unexpected_keys)}): {msg.unexpected_keys[:20]}")
#             if len(msg.unexpected_keys) > 20:
#                 print("    ...")
#     else:
#         print("✅ 체크포인트 완벽히 로드됨")

#     model.eval()
#     model.requires_grad_(False)
#     return model


# def process_image(img_path: str, size: int = 256):
#     transform = T.Compose([
#         T.Resize(int(size * 1.14)),
#         T.CenterCrop(size),
#         T.ToTensor(),
#     ])
#     img = Image.open(img_path).convert("RGB")
#     return transform(img).unsqueeze(0).to(device)


# def find_fallback_image(default_path: str):
#     if os.path.exists(default_path):
#         return default_path

#     print(f"⚠️ 지정한 이미지({default_path})가 없어서 자동으로 검색합니다...")
#     found_imgs = glob.glob("imagenette2/train/*/*.JPEG")
#     if found_imgs:
#         print(f"🔎 찾은 이미지: {found_imgs[0]}")
#         return found_imgs[0]

#     raise FileNotFoundError("❌ 이미지를 찾을 수 없습니다. 경로를 확인해주세요.")


# def print_vq_info(model):
#     # dual VQ / single VQ / no VQ 모두 안전하게 처리
#     if hasattr(model, "use_vq") and bool(model.use_vq):
#         print("\n✅ VQ 모드 활성화됨")

#         # dual-codebook
#         if hasattr(model, "vq_sem") and hasattr(model, "vq_tex"):
#             print(f"   - SEM codebook: K={model.vq_sem.num_embeddings}, D={model.vq_sem.embedding_dim}")
#             print(f"   - TEX codebook: K={model.vq_tex.num_embeddings}, D={model.vq_tex.embedding_dim}")

#             sem_used = int((model.vq_sem.cluster_size.detach().cpu() > 0.1).sum().item())
#             tex_used = int((model.vq_tex.cluster_size.detach().cpu() > 0.1).sum().item())
#             print(f"   - SEM used codes: {sem_used}/{model.vq_sem.num_embeddings} ({sem_used/model.vq_sem.num_embeddings*100:.2f}%)")
#             print(f"   - TEX used codes: {tex_used}/{model.vq_tex.num_embeddings} ({tex_used/model.vq_tex.num_embeddings*100:.2f}%)")

#         # single-codebook (구버전 호환)
#         elif hasattr(model, "vq_layer"):
#             print(f"   - codebook: K={model.vq_layer.num_embeddings}, D={model.vq_layer.embedding_dim}")
#             used = int((model.vq_layer.cluster_size.detach().cpu() > 0.1).sum().item())
#             print(f"   - used codes: {used}/{model.vq_layer.num_embeddings} ({used/model.vq_layer.num_embeddings*100:.2f}%)")

#         else:
#             print("   - (VQ 활성화로 보이지만 vq 모듈을 찾지 못함: vq_sem/vq_tex 또는 vq_layer 없음)")
#     else:
#         print("\n❌ VQ 모드 비활성화 (일반 RAE)")


# def main():
#     global image_path
#     image_path = find_fallback_image(image_path)

#     # 1) 모델 로드
#     model = load_model(config_path, ckpt_path)
#     print_vq_info(model)

#     # 2) 이미지 로드
#     print(f"\n🖼️ 테스트 이미지: {image_path}")
#     img = process_image(image_path, size=image_size)

#     # 3) 추론
#     with torch.no_grad():
#         out = model(img)

#         # training 모드에서 tuple을 반환하도록 설계된 경우 방어
#         if isinstance(out, tuple):
#             recon = out[0]
#             maybe_vq_loss = out[1] if len(out) > 1 else None
#             if isinstance(maybe_vq_loss, torch.Tensor):
#                 print(f"📊 VQ Loss (returned): {maybe_vq_loss.item():.6f}")
#         else:
#             recon = out

#         # indices 출력 (dual / single 모두 대응)
#         if hasattr(model, "last_vq_indices") and model.last_vq_indices is not None:
#             try:
#                 # dual: (idx_sem, idx_tex)
#                 idx_sem, idx_tex = model.last_vq_indices
#                 sem_unique = torch.unique(idx_sem).numel()
#                 tex_unique = torch.unique(idx_tex).numel()
#                 print(f"🎨 SEM unique codes in this image: {sem_unique}")
#                 print(f"🎨 TEX unique codes in this image: {tex_unique}")
#                 print(f"   SEM index range: [{int(idx_sem.min())}, {int(idx_sem.max())}]")
#                 print(f"   TEX index range: [{int(idx_tex.min())}, {int(idx_tex.max())}]")
#             except Exception:
#                 # single: (H,W) 혹은 (B,H,W)
#                 idx = model.last_vq_indices
#                 if idx.dim() == 3:
#                     idx = idx[0]
#                 uniq = torch.unique(idx).numel()
#                 print(f"🎨 unique codes in this image: {uniq}")
#                 print(f"   index range: [{int(idx.min())}, {int(idx.max())}]")

#     # 4) numpy 변환
#     orig_np = img[0].permute(1, 2, 0).detach().cpu().numpy()
#     recon_np = recon[0].permute(1, 2, 0).detach().cpu().numpy()

#     print(f"\n🔍 원본 픽셀 범위: [{orig_np.min():.3f}, {orig_np.max():.3f}]")
#     print(f"🔍 복원 픽셀 범위: [{recon_np.min():.3f}, {recon_np.max():.3f}]")

#     recon_np = np.clip(recon_np, 0.0, 1.0)

#     # 5) MSE/PSNR
#     mse = float(((orig_np - recon_np) ** 2).mean())
#     psnr = 10.0 * np.log10(1.0 / max(mse, 1e-12))
#     print(f"📈 MSE: {mse:.6f}, PSNR: {psnr:.2f} dB")

#     # 6) 저장
#     plt.figure(figsize=(12, 5))

#     plt.subplot(1, 2, 1)
#     plt.title("Original", fontsize=14)
#     plt.imshow(orig_np)
#     plt.axis("off")

#     plt.subplot(1, 2, 2)
#     step_str = os.path.basename(ckpt_path).split(".")[0]
#     plt.title(f"Reconstruction ({step_str})\nPSNR: {psnr:.1f} dB", fontsize=14)
#     plt.imshow(recon_np)
#     plt.axis("off")

#     plt.tight_layout()
#     plt.savefig(out_path, dpi=150, bbox_inches="tight")
#     print(f"\n✨ 결과 이미지 저장: {out_path}")


# if __name__ == "__main__":
#     main()


