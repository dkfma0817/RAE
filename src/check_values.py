import os
import glob
import torch
import torchvision.transforms as T
from PIL import Image
from omegaconf import OmegaConf
from utils.model_utils import instantiate_from_config


# ================= 설정 =================
config_path = "configs/stage1/training/DINOv2-B_decXL.yaml"
ckpt_path   = "results/002-RAE-bf16/checkpoints/0015000.pt"   # 체크포인트 경로
image_path  = "imagenette2/train/n01440764/n01440764_18.JPEG" # 테스트 이미지
# =======================================

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(config_path: str, ckpt_path: str):
    print("🚀 모델 로딩 중...")
    cfg = OmegaConf.load(config_path)
    model = instantiate_from_config(cfg.stage_1).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"✅ 체크포인트 로드 완료: {msg}")

    model.eval()
    return model


def load_image(image_path: str):
    if not os.path.exists(image_path):
        print(f"⚠️ 지정 이미지가 없어 자동 검색: {image_path}")
        fs = glob.glob("imagenette2/train/*/*.JPEG") + glob.glob("imagenette2/val/*/*.JPEG")
        if not fs:
            raise FileNotFoundError("이미지를 찾지 못했습니다. image_path 또는 dataset 경로를 확인하세요.")
        image_path = fs[0]
        print(f"🔎 찾은 이미지: {image_path}")

    img = Image.open(image_path).convert("RGB")

    # 입력은 0~1 텐서
    x = T.Compose([
        T.Resize((224, 224)),   # encoder_input_size=224 기준
        T.ToTensor(),
    ])(img).unsqueeze(0).to(device)

    return image_path, x


@torch.no_grad()
def inspect_vq(model, x):
    print("\n================= INSPECT =================")

    print(f"📌 device: {device}")
    print(f"📌 input shape: {tuple(x.shape)} | min={x.min().item():.4f} max={x.max().item():.4f} mean={x.mean().item():.4f}")

    # 1) encoder output
    z_enc = model.encode(x)
    print(f"\n📊 [1. encoder z]")
    print(f"   shape: {tuple(z_enc.shape)} | min={z_enc.min().item():.4f} max={z_enc.max().item():.4f} mean={z_enc.mean().item():.4f}")

    # 2) VQ path 여부 확인
    use_vq = getattr(model, "use_vq", False)
    has_vq = hasattr(model, "vq_layer") and hasattr(model, "vq_pre") and hasattr(model, "vq_post")

    if use_vq and has_vq:
        print("\n⚡ VQ path detected: vq_pre -> vq_layer -> vq_post")

        # vq_pre (768 -> 256)
        z_small = model.vq_pre(z_enc)
        print(f"\n📊 [2.1 vq_pre(z)]")
        print(f"   shape: {tuple(z_small.shape)} | min={z_small.min().item():.4f} max={z_small.max().item():.4f} mean={z_small.mean().item():.4f}")

        # z_norm (학습과 동일하게)
        if getattr(model, "vq_z_norm", False):
            denom = z_small.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
            z_small = z_small / denom
            print(f"   (after z_norm) min={z_small.min().item():.4f} max={z_small.max().item():.4f} mean={z_small.mean().item():.4f}")

        # vq_layer
        z_q, vq_loss, indices = model.vq_layer(z_small)
        print(f"\n📊 [2.2 vq_layer]")
        print(f"   vq_loss: {vq_loss.item():.6f}")
        print(f"   z_q shape: {tuple(z_q.shape)} | min={z_q.min().item():.4f} max={z_q.max().item():.4f} mean={z_q.mean().item():.4f}")

        idx_flat = indices.view(-1)
        num_embeddings = getattr(model.vq_layer, "num_embeddings", None)
        if num_embeddings is None:
            num_embeddings = getattr(model.vq_layer, "n_embed", None)
        if num_embeddings is None:
            num_embeddings = getattr(model.vq_layer, "K", None)
        if num_embeddings is None:
            num_embeddings = 1024  # fallback

        unique_codes = torch.unique(idx_flat).numel()
        print(f"   indices sample: {idx_flat[:20].tolist()}")
        print(f"   ✅ unique_codes: {unique_codes} / {num_embeddings}")

        # perplexity
        counts = torch.bincount(idx_flat, minlength=num_embeddings).float()
        probs = counts / counts.sum().clamp(min=1.0)
        perplexity = torch.exp(-(probs * torch.log(probs + 1e-10)).sum()).item()
        print(f"   ✅ perplexity: {perplexity:.2f}")

        # vq_post (256 -> 768)
        z_quant = model.vq_post(z_q)
        print(f"\n📊 [2.3 vq_post(z_q)]")
        print(f"   shape: {tuple(z_quant.shape)} | min={z_quant.min().item():.4f} max={z_quant.max().item():.4f} mean={z_quant.mean().item():.4f}")

        # decoder
        recon = model.decode(z_quant)

    else:
        if use_vq and not has_vq:
            print("\n❌ use_vq=True 인데 vq_pre/vq_layer/vq_post가 없습니다. 모델 코드/체크포인트 불일치 가능")
        else:
            print("\nℹ️ VQ 모드가 아니거나(use_vq=False) VQ 모듈이 없습니다. encoder->decoder만 테스트합니다.")
        recon = model.decode(z_enc)

        # last_vq_indices가 있다면 찍어주기 (forward를 한 번 돌려야 생길 수 있음)
        if getattr(model, "last_vq_indices", None) is not None:
            idx_flat = model.last_vq_indices.view(-1)
            print(f"   last_vq_indices sample: {idx_flat[:20].tolist()}")

    # 3) recon stats
    print(f"\n📊 [3. recon]")
    print(f"   shape: {tuple(recon.shape)} | min={recon.min().item():.4f} max={recon.max().item():.4f} mean={recon.mean().item():.4f}")

    # quick diagnosis
    if recon.min() < -2 or recon.max() > 2:
        print("\n🚨 [진단] 출력값 범위가 비정상적으로 큼. (scale/unnormalize 또는 학습 불안정 가능)")
    else:
        print("\n✅ [진단] 출력 range 자체는 크게 이상하지 않아 보임.")

    return recon


def main():
    model = load_model(config_path, ckpt_path)
    p, x = load_image(image_path)
    print(f"🔎 테스트 이미지: {p}")

    _ = inspect_vq(model, x)


if __name__ == "__main__":
    main()
