import torch
import torchvision.transforms as T
from PIL import Image

# -------------------------
# preprocess (너가 쓰던 방식)
# -------------------------
def process_image_center_crop(img_path, size=256, device="cuda"):
    tfm = T.Compose([
        T.Resize(int(size * 1.14)),
        T.CenterCrop(size),
        T.ToTensor(),
    ])
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)  # (1,3,256,256) in [0,1]
    return img, x


# -------------------------
# encode -> z_q (RAE 경로와 동일)
# -------------------------
@torch.no_grad()
def encode_to_zq_exact(model, x):
    """
    model.encode(x) -> pca_reweight -> vq_pre -> (optional z_norm) -> vq_layer
    return:
      z_q: (B, Cq, H, W)  e.g. (1,256,16,16)
      indices: (B,H,W) or (B,N)
      vq_loss (optional)
    """
    # encode(): 내부에서 encoder_input_size로 resize + normalize 등을 처리하는 경우가 많음
    z = model.encode(x)  # 보통 (B, latent_dim, H, W)

    # PCA reweight (optional)
    if getattr(model, "pca_reweight", None) is not None:
        z = model.pca_reweight(z)

    # vq_pre (필수라고 가정)
    if getattr(model, "vq_pre", None) is not None:
        z_small = model.vq_pre(z)
    else:
        z_small = z

    # vq_z_norm (optional)
    if bool(getattr(model, "vq_z_norm", False)):
        denom = z_small.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
        z_small = z_small / denom

    # vq_layer
    out = model.vq_layer(z_small)

    # 흔한 형태: (z_q, vq_loss, indices)
    if isinstance(out, (tuple, list)) and len(out) >= 3:
        z_q, vq_loss, indices = out[0], out[1], out[2]
    else:
        # 구현이 다르면 여기서 조정 필요
        raise RuntimeError(f"Unexpected vq_layer output: type={type(out)}, len={len(out) if hasattr(out,'__len__') else 'NA'}")

    return z_q, indices, vq_loss


# -------------------------
# decode from z_q (RAE 경로와 동일)
# -------------------------
@torch.no_grad()
def decode_from_zq_exact(model, z_q):
    """
    z_q -> vq_post -> model.decode()  (pixel space recon)
    """
    z_latent = model.vq_post(z_q) if getattr(model, "vq_post", None) is not None else z_q
    x_rec = model.decode(z_latent)   # (B,3,H,W), 보통 decode 안에서 denorm 처리까지 함
    x_rec = torch.clamp(x_rec, 0.0, 1.0)
    return x_rec


# -------------------------
# rotation test (이게 핵심)
# -------------------------
@torch.no_grad()
def rotation_on_quantized_latent(model, x, k=1):
    """
    1) z_q 추출
    2) z_q를 spatial dims에서 rot90
    3) decode해서 비교
    """
    z_q, indices, vq_loss = encode_to_zq_exact(model, x)

    # baseline recon (정상 경로)
    recon_normal = decode_from_zq_exact(model, z_q)

    # rotate z_q (spatial grid만 회전)
    z_q_rot = torch.rot90(z_q, k=k, dims=(2, 3)).contiguous()
    recon_rot = decode_from_zq_exact(model, z_q_rot)

    return recon_normal, recon_rot, vq_loss


# -------------------------
# 사용 예시
# -------------------------
def run_rotation(model, image_path, device="cuda", input_size=256, k=1):
    img_pil, x = process_image_center_crop(image_path, size=input_size, device=device)
    recon_normal, recon_rot, vq_loss = rotation_on_quantized_latent(model, x, k=k)

    return img_pil, x, recon_normal, recon_rot, vq_loss



from omegaconf import OmegaConf
from utils.model_utils import instantiate_from_config

CONFIG_PATH = "configs/stage1/training/DINOv2-B_decXL.yaml"
CKPT_PATH   = "pca_results/008-RAE/checkpoints/1280000.pt"
IMAGE_PATH  = "dog.png"

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(config_path, ckpt_path, device):
    print("🚀 Loading model...")
    conf = OmegaConf.load(config_path)

    model_conf = conf.stage_1 if "stage_1" in conf else conf
    model = instantiate_from_config(model_conf).to(device).eval()

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # EMA 우선
    if isinstance(ckpt, dict) and "ema" in ckpt:
        print("📦 Using EMA weights")
        state_dict = ckpt["ema"]
    else:
        state_dict = ckpt["model"]

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    return model


if __name__ == "__main__":
    model = load_model(CONFIG_PATH, CKPT_PATH, device)

    img_pil, x, recon_normal, recon_rot, vq_loss = run_rotation(
        model=model,
        image_path=IMAGE_PATH,
        device=device,
        input_size=256,
        k=1,     # 1=90도, 2=180도
    )

    print(f"VQ loss: {vq_loss.item():.6f}")

    # -------------------------
    # 시각화
    # -------------------------
    import matplotlib.pyplot as plt
    import numpy as np

    orig = np.array(img_pil.resize((256, 256)))
    recon_n = recon_normal[0].permute(1,2,0).cpu().numpy()
    recon_r = recon_rot[0].permute(1,2,0).cpu().numpy()

    plt.figure(figsize=(12,4))


    plt.subplot(1,3,1)
    plt.title("Input")
    plt.imshow(orig)
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Recon (normal)")
    plt.imshow(recon_n)
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Recon (latent rotated 90°)")
    plt.imshow(recon_r)
    plt.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("rotation_result.png", dpi=150)
    print("✅ Saved: rotation_result.png")
    plt.show()
