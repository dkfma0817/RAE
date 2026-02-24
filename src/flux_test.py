import torch
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import argparse
from pathlib import Path

def load_img(p: Path):
    img = Image.open(p).convert("RGB")
    # FLUX VAE는 입력 크기가 8의 배수여야 안전합니다.
    x = transforms.ToTensor()(img).unsqueeze(0) 
    return x

@torch.no_grad()
def step_flux_vae(vae, x, deterministic=True):
    # 1. [-1, 1] 범위로 변환
    x_in = (x * 2.0 - 1.0).to(vae.device, dtype=vae.dtype)

    # 2. Encode
    posterior = vae.encode(x_in).latent_dist
    z = posterior.mode() if deterministic else posterior.sample()

    # 3. FLUX 특유의 Latent Scaling & Shifting
    # 공식: (latent - shift) * scale
    shift = vae.config.shift_factor if hasattr(vae.config, "shift_factor") else 0.0
    scale = vae.config.scaling_factor if hasattr(vae.config, "scaling_factor") else 1.0
    
    z_processed = (z - shift) * scale
    
    # 4. Decode (역연산 적용 후 Decode)
    z_unprocessed = (z_processed / scale) + shift
    dec = vae.decode(z_unprocessed).sample
    
    # 5. [0, 1] 범위로 복원
    x_out = (dec + 1.0) / 2.0
    return z, x_out.clamp(0.0, 1.0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--outdir", type=Path, default=Path("flux_repeat"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # FLUX VAE 모델 로드 (black-forest-labs/FLUX.1-dev 또는 schnell 동일)
    vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae", torch_dtype=torch.float16).to(device)
    vae.eval()

    x = load_img(args.image).to(device)
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    # 초기 z0 저장용
    z0, _ = step_flux_vae(vae, x)
    z0f = z0.reshape(1, -1)

    print("k\tmse_to_z0\tcos_to_z0\tmse_to_prev")
    prev_zf = z0f

    for k in range(1, args.steps + 1):
        z, x = step_flux_vae(vae, x)
        zf = z.reshape(1, -1)
        
        mse_z0 = (zf - z0f).pow(2).mean().item()
        cos = torch.nn.functional.cosine_similarity(zf, z0f, dim=1).item()
        mse_prev = (zf - prev_zf).pow(2).mean().item()
        
        print(f"{k:03d}\t{mse_z0:.6e}\t{cos:.6f}\t{mse_prev:.6e}")
        save_image(x, args.outdir / f"x_{k:03d}.png")
        prev_zf = zf

if __name__ == "__main__":
    main()