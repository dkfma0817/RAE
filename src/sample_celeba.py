import os, math
import torch
from torchvision.utils import save_image
from omegaconf import OmegaConf

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# repo import
try:
    from utils.model_utils import instantiate_from_config
except ImportError:
    from src.utils.model_utils import instantiate_from_config

from stage1 import RAE
from models.t2i_model import RAE_T2I_Model


# -------------------------
# Rectified Flow sampling (Euler ODE)
# -------------------------
@torch.no_grad()
def sample_rectified_flow_ode(model, prompts, latent_shape, steps=50, cfg_scale=5.0, device="cuda"):
    """
    model: RAE_T2I_Model (predicts v)
    prompts: list[str], length = B
    latent_shape: tuple like (C,H,W) OR (N,C) depending on your RAE latent
    steps: ODE steps
    cfg_scale: classifier-free guidance scale (>=1)
    """
    B = len(prompts)

    # init noise
    z = torch.randn((B, *latent_shape), device=device)

    # prepare CFG batch: [cond; uncond]
    if cfg_scale > 1.0:
        z_in = torch.cat([z, z], dim=0)
        prompts_in = prompts + [""] * B
    else:
        z_in = z
        prompts_in = prompts

    # time grid: 1 -> 0
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)

    zt = z_in
    for i in range(steps):
        t = ts[i].expand(zt.shape[0])  # (2B,) or (B,)
        dt = ts[i+1] - ts[i]           # negative

        # predict v
        v = model(zt, t, prompts_in)

        if cfg_scale > 1.0:
            v_cond, v_uncond = v.chunk(2, dim=0)
            v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)
            v = torch.cat([v_guided, v_guided], dim=0)

        # Euler update
        zt = zt + dt.view(-1, *([1] * (zt.dim() - 1))) * v

    # return only conditional half
    if cfg_scale > 1.0:
        z_final, _ = zt.chunk(2, dim=0)
    else:
        z_final = zt
    return z_final


def load_stage1_rae(stage1_yaml: str, device: str) -> RAE:
    cfg = OmegaConf.load(stage1_yaml)
    rae: RAE = instantiate_from_config(cfg.stage_1).to(device)
    rae.eval()
    for p in rae.parameters():
        p.requires_grad = False
    return rae


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- paths ----
    ckpt_path = "checkpoints/celeba_t2i_ep2.pt"
    stage1_yaml = "configs/stage1/pretrained/SigLIP2.yaml"

    # ---- prompts to test ----
    # 원하는 대로 바꿔도 됨
    prompts = [
        "a photo of a person smiling with glasses",
        # "a face photo, not smiling, no eyeglasses, no bangs, not blond hair, no mustache",
    #    "a face photo blond hair",
        "a photo of a person with bangs"
    ]

    # ---- load RAE ----
    rae = load_stage1_rae(stage1_yaml, device)

    # ---- infer latent shape by encoding one dummy image? (skip) ----
    # 여기서는 학습에서 쓰던 latent_size를 직접 지정하는 게 제일 확실함.
    # 우선 네 stage1이 보통 (B,768,16,16) 쓰는 세팅으로 가정:
    latent_shape = (768, 16, 16)

    # ---- load model (must match training config) ----
    model = RAE_T2I_Model(
        siglip_name="google/siglip2-base-patch16-256",
        train_text_encoder=False,
        input_size=16,
        in_channels=latent_shape[0],
        hidden_size=[1152, 2048],
        depth=[28, 2],       # 학습 때랑 동일해야 함
        num_heads=[16, 16],
    ).to(device)
    model.eval()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    # 너 저장 방식이 {"model": state_dict} 형태였지
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    # ---- sample latent ----
    z = sample_rectified_flow_ode(
        model=model,
        prompts=prompts,
        latent_shape=latent_shape,
        steps=50,
        cfg_scale=5.0,
        device=device,
    )

    # ---- decode ----
    # 학습에서 latent_scale을 곱했다면, 여기서도 역으로 나눠줘야 함.
    latent_scale = 1.0
    z = z / latent_scale

    with torch.no_grad():
        imgs = rae.decode(z.to(torch.float32))

    os.makedirs("samples", exist_ok=True)
    out_path = "samples/ep2_test6.png"
    save_image(imgs, out_path, nrow=len(prompts), normalize=True, value_range=(0, 1))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()