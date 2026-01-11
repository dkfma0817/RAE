import os
import sys
import math
import argparse
from time import time

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs
from stage2.transport import create_transport
from stage1 import RAE
from stage2.models import Stage2ModelProtocol

# For relative imports in repo style
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# -----------------------------
# Helpers: image loading
# -----------------------------
def load_image_256(path: str, device: str) -> torch.Tensor:
    """
    Loads an image and returns a float tensor in [0,1], shape (1,3,256,256).
    NOTE: RAE's encoder wrapper often handles its own normalization; this keeps it simple.
    """
    img = Image.open(path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(256),
        transforms.ToTensor(),  # [0,1], (3,H,W)
    ])
    x = tfm(img).unsqueeze(0).to(device)
    return x


# -----------------------------
# Helpers: SigLIP text embedding
# -----------------------------
def get_siglip_text_embed_from_rae(rae: RAE, text: str, device: str) -> torch.Tensor:
    """
    Tries to obtain SigLIP text embedding aligned with the RAE's SigLIP image features.
    We try:
      1) rae.encoder.text_embed / encode_text / get_text_features (repo-specific)
      2) fallback to HuggingFace transformers (if installed)
    Returns: (1, C) float tensor on device.
    """
    # --- 1) repo-specific: look for methods on rae or rae.encoder
    for obj in [rae, getattr(rae, "encoder", None)]:
        if obj is None:
            continue
        for name in ["encode_text", "get_text_features", "text_features", "text_embed", "embed_text"]:
            fn = getattr(obj, name, None)
            if callable(fn):
                out = fn([text] if isinstance(text, str) else text)  # common pattern: list[str]
                if isinstance(out, (tuple, list)):
                    out = out[0]
                if out.dim() == 1:
                    out = out.unsqueeze(0)
                return out.to(device)

    # --- 2) fallback: HuggingFace SigLIP2
    try:
        from transformers import AutoProcessor, AutoModel
    except Exception as e:
        raise RuntimeError(
            "Could not find a text-embedding method on RAE/SigLIP2wNorm, and transformers is not available.\n"
            "Either (a) expose a text encoder method in stage1 encoder, or (b) install transformers.\n"
            f"Original error: {e}"
        )

    # Try to reuse model name from config if present
    model_name = None
    # Best effort to fetch from rae config if stored
    if hasattr(rae, "encoder_config_path"):
        model_name = getattr(rae, "encoder_config_path")
    if model_name is None:
        model_name = "google/siglip2-base-patch16-256"

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        # HF SigLIP returns text_embeds for some variants; for others you may need model.get_text_features
        if hasattr(model, "get_text_features"):
            e = model.get_text_features(**inputs)
        else:
            out = model(**inputs)
            e = getattr(out, "text_embeds", None)
            if e is None:
                raise RuntimeError("Could not obtain text embedding from HF SigLIP model outputs.")
    if e.dim() == 1:
        e = e.unsqueeze(0)
    return e


# -----------------------------
# Text loss: patch-wise similarity + soft-attention pooling
# -----------------------------
def patch_text_alignment_loss(z_bchw: torch.Tensor, e_bc: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
    """
    z_bchw: (B, C, H, W)  - RAE latent map
    e_bc:   (B, C)        - text embedding
    tau: temperature for softmax over patches

    Steps:
      1) compute cosine sim per patch s_ij = cos(z_ij, e)
      2) w_ij = softmax(s_ij / tau)
      3) z_bar = sum w_ij * z_ij
      4) loss = -cos(z_bar, e)
    """
    B, C, H, W = z_bchw.shape
    z = z_bchw.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, HW, C)

    z_n = F.normalize(z, dim=-1)          # (B, HW, C)
    e_n = F.normalize(e_bc, dim=-1)       # (B, C)
    e_n = e_n.unsqueeze(1)                # (B, 1, C)

    sims = (z_n * e_n).sum(dim=-1)        # (B, HW)
    weights = F.softmax(sims / max(tau, 1e-6), dim=-1)  # (B, HW)

    z_bar = (weights.unsqueeze(-1) * z).sum(dim=1)      # (B, C)
    z_bar_n = F.normalize(z_bar, dim=-1)
    # maximize cosine => minimize negative cosine
    loss = -(z_bar_n * e_n.squeeze(1)).sum(dim=-1).mean()
    return loss


# -----------------------------
# Prior loss: velocity matching (Linear path)
# -----------------------------
# def dit_velocity_prior_loss(
#     model_fwd,
#     z_bchw: torch.Tensor,
#     y: torch.Tensor,
#     time_dist_shift: float,
#     device: str
# ) -> torch.Tensor:
#     """
#     Uses flow-matching style velocity prediction.
#     For a Linear path:
#         z_t = (1 - t) * z + t * eps
#         v*(z,t) = eps - z
#     Loss:
#         || v_pred(z_t, t, y) - (eps - z) ||^2
#     Note: We apply a simple "time_dist_shift" by shifting the sampled t distribution:
#       - repo uses time_dist_shift for dimension-dependent shift; transport handles it in sampling.
#       - Here we approximate by scaling t's logit; simplest first implementation uses power-law-ish shift.
#     """
#     B = z_bchw.shape[0]
#     eps = torch.randn_like(z_bchw)

#     # Sample t ~ Uniform(0,1) then apply a simple shift heuristic
#     # (If you later hook into transport's official time sampling, replace this.)
#     t = torch.rand(B, device=device)

#     # Heuristic shift: push mass toward small t for high-dim (similar spirit to schedule shift)
#     # Map t via t' = t^(1/time_dist_shift). If time_dist_shift>1 => smaller effective t on average.
#     if time_dist_shift is not None and time_dist_shift > 1e-6:
#         t = torch.clamp(t, 1e-6, 1.0)
#         t = t.pow(1.0 / time_dist_shift)

#     t_b = t.view(B, 1, 1, 1)

#     z_t = (1.0 - t_b) * z_bchw + t_b * eps
#     v_target = eps - z_bchw

#     # Model forward: expects (z_t, t, y=...)
#     # t shape: (B,) is usually fine; if not, change to (B,1) or (B,).
#     v_pred = model_fwd(z_t, t, y=y)

#     # Ensure same shape
#     if v_pred.shape != v_target.shape:
#         raise RuntimeError(f"v_pred shape {v_pred.shape} != v_target shape {v_target.shape}")

#     loss = F.mse_loss(v_pred, v_target)
#     return loss

# -----------------------------
# Prior loss: Latent Diffusion (DiT operates on RAE latent space)
# -----------------------------
def rae_latent_diffusion_prior(
    model_fwd,
    z_bchw: torch.Tensor,
    y: torch.Tensor,
    device: str,
    n_timesteps: int = 1
) -> torch.Tensor:
    """
    DiT가 RAE latent space에서 직접 denoising하는 경우.
    논문 식(1)의 구현: L_prior = E_t,ε[‖ε - ε_θ(z_t, t, z, c)‖²]
    
    Args:
        model_fwd: DiT forward function
        z_bchw: (B, C, H, W) - RAE latent representation
        y: (B,) - class labels for conditioning
        device: device string
        n_timesteps: number of timesteps to average over
    
    Returns:
        Diffusion prior loss (scalar)
    """
    B = z_bchw.shape[0]
    
    losses = []
    for _ in range(n_timesteps):
        # Sample random timestep and noise
        t = torch.rand(B, device=device)
        eps = torch.randn_like(z_bchw)
        
        # Forward diffusion in latent space: z_t = α_t·z + σ_t·ε
        alpha_t = (1 - t).view(B, 1, 1, 1)
        sigma_t = t.view(B, 1, 1, 1)
        z_t = alpha_t * z_bchw + sigma_t * eps
        
        # DiT predicts noise in latent space
        eps_pred = model_fwd(z_t, t, y=y)
        
        # Denoising score matching loss
        loss = F.mse_loss(eps_pred, eps)
        losses.append(loss)
    
    return torch.stack(losses).mean() if len(losses) > 1 else losses[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    parser.add_argument("--text", type=str, required=True, help="Text query/prompt.")
    parser.add_argument("--class_id", type=int, default=1000, help="Class label for stage2 conditioning (1000 often used as null).")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--lambda_prior", type=float, default=0.1)
    parser.add_argument("--beta_anchor", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--save_prefix", type=str, default="out/opt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_prefix), exist_ok=True)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # IMPORTANT: we need gradients for z optimization
    torch.set_grad_enabled(True)

    # Load configs and models (same style as sample.py)
    rae_config, model_config, transport_config, sampler_config, guidance_config, misc, _ = parse_configs(args.config)

    rae: RAE = instantiate_from_config(rae_config).to(device)
    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)

    rae.eval()
    model.eval()

    # Freeze all model params (we only optimize z)
    for p in rae.parameters():
        p.requires_grad_(False)
    for p in model.parameters():
        p.requires_grad_(False)

    # Time distribution shift (as in sample.py)
    shift_dim = misc.get("time_dist_shift_dim", 768 * 16 * 16)
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)
    print(f"[info] time_dist_shift={time_dist_shift:.4f} = sqrt({shift_dim}/{shift_base})")

    # Create transport (kept for future extension; current code uses a simple shifted t heuristic)
    _transport = create_transport(**transport_config["params"], time_dist_shift=time_dist_shift)

    # Load image, encode to z0
    x = load_image_256(args.image, device=device)

    # Save original (for comparison)
    save_image(x, f"{args.save_prefix}_x.png", normalize=True, value_range=(0, 1))

    with torch.no_grad():
        # Most RAE implementations expose encode()
        if hasattr(rae, "encode") and callable(getattr(rae, "encode")):
            z0 = rae.encode(x)
        else:
            # fallback: try encoder attribute
            enc = getattr(rae, "encoder", None)
            if enc is None:
                raise RuntimeError("RAE has no encode() method and no encoder attribute.")
            z0 = enc(x)

    if z0.dim() != 4:
        raise RuntimeError(f"Expected z0 to be BCHW (B,C,H,W). Got shape: {tuple(z0.shape)}")
    print(f"[info] z0 shape: {tuple(z0.shape)} (expected ~ (B,768,16,16))")

    # Prepare optimizable latent
    z = z0.detach().clone()
    z.requires_grad_(True)

    # Prepare text embedding (SigLIP text encoder)
    with torch.no_grad():
        e_q = get_siglip_text_embed_from_rae(rae, args.text, device=device)
        # Ensure batch match
        if e_q.shape[0] == 1 and z.shape[0] > 1:
            e_q = e_q.repeat(z.shape[0], 1)
        if e_q.shape[0] != z.shape[0]:
            raise RuntimeError(f"text embed batch {e_q.shape[0]} != z batch {z.shape[0]}")
        # If text embed dim mismatches channel dim, you need a projection head.
        if e_q.shape[1] != z.shape[1]:
            raise RuntimeError(
                f"text embed dim {e_q.shape[1]} != latent channels {z.shape[1]}.\n"
                "You need to add a small projection layer (learned or fixed) to align dims."
            )

    # Class label y for stage2
    y = torch.tensor([args.class_id] * z.shape[0], device=device, dtype=torch.long)

    # Choose model forward (guidance scale=1 in your YAML; keep it simple)
    model_fwd = model.forward

    # Optimizer for z only
    opt = torch.optim.Adam([z], lr=args.lr)

    # Optional: decode z0 for reference
    with torch.no_grad():
        x0_hat = rae.decode(z0)
        save_image(x0_hat, f"{args.save_prefix}_x0_hat.png", normalize=True, value_range=(0, 1))

    print("[info] Starting optimization...")
    start = time()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)

        # Text (patch-wise) alignment loss
        l_text = patch_text_alignment_loss(z, e_q, tau=args.tau)

        # DiT prior loss (latent diffusion) - 이 부분만 교체!
        l_prior = rae_latent_diffusion_prior(
            model_fwd=model_fwd,
            z_bchw=z,
            y=y,
            device=device,
            n_timesteps=1  # 필요시 증가 (계산량↑, 안정성↑)
        )

        # Anchor loss
        l_anchor = F.mse_loss(z, z0)

        loss = l_text + args.lambda_prior * l_prior + args.beta_anchor * l_anchor
        loss.backward()
        opt.step()

        if step % 20 == 0 or step == 1 or step == args.steps:
            with torch.no_grad():
                # decode current z for monitoring (optional but helpful early on)
                x_hat = rae.decode(z.detach())
                save_image(x_hat, f"{args.save_prefix}_step{step:04d}.png", normalize=True, value_range=(0, 1))

            print(
                f"[step {step:04d}/{args.steps}] "
                f"loss={loss.item():.4f} | "
                f"text={l_text.item():.4f} "
                f"prior={l_prior.item():.4f} "
                f"anchor={l_anchor.item():.4f}"
            )

    elapsed = time() - start
    print(f"[info] Optimization done in {elapsed:.2f}s")

    # Save final decode
    with torch.no_grad():
        x_star = rae.decode(z)
        save_image(x_star, f"{args.save_prefix}_x_star.png", normalize=True, value_range=(0, 1))

    # Save final latent
    torch.save({"z0": z0.cpu(), "z_star": z.detach().cpu(), "text": args.text}, f"{args.save_prefix}_latents.pt")
    print(f"[info] Saved outputs with prefix: {args.save_prefix}")


if __name__ == "__main__":
    main()


# # Usage example:
#   CUDA_VISIBLE_DEVICES=1 python src/inference_opt.py \
#     --config configs/stage2/sampling/ImageNet256/sample_DiTDH-XL_SigLIP2.yaml \
#     --image dog.png \
#     --text "maltese's face" \
#     --class_id 153 \
#     --steps 200 --lr 5e-3 \
#     --lambda_prior 0.1 --beta_anchor 1.0 \
#     --tau 0.05 \
#     --save_prefix out/run1