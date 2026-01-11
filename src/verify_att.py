import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# For relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs
from stage1 import RAE


def load_image_256(path: str, device: str) -> torch.Tensor:
    """Load and preprocess image to 256x256"""
    img = Image.open(path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    x = tfm(img).unsqueeze(0).to(device)
    return x


def get_siglip_text_embed(rae: RAE, text: str, device: str) -> torch.Tensor:
    """Get SigLIP text embedding"""
    for obj in [rae, getattr(rae, "encoder", None)]:
        if obj is None:
            continue
        for name in ["encode_text", "get_text_features", "text_features", "text_embed", "embed_text"]:
            fn = getattr(obj, name, None)
            if callable(fn):
                out = fn([text] if isinstance(text, str) else text)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                if out.dim() == 1:
                    out = out.unsqueeze(0)
                return out.to(device)
    
    # Fallback to HuggingFace
    try:
        from transformers import AutoProcessor, AutoModel
        model_name = "google/siglip2-base-patch16-256"
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
        
        inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            if hasattr(model, "get_text_features"):
                e = model.get_text_features(**inputs)
            else:
                out = model(**inputs)
                e = getattr(out, "text_embeds", None)
        if e.dim() == 1:
            e = e.unsqueeze(0)
        return e
    except Exception as e:
        raise RuntimeError(f"Could not get text embedding: {e}")


def compute_patch_attention(z_bchw: torch.Tensor, e_bc: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
    """
    Compute patch-wise attention weights based on text query
    
    Args:
        z_bchw: (B, C, H, W) - latent features
        e_bc: (B, C) - text embedding
        tau: temperature for softmax
    
    Returns:
        attention_map: (B, H, W) - attention weights
    """
    B, C, H, W = z_bchw.shape
    z = z_bchw.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, HW, C)
    
    # Normalize
    z_n = F.normalize(z, dim=-1)  # (B, HW, C)
    e_n = F.normalize(e_bc, dim=-1)  # (B, C)
    e_n = e_n.unsqueeze(1)  # (B, 1, C)
    
    # Cosine similarity per patch
    sims = (z_n * e_n).sum(dim=-1)  # (B, HW)
    
    # Softmax attention
    attention = F.softmax(sims / max(tau, 1e-6), dim=-1)  # (B, HW)
    
    # Reshape to spatial
    attention_map = attention.reshape(B, H, W)
    
    return attention_map


def visualize_attention_comparison(
    image_orig: torch.Tensor,
    z_before: torch.Tensor,
    z_after: torch.Tensor,
    text_query: str,
    text_embed: torch.Tensor,
    tau: float,
    save_path: str
):
    """
    Visualize attention maps before and after optimization
    
    Args:
        image_orig: (1, 3, 256, 256) - original image
        z_before: (1, C, H, W) - latent before optimization
        z_after: (1, C, H, W) - latent after optimization
        text_query: text query string
        text_embed: (1, C) - text embedding
        tau: temperature parameter
        save_path: output path
    """
    # Compute attention maps
    with torch.no_grad():
        attn_before = compute_patch_attention(z_before, text_embed, tau)  # (1, H, W)
        attn_after = compute_patch_attention(z_after, text_embed, tau)
    
    # To numpy
    img = image_orig.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    
    attn_before = attn_before.squeeze(0).cpu().numpy()
    attn_after = attn_after.squeeze(0).cpu().numpy()
    
    # Upsample attention to image size
    from scipy.ndimage import zoom
    h, w = attn_before.shape
    scale_h = 256 / h
    scale_w = 256 / w
    attn_before_up = zoom(attn_before, (scale_h, scale_w), order=1)
    attn_after_up = zoom(attn_after, (scale_h, scale_w), order=1)
    
    # Create figure
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Row 1: Before optimization
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(attn_before, cmap='hot', interpolation='nearest')
    ax2.set_title('Attention (Before)\nPatch-level', fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(img)
    im3 = ax3.imshow(attn_before_up, cmap='hot', alpha=0.5, interpolation='bilinear')
    ax3.set_title('Attention Overlay (Before)', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(attn_before_up, cmap='viridis', interpolation='bilinear')
    ax4.set_title('Attention Heatmap (Before)', fontsize=14, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    # Row 2: After optimization
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(img)
    ax5.set_title('Original Image', fontsize=14, fontweight='bold')
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 1])
    im6 = ax6.imshow(attn_after, cmap='hot', interpolation='nearest')
    ax6.set_title('Attention (After)\nPatch-level', fontsize=14, fontweight='bold', color='red')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.imshow(img)
    im7 = ax7.imshow(attn_after_up, cmap='hot', alpha=0.5, interpolation='bilinear')
    ax7.set_title('Attention Overlay (After)', fontsize=14, fontweight='bold', color='red')
    ax7.axis('off')
    
    ax8 = fig.add_subplot(gs[1, 3])
    im8 = ax8.imshow(attn_after_up, cmap='viridis', interpolation='bilinear')
    ax8.set_title('Attention Heatmap (After)', fontsize=14, fontweight='bold', color='red')
    ax8.axis('off')
    plt.colorbar(im8, ax=ax8, fraction=0.046)
    
    # Compute attention shift statistics
    max_before = attn_before.max()
    max_after = attn_after.max()
    entropy_before = -np.sum(attn_before * np.log(attn_before + 1e-10))
    entropy_after = -np.sum(attn_after * np.log(attn_after + 1e-10))
    
    plt.suptitle(
        f'Text Query: "{text_query}"\n'
        f'Max Attention: {max_before:.4f} → {max_after:.4f} | '
        f'Entropy: {entropy_before:.2f} → {entropy_after:.2f}',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[info] Attention visualization saved: {save_path}")
    plt.close()


def visualize_top_patches(
    image_orig: torch.Tensor,
    z_before: torch.Tensor,
    z_after: torch.Tensor,
    text_query: str,
    text_embed: torch.Tensor,
    tau: float,
    save_path: str,
    top_k: int = 9
):
    """
    Visualize top-K attended patches before and after optimization
    """
    with torch.no_grad():
        attn_before = compute_patch_attention(z_before, text_embed, tau).squeeze(0).cpu().numpy()
        attn_after = compute_patch_attention(z_after, text_embed, tau).squeeze(0).cpu().numpy()
    
    img = image_orig.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    
    H, W = attn_before.shape
    patch_h = 256 // H
    patch_w = 256 // W
    
    # Get top-K patches
    flat_before = attn_before.flatten()
    flat_after = attn_after.flatten()
    
    top_idx_before = np.argsort(flat_before)[-top_k:][::-1]
    top_idx_after = np.argsort(flat_after)[-top_k:][::-1]
    
    fig, axes = plt.subplots(2, top_k, figsize=(top_k * 2, 5))
    
    for i, (idx_b, idx_a) in enumerate(zip(top_idx_before, top_idx_after)):
        # Before
        py_b, px_b = divmod(idx_b, W)
        y1_b, y2_b = py_b * patch_h, (py_b + 1) * patch_h
        x1_b, x2_b = px_b * patch_w, (px_b + 1) * patch_w
        patch_b = img[y1_b:y2_b, x1_b:x2_b]
        
        axes[0, i].imshow(patch_b)
        axes[0, i].set_title(f'{flat_before[idx_b]:.3f}', fontsize=10)
        axes[0, i].axis('off')
        
        # After
        py_a, px_a = divmod(idx_a, W)
        y1_a, y2_a = py_a * patch_h, (py_a + 1) * patch_h
        x1_a, x2_a = px_a * patch_w, (px_a + 1) * patch_w
        patch_a = img[y1_a:y2_a, x1_a:x2_a]
        
        axes[1, i].imshow(patch_a)
        axes[1, i].set_title(f'{flat_after[idx_a]:.3f}', fontsize=10, color='red')
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('Before', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('After', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Top-{top_k} Attended Patches\nQuery: "{text_query}"', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[info] Top patches visualization saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize attention maps before/after optimization")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--text", type=str, required=True, help="Text query")
    parser.add_argument("--latents", type=str, required=True, help="Path to saved latents (.pt file)")
    parser.add_argument("--tau", type=float, default=0.05, help="Temperature for attention")
    parser.add_argument("--save_prefix", type=str, default="out/viz", help="Output prefix")
    parser.add_argument("--top_k", type=int, default=9, help="Number of top patches to visualize")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.save_prefix), exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load config and RAE
    rae_config, _, _, _, _, _, _ = parse_configs(args.config)
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()
    
    # Load image
    x = load_image_256(args.image, device)
    
    # Load saved latents
    latents_data = torch.load(args.latents, map_location=device)
    z_before = latents_data['z0'].to(device)
    z_after = latents_data['z_star'].to(device)
    text_query = latents_data.get('text', args.text)
    
    print(f"[info] Loaded latents:")
    print(f"  z_before shape: {tuple(z_before.shape)}")
    print(f"  z_after shape: {tuple(z_after.shape)}")
    print(f"  text query: {text_query}")
    
    # Get text embedding
    with torch.no_grad():
        text_embed = get_siglip_text_embed(rae, text_query, device)
        if text_embed.shape[0] == 1 and z_before.shape[0] > 1:
            text_embed = text_embed.repeat(z_before.shape[0], 1)
    
    print(f"[info] Text embedding shape: {tuple(text_embed.shape)}")
    
    # Visualize attention comparison
    print("[info] Creating attention comparison visualization...")
    visualize_attention_comparison(
        image_orig=x,
        z_before=z_before,
        z_after=z_after,
        text_query=text_query,
        text_embed=text_embed,
        tau=args.tau,
        save_path=f"{args.save_prefix}_attention_comparison.png"
    )
    
    # Visualize top patches
    print("[info] Creating top patches visualization...")
    visualize_top_patches(
        image_orig=x,
        z_before=z_before,
        z_after=z_after,
        text_query=text_query,
        text_embed=text_embed,
        tau=args.tau,
        save_path=f"{args.save_prefix}_top_patches.png",
        top_k=args.top_k
    )
    
    print(f"[info] All visualizations saved with prefix: {args.save_prefix}")


if __name__ == "__main__":
    main()


# Usage:
# python visualize_attention.py \
#   --config configs/stage2/sampling/ImageNet256/sample_DiTDH-XL_SigLIP2.yaml \
#   --image dog.png \
#   --text "maltese's face" \
#   --latents out/run3_latents.pt \
#   --tau 0.05 \
#   --save_prefix out/run3_viz