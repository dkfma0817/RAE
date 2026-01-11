# src/probe_text_attention.py

import torch
import yaml
from omegaconf import OmegaConf
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

from transformers import AutoProcessor, AutoModel

# --- RAE imports
from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs
from stage1 import RAE


# ============================================
# Stage 1: RAE Loading
# ============================================

@torch.no_grad()
def load_stage1_rae(config_path, device):
    """
    RAE 로드 (checkpoint는 자동으로 config에서 처리됨)
    
    중요: RAE는 instantiate_from_config 내부에서 자동으로 checkpoint를 로드합니다.
    config 파일에 pretrained_decoder_path가 지정되어 있으면 자동 로드됩니다.
    """
    rae_config, *_ = parse_configs(config_path)
    if rae_config is None:
        raise ValueError(
            f"No stage_1 section found in config {config_path}. "
            "Please supply a config with a stage_1 target."
        )
    
    # instantiate_from_config가 내부적으로 checkpoint 로드
    rae: RAE = instantiate_from_config(rae_config).to(device)
    
    rae.eval()
    rae.requires_grad_(False)
    
    print("✅ RAE loaded successfully")
    print(f"   Encoder: {type(rae.encoder).__name__}")
    print(f"   Decoder: {type(rae.decoder).__name__}")
    
    return rae


# ============================================
# Stage 2: Image Encoding
# ============================================

@torch.no_grad()
def encode_image(rae, image_path, device):
    """
    이미지를 RAE로 인코딩
    
    Returns:
        z: latent representation
        image: PIL Image
        latent_shape: original shape info (for reshape later)
    """
    image = Image.open(image_path).convert("RGB")

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(256),
        T.ToTensor(),
    ])

    x = transform(image).unsqueeze(0).to(device)

    # Stage1 encoder
    z = rae.encode(x)
    
    # 🔧 여기가 중요: z의 shape 확인
    print(f"Original latent shape: {z.shape}")
    
    original_shape = z.shape
    
    # Case 1: (1, C, H, W) - spatial latent
    if len(z.shape) == 4:
        B, C, H, W = z.shape
        z_flat = z.reshape(B, C, H * W).permute(0, 2, 1)  # (1, H*W, C)
        z_flat = z_flat.squeeze(0)  # (H*W, C)
        spatial_shape = (H, W)
        print(f"Spatial latent detected: {H}x{W}, {C} channels")
    
    # Case 2: (1, N_tokens, C) - already tokenized
    elif len(z.shape) == 3:
        B, N, C = z.shape
        z_flat = z.squeeze(0)  # (N, C)
        # Assume square spatial layout
        H = W = int(np.sqrt(N))
        spatial_shape = (H, W)
        print(f"Token latent detected: {N} tokens, {C} channels")
    
    # Case 3: (1, C) - global feature
    else:
        raise ValueError(f"Unexpected latent shape: {z.shape}")
    
    return z_flat, image, spatial_shape


# ============================================
# Stage 3: Text Encoding
# ============================================

# @torch.no_grad()
# def load_text_encoder(device):
#     """SigLIP text encoder 로드"""
#     name = "google/siglip2-base-patch16-256"
#     processor = AutoProcessor.from_pretrained(name)
#     model = AutoModel.from_pretrained(name).to(device).eval()
#     model.requires_grad_(False)
#     return processor, model

@torch.no_grad()
def load_text_encoder(device):
    name = "google/siglip2-base-patch16-256"
    processor = AutoProcessor.from_pretrained(
        name, local_files_only=True
    )
    model = AutoModel.from_pretrained(
        name, local_files_only=True
    ).to(device).eval()
    model.requires_grad_(False)
    return processor, model



@torch.no_grad()
def encode_text(processor, model, text, device):
    """
    텍스트를 SigLIP으로 인코딩
    
    Args:
        text: str or list of str
    Returns:
        t: (n_texts, d) tensor
    """
    if isinstance(text, str):
        text = [text]
    
    inputs = processor(text=text, return_tensors="pt", padding=True).to(device)
    t = model.get_text_features(**inputs)  # (n_texts, d)
    
    return t


# ============================================
# Stage 4: Relevance Computation
# ============================================

def compute_relevance(z, t):
    """
    이미지 latent와 텍스트 간 relevance 계산
    
    Args:
        z: (N_spatial, d) - 이미지 spatial tokens
        t: (n_texts, d) - 텍스트 embeddings
    
    Returns:
        scores: (n_texts, N_spatial) - 각 텍스트에 대한 spatial attention
    """
    # L2 normalize
    z_norm = z / z.norm(dim=-1, keepdim=True)  # (N_spatial, d)
    t_norm = t / t.norm(dim=-1, keepdim=True)  # (n_texts, d)
    
    # Cosine similarity: (n_texts, d) @ (d, N_spatial) = (n_texts, N_spatial)
    scores = t_norm @ z_norm.T
    
    return scores


# ============================================
# Stage 5: Visualization
# ============================================

def visualize_attention(image, attention_maps, concepts, spatial_shape):
    """
    Attention maps 시각화
    
    Args:
        image: PIL Image
        attention_maps: (n_concepts, H, W) tensor
        concepts: list of str
        spatial_shape: (H, W)
    """
    n_concepts = len(concepts)
    n_cols = min(4, n_concepts + 1)
    n_rows = (n_concepts + n_cols) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Each concept's attention
    img_w, img_h = image.size
    
    for idx, (concept, attn_map) in enumerate(zip(concepts, attention_maps)):
        # Upscale attention map to image size
        H, W = spatial_shape
        attn_2d = attn_map.cpu().numpy()
        
        # Zoom to image size
        scale_h = img_h / H
        scale_w = img_w / W
        attn_upscaled = zoom(attn_2d, (scale_h, scale_w), order=1)
        
        # Normalize
        attn_norm = (attn_upscaled - attn_upscaled.min()) / (attn_upscaled.max() - attn_upscaled.min() + 1e-8)
        
        # Plot
        axes[idx + 1].imshow(image, alpha=0.5)
        im = axes[idx + 1].imshow(attn_norm, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        axes[idx + 1].set_title(f'"{concept}"', fontsize=11, fontweight='bold')
        axes[idx + 1].axis('off')
    
    # Hide extra subplots
    for idx in range(n_concepts + 1, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig("rae_text_attention.png", dpi=150, bbox_inches='tight')
    print("✅ Saved: rae_text_attention.png")
    plt.show()


# ============================================
# Main Function
# ============================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 🔧 설정 - checkpoint path는 제거 (config에서 자동 로드)
    stage1_config = "configs/stage1/pretrained/SigLIP2.yaml"
    image_path    = "../RL/data/train2014/COCO_train2014_000000000036.jpg"

    # 여러 개념 동시 분석
    concepts = [
        "What is the woman holding?",
        "What color is the umbrella?",
        "the woman",
        "the background",
        "the pink umbrella",
    ]


    print("="*60)
    print("RAE + Text Attention Analysis")
    print("="*60)

    # 1) Load models - checkpoint path 제거
    print("\n📦 Loading models...")
    rae = load_stage1_rae(stage1_config, device)
    processor, text_model = load_text_encoder(device)

    # 2) Encode image
    print("\n🖼️  Encoding image...")
    z, image, spatial_shape = encode_image(rae, image_path, device)
    print(f"Flattened latent shape: {z.shape}")
    print(f"Spatial shape: {spatial_shape}")

    # 3) Encode texts
    print(f"\n📝 Encoding {len(concepts)} concepts...")
    t = encode_text(processor, text_model, concepts, device)
    print(f"Text embeddings shape: {t.shape}")

    # 4) Compute relevance
    print("\n🔍 Computing attention...")
    scores = compute_relevance(z, t)  # (n_concepts, N_spatial)
    print(f"Scores shape: {scores.shape}")

    # 5) Reshape to spatial maps
    H, W = spatial_shape
    attention_maps = scores.reshape(len(concepts), H, W)

    # 6) Print statistics
    print("\n📊 Attention Statistics:")
    for i, concept in enumerate(concepts):
        max_val = scores[i].max().item()
        mean_val = scores[i].mean().item()
        top5_indices = scores[i].topk(5).indices.tolist()
        print(f"  {concept:15s}: max={max_val:.3f}, mean={mean_val:.3f}")
        print(f"    Top-5 indices: {top5_indices}")

    # 7) Visualize
    print("\n🎨 Visualizing...")
    visualize_attention(image, attention_maps, concepts, spatial_shape)


if __name__ == "__main__":
    main()