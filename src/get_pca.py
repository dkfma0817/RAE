import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
from sklearn.decomposition import PCA
from omegaconf import OmegaConf

from utils.train_utils import prepare_dataloader, parse_configs
from utils.model_utils import instantiate_from_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--image-size", type=int, default=256)

    # single GPU / small footprint defaults
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=6)

    # token budget (NOT images)
    p.add_argument("--num-tokens", type=int, default=200_000,
                   help="Total patch tokens to collect for PCA (NOT images).")
    p.add_argument("--tokens-per-batch", type=int, default=2048,
                   help="How many patch tokens to sample per batch.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-path", type=str, default="pca_stats.pth")
    p.add_argument("--fp16", action="store_true")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    # load model config
    (rae_config, *_) = parse_configs(args.config)
    _ = OmegaConf.load(args.config)  # keep parity with your repo; not strictly needed

    # dataloader (same as train_stage1.py, but single GPU => rank=0, world_size=1)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    loader, _ = prepare_dataloader(
        args.data_path,
        args.batch_size,
        args.num_workers,
        rank=0,
        world_size=1,
        transform=transform,
    )

    # model
    rae = instantiate_from_config(rae_config).to(device).eval()

    # IMPORTANT: deterministic latent collection
    if hasattr(rae, "noise_tau"):
        rae.noise_tau = 0.0
    # collect PCA on continuous latent (pre-VQ)
    if hasattr(rae, "use_vq") and rae.use_vq:
        rae.use_vq = False

    feats_buf = []
    total = 0
    feat_dim = None

    pbar = tqdm(loader, desc="Collect tokens")
    for images, _ in pbar:
        images = images.to(device, non_blocking=True)

        if args.fp16 and device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                z = rae.encode(images)
        else:
            z = rae.encode(images)

        # z: (B,C,H,W) or (B,N,C)
        if z.dim() == 4:
            B, C, H, W = z.shape
            flat = z.permute(0, 2, 3, 1).reshape(B * H * W, C)
        elif z.dim() == 3:
            B, N, C = z.shape
            flat = z.reshape(B * N, C)
        else:
            raise ValueError(f"Unexpected latent shape {tuple(z.shape)}")

        if feat_dim is None:
            feat_dim = flat.shape[-1]

        flat = flat.float().cpu().numpy()

        need = args.num_tokens - total
        if need <= 0:
            break

        take = min(args.tokens_per_batch, flat.shape[0], need)
        if take < flat.shape[0]:
            idx = rng.choice(flat.shape[0], size=take, replace=False)
            flat = flat[idx]
        else:
            flat = flat[:take]

        feats_buf.append(flat)
        total += flat.shape[0]
        pbar.set_postfix(tokens=total)

        if total >= args.num_tokens:
            break

    all_feats = np.concatenate(feats_buf, axis=0)
    print(f">>> PCA fit on {all_feats.shape} (tokens, dim)")

    # Full-D PCA (reweighting용)
    pca = PCA(n_components=all_feats.shape[1], svd_solver="randomized")
    pca.fit(all_feats)

    stats = {
        "mean": torch.tensor(pca.mean_, dtype=torch.float32),
        "comp": torch.tensor(pca.components_, dtype=torch.float32),
        "var": torch.tensor(pca.explained_variance_, dtype=torch.float32),
        "meta": {
            "num_tokens": int(all_feats.shape[0]),
            "dim": int(all_feats.shape[1]),
            "image_size": int(args.image_size),
            "batch_size": int(args.batch_size),
            "tokens_per_batch": int(args.tokens_per_batch),
        }
    }

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save(stats, args.save_path)
    print(f">>> Saved to {args.save_path}")


if __name__ == "__main__":
    main()
