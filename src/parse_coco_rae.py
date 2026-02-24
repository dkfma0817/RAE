import os
import json
import pickle
import argparse
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


# -------------------------------------------------
# Load trained RAE
# -------------------------------------------------
def load_rae(rae_config, rae_ckpt, device):
    from omegaconf import OmegaConf
    from utils.model_utils import instantiate_from_config

    conf = OmegaConf.load(rae_config)
    model_conf = conf.stage_1 if "stage_1" in conf else conf

    model = instantiate_from_config(model_conf).to(device).eval()

    ckpt = torch.load(rae_ckpt, map_location="cpu")

    # stage1 trainer checkpoint 대응
    if isinstance(ckpt, dict):
        sd = ckpt.get("ema", ckpt.get("model", ckpt))
    else:
        sd = ckpt

    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)

    for p in model.parameters():
        p.requires_grad = False

    return model


# -------------------------------------------------
# Extract VQ indices
# -------------------------------------------------
@torch.no_grad()
def rae_vq_indices(model, x):
    """
    x: (B,3,H,W) in [0,1]
    returns: (B,Hq,Wq)
    """

    # ---- encode ----
    z = model.encode(x)  # (B,C,H,W)

    # ---- PCA reweight ----
    if hasattr(model, "pca_reweight") and model.pca_reweight is not None:
        z = model.pca_reweight(z)

    # ---- project ----
    if hasattr(model, "vq_pre") and model.vq_pre is not None:
        z = model.vq_pre(z)

    # ---- z norm ----
    if hasattr(model, "vq_z_norm") and model.vq_z_norm:
        denom = z.std(dim=(1,2,3), keepdim=True).clamp(min=1e-6)
        z = z / denom

    # ---- VQ ----
    out = model.vq_layer(z)

    if isinstance(out, (tuple, list)) and len(out) >= 3:
        idx = out[2]
    elif hasattr(model, "last_vq_indices"):
        idx = model.last_vq_indices
    else:
        raise RuntimeError("Cannot extract VQ indices")

    if idx.dim() == 2:
        B, N = idx.shape
        H = W = int(N ** 0.5)
        idx = idx.view(B, H, W)

    return idx.long()


# -------------------------------------------------
# Get codebook embedding matrix
# -------------------------------------------------
def get_codebook_embeddings(model):

    vq = model.vq_layer

    if hasattr(vq, "embedding"):
        return vq.embedding.weight

    if hasattr(vq, "codebook"):
        return vq.codebook.weight

    if hasattr(vq, "quantize"):
        return vq.quantize.embedding.weight

    raise RuntimeError("Cannot locate codebook embedding weights")


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--ann", required=True)
    parser.add_argument("--img_root", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument("--rae_config", required=True)
    parser.add_argument("--rae_ckpt", required=True)

    # 🔥 COCO image meta 추가
    parser.add_argument("--coco_images_json", required=True, nargs="+",
                    help="e.g., captions_train2014.json captions_val2014.json")
    
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch", type=int, default=64)

    parser.add_argument("--proj_dim", type=int, default=0)
    parser.add_argument("--dtype", default="fp16", choices=["fp16","fp32"])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------
    # Load RAE
    # -------------------------------------------------
    model = load_rae(args.rae_config, args.rae_ckpt, device)

    codebook = get_codebook_embeddings(model).to(device)
    Dcb = codebook.shape[1]

    proj = None
    if args.proj_dim and args.proj_dim != Dcb:
        proj = nn.Linear(Dcb, args.proj_dim, bias=False).to(device).eval()

    tfm = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
    ])

    # -------------------------------------------------
    # Load captions (LIST FORMAT)
    # -------------------------------------------------
    with open(args.ann, "r") as f:
        captions = json.load(f)

    # -------------------------------------------------
    # Load image meta (id → file_name)
    # -------------------------------------------------
    with open(args.coco_images_json, "r") as f:
        coco = json.load(f)

    id2file = {}
    for meta_path in args.coco_images_json:
        with open(meta_path, "r") as f:
            coco = json.load(f)
        for im in coco["images"]:
            id2file[im["id"]] = im["file_name"]
    # unique image ids
    image_ids = sorted({c["image_id"] for c in captions})

    # -------------------------------------------------
    # Path resolver
    # -------------------------------------------------
    def resolve_path(fname):

        p = os.path.join(args.img_root, fname)
        if os.path.exists(p): return p

        p2 = os.path.join(args.img_root, "train2014", fname)
        if os.path.exists(p2): return p2

        p3 = os.path.join(args.img_root, "val2014", fname)
        if os.path.exists(p3): return p3

        raise FileNotFoundError(fname)

    # -------------------------------------------------
    # Image → RAE-VQ embedding
    # -------------------------------------------------
    fname2eidx = {}
    imageid2eidx = {}
    embeddings = []

    batch_imgs, batch_ids = [], []

    for image_id in tqdm(image_ids, desc="Encoding images"):

        fname = id2file[image_id]

        path = resolve_path(fname)
        img = Image.open(path).convert("RGB")
        x = tfm(img)

        batch_imgs.append(x)
        batch_ids.append(image_id)

        if len(batch_imgs) >= args.batch:

            xbat = torch.stack(batch_imgs).to(device)

            idx = rae_vq_indices(model, xbat)
            flat = idx.view(idx.size(0), -1)

            emb = codebook[flat]
            emb = F.normalize(emb, dim=-1)
            emb = emb.mean(dim=1)
            emb = F.normalize(emb, dim=-1)

            if proj is not None:
                emb = proj(emb)

            emb = emb.half() if args.dtype == "fp16" else emb.float()

            for i, iid in enumerate(batch_ids):
                imageid2eidx[iid] = len(embeddings)
                embeddings.append(emb[i].cpu())

            batch_imgs, batch_ids = [], []

    # leftover
    if len(batch_imgs) > 0:

        xbat = torch.stack(batch_imgs).to(device)
        idx = rae_vq_indices(model, xbat)

        flat = idx.view(idx.size(0), -1)

        emb = codebook[flat]
        emb = F.normalize(emb, dim=-1)
        emb = emb.mean(dim=1)
        emb = F.normalize(emb, dim=-1)

        if proj is not None:
            emb = proj(emb)

        emb = emb.half() if args.dtype == "fp16" else emb.float()

        for i, iid in enumerate(batch_ids):
            imageid2eidx[iid] = len(embeddings)
            embeddings.append(emb[i].cpu())

    clip_embedding = torch.stack(embeddings)

    # -------------------------------------------------
    # Caption list 생성
    # -------------------------------------------------
    captions_out = []

    for cid, c in enumerate(captions):

        image_id = c["image_id"]
        cap = c["caption"]

        captions_out.append({
            "image_id": image_id,
            "id": cid,
            "caption": cap,
            "clip_embedding": imageid2eidx[image_id],
        })

    # -------------------------------------------------
    # Save
    # -------------------------------------------------
    out = {
        "clip_embedding": clip_embedding,
        "captions": captions_out,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "wb") as f:
        pickle.dump(out, f)

    print("\n[DONE]")
    print(" embedding:", tuple(clip_embedding.shape))
    print(" captions:", len(captions_out))