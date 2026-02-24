# import torch
# from PIL import Image
# from torchvision import transforms

# from train_adapter import VQToGPT2   # 네 train 파일 이름

# device = "cuda"

# model = VQToGPT2(
#     rae_ckpt_path="pca_results/008-RAE/checkpoints/1280000.pt",
#     rae_config_path="configs/stage1/training/DINOv2-B_decXL.yaml",
# ).to(device)

# ckpt = torch.load("adapter_ckpts/flickr30k_vq1024_gpt2/adapter_last.pt")
# model.adapter.load_state_dict(ckpt["adapter"])

# model.eval()

# # imagenette2/train/n02102040/ILSVRC2012_val_00000665.JPEG
# # imagenette2/train/n01440764/n01440764_18.JPEG

# img = Image.open("imagenette2/train/n02102040/ILSVRC2012_val_00000665.JPEG").convert("RGB")

# tfm = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(256),
#     transforms.ToTensor()
# ])

# img = tfm(img).unsqueeze(0).to(device)

# caps = model.generate(img, prompt="A photo of", max_new_tokens=30)

# print(caps[0])







import os
import torch
from PIL import Image
from torchvision import transforms

from train_adapter import VQToGPT2  # 학습에 쓴 그 파일/클래스랑 동일해야 함

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_DIR = "eval_imgs"

MODELS = {
    "crossattn16_imagenet": "adapter_ckpts/flickr30k_v3_imagenet/adapter_last.pt",
    # 필요하면 추가:
    # "step3000": "adapter_ckpts/flickr30k_v3_imagenet/adapter_step0003000.pt",
}

RAE_CKPT = "pca_results/008-RAE/checkpoints/1280000.pt"
RAE_CFG = "configs/stage1/training/DINOv2-B_decXL.yaml"

# ✅ 학습과 동일하게 맞추기 (지금은 imagenet)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_images():
    imgs = []
    names = []
    for fn in sorted(os.listdir(IMAGE_DIR)):
        path = os.path.join(IMAGE_DIR, fn)
        img = Image.open(path).convert("RGB")
        img = tf(img)
        imgs.append(img)
        names.append(fn)
    return torch.stack(imgs).to(DEVICE), names


@torch.no_grad()
def main():
    images, names = load_images()

    for tag, ckpt_path in MODELS.items():
        print("\n" + "=" * 60)
        print(f"MODEL = {tag}")
        print("=" * 60)

        # ✅ 학습 때 설정이랑 맞추기: num_prefix=16, max_text_len=32 등
        model = VQToGPT2(
            rae_ckpt_path=RAE_CKPT,
            rae_config_path=RAE_CFG,
            gpt_model_name="gpt2",
            max_text_len=32,
            use_adapter_mlp=True,
            num_prefix=16,   # ⭐ cross-attn 버전이면 꼭!
        ).to(DEVICE)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.adapter.load_state_dict(ckpt["adapter"], strict=True)
        model.eval()

        # (옵션) VQ 인덱스 체크
        idx = model.get_vq_indices(images)  # [B, 324]
        for i, name in enumerate(names):
            u = idx[i].unique().numel()
            print(f"[{name}] unique_codes={u} min={idx[i].min().item()} max={idx[i].max().item()}")

        # ✅ 조건부 확인용: prompt 없이 + greedy 권장
        outputs = model.generate(
            images,
            prompt="",              # 중요: prompt가 덮지 않게
            max_new_tokens=25,
        )

        for name, cap in zip(names, outputs):
            print(f"[{name}] {cap}")


if __name__ == "__main__":
    main()

