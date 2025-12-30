from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
import os

class ArrowImageNetDataset(Dataset):
    def __init__(self, arrow_dir, split="train", transform=None):
        self.transform = transform

        pattern = os.path.join(arrow_dir, f"imagenet-1k-{split}-*.arrow")
        files = sorted(glob.glob(pattern))
        if len(files) == 0:
            raise FileNotFoundError(f"No arrow shards found with pattern: {pattern}")

        # arrow shard 파일 리스트로 데이터셋 로드
        self.dataset = load_dataset(
            "arrow",
            data_files={split: files},
            split=split,
        )

        # image column이 datasets.Image로 decode 되게 (보통 자동이지만 안전빵)
        if "image" in self.dataset.column_names:
            self.dataset = self.dataset.cast_column("image", self.dataset.features["image"])

        print(f"Loaded split={split}, num={len(self.dataset)} from {arrow_dir}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = int(item["label"])
        return image, label
