from omegaconf import OmegaConf, DictConfig
from typing import List, Tuple, Union
from PIL import Image
import numpy as np
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pathlib import Path
from copy import deepcopy
from torch.cuda.amp import GradScaler

from torch.utils.data import Dataset
from datasets import load_dataset




def parse_configs(config: Union[DictConfig, str]) -> Tuple[DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig]:
    """Load a config file and return component sections as DictConfigs."""
    if isinstance(config, str):
        config = OmegaConf.load(config)
    rae_config = config.get("stage_1", None)
    stage2_config = config.get("stage_2", None)
    transport_config = config.get("transport", None)
    sampler_config = config.get("sampler", None)
    guidance_config = config.get("guidance", None)
    misc = config.get("misc", None)
    training_config = config.get("training", None)
    eval_config = config.get("eval", None)
    return rae_config, stage2_config, transport_config, sampler_config, guidance_config, misc, training_config, eval_config

def none_or_str(value):
    if value == 'None':
        return None
    return value

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def prepare_dataloader(
    data_path: Path,
    batch_size: int,
    workers: int,
    rank: int,
    world_size: int,
    transform: List = None,
):
    parquet_glob = str(data_path)  # e.g. "/dataset/imagenet-1k/data/train-*.parquet"
    dataset = HFParquetImageNet(parquet_glob=parquet_glob, transform=transform, split="train")

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, sampler


def get_autocast_scaler(args) -> Tuple[dict, torch.cuda.amp.GradScaler | None]:
    if args.precision == "fp16":
        scaler = GradScaler()
        autocast_kwargs = dict(enabled=True, dtype=torch.float16)
    elif args.precision == "bf16":
        scaler = None
        autocast_kwargs = dict(enabled=True, dtype=torch.bfloat16)
    else:
        scaler = None
        autocast_kwargs = dict(enabled=False)
    
    return scaler, autocast_kwargs


class HFParquetImageNet(Dataset):
    """
    ImageNet-1k parquet shards loaded via HuggingFace datasets.
    Returns (image_tensor, label_int) to match ImageFolder behavior.
    """
    def __init__(self, parquet_glob: str, transform=None, split: str = "train"):
        self.ds = load_dataset(
            "parquet",
            data_files={split: parquet_glob},
            split=split,
        )
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex["image"].convert("RGB")  # your columns: image, label
        if self.transform is not None:
            img = self.transform(img)
        label = int(ex["label"])
        return img, label






# ## 이미지폴더로 되어있을 때

# from omegaconf import OmegaConf, DictConfig
# from typing import List, Tuple, Union
# from PIL import Image
# import numpy as np
# from collections import OrderedDict
# import torch
# from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
# from torchvision import transforms
# from torchvision.datasets import ImageFolder
# from pathlib import Path
# from copy import deepcopy
# # from .dist_utils import setup_distributed
# from torch.cuda.amp import GradScaler




# def parse_configs(config: Union[DictConfig, str]) -> Tuple[DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig]:
#     """Load a config file and return component sections as DictConfigs."""
#     if isinstance(config, str):
#         config = OmegaConf.load(config)
#     rae_config = config.get("stage_1", None)
#     stage2_config = config.get("stage_2", None)
#     transport_config = config.get("transport", None)
#     sampler_config = config.get("sampler", None)
#     guidance_config = config.get("guidance", None)
#     misc = config.get("misc", None)
#     training_config = config.get("training", None)
#     eval_config = config.get("eval", None)
#     return rae_config, stage2_config, transport_config, sampler_config, guidance_config, misc, training_config, eval_config

# def none_or_str(value):
#     if value == 'None':
#         return None
#     return value

# def center_crop_arr(pil_image, image_size):
#     """
#     Center cropping implementation from ADM.
#     https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
#     """
#     while min(*pil_image.size) >= 2 * image_size:
#         pil_image = pil_image.resize(
#             tuple(x // 2 for x in pil_image.size), resample=Image.BOX
#         )

#     scale = image_size / min(*pil_image.size)
#     pil_image = pil_image.resize(
#         tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
#     )

#     arr = np.array(pil_image)
#     crop_y = (arr.shape[0] - image_size) // 2
#     crop_x = (arr.shape[1] - image_size) // 2
#     return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

# #################################################################################
# #                             Training Helper Functions                         #
# #################################################################################

# def requires_grad(model, flag=True):
#     """
#     Set requires_grad flag for all parameters in a model.
#     """
#     for p in model.parameters():
#         p.requires_grad = flag

# @torch.no_grad()
# def update_ema(ema_model, model, decay=0.9999):
#     """
#     Step the EMA model towards the current model.
#     """
#     ema_params = OrderedDict(ema_model.named_parameters())
#     model_params = OrderedDict(model.named_parameters())

#     for name, param in model_params.items():
#         # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
#         ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

# def prepare_dataloader(
#     data_path: Path,
#     batch_size: int,
#     workers: int,
#     rank: int,
#     world_size: int,
#     transform=None,
# ):
#     dataset = ImageFolder(str(data_path), transform=transform)

#     if world_size > 1:
#         sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
#         shuffle = False
#     else:
#         sampler = None
#         shuffle = True

#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         sampler=sampler,
#         num_workers=workers,
#         pin_memory=True,
#         drop_last=True,
#     )
#     return loader, sampler

# def get_autocast_scaler(args) -> Tuple[dict, torch.cuda.amp.GradScaler | None]:
#     if args.precision == "fp16":
#         scaler = GradScaler()
#         autocast_kwargs = dict(enabled=True, dtype=torch.float16)
#     elif args.precision == "bf16":
#         scaler = None
#         autocast_kwargs = dict(enabled=True, dtype=torch.bfloat16)
#     else:
#         scaler = None
#         autocast_kwargs = dict(enabled=False)
    
#     return scaler, autocast_kwargs