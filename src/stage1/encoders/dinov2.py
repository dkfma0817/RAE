from transformers import Dinov2WithRegistersModel
from torch import nn
import torch
from math import *
from . import register_encoder


@register_encoder()
class Dinov2withNorm(nn.Module):
    def __init__(
        self,
        dinov2_path: str,
        normalize: bool = True,
    ):
        super().__init__()
        # Support both local paths and HuggingFace model IDs
        try:
            self.encoder = Dinov2WithRegistersModel.from_pretrained(dinov2_path, local_files_only=True)
        except (OSError, ValueError, AttributeError):
            self.encoder = Dinov2WithRegistersModel.from_pretrained(dinov2_path, local_files_only=False)
        self.encoder.requires_grad_(False)
        if normalize:
            self.encoder.layernorm.elementwise_affine = False
            self.encoder.layernorm.weight = None
            self.encoder.layernorm.bias = None
        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size
        
    def dinov2_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x, output_hidden_states=True)
        unused_token_num = 5  # 1 CLS + 4 register tokens
        image_features = x.last_hidden_state[:, unused_token_num:]
        return image_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dinov2_forward(x)






## DINO-Tok 처럼 shallow, deep 둘 다 뽑기

# from transformers import Dinov2WithRegistersModel
# from torch import nn
# import torch
# from . import register_encoder


# @register_encoder()
# class Dinov2withNorm(nn.Module):
#     def __init__(
#         self,
#         dinov2_path: str,
#         normalize: bool = True,
#         shallow_layer: int = 2,   # NEW: early block index for texture features
#     ):
#         super().__init__()
#         try:
#             self.encoder = Dinov2WithRegistersModel.from_pretrained(
#                 dinov2_path, local_files_only=True
#             )
#         except (OSError, ValueError, AttributeError):
#             self.encoder = Dinov2WithRegistersModel.from_pretrained(
#                 dinov2_path, local_files_only=False
#             )

#         self.encoder.requires_grad_(False)

#         if normalize:
#             self.encoder.layernorm.elementwise_affine = False
#             self.encoder.layernorm.weight = None
#             self.encoder.layernorm.bias = None

#         self.patch_size = self.encoder.config.patch_size
#         self.hidden_size = self.encoder.config.hidden_size
#         self.shallow_layer = int(shallow_layer)

#         # DINOv2 w/ registers: 1 CLS + 4 registers
#         self.unused_token_num = 5

#     def _strip_special_tokens(self, h: torch.Tensor) -> torch.Tensor:
#         # h: (B, 1+4+N, C) -> (B, N, C)
#         return h[:, self.unused_token_num:]

#     @torch.no_grad()
#     def dinov2_forward(
#         self,
#         x: torch.Tensor,
#         return_shallow: bool = False,
#     ):
#         out = self.encoder(x, output_hidden_states=True)

#         deep = self._strip_special_tokens(out.last_hidden_state)

#         if not return_shallow:
#             return deep

#         hs = out.hidden_states  # tuple length ~ num_layers+1
#         # safety clamp
#         idx = max(0, min(self.shallow_layer, len(hs) - 1))
#         shallow = self._strip_special_tokens(hs[idx])

#         return shallow, deep

#     def forward(self, x: torch.Tensor, return_shallow: bool = False):
#         return self.dinov2_forward(x, return_shallow=return_shallow)
