import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoders import GeneralDecoder
from .encoders import ARCHS
from transformers import AutoConfig, AutoImageProcessor
from typing import Optional, Protocol
from math import sqrt

# expects your vq_layer.py to define VectorQuantizer (can be EMA version)
from .vq_layer import VectorQuantizer


class Stage1Protocal(Protocol):
    patch_size: int
    hidden_size: int

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        ...


class PCAReweightLayer(nn.Module):
    """
    DINO-Tok style PCA-based reweighting (NOT whitening).
    Applies:
        x_pca = (x - mean) @ V^T
        x_rw  = x_pca * (var + eps)^alpha
    """
    def __init__(self, stats_path: str, alpha: float = 0.25, eps: float = 1e-6):
        super().__init__()
        stats = torch.load(stats_path, map_location="cpu")

        if "mean" not in stats or "comp" not in stats or "var" not in stats:
            raise KeyError(f"PCA stats at {stats_path} must contain keys: mean, comp, var")

        self.register_buffer("mean", stats["mean"].float())          # (D,)
        self.register_buffer("comp_t", stats["comp"].t().float())    # (D,D) = V^T
        scale = (stats["var"].float() + eps).pow(alpha)              # (D,)
        self.register_buffer("scale", scale)

        self.alpha = float(alpha)
        self.eps = float(eps)

        print(f"[PCAReweightLayer] Loaded stats from {stats_path}")
        print(f" - Dim: {self.mean.shape[0]}, alpha={self.alpha}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D) or (B, D, H, W)
        returns: same shape as x
        """
        if x.dim() == 4:
            # (B,D,H,W) -> (B,H,W,D)
            x_perm = x.permute(0, 2, 3, 1)
            # center
            x_perm = x_perm - self.mean
            # rotate to PCA basis
            x_perm = x_perm @ self.comp_t
            # reweight
            x_perm = x_perm * self.scale
            # back
            return x_perm.permute(0, 3, 1, 2)

        if x.dim() == 3:
            # (B,N,D)
            x = x - self.mean
            x = x @ self.comp_t
            x = x * self.scale
            return x

        raise ValueError(f"PCAReweightLayer expected 3D or 4D input, got {x.dim()}D")


class RAE(nn.Module):
    def __init__(
        self,
        # ---- encoder configs ----
        encoder_cls: str = "Dinov2withNorm",
        encoder_config_path: str = "facebook/dinov2-base",
        encoder_input_size: int = 224,
        encoder_params: dict = {},
        # ---- decoder configs ----
        decoder_config_path: str = "vit_mae-base",
        decoder_patch_size: int = 16,
        pretrained_decoder_path: Optional[str] = None,
        # ---- noising, reshaping and normalization ----
        noise_tau: float = 0.8,
        reshape_to_2d: bool = True,
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
        # ---- VQ configs ----
        use_vq: bool = False,
        vq_codebook_size: int = 1024,
        vq_embed_dim: int = 256,            # project latent_dim -> vq_embed_dim before VQ
        vq_commitment_cost: float = 0.25,
        vq_decay: float = 0.99,
        vq_eps: float = 1e-5,
        vq_z_norm: bool = True,
        # ---- PCA reweighting configs (NEW) ----
        pca_stat_path: Optional[str] = None,  # path to pca_stats.pth produced by get_pca.py
        pca_alpha: float = 0.25,
        pca_eps: float = 1e-6,
    ):
        super().__init__()

        encoder_cls = ARCHS[encoder_cls]
        self.encoder: Stage1Protocal = encoder_cls(**encoder_params)

        self.use_vq = use_vq
        self.vq_z_norm = vq_z_norm
        self.eps = eps

        # encoder stats as buffers (move with model)
        proc = AutoImageProcessor.from_pretrained(encoder_config_path)
        encoder_mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
        encoder_std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
        self.register_buffer("encoder_mean", encoder_mean, persistent=False)
        self.register_buffer("encoder_std", encoder_std, persistent=False)

        print(f"encoder_config_path: {encoder_config_path}")
        _ = AutoConfig.from_pretrained(encoder_config_path)

        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size = self.encoder.patch_size
        self.latent_dim = self.encoder.hidden_size

        assert (
            self.encoder_input_size % self.encoder_patch_size == 0
        ), f"encoder_input_size {self.encoder_input_size} must be divisible by encoder_patch_size {self.encoder_patch_size}"

        self.base_patches = (self.encoder_input_size // self.encoder_patch_size) ** 2

        # decoder
        decoder_config = AutoConfig.from_pretrained(decoder_config_path)
        decoder_config.hidden_size = self.latent_dim
        decoder_config.patch_size = decoder_patch_size
        decoder_config.image_size = int(decoder_patch_size * sqrt(self.base_patches))
        self.decoder = GeneralDecoder(decoder_config, num_patches=self.base_patches)

        if pretrained_decoder_path is not None:
            print(f"Loading pretrained decoder from {pretrained_decoder_path}")
            state_dict = torch.load(pretrained_decoder_path, map_location="cpu")
            keys = self.decoder.load_state_dict(state_dict, strict=False)
            if len(keys.missing_keys) > 0:
                print(f"Missing keys when loading pretrained decoder: {keys.missing_keys}")

        self.noise_tau = noise_tau
        self.reshape_to_2d = reshape_to_2d

        # latent normalization stats (optional)
        if normalization_stat_path is not None:
            stats = torch.load(normalization_stat_path, map_location="cpu")
            latent_mean = stats.get("mean", None)
            latent_var = stats.get("var", None)

            self.do_normalization = True
            if latent_mean is not None:
                self.register_buffer("latent_mean", latent_mean, persistent=False)
            else:
                self.latent_mean = None
            if latent_var is not None:
                self.register_buffer("latent_var", latent_var, persistent=False)
            else:
                self.latent_var = None
            print(f"Loaded normalization stats from {normalization_stat_path}")
        else:
            self.do_normalization = False
            self.latent_mean = None
            self.latent_var = None

        # -------------------------
        # PCA reweighting (NEW)
        # -------------------------
        # Only meaningful when used with VQ (but can be enabled regardless)
        if pca_stat_path is not None:
            self.pca_reweight = PCAReweightLayer(
                pca_stat_path,
                alpha=pca_alpha,
                eps=pca_eps,
            )
        else:
            self.pca_reweight = None

        # -------------------------
        # VQ modules
        # -------------------------
        if self.use_vq:
            print(
                f"❄️ VQ Mode Enabled: Freezing Encoder, Codebook Size: {vq_codebook_size}, "
                f"VQ embed dim: {vq_embed_dim}, z_norm: {vq_z_norm}"
            )
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False

            # project latent to smaller dim before VQ (reduces collapse)
            self.vq_pre = nn.Conv2d(self.latent_dim, vq_embed_dim, kernel_size=1)
            self.vq_post = nn.Conv2d(vq_embed_dim, self.latent_dim, kernel_size=1)

            # instantiate VQ layer; support both (vanilla) and (EMA) signatures
            try:
                self.vq_layer = VectorQuantizer(
                    num_embeddings=vq_codebook_size,
                    embedding_dim=vq_embed_dim,
                    commitment_cost=vq_commitment_cost,
                    decay=vq_decay,
                    eps=vq_eps,
                )
            except TypeError:
                self.vq_layer = VectorQuantizer(
                    num_embeddings=vq_codebook_size,
                    embedding_dim=vq_embed_dim,
                    commitment_cost=vq_commitment_cost,
                )

            self.last_vq_indices = None
        else:
            self.vq_pre = None
            self.vq_post = None
            self.vq_layer = None
            self.last_vq_indices = None

    def noising(self, x: torch.Tensor) -> torch.Tensor:
        noise_sigma = self.noise_tau * torch.rand(
            (x.size(0),) + (1,) * (len(x.shape) - 1), device=x.device
        )
        noise = noise_sigma * torch.randn_like(x)
        return x + noise

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # resize to encoder_input_size
        _, _, h, w = x.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            x = F.interpolate(
                x,
                size=(self.encoder_input_size, self.encoder_input_size),
                mode="bicubic",
                align_corners=False,
            )

        # normalize input for encoder
        x = (x - self.encoder_mean.to(x.device)) / self.encoder_std.to(x.device)

        z = self.encoder(x)

        if self.training and self.noise_tau > 0:
            z = self.noising(z)

        if self.reshape_to_2d:
            b, n, c = z.shape
            hh = ww = int(sqrt(n))
            z = z.transpose(1, 2).contiguous().view(b, c, hh, ww)

        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = z * torch.sqrt(latent_var + self.eps) + latent_mean

        if self.reshape_to_2d:
            b, c, h, w = z.shape
            n = h * w
            z = z.view(b, c, n).transpose(1, 2).contiguous()

        output = self.decoder(z, drop_cls_token=False).logits
        x_rec = self.decoder.unpatchify(output)

        # unnormalize back to pixel space
        x_rec = x_rec * self.encoder_std.to(x_rec.device) + self.encoder_mean.to(x_rec.device)
        return x_rec

    def _vq_process(self, z: torch.Tensor):
        """
        z: (B, C, H, W) where C = latent_dim
        returns: z_out (B, C, H, W), vq_loss (scalar tensor)
        """
        # optional PCA reweighting (DINO-Tok style) BEFORE projection + VQ
        if self.pca_reweight is not None:
            z = self.pca_reweight(z)

        # project to smaller dim
        z_small = self.vq_pre(z)

        # optional scale normalization to reduce collapse
        if self.vq_z_norm:
            denom = z_small.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
            z_small = z_small / denom

        # VQ
        z_q, vq_loss, indices = self.vq_layer(z_small)
        self.last_vq_indices = indices

        # project back
        z_out = self.vq_post(z_q)
        return z_out, vq_loss

    def forward(self, x: torch.Tensor):
        # encode (freeze encoder under VQ mode)
        if self.use_vq:
            with torch.no_grad():
                z = self.encode(x)
        else:
            z = self.encode(x)

        vq_loss = torch.zeros((), device=x.device)

        # VQ bottleneck
        if self.use_vq:
            z, vq_loss = self._vq_process(z)

        # decode
        x_rec = self.decode(z)

        # keep your original convention (train: tuple, eval: recon only)
        if self.training:
            return x_rec, vq_loss
        else:
            return x_rec


# import torch
# import torch.nn as nn
# from .decoders import GeneralDecoder
# from .encoders import ARCHS
# from transformers import AutoConfig, AutoImageProcessor
# from typing import Optional
# from math import sqrt
# from typing import Protocol

# class Stage1Protocal(Protocol):
#     # must have patch size attribute
#     patch_size: int
#     hidden_size: int 
#     def encode(self, x: torch.Tensor) -> torch.Tensor:
#         ...

# class RAE(nn.Module):
#     def __init__(self, 
#         # ---- encoder configs ----
#         encoder_cls: str = 'Dinov2withNorm',
#         encoder_config_path: str = 'facebook/dinov2-base',
#         encoder_input_size: int = 224,
#         encoder_params: dict = {},
#         # ---- decoder configs ----
#         decoder_config_path: str = 'vit_mae-base',
#         decoder_patch_size: int = 16,
#         pretrained_decoder_path: Optional[str] = None,
#         # ---- noising, reshaping and normalization-----
#         noise_tau: float = 0.8,
#         reshape_to_2d: bool = True,
#         normalization_stat_path: Optional[str] = None,
#         eps: float = 1e-5,
#     ):
#         super().__init__()
#         encoder_cls = ARCHS[encoder_cls]
#         self.encoder: Stage1Protocal = encoder_cls(**encoder_params)
#         print(f"encoder_config_path: {encoder_config_path}")
#         proc = AutoImageProcessor.from_pretrained(encoder_config_path)
#         self.encoder_mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
#         self.encoder_std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
#         encoder_config = AutoConfig.from_pretrained(encoder_config_path)
#         # see if the encoder has patch size attribute            
#         self.encoder_input_size = encoder_input_size
#         self.encoder_patch_size = self.encoder.patch_size
#         self.latent_dim = self.encoder.hidden_size
#         assert self.encoder_input_size % self.encoder_patch_size == 0, f"encoder_input_size {self.encoder_input_size} must be divisible by encoder_patch_size {self.encoder_patch_size}"
#         self.base_patches = (self.encoder_input_size // self.encoder_patch_size) ** 2 # number of patches of the latent
        
#         # decoder
#         decoder_config = AutoConfig.from_pretrained(decoder_config_path)
#         decoder_config.hidden_size = self.latent_dim # set the hidden size of the decoder to be the same as the encoder's output
#         decoder_config.patch_size = decoder_patch_size
#         decoder_config.image_size = int(decoder_patch_size * sqrt(self.base_patches)) 
#         self.decoder = GeneralDecoder(decoder_config, num_patches=self.base_patches)
#         # load pretrained decoder weights
#         if pretrained_decoder_path is not None:
#             print(f"Loading pretrained decoder from {pretrained_decoder_path}")
#             state_dict = torch.load(pretrained_decoder_path, map_location='cpu')
#             keys = self.decoder.load_state_dict(state_dict, strict=False)
#             if len(keys.missing_keys) > 0:
#                 print(f"Missing keys when loading pretrained decoder: {keys.missing_keys}")
#         self.noise_tau = noise_tau
#         self.reshape_to_2d = reshape_to_2d
#         print("Zeroing decoder CLS token")
#         self.decoder.trainable_cls_token.data.zero_()

#         if normalization_stat_path is not None:
#             stats = torch.load(normalization_stat_path, map_location='cpu')
#             self.latent_mean = stats.get('mean', None)
#             self.latent_var = stats.get('var', None)
#             self.do_normalization = True
#             self.eps = eps
#             print(f"Loaded normalization stats from {normalization_stat_path}")
#         else:
#             self.do_normalization = False
#     def noising(self, x: torch.Tensor) -> torch.Tensor:
#         noise_sigma = self.noise_tau * torch.rand((x.size(0),) + (1,) * (len(x.shape) - 1), device=x.device)
#         noise = noise_sigma * torch.randn_like(x)
#         return x + noise
#     @torch.no_grad()
#     def encode(self, x: torch.Tensor) -> torch.Tensor:
#         # normalize input
#         _, _, h, w = x.shape
#         if h != self.encoder_input_size or w != self.encoder_input_size:
#             x = nn.functional.interpolate(x, size=(self.encoder_input_size, self.encoder_input_size), mode='bicubic', align_corners=False)
#         x = (x - self.encoder_mean.to(x.device)) / self.encoder_std.to(x.device)
#         z = self.encoder(x)
#         if self.training and self.noise_tau > 0:
#             z = self.noising(z)
#         if self.reshape_to_2d:
#             b, n, c = z.shape
#             h = w = int(sqrt(n))
#             z = z.transpose(1, 2).view(b, c, h, w)
#         if self.do_normalization:
#             latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
#             latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
#             z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)
#         return z
    
#     def decode(self, z: torch.Tensor) -> torch.Tensor:
#         if self.do_normalization:
#             latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
#             latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
#             z = z * torch.sqrt(latent_var + self.eps) + latent_mean
#         if self.reshape_to_2d:
#             b, c, h, w = z.shape
#             n = h * w
#             z = z.view(b, c, n).transpose(1, 2)
#         output = self.decoder(z, drop_cls_token=False).logits
#         x_rec = self.decoder.unpatchify(output)
#         x_rec = x_rec * self.encoder_std.to(x_rec.device) + self.encoder_mean.to(x_rec.device)
#         return x_rec
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         z = self.encode(x)
#         x_rec = self.decode(z)
#         return x_rec