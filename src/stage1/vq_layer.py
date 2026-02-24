import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.normal_()

        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", torch.zeros(num_embeddings, embedding_dim))
        self.ema_w.data.normal_()

    def forward(self, x):
        # x: (B,C,H,W)
        x = x.permute(0, 2, 3, 1).contiguous()   # (B,H,W,C)
        B, H, W, C = x.shape
        flat_x = x.view(-1, C)                   # (N,C)

        # ---------------------------------------------------------
        # (1) DINO-Tok 스타일: cosine VQ (normalize 후 거리)
        #     autocast(fp16)여도 여기서는 fp32로 하는 게 안정적
        # ---------------------------------------------------------
        flat_x = F.normalize(flat_x.float(), p=2, dim=1)  # (N,C) fp32
        embed = F.normalize(self.embedding.weight.float(), p=2, dim=1)  # (K,C) fp32

        # squared L2 on unit sphere == 2 - 2*cos
        distances = (
            flat_x.pow(2).sum(dim=1, keepdim=True)
            + embed.pow(2).sum(dim=1)
            - 2 * flat_x @ embed.t()
        )
        encoding_indices = torch.argmin(distances, dim=1)  # (N,)

        quantized = F.embedding(encoding_indices, embed).view(B, H, W, C)  # fp32

        # ---------------------------------------------------------
        # (2) EMA update (항상 fp32, autocast off)
        # ---------------------------------------------------------
        if self.training:
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                encodings_sum = torch.bincount(
                    encoding_indices, minlength=self.num_embeddings
                ).float()

                dw = torch.zeros_like(self.ema_w)  # fp32
                dw.index_add_(0, encoding_indices, flat_x)  # flat_x is fp32 already

                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(encodings_sum)
                    torch.distributed.all_reduce(dw)

                self.cluster_size.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)
                self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)

                n = self.cluster_size.sum()
                cluster_size = (self.cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n
                embed_normalized = self.ema_w / cluster_size.unsqueeze(1)

                # 코드북 업데이트 + normalize (fp32)
                self.embedding.weight.data.copy_(F.normalize(embed_normalized, p=2, dim=1))

        # ---------------------------------------------------------
        # (3) commitment loss (normalize 공간에서)
        # ---------------------------------------------------------
        x_norm = flat_x.view(B, H, W, C)  # normalized fp32
        loss = self.commitment_cost * F.mse_loss(quantized.detach(), x_norm)

        # ---------------------------------------------------------
        # (4) straight-through + dtype 원상복구
        #     디코더로 넘어가는 값은 원래 x dtype(fp16)으로 맞춰줌
        # ---------------------------------------------------------
        quantized_st = x_norm + (quantized - x_norm).detach()
        quantized_st = quantized_st.to(x.dtype)  # fp16으로 되돌림 (autocast 호환)
        quantized_st = quantized_st.permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)

        return quantized_st, loss.to(x.dtype), encoding_indices.view(B, H, W)
