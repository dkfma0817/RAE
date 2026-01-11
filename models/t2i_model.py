import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from src.stage2.models.DDT import DiTwDDTHead

class SigLIPTextEncoder(nn.Module):
    def __init__(self, model_name="google/siglip2-base-patch16-256", trainable=False, max_length=64):
        super().__init__()
        self.trainable = trainable
        self.max_length = max_length

        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if hasattr(self.model.config, "text_config"):
            self.hidden_size = self.model.config.text_config.hidden_size
        else:
            self.hidden_size = self.model.config.hidden_size

        if not trainable:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
        else:
            # optional
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()

    def forward(self, text_list, device=None):
        if device is None:
            device = next(self.model.parameters()).device

        inputs = self.tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,          # max_length padding은 낭비 큼. 일단 True 권장.
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if self.trainable:
            out = self.model.text_model(**inputs) if hasattr(self.model, "text_model") else self.model(**inputs)
        else:
            with torch.no_grad():
                out = self.model.text_model(**inputs) if hasattr(self.model, "text_model") else self.model(**inputs)

        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            text_emb = out.pooler_output
        else:
            text_emb = out.last_hidden_state.mean(dim=1)

        return text_emb.float()

class SimpleProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),   # ✅ 추가
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )
    def forward(self, x):
        return self.net(x)


class RAE_T2I_Model(nn.Module):
    def __init__(
        self,
        siglip_name="google/siglip2-base-patch16-256",
        train_text_encoder=False,
        input_size=16,
        patch_size=[1, 1],
        in_channels=768,
        hidden_size=[1152, 2048],
        depth=[28, 2],
        num_heads=[16, 16],
        **dit_kwargs
    ):
        super().__init__()
        self.text_encoder = SigLIPTextEncoder(siglip_name, trainable=train_text_encoder)
        self.projector = SimpleProjector(self.text_encoder.hidden_size, hidden_size[0])

        # num_classes는 안전하게 1로 (안 쓰더라도)
        self.dit = DiTwDDTHead(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            num_classes=1,
            **dit_kwargs
        )

    def forward(self, x, t, text_list, s=None):
        device = x.device

        # 1) text encode
        text_emb = self.text_encoder(text_list, device=device)   # (B, text_dim)

        # 2) projector
        cond = self.projector(text_emb)                          # (B, hidden)

        # ✅ [필수] L2 normalize (AdaLN 폭주 방지)
        cond = cond / (cond.norm(dim=-1, keepdim=True) + 1e-6)

        # ✅ [권장] condition strength를 약하게 시작 (너무 강하면 NaN 잘 남)
        cond = 0.5 * cond   # 0.2~1.0 사이 조절 가능. 처음엔 0.5 추천.

        return self.dit(x, t, text_embedding=cond, s=s)
