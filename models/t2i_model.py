import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.stage2.models.DDT import DiTwDDTHead 

class TextEncoder(nn.Module):
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B", trainable=False):
        super().__init__()
        print(f"Loading LLM: {model_name}...")
        self.trainable = trainable # (추가) 변수 저장 필요
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
        )
        self.model.to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Hidden Size 자동 감지 (추가)
        self.hidden_size = self.model.config.hidden_size

        # 학습 여부 설정
        if not trainable:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            # 학습하더라도 메모리 절약을 위해 Gradient Checkpointing 추천
            self.model.gradient_checkpointing_enable()
            
    def forward(self, text_list):
        inputs = self.tokenizer(
            text_list, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(self.model.device)
        
        # trainable일 때만 gradient 계산
        if self.trainable:
            outputs = self.model(**inputs, output_hidden_states=True)
        else:
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
        last_hidden_state = outputs.hidden_states[-1]
        
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        masked_hidden = last_hidden_state * attention_mask
        text_emb = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)
        
        return text_emb.to(dtype=torch.float32)

class TextAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(), 
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim) # (선택) 마지막 LayerNorm은 상황에 따라 빼기도 함
        )

    def forward(self, x):
        return self.net(x)

class RAE_T2I_Model(nn.Module):
    def __init__(
        self, 
        llm_name="Qwen/Qwen2.5-1.5B",
        train_llm=False, # 처음엔 False 추천
        # --- RAE 설정값으로 수정 ---
        input_size=16,          # 16x16 Latent
        patch_size=[1, 1],      # RAE는 보통 patch 1
        in_channels=768,        # DINO/SigLIP 차원 (설정에 맞게 확인!)
        hidden_size=[1152, 2048], 
        depth=[28, 2],
        num_heads=[16, 16],
        # ------------------------------
        **dit_kwargs
    ):
        super().__init__()
        
        # 1. 텍스트 인코더
        self.text_encoder = TextEncoder(llm_name, trainable=train_llm)
        
        # 2. Adapter
        # DiT Encoder의 hidden size (1152)로 맞춰줌
        self.adapter = TextAdapter(
            input_dim=self.text_encoder.hidden_size, 
            output_dim=hidden_size[0] 
        )
        
        # 3. DiT
        self.dit = DiTwDDTHead(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            num_classes=1000, # 이건 무시됨
            **dit_kwargs
        )

    def forward(self, x, t, text_list, s=None):
        # 1. Text -> Embedding
        text_emb = self.text_encoder(text_list) # [B, 1536]
        
        # 2. Adaptation
        cond = self.adapter(text_emb) # [B, 1152]
        
        # 3. DiT (수정된 forward 호출)
        noise_pred = self.dit(x, t, text_embedding=cond, s=s)
        
        return noise_pred