import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer 
from src.stage2.models.DDT import DiTwDDTHead 

class SigLIPTextEncoder(nn.Module):
    def __init__(self, model_name="google/siglip-so400m-patch14-384", trainable=False):
        super().__init__()
        print(f"Loading SigLIP Text Encoder: {model_name}...")
        self.trainable = trainable
        
        
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Hidden Size 자동 감지
        # SigLIP SO400M의 경우 text_config.hidden_size는 1152입니다.
        if hasattr(self.model.config, "text_config"):
             self.hidden_size = self.model.config.text_config.hidden_size
        else:
             # fallback
             self.hidden_size = self.model.config.hidden_size

        # 학습 여부 설정 (Frozen)
        if not trainable:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model.gradient_checkpointing_enable()
            
    def forward(self, text_list):
        # SigLIP 토크나이징
        inputs = self.tokenizer(
            text_list, 
            return_tensors="pt", 
            padding="max_length", # SigLIP은 고정 길이를 선호하는 편입니다
            truncation=True, 
            max_length=64 # 캡션 길이에 따라 조절 (보통 64~128)
        ).to(self.model.device)
        
        if self.trainable:
            outputs = self.model.text_model(**inputs)
        else:
            with torch.no_grad():
                # self.model.text_model을 호출하여 텍스트 인코딩만 수행
                outputs = self.model.text_model(**inputs)
        
       
        text_emb = outputs.pooler_output
        
        return text_emb.to(dtype=torch.float32)

class SimpleProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
     
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),  # 비선형성 하나 정도는 있는 게 학습에 도움됨
            nn.Linear(output_dim, output_dim) 
        )

    def forward(self, x):
        return self.net(x)

class RAE_T2I_Model(nn.Module):
    def __init__(
        self, 
        # SigLIP 모델명 예시
        siglip_name="google/siglip-so400m-patch14-384", 
        train_text_encoder=False, 
        # RAE 설정값
        input_size=32,          # 256x256 이미지 -> Patch 16이면 16x16=256 토큰, Patch 8이면 32x32=1024 토큰
                                # *주의*: DiT 내부 설정과 RAE Latent 사이즈를 맞춰야 함
        patch_size=[1, 1],      
        in_channels=768,        
        hidden_size=[1152, 2048], # [DiT Hidden, DDT Head Hidden]
        depth=[28, 2],
        num_heads=[16, 16],
        **dit_kwargs
    ):
        super().__init__()
        
        # 1. SigLIP 텍스트 인코더 (Frozen)
        self.text_encoder = SigLIPTextEncoder(siglip_name, trainable=train_text_encoder)
        
        # 2. Projector (Adapter 대체)
        # SigLIP output(1152) -> DiT Hidden(1152)
        # 차원이 같더라도 공간 매핑을 위해 Projector는 유지하는 게 좋습니다.
        self.projector = SimpleProjector(
            input_dim=self.text_encoder.hidden_size, 
            output_dim=hidden_size[0] 
        )
        
        # 3. DiT Backbone
        self.dit = DiTwDDTHead(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            num_classes=0, # Class label 대신 Text Embedding을 쓰므로 num_classes는 사용 안 함 (혹은 1)
            **dit_kwargs
        )

    def forward(self, x, t, text_list, s=None):
        # 1. Text -> SigLIP Embedding [B, 1152]
        text_emb = self.text_encoder(text_list) 
        
        # 2. Projection [B, 1152]
        cond = self.projector(text_emb) 
        
        # 3. DiT
        noise_pred = self.dit(x, t, text_embedding=cond, s=s)
        
        return noise_pred