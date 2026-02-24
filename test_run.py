import torch

from models.t2i_model import RAE_T2I_Model 

def run_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on device: {device}")

    # 1. 모델 초기화 (체크포인트 없이 껍데기만 생성)
    print("\n[1] 모델 초기화 중...")
    try:
        model = RAE_T2I_Model(
            llm_name="Qwen/Qwen2.5-1.5B",
            train_llm=False,
            # RAE 설정 (SigLIP/DINO 기준)
            input_size=16,          # Latent 크기 (16x16)
            in_channels=768,        # Latent 채널
            hidden_size=[1152, 2048], # DiT-XL 설정
            depth=[28, 2],
            num_heads=[16, 16]
        ).to(device)
        print("✅ 모델 빌드 성공!")
    except Exception as e:
        print(f"❌ 모델 빌드 실패: {e}")
        return

    # 2. 더미 데이터 생성 (가짜 입력)
    print("\n[2] 더미 데이터 생성 중...")
    batch_size = 2
    
    # RAE Latent 모양: [Batch, Channel, Height, Width]
    # SigLIP/DINO의 경우 보통 256토큰 -> 16x16 그리드
    dummy_latent = torch.randn(batch_size, 768, 16, 16).to(device)
    
    # 타임스텝 (0~1 사이의 실수 or 정수, DiT 구현에 따라 다름)
    dummy_t = torch.rand(batch_size).to(device) 
    
    # 텍스트 프롬프트
    dummy_texts = ["A photo of a cute dog", "A futuristic city"]
    
    print(f"   - Latent Shape: {dummy_latent.shape}")
    print(f"   - Texts: {dummy_texts}")

    # 3. Forward Pass (실행)
    print("\n[3] Forward Pass 실행 (Qwen -> Adapter -> DiT)...")
    try:
        # 학습 모드와 동일하게 실행
        output = model(dummy_latent, dummy_t, dummy_texts)
        
        print("✅ Forward Pass 성공!")
        print(f"   - Output Shape: {output.shape}")
        
        # 4. 차원 검증 (입력과 출력이 같아야 함)
        if output.shape == dummy_latent.shape:
            print("검증 완료: 입력과 출력의 크기가 정확히 일치합니다.")
            print("   (이제 학습(train.py)으로 넘어가셔도 됩니다!)")
        else:
            print(f"⚠️ 경고: 입력({dummy_latent.shape})과 출력({output.shape}) 크기가 다릅니다.")
            
    except Exception as e:
        print(f"❌ 실행 중 에러 발생:\n{e}")
       

if __name__ == "__main__":
    run_test()