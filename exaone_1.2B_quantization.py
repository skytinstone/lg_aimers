"""
LG AImers 8기 해커톤 - EXAONE 4.0 Quantization
Weight: INT4, Activation: FP16
작성일 : 2026.01.13
작성자 : 신민석

[학습 환경] 
OS : window 11
GPU : 1080ti 12VRAM

[양자화 전략]
1. 양자화 파이프라인 구성(최초)
- Pruning, Knowledge Distilation, Quantization

2. 


[학습 방향]
1. 32B 모델을 양자화 진행하려고하나, 안전한 환경에서 파이프라인 구성을 위해 1.2B 모델 학습을 진행!
2. 이후, 32B 모델로 바꿔끼워 한번 더 진행!

"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dotenv import load_dotenv
import gc

# 환경 변수 로드
load_dotenv()

class ExaoneQuantizer:
    def __init__(self, model_name="LGAI-EXAONE/EXAONE-4.0-1.2B"):
        """
        EXAONE 모델 양자화 클래스
        
        Args:
            model_name: Hugging Face 모델 이름
        """
        self.model_name = model_name
        self.hf_token = os.getenv("HF_TOKEN")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if not self.hf_token:
            print("Warning: HF_TOKEN이 설정되지 않았습니다. .env 파일을 확인하세요.")
        
        print(f"Device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    def quantize_int4_fp16(self, output_dir="./quantized_models"):
        """
        INT4 Weight + FP16 Activation 양자화 수행
        
        Args:
            output_dir: 양자화된 모델 저장 경로
        """
        print(f"\n{'='*60}")
        print(f"EXAONE 모델 양자화 시작: {self.model_name}")
        print(f"양자화 설정: INT4 Weight + FP16 Activation")
        print(f"{'='*60}\n")
        
        # BitsAndBytes를 사용한 4-bit 양자화 설정
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,                      # 4-bit 양자화 활성화
            bnb_4bit_compute_dtype=torch.float16,   # Activation: FP16
            bnb_4bit_use_double_quant=True,         # Double quantization (더 나은 압축)
            bnb_4bit_quant_type="nf4"               # NormalFloat 4-bit (추천)
        )
        
        try:
            print("1. 토크나이저 로드 중...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token,
                trust_remote_code=True
            )
            print("   ✓ 토크나이저 로드 완료")
            
            print("\n2. 모델 로드 및 양자화 중...")
            print("   (이 과정은 모델 크기에 따라 수 분 소요될 수 있습니다)")
            
            # 첫 번째 시도: safetensors 사용
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    token=self.hf_token,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    revision="main"
                )
                print("   ✓ 모델 로드 및 양자화 완료 (safetensors)")
                
            except Exception as e:
                print(f"   ⚠️  safetensors 로드 실패: {str(e)[:100]}")
                print("   재시도: pytorch 형식으로 로드...")
                
                # 두 번째 시도: pytorch 형식 사용
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    token=self.hf_token,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,
                    use_safetensors=False
                )
                print("   ✓ 모델 로드 및 양자화 완료 (pytorch)")
            
            # 메모리 사용량 확인
            if self.device == "cuda":
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                print(f"\n3. GPU 메모리 사용량:")
                print(f"   - Allocated: {allocated:.2f} GB")
                print(f"   - Reserved: {reserved:.2f} GB")
            
            # 간단한 추론 테스트
            print("\n4. 양자화된 모델 테스트 중...")
            test_prompt = "안녕하세요, EXAONE입니다."
            inputs = tokenizer(test_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.5,
                    do_sample=True
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"   입력: {test_prompt}")
            print(f"   출력: {generated_text}")
            print("   ✓ 추론 테스트 완료")
            
            # 모델 저장
            print(f"\n5. 양자화된 모델 저장 중: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"   ✓ 저장 완료: {output_dir}")
            
            # 모델 정보 출력
            self._print_model_info(model)
            
            return model, tokenizer
            
        except Exception as e:
            print(f"\n❌ 오류 발생: {str(e)}")
            import traceback
            print("\n상세 오류 정보:")
            traceback.print_exc()
            raise
            
        finally:
            # 메모리 정리
            if 'model' in locals():
                del model
            if 'tokenizer' in locals():
                del tokenizer
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
    
    def _print_model_info(self, model):
        """모델 정보 출력"""
        print(f"\n{'='*60}")
        print("양자화된 모델 정보")
        print(f"{'='*60}")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"총 파라미터 수: {total_params:,}")
        print(f"학습 가능 파라미터 수: {trainable_params:,}")
        
        # 레이어별 dtype 확인
        print("\n레이어 dtype 정보:")
        dtype_counts = {}
        for name, param in model.named_parameters():
            dtype = str(param.dtype)
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        
        for dtype, count in dtype_counts.items():
            print(f"  - {dtype}: {count} layers")
        
        print(f"{'='*60}\n")


def main():
    """메인 실행 함수"""
    print("LG AImers 8기 해커톤 - EXAONE Quantization")
    print("="*60)
    
    # 1.2B 모델로 먼저 테스트
    quantizer_1_2b = ExaoneQuantizer("LGAI-EXAONE/EXAONE-4.0-1.2B")
    
    try:
        model, tokenizer = quantizer_1_2b.quantize_int4_fp16(
            output_dir="./quantized_models/exaone-1.2b-int4"
        )
        print("\n✓ 1.2B 모델 양자화 완료!")
        
    except Exception as e:
        print(f"\n❌ 양자화 실패: {str(e)}")
        print("\n다음 사항을 확인하세요:")
        print("1. .env 파일에 HF_TOKEN이 올바르게 설정되어 있는지")
        print("2. 필요한 라이브러리가 모두 설치되어 있는지")
        print("3. GPU 메모리가 충분한지")
        print("4. Hugging Face 모델 페이지에서 라이선스에 동의했는지")
        print("   → https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B")
        return
    
    print("\n" + "="*60)
    print("다음 단계:")
    print("1. 양자화된 1.2B 모델 성능 평가")
    print("2. 32B 모델로 확장 (Layer-by-layer 방식)")
    print("="*60)


if __name__ == "__main__":
    main()