"""
EXAONE 4.0 32B 모델 INT4 양자화
Meta 텐서 오류 완전 해결 버전 (device_map=None 방식)
수정일: 2026.01.13
버전: v3 - device_map 없이 로드 후 수동 배치
"""

import os
import gc
import time
import shutil

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# .env 파일 로드
load_dotenv()

class Exaone32BQuantizer:
    def __init__(self):
        self.model_name = "LGAI-EXAONE/EXAONE-4.0-32B"
        self.hf_token = os.getenv("HF_TOKEN")

        if not self.hf_token:
            raise ValueError("HF_TOKEN이 .env 파일에 없습니다. .env 파일을 확인해주세요.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")

        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Total VRAM: {total_vram:.2f} GB")

    def quantize_int4_fp16(self, output_dir="./quantized_models/exaone-32b-int4"):
        start_time = time.time()

        print(f"\n{'='*70}")
        print("EXAONE 32B INT4 양자화 (Meta Tensor 완전 해결판 v3)")
        print("설정: INT4 Weight (NF4) + FP16 Activation")
        print("방식: device_map=None으로 로드 후 수동 배치")
        print(f"{'='*70}\n")

        # 4bit 양자화 설정
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True,
        )

        offload_dir = "./offload_temp"
        os.makedirs(offload_dir, exist_ok=True)

        model = None
        tokenizer = None

        try:
            print("1. 토크나이저 로드 중...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token,
                trust_remote_code=True,
            )
            
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            print("   ✓ 토크나이저 로드 완료\n")

            print("2. 32B 모델 로드 + INT4 양자화 중...")
            print("   예상 시간: 10-30분 (시스템 사양에 따라 다름)")
            print("   ⚠️ 주의: device_map=None 방식으로 메타 텐서 생성을 방지합니다.")
            print("   시스템 RAM을 많이 사용할 수 있습니다.\n")

            # [핵심 수정] device_map 없이 먼저 로드한 후 수동으로 디바이스 배치
            # 이 방법으로 메타 텐서 생성을 완전히 방지합니다.
            print("   [진행 중] 모델 로드 중 (device_map 없음)...\n")

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=self.hf_token,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map=None,  # <--- device_map을 None으로 설정
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )

            print("   [진행 중] 모델을 디바이스로 이동 중...\n")
            # 양자화된 모델을 GPU로 이동 (VRAM이 부족하면 자동으로 CPU로 폴백)
            try:
                model = model.to(self.device)
            except RuntimeError as e:
                print(f"   ⚠️ GPU 메모리 부족: {str(e)}")
                print("   CPU 모드로 전환합니다...\n")
                model = model.to("cpu")
            print("   ✓ 모델 로드 및 INT4 양자화 완료!\n")

            elapsed_min = (time.time() - start_time) / 60
            print(f"   로드 경과 시간: {elapsed_min:.1f}분\n")

            # 모델이 어떤 디바이스에 있는지 확인
            first_param = next(model.parameters())
            model_device = first_param.device
            print(f"3. 모델 디바이스: {model_device}\n")

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                reserved = torch.cuda.memory_reserved(0) / (1024**3)
                print("4. GPU 메모리 사용량:")
                print(f"   - Allocated: {allocated:.2f} GB")
                print(f"   - Reserved: {reserved:.2f} GB\n")

            # 추론 테스트
            print("5. 추론 테스트...")
            test_prompt = "인공지능의 미래에 대해 한 문장으로 설명해줘."
            inputs = tokenizer(test_prompt, return_tensors="pt")
            
            # 모델의 첫 번째 파라미터가 있는 디바이스로 입력 이동
            first_device = next(model.parameters()).device
            inputs = {k: v.to(first_device) for k, v in inputs.items()}

            infer_start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.5,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            infer_time = time.time() - infer_start

            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"   입력: {test_prompt}")
            print(f"   출력: {generated}")
            print(f"   추론 시간: {infer_time:.2f}초")
            print("   ✓ 추론 테스트 완료\n")

            # 모델 저장
            print(f"6. 모델 저장 중...")
            print(f"   경로: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

            save_start = time.time()
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            save_min = (time.time() - save_start) / 60
            
            print(f"   ✓ 저장 완료 (소요: {save_min:.1f}분)\n")

            self._print_model_info(model)

            total_min = (time.time() - start_time) / 60
            print(f"총 소요 시간: {total_min:.1f}분")

            return model, tokenizer

        except Exception as e:
            print(f"\n❌ 오류 발생: {str(e)}\n")
            import traceback
            traceback.print_exc()
            raise

        finally:
            # 메모리 정리
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if os.path.exists(offload_dir):
                try:
                    shutil.rmtree(offload_dir)
                    print("\n✓ 임시 폴더 정리 완료")
                except:
                    pass

    def _print_model_info(self, model):
        print(f"{'='*70}")
        print("양자화된 32B 모델 정보")
        print(f"{'='*70}")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"총 파라미터: {total_params:,}")

        dtype_counts = {}
        for _, param in model.named_parameters():
            dtype = str(param.dtype)
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1

        print("\n레이어 dtype:")
        for dtype, count in dtype_counts.items():
            print(f"  - {dtype}: {count} layers")

        model_size_gb = total_params * 0.5 / (1024**3)
        print(f"\n추정 크기 (INT4): {model_size_gb:.2f} GB")
        print(f"{'='*70}\n")


def main():
    print("=" * 70)
    print("LG AImers 8기 - EXAONE 32B Quantization Fix")
    print("=" * 70)

    try:
        quantizer = Exaone32BQuantizer()
        quantizer.quantize_int4_fp16()
        
        print("\n🎉 32B 모델 양자화 완료!")

    except Exception as e:
        print(f"\n❌ 실패: {str(e)}")


if __name__ == "__main__":
    main()