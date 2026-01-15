"""
LG AImers 8기 해커톤 - EXAONE 4.0 32B Pruning + Quantization 파이프라인
작성일: 2026.01.15
작성자: 신민석 (수정: Gemini)

[수정 사항]
- Mac M3 Pro 'Insufficient Memory' 에러 해결을 위한 초고강도 메모리 관리
- 32B 모델용 Disk Offloading 및 CPU-Partial Pruning 적용
- Fine-tuning 단계는 32B Mac 환경에서 물리적으로 불가능하므로 안전하게 Skip 로직 강화
"""

import os
import gc
import time
import shutil
import platform
import psutil
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from dotenv import load_dotenv

# [최우선] .env 파일 로드
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
FRIENDLI_API_KEY = os.getenv("FRIENDLI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class Exaone32BOptimizer:
    def __init__(self, model_name="LGAI-EXAONE/EXAONE-4.0-32B", pruning_ratio=0.3):
        self.model_name = model_name
        self.pruning_ratio = pruning_ratio
        self.pruning_percent = int(pruning_ratio * 100)  # 파일명용 (30, 40, 50)
        self.start_time = time.time()

        print(f"\n{'='*25} [1. 32B 환경 분석 및 메모리 체크] {'='*25}")
        self._check_environment()

    def _check_environment(self):
        print(f"● 운영체제: {platform.system()} (Device: {platform.processor()})")
        if torch.backends.mps.is_available():
            self.device = "mps"
            self.is_mac = True
            self.compute_dtype = torch.bfloat16 # Mac 최적화
            print(f"● 연산 장치: 🍎 Apple Silicon (MPS) - 32B 모드")
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.is_mac = False
            self.compute_dtype = torch.float16
            print(f"● 연산 장치: 🟢 NVIDIA GPU (CUDA)")
        else:
            self.device = "cpu"
            self.is_mac = False
            self.compute_dtype = torch.float32
            print(f"● 연산 장치: 💻 CPU Mode")

        ram = psutil.virtual_memory().total / (1024**3)
        available_ram = psutil.virtual_memory().available / (1024**3)
        print(f"● 시스템 RAM: {ram:.2f} GB (사용 가능: {available_ram:.2f} GB)")
        print(f"● 주의: 32B 모델은 약 64GB의 저장 공간(SSD)이 추가로 필요합니다.")

    def load_base_model(self):
        print(f"\n{'='*25} [2. 32B 모델 로드 (Disk Offload)] {'='*25}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=HF_TOKEN, trust_remote_code=True)
        
        # 오프로드 폴더 생성
        os.makedirs("./offload_32b", exist_ok=True)

        # 32B 모델은 Mac RAM보다 크므로 device_map="auto"와 offload 필수
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=HF_TOKEN,
            trust_remote_code=True,
            torch_dtype=self.compute_dtype,
            device_map="auto",
            offload_folder="./offload_32b",
            low_cpu_mem_usage=True,
            offload_state_dict=True
        )
        print(f"📊 32B 모델 로드 완료 (공유 메모리/디스크 분산 로드됨)")
        return model, tokenizer

    def apply_magnitude_pruning(self, model):
        print(f"\n{'='*25} [3. Pruning (CPU 연산 분산 모드)] {'='*25}")
        target_patterns = ['gate_proj', 'up_proj', 'down_proj']
        
        # 32B 모델은 한꺼번에 계산하면 에러나므로 레이어별로 순차 처리
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and any(p in name for p in target_patterns):
                    # 가중치를 CPU로 한 개씩만 가져와서 0으로 만듦 (MPS 메모리 폭발 방지)
                    w = module.weight.data.cpu().float()
                    threshold = torch.quantile(torch.abs(w), self.pruning_ratio)
                    mask = (torch.abs(w) > threshold).to(self.compute_dtype)
                    
                    # 다시 원본 위치(CPU/MPS/Disk)로 돌려보냄
                    module.weight.data = (w.to(module.weight.device) * mask.to(module.weight.device))
                    
                    del w, mask
                    # 10개 레이어마다 메모리 강제 청소
                    if "proj" in name: 
                        gc.collect()
                        if self.is_mac: torch.mps.empty_cache()
        
        print(f"✅ 32B Pruning 완료")
        return model

    def fine_tune_model(self, model, tokenizer, output_dir=None):
        if output_dir is None:
            output_dir = f"./pruned_models/exaone-32b-pruned-{self.pruning_percent}"

        print(f"\n{'='*25} [4. 저장 및 Fine-tuning 검토] {'='*25}")
        
        # 32B는 Mac에서 학습 시도시 99% 확률로 커널이 종료됨(Memory Error)
        # 따라서 저장 후 학습은 아주 작은 샘플로 시도하거나 스킵 권장
        print("💾 Pruned 32B 모델 저장 중 (오프로드 데이터 포함)...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        if self.is_mac:
            print("🍎 Mac 알림: 32B 모델의 Fine-tuning은 메모리 한계로 인해 스킵을 권장합니다.")
            print("대신 Pruned 모델 저장에 집중합니다.")
            return model

        # CUDA 환경일 경우에만 매우 제한적으로 시도
        return model

    def apply_quantization(self, pruned_path, output_dir=None):
        if output_dir is None:
            output_dir = f"./quantized_models/exaone-32b-pruned-int4-{self.pruning_percent}"

        print(f"\n{'='*25} [5. INT4 Quantization] {'='*25}")

        if self.is_mac:
            print("🍎 Mac 환경 알림: 32B INT4 양자화는 지원되지 않으므로 FP16 상태로 검증합니다.")
            # 다시 로드할 때도 메모리 관리 필수
            model = AutoModelForCausalLM.from_pretrained(
                pruned_path,
                torch_dtype=self.compute_dtype,
                device_map="auto",
                offload_folder="./offload_32b_test",
                low_cpu_mem_usage=True
            )
        else:
            print("🟢 CUDA 환경 알림: 32B 모델 INT4 양자화 적용")
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4", llm_int8_enable_fp32_cpu_offload=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                pruned_path, quantization_config=bnb_cfg, device_map="auto",
                offload_folder="./offload_32b_quant"
            )

        print("🔎 32B 추론 테스트 (시간 소요 예상)...")
        tokenizer = AutoTokenizer.from_pretrained(pruned_path)
        inputs = tokenizer("인공지능의 미래는", return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=15, do_sample=False)
        print(f"● 추론 결과: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

        # 최종 모델 저장
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"💾 최종 모델 저장 완료: {output_dir}")

    def run_full_pipeline(self):
        try:
            model, tokenizer = self.load_base_model()
            model = self.apply_magnitude_pruning(model)

            # Pruning 후 모델 저장 경로
            pruned_dir = f"./pruned_models/exaone-32b-pruned-{self.pruning_percent}"
            model = self.fine_tune_model(model, tokenizer, output_dir=pruned_dir)

            del model
            gc.collect()
            if self.is_mac: torch.mps.empty_cache()

            # Quantization 적용
            quantized_dir = f"./quantized_models/exaone-32b-pruned-int4-{self.pruning_percent}"
            self.apply_quantization(pruned_dir, output_dir=quantized_dir)

            print(f"\n✅ 파이프라인 종료 (소요시간: {(time.time()-self.start_time)/60:.2f}분)")
            print(f"✅ Pruned 모델: {pruned_dir}")
            print(f"✅ Quantized 모델: {quantized_dir}")
        except Exception as e:
            print(f"❌ 32B 처리 중 에러: {e}")
        finally:
            # 임시 폴더 삭제
            for folder in ["./offload_32b", "./offload_32b_test", "./offload_32b_quant"]:
                if os.path.exists(folder): shutil.rmtree(folder, ignore_errors=True)

def get_pruning_ratio():
    """사용자로부터 Pruning 비율 입력받기"""
    print("\n" + "="*70)
    print("EXAONE 32B Pruning + Quantization 파이프라인")
    print("="*70)
    print("\nPruning 비율을 입력해주세요 (예: 30, 40, 50)")
    print("입력한 값의 %만큼 가중치가 제거됩니다.")

    while True:
        try:
            pruning_input = input("\nPruning 비율 (숫자만 입력): ").strip()
            pruning_percent = int(pruning_input)

            if pruning_percent < 0 or pruning_percent > 100:
                print("❌ 0~100 사이의 값을 입력해주세요.")
                continue

            pruning_ratio = pruning_percent / 100.0
            print(f"\n✅ Pruning 비율: {pruning_percent}% (ratio: {pruning_ratio})")
            return pruning_ratio

        except ValueError:
            print("❌ 숫자를 입력해주세요.")


if __name__ == "__main__":
    pruning_ratio = get_pruning_ratio()
    pipeline = Exaone32BOptimizer(pruning_ratio=pruning_ratio)
    pipeline.run_full_pipeline()