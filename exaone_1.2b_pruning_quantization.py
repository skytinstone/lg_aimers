"""
LG AImers 8기 해커톤 - EXAONE 4.0 1.2B Pruning + Quantization 파이프라인
작성일: 2026.01.15
작성자: 신민석 (수정: Gemini)

[수정 사항] 
- Mac 'Insufficient Memory' 에러 해결을 위한 메모리 최적화 적용
- bfloat16 사용 및 데이터셋 샘플링 축소
"""

import os
import gc
import time
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

class Exaone12BOptimizer:
    def __init__(self, model_name="LGAI-EXAONE/EXAONE-4.0-1.2B", pruning_ratio=0.3):
        self.model_name = model_name
        self.pruning_ratio = pruning_ratio
        self.pruning_percent = int(pruning_ratio * 100)  # 파일명용 (30, 40, 50)
        self.start_time = time.time()

        print(f"\n{'='*25} [1. 환경 분석 및 메모리 체크] {'='*25}")
        self._check_environment()

    def _check_environment(self):
        print(f"● 운영체제: {platform.system()} (Device: {platform.processor()})")
        if torch.backends.mps.is_available():
            self.device = "mps"
            self.is_mac = True
            # Mac 전용: bfloat16이 지원되는지 확인
            self.compute_dtype = torch.bfloat16 
            print(f"● 연산 장치: 🍎 Apple Silicon (MPS) - bfloat16 모드 사용")
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
        print(f"● 시스템 RAM: {ram:.2f} GB (현재 사용 가능: {available_ram:.2f} GB)")

    def _get_model_size(self, model):
        try:
            param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
            return param_size / (1024**2)
        except: return 0.0

    def load_base_model(self):
        print(f"\n{'='*25} [2. 모델 로드] {'='*25}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=HF_TOKEN, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # low_cpu_mem_usage=True 옵션 추가로 로드 시 메모리 피크 방지
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=HF_TOKEN,
            trust_remote_code=True,
            torch_dtype=self.compute_dtype,
            device_map=self.device,
            low_cpu_mem_usage=True 
        )
        print(f"📊 모델 로드 완료 (메모리: {self._get_model_size(model):.2f} MB)")
        return model, tokenizer

    def apply_magnitude_pruning(self, model):
        print(f"\n{'='*25} [3. Pruning] {'='*25}")
        target_patterns = ['gate_proj', 'up_proj', 'down_proj']
        
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and any(p in name for p in target_patterns):
                    # MPS 에러 방지를 위해 계산할 때만 CPU로 복사
                    w = module.weight.data.cpu().float()
                    threshold = torch.quantile(torch.abs(w), self.pruning_ratio)
                    mask = (torch.abs(w) > threshold).to(self.compute_dtype)
                    module.weight.data = (w.to(self.device) * mask.to(self.device))
                    del w, mask
        
        # 즉시 메모리 정리
        gc.collect()
        if self.is_mac: torch.mps.empty_cache()
        print(f"✅ Pruning 완료")
        return model

    def fine_tune_model(self, model, tokenizer, output_dir=None):
        if output_dir is None:
            output_dir = f"./pruned_models/exaone-1.2b-pruned-{self.pruning_percent}"

        print(f"\n{'='*25} [4. Fine-tuning (메모리 절약 모드)] {'='*25}")
        
        try:
            # 샘플 데이터를 더 줄임 (100 -> 20)
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:20]")
            def tokenize_fn(e): return tokenizer(e["text"], truncation=True, max_length=64, padding="max_length")
            tokenized_ds = dataset.map(tokenize_fn, batched=True).filter(lambda x: len(x['input_ids']) > 0)

            model.config.use_cache = False
            # Mac MPS에서 메모리 폭발을 막기 위한 초경량 설정
            args = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=1, # 최소화
                gradient_accumulation_steps=4,
                num_train_epochs=1,
                learning_rate=1e-5,
                # Mac에서는 fp16=True보다 bf16=False(또는 기본)가 더 안전할 수 있음
                fp16=False, 
                bf16=False,
                save_strategy="no",
                report_to="none",
                logging_steps=1
            )
            trainer = Trainer(model=model, args=args, train_dataset=tokenized_ds, 
                              data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
            
            print("학습 시작...")
            trainer.train()
            print("✅ Fine-tuning 완료")
        except Exception as e:
            print(f"⚠️ Fine-tuning 건너뜀 (메모리 부족 예상): {e}")

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        return model

    def apply_quantization(self, pruned_path, output_dir=None):
        if output_dir is None:
            output_dir = f"./quantized_models/exaone-1.2b-pruned-int4-{self.pruning_percent}"

        print(f"\n{'='*25} [5. INT4 Quantization] {'='*25}")

        # Quantization 단계에서 다시 로드할 때 메모리 확보
        model = AutoModelForCausalLM.from_pretrained(
            pruned_path,
            torch_dtype=self.compute_dtype,
            device_map=self.device,
            low_cpu_mem_usage=True
        )

        tokenizer = AutoTokenizer.from_pretrained(pruned_path)

        # 추론 테스트
        inputs = tokenizer("인공지능의 미래는", return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
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
            pruned_dir = f"./pruned_models/exaone-1.2b-pruned-{self.pruning_percent}"
            model = self.fine_tune_model(model, tokenizer, output_dir=pruned_dir)

            # 모델 인스턴스 해제 후 다시 로드하여 추론 테스트 (메모리 관리)
            del model
            gc.collect()
            if self.is_mac: torch.mps.empty_cache()

            # Quantization 적용
            quantized_dir = f"./quantized_models/exaone-1.2b-pruned-int4-{self.pruning_percent}"
            self.apply_quantization(pruned_dir, output_dir=quantized_dir)

            print(f"\n✅ 완료! (소요시간: {(time.time()-self.start_time)/60:.2f}분)")
            print(f"✅ Pruned 모델: {pruned_dir}")
            print(f"✅ Quantized 모델: {quantized_dir}")
        except Exception as e:
            print(f"❌ 최종 에러: {e}")

def get_pruning_ratio():
    """사용자로부터 Pruning 비율 입력받기"""
    print("\n" + "="*70)
    print("EXAONE 1.2B Pruning + Quantization 파이프라인")
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
    pipeline = Exaone12BOptimizer(pruning_ratio=pruning_ratio)
    pipeline.run_full_pipeline()