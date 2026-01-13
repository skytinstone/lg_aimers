"""
EXAONE 모델 공식 평가 스크립트 (LG AImers 8기 해커톤)
- MMLU (Multi-task Language Understanding)
- GPQA (Graduate-Level Physics, Chemistry, Biology)
- IFEval (Instruction Following Evaluation)
- LiveCodeBench (Coding) - 준비 단계

가이드라인 100% 준수
작성일: 2026.01.13
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from dotenv import load_dotenv
import time
from tqdm import tqdm
from datetime import datetime
import re

load_dotenv()


class OfficialExaoneEvaluator:
    def __init__(self, model_path, model_name="EXAONE"):
        """
        공식 평가 스크립트
        Args:
            model_path: 양자화된 모델 경로
            model_name: 모델 이름 (로그 파일명에 사용)
        """
        self.model_path = model_path
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 날짜와 시간을 포함한 로그 파일명
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"evaluation_log_{model_name}_{timestamp}.json"
        self.results_filename = f"evaluation_results_{model_name}_{timestamp}.json"
        
        print(f"=" * 70)
        print(f"공식 평가 시작")
        print(f"=" * 70)
        print(f"평가 모델: {model_name}")
        print(f"모델 경로: {model_path}")
        print(f"Device: {self.device}")
        print(f"로그 파일: {self.log_filename}")
        print(f"결과 파일: {self.results_filename}")
        print(f"=" * 70)
        
        self.model = None
        self.tokenizer = None
        
        # 전체 로그 저장용
        self.evaluation_log = {
            "metadata": {
                "model_name": model_name,
                "model_path": model_path,
                "device": self.device,
                "start_time": timestamp,
                "guideline_version": "LG_AImers_8th_Official"
            },
            "evaluations": {}
        }
        
    def load_model(self):
        """저장된 양자화 모델 로드"""
        print("\n모델 로드 중...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # device_map 없이 로드 후 수동 배치 (메타 텐서 오류 방지)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )

            # 모델을 eval 모드로 전환 (가이드라인 준수)
            self.model.eval()

            # 디바이스로 이동
            print(f"모델을 {self.device}로 이동 중...")
            try:
                self.model = self.model.to(self.device)
            except RuntimeError as e:
                print(f"⚠️ GPU 메모리 부족, CPU로 전환: {e}")
                self.device = "cpu"
                self.model = self.model.to("cpu")

            print(f"✓ 모델 로드 완료 (디바이스: {self.device})")

            # 메모리 사용량
            if self.device == "cuda":
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                print(f"GPU 메모리 사용: {allocated:.2f} GB\n")
                
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            raise
    
    def generate_text(self, prompt, max_new_tokens=64, temperature=0.0, top_p=1.0):
        """
        텍스트 생성 (가이드라인 준수)
        - temperature=0.0: greedy decoding
        - top_p=1.0: 제한 없음
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                do_sample=(temperature > 0),
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 프롬프트 제거
        response = generated[len(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):].strip()
        return response
    
    # ========== MMLU 평가 (가이드라인 준수) ==========
    def evaluate_mmlu(self, num_samples=None):
        """
        MMLU 평가 (cais/mmlu 데이터셋 사용)
        가이드라인:
        - temperature=0.0, top_p=1.0
        - max_new_tokens=64
        - greedy decoding
        """
        print("\n" + "="*70)
        print("MMLU (Multi-task Language Understanding) 평가")
        print("데이터셋: cais/mmlu")
        print("="*70)
        
        eval_start = time.time()
        
        try:
            # 데이터셋 로드
            print("데이터셋 로드 중...")
            dataset = load_dataset("cais/mmlu", "all")
            test_data = dataset["test"]
            
            if num_samples:
                test_data = test_data.select(range(min(num_samples, len(test_data))))
            
            print(f"평가 샘플 수: {len(test_data)}\n")
            
            correct = 0
            total = len(test_data)
            results_by_subject = {}
            detailed_results = []
            
            for idx, example in enumerate(tqdm(test_data, desc="MMLU 평가")):
                # 가이드라인 프롬프트 템플릿
                prompt = self._build_mmlu_prompt_guideline(example)
                
                # 가이드라인 파라미터로 생성
                response = self.generate_text(
                    prompt, 
                    max_new_tokens=64,  # 가이드라인: 16-64
                    temperature=0.0,     # 가이드라인: greedy
                    top_p=1.0           # 가이드라인: 1.0
                )
                
                # 답변 파싱
                predicted = self._parse_choice_letter(response)
                correct_answer = self._get_answer_index(example)
                
                is_correct = (predicted == correct_answer)
                correct += is_correct
                
                # 과목별 집계
                subject = example.get("subject", "unknown")
                if subject not in results_by_subject:
                    results_by_subject[subject] = {"correct": 0, "total": 0}
                results_by_subject[subject]["total"] += 1
                results_by_subject[subject]["correct"] += is_correct
                
                # 상세 로그
                detailed_results.append({
                    "index": idx,
                    "subject": subject,
                    "question": example["question"],
                    "choices": example["choices"],
                    "correct_answer": chr(65 + correct_answer),
                    "predicted_answer": chr(65 + predicted) if predicted >= 0 else "N/A",
                    "is_correct": is_correct,
                    "model_response": response
                })
                
                if (idx + 1) % 10 == 0:
                    current_acc = correct / (idx + 1) * 100
                    print(f"  진행률: {idx+1}/{total}, 현재 정확도: {current_acc:.2f}%")
            
            # 최종 결과
            accuracy = correct / total * 100
            eval_time = time.time() - eval_start
            
            print(f"\n{'='*70}")
            print(f"MMLU 평가 완료")
            print(f"{'='*70}")
            print(f"전체 정확도: {correct}/{total} = {accuracy:.2f}%")
            print(f"평가 시간: {eval_time/60:.1f}분")
            
            print(f"\n과목별 정확도 (상위 10개):")
            sorted_subjects = sorted(
                results_by_subject.items(), 
                key=lambda x: x[1]["correct"] / x[1]["total"], 
                reverse=True
            )
            for subject, result in sorted_subjects[:10]:
                acc = result["correct"] / result["total"] * 100
                print(f"  {subject}: {result['correct']}/{result['total']} = {acc:.1f}%")
            
            # 결과 저장
            mmlu_results = {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "evaluation_time_seconds": eval_time,
                "by_subject": results_by_subject,
                "detailed_results": detailed_results,
                "parameters": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_new_tokens": 64
                }
            }
            
            self.evaluation_log["evaluations"]["mmlu"] = mmlu_results
            self._save_log()
            
            return mmlu_results
            
        except Exception as e:
            print(f"\n❌ MMLU 평가 실패: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def _build_mmlu_prompt_guideline(self, example):
        """가이드라인 준수 MMLU 프롬프트"""
        question = example["question"]
        choices = example["choices"]
        
        prompt = "You are a helpful AI assistant.\n"
        prompt += "Choose the correct answer to the following multiple-choice question.\n\n"
        prompt += f"Question: {question}\n\n"
        prompt += "Choices:\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "\nAnswer with only the letter of the correct option (A, B, C, or D)."
        
        return prompt
    
    def _get_answer_index(self, example):
        """정답 인덱스 추출 (0-3)"""
        answer = example.get("answer")
        if isinstance(answer, int):
            return answer
        elif isinstance(answer, str):
            if answer in ['A', 'B', 'C', 'D']:
                return ord(answer) - 65
            elif answer.isdigit():
                return int(answer)
        return 0
    
    def _parse_choice_letter(self, response):
        """응답에서 A/B/C/D 추출"""
        response = response.strip().upper()
        
        # 첫 글자 확인
        if response and response[0] in ['A', 'B', 'C', 'D']:
            return ord(response[0]) - 65
        
        # 정규표현식으로 찾기
        match = re.search(r'\b([A-D])\b', response)
        if match:
            return ord(match.group(1)) - 65
        
        return -1
    
    # ========== GPQA 평가 (가이드라인 준수) ==========
    def evaluate_gpqa(self, num_samples=None):
        """
        GPQA 평가 (Idavidrein/gpqa 데이터셋 사용)
        가이드라인:
        - temperature=0.0, top_p=1.0
        - max_new_tokens=32
        """
        print("\n" + "="*70)
        print("GPQA (Graduate-Level Science QA) 평가")
        print("데이터셋: Idavidrein/gpqa")
        print("="*70)
        
        eval_start = time.time()
        
        try:
            # 데이터셋 로드
            print("데이터셋 로드 중...")
            dataset = load_dataset("Idavidrein/gpqa", "gpqa_main")
            test_data = dataset["train"]  # GPQA는 train split 사용
            
            if num_samples:
                test_data = test_data.select(range(min(num_samples, len(test_data))))
            
            print(f"평가 샘플 수: {len(test_data)}\n")
            
            correct = 0
            total = len(test_data)
            results_by_domain = {}
            detailed_results = []
            
            for idx, example in enumerate(tqdm(test_data, desc="GPQA 평가")):
                # 가이드라인 프롬프트
                prompt = self._build_gpqa_prompt_guideline(example)
                
                # 가이드라인 파라미터
                response = self.generate_text(
                    prompt,
                    max_new_tokens=32,  # 가이드라인: 16-32
                    temperature=0.0,
                    top_p=1.0
                )
                
                # 답변 파싱
                predicted = self._parse_choice_letter(response)
                correct_answer = self._get_gpqa_answer(example)
                
                is_correct = (predicted == correct_answer)
                correct += is_correct
                
                # 도메인별 집계
                domain = example.get("High-level domain", "unknown")
                if domain not in results_by_domain:
                    results_by_domain[domain] = {"correct": 0, "total": 0}
                results_by_domain[domain]["total"] += 1
                results_by_domain[domain]["correct"] += is_correct
                
                # 상세 로그
                detailed_results.append({
                    "index": idx,
                    "domain": domain,
                    "question": example.get("Question", ""),
                    "correct_answer": chr(65 + correct_answer),
                    "predicted_answer": chr(65 + predicted) if predicted >= 0 else "N/A",
                    "is_correct": is_correct,
                    "model_response": response
                })
            
            # 최종 결과
            accuracy = correct / total * 100
            eval_time = time.time() - eval_start
            
            print(f"\n{'='*70}")
            print(f"GPQA 평가 완료")
            print(f"{'='*70}")
            print(f"전체 정확도: {correct}/{total} = {accuracy:.2f}%")
            print(f"평가 시간: {eval_time/60:.1f}분")
            
            print(f"\n도메인별 정확도:")
            for domain, result in results_by_domain.items():
                acc = result["correct"] / result["total"] * 100
                print(f"  {domain}: {result['correct']}/{result['total']} = {acc:.1f}%")
            
            # 결과 저장
            gpqa_results = {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "evaluation_time_seconds": eval_time,
                "by_domain": results_by_domain,
                "detailed_results": detailed_results,
                "parameters": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_new_tokens": 32
                }
            }
            
            self.evaluation_log["evaluations"]["gpqa"] = gpqa_results
            self._save_log()
            
            return gpqa_results
            
        except Exception as e:
            print(f"\n❌ GPQA 평가 실패: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def _build_gpqa_prompt_guideline(self, example):
        """가이드라인 준수 GPQA 프롬프트"""
        domain = example.get("High-level domain", "science")
        question = example.get("Question", "")
        
        # 선택지 추출 (Correct Answer, Incorrect Answer 1, 2, 3)
        choices = [
            example.get("Correct Answer", ""),
            example.get("Incorrect Answer 1", ""),
            example.get("Incorrect Answer 2", ""),
            example.get("Incorrect Answer 3", "")
        ]
        
        prompt = f"You are an expert in {domain}.\n"
        prompt += "Answer the following multiple-choice question.\n\n"
        prompt += f"Question: {question}\n\n"
        prompt += "Choices:\n"
        for i, choice in enumerate(choices):
            if choice:
                prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "\nAnswer with only the letter (A, B, C, or D)."
        
        return prompt
    
    def _get_gpqa_answer(self, example):
        """GPQA 정답 추출 (정답은 항상 A - Correct Answer)"""
        return 0  # GPQA에서 Correct Answer는 항상 첫 번째
    
    # ========== IFEval 평가 준비 ==========
    def evaluate_ifeval(self):
        """
        IFEval 평가 (준비 단계)
        실제 실행을 위해서는 ifeval 라이브러리 설치 필요:
        git clone https://github.com/oKatanaaa/ifeval.git
        cd ifeval
        pip install -e .
        """
        print("\n" + "="*70)
        print("IFEval (Instruction Following Evaluation) 평가")
        print("="*70)
        
        print("⚠️ IFEval 평가를 위해서는 다음 라이브러리 설치가 필요합니다:")
        print("   git clone https://github.com/oKatanaaa/ifeval.git")
        print("   cd ifeval")
        print("   pip install -e .")
        print("\n설치 후 별도 스크립트로 실행해주세요.")
        
        return {"status": "not_implemented", "note": "ifeval library required"}
    
    # ========== 종합 평가 ==========
    def run_full_evaluation(self, mmlu_samples=None, gpqa_samples=None):
        """
        전체 평가 실행
        Args:
            mmlu_samples: MMLU 샘플 수 (None=전체)
            gpqa_samples: GPQA 샘플 수 (None=전체)
        """
        print("\n" + "="*70)
        print(f"{self.model_name} 모델 종합 평가 (공식)")
        print("="*70)
        
        overall_start = time.time()
        
        # 1. MMLU
        print("\n[1/3] MMLU 평가 시작")
        try:
            mmlu_results = self.evaluate_mmlu(num_samples=mmlu_samples)
        except Exception as e:
            print(f"MMLU 평가 실패: {e}")
            mmlu_results = {"error": str(e)}
        
        # 2. GPQA
        print("\n[2/3] GPQA 평가 시작")
        try:
            gpqa_results = self.evaluate_gpqa(num_samples=gpqa_samples)
        except Exception as e:
            print(f"GPQA 평가 실패: {e}")
            gpqa_results = {"error": str(e)}
        
        # 3. IFEval (준비 단계)
        print("\n[3/3] IFEval 평가 (준비 단계)")
        ifeval_results = self.evaluate_ifeval()
        
        overall_time = time.time() - overall_start
        
        # 최종 결과 저장
        self.evaluation_log["metadata"]["end_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.evaluation_log["metadata"]["total_evaluation_time_seconds"] = overall_time
        
        self._save_log()
        self._save_results_summary()
        
        print("\n" + "="*70)
        print("전체 평가 완료!")
        print("="*70)
        print(f"총 소요 시간: {overall_time/60:.1f}분")
        print(f"로그 파일: {self.log_filename}")
        print(f"결과 파일: {self.results_filename}")
        print("="*70)
        
        return self.evaluation_log
    
    def _save_log(self):
        """평가 로그 저장 (실시간)"""
        with open(self.log_filename, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_log, f, indent=2, ensure_ascii=False)
    
    def _save_results_summary(self):
        """결과 요약 저장"""
        summary = {
            "model_name": self.model_name,
            "evaluation_date": self.evaluation_log["metadata"]["start_time"],
            "total_time_minutes": self.evaluation_log["metadata"].get("total_evaluation_time_seconds", 0) / 60,
            "results": {}
        }
        
        # MMLU 요약
        if "mmlu" in self.evaluation_log["evaluations"]:
            mmlu = self.evaluation_log["evaluations"]["mmlu"]
            if "accuracy" in mmlu:
                summary["results"]["mmlu"] = {
                    "accuracy": mmlu["accuracy"],
                    "correct": mmlu["correct"],
                    "total": mmlu["total"]
                }
        
        # GPQA 요약
        if "gpqa" in self.evaluation_log["evaluations"]:
            gpqa = self.evaluation_log["evaluations"]["gpqa"]
            if "accuracy" in gpqa:
                summary["results"]["gpqa"] = {
                    "accuracy": gpqa["accuracy"],
                    "correct": gpqa["correct"],
                    "total": gpqa["total"]
                }
        
        with open(self.results_filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


def main():
    """메인 실행"""
    print("="*70)
    print("LG AImers 8기 - EXAONE 모델 공식 평가 시스템")
    print("="*70)
    
    # 평가할 모델 선택
    models_to_evaluate = []
    
    # 1.2B 모델
    if os.path.exists("./quantized_models/exaone-1.2b-int4"):
        models_to_evaluate.append({
            "path": "./quantized_models/exaone-1.2b-int4",
            "name": "EXAONE-1.2B-INT4"
        })
    
    # 32B 모델
    if os.path.exists("./quantized_models/exaone-32b-int4"):
        models_to_evaluate.append({
            "path": "./quantized_models/exaone-32b-int4",
            "name": "EXAONE-32B-INT4"
        })
    
    if not models_to_evaluate:
        print("❌ 평가할 양자화된 모델이 없습니다.")
        print("먼저 양자화를 수행해주세요.")
        return
    
    # 각 모델 평가
    for model_info in models_to_evaluate:
        print(f"\n{'='*70}")
        print(f"모델 평가: {model_info['name']}")
        print(f"{'='*70}")
        
        evaluator = OfficialExaoneEvaluator(
            model_path=model_info["path"],
            model_name=model_info["name"]
        )
        
        try:
            evaluator.load_model()
            
            # 전체 평가 실행
            # num_samples를 지정하면 샘플 평가, None이면 전체 평가
            results = evaluator.run_full_evaluation(
                mmlu_samples=100,   # MMLU 100 샘플 (전체: None)
                gpqa_samples=50     # GPQA 50 샘플 (전체: None)
            )
            
            print(f"\n✓ {model_info['name']} 평가 완료!")
            
        except Exception as e:
            print(f"\n❌ {model_info['name']} 평가 실패: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("모든 평가 완료!")
    print("="*70)


if __name__ == "__main__":
    main()