"""
EXAONE 평가 스크립트 (1.2B -> 32B 순차 평가 + 최종 표 출력 + 시간 로그)
- Core: MMLU, GPQA, IFEval
- Extra: LiveCodeBench, AIME, HELMET, LongBench, BFCL, KSM
- 각 벤치마크 start/end/elapsed 로그 저장
- 실행 후 결과를 표로 출력

작성일: 2026.01.14
"""

import os
import json
import time
import re
import gc
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


# -------------------------
# IFEval import (safe)
# -------------------------
try:
    from ifeval import get_default_dataset, get_default_instruction_registry, Evaluator
    IFEVAL_AVAILABLE = True
    IFEVAL_IMPORT_ERROR = None
except Exception as e:
    IFEVAL_AVAILABLE = False
    IFEVAL_IMPORT_ERROR = str(e)
    print(f"[WARN] IFEval 사용 불가(미설치/버전불일치 가능): {e}")


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def fmt_score(x):
    """표 출력용 점수 포맷: float -> '12.34', None -> '-', dict error/skip -> 'SKIP/ERR'"""
    if x is None:
        return "-"
    if isinstance(x, (int, float)):
        return f"{x:.2f}"
    if isinstance(x, dict):
        if x.get("status") == "skipped":
            return "SKIP"
        if x.get("status") == "error":
            return "ERR"
        if "accuracy" in x and isinstance(x["accuracy"], (int, float)):
            return f"{x['accuracy']:.2f}"
    return "-"


def print_final_table(scores_12b: dict, scores_32b: dict):
    """
    요청한 표 형식:
    Score | 1.2B Model | 32B Model
    rows: MMLU-Diamond, GPQA, IFEval, LiveCodeBench, AIME, HELMET, LongBench, BFCL, KSM
    """
    rows = [
        ("MMLU-Diamond", "mmlu"),
        ("GPQA", "gpqa_diamond"),
        ("IFEval", "ifeval"),
        ("LiveCodeBench", "livecodebench"),
        ("AIME", "aime_2025"),
        ("HELMET", "helmet"),
        ("LongBench", "longbench"),
        ("BFCL", "bfcl_v3"),
        ("KSM", "ksm"),
    ]

    col1 = "1.2B Model"
    col2 = "32B Model"

    print("\n" + "=" * 80)
    print("최종 평가 결과 표")
    print("=" * 80)
    print(f"{'Score':<18} {col1:>15} {col2:>15}")
    print("-" * 80)

    for display_name, key in rows:
        s1 = fmt_score(scores_12b.get(key))
        s2 = fmt_score(scores_32b.get(key))
        # 숫자로 들어오면 % 표시
        if s1 not in ("-", "SKIP", "ERR"):
            s1 = s1 + "%"
        if s2 not in ("-", "SKIP", "ERR"):
            s2 = s2 + "%"
        print(f"{display_name:<18} {s1:>15} {s2:>15}")

    print("=" * 80 + "\n")


class ExaoneEvaluator:
    CONFIG = {
        "mmlu": {
            "dataset": "cais/mmlu",
            "config": "all",
            "split": "test",
            "n_shot": 5,
            "max_new_tokens": 5,
        },
        "gpqa_diamond": {
            "dataset": "Idavidrein/gpqa",
            "config": "gpqa_diamond",
            "split": "train",
            "max_new_tokens": 5,
        },
        "ifeval": {
            "max_new_tokens": 512,
        },
        "livecodebench": {
            "dataset": "livecodebench/code_generation_lite",
            "config": "release_v6",
            "split": "test",
            "max_new_tokens": 1024,
        },
        "aime_2025": {
            "dataset": "AI-MO/aimo-validation-aime",
            "config": None,
            "split": "train",
            "max_new_tokens": 512,
        },
        "helmet": {
            "dataset": "princeton-nlp/HELMET",
            "config": None,
            "split": "test",
            "max_new_tokens": 128,
        },
        "longbench": {
            "dataset": "THUDM/LongBench",
            "config": "2wikimqa_e",
            "split": "test",
            "max_new_tokens": 64,
        },
        "bfcl_v3": {
            "dataset": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
            "config": None,
            "split": "train",
            "max_new_tokens": 512,
        },
        "ksm": {
            "dataset": "HAERAE-HUB/KSM",
            "config": None,
            "split": "test",
            "max_new_tokens": 64,
        },
    }

    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name
        self.hf_token = os.getenv("HF_TOKEN")

        # Mac M3 Pro 지원 추가
        if torch.cuda.is_available():
            self.device = "cuda"
            self.compute_dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.compute_dtype = torch.bfloat16
            print("[INFO] Apple Silicon (MPS) 감지 - bfloat16 모드 사용")
        else:
            self.device = "cpu"
            self.compute_dtype = torch.float32

        ts = now_tag()
        self.log_filename = f"evaluation_log_{model_name}_{ts}.json"
        self.results_filename = f"evaluation_results_{model_name}_{ts}.json"

        self.model = None
        self.tokenizer = None

        self.log = {
            "metadata": {
                "model_name": model_name,
                "model_path": model_path,
                "device": self.device,
                "start_time": now_str(),
            },
            "evaluations": {}
        }

    def _save_log(self):
        with open(self.log_filename, "w", encoding="utf-8") as f:
            json.dump(self.log, f, indent=2, ensure_ascii=False)

    def _start_eval(self, key: str):
        self.log["evaluations"].setdefault(key, {})
        self.log["evaluations"][key]["start_time"] = now_str()
        self.log["evaluations"][key]["status"] = "running"
        self._save_log()

    def _end_eval(self, key: str, t0: float, status: str = "ok", error: str = None):
        self.log["evaluations"].setdefault(key, {})
        self.log["evaluations"][key]["end_time"] = now_str()
        self.log["evaluations"][key]["elapsed_sec"] = round(time.time() - t0, 2)
        self.log["evaluations"][key]["status"] = status
        if error:
            self.log["evaluations"][key]["error_message"] = str(error)
        self._save_log()

    def load_model(self):
        print("\n" + "=" * 70)
        print(f"[MODEL] {self.model_name} 로드")
        print("=" * 70)

        # tokenizer: fix_mistral_regex는 모델/버전 조합에 따라 깨질 수 있음 → 재시도
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                fix_mistral_regex=True,
            )
        except TypeError as e:
            print(f"[WARN] fix_mistral_regex 실패 → 옵션 없이 재시도: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        try:
            self.model = self.model.to(self.device)
        except RuntimeError as e:
            print(f"[WARN] GPU 메모리 부족 → CPU 전환: {e}")
            self.device = "cpu"
            self.model = self.model.to("cpu")
            self.log["metadata"]["device"] = self.device

        print(f"[MODEL] 로드 완료 (device={self.device})")
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f"[MODEL] GPU 메모리 사용: {allocated:.2f} GB")

        self._save_log()

    def unload_model(self):
        print("\n[MODEL] 언로드/메모리 정리")
        try:
            del self.model
            del self.tokenizer
        except Exception:
            pass
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(2)

    def generate_text(self, prompt, max_new_tokens=64, max_length=4096):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )

        full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        in_text = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        return full[len(in_text):].strip()

    # -------------------------
    # helpers
    # -------------------------
    def _parse_choice_letter(self, response):
        r = response.strip().upper()
        if r and r[0] in ["A", "B", "C", "D"]:
            return ord(r[0]) - 65
        m = re.search(r"\b([A-D])\b", r)
        if m:
            return ord(m.group(1)) - 65
        return -1

    def _get_answer_index(self, ex):
        ans = ex.get("answer")
        if isinstance(ans, int):
            return ans
        if isinstance(ans, str):
            if ans in ["A", "B", "C", "D"]:
                return ord(ans) - 65
            if ans.isdigit():
                return int(ans)
        return 0

    # -------------------------
    # benchmarks
    # -------------------------
    def eval_mmlu(self, samples=1500):
        key = "mmlu"
        self._start_eval(key)
        t0 = time.time()
        try:
            cfg = self.CONFIG[key]
            ds = load_dataset(cfg["dataset"], cfg["config"])
            test_data = ds[cfg["split"]]
            dev_data = ds["dev"]

            # 샘플 제한
            test_data = test_data.select(range(min(samples, len(test_data))))

            # subject별 dev 묶기 (5-shot)
            dev_by_subject = {}
            for ex in dev_data:
                dev_by_subject.setdefault(ex.get("subject", "unknown"), []).append(ex)

            correct = 0
            total = len(test_data)

            for ex in tqdm(test_data, desc=f"{self.model_name} MMLU"):
                subject = ex.get("subject", "unknown")
                shots = dev_by_subject.get(subject, [])[:5]

                prompt = "The following are multiple choice questions (with answers).\n\n"
                for s in shots:
                    prompt += f"Question: {s['question']}\n"
                    for i, c in enumerate(s["choices"]):
                        prompt += f"{chr(65+i)}. {c}\n"
                    a = self._get_answer_index(s)
                    prompt += f"Answer: {chr(65+a)}\n\n"

                prompt += f"Question: {ex['question']}\n"
                for i, c in enumerate(ex["choices"]):
                    prompt += f"{chr(65+i)}. {c}\n"
                prompt += "Answer:"

                resp = self.generate_text(prompt, max_new_tokens=cfg["max_new_tokens"])
                pred = self._parse_choice_letter(resp)
                gold = self._get_answer_index(ex)
                if pred == gold:
                    correct += 1

            acc = correct / total * 100
            self.log["evaluations"][key].update({"accuracy": round(acc, 4), "correct": correct, "total": total})
            self._end_eval(key, t0, status="ok")
            return self.log["evaluations"][key]
        except Exception as e:
            self._end_eval(key, t0, status="error", error=e)
            return {"status": "error", "error_message": str(e)}

    def eval_gpqa(self, samples=None):
        key = "gpqa_diamond"
        self._start_eval(key)
        t0 = time.time()
        try:
            cfg = self.CONFIG[key]
            ds = load_dataset(cfg["dataset"], cfg["config"], token=self.hf_token)
            data = ds[cfg["split"]]
            if samples is not None:
                data = data.select(range(min(samples, len(data))))

            correct = 0
            total = len(data)
            for ex in tqdm(data, desc=f"{self.model_name} GPQA"):
                q = ex.get("Question", "")
                choices = [
                    ex.get("Correct Answer", ""),
                    ex.get("Incorrect Answer 1", ""),
                    ex.get("Incorrect Answer 2", ""),
                    ex.get("Incorrect Answer 3", "")
                ]
                prompt = "Answer the following multiple choice question.\n\n"
                prompt += f"Question: {q}\n\nChoices:\n"
                for i, c in enumerate(choices):
                    if c:
                        prompt += f"{chr(65+i)}. {c}\n"
                prompt += "\nAnswer:"

                resp = self.generate_text(prompt, max_new_tokens=cfg["max_new_tokens"])
                pred = self._parse_choice_letter(resp)
                if pred == 0:  # Correct Answer를 A로 둔 구성
                    correct += 1

            acc = correct / total * 100
            self.log["evaluations"][key].update({"accuracy": round(acc, 4), "correct": correct, "total": total})
            self._end_eval(key, t0, status="ok")
            return self.log["evaluations"][key]
        except Exception as e:
            self._end_eval(key, t0, status="error", error=e)
            return {"status": "error", "error_message": str(e)}

    def eval_ifeval(self, samples=150):
        key = "ifeval"
        self._start_eval(key)
        t0 = time.time()
        try:
            if not IFEVAL_AVAILABLE:
                self.log["evaluations"][key].update({"status": "skipped", "reason": IFEVAL_IMPORT_ERROR})
                self._end_eval(key, t0, status="skipped")
                return {"status": "skipped", "reason": IFEVAL_IMPORT_ERROR}

            cfg = self.CONFIG[key]
            registry = get_default_instruction_registry("en")
            evaluator = Evaluator(registry)
            examples = get_default_dataset("en")
            examples = examples[:samples]

            responses = {}
            for ex in tqdm(examples, desc=f"{self.model_name} IFEval"):
                responses[ex.prompt] = self.generate_text(ex.prompt, max_new_tokens=cfg["max_new_tokens"], max_length=8192)

            report, _ = evaluator.evaluate(examples, responses)
            acc = report.overall_score * 100

            self.log["evaluations"][key].update({"accuracy": round(acc, 4), "total": len(examples)})
            self._end_eval(key, t0, status="ok")
            return self.log["evaluations"][key]
        except Exception as e:
            self._end_eval(key, t0, status="error", error=e)
            return {"status": "error", "error_message": str(e)}

    # LiveCodeBench (proxy: compile check)
    def eval_livecodebench(self, samples=50):
        key = "livecodebench"
        self._start_eval(key)
        t0 = time.time()
        try:
            cfg = self.CONFIG[key]
            ds = load_dataset(cfg["dataset"], cfg["config"], token=self.hf_token)
            data = ds[cfg["split"]]
            data = data.select(range(min(samples, len(data))))

            correct = 0
            total = len(data)
            for ex in tqdm(data, desc=f"{self.model_name} LiveCodeBench"):
                q = ex.get("question_content", ex.get("prompt", ""))
                prompt = f"Solve the following programming problem.\n\n{q}\n\nWrite the solution in Python:\n```python\n"
                resp = self.generate_text(prompt, max_new_tokens=cfg["max_new_tokens"], max_length=8192)

                code = resp.split("```")[0] if "```" in resp else resp
                try:
                    compile(code, "<string>", "exec")
                    correct += 1
                except Exception:
                    pass

            acc = correct / total * 100
            self.log["evaluations"][key].update({
                "accuracy": round(acc, 4), "correct": correct, "total": total,
                "note": "proxy metric: compile check"
            })
            self._end_eval(key, t0, status="ok")
            return self.log["evaluations"][key]
        except Exception as e:
            self._end_eval(key, t0, status="error", error=e)
            return {"status": "error", "error_message": str(e)}

    # AIME (proxy: exact final number match)
    def eval_aime(self, samples=50):
        key = "aime_2025"
        self._start_eval(key)
        t0 = time.time()
        try:
            cfg = self.CONFIG[key]
            ds = load_dataset(cfg["dataset"], token=self.hf_token)
            data = ds[cfg["split"]]
            data = data.select(range(min(samples, len(data))))

            correct = 0
            total = len(data)
            for ex in tqdm(data, desc=f"{self.model_name} AIME"):
                problem = ex.get("problem", ex.get("question", ""))
                gold = str(ex.get("answer", ex.get("solution", ""))).strip()
                prompt = f"Solve the following math problem. Give your final answer as a single number.\n\nProblem: {problem}\n\nAnswer:"
                resp = self.generate_text(prompt, max_new_tokens=cfg["max_new_tokens"], max_length=8192)

                nums = re.findall(r"-?\d+\.?\d*", resp)
                pred = nums[-1].strip() if nums else ""
                if pred == gold:
                    correct += 1

            acc = correct / total * 100
            self.log["evaluations"][key].update({"accuracy": round(acc, 4), "correct": correct, "total": total})
            self._end_eval(key, t0, status="ok")
            return self.log["evaluations"][key]
        except Exception as e:
            self._end_eval(key, t0, status="error", error=e)
            return {"status": "error", "error_message": str(e)}

    # HELMET (proxy: string-inclusion)
    def eval_helmet(self, samples=20):
        key = "helmet"
        self._start_eval(key)
        t0 = time.time()
        try:
            cfg = self.CONFIG[key]
            ds = load_dataset(cfg["dataset"], token=self.hf_token)
            split_name = cfg["split"] if cfg["split"] in ds else list(ds.keys())[0]
            data = ds[split_name]
            data = data.select(range(min(samples, len(data))))

            correct = 0
            total = len(data)
            for ex in tqdm(data, desc=f"{self.model_name} HELMET"):
                context = ex.get("context", ex.get("input", ""))
                question = ex.get("question", ex.get("query", ""))
                prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                resp = self.generate_text(prompt, max_new_tokens=cfg["max_new_tokens"], max_length=8192)

                answers = ex.get("answers", ex.get("answer", []))
                if isinstance(answers, str):
                    answers = [answers]
                low = resp.lower()
                ok = any(str(a).lower().strip() in low for a in answers)
                if ok:
                    correct += 1

            acc = correct / total * 100
            self.log["evaluations"][key].update({
                "accuracy": round(acc, 4), "correct": correct, "total": total,
                "note": "proxy metric: string inclusion"
            })
            self._end_eval(key, t0, status="ok")
            return self.log["evaluations"][key]
        except Exception as e:
            self._end_eval(key, t0, status="error", error=e)
            return {"status": "error", "error_message": str(e)}

    # LongBench (proxy: string-inclusion)
    def eval_longbench(self, samples=50):
        key = "longbench"
        self._start_eval(key)
        t0 = time.time()
        try:
            cfg = self.CONFIG[key]
            ds = load_dataset(cfg["dataset"], cfg["config"], token=self.hf_token)
            data = ds[cfg["split"]]
            data = data.select(range(min(samples, len(data))))

            correct = 0
            total = len(data)
            for ex in tqdm(data, desc=f"{self.model_name} LongBench"):
                context = ex.get("context", ex.get("input", ""))
                question = ex.get("question", ex.get("query", ""))
                prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                resp = self.generate_text(prompt, max_new_tokens=cfg["max_new_tokens"], max_length=8192)

                answers = ex.get("answers", ex.get("answer", []))
                if isinstance(answers, str):
                    answers = [answers]
                low = resp.lower()
                ok = any(str(a).lower().strip() in low for a in answers)
                if ok:
                    correct += 1

            acc = correct / total * 100
            self.log["evaluations"][key].update({
                "accuracy": round(acc, 4), "correct": correct, "total": total,
                "note": "proxy metric: string inclusion"
            })
            self._end_eval(key, t0, status="ok")
            return self.log["evaluations"][key]
        except Exception as e:
            self._end_eval(key, t0, status="error", error=e)
            return {"status": "error", "error_message": str(e)}

    # BFCL (proxy: function name match)
    def eval_bfcl(self, samples=50):
        key = "bfcl_v3"
        self._start_eval(key)
        t0 = time.time()
        try:
            cfg = self.CONFIG[key]
            ds = load_dataset(cfg["dataset"], token=self.hf_token)
            split_name = cfg["split"] if cfg["split"] in ds else list(ds.keys())[0]
            data = ds[split_name]
            data = data.select(range(min(samples, len(data))))

            correct = 0
            total = len(data)
            for ex in tqdm(data, desc=f"{self.model_name} BFCL"):
                q = ex.get("question", ex.get("prompt", ""))
                funcs = ex.get("functions", ex.get("tools", []))

                prompt = (
                    "You have access to the following functions:\n"
                    + json.dumps(funcs, indent=2, ensure_ascii=False)
                    + "\n\nUser request: "
                    + str(q)
                    + "\n\nGenerate the appropriate function call in JSON format:"
                )
                resp = self.generate_text(prompt, max_new_tokens=cfg["max_new_tokens"], max_length=8192)

                # JSON에서 함수명만 비교
                ok = False
                try:
                    m = re.search(r"\{.*\}", resp, flags=re.DOTALL)
                    if m:
                        obj = json.loads(m.group(0))
                        expected = ex.get("answer", ex.get("ground_truth", {}))
                        if obj.get("name") == expected.get("name"):
                            ok = True
                except Exception:
                    ok = False

                if ok:
                    correct += 1

            acc = correct / total * 100
            self.log["evaluations"][key].update({
                "accuracy": round(acc, 4), "correct": correct, "total": total,
                "note": "proxy metric: function-name match"
            })
            self._end_eval(key, t0, status="ok")
            return self.log["evaluations"][key]
        except Exception as e:
            self._end_eval(key, t0, status="error", error=e)
            return {"status": "error", "error_message": str(e)}

    # KSM (numeric match)
    def eval_ksm(self, samples=100):
        key = "ksm"
        self._start_eval(key)
        t0 = time.time()
        try:
            cfg = self.CONFIG[key]
            ds = load_dataset(cfg["dataset"], token=self.hf_token)
            split_name = cfg["split"] if cfg["split"] in ds else list(ds.keys())[0]
            data = ds[split_name]
            data = data.select(range(min(samples, len(data))))

            correct = 0
            total = len(data)
            for ex in tqdm(data, desc=f"{self.model_name} KSM"):
                q = ex.get("question", ex.get("problem", ""))
                gold = str(ex.get("answer", "")).strip()

                prompt = f"다음 수학 문제를 풀고 정답을 숫자로만 답하세요.\n\n문제: {q}\n\n정답:"
                resp = self.generate_text(prompt, max_new_tokens=cfg["max_new_tokens"], max_length=8192)

                nums = re.findall(r"-?\d+\.?\d*", resp)
                pred = nums[-1].strip() if nums else ""
                if pred == gold:
                    correct += 1

            acc = correct / total * 100
            self.log["evaluations"][key].update({"accuracy": round(acc, 4), "correct": correct, "total": total})
            self._end_eval(key, t0, status="ok")
            return self.log["evaluations"][key]
        except Exception as e:
            self._end_eval(key, t0, status="error", error=e)
            return {"status": "error", "error_message": str(e)}

    # -------------------------
    # runner
    # -------------------------
    def run_full(
        self,
        mmlu_samples=1500,
        gpqa_samples=None,
        ifeval_samples=150,
        lcb_samples=50,
        aime_samples=50,
        helmet_samples=20,
        longbench_samples=50,
        bfcl_samples=50,
        ksm_samples=100,
    ):
        # Core
        self.eval_mmlu(samples=mmlu_samples)
        self.eval_gpqa(samples=gpqa_samples)
        self.eval_ifeval(samples=ifeval_samples)

        # Extra
        self.eval_livecodebench(samples=lcb_samples)
        self.eval_aime(samples=aime_samples)
        self.eval_helmet(samples=helmet_samples)
        self.eval_longbench(samples=longbench_samples)
        self.eval_bfcl(samples=bfcl_samples)
        self.eval_ksm(samples=ksm_samples)

        self.log["metadata"]["end_time"] = now_str()
        self._save_log()

        # summary 저장(간단)
        summary = {"model": self.model_name, "time": now_str(), "scores": {}}
        for k, v in self.log["evaluations"].items():
            summary["scores"][k] = v.get("accuracy") if isinstance(v.get("accuracy"), (int, float)) else v
        with open(self.results_filename, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n[OK] 결과 저장: {self.results_filename}")
        print(f"[OK] 로그 저장: {self.log_filename}")

        # 점수 dict 리턴
        return summary["scores"]


def run_one_model(model_path, model_name, **kwargs):
    ev = ExaoneEvaluator(model_path, model_name)
    ev.load_model()
    scores = ev.run_full(**kwargs)
    ev.unload_model()
    return scores


def get_user_model_choice():
    """사용자로부터 모델 타입과 크기를 입력받는 함수"""
    import platform

    print("\n" + "="*70)
    print("EXAONE 모델 평가")
    print("="*70)

    # 환경 정보 표시
    print("\n[현재 평가 환경]")
    print(f"  운영체제: {platform.system()}")
    print(f"  프로세서: {platform.processor()}")
    print(f"  아키텍처: {platform.machine()}")

    if torch.cuda.is_available():
        print(f"  연산 장치: NVIDIA GPU (CUDA)")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        print(f"  연산 장치: Apple Silicon (MPS)")
    else:
        print(f"  연산 장치: CPU")

    # 1. 모델 타입 선택
    print("\n평가하고 싶은 모델을 선택해주세요:")
    print("1. Quantized(INT4) 모델")
    print("2. Pruning + Quantized(INT4, Mac은 제외)")

    model_type = None
    while model_type not in ["1", "2"]:
        model_type = input("\n선택 (1 또는 2): ").strip()
        if model_type not in ["1", "2"]:
            print("❌ 잘못된 입력입니다. 1 또는 2를 입력해주세요.")

    # 2. 모델 크기 선택
    print("\n어떤 모델로 선택하시겠습니까?")
    print("- 32B: 32B 모델")
    print("- 1.2B: 1.2B 모델")

    model_size = None
    while model_size not in ["32B", "32b", "1.2B", "1.2b"]:
        model_size = input("\n선택 (32B 또는 1.2B): ").strip()
        if model_size not in ["32B", "32b", "1.2B", "1.2b"]:
            print("❌ 잘못된 입력입니다. 32B 또는 1.2B를 입력해주세요.")

    # 정규화
    model_size = model_size.lower()

    # 3. Pruning 비율 선택 (Pruning 모델인 경우)
    pruning_percent = None
    if model_type == "2":
        print("\nPruning 비율을 선택해주세요 (예: 30, 40, 50)")
        while True:
            try:
                pruning_input = input("Pruning 비율 (숫자만 입력): ").strip()
                pruning_percent = int(pruning_input)
                if pruning_percent < 0 or pruning_percent > 100:
                    print("❌ 0~100 사이의 값을 입력해주세요.")
                    continue
                break
            except ValueError:
                print("❌ 숫자를 입력해주세요.")

    # 4. 경로 결정
    if model_type == "1":
        # Quantized만
        if model_size == "1.2b":
            model_path = "./quantized_models/exaone-1.2b-int4"
            model_name = "EXAONE-1.2B-INT4"
        else:  # 32b
            model_path = "./quantized_models/exaone-32b-int4"
            model_name = "EXAONE-32B-INT4"
    else:
        # Pruning + Quantized
        if model_size == "1.2b":
            model_path = f"./quantized_models/exaone-1.2b-pruned-int4-{pruning_percent}"
            model_name = f"EXAONE-1.2B-Pruned-INT4-{pruning_percent}%"
        else:  # 32b
            model_path = f"./quantized_models/exaone-32b-pruned-int4-{pruning_percent}"
            model_name = f"EXAONE-32B-Pruned-INT4-{pruning_percent}%"

    print(f"\n✅ 선택된 모델: {model_name}")
    print(f"✅ 모델 경로: {model_path}")

    return model_path, model_name, model_size


def main():
    import argparse
    p = argparse.ArgumentParser(description="EXAONE full evaluation")
    p.add_argument("--auto", action="store_true", help="자동 모드 (기존 동작)")
    args = p.parse_args()

    # 설정
    cfg_12b = dict(
        mmlu_samples=3000,
        gpqa_samples=None,
        ifeval_samples=300,
        lcb_samples=100,
        aime_samples=100,
        helmet_samples=30,
        longbench_samples=100,
        bfcl_samples=100,
        ksm_samples=200,
    )

    cfg_32b = dict(
        mmlu_samples=1500,
        gpqa_samples=None,
        ifeval_samples=150,
        lcb_samples=50,
        aime_samples=50,
        helmet_samples=20,
        longbench_samples=50,
        bfcl_samples=50,
        ksm_samples=100,
    )

    if args.auto:
        # 기존 자동 모드 (--auto 플래그 사용 시)
        print("[자동 모드] 기본 설정으로 평가를 시작합니다...")
        scores_12b = run_one_model("./quantized_models/exaone-1.2b-int4", "EXAONE-1.2B-INT4", **cfg_12b)
        scores_32b = run_one_model("./quantized_models/exaone-32b-int4", "EXAONE-32B-INT4", **cfg_32b)
        print_final_table(scores_12b, scores_32b)
    else:
        # 사용자 입력 모드 (기본)
        model_path, model_name, model_size = get_user_model_choice()

        # 모델 크기에 따라 설정 선택
        cfg = cfg_12b if model_size == "1.2b" else cfg_32b

        # 평가 실행
        print(f"\n평가를 시작합니다...")
        scores = run_one_model(model_path, model_name, **cfg)

        # 결과 출력 (단일 모델)
        print(f"\n{model_name} 평가 결과:")
        print_final_table(scores if model_size == "1.2b" else {}, scores if model_size == "32b" else {})

    print("\n✅ 완료")


if __name__ == "__main__":
    main()
