# LG AImers 8기 해커톤 - EXAONE 4.0 경량화

## 프로젝트 개요
- **주제**: EXAONE 4.0 모델 경량화 (Pruning + Quantization)
  *(Pruning → Fine-tuning → INT4 Quantization)*
- **목표**: LG AI Research에서 공개한 EXAONE 4.0 모델(1.2B, 32B)을 경량화하여
  모델 크기와 메모리 효율성을 극대화하면서도 성능을 최대한 유지
- **작성자**: 신민석
- **작성일**: 2026.01.13 (업데이트: 2026.01.15)
- **대회**: LG AImers 8기 해커톤  

---

## 프로젝트 목표
- **메모리 효율화**  
  - 32B 파라미터 모델을 제한된 GPU 환경(1080 Ti, 11GB VRAM)에서 실행 가능하도록 양자화
- **성능 유지**  
  - INT4 양자화 이후에도 원본 모델 성능 최대한 유지
- **실용성**  
  - 로컬 환경에서 실행 가능한 경량화 LLM 제공

---

## 개발 환경

### 하드웨어(Hardware)
- **GPU**: NVIDIA GeForce GTX 1080 Ti (11GB VRAM)
- **CPU**: 고성능 멀티코어 프로세서
- **RAM**: 32GB 이상
- **Storage**: SSD 100GB 이상 여유 공간

### 소프트웨어(Software)
- **OS**: Windows 11
- **Python**: 3.10
- **CUDA**: 11.3 (Driver: 13.0)

### 주요 라이브러리(Depedencies)
- PyTorch 2.x (CUDA 11.8)
- Transformers 4.40+
- BitsAndBytes 0.43.2+
- Accelerate 0.28+

---

## 경량화 방법론

### 전체 파이프라인
```
원본 모델 → Pruning (구조 경량화) → Fine-tuning (성능 회복) → Quantization INT4 (메모리 경량화) → 평가
```

### 1. Pruning (가지치기)
- **방법**: Magnitude-based Pruning
  - L1 norm 기반 가중치 중요도 계산
  - 중요도가 낮은 가중치를 0으로 설정 (제거)
- **비율**: 30% (조절 가능)
- **대상**: Linear 레이어
- **효과**: 모델 구조 경량화, Sparsity 증가

### 2. Fine-tuning (성능 회복)
- **목적**: Pruning으로 인한 성능 손실 최소화
- **데이터**: WikiText-2 소규모 데이터셋
- **방법**: 매우 가벼운 Fine-tuning (1 epoch)
- **최적화**: Gradient checkpointing, 낮은 learning rate

### 3. Quantization (양자화 전략)

#### Weight: INT4 (NF4)
- NormalFloat 4-bit (NF4) 사용
- 가중치를 4-bit로 압축하여 **메모리 사용량 약 75% 감소**
- Double Quantization 적용

#### Activation: FP16
- 활성화 함수는 FP16 유지
- 정확도와 연산 속도의 균형 유지

#### CPU Offloading
- GPU 메모리 부족 시 일부 레이어를 CPU로 offload
- `device_map="auto"` 기반 자동 분산
- `llm_int8_enable_fp32_cpu_offload=True` 설정

### 기술적 세부사항
```python
BitsAndBytesConfig(
    load_in_4bit=True,                      # 4-bit 양자화
    bnb_4bit_compute_dtype=torch.float16,   # FP16 계산
    bnb_4bit_use_double_quant=True,         # 이중 양자화
    bnb_4bit_quant_type="nf4",              # NormalFloat 4-bit
    llm_int8_enable_fp32_cpu_offload=True   # CPU offload 허용
)
양자화 결과
EXAONE 1.2B 모델
항목	원본	INT4 양자화	감소율
모델 크기	~2.5GB	~0.9GB	64%
GPU 메모리	~2.5GB	~1.0GB	60%
파라미터 수	1.2B	1.2B	-
추론 속도	기준	유사	-

레이어 구성

FP16 레이어: 122개

INT4 레이어: 210개

특징

전체 모델 GPU 로드 (CPU offload 불필요)

빠른 추론 속도 유지

양자화 시간: 약 5~10분

EXAONE 32B 모델
항목	원본	INT4 양자화	감소율
모델 크기	~64GB	~16GB	75%
GPU 메모리	~64GB	~8~10GB	85%
파라미터 수	32B	32B	-
추론 속도	기준	느림 (CPU offload)	-

메모리 분산(예상)

GPU 레이어: ~50~55

CPU 레이어: ~10~15

총 레이어 수: ~64

특징

GPU 메모리 한계로 CPU offload 필수

추론 속도 약 2~5배 저하

양자화 시간: 20~40분 (첫 실행 시 다운로드 포함)

## 프로젝트 구조
```
lg_aimers/
├── .env
├── .gitignore
├── requirements.txt
│
├── exaone_1.2B_quantization.py              # 1.2B INT4 양자화만
├── exaone_32B_quantization.py               # 32B INT4 양자화만
│
├── exaone_1.2b_pruning_quantization.py      # 1.2B Pruning + Quantization (NEW)
├── exaone_32b_pruning_quantization.py       # 32B Pruning + Quantization (NEW)
│
├── exaone_evaluation.py                     # 평가 스크립트 (예정)
│
├── pruned_models/                           # Pruning 후 모델 저장
│   ├── exaone-1.2b-pruned/
│   └── exaone-32b-pruned/
│
├── quantized_models/                        # 최종 양자화 모델 저장
│   ├── exaone-1.2b-int4/                   # 양자화만 적용
│   ├── exaone-32b-int4/                    # 양자화만 적용
│   ├── exaone-1.2b-pruned-int4/           # Pruning + Quantization
│   └── exaone-32b-pruned-int4/            # Pruning + Quantization
│
└── README.md
```
## 실행 방법

### 1. 환경 설정
```bash
python -m venv exaone32b
exaone32b\Scripts\activate
pip install -r requirements.txt
```

### 2. API 키 설정
.env 파일 생성:
```env
HF_TOKEN=your_huggingface_token_here
FRIENDLI_API_KEY=your_friendli_api_key_here
```

### 3. 경량화 실행

#### 방법 A: Quantization만 적용 (빠른 실행)
```bash
# 1.2B 모델 (5~10분)
python exaone_1.2B_quantization.py

# 32B 모델 (20~40분)
python exaone_32B_quantization.py
```

#### 방법 B: Pruning + Quantization 파이프라인 (권장)
```bash
# 1.2B 모델 (15~30분)
python exaone_1.2b_pruning_quantization.py

# 32B 모델 (60~120분)
python exaone_32b_pruning_quantization.py
```

### 4. 모델 평가 (예정)
```bash
python exaone_evaluation.py
```

성능 평가 계획
평가 벤치마크
MMLU: 다중 분야 언어 이해
GPQA: 대학원 수준 과학 QA
IFEval: 지시 준수 평가
LiveCodeBench: 코딩 성능 평가

평가 지표
모델	MMLU	GPQA	IFEval	Code
EXAONE 1.2B (원본)	TBD	TBD	TBD	TBD
EXAONE 1.2B (INT4)	TBD	TBD	TBD	TBD
EXAONE 32B (원본)	TBD	TBD	TBD	TBD
EXAONE 32B (INT4)	TBD	TBD	TBD	TBD

핵심 기술
BitsAndBytes NF4 양자화
CPU Offloading & Device Mapping
메모리 최적화 (max_memory, offload_folder)

트러블슈팅
PyTorch/Transformers 버전 충돌
BitsAndBytes CPU offload 옵션 누락
Meta Tensor 오류
NumPy / 컴파일러 문제

기대 효과
제한된 GPU에서 32B 모델 실행
비용 절감 및 접근성 향상
로컬 실행 기반 프라이버시 보호

## 향후 계획
- [ ] Pruning + Quantization 파이프라인 성능 평가
- [ ] Pruning 비율 최적화 (20%, 30%, 40% 비교)
- [ ] Quantization만 vs Pruning+Quantization 성능 비교
- [ ] vLLM 추론 엔진 적용
- [ ] 결과 문서화 및 모델 공유

## 주요 변경 사항 (2026.01.15)
- Pruning + Quantization 파이프라인 추가
- Magnitude-based Pruning 구현
- Fine-tuning 단계 추가 (성능 회복)
- 1.2B 및 32B 모델 모두 지원
- 메모리 효율적 처리 (Layer-wise, CPU offloading)

연락처(Contacts)
작성자: 신민석
소속: LG AImers 8기
GitHub: https://www.github.com/skytinstone
Email: stevenshin16@gmail.com

📄 라이선스
본 프로젝트는
LG AI Research - EXAONE AI Model License Agreement 1.2 (NC) 를 따릅니다.

Last Updated: 2026.01.15




