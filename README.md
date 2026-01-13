LG AImers 8기 해커톤 - EXAONE 4.0 Quantization
📌 프로젝트 개요
주제: EXAONE 4.0 모델 INT4 양자화 (Weight: INT4, Activation: FP16)

목표: LG AI Research에서 공개한 EXAONE 4.0 모델(1.2B, 32B)을 INT4로 양자화하여 메모리 효율성을 극대화하면서도 성능을 유지하는 것

작성자: 신민석

작성일: 2026.01.13

대회: LG AImers 8기 해커톤

🎯 프로젝트 목표
메모리 효율화: 32B 파라미터 모델을 제한된 GPU 환경(1080 Ti, 11GB VRAM)에서 실행 가능하도록 양자화
성능 유지: INT4 양자화 후에도 원본 모델의 성능을 최대한 유지
실용성: 로컬 환경에서 실행 가능한 경량화 모델 제공
🖥️ 개발 환경
하드웨어
GPU: NVIDIA GeForce GTX 1080 Ti (11GB VRAM)
CPU: 고성능 멀티코어 프로세서
RAM: 32GB 이상
Storage: SSD 100GB+ 여유 공간
소프트웨어
OS: Windows 11
Python: 3.10
CUDA: 11.3 (Driver: 13.0)
주요 라이브러리:
PyTorch 2.x (CUDA 11.8)
Transformers 4.40+
BitsAndBytes 0.43.2+
Accelerate 0.28+
🔬 양자화 방법론
양자화 전략
1. Weight: INT4 (NF4)
NormalFloat 4-bit (NF4) 사용
가중치를 4-bit로 압축하여 메모리 사용량 75% 감소
Double Quantization 적용으로 추가 압축
2. Activation: FP16
활성화 함수는 FP16 (Float16) 유지
계산 정확도와 속도의 균형
3. CPU Offloading
GPU 메모리 부족 시 일부 레이어를 CPU로 offload
llm_int8_enable_fp32_cpu_offload=True 설정
device_map="auto" 사용하여 자동 분산
기술적 세부사항

BitsAndBytesConfig(
    load_in_4bit=True,                      # 4-bit 양자화
    bnb_4bit_compute_dtype=torch.float16,   # FP16 계산
    bnb_4bit_use_double_quant=True,         # 이중 양자화
    bnb_4bit_quant_type="nf4",              # NormalFloat 4-bit
    llm_int8_enable_fp32_cpu_offload=True   # CPU offload 허용
)
📊 양자화 결과
EXAONE 1.2B 모델
항목	원본	양자화 (INT4)	감소율
모델 크기	~2.5GB	~0.9GB	64%
GPU 메모리	~2.5GB	~1.0GB	60%
파라미터 수	1.2B	1.2B	-
추론 속도	기준	유사	-
레이어 구성:

FP16 레이어: 122개
INT4 레이어 (uint8로 저장): 210개
특징:

전체 모델이 GPU에 로드됨 (CPU offload 불필요)
빠른 추론 속도 유지
약 5-10분 만에 양자화 완료
EXAONE 32B 모델
항목	원본	양자화 (INT4)	감소율
모델 크기	~64GB	~16GB	75%
GPU 메모리	~64GB	~8-10GB	85%
파라미터 수	32B	32B	-
추론 속도	기준	느림 (CPU offload)	-
메모리 분산 (예상):

GPU 레이어: ~50-55개
CPU 레이어: ~10-15개
총 레이어: ~64개
특징:

GPU 메모리 부족으로 일부 레이어가 CPU로 offload
CPU offload로 인한 추론 속도 저하 (약 2-5배)
양자화 시간: 20-40분 (첫 실행 시 다운로드 포함)
🏗️ 프로젝트 구조

hackathon/
├── .env                              # API 키 (보안)
├── .gitignore                        # Git 제외 파일
├── requirements.txt                  # 의존성 패키지
│
├── exaone_quantization.py            # 1.2B 모델 양자화
├── exaone_32B_quantization.py        # 32B 모델 양자화
├── exaone_evaluation.py              # 모델 평가
│
├── api_token_test.py                 # API 토큰 테스트
├── test_imports.py                   # Import 테스트
│
├── quantized_models/                 # 양자화된 모델 저장
│   ├── exaone-1.2b-int4/            # 1.2B INT4 모델
│   └── exaone-32b-int4/             # 32B INT4 모델
│
└── README.md                         # 프로젝트 문서
🚀 실행 방법
1. 환경 설정

# 가상환경 생성
python -m venv exaone32b

# 가상환경 활성화 (Windows)
exaone32b\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
2. API 키 설정
.env 파일 생성:


HF_TOKEN=your_huggingface_token_here
FRIENDLI_API_KEY=your_friendli_api_key_here
3. 양자화 실행

# 1.2B 모델 양자화 (빠름, 5-10분)
python exaone_quantization.py

# 32B 모델 양자화 (느림, 20-40분)
python exaone_32B_quantization.py
4. 모델 평가 (예정)

# 양자화된 모델 성능 평가
python exaone_evaluation.py
📈 성능 평가 계획
평가 벤치마크
1. MMLU (Multi-task Language Understanding)
57개 과목의 4지선다 문제
Zero-shot / 5-shot 평가
전체 정확도 및 분야별 정확도 측정
2. GPQA (Graduate-Level Science QA)
물리/화학/생물 대학원 수준 문제
고난도 벤치마크
과목별 정확도 분석
3. IFEval (Instruction Following Evaluation)
검증 가능한 지시 준수 능력 평가
25종류의 instruction 유형
준수율(compliance rate) 측정
4. LiveCodeBench (Coding)
실제 코딩 문제 해결 능력
pass@1 지표
Self-repair 능력 평가
평가 지표
모델	MMLU	GPQA	IFEval	Code
EXAONE 1.2B (원본)	TBD	TBD	TBD	TBD
EXAONE 1.2B (INT4)	TBD	TBD	TBD	TBD
EXAONE 32B (원본)	TBD	TBD	TBD	TBD
EXAONE 32B (INT4)	TBD	TBD	TBD	TBD
💡 핵심 기술
1. BitsAndBytes 양자화
Meta의 BitsAndBytes 라이브러리 사용
NF4 (NormalFloat 4-bit) 양자화 방식
Double Quantization으로 추가 압축
2. CPU Offloading
Accelerate 라이브러리의 device_map 기능
GPU 메모리 부족 시 자동으로 CPU로 레이어 이동
유연한 메모리 관리
3. 메모리 최적화
low_cpu_mem_usage=True: CPU 메모리 사용 최소화
max_memory 설정: GPU/CPU 메모리 할당 제어
offload_folder: Disk 기반 offload 지원
🎓 학습 내용
양자화의 이해
Weight Quantization: 모델 가중치 압축
Activation Quantization: 활성화 함수 정밀도 제어
Trade-off: 메모리 vs 성능
대규모 모델 처리
Device Mapping: 레이어 단위 디바이스 분산
CPU Offloading: GPU 메모리 한계 극복
메모리 관리: 효율적인 리소스 사용
실무 기술
Hugging Face 생태계 활용
모델 평가 및 벤치마킹
재현 가능한 실험 설계
🔧 트러블슈팅
주요 해결 문제
PyTorch 버전 호환성

문제: Transformers 4.54+는 PyTorch 2.1+ 필요
해결: PyTorch 2.x로 업그레이드
BitsAndBytes CPU Offload

문제: llm_int8_enable_fp32_cpu_offload 누락
해결: BitsAndBytesConfig에 옵션 추가
Meta 텐서 오류

문제: low_cpu_mem_usage=False 사용 시 오류
해결: low_cpu_mem_usage=True로 변경
NumPy 컴파일 오류

문제: C 컴파일러 없음
해결: 미리 컴파일된 바이너리 사용
📚 참고 자료
공식 문서
EXAONE 4.0 Model Card
EXAONE 4.0 Paper
BitsAndBytes Documentation
Transformers Documentation
기술 블로그
LG AI Research Blog
Hugging Face Blog - Quantization
BitsAndBytes GitHub
🏆 기대 효과
1. 메모리 효율성
75% 메모리 절약으로 제한된 하드웨어에서 대규모 모델 실행 가능
2. 비용 절감
고가의 GPU 없이도 32B 모델 실행
클라우드 비용 절감
3. 접근성 향상
일반 소비자용 GPU(1080 Ti)로 대규모 LLM 실행
연구 및 개발 진입 장벽 낮춤
4. 실용성
로컬 환경에서 프라이버시 보호
빠른 프로토타이핑 가능
🔮 향후 계획
성능 평가 완료: MMLU, GPQA, IFEval, LiveCodeBench 실행
최적화: 더 나은 양자화 기법 탐색
비교 분석: 원본 vs 양자화 모델 상세 비교
문서화: 재현 가능한 실험 결과 정리
배포: 양자화된 모델 공유 (라이선스 허용 시)
📧 연락처
작성자: 신민석

소속: LG AImers 8기

GitHub: https://www.github.com/skytinstone

이메일: stevenshin16@gmail.com

📄 라이선스
이 프로젝트는 LG AI Research의 EXAONE AI Model License Agreement 1.2 - NC를 따릅니다.

Last Updated: 2026.01.13
