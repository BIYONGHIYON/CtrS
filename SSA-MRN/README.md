# SSA-MRN 원 논문 재현

> 연구 기준: Xu et al., “Spectral–Spatial Attention-Guided Multi-Resolution Network for Pansharpening,” IEEE JSTARS, 2025.
> 논문: https://doi.org/10.1109/JSTARS.2025.3543827
> 공식 저장소: https://github.com/zhouchuanxu/SSA-MRN

## 현재 연구 단계

SSA-MRN 담당 팀원 2명이 **원 논문의 pansharpening 결과 재현**을 진행합니다. 동시에 다른 팀원 2명은 ECRformer 원 논문을 재현합니다. 공식 저장소의 네트워크 구현을 확인하고, 논문에 공개된 학습 조건·데이터 처리·평가 절차를 재구성합니다. RGB–HSI 확장은 원 논문 재현 결과를 확인한 뒤 검토합니다.

원 문제의 입력과 출력은 다음과 같습니다.

- 입력: 고해상도 PAN, 저해상도 PAN(LRPAN), 저해상도 MS
- 출력: 고해상도 MS(HRMS)
- 축척 비율: 4
- 주요 구성: 다중 해상도 SSAI와 점진적 특징 융합

## 공식 코드에서 확인한 범위

공식 GitHub 저장소에는 `network.py`와 README가 공개되어 있습니다. README는 PanCollection 사용을 안내하지만, 저장소에는 논문 전체 재현에 필요한 학습 스크립트, 데이터 로더와 설정, 평가 코드, 학습 가중치가 포함되어 있지 않습니다. 따라서 공개 네트워크를 출발점으로 누락된 학습·평가 파이프라인을 별도로 구현하고, 원 논문과의 차이를 기록합니다.

논문 식 (10)–(12)의 `W = Softmax(C_pan C_ms^T)`는 입력 특징에서 계산하는 **어텐션 가중치**이며, 저장된 학습 파라미터(체크포인트)가 아닙니다. 논문은 SSAI의 차원 `K=6`을 명시하지만, 공개 `network.py`의 `SSA.fixed`는 `4`이고 어텐션도 행렬곱 대신 원소별 곱으로 계산합니다. 현재 복구 모델은 공개 코드를 보존한 실행 경로이므로 이 불일치를 그대로 가지며, 논문과 동일한 학습 결과를 재현했다고 볼 수 없습니다.

## 논문 설정

| 항목 | 논문에 기재된 값 |
|---|---|
| 데이터 | PanCollection: QB, GF2, WV3; 교차 위성 시험은 WV3 학습 → WV2 평가 |
| 학습 표본 | QB 19,044쌍, GF2 22,010쌍, WV3 10,794쌍 |
| 해상도 비율 | 4× |
| 손실 | MSE (L2) |
| 최적화 | Adam |
| 배치 크기 | 32 |
| 초기 학습률 | 1 × 10⁻⁴ |
| 학습 길이 | 100 epochs |
| SSAI 내부 차원 K | 6 |
| 논문 실험 GPU | NVIDIA GeForce RTX 2080 Ti |

축소 해상도(RR) 평가는 Wald protocol을 따르며, 참조 지표로 SAM, ERGAS, PSNR, SCC, Q2ⁿ을 사용합니다. 원 해상도(FR) 평가는 참조 영상이 없는 QNR, Dλ, Ds를 사용합니다. 먼저 논문 설정을 고정해 재현하고, 메모리 한계 등으로 설정을 바꿀 때는 논문 값과 실제 실행 값을 분리해 기록합니다.

## 재현 순서

1. `docs/reproduction_status.md`에서 논문·공식 저장소와 구현 누락 항목을 확인합니다.
2. PanCollection 파일을 내려받아 `data/raw/`에 두고, 원본 파일은 수정하지 않습니다.
3. 센서별 입력 채널 수, 데이터 분할, Wald 열화 방식과 배열 범위를 확인합니다.
4. 모델 입력/출력 shape 및 단일 배치 forward 검증을 구현합니다.
5. 손실 감소, 체크포인트 저장·재시작을 포함한 짧은 실행으로 파이프라인을 점검합니다.
6. 논문 설정으로 QB, GF2, WV3 학습을 진행하고 RR 및 FR 평가를 수행합니다.
7. WV3로 학습한 모델의 WV2 교차 위성 평가와 논문 ablation을 추가합니다.
8. 로그, 설정, 코드 revision, 지표와 실행 환경을 함께 `experiments/`에 보관합니다.

## 저장소 사용

```text
SSA-MRN/
├── configs/                 # 논문 재현 설정
├── data/
│   ├── raw/                 # QuickBird 테스트 H5만 Git 포함
│   └── processed/           # 준비된 데이터 (Git 제외)
├── docs/                    # 재현 계획과 실험 기록
├── experiments/
│   ├── checkpoints/         # 모델 가중치 (Git 제외)
│   ├── logs/                # 학습 로그 (Git 제외)
│   └── results/             # QuickBird 스모크 결과만 Git 포함
├── references/              # 공식 코드 위치·버전 기록
├── scripts/                 # 학습·평가 실행 진입점
├── src/ssamrn/              # 재현용 모델·데이터·평가 모듈
└── tests/                   # shape와 데이터 처리 점검
```

### 환경 준비

PyTorch는 사용 장비의 운영체제·CUDA에 맞는 버전을 먼저 설치한 뒤 나머지 의존성을 설치합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -r requirements.txt
```

설치한 PyTorch/CUDA 버전과 GPU 모델은 실행 기록에 남깁니다. 서로 다른 CUDA 환경에서 같은 설치 명령이 통한다고 가정하지 않습니다.

### QuickBird 실행 확인

```bash
python scripts/smoke_test.py --size 32
python scripts/smoke_test.py --size 256
python -m unittest discover -s tests -v
```

`--size 32`는 첫 테스트 샘플 전체의 PAN·LMS 256×256과 MS 64×64를 각각 32×32와 8×8로 축소합니다. `--size 256`은 전체 샘플을 그대로 사용합니다. 입력 MS 4밴드의 RGB 합성·입력 PAN·출력 MS RGB 합성은 [QuickBird 스모크 결과](./experiments/results/quickbird_smoke/README.md)에서 나란히 볼 수 있습니다. RGB 합성에는 QuickBird B·G·R·NIR 순서가 H5에서도 유지됐다고 가정하며, 각 이미지는 보기용으로 대비를 별도 조정했습니다. 학습된 가중치가 없는 무작위 초기화 결과이므로 화질 평가는 할 수 없습니다.

## 코드와 데이터 원칙

- 공식 `network.py`를 기준 구현으로 보존하고, 변경분은 재현 코드와 분리해 추적합니다.
- 논문에 명시되지 않은 값은 임의로 논문 설정인 것처럼 기재하지 않고 `미확인`으로 표시합니다.
- `data/raw/QuickBird/test_qb_multiExm1.h5`와 `experiments/results/quickbird_smoke/`만 Git에 포함합니다. 다른 원본 데이터, 체크포인트, 로그는 추가하지 않습니다.
- 재현 완료는 코드 실행 성공과 구분합니다. 논문 수치와의 비교가 끝나기 전에는 “논문 재현 완료”로 표현하지 않습니다.
- `scripts/visualize_arad_hsi.py`는 기존 HSI 시각화 유틸리티로 보존하며 원 논문 pansharpening 재현의 일부로 사용하지 않습니다.

## 이후 확장

원 논문 재현 결과와 누락 설정을 정리한 뒤 ECRformer 팀의 재현 결과와 비교합니다. 중간 시점에 최종 주제를 선택하고, 선택된 연구에 팀원 4명이 함께 참여합니다. SSA-MRN이 선택되면 RGB–HSI 초해상도 확장을 별도 실험으로 검토합니다.
