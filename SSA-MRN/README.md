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
6. QB, GF2, WV3의 100-epoch 학습을 완료했습니다. RR 단일 샘플 실행을 확인했으며 전체 RR·FR 평가는 진행 예정입니다.
7. WV3로 학습한 모델의 WV2 교차 위성 평가와 논문 ablation을 추가합니다.
8. 로그, 설정, 코드 revision, 지표와 실행 환경을 함께 `experiments/`에 보관합니다.

## 저장소 사용

```text
SSA-MRN/
├── configs/                 # 논문 재현 설정
├── data/
│   ├── raw/                 # 원본 H5 전부 Git 제외 (별도 다운로드)
│   └── processed/           # 준비된 데이터 (Git 제외)
├── docs/                    # 재현 계획과 실험 기록
├── experiments/
│   ├── checkpoints/         # QB·GF2·WV3 최신 100-epoch 가중치 Git 포함
│   ├── logs/                # 세 센서 학습 로그 Git 포함
│   └── results/             # 단일 샘플 비교 이미지·수치 Git 포함
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

### 3개 센서 학습·단일 샘플 테스트 현황 (2026-09-28)

QB·GF2·WV3를 서버의 RTX A6000에서 각각 100 epochs 학습했습니다. 사용 코드는 공개 `network.py`를 기반으로 한 현재 복구 구현이며, Adam·MSE·배치 32·학습률 1e-4·시드 42를 사용했습니다. 마지막 에포크 가중치인 `latest.pt`를 Git에 포함했습니다(`best` 체크포인트가 아님). 모든 테스트는 Mac CPU에서 별도 ReducedData H5의 첫 샘플(0번)을 `strict=True`로 불러와 수행했습니다.

| 센서 | 채널 | epoch 100 검증 MSE | 테스트 첫 샘플 LMS PSNR | 모델 PSNR | 결과 |
|---|---:|---:|---:|---:|---|
| QB | 4 | 0.00017375 | 35.45 dB | 41.35 dB | [이미지·수치](./experiments/results/quickbird_trained/metrics.json) |
| GF2 | 4 | 0.00008431 | 31.26 dB | 38.65 dB | [이미지·수치](./experiments/results/gaofen2_trained/metrics.json) |
| WV3 | 8 | 0.00035871 | 29.06 dB | 37.66 dB | [이미지·수치](./experiments/results/worldview3_trained/metrics.json) |

![QuickBird 비교](./experiments/results/quickbird_trained/comparison.png)
![Gaofen 2 비교](./experiments/results/gaofen2_trained/comparison.png)
![WorldView 3 비교](./experiments/results/worldview3_trained/comparison.png)

이미지는 왼쪽부터 입력 MS·PAN·LMS 기준·모델 출력·GT입니다. 원본 MS는 64×64, PAN/LMS/GT와 출력은 256×256이며 **추론 입력을 자르거나 다운샘플링하지 않았습니다**. 그림의 MS만 보기 위해 확대했습니다. MS 계열의 RGB 대비는 GT 기반 채널별 1–99 백분위를 공유하고, RGB 밴드 순서는 GF2/QB `[2,1,0]`, WV3 `[4,2,1]`을 **가정**합니다. PSNR은 정규화된 전 밴드 MSE를 기준 최댓값 1로 환산한 값입니다. 센서 간 수치를 직접 우열 비교하지 마세요.

이는 각 20개 중 **첫 1개 샘플 실행 확인**이며, 논문의 전체 RR/FR 지표나 재현 성능 비교가 아닙니다. 앞으로 20개 전체에 대해 논문 지표(SAM·ERGAS·PSNR·SCC·Q2ⁿ 및 FR 지표)를 구현·검증하고 논문 표와 같은 조건에서 수치를 비교할 예정입니다. 공개 구현과 논문 설명의 SSA 차이(K=4/6 등)도 따로 확인해야 합니다.

### 다른 사람이 테스트 재실행하기

`git clone --recurse-submodules` 후 위 환경을 설치합니다. **테스트 H5는 Git에 포함되지 않습니다.** [PanCollection 공개 다운로드 안내](https://github.com/liangjiandeng/PanCollection)의 각 센서 **Testing Dataset (ReducedData, H5 Format)**에서 받아 아래 경로에 둡니다. FullData H5에는 GT가 없어 이 스크립트와 대상이 다릅니다.

| 센서 | 놓을 위치 | 원본 SHA-256 |
|---|---|---|
| [QB](https://drive.google.com/drive/folders/1g4kB3Yxmn6Y8_OCqE1Mra1GoUKCmHOUm?usp=sharing) | `data/raw/QuickBird/test_qb_multiExm1.h5` | `9842a1232ad2d5b9a61c9fd7fb353e8fc6214eeac6e947d888f221ad6ecdda35` |
| [GF2](https://drive.google.com/drive/folders/1g4f2NElV7By2gWhCavrDaglzCxiDT6CP?usp=sharing) | `data/raw/Gaofen2/test_gf2_multiExm1.h5` | `709a9a53b2e0f29d3dcd5c6ca4410c2913b62cc03c104fed6c049911dbe1c8ea` |
| [WV3](https://drive.google.com/drive/folders/1EYjaAxTheNPvukvifKXMq8m_dJ-8qz8G?usp=sharing) | `data/raw/WorldView3/test_wv3_multiExm1.h5` | `00db0d62f63693410935208e287aea21a736d11840d92eedd2d093c99be5314a` |

```bash
cd SSA-MRN
python scripts/test_checkpoint.py --sensor QB --checkpoint experiments/checkpoints/qb_full/latest.pt --data data/raw/QuickBird/test_qb_multiExm1.h5 --output-dir experiments/results/quickbird_trained
python scripts/test_checkpoint.py --sensor GF2 --checkpoint experiments/checkpoints/gf2_full/latest.pt --data data/raw/Gaofen2/test_gf2_multiExm1.h5 --output-dir experiments/results/gaofen2_trained
python scripts/test_checkpoint.py --sensor WV3 --checkpoint experiments/checkpoints/wv3_full/latest.pt --data data/raw/WorldView3/test_wv3_multiExm1.h5 --output-dir experiments/results/worldview3_trained
```

각 결과 폴더에 `comparison.png`, 개별 입력/출력 PNG, 전 밴드 `prediction_bands.npy`, `metrics.json`이 생성됩니다. 재실행하면 공유된 결과 파일을 덮어쓰므로 별도 `--output-dir`을 쓰면 안전합니다. 각 센서 학습 기록은 `experiments/logs/`, 가중치는 `experiments/checkpoints/`에 있습니다. 대용량 `train_*.h5`/`valid_*.h5`는 공유하지 않아 **학습 자체를 똑같이 재실행하려면 별도 다운로드**가 필요합니다.

## 코드와 데이터 원칙

- 공식 `network.py`를 기준 구현으로 보존하고, 변경분은 재현 코드와 분리해 추적합니다.
- 논문에 명시되지 않은 값은 임의로 논문 설정인 것처럼 기재하지 않고 `미확인`으로 표시합니다.
- 원본 학습·검증·테스트 H5는 **모두 Git에서 제외**합니다. 코드, 세 센서의 마지막 가중치·학습 로그, 실행 결과 이미지만 공유합니다. QuickBird H5도 이번 변경부터 추적 해제합니다(과거 커밋 이력에는 남아 있으므로 완전 삭제가 필요한 경우 이력 정리가 별도 필요합니다).
- 재현 완료는 코드 실행 성공과 구분합니다. 논문 수치와의 비교가 끝나기 전에는 “논문 재현 완료”로 표현하지 않습니다.
- `scripts/visualize_arad_hsi.py`는 기존 HSI 시각화 유틸리티로 보존하며 원 논문 pansharpening 재현의 일부로 사용하지 않습니다.

## 이후 확장

원 논문 재현 결과와 누락 설정을 정리한 뒤 ECRformer 팀의 재현 결과와 비교합니다. 중간 시점에 최종 주제를 선택하고, 선택된 연구에 팀원 4명이 함께 참여합니다. SSA-MRN이 선택되면 RGB–HSI 초해상도 확장을 별도 실험으로 검토합니다.
