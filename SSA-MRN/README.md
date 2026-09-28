# SSA-MRN 원 논문 재현 결과

> `experiment` 브랜치의 Radeon 780M 로컬 K=6 재학습 절차는 [실험 가이드](./docs/experiment_k6.md)를 참조하세요.

> 연구 기준: Xu et al., “Spectral–Spatial Attention-Guided Multi-Resolution Network for Pansharpening,” IEEE JSTARS, 2025.
> 논문: https://doi.org/10.1109/JSTARS.2025.3543827
> 공식 저장소: https://github.com/zhouchuanxu/SSA-MRN

## 현재 연구 단계

SSA-MRN 담당 팀원 2명의 **공개 코드 기반 재현 실험은 완료**했습니다. 누락된 학습·평가 파이프라인을 복구하고, QB·GF2·WV3를 각각 100 epochs 학습한 뒤 RR·FR 테스트와 WV3→WV2 교차 위성 평가까지 수행했습니다. 여기서 ‘완료’는 계획한 실험과 논문 수치 비교를 마쳤다는 뜻이며, 논문 모델·평가 설정과 완전히 일치한다는 뜻은 아닙니다. ECRformer 팀은 별도로 재현을 진행합니다.

원 문제의 입력과 출력은 다음과 같습니다.

- 입력: 고해상도 PAN, 저해상도 PAN(LRPAN), 저해상도 MS
- 출력: 고해상도 MS(HRMS)
- 축척 비율: 4
- 주요 구성: 다중 해상도 SSAI와 점진적 특징 융합

## 공식 코드에서 확인한 범위

공식 GitHub 저장소에는 `network.py`와 README가 공개되어 있습니다. README는 PanCollection 사용을 안내하지만, 저장소에는 논문 전체 재현에 필요한 학습 스크립트, 데이터 로더와 설정, 평가 코드, 학습 가중치가 포함되어 있지 않습니다. 따라서 공개 네트워크를 출발점으로 누락된 학습·평가 파이프라인을 별도로 구현하고, 원 논문과의 차이를 기록합니다.

논문 식 (10)–(12)의 `W = Softmax(C_pan C_ms^T)`는 입력 특징에서 계산하는 **어텐션 가중치**이며, 저장된 학습 파라미터(체크포인트)가 아닙니다. 논문은 SSAI의 차원 `K=6`을 명시하지만, 공개 `network.py`의 `SSA.fixed`는 `4`이고 어텐션도 행렬곱 대신 원소별 곱으로 계산합니다. 현재 복구 모델은 공개 코드를 보존한 실행 경로이므로 이 불일치를 그대로 가집니다.

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

축소 해상도(RR) 평가는 Wald protocol을 따르며, 참조 지표로 SAM, ERGAS, PSNR, SCC, Q2ⁿ을 사용합니다. 원 해상도(FR) 평가는 참조 영상이 없는 QNR, Dλ, Ds를 사용합니다. 논문 값과 실제 실행 값은 구분해 기록했습니다.

## 재현 결과: 논문과 정량 비교

QB·GF2·WV3의 100-epoch `latest.pt`로 각각 테스트하고, WV2에는 WV3 가중치를 적용했습니다. PanCollection RR·FR 테스트를 센서별 20장씩 총 160장 평가했으며, 표의 현재 값은 이미지별 지표의 평균입니다. 왼쪽은 논문 Tables I–IV의 *Ours*, 오른쪽은 현재 공개 코드 기반 복구 모델의 결과입니다.

| 센서 | SAM↓ 논문/현재 | ERGAS↓ 논문/현재 | PSNR↑ 논문/현재 | SCC↑ 논문/현재 | Q4·Q8↑ 논문/현재 |
|---|---:|---:|---:|---:|---:|
| QB | 4.8478 / 4.9557 | 4.0726 / 4.1555 | 37.6645 / 37.3608 | .9702 / .9765 | .9243 / .9214 |
| GF2 | .9434 / .9668 | .8755 / .9125 | 46.9734 / 46.3520 | .9898 / .9816 | .9688 / .9661 |
| WV3 | 3.4873 / 3.5277 | 2.5866 / 2.5822 | 37.5691 / 37.3894 | .9735 / .9779 | .8930 / .8753 |
| WV2 | 5.8845 / 5.9265 | 4.7254 / 4.8002 | 29.4148 / 29.0611 | .9217 / .8959 | .8215 / .8190 |

| 센서 | Dλ↓ 논문/현재 | Ds↓ 논문/현재 | QNR↑ 논문/현재 |
|---|---:|---:|---:|
| QB | .0341 / .0479 | .0360 / .0396 | .9311 / .9147 |
| GF2 | .0395 / .0350 | .0487 / .0565 | .9137 / .9106 |
| WV3 | .0329 / .0221 | .0617 / .0382 | .9077 / .9408 |
| WV2 | .0657 / .0511 | .0549 / .0382 | .8831 / .9126 |

**해석:** RR의 SAM·ERGAS와 PSNR은 대체로 논문 수치에 가깝지만 완전히 같지는 않고, WV3 Q8·WV2 SCC와 일부 FR 지표에는 차이가 남습니다. 특히 PSNR은 논문이 계산 세부 설정을 공개하지 않아 확인한 방식 중 논문 수치에 가장 가까운 전 센서 peak 2047·밴드별 평균으로 고정했습니다. 이는 성능 향상이 아니라 평가식 선택입니다. FR의 PAN 축소·보간은 MATLAB 평가 코드와 동일성이 검증되지 않아 QNR 등의 비교는 잠정적입니다. 공개 코드의 SSAI `K=4`와 논문의 `K=6`도 다릅니다. 따라서 **공개 코드 기반 재현 실험 완료**이며 **논문 구현·성능의 완전한 일치까지 입증한 것은 아닙니다**. 계산 조건과 실행 방법은 [상세 비교 문서](./docs/paper_metric_comparison.md)에 기록했습니다.

```bash
python scripts/evaluate_paper.py --sensor all --protocol both
```

## 재현 순서

1. `docs/reproduction_status.md`에서 논문·공식 저장소와 구현 누락 항목을 확인합니다.
2. PanCollection 파일을 내려받아 `data/raw/`에 두고, 원본 파일은 수정하지 않습니다.
3. 센서별 입력 채널 수, 데이터 분할, Wald 열화 방식과 배열 범위를 확인합니다.
4. 모델 입력/출력 shape 및 단일 배치 forward 검증을 구현합니다.
5. 손실 감소, 체크포인트 저장·재시작을 포함한 짧은 실행으로 파이프라인을 점검합니다.
6. QB·GF2·WV3의 100-epoch 학습과 RR·FR 각 20장 평가를 완료했습니다.
7. WV3 모델의 WV2 교차 위성 평가 및 논문 수치 비교를 완료했습니다. 논문 ablation 재현은 이번 범위에 포함하지 않았습니다.
8. 학습 로그·가중치는 `experiments/`에, 전체 평가 결과와 한계는 [비교 문서](./docs/paper_metric_comparison.md)에 기록했습니다.

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

위 이미지는 각 20개 중 **첫 1개 샘플 시각화**입니다. 바로 위의 단일 샘플 PSNR은 이 문서의 20장 평균 PSNR과 계산 조건·평가 범위가 달라 직접 비교하지 마세요. 전체 논문 지표 비교는 [재현 결과](#재현-결과-논문과-정량-비교)에 있습니다.

### 다른 사람이 테스트 재실행하기

`git clone --recurse-submodules` 후 위 환경을 설치합니다. **테스트 H5는 Git에 포함되지 않습니다.** [PanCollection 공개 다운로드 안내](https://github.com/liangjiandeng/PanCollection)의 각 센서 **Testing Dataset (ReducedData, H5 Format)**에서 받아 아래 경로에 둡니다. FullData H5에는 GT가 없어 이 스크립트와 대상이 다릅니다.

| 센서 | 놓을 위치 | 원본 SHA-256 |
|---|---|---|
| [QB](https://drive.google.com/drive/folders/1g4kB3Yxmn6Y8_OCqE1Mra1GoUKCmHOUm?usp=sharing) | `data/raw/QuickBird/test_qb_multiExm1.h5` | `9842a1232ad2d5b9a61c9fd7fb353e8fc6214eeac6e947d888f221ad6ecdda35` |
| [GF2](https://drive.google.com/drive/folders/1g4f2NElV7By2gWhCavrDaglzCxiDT6CP?usp=sharing) | `data/raw/Gaofen2/test_gf2_multiExm1.h5` | `709a9a53b2e0f29d3dcd5c6ca4410c2913b62cc03c104fed6c049911dbe1c8ea` |
| [WV3](https://drive.google.com/drive/folders/1EYjaAxTheNPvukvifKXMq8m_dJ-8qz8G?usp=sharing) | `data/raw/WorldView3/test_wv3_multiExm1.h5` | `00db0d62f63693410935208e287aea21a736d11840d92eedd2d093c99be5314a` |

전체 논문 수치 평가에는 위 RR 파일에 더해 `data/raw/WorldView2/test_wv2_multiExm1.h5`가 필요합니다. FR 평가에는 각 센서 폴더에 `test_qb_OrigScale_multiExm1.h5`, `test_gf2_OrigScale_multiExm1.h5`, `test_wv3_OrigScale_multiExm1.h5`, `test_wv2_OrigScale_multiExm1.h5`를 각각 둡니다. 모두 [PanCollection](https://github.com/liangjiandeng/PanCollection)의 해당 센서 FullData H5에서 받으며 Git에는 포함되지 않습니다. 파일을 배치한 뒤 `python scripts/evaluate_paper.py --sensor all --protocol both`로 표의 수치를 재계산할 수 있습니다.

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
- 공개 코드 기반 재현 실험은 학습·평가·논문 수치 비교까지 완료했습니다. 미확인 평가 설정과 모델 구조 차이는 결과 해석에 유지합니다.
- `scripts/visualize_arad_hsi.py`는 기존 HSI 시각화 유틸리티로 보존하며 원 논문 pansharpening 재현의 일부로 사용하지 않습니다.

## 이후 확장

원 논문 재현 결과와 누락 설정을 정리한 뒤 ECRformer 팀의 재현 결과와 비교합니다. 중간 시점에 최종 주제를 선택하고, 선택된 연구에 팀원 4명이 함께 참여합니다. SSA-MRN이 선택되면 RGB–HSI 초해상도 확장을 별도 실험으로 검토합니다.
