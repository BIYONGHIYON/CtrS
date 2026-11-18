# ADWM

| 평가자 | 설정 | 상태 |
|:---:|:---:|:---:|
| 김경찬 | `author` WV3 (저자 가중치) · `retrain` QB 200 · GF2 500에폭 (논문), seed 42 | 계획 |

**출처:** Huang 외, A General Adaptive Dual-level Weighting Mechanism for Remote Sensing Pansharpening, CVPR, 2025, [arXiv 2503.13214](https://arxiv.org/abs/2503.13214) · 공식 코드 [Jie-1203/ADWM](https://github.com/Jie-1203/ADWM) `aa185a6` · MIT(코드와 WV3 가중치가 같은 저장소, 재배포 가능) · 이슈 [#183](https://github.com/BIYONGHIYON/CtrS/issues/183)

**결과:** 아직 평가하지 않았습니다.

[평가 표](../README.md#models) · [결과](#results) · [해석](#verdict) · [방법](#method) · [그림](#figures) · [재현 자료](#evidence)

<a id="results"></a>

## 결과

아직 평가하지 않았습니다. 비교할 논문 보고값(표 1, RR test 20장)은 다음과 같습니다.

| 센서 | PSNR ↑ | SAM ↓ | ERGAS ↓ | Q ↑ | 우리 평가 설정 |
|---|---:|---:|---:|---:|---|
| WV3 | 39.170 | 2.913 | 2.145 | 0.921 (Q8) | `author` |
| QB | 38.466 | 4.450 | 3.705 | 0.937 (Q4) | `retrain` seed 42 |
| GF2 | 43.884 | 0.672 | 0.597 | 0.985 (Q4) | `retrain` seed 42 |
| WV2 | — | — | — | — | `author` WV3 가중치 전이(참고) |

GF2 PSNR은 논문마다 약 43-45 dB와 49-50 dB 계열이 함께 있어([성능조사 문서](../../docs/pansharpening_survey/QB_GF_WV2_WV3_팬샤프닝_성능조사_2026-10-27.md)) 우리 평가기 값과 직접 비교하지 않습니다.

<a id="verdict"></a>

## 해석과 정리

평가 후 작성합니다.

<a id="method"></a>

<details>
<summary><b>방법</b> · 목적과 상태 · 모델과 비교 조건</summary>

### 목적과 상태

논문의 QB RR PSNR(38.466 dB)이 C2MS-SP(3시드 평균 38.373 dB)보다 높게 보고된 모델입니다. 기존 CNN에 공분산 기반 가중(채널 내 IFW, 층 간 CFW)을 붙이는 구조라 모델이 작습니다. 같은 데이터·지표로 이 차이가 재현되는지 확인합니다.

| 설정 | 센서 | 상태 |
|---|---|---|
| `author` | WV3, WV2 전이 | 계획 |
| `retrain` seed 42 | QB, GF2 | 계획 |

### 모델

논문은 ADWM을 LAGNet에 붙인 모델을 제안 방법으로 씁니다(3.2절). 공식 코드의 `model/bwnet_lagconv_sota.py`(BWNet)입니다. 다른 적용판(FusionNet·LGPNet·DiCNN)은 평가하지 않습니다.

- **입력 → 출력:** 저해상도 MS를 bicubic으로 4배 보간해 PAN과 이어 붙입니다. 국소 적응 합성곱(LAConv) 잔차 블록 6개(48채널)를 거친 뒤, 블록 출력 6개를 공분산 기반 가중(CACW)으로 채널 내(IFW)·블록 간(CFW) 가중합하고, 보간한 MS에 더합니다.
- **어댑터:** 공식 입력 순서 `(ms, pan)`으로 호출하고 데이터의 `lms`는 쓰지 않습니다. 추론 출력은 공식 test 스크립트처럼 [0, 1]로 자릅니다. 공식 코드는 고치지 않습니다.
- **CACW 중간층 크기:** 논문 부록 D대로 QB는 0.6n(29), GF2·WV3는 0.8n(38)입니다. 공식 코드는 0.8n 고정이라 QB는 어댑터에서 공식 `CovBlock`으로 다시 만듭니다.

### 저자 가중치

공개된 WV3 가중치 세 개 중 주 모델과 이름이 맞는 `BWNET_LAGConv_WV3/model.pth`를 평가 전에 정해 씁니다.

- 이 파일은 공식 코드와 키·크기가 모두 같고 CovBlock의 LayerNorm(텐서 14개)만 없습니다. 공식 `main_test.py`의 strict 읽기로는 읽히지 않습니다.
- 파일을 설명하는 해석이 두 가지입니다. LayerNorm 없이 학습했거나(`none`), 파라미터 없는 LayerNorm으로 학습했을 수 있습니다(`parameter_free`, 키를 남기지 않음). 둘 다 strict로 읽힙니다.
- 해석은 **WV3 validation** L1이 낮은 쪽으로 고르고(`model.py --select-author`, 결과는 `results/author_selection.json`), test는 고른 해석으로만 평가합니다.
- `BWNET_FusionNet_WV3/model.pth`는 공식 마지막 커밋에서 교체돼 FusionNet 구조와 맞지 않고, LAGConv 적용판과 같은 구조의 다른 가중치입니다. `BWNET_LGPNet_WV3/model.pth`에는 공식 코드에 없는 키가 하나 있습니다. 둘 다 평가에 쓰지 않습니다.

### 비교 조건

| 항목 | C2MS-SP | ADWM `retrain` |
|---|---|---|
| 학습 데이터·분할 | PanCollection 학습·validation | 같음 (팀 고정 파일, SHA-256 확인) |
| 정규화 | QB·WV3 2047, GF2 1023 | 같음 |
| 손실 | L1 + 거친 쌍 L1 + 0.46 분광 | L1 (논문, 공식 코드) |
| 에폭(출처) | 150 | QB 200 · GF2 500 (논문 부록 D, GF2는 WV3와 같은 설정) |
| 배치·학습률 | 32 · 1e-4 → 1e-6 cosine | 64 · 2e-3, 150에폭마다 0.5배 (공식 코드처럼 149·299·449에폭 뒤) |
| 가중치 선택 | validation MSE 최소 | 매 에폭 validation L1 최소 (공식 코드는 10에폭마다 저장) |
| 후처리 | 관측 일치 사영(K10) | 없음 |
| 시드 | 42·43·44 | 42 |
| 파라미터 수 | 2,735,610 | QB 389,692 · GF2 392,392 · WV3 396,024 |

**메모리:** 배치 64의 학습에는 역전파용 활성값이 약 20 GB 필요해 Kaggle T4(15 GB)에 들어가지 않습니다. 배치를 줄이지 않고, LAConv 층의 중간값을 저장하는 대신 역전파 때 다시 계산합니다(`torch.utils.checkpoint`). LAConv 층에는 BatchNorm·dropout이 없어 손실·기울기·BatchNorm 통계가 배치 64와 같습니다(같은 입력에서 기울기 차이 0으로 확인). 저장 활성값은 약 2.7 GB로 줄고 시간이 더 듭니다.

</details>

<a id="figures"></a>

<details>
<summary><b>그림</b> · 학습·평가 그래프 · 결과 이미지</summary>

평가 후 작성합니다.

</details>

<a id="evidence"></a>

<details>
<summary><b>재현 자료</b> · 코드, 가중치, 설정, 환경</summary>

- 공식 코드: `upstream/` 하위 모듈 (고정 커밋과 저자 가중치 SHA-256은 [config.json](config.json))
- 어댑터: [model.py](model.py) · 학습: [train.py](train.py) · Kaggle 실행 원본: [provenance/kaggle_adwm.py](provenance/kaggle_adwm.py)
- 데이터: 팀 고정 PanCollection 파일(성능 53·54번 config.json의 SHA-256)과 같은지 Kaggle 노트북이 확인합니다.

</details>
