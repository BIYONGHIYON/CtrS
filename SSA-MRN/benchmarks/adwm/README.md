# ADWM: Adaptive Dual-level Weighting Mechanism

| 평가자 | 설정 | 상태 |
|:---:|:---:|:---:|
| 김경찬 | `author` WV3 (저자 가중치, 재학습 없음) · `retrain` QB 200에폭 · GF2 500에폭 (논문) | 계획 |

**출처:** Jie Huang, Haorui Chen, Jiaxuan Ren, Siran Peng, Liang-Jian Deng. *A General Adaptive Dual-level Weighting Mechanism for Remote Sensing Pansharpening*. CVPR 2025. [논문](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_A_General_Adaptive_Dual-level_Weighting_Mechanism_for_Remote_Sensing_Pansharpening_CVPR_2025_paper.html) · [arXiv](https://arxiv.org/abs/2503.13214) · [공식 코드](https://github.com/Jie-1203/ADWM) · MIT(재배포 가능). 고정 버전과 해시는 [config.json](config.json)에 있습니다.

**결과:** 아직 평가하지 않았습니다.

[평가 표](../README.md#models) · [결과](#results) · [해석](#verdict) · [방법](#method) · [그림](#figures) · [재현 자료](#evidence)

<a id="results"></a>

## 결과

아직 평가하지 않았습니다. 비교할 논문 보고값(Table 1, RR test)은 다음과 같습니다.

| 센서 | PSNR ↑ | SAM ↓ | ERGAS ↓ | Q ↑ | 우리 평가 설정 |
|---|---:|---:|---:|---:|---|
| WV3 | 39.170 | 2.913 | 2.145 | 0.921 (Q8) | `author` |
| QB | 38.466 | 4.450 | 3.705 | 0.937 (Q4) | `retrain` seed 42 |
| GF2 | 43.884 | 0.672 | 0.597 | 0.985 (Q4) | `retrain` seed 42 |

GF2 PSNR은 논문마다 약 43-45 dB와 49-50 dB 계열이 함께 있습니다([성능조사 문서](../../docs/pansharpening_survey/QB_GF_WV2_WV3_팬샤프닝_성능조사_2026-10-27.md)). 우리 평가기 값과 직접 비교하지 않습니다.

<a id="verdict"></a>

## 해석과 정리

평가 후 작성합니다.

<a id="method"></a>

<details>
<summary><b>방법</b> · 목적과 상태 · 모델과 비교 조건</summary>

### 목적과 상태

논문 QB RR PSNR(38.466 dB)이 C2MS-SP(38.373 dB)보다 높게 보고된 모델입니다. 기존 CNN에 공분산 기반 가중을 더하는 구조라 모델이 작습니다. 우리 환경에서 같은 데이터·지표로 이 차이가 재현되는지 확인합니다.

| 설정 | 센서 | 상태 |
|---|---|---|
| `author` | WV3, WV2 전이 | 계획 |
| `retrain` seed 42 | QB, GF2 | 계획 |

### 모델

논문은 ADWM을 LAGNet에 적용한 모델을 제안 방법으로 씁니다(3.2절). 공식 코드의 `model/bwnet_lagconv_sota.py`(BWNet)입니다.

- **입력 → 출력:** 저해상도 MS를 bicubic으로 4배 보간하고 PAN과 이어 붙입니다. 국소 적응 합성곱(LAConv) 잔차 블록 6개(48채널)를 거친 뒤, 블록별 특징을 공분산 기반 가중(채널 내 IFW, 블록 간 CFW)으로 합쳐 보간한 MS에 더합니다.
- **어댑터에서 맞춘 점:** 공식 입력 순서 `(ms, pan)`으로 호출합니다. 데이터의 `lms`는 쓰지 않습니다. 추론 출력은 공식 test 스크립트처럼 [0, 1]로 자릅니다.
- **저자 가중치와 LayerNorm:** 공개 WV3 가중치는 공식 코드와 키·크기가 모두 같고, 공분산 블록의 LayerNorm(텐서 14개)만 없습니다. 이를 설명하는 해석이 두 가지라 둘 다 평가합니다.
  - `author`: LayerNorm 없이 학습한 가중치로 보고 이 층을 항등으로 바꿉니다.
  - `author_ln` (진단): 학습 파라미터가 없는 LayerNorm(`elementwise_affine=False`)으로 학습한 가중치로 봅니다. 이런 LayerNorm은 가중치 파일에 키를 남기지 않고, 공식 LayerNorm의 초깃값(weight 1, bias 0)과 계산이 같습니다.
  - 어느 해석이 맞는지는 WV3 논문 값 재현으로 확인합니다. 두 결과를 모두 보고합니다.
- **사용하지 않은 가중치:** 공식 저장소의 `BWNET_FusionNet_WV3/model.pth`는 마지막 커밋에서 교체돼, FusionNet 구조가 아니라 LAGConv 적용판과 같은 키를 가진 다른 가중치입니다. 이름이 맞는 `BWNET_LAGConv_WV3`만 미리 정해 쓰고, test 결과를 보고 둘 중 고르지 않습니다.
- **공식 코드 수정:** 없습니다. QB `retrain`의 공분산 블록 크기만 어댑터에서 공식 클래스로 다시 만듭니다(아래 표).

### 비교 조건

| 항목 | C2MS-SP | ADWM `retrain` |
|---|---|---|
| 학습 데이터·분할 | PanCollection 학습·validation | 같음 |
| 정규화 | QB·WV3 2047, GF2 1023 | 같음 |
| 손실 | L1 + 거친 쌍 L1 + 0.46 분광 | L1 (논문, 공식 코드) |
| 에폭(출처)·배치·학습률 | 150 · 32 · 1e-4 → 1e-6 cosine | QB 200 · GF2 500 (논문) · 64 · 2e-3, 150에폭마다 0.5배 |
| 배치 처리 | — | 배치 64를 같은 크기 조각으로 나눠 기울기를 누적한 뒤 한 번 갱신(아래 설명) |
| 공분산 블록 은닉 크기 | — | QB 0.6n, GF2 0.8n (논문). 공식 코드는 0.8n 고정 |
| 가중치 선택 | validation MSE 최소 | 매 에폭 validation L1 최소 (공식 코드는 10에폭마다 저장) |
| 후처리 | 관측 일치 사영(K10) | 없음 |
| 시드 | 42·43·44 | 42 |
| 파라미터 수 | 2,735,610 | QB 389,692 · GF2 392,392 · WV3 396,024 (어댑터에서 센 값) |

에폭은 논문 구현 세부사항을 따릅니다. 공식 `main_train.py`의 기본값은 모든 센서 500에폭입니다.

**기울기 누적:** 배치 64의 학습에는 역전파용 활성값만 약 19 GB가 필요합니다(어댑터로 계산한 값). 논문은 24 GB인 RTX 3090을 썼고, Kaggle T4(15 GB)에서는 첫 Kaggle 실행이 메모리 부족으로 멈췄습니다. 그래서 배치 64를 같은 크기 조각으로 나눠 조각마다 손실/조각 수로 역전파하고 한 번 갱신합니다. 손실과 기울기는 배치 64와 같습니다(BatchNorm을 고정하고 확인한 차이 3e-18). BatchNorm만 더 작은 조각의 통계를 씁니다. 조각 수는 학습 전에 실제 배치 하나로 메모리에 들어가는 가장 작은 값(1, 2, 4, 8 순)을 고르고 `history.json`에 기록합니다.

</details>

<a id="figures"></a>

<details>
<summary><b>그림</b> · 학습·평가 그래프 · 결과 이미지</summary>

평가 후 작성합니다.

</details>

<a id="evidence"></a>

<details>
<summary><b>재현 자료</b> · 코드, 가중치, 설정, 환경</summary>

- 공식 코드: `upstream/` 하위 모듈 (고정 커밋은 [config.json](config.json))
- 저자 WV3 가중치: `upstream/weights/BWNET_LAGConv_WV3/model.pth` (SHA-256은 config.json)
- 어댑터: [model.py](model.py) · 학습: [train.py](train.py) · Kaggle 실행 원본: [provenance/kaggle_adwm.py](provenance/kaggle_adwm.py)

</details>
