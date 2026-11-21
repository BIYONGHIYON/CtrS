# ADWM (Adaptive Dual-level Weighting Mechanism)

| 평가자 | 설정 | 상태 |
|:---:|:---:|:---:|
| 김경찬 | `author` WV3(WV2 전이) · `retrain` QB 200·GF2 500에폭(논문 부록 D) | 계획 |

**출처:** Huang, Chen, Ren, Peng, Deng. A General Adaptive Dual-level Weighting Mechanism for Remote Sensing Pansharpening. CVPR 2025. [arXiv 2503.13214](https://arxiv.org/abs/2503.13214) · 공식 코드 [Jie-1203/ADWM](https://github.com/Jie-1203/ADWM) `aa185a6` · MIT(코드). 저자 가중치는 복사하지 않고 고정 커밋 하위 모듈에서 읽습니다.

**결과:** 아직 평가하지 않았습니다.

[평가 표](../README.md#models) · [결과](#results) · [해석](#verdict) · [방법](#method) · [그림](#figures) · [재현 자료](#evidence)

<a id="results"></a>

## 결과

아직 평가하지 않았습니다. 첫 Kaggle 세션에서 배치 64의 GPU 메모리와 에폭 시간을 측정하고, 저자 WV3 가중치의 LayerNorm 처리를 WV3 validation으로 고른 뒤 평가와 학습을 시작합니다.

<a id="verdict"></a>

## 해석과 정리

아직 평가하지 않았습니다.

<a id="method"></a>

<details>
<summary><b>방법</b> · 목적과 상태 · 모델과 비교 조건 · 공식 코드에서 확인한 점</summary>

### 목적과 상태

논문 RR 보고값이 QB PSNR 38.466으로 C2MS-SP보다 높게 보고된 모델입니다. 기존 CNN의 중간 특징에 공분산 기반 가중을 붙이는 방식이라 우리 모델에 가져올 점이 있는지 봅니다([#183](https://github.com/BIYONGHIYON/CtrS/issues/183)).

| 설정 | 센서 | 상태 |
|---|---|---|
| `author` | WV3 (WV2 전이) | 계획: LayerNorm 처리 validation 선택 전 |
| `retrain` seed 42 | QB 200에폭 · GF2 500에폭 | 계획: Kaggle 측정 전 |

### 모델과 비교 조건

PAN과 4배 bicubic으로 키운 LR MS를 합쳐 LAGConv 층으로 48채널로 올립니다. LAGConv 잔차 블록 6개를 지나며 나온 중간 특징마다 IFW가 채널 공분산으로 채널 가중을 곱합니다. CFW는 층 평균 특징의 공분산으로 층 가중(softmax)을 만들어 6개 특징을 더합니다. 마지막 LAGConv로 밴드 수를 맞춘 뒤 bicubic MS를 더해 출력합니다. 논문 3.2절의 주 모델(LAGNet + ADWM, `model/bwnet_lagconv_sota.py`)을 씁니다.

어댑터(`model.py`)는 공식 클래스를 고치지 않고 불러옵니다. 입력은 DN을 정규화 값으로 나눈 LR MS와 PAN이며 데이터셋의 LMS는 쓰지 않습니다. 추론 출력은 공식 `main_test.py`처럼 0-1로 자릅니다. QB만 논문대로 IFW 은닉 크기를 0.6n(29)으로 만듭니다.

| 항목 | C2MS-SP | ADWM (`retrain`) |
|---|---|---|
| 학습 데이터·분할 | PanCollection 학습·validation | 같음 (QB 17,139·1,905, GF2 19,809·2,201) |
| 정규화 | QB·WV3 2047, GF2 1023 | 같음 (공식 `data.py`와도 같음) |
| 손실 | L1 + 거친 쌍 L1 + 0.46 분광 | L1 |
| 에폭(출처)·배치·학습률 | 150 · 32 · 1e-4 → 1e-6 cosine | QB 200·GF2 500(논문 부록 D) · 64 · 2e-3, 150에폭마다 0.5배 |
| 가중치 선택 | validation MSE 최소 | 매 에폭 validation L1 최소 |
| 후처리 | 관측 일치 사영(K10) | 없음 (출력 0-1 자르기만) |
| 시드 | 42·43·44 | 42 |
| 파라미터 수 | | QB 389,692 · GF2 392,392 · WV3 저자 396,024 또는 396,612 |

### 공식 코드에서 확인한 점

- **저자 WV3 가중치의 LayerNorm:** `BWNET_LAGConv_WV3`에는 공식 클래스의 공분산 블록 LayerNorm 텐서 14개가 없고, 나머지 키와 크기는 모두 같습니다. 공식 `main_test.py`의 strict load는 이 가중치에서 실패합니다. LayerNorm을 빼는 방법(`identity`)과 기본값(가중치 1, 편향 0)으로 두는 방법(`default`) 중 WV3 validation L1이 작은 쪽을 test 전에 고릅니다. 직접 학습은 공식 클래스 그대로 LayerNorm을 포함합니다.
- **가중치 파일:** `BWNET_FusionNet_WV3`는 공식 저장소 마지막 커밋에서 교체돼 FusionNet이 아니라 LAGNet + ADWM과 같은 키에 다른 값을 가집니다. 평가 전에 이름이 맞는 `BWNET_LAGConv_WV3`로 정했고 이 파일은 기록만 합니다.
- **IFW 은닉 크기:** 논문은 QB 0.6n, WV3·GF2 0.8n이고 공식 코드는 0.8n으로 고정돼 있습니다. 논문값을 따릅니다.
- **에폭:** 논문은 QB 200, GF2는 WV3와 같은 500에폭이고 공식 코드 기본값은 500입니다. 논문값을 따릅니다.
- **학습률 감소 시점:** 공식 코드는 `initial_lr`과 `last_epoch=0`으로 StepLR을 만들어 150·300·450번째 에폭부터 절반 학습률을 씁니다(기본 StepLR보다 한 에폭 이릅니다). 같은 방식으로 만듭니다.
- **validation:** 공식 코드는 10에폭마다 섞고 마지막 묶음을 버린 validation 손실을 기록만 합니다. 우리는 매 에폭 validation 전체의 L1로 best를 고릅니다.
- **층 재계산:** 배치 64가 T4 메모리에 들어가지 않을 때만 씁니다. 블록 안 BatchNorm의 이동 평균이 재계산에서 한 번 더 갱신되므로 재계산 동안 통계를 저장했다가 되돌립니다. CPU에서 재계산 여부, 이어서 학습, 데이터 로더 worker 수와 관계없이 가중치·BatchNorm 통계·validation 기록이 같음을 확인했습니다.

</details>

<a id="figures"></a>

<details>
<summary><b>그림</b> · 학습·평가 그래프 · 결과 이미지</summary>

아직 평가하지 않았습니다.

</details>

<a id="evidence"></a>

<details>
<summary><b>재현 자료</b> · 코드, 가중치, 설정, 환경</summary>

- [config.json](config.json): 출처, 저자 가중치 SHA-256, 설정별 학습 조건, 논문 보고값
- [model.py](model.py): 어댑터 · [train.py](train.py): 학습(`train`), 측정(`measure`), 저자 LayerNorm 선택(`author-select`)
- [provenance/kaggle_session.py](provenance/kaggle_session.py): Kaggle 세션 실행 셀(T4 x2에서 QB·GF2 동시 학습, 이전 세션 출력에서 이어서 학습)
- `upstream/`: 공식 코드 하위 모듈(수정 없음)

</details>
