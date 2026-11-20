# 90. 모델 내부 축소·확대를 데이터셋 방식(23탭 보간, MTF 블러 + 솎음)으로 교체

| 분야 | 실험자 | 상태 | 판정 |
|:---:|:---:|:---:|:---:|
| 성능 개선 | 이병현 | 계획 | **판단 보류** |

**변경:** 원본 코드가 모델 안에서 쓰는 bilinear 축소·확대(`nn.Upsample`, `align_corners=True`)를 데이터셋을 만든 연산으로 바꿉니다. A는 입력 확대(LRPAN 4배, MS 2배)를 23탭 보간으로, B는 A에 더해 LRPAN을 PAN MTF 블러 + 4배 솎음으로, C는 B에 더해 내부 축소를 23탭 반대역 저역 필터 + 솎음으로 바꿉니다. 학습할 파라미터는 바뀌지 않습니다. 대조군은 같은 조건에서 연산을 바꾸지 않은 N이며, N은 [89번](../89_spectral_coarse_loss_weights/README.md)에서 고른 λ = 1.2, c = 0.25의 확정 모델입니다.

**결과:** 아직 학습하지 않았습니다.

[실험 표](../README.md#experiments) · [결과](#results) · [해석과 판정](#verdict) · [방법](#method) · [그림](#figures) · [재현 자료](#evidence)

<a id="results"></a>

## 결과

아직 학습하지 않았습니다. 갈래별 validation 차이(N 대비)와 선택 과정, RR test(사영 포함, N·88번 K2D·C2MS-SP 대비), FR QNR과 WV2 전이(참고)를 채웁니다.

<a id="verdict"></a>

## 해석과 판정

### 판정과 다음 단계

아직 판단하지 않았습니다(판단 보류). 고른 갈래를 같은 시드 N과 RR test로 비교합니다([v2.4](../../../docs/judgement_criteria.md), 단일 시드 문턱 0.06 dB, 보고용). 고른 갈래(또는 N)가 최종 모델이 되고, 기준 모델 교체 판정은 최종 모델 3시드에서 한 번만 합니다.

미리 밝혀 둘 한계:

- seed 42 하나로 고릅니다(2026-11-19 결정).
- 갈래는 validation(사영 없는 네트워크 출력)으로만 고릅니다. test는 측정·보고만 합니다.
- B는 학습 센서의 PAN MTF를 밴드 수로 정합니다(4밴드 0.15 = QB·GF2, 8밴드 0.14 = WV3). WV2 전이에서도 WV3 커널을 그대로 씁니다.
- C의 내부 축소 필터는 센서와 무관한 23탭 반대역 필터이고, 학습 전에 정했습니다.

<a id="method"></a>

<details>
<summary><b>방법</b> · 목적과 상태 · 변경한 구조와 비교 조건</summary>

### 목적과 상태

최종 모델(C2MS-SP보다 정확도가 같거나 높은 모델)을 만드는 세 번째 단계입니다([#203](https://github.com/BIYONGHIYON/CtrS/issues/203)). 같은 교체는 원본 SSA-MRN(K=6) 시절에 입력 확대만 바꿨을 때 개선이었지만(성능 6번), 지금 모델(K=4, W2 가지, L1·분광 손실, K2 분해, 사영)에서는 확인한 적이 없습니다.

**상태:** 계획입니다. 2026-11-20 사용자 결정으로 89번 B단계가 c\* = 0.25를 고르면 서버에서 자동으로 시작하도록 예약했습니다([chain90.ps1](provenance/chain90.ps1)). 순서는 서버 정리(빈 폴더·임시 스크립트만) → 89번 소스 복사와 90번 덮어쓰기 → 캐시 준비 → GPU가 빈 상태에서 구조 점검·대조군 평가·사전 점검 → 학습입니다. c\*가 0.25가 아니면 아무것도 시작하지 않고 사용자 결정을 기다립니다. N 갈래(89번 확정)를 세 센서에서 먼저 학습하고, 이어서 A·B·C를 학습합니다.

**중단 규칙(89번과 같음, PSNR 문턱 없음):** 프로세스 오류, 끝나지 않은 실행, validation 이상(best 에폭 0, 유한하지 않은 값) 중 하나라도 생기면 새 학습을 띄우지 않고, 돌고 있는 학습은 마치게 둔 뒤 `stopped.json`을 남기고 멈춥니다.

### 변경한 구조와 비교 조건

| 위치 | 원본 (N) | A | B | C |
|---|---|---|---|---|
| `downsample1` (PAN 1/4, LRPAN용) | bilinear | bilinear | PAN MTF 블러(41×41) + 4배 솎음(위상 2) | B와 같음 |
| `upsample1` (LRPAN 4배) | bilinear | 23탭 보간 | 23탭 보간 | 23탭 보간 |
| `upsample100` (MS 2배) | bilinear | 23탭 보간(4배 사슬의 첫 단) | 23탭 보간 | 23탭 보간 |
| `downsample100` (특징·PAN 1/2) | bilinear | bilinear | bilinear | 23탭 반대역 저역 + 솎음(위상 0) |
| `downsample200` (특징 1/2 → 1/4) | bilinear | bilinear | bilinear | 23탭 반대역 저역 + 솎음(위상 1) |
| `downsample2` (PAN 1/4) | bilinear | bilinear | bilinear | 반대역 두 번(전체 위상 2) |
| 출력 확대(`upsample101`·`upsample102`), W2 2×2 층 | 그대로 | 그대로 | 그대로 | 그대로 |

- 솎음 위상은 데이터의 격자에 맞췄습니다. MS 화소 j는 원래 크기 4j + 2 위치에 놓이고, 1/2 크기 화소 k는 원래 크기 2k 위치입니다.
- B의 PAN 축소는 데이터 열화 연산(54번 `Wald.pan`)과 1e-6 이내로 같습니다.
- 필터는 저장되지 않는 고정 버퍼라서 상태와 시드 초기값은 N과 같습니다. K2 분해 대상(3×3 합성곱 24개)도 같습니다.

| 항목 | 내용 |
|---|---|
| 학습법 | 89번 확정: W2 2×2(convT), 88번 K2D(원래 크기 120에폭 학습률 1e-4 고정 → K2 분해 → 30에폭 cosine 1e-4 → 1e-6), λ = 1.2, c = 0.25 |
| 센서·시드 | QB·GF2·WV3, seed 42 (갈래 4개 × 센서 3개 = 학습 24개) |
| 가중치 선택 | K2 30에폭 중 원래 축척 validation MSE 최소 |
| 갈래 선택 | 세 센서 validation PSNR 차이(N 대비) 평균이 가장 큰 갈래. 평균이 0보다 크고 어느 센서에서도 validation SAM이 τ(QB·GF2 0.004°, WV3 0.024°)보다 크게 나빠지지 않을 때만 고르고, 아니면 다음 갈래, 모두 안 되면 N |
| 평가 | RR test 20장(사영 유무), FR QNR(참고), WV3는 WV2 전이(82번 보정, 참고). 대조군 N, 88번 K2D, C2MS-SP(같은 시드) |
| 89번 확정 | N 갈래가 89번 확정 모델입니다. 88번 K2D + 사영과 RR test로 비교해 89번 보고서에 적습니다 |
| 실행 | 공용 서버 `C:\CtrS-p90`, 두 개씩 동시 |
| 예상 GPU 시간 | N 약 8.5, A·B·C 약 25, 합계 약 33.5시간 |

</details>

<a id="figures"></a>

<details>
<summary><b>그림</b> · 학습·평가 그래프 · 결과 이미지</summary>

아직 학습하지 않았습니다. 갈래별 학습·validation 곡선(N 기준선)과 고정 6장면(0·1·3·7·14·17)에서 N과 고른 갈래를 비교한 이미지를 남깁니다.

</details>

<a id="evidence"></a>

<details>
<summary><b>재현 자료</b> · 코드, 가중치, 설정, 실행 기록</summary>

- [config.json](config.json): 가설, 갈래 정의, 손실 계수, 선택 규칙, 평가, 중단 규칙
- [model.py](model.py): 갈래별 내부 축소·확대 연산(`Interp23`, `PanMTFDown`, `HalfDown`, `QuarterDown`)과 갈래 이름(`<v>T`·`<v>K`)
- [train.py](train.py): 89번 학습기에 K2D 일정과 89번 손실 계수를 적용
- [evaluate.py](evaluate.py): RR·FR test와 WV2 전이 평가(보고용)
- [run.py](run.py): `prepare` → `optimize` → `run`(학습 → 분해 → 학습 → 평가, 갈래 선택), 중단 규칙
- [chain90.ps1](provenance/chain90.ps1): 89번 B단계 뒤 자동 시작

</details>
