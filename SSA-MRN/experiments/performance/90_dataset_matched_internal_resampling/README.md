# 90. 모델 내부 축소·확대를 데이터셋 방식(23탭 보간, MTF 블러 + 솎음)으로 교체

| 분야 | 실험자 | 상태 | 판정 |
|:---:|:---:|:---:|:---:|
| 성능 개선 | 이병현 | 계획 | **판단 보류** |

**변경:** 원본 코드가 모델 안에서 쓰는 bilinear 축소·확대(`nn.Upsample`, `align_corners=True`)를 데이터셋을 만든 연산으로 바꿉니다. A는 입력 확대(LRPAN 4배, MS 2배)를 23탭 보간으로, B는 A에 더해 LRPAN을 PAN MTF 블러 + 4배 솎음으로 바꿉니다. C는 모델 안의 고정 축소·확대를 모두 데이터셋 연산으로 맞춘 모델로, B에 더해 내부 축소를 센서 MTF 블러 + 솎음으로, 출력 확대를 23탭 보간으로 바꿉니다. 학습할 파라미터는 바뀌지 않습니다. 대조군은 같은 조건에서 연산을 바꾸지 않은 N이며, N은 [89번](../89_spectral_coarse_loss_weights/README.md)에서 고른 λ = 1.2, c = 0.25의 확정 모델입니다.

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

- seed 42 하나로 고릅니다(2026-11-19 결정). 갈래는 50에폭 선별로 고르므로, 150에폭 K2D에서는 순위가 바뀔 수 있습니다.
- 갈래는 validation(사영 없는 네트워크 출력)으로만 고릅니다. test는 측정·보고만 합니다.
- B·C는 학습 센서의 MTF를 씁니다. WV2 전이에서도 WV3 커널을 그대로 씁니다.
- C는 내부 축소와 출력 확대를 함께 바꿉니다(완전히 맞춘 모델). C가 뽑혀도 두 부분의 효과는 나누지 않습니다.
- C의 1/4 크기 특징은 2배 MTF 두 번(사이에 SSAI 융합)으로 만들어, 데이터의 4배 MTF와 비슷하지만 같지는 않습니다.

<a id="method"></a>

<details>
<summary><b>방법</b> · 목적과 상태 · 변경한 구조와 비교 조건</summary>

### 목적과 상태

최종 모델(C2MS-SP보다 정확도가 같거나 높은 모델)을 만드는 세 번째 단계입니다([#203](https://github.com/BIYONGHIYON/CtrS/issues/203)). 같은 교체는 원본 SSA-MRN(K=6) 시절에 입력 확대만 바꿨을 때 개선이었지만(성능 6번), 지금 모델(K=4, W2 가지, L1·분광 손실, K2 분해, 사영)에서는 확인한 적이 없습니다.

**상태:** 계획입니다. 2026-11-20 사용자 결정으로 89번 B단계가 c\* = 0.25를 고른 뒤 서버에서 자동으로 시작했습니다(2026-11-21 02:11, [chain90.ps1](provenance/chain90.ps1)). 첫 사전 점검이 58번 점검 단계의 모델 생성에 학습 센서가 빠져 멈췄고, 고친 뒤 사전 점검부터 다시 실행했습니다([chain90_resume.ps1](provenance/chain90_resume.ps1)). 지금은 N 갈래(89번 확정)를 세 센서에서 학습하고 있습니다.

**갈래 선별 방식 변경(2026-11-21 사용자 결정):** A·B·C는 150에폭 K2D 대신 89번 λ 탐색과 같은 50에폭 선별 학습(원래 크기 2×2 모델, 40에폭 1e-4 고정 + 10에폭 cosine, λ 1.2, c 0.25, `AS`·`BS`·`CS`)으로 고릅니다. N의 선별 대조군은 같은 조건으로 이미 학습한 89번 B단계 `BL1.20C0.25`입니다(같은 학습법·시드, 별도 실행이라 재현 차이 최대 0.03 dB). 실행 중인 컨트롤러는 그대로 두고 [block_abc.py](provenance/block_abc.py)가 150에폭 A·B·C 실행의 작업 잠금을 잡아, 컨트롤러는 N을 마치는 대로 멈춥니다. `run.py --mode screen`(같은 runtime, 별도 소스 사본)이 빈 칸부터 선별 학습을 돌리고, 컨트롤러가 남긴 N 작업(WV3 분해·감쇠·평가)을 마친 뒤 같은 선택 규칙으로 갈래를 고릅니다(`screen_selection.json`). A·B·C 중 하나가 뽑히면 `run.py --mode chosen`이 이어서 그 갈래를 K2D 150에폭으로 QB·GF2·WV3 seed 42에서 학습하고, 89번 확정(NK)과 validation·RR test(사영 포함)·WV2 전이(참고)로 비교합니다(`chosen_comparison.json`). N이 남으면 아무것도 학습하지 않습니다. 선별 결과 갈래 C가 뽑혀(세 센서 평균 +0.016 dB) 2026-11-21 18:34부터 C를 150에폭으로 학습하고 있습니다.

**최종 모델 3시드(2026-11-21 사용자 결정):** `run.py --mode final`(네 번째 소스 사본, 같은 runtime)이 고른 갈래를 K2D 150에폭으로 seed 43·44 × QB·GF2·WV3에서 학습합니다(seed 42는 `chosen`). `chosen`이 마지막 1단계(WV3)를 돌리기 시작하면 남는 한 칸부터 쓰고, 서버 전체 작업 수는 두 개를 넘지 않습니다. 끝나면 같은 시드 C2MS-SP·C2MS-SP-Lite(둘 다 사영 포함)와 시드별로 짝지어 RR test 3시드 평균을 비교합니다(v2.4 3시드 문턱 0.04 dB, 보고용, WV2 전이·FR 참고, `final_summary.json`). 기준 모델 교체는 이 결과를 보고 사용자가 정합니다.

**중단 규칙(89번과 같음, PSNR 문턱 없음):** 프로세스 오류, 끝나지 않은 실행, validation 이상(best 에폭 0, 유한하지 않은 값) 중 하나라도 생기면 새 학습을 띄우지 않고, 돌고 있는 학습은 마치게 둔 뒤 `stopped.json`을 남기고 멈춥니다.

### 변경한 구조와 비교 조건

| 위치 | 원본 (N) | A | B | C |
|---|---|---|---|---|
| `downsample1` (PAN 1/4, LRPAN용) | bilinear | bilinear | PAN MTF(4배) 블러 + 솎음(위상 2) | B와 같음 |
| `upsample1` (LRPAN 4배) | bilinear | 23탭 보간 | 23탭 보간 | 23탭 보간 |
| `upsample100` (MS 2배) | bilinear | 23탭 보간(4배 사슬의 첫 단) | 23탭 보간 | 23탭 보간 |
| `downsample100` (특징·PAN 1/2) | bilinear | bilinear | bilinear | 특징은 밴드별 MS MTF(2배), PAN은 PAN MTF(2배) 블러 + 솎음(위상 0) |
| `downsample200` (특징 1/2 → 1/4) | bilinear | bilinear | bilinear | MS MTF(2배) 블러 + 솎음(위상 1) |
| `downsample2` (PAN 1/4) | bilinear | bilinear | bilinear | PAN MTF(4배) 블러 + 솎음(위상 2) |
| `upsample101`·`upsample102` (출력 2배씩) | bilinear | bilinear | bilinear | 23탭 보간 두 단(삽입 위상 1, 0) |
| W2 2×2 층 | 학습 | 학습 | 학습 | 학습 |

- 솎음 위상은 데이터의 격자에 맞췄습니다. MS 화소 j는 원래 크기 4j + 2 위치에 놓이고, 1/2 크기 화소 k는 원래 크기 2k 위치입니다.
- PAN 4배 축소는 데이터 열화 연산(54번 `Wald.pan`)과 1e-6 이내로, 2배 MS MTF 커널은 2배 거친 학습 쌍의 커널과 같습니다. 출력 확대 두 단을 이으면 데이터셋 23탭 4배 보간과 정확히 같습니다.
- 필터는 저장되지 않는 고정 버퍼라서 상태와 시드 초기값은 N과 같습니다. K2 분해 대상(3×3 합성곱 24개)도 같습니다.

| 항목 | 내용 |
|---|---|
| 학습법 | 89번 확정: W2 2×2(convT), 88번 K2D(원래 크기 120에폭 학습률 1e-4 고정 → K2 분해 → 30에폭 cosine 1e-4 → 1e-6), λ = 1.2, c = 0.25 |
| 센서·시드 | QB·GF2·WV3, seed 42. N은 K2D 150에폭(89번 확정), A·B·C는 50에폭 선별 |
| 가중치 선택 | 원래 축척 validation MSE 최소 에폭 |
| 갈래 선택 | 50에폭 선별 validation에서 세 센서 PSNR 차이(N 선별 대조군 대비) 평균이 가장 큰 갈래. 평균이 0보다 크고 어느 센서에서도 validation SAM이 τ(QB·GF2 0.004°, WV3 0.024°)보다 크게 나빠지지 않을 때만 고르고, 아니면 다음 갈래, 모두 안 되면 N |
| 평가 | RR test 20장(사영 유무), FR QNR(참고), WV3는 WV2 전이(82번 보정, 참고). 대조군 N, 88번 K2D, C2MS-SP(같은 시드) |
| 89번 확정 | N 갈래가 89번 확정 모델입니다. 88번 K2D + 사영과 RR test로 비교해 89번 보고서에 적습니다 |
| 실행 | 공용 서버 `C:\CtrS-p90`, 두 개씩 동시 |
| 예상 GPU 시간 | N 약 8.5, A·B·C 선별 약 8.5시간 |

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
- [model.py](model.py): 갈래별 내부 축소·확대 연산(`Interp23`, `Up23`, `MTFDown`)과 갈래 이름(`<v>T`·`<v>K`)
- [train.py](train.py): 89번 학습기에 K2D 일정과 89번 손실 계수를 적용
- [evaluate.py](evaluate.py): RR·FR test와 WV2 전이 평가(보고용)
- [run.py](run.py): `prepare` → `optimize` → `run`(학습 → 분해 → 학습 → 평가), `screen`(50에폭 선별과 갈래 선택), `chosen`(고른 갈래 150에폭과 89번 확정 비교), `final`·`finalctrl`(최종 모델 seed 43·44와 3시드 비교), 중단 규칙
- [chain90.ps1](provenance/chain90.ps1)·[chain90_resume.ps1](provenance/chain90_resume.ps1): 89번 B단계 뒤 자동 시작과 사전 점검 수정 뒤 재시작
- [block_abc.py](provenance/block_abc.py): 150에폭 A·B·C 실행 차단

</details>
