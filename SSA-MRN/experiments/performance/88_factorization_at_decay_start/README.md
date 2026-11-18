# 88. 일정 학습률 학습 뒤 감쇠 시작 시점의 저랭크 분해(K2)와 감쇠 구간 학습: 150에폭 경량 C2MS-SP 단일 시드 확인

| 분야 | 실험자 | 상태 | 판정 |
|:---:|:---:|:---:|:---:|
| 성능 개선 | 이병현 | 계획 | **판단 보류** |

**변경:** 원래 크기 C2MS를 학습률 1e-4로 고정해 120에폭 학습합니다([84번](../84_long_training_decay_branches/README.md) TRUNK와 같은 방식). 감쇠를 시작하는 120에폭에 [83번](../83_factorized_c2ms_finetuning/README.md) K2 구조(연산 66%, C2MS-SP-Lite와 같은 구조)로 분해하고, 남은 30에폭 cosine 감쇠를 K2 모델로 학습합니다. 감쇠 구간이 분해 손실을 되찾는 미세조정 역할을 겸하므로 전체 학습은 C2MS-SP와 같은 150에폭입니다. 추론에는 [86번](../86_fft_adjoint_fast_inference/README.md)의 빠른 K10 사영을 씁니다. 대조군은 같은 시드 C2MS-SP이고, seed 42 한 개로 먼저 확인합니다.

**결과:** 아직 학습하지 않았습니다.

[실험 표](../README.md#experiments) · [결과](#results) · [해석과 판정](#verdict) · [방법](#method) · [그림](#figures) · [재현 자료](#evidence)

<a id="results"></a>

## 결과

아직 학습하지 않았습니다. 갈래·센서별 validation과 test RR(사영 포함, C2MS-SP·C2MS-SP-Lite 대비), 비용(파라미터·연산·GPU 시간·학습 GPU 시간)을 채웁니다.

<a id="verdict"></a>

## 해석과 판정

### 판정과 다음 단계

아직 판단하지 않았습니다(판단 보류). K2D + 사영을 같은 시드 C2MS-SP와 RR test로 비교합니다([v2.4](../../../docs/judgement_criteria.md), 단일 시드 문턱 0.06 dB). QB·GF2·WV3가 모두 개선이어야 개선으로 적습니다. 세 센서 모두 C2MS-SP 대비 −0.04 dB 이내이면 "경량 비열등"으로 참고 표기합니다(판정 아님). 기준 모델 교체에는 3시드 확인과 사용자 결정이 필요합니다.

**2단계(3시드 확인):** seed 42 결과를 보기 전에 정한 조건(K2D가 세 센서 모두 개선이거나 경량 비열등)을 만족하면 서버에서 자동으로 seed 43·44를 학습합니다. TRUNK와 K2D만 학습하고, 3시드 평균(seed 42 포함)을 C2MS-SP와 비교합니다(v2.4, 2시드 이상 문턱 0.04 dB). FULL·K2F와 84번 B150 대조는 seed 42 참고로만 둡니다. 조건을 만족하지 않으면 2단계를 실행하지 않습니다.

예상과 미리 밝혀 둘 한계:

- **예상치:** 84번 B150(같은 150에폭, 일정만 다름)은 C2MS-SP보다 QB +0.060, GF2 +0.184, WV3 +0.342 dB였습니다. Lite의 분해 손실(seed 42 C2MS-SP 대비 QB −0.026, GF2 −0.058, WV3 −0.010 dB)을 빼면 GF2·WV3는 개선, QB는 잡음 수준이 예상됩니다. 그러면 판정은 부분 개선이 됩니다.
- **학습률 위험:** K2D는 분해 직후 새 Adam으로 학습률 1e-4에서 감쇠를 시작합니다. 83번은 감쇠가 끝난 모델을 5e-5에서 미세조정했으므로, 첫 몇 에폭에 분해 직후 해를 일부 잃을 수 있습니다. 분해 직후(0에폭)도 선택 후보에 넣고, 감쇠 후 분해(K2F)를 함께 봅니다.

<a id="method"></a>

<details>
<summary><b>방법</b> · 목적과 상태 · 변경한 구조와 비교 조건</summary>

### 목적과 상태

경량판 하나를 기준 모델로 삼으려면 C2MS-SP보다 가벼우면서 정확도가 같거나 높아야 합니다.

- **[87번](../87_reduced_width_short_finetuning/README.md):** 가운데 채널을 줄이는 방식은 seed 42에서 C2MS-SP보다 B32W96이 QB −0.076, GF2 −0.253, WV3 −0.146 dB, B32W64가 −0.236, −0.510, −0.443 dB였습니다. 같은 시드 Lite(K2)보다 손실이 훨씬 커서, 88번은 K2 구조를 씁니다.
- **[84번](../84_long_training_decay_branches/README.md):** 같은 150에폭에서도 고정 학습률 + 감쇠 일정이 cosine 일정보다 세 센서 모두 높았습니다.
- **88번의 질문:** 이 일정의 이득을 K2 구조에서도 얻을 수 있는지 봅니다.

**상태:** 2026-11-18 17:34에 서버에서 1단계(seed 42) 학습을 시작했습니다. 2단계(seed 43·44, TRUNK + K2D)는 1단계가 끝난 뒤 조건을 확인해 자동으로 시작합니다(2026-11-18 사용자 결정, seed 42 결과 전에 등록).

### 변경한 구조와 비교 조건

| 갈래 | 시작 | 학습 | 역할 |
|---|---|---|---|
| TRUNK | seed 42 초기값 | C2MS, Adam 1e-4 고정, 120에폭 | 모든 갈래의 공통 앞부분 |
| FULL | TRUNK 120에폭 전체 상태 | C2MS, 1e-4 → 1e-6 cosine 30에폭 | 같은 줄기의 원래 크기 감쇠(84번 B150 재실행), 참고 |
| **K2D** | TRUNK 120에폭 가중치 → K2 분해 | K2, 새 Adam, 1e-4 → 1e-6 cosine 30에폭 | **판정 후보** (총 150에폭) |
| K2F | FULL best → K2 분해 | K2, 5e-5 → 1e-6 cosine 10에폭(83번 Lite 방식) | 감쇠 후 분해, 참고 (총 160에폭) |

| 항목 | 내용 |
|---|---|
| 구조 | K2: backbone 3×3 64→64 8개를 3×3 64→32 + 1×1 32→64로, W2 3×3 128→128 16개를 1×1 128→384 · depthwise 3×3 · 1×1 384→128로 분해. 연산 66%, 파라미터 68%(C2MS-SP-Lite와 같음) |
| 분해 | 83번 방식: backbone은 학습 패치 512개(고정 생성기, seed 0)의 입력 2차 모멘트로 활성 인식 SVD, W2는 CP-ALS rank 384(150회, seed 0) |
| 데이터 | QB·GF2·WV3 학습·validation·test(각 20장면 RR·FR). WV3 모델은 WV2 test에도 적용(참고, 82번 보정 + 사영) |
| 손실 | 65번과 같음: 원래 쌍 L1 + 1.0 × 2배 거친 쌍 L1 + 0.46 × 분광 손실 |
| 학습 | batch 32, FP16 AMP(84번 학습기), 두 개씩 동시 학습. K2D는 데이터 순서·난수·GradScaler를 TRUNK에서 이어받고 Adam만 새로 만듭니다 |
| 가중치 선택 | 원래 축척 validation MSE 최소. FULL은 줄기 + 감쇠 에폭, K2D는 K2 에폭 0-30(0 = 분해 직후), K2F는 0-10 |
| 추론 | 네트워크 float32(TF32 끔), 사영 K10 float64(86번 FFT 전치) |
| 비교 | 공식: K2D + P vs C2MS-SP. 참고: K2D vs Lite(같은 구조, 학습법 효과), K2D vs FULL(분해 비용), K2F vs K2D(분해 시점), FULL vs 84번 B150(재실행 차이) |
| 시드 | 1단계 42(네 갈래). 2단계 43·44(TRUNK + K2D만, 조건부 자동 시작) |
| 시간 | 85번 측정 규칙으로 공식 설정과 86번 graph_fp16 설정, 256²·512². 학습 GPU 시간도 C2MS-SP·Lite와 비교 |
| 예상 소요 | 1단계 학습 약 9-10시간 + 평가 약 1시간, 2단계 학습 약 15시간 + 평가 약 1시간(실측 에폭 시간 기준 추정) |

</details>

<a id="figures"></a>

<details>
<summary><b>그림</b> · 학습·평가 그래프 · 결과 이미지</summary>

아직 학습하지 않았습니다. 갈래별 학습 L1과 validation PSNR·SAM 곡선(120에폭 분해 시점 표시, 65번·84번 B150 기준선)을 남깁니다. 고정 6장면(0·1·3·7·14·17)에서 C2MS-SP·Lite와 K2D + 사영을 비교한 이미지를 만듭니다.

</details>

<a id="evidence"></a>

<details>
<summary><b>재현 자료</b> · 코드, 가중치, 설정, 실행 기록</summary>

- [config.json](config.json): 가설, 근거, 갈래, 학습률 일정, 분해, 선택 규칙, 비교, 데이터 해시
- [model.py](model.py): 갈래별 구조(TRUNK·FULL은 C2MS, K2D·K2F는 83번 K2)와 83번 분해 함수
- [prepare.py](prepare.py): 86번 센서 연산·사영을 다시 내보냅니다(`FastProjected` = 86번 FFT 전치 사영)
- [train.py](train.py): 65번 손실의 58번 학습기에 84번 줄기·갈래 방식(학습률 일정, 120에폭 스냅샷, FULL 시작)과 분해 초기값 불러오기를 적용
- [evaluate.py](evaluate.py): 실행별 평가(C2MS-SP·Lite·84번 B150 대조), 단일 시드 요약(`summarize_arms`), 시간 측정
- [run.py](run.py): `prepare` → `optimize` → `run`(TRUNK → FULL·분해·K2D → 분해·K2F 순서로 두 개씩, 평가, 요약, 시간). `--stage 2`는 `gate` → `prepare` → `optimize` → `run`(seed 43·44 TRUNK + K2D, 3시드 요약)

</details>
