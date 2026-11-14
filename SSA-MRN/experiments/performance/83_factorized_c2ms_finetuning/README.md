# 83. 학습된 C2MS 가중치의 저랭크·depthwise 분해와 짧은 미세조정

**연구 분야:** 성능 개선

**실험자:** 이병현

**모델 이름:** 선택된 갈래 K2에 관측 일치 사영을 붙인 모델을 **C2MS-SP-Lite**라고 부릅니다(2026-11-14 결정). 분해 전 모델은 [81번](../81_l1_spectral_loss_projection/README.md)의 **C2MS-SP**(65번 C2MS_L1S + 사영)입니다. 코드와 가중치 안의 이름(`K2`)은 바꾸지 않았습니다.

**변경 내용:** 새 구조를 처음부터 학습하지 않습니다. [65번](../65_l1_spectral_loss/README.md) C2MS_L1S의 학습된 가중치에서 출발해 원래 해상도 backbone의 3×3 합성곱을 랭크 32 병목으로, W2 가지의 3×3 합성곱을 1×1 · depthwise 3×3 · 1×1로 분해합니다. 그다음 65번 손실로 10에폭만 미세조정하고, 추론 때 [81번](../81_l1_spectral_loss_projection/README.md)의 관측 일치 사영(K10)을 붙여 같은 시드 C2MS와 비교합니다. 목표는 연산을 줄이면서 수치도 C2MS보다 좋게 만드는 것입니다.

| 확인된 결과 | 상태 | 개선 여부 |
|---|:---:|:---:|
| validation 관문으로 고른 K2(연산 66%, 실사용 파라미터 68%)에 사영을 붙인 C2MS-SP-Lite는 같은 시드 C2MS보다 3시드 평균 QB +0.051 dB·SAM −0.100°·ERGAS −0.018, GF2 +0.157 dB·−0.024°·−0.014, WV3 +0.119 dB·−0.085°·−0.015로 세 센서 모두 개선이며 기준 모델 교체 조건을 충족합니다(자동 교체 안 함). 분해 전 C2MS-SP(65번 + 사영)와의 차이는 −0.012에서 −0.034 dB(경량 비열등)입니다. 다만 RTX 3060 Ti에서 K2의 실제 GPU 시간은 C2MS보다 31-37% 깁니다(256² 19.1 → 25.0 ms). | 평가&nbsp;완료<br>3seed | 개선 |

[실험 목록](../../../research/performance/README.md) · [수치](#results) · [그래프](#graphs) · [이미지](#images) · [재현 자료](#evidence)

## 1. 목적과 상태

경량화와 수치 개선을 함께 노립니다. 지금까지 처음부터 학습한 작은 구조는 정확도를 잃었습니다. [74번](../74_backbone_matched_parallel_branch/README.md)·[75번](../75_parallel_branch_width_depth/README.md)은 W2 가지를 줄이자 −0.13에서 −0.52 dB였고, [79번](../79_deep_narrow_branch_projection/README.md)·[80번](../80_multi_sensor_deep_narrow_projection/README.md)의 64×16 가지는 QB·GF2에서는 C2MS와 비슷했지만 WV3(8밴드)에서는 이득의 54-63%만 지켰습니다. [78번](../78_seed_ensemble_distillation/README.md)의 앙상블 증류도 학생에게 거의 옮겨지지 않았습니다. 이번에는 이미 잘 학습된 가중치를 근사하는 데서 출발합니다.

### 0단계 사전 진단 (로컬, 2026-11-14)

보관 가중치로 학습 없이(또는 아주 짧게) 확인했습니다. QB seed 42이고, **test를 보고 한 사후 확인**이라 갈래를 정하는 근거로만 쓰고 판정에는 쓰지 않습니다. 자료는 [provenance/stage0](provenance/stage0/)에 있습니다.

| 바꾼 것 (53번 C2MS) | 연산 (256²) | 분해 직후 test PSNR | 미세조정 뒤 |
|---|---:|---:|---:|
| backbone을 랭크 32 병목으로 (활성값 기준 SVD) | 86% | −0.063 dB | 안 함 |
| backbone 랭크 24 | 81% | −0.262 dB | 안 함 |
| W2를 랭크 96 병목으로 | 89% | −0.248 dB | 안 함 |
| W2를 랭크 64 병목으로 | 71% | −0.731 dB | 안 함 |
| W2를 1×1 · depthwise · 1×1 (CP 랭크 384) | 80% | −0.318 dB | 안 함 |
| backbone 랭크 32 + W2 CP 384 | 66% | −0.400 dB | **−0.045 dB** (batch 16 × 750스텝, 약 0.7에폭) |
| backbone 랭크 32 + W2 CP 256 | 51% | −1.186 dB | −0.190 dB (250스텝에서 중단) |

- 원래 해상도 backbone은 랭크를 절반으로 줄여도 거의 그대로입니다. W2 가지는 랭크가 꽉 차 있어 병목으로는 줄일 수 없지만, depthwise 형태(CP 분해)로 바꾸면 짧은 미세조정으로 회복됩니다.
- 65번 C2MS_L1S 가중치에서도 분해 직후 손실은 같은 양상이었습니다(backbone 랭크 32 −0.048 dB, W2 랭크 96 −0.264 dB).
- depthwise 블록을 처음부터 학습해 W2 출력을 흉내 내게 한 간이 실험은 일반 블록보다 훨씬 느리게 수렴했습니다(1,500스텝 뒤 −4.08 dB, 일반 블록 −2.36 dB). 처음부터 학습하지 않고 분해에서 출발하는 이유입니다.

**가설:** 65번 가중치를 분해하고 65번 손실로 짧게 미세조정하면, 연산 51-86%에서 65번과 같은 validation 정확도를 되찾습니다. 여기에 사영을 붙이면 81번처럼 같은 시드 C2MS보다 RR 세 지표가 좋아집니다.

**상태(2026-11-14):** 평가 완료입니다. 서버에서 11:03에 1단계 학습을 시작했고, 14:05에 관문이 K2를 골라 2단계로 넘어갔으며, 15:40에 17개 실행의 학습과 평가를 모두 마쳤습니다. 재부팅이나 재개 없이 진행됐습니다.

## 2. 변경한 구조와 비교 조건

```mermaid
flowchart LR
  W["65번 C2MS_L1S 가중치<br>(같은 시드·센서)"] --> F["분해 (학습 없음)<br>backbone 3×3: 활성값 기준 SVD → 3×3 64→32 + 1×1 32→64<br>W2 3×3: CP 분해 → 1×1 128→R · depthwise 3×3 · 1×1 R→128"]
  F --> T["미세조정 10에폭<br>65번 손실, 5e-5 → 1e-6 cosine"] --> N["분해 모델"] --> P["관측 일치 사영 K10<br>(추론 시, 파라미터 없음)"]
```

SSA 3단계, decoder, 입력 처리는 바꾸지 않습니다.

| 갈래 | backbone (64채널 3×3 여덟 개) | W2 가지 (128채널 3×3 열여섯 개) | 미세조정 | 연산 (QB 256²) | 실사용 파라미터 |
|---|---|---|---:|---:|---:|
| C2MS (53·54번, 공식 대조군) | 그대로 | 그대로 | 없음 | 59.63 GMAC | 2,735,610 |
| FT0 (대조군) | 그대로 | 그대로 | 10에폭 | 59.63 GMAC | 2,735,610 |
| K1 | 랭크 32 | 그대로 | 10에폭 | 51.04 GMAC (86%) | 2,604,538 |
| **K2** | 랭크 32 | CP 랭크 384 | 10에폭 | **39.06 GMAC (66%)** | 1,873,402 |
| K3 | 랭크 32 | CP 랭크 256 | 10에폭 | 30.17 GMAC (51%) | 1,330,682 |
| K2L | K2와 같음 | K2와 같음 | 40에폭 | 39.06 GMAC (66%) | 1,873,402 |

- **backbone 분해:** QB 학습 패치 512장(고정 난수)에서 각 합성곱 입력의 3×3 이웃 벡터의 2차 모멘트를 구하고, 그 위에서 출력 오차가 가장 작은 랭크 32 근사를 SVD로 구합니다.
- **W2 분해:** 가중치 텐서(출력 128 × 입력 128 × 3×3)를 CP 분해(교대 최소제곱 150회, 난수 0)해 1×1 · depthwise 3×3 · 1×1 세 층으로 옮깁니다. 데이터는 쓰지 않습니다.
- **FT0:** 분해 없이 같은 미세조정만 합니다. 미세조정 자체의 효과와 분해의 효과를 나눕니다.
- **K2L:** 미세조정 길이(10 대 40에폭)의 효과를 봅니다. QB는 학습을 늘리면 validation은 오르고 test는 내려간 전례([59번](../59_extended_cosine_training/README.md), [73번](../73_l1_step_learning_rate_schedule/README.md))가 있어 기본은 짧게 둡니다.
- **계획에서 뺀 갈래:** 사영을 넣은 채 미세조정하는 갈래는 하지 않습니다. 평가 사영은 가장자리 MS 5픽셀을 빼는데, 64×64 학습 패치(MS 16×16)에서는 가운데 6×6만 남아 학습 때의 정의가 추론과 달라지기 때문입니다.

| 항목 | 내용 |
|---|---|
| 데이터 | QB·GF2·WV3 학습·validation·test(각 20장면 RR·FR). WV3 모델은 WV2 test에도 적용(참고). 캐시는 H5에서 다시 만들고 63번(QB)·65번 3단계(GF2·WV3) 해시와 대조 |
| 손실 | 65번과 같음: 원래 쌍 L1 + 1.0 × 2배 거친 쌍 L1 + 0.46 × 분광 손실 |
| 학습 | Adam 5e-5 → 1e-6 cosine, 10에폭(K2L 40), batch 32, FP16 AMP, 서버 RTX 3060 Ti에서 두 개씩 동시 학습 |
| 가중치 선택 | 원래 축척 validation MSE 최소. 0에폭(분해 직후, FT0는 65번 그대로)도 후보에 포함 |
| 사영 | 81번 사영을 공액 기울기법 10회에서 멈춘 K10. 81번에서 기본 사영과 0.0003 dB 이내였습니다 |
| 대조군 | 같은 시드 C2MS(QB 53번, GF2·WV3 54번), 65번(분해 전 가중치), 원본 C0(이득 유지율) |

| 단계 | 실행 | 다음 단계 조건 (학습 전 고정) |
|---|---|---|
| 1 선별 | QB·WV3 seed 42, 다섯 갈래 | **validation만 씁니다.** 두 센서 모두에서 선택 가중치가 같은 시드 65번보다 validation PSNR −0.04 dB 이상, SAM +0.02° 이하인 갈래 가운데 가장 가벼운 것(K3 → K2 → K2L → K1 순)을 고릅니다. 통과한 갈래가 없으면 2단계를 하지 않습니다 |
| 2 확장 | 고른 갈래, QB·GF2·WV3 × seed 42-44 (1단계 실행 재사용) | 같은 시드 C2MS와 짝지은 센서별 3시드 평균, v2.4 |
| 3 (선택) | 고른 구조를 처음부터 150에폭 학습(QB seed 42) | 2단계 뒤 사용자 결정. 구조 자체와 초기값의 효과를 나눕니다 |

| 비교 | 의미 |
|---|---|
| **갈래 + 사영 대 같은 시드 C2MS** | 공식 판정(v2.4, RR만, 센서별 3시드 평균, 문턱 0.04 dB). QB·GF2·WV3가 모두 개선일 때만 실험 전체를 개선으로 표기합니다 |
| 갈래 + 사영 대 65번 + 사영 | 분해의 손실. 3시드 평균 −0.04 dB 이상이면 "경량 비열등"으로 참고 표기합니다(판정 아님) |
| 갈래 대 FT0 (1단계) | 미세조정만의 효과를 뺀 분해의 효과 |
| 갈래 + 사영 대 C2MS + 사영 | 사영을 양쪽에 붙인 공정 비교 |
| WV2 전이 | 참고만 합니다 |
| 비용 | 실사용 파라미터, 256² 합성곱 연산량, GPU 시간(256²·512², 사영 유무). depthwise는 연산량만큼 빨라지지 않을 수 있어 실제 시간을 함께 보고합니다 |

<a id="results"></a>

## 3. 정량 결과

모든 test 수치는 RR 20장면 평균입니다. 사영(+P)은 K10이며, FR QNR은 참고, WV2 전이(WV3 학습 모델)는 모두 참고입니다. 전체 수치는 [summary.json](results/summary.json)(3시드), [stage_gate.json](results/stage_gate.json)(관문), 실행별 `results/<갈래>_<센서>_s<시드>/`에 있습니다.

### 1단계 선별 (seed 42)

**validation 관문** (선택 가중치, 같은 시드 65번 대비, 문턱 PSNR −0.04 dB 이상·SAM +0.02° 이하)

| 갈래 (연산) | QB ΔPSNR / ΔSAM | WV3 ΔPSNR / ΔSAM | 선택 에폭 (QB · WV3) | 관문 |
|---|---|---|---|:---:|
| FT0 (100%) | +0.008 / −0.003 | +0.023 / −0.007 | 10 · 10 | 통과 (대조군) |
| K1 (86%) | +0.006 / −0.002 | +0.016 / −0.005 | 10 · 10 | 통과 |
| **K2 (66%)** | **−0.023 / +0.008** | **−0.023 / +0.007** | 10 · 10 | **통과 → 선택** |
| K3 (51%) | −0.103 / +0.037 | −0.128 / +0.042 | 10 · 10 | 탈락 |
| K2L (66%, 40에폭) | +0.003 / −0.002 | +0.063 / −0.018 | 38 · 39 | 통과 |

미리 정한 순서(K3 → K2 → K2L → K1)에 따라 통과한 갈래 가운데 가장 가벼운 K2를 골랐습니다. validation PSNR은 분해 직후(0에폭) QB 39.674 dB에서 1에폭 뒤 40.151 dB로 바로 회복했고, 이후 10에폭까지 천천히 올랐습니다.

**test (참고, 단일 시드)**

| 갈래 | QB: 갈래 + P 대 C2MS<br>PSNR / SAM / ERGAS | WV3: 갈래 + P 대 C2MS | 사영 없이 대 C2MS<br>QB · WV3 | 대 65번 + P<br>QB · WV3 |
|---|---|---|---|---|
| FT0 | +0.122 / −0.114 / −0.049 | +0.199 / −0.108 / −0.034 | +0.077 · +0.123 | +0.000 · +0.029 |
| K1 | +0.123 / −0.113 / −0.050 | +0.193 / −0.106 / −0.033 | +0.079 · +0.116 | +0.001 · +0.022 |
| **K2** | **+0.097 / −0.109 / −0.039** | **+0.161 / −0.096 / −0.025** | +0.050 · +0.081 | −0.026 · −0.010 |
| K3 | +0.038 / −0.095 / −0.014 | +0.061 / −0.063 / +0.001 | −0.014 · −0.026 | −0.084 · −0.110 |
| K2L | +0.127 / −0.116 / −0.051 | +0.235 / −0.121 / −0.043 | +0.083 · +0.159 | +0.005 · +0.064 |

![1단계 갈래 비교](results/stage1_arms.png)

### 2단계 판정 (K2, seed 42·43·44)

| 시스템 | QB<br>PSNR / SAM / ERGAS | GF2<br>PSNR / SAM / ERGAS | WV3<br>PSNR / SAM / ERGAS | WV2 전이 (참고) |
|---|---|---|---|---|
| 원본 SSA-MRN C0 | 37.429 / 4.941 / 4.124 | 46.549 / 0.953 / 0.896 | 37.536 / 3.494 / 2.571 | 28.586 / 6.253 / 5.046 |
| C2MS (공식 대조군) | 38.302 / 4.576 / 3.712 | 48.594 / 0.791 / 0.708 | 38.543 / 3.075 / 2.258 | 29.413 / 5.854 / 4.620 |
| C2MS + 사영 | 38.341 / 4.566 / 3.696 | 48.679 / 0.783 / 0.701 | 38.624 / 3.053 / 2.239 | 29.598 / 5.675 / 4.534 |
| 65번 + 사영 (분해 전) | 38.373 / 4.474 / 3.685 | 48.785 / 0.763 / 0.691 | 38.673 / 2.984 / 2.241 | 29.584 / 5.506 / 4.533 |
| K2 | 38.308 / 4.478 / 3.713 | 48.659 / 0.774 / 0.702 | 38.576 / 3.003 / 2.263 | 29.064 / 5.813 / 4.781 |
| **K2 + 사영** | **38.353 / 4.476 / 3.693** | **48.751 / 0.767 / 0.694** | **38.662 / 2.990 / 2.243** | 29.540 / 5.532 / 4.557 |

| 비교 (3시드 평균, v2.4) | QB | GF2 | WV3 | WV2 전이 (참고) |
|---|---|---|---|---|
| **K2 + 사영 대 C2MS (공식)** | **+0.051 / −0.100 / −0.018<br>개선** | **+0.157 / −0.024 / −0.014<br>개선** | **+0.119 / −0.085 / −0.015<br>개선** | +0.126 / −0.322 / −0.063 |
| K2 대 C2MS (사영 없이) | +0.006 / −0.098 / +0.001<br>부분 개선 | +0.064 / −0.017 / −0.006<br>개선 | +0.033 / −0.072 / +0.006<br>부분 개선 | −0.350 / −0.040 / +0.161 |
| K2 + 사영 대 C2MS + 사영 | +0.013 / −0.091 / −0.002<br>부분 개선 | +0.072 / −0.017 / −0.007<br>개선 | +0.038 / −0.063 / +0.003<br>부분 개선 | −0.059 / −0.144 / +0.022 |
| K2 + 사영 대 65번 + 사영 (분해의 손실) | −0.020 / +0.002 / +0.008 | −0.034 / +0.003 / +0.003 | −0.012 / +0.006 / +0.002 | −0.044 / +0.025 / +0.024 |

표 안의 숫자는 PSNR(dB) / SAM(°) / ERGAS 차이입니다. 분해의 손실은 세 센서 모두 −0.04 dB 이내라 계획에서 정한 "경량 비열등"에 해당합니다(판정 아님).

공식 비교의 시드별 결과와 장면 승수입니다. 판정은 세 시드 평균으로 하며 일부 시드만 골라 판정하지 않습니다.

| 센서 | seed 42 | seed 43 | seed 44 | C2MS보다 좋은 장면<br>PSNR · SAM (60장 중) |
|---|---|---|---|:---:|
| QB | +0.097 / −0.109 / −0.039 | +0.027 / −0.109 / −0.008 | +0.030 / −0.083 / −0.008 | 48 · 60 |
| GF2 | +0.167 / −0.024 / −0.014 | +0.202 / −0.030 / −0.019 | +0.102 / −0.019 / −0.008 | 56 · 60 |
| WV3 | +0.161 / −0.096 / −0.025 | +0.081 / −0.071 / −0.005 | +0.115 / −0.087 / −0.015 | 41 · 58 |
| WV2 (참고) | −0.582 | +0.243 | +0.718 | 38 · 58 |

| FR QNR (참고) | QB | GF2 | WV3 | WV2 |
|---|---:|---:|---:|---:|
| C2MS | 0.810 | 0.941 | 0.952 | 0.910 |
| K2 + 사영 | 0.910 | 0.963 | 0.950 | 0.924 |

**WV2 전이와 82번 밴드 보정 (참고):** [82번](../82_cross_sensor_band_calibration/README.md)의 밴드별 센서 보정을 붙이면 K2 + 사영이 WV2에서 29.54 → 31.55 dB·SAM 5.53 → 5.07°가 됩니다(3시드). 같은 보정을 붙인 C2MS는 31.75 dB, 65번은 31.54 dB입니다([wv2_band_calibration.json](results/wv2_band_calibration.json)).

![K2 3시드 비교](results/k2_3seed_comparison.png)

### 비용과 추론 시간

| 모델 | 실사용 파라미터 (4밴드 · 8밴드) | 256² 합성곱 연산량 (4밴드) | GPU 시간 RR 256² | GPU 시간 FR 512² |
|---|---:|---:|---:|---:|
| C2MS | 2,735,610 · 2,755,942 | 59.63 GMAC | 19.1 ms | 55.4 ms |
| K1 | 2,604,538 · 2,624,870 | 51.04 GMAC (86%) | 19.0 ms | 53.6 ms |
| **K2** | **1,873,402 · 1,893,734 (68%)** | **39.06 GMAC (66%)** | **25.0 ms** | **76.0 ms** |
| K3 | 1,330,682 · 1,351,014 | 30.17 GMAC (51%) | 21.5 ms | 62.6 ms |
| C2MS + 사영 (K10) | | | 56.6 ms | 167.7 ms |
| K2 + 사영 (K10) | | | 62.3 ms | 187.9 ms |

GPU 시간은 RTX 3060 Ti에서 학습이 돌지 않을 때 QB seed 42 평가 중에 잰 장면 0의 중앙값입니다(WV3: C2MS 23.0·61.2 ms, K2 26.7·82.1 ms). 학습 한 스텝은 K2가 FT0보다 짧았습니다(QB 0.158초 대 0.195초, 사전 벤치마크).

<a id="graphs"></a>

## 4. 기준선 대비 차이와 해석

- **공식 판정은 세 센서 모두 개선입니다.** K2 + 사영은 같은 시드 C2MS보다 QB +0.051 dB, GF2 +0.157 dB, WV3 +0.119 dB였고 SAM·ERGAS도 모두 좋아졌습니다. SAM은 178/180장에서 낮아졌습니다. QB seed 43·44(+0.027·+0.030 dB)는 문턱 아래였지만 판정은 규칙대로 3시드 평균으로 했습니다. 기준 모델 교체 조건(3시드 개선)을 충족하지만 자동으로 교체하지 않습니다.
- **연산 66%로 분해 전 65번 + 사영을 거의 그대로 지켰습니다.** 차이는 QB −0.020, GF2 −0.034, WV3 −0.012 dB입니다. 80번에서 처음부터 학습한 64×16 가지가 WV3에서 C2MS보다 −0.43 dB였던 것과 달리, 학습된 가중치를 분해해 출발하면 8밴드에서도 손해가 작았습니다.
- **분해 직후의 손실은 1에폭 만에 대부분 회복됩니다.** K2는 0에폭에 validation이 65번보다 QB 0.56 dB, WV3 0.89 dB 낮았지만 1에폭 뒤 0.09 dB, 0.18 dB 차이로 줄었습니다. K3(51%)는 10에폭 뒤에도 0.10-0.13 dB가 남아, W2 가지를 CP 랭크 256까지 줄이는 것은 이 미세조정 길이로는 무리였습니다.
- **미세조정 자체도 효과가 있었습니다.** 분해 없이 같은 미세조정만 한 FT0는 65번 + 사영 대비 QB +0.000, WV3 +0.029 dB였고, 40에폭 K2L은 QB +0.005, WV3 +0.064 dB였습니다. K2의 손실은 분해에서 오고 미세조정이 일부를 되찾은 것입니다. 더 긴 미세조정(K2L)이 K2의 손실을 메우지만, 관문 순서에 따라 2단계는 10에폭 K2로 했습니다.
- **연산량은 줄었지만 이 GPU에서 빨라지지는 않았습니다.** K2는 depthwise 합성곱(384채널, 그룹 384)을 쓰는데, RTX 3060 Ti에서는 일반 합성곱보다 효율이 낮아 추론이 C2MS보다 31-37% 느렸습니다. K1(backbone만 분해)은 C2MS와 같은 속도였습니다. 파라미터 수와 연산량 기준의 경량화이며, 이 GPU의 지연 시간 기준으로는 경량화가 아닙니다.
- **WV2 전이(참고)는 시드에 따라 크게 갈립니다**(−0.58, +0.24, +0.72 dB). 65번 seed 42가 원래 WV2에서 가장 나빴던 시드이고 K2도 같은 경향을 물려받았습니다. 82번 보정을 붙이면 세 시드 모두 31.3 dB 이상으로 안정됩니다.
- **validation과 test 방향:** GF2 K2는 validation이 65번보다 0.06 dB 낮았지만(관문 센서가 아님) test는 C2MS보다 +0.157 dB였습니다.

## 5. 학습·평가 그래프

실행마다 학습 L1, validation PSNR·SAM(0에폭은 분해 직후)과 C2MS·65번 validation 기준선을 남겼습니다. 아래는 선택된 K2의 seed 42 세 센서입니다.

![K2 QB seed 42 학습 곡선](results/K2_QB_s42/evaluation/learning.png)

![K2 GF2 seed 42 학습 곡선](results/K2_GF2_s42/evaluation/learning.png)

![K2 WV3 seed 42 학습 곡선](results/K2_WV3_s42/evaluation/learning.png)

<details><summary>다른 갈래와 seed 43·44 학습 곡선</summary>

![FT0 QB](results/FT0_QB_s42/evaluation/learning.png)

![K1 QB](results/K1_QB_s42/evaluation/learning.png)

![K3 QB](results/K3_QB_s42/evaluation/learning.png)

![K2L QB](results/K2L_QB_s42/evaluation/learning.png)

![K2 QB seed 43](results/K2_QB_s43/evaluation/learning.png)

![K2 QB seed 44](results/K2_QB_s44/evaluation/learning.png)

![K2 GF2 seed 43](results/K2_GF2_s43/evaluation/learning.png)

![K2 GF2 seed 44](results/K2_GF2_s44/evaluation/learning.png)

![K2 WV3 seed 43](results/K2_WV3_s43/evaluation/learning.png)

![K2 WV3 seed 44](results/K2_WV3_s44/evaluation/learning.png)

</details>

<a id="images"></a>

## 6. 결과 이미지

고정 6장면(0-based 0·1·3·7·14·17, seed 42)에서 현재 기준 C2MS와 K2 + 사영을 비교했습니다. 장면은 평가 전에 정했고 `common/result_images.py`로 서버 평가 중에 만들었습니다. 원본 크기 비교는 LR MS · PAN · C2MS · K2 + 사영 · 정답과 두 오차 지도(같은 색 척도)이고, 확대 비교는 K2 + 사영의 개선이 가장 큰 64×64 창을 4배 확대한 것입니다. **확대 창은 후보에 유리하게 고른 창**이므로 전체 차이는 장면 PSNR로 봅니다.

| 장면 | QB C2MS → K2 + 사영 | GF2 | WV3 |
|---:|---|---|---|
| 0 | 42.81 → 43.03 | 47.62 → 47.72 | 39.81 → 39.76 |
| 1 | 43.18 → 43.30 | 49.16 → 49.39 | 37.60 → 37.50 |
| 3 | 37.96 → 38.05 | 49.67 → 49.81 | 41.25 → 41.78 |
| 7 | 39.41 → 39.69 | 50.65 → 50.74 | 38.39 → 38.46 |
| 14 | 38.47 → 38.53 | 47.87 → 48.00 | 33.56 → 33.51 |
| 17 | 39.40 → 39.58 | 49.23 → 49.57 | 36.01 → 36.10 |

WV3의 고정 6장면에서는 세 장면이 조금 나빠졌습니다(20장 전체 seed 42는 +0.161 dB).

**QB 장면 0 (바다·항구 방파제)**

![QB 장면 0 원본 크기 비교](results/K2_QB_s42/evaluation/QB/full_scene_00.png)

![QB 장면 0 확대 비교](results/K2_QB_s42/evaluation/QB/compare_scene_00.png)

<details><summary>QB 장면 1·3·7·14·17</summary>

![QB 장면 1 원본 크기 비교](results/K2_QB_s42/evaluation/QB/full_scene_01.png)

![QB 장면 1 확대 비교](results/K2_QB_s42/evaluation/QB/compare_scene_01.png)

![QB 장면 3 원본 크기 비교](results/K2_QB_s42/evaluation/QB/full_scene_03.png)

![QB 장면 3 확대 비교](results/K2_QB_s42/evaluation/QB/compare_scene_03.png)

![QB 장면 7 원본 크기 비교](results/K2_QB_s42/evaluation/QB/full_scene_07.png)

![QB 장면 7 확대 비교](results/K2_QB_s42/evaluation/QB/compare_scene_07.png)

![QB 장면 14 원본 크기 비교](results/K2_QB_s42/evaluation/QB/full_scene_14.png)

![QB 장면 14 확대 비교](results/K2_QB_s42/evaluation/QB/compare_scene_14.png)

![QB 장면 17 원본 크기 비교](results/K2_QB_s42/evaluation/QB/full_scene_17.png)

![QB 장면 17 확대 비교](results/K2_QB_s42/evaluation/QB/compare_scene_17.png)

</details>

<details><summary>GF2 장면 0·1·3·7·14·17</summary>

![GF2 장면 0 원본 크기 비교](results/K2_GF2_s42/evaluation/GF2/full_scene_00.png)

![GF2 장면 0 확대 비교](results/K2_GF2_s42/evaluation/GF2/compare_scene_00.png)

![GF2 장면 1 원본 크기 비교](results/K2_GF2_s42/evaluation/GF2/full_scene_01.png)

![GF2 장면 1 확대 비교](results/K2_GF2_s42/evaluation/GF2/compare_scene_01.png)

![GF2 장면 3 원본 크기 비교](results/K2_GF2_s42/evaluation/GF2/full_scene_03.png)

![GF2 장면 3 확대 비교](results/K2_GF2_s42/evaluation/GF2/compare_scene_03.png)

![GF2 장면 7 원본 크기 비교](results/K2_GF2_s42/evaluation/GF2/full_scene_07.png)

![GF2 장면 7 확대 비교](results/K2_GF2_s42/evaluation/GF2/compare_scene_07.png)

![GF2 장면 14 원본 크기 비교](results/K2_GF2_s42/evaluation/GF2/full_scene_14.png)

![GF2 장면 14 확대 비교](results/K2_GF2_s42/evaluation/GF2/compare_scene_14.png)

![GF2 장면 17 원본 크기 비교](results/K2_GF2_s42/evaluation/GF2/full_scene_17.png)

![GF2 장면 17 확대 비교](results/K2_GF2_s42/evaluation/GF2/compare_scene_17.png)

</details>

<details><summary>WV3 장면 0·1·3·7·14·17</summary>

![WV3 장면 0 원본 크기 비교](results/K2_WV3_s42/evaluation/WV3/full_scene_00.png)

![WV3 장면 0 확대 비교](results/K2_WV3_s42/evaluation/WV3/compare_scene_00.png)

![WV3 장면 1 원본 크기 비교](results/K2_WV3_s42/evaluation/WV3/full_scene_01.png)

![WV3 장면 1 확대 비교](results/K2_WV3_s42/evaluation/WV3/compare_scene_01.png)

![WV3 장면 3 원본 크기 비교](results/K2_WV3_s42/evaluation/WV3/full_scene_03.png)

![WV3 장면 3 확대 비교](results/K2_WV3_s42/evaluation/WV3/compare_scene_03.png)

![WV3 장면 7 원본 크기 비교](results/K2_WV3_s42/evaluation/WV3/full_scene_07.png)

![WV3 장면 7 확대 비교](results/K2_WV3_s42/evaluation/WV3/compare_scene_07.png)

![WV3 장면 14 원본 크기 비교](results/K2_WV3_s42/evaluation/WV3/full_scene_14.png)

![WV3 장면 14 확대 비교](results/K2_WV3_s42/evaluation/WV3/compare_scene_14.png)

![WV3 장면 17 원본 크기 비교](results/K2_WV3_s42/evaluation/WV3/full_scene_17.png)

![WV3 장면 17 확대 비교](results/K2_WV3_s42/evaluation/WV3/compare_scene_17.png)

</details>

1단계 다른 갈래의 seed 42 이미지는 각 `results/<갈래>_<센서>_s42/evaluation/<센서>/`에 있습니다.

<a id="evidence"></a>

## 7. 재현 자료

- [config.json](config.json): 가설, 0단계 근거, 갈래, 학습 설정, 관문(validation만), 비교, 데이터 해시. 평가 뒤에는 첫머리의 상태·개선 여부·결과 요약 필드만 바꿨고, 서버에 봉인된 설정은 `provenance/run_records/runtime_config.json`에 있습니다
- [model.py](model.py): 분해 구조(`Model`), 활성값 기준 SVD(`svd_factor`), CP 분해(`cp_factor`), `factorize`
- [train.py](train.py): 58번 학습기에 65번 L1 치환과 분해 초기값 불러오기를 적용
- [prepare.py](prepare.py): 79번 센서 연산과 K10 사영
- [evaluate.py](evaluate.py): 실행별 평가(대조군은 센서·시드별로 한 번만 계산), validation 관문, 3시드 요약, `--figures`(보관 후 그래프)
- [run.py](run.py): 서버 실행 순서. `prepare`(캐시 재생성·해시 대조) → `factorize`(분해 초기값 45개, 해시 봉인) → `optimize`(벤치마크·사전 점검) → `run`(1단계 → 관문 → 2단계 → 요약)
- 가중치: `checkpoints/<갈래>_<센서>_s<시드>/best.pt`(선택 가중치)와 `best_ema.pt`(EMA 기록)는 Git에, 재개용 `latest.pt`와 분해 초기값 `initial.pt`는 GitHub Release `weights-experiments-performance-83`에 있고 [manifest.json](manifest.json)의 `release_assets`에 기록했습니다
- 결과: [summary.json](results/summary.json), [stage_gate.json](results/stage_gate.json), [factorization.json](results/factorization.json)(분해 손실), `results/controls/`(대조군 평가), 실행별 `history.json`·`training_complete.json`·`evaluation_complete.json`·`evaluation/<대상>_v2_4.json`(장면별 수치)·그래프·이미지, [wv2_band_calibration.json](results/wv2_band_calibration.json)(82번 보정 참고, [스크립트](provenance/wv2_calibration_check.py))
- 실행 기록: `provenance/run_records/`(봉인된 설정, 캐시 manifest, 환경, 사전 점검·벤치마크, 대기열 상태, 완료 기록), `provenance/ctrs83-source-manifest.json`·`source_verification.json`(서버에 보낸 소스 118개 대조), 실행·재개·검증 스크립트
- [provenance/stage0](provenance/stage0/): 0단계 진단 스크립트와 결과

## 8. 개선 여부와 다음 단계

**개선**입니다. 미리 고정한 비교(K2 + 사영 대 같은 시드 C2MS, 센서별 3시드 평균)에서 QB +0.051 dB, GF2 +0.157 dB, WV3 +0.119 dB로 세 센서 모두 개선이었고, 연산은 66%, 실사용 파라미터는 68%입니다. 기준 모델 교체 조건을 충족하지만 자동으로 교체하지 않고 팀이 정합니다.

**기준 모델 결정(2026-11-14):** 사용자 결정으로 기준 모델은 C2MS-SP(81번)로 정했고, C2MS-SP-Lite는 경량판으로 루트 README·결과 비교 사이트에 함께 보고합니다.

함께 볼 점입니다.

- **수치:** 분해 전 C2MS-SP(81번)보다 0.012-0.034 dB 낮습니다. 수치만 보면 C2MS-SP가 조금 더 높고, C2MS-SP-Lite는 같은 수준을 더 적은 파라미터·연산으로 냅니다.
- **속도:** 이 GPU에서는 depthwise 합성곱 때문에 K2가 C2MS보다 느립니다. 지연 시간이 중요하면 backbone만 분해한 K1(86%, 같은 속도, 65번 + 사영과 같은 수치)이 대안입니다.
- **전이:** 미학습 센서에는 82번 밴드 보정을 함께 쓰면 안정됩니다.

다음 단계 후보입니다.

1. **기준 모델:** 2026-11-14에 C2MS-SP(81번)로 정했습니다.
2. **빠른 분해 형태:** GPU 지연 시간까지 줄이려면 W2 가지를 depthwise 대신 일반 합성곱 기반의 작은 형태(예: 3×3 128→64 + 1×1)로 분해하고, 더 긴 미세조정(K2L처럼 40에폭)과 함께 확인합니다.
3. **선택적 3단계:** 같은 K2 구조를 처음부터 학습해 초기값(분해)의 효과를 따로 확인할 수 있습니다.
