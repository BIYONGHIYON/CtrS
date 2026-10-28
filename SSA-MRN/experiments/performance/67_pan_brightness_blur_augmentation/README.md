# 67. PAN 밝기·흐림 증강 학습

**연구 분야:** 성능 개선 · **실험자:** 이병현

**변경 내용:** 기준 모델 C2MS의 구조·축척 혼합·학습 일정은 그대로 둡니다. 학습할 때 입력 PAN만 표본마다 무작위로 밝기를 바꾸거나(×0.70-1.42) 흐리게(가우시안 σ 0-1.2 픽셀) 합니다. 정답·MS·LMS는 바꾸지 않습니다. 손실 두 가지(C2MS의 MSE, [성능 65번](../65_l1_spectral_loss/README.md)의 L1 + 분광)와 증강 세 가지(밝기, 흐림, 둘 다)를 조합한 6개 갈래를 WV3 seed 42로 학습하고, WV3 test와 WV2 전이로 효과를 나눠 봅니다.

| 확인된 결과 | 상태 | 개선 여부 |
|---|:---:|:---:|
| WV3 seed 42 단일 시드, 54번 C2MS 대비. **흐림 증강만 넣은 C2MS_B는 WV2 전이에서 RR PSNR +1.025 dB·SAM −0.070°(20장 중 19장 PSNR 상승)로 흐려짐이 줄었지만, 학습한 WV3에서는 −0.204 dB(20/20장 하락)였습니다.** 65번 손실에 두 증강을 함께 넣은 L1S_GB도 WV2 +0.827 dB, WV3 −0.167 dB로 같은 맞바꿈이 나타났습니다. 학습 전에 정한 1단계 관문(WV3를 지키면서 WV2 개선)을 통과한 갈래는 없습니다. v2.3 판정은 WV3에서 L1S 세 갈래 부분 개선·C2MS 세 갈래 개선 없음, WV2 전이에서 C2MS_B·L1S 세 갈래 부분 개선입니다. | 평가&nbsp;완료<br>seed42 | 부분&nbsp;개선 |

[실험 목록](../../../research/performance/README.md) · [수치](#results) · [이미지](#images) · [재현 자료](#evidence)

## 1. 목적과 상태

65번(L1 + 분광 손실)은 같은 센서에서 SAM을 줄였지만, WV3 모델을 WV2에 적용하면 C2MS보다 눈에 띄게 흐리고 어두운 경우가 있었습니다(3시드 평균 RR PSNR −0.28 dB, 시드별 −1.08, −0.16, +0.41 dB). 원인을 학습에 쓰지 않은 test가 아닌 WV3 validation에서 먼저 확인했습니다([진단 기록](provenance/stage0/valdiag.json)).

| WV3 validation 512장, seed 42 | C2MS | 58번 (MSE + 분광 0.01) | 65번 (L1 + 분광 0.46) |
|---|---|---|---|
| 넣은 세부(정답 대비) / 밴드 밝기 치우침: 원래 PAN | 0.94 / 0.1 DN | 0.94 / 0.2 DN | 0.93 / 0.5 DN |
| PAN × 0.70 | 0.92 / 4.3 DN | 0.80 / 5.0 DN | **0.66 / 18.5 DN** |
| PAN 가우시안 σ 0.88 픽셀 | 0.56 / 0.1 DN | 0.56 / 0.1 DN | 0.55 / 0.4 DN |

- PAN이 흐리면 세 모델 모두 세부를 덜 넣습니다. C2MS도 WV2에서 정답 세부의 58%만 넣는 이유로 봅니다.
- PAN이 어두우면 분광 손실이 강한 모델일수록 세부를 덜 넣고 출력이 어두워집니다. WV2 test에서 65번이 보인 밝기 치우침(18-23 DN)과 같은 현상입니다.
- 학습 데이터의 PAN/MS 밝기 비율은 센서마다 다릅니다(QB 0.96, GF2 0.82, WV3 1.16, [측정 기록](provenance/stage0/panratio.json)).

PAN의 밝기와 흐림을 학습 중에 바꿔 주면, 모델이 PAN의 절대 밝기에 덜 의존하고 흐린 PAN에서도 세부를 넣도록 배울 것이라는 가설입니다. 2026-10-27 사용자 결정으로 효과를 나누어 1시드로 먼저 확인했습니다.

2026-10-27 23:40 KST에 사용자 지시로 preflight(증강 범위·확률·재현성 확인 포함)를 마치고 학습을 시작했습니다. 2026-10-28 00:12에 사용자 요청(서버 재부팅)으로 첫 갈래 L1S_GB를 안전하게 멈추고 재개용 가중치를 저장했으며, 00:25에 같은 runtime에서 이어 학습했습니다. 여섯 갈래의 학습과 평가는 16:17에 끝났습니다. 기준 모델은 C2MS로 유지합니다.

## 2. 변경한 구조와 비교 조건

| | 증강 없음 (보관 모델) | 밝기 (_G) | 흐림 (_B) | 둘 다 (_GB) |
|---|---|---|---|---|
| C2MS 손실 (MSE + 거친 쌍 MSE) | [54번](../54_multi_sensor_scale_mixed_training/README.md) C2MS (공식 대조군) | C2MS_G | C2MS_B | C2MS_GB |
| 65번 손실 (L1 + 거친 쌍 L1 + 0.46 × 분광) | 65번 C2MS_L1S (참조) | L1S_G | L1S_B | L1S_GB |

- **밝기 증강:** 확률 0.5로 PAN에 `exp(U(−0.35, 0.35))`(0.70-1.42배)를 곱합니다. 0.35는 학습 센서 사이의 밝기 비율 차이 `ln(1.16 / 0.82)`입니다.
- **흐림 증강:** 확률 0.5로 σ ~ U(0, 1.2) 픽셀의 가우시안(복제 경계, 직류 이득 1)을 적용합니다. σ 1.2는 MS 나이퀴스트에서 PAN 응답을 0.64배로 낮추며, 공개된 센서 PAN MTF 범위(약 0.11-0.17)를 덮도록 정했습니다. 이 센서표에는 WV2도 들어 있지만, WV2 test 결과를 보고 고르지는 않았습니다.
- **적용 위치:** 원래 쌍과 2배 거친 쌍 모두의 PAN 입력에 표본마다 따로 적용합니다. validation과 test 입력에는 적용하지 않습니다.
- **난수:** 표본마다 균등 난수 4개를 전용 생성기(seed + 200000)에서 항상 뽑습니다. 그래서 같은 표본은 _G와 _GB에서 같은 밝기를, _B와 _GB에서 같은 흐림을 받습니다. 생성기 상태는 재개용 가중치에 저장합니다.
- **구현:** 58번 학습기(54번 학습기 + 분광 손실 + EMA 기록)를 그대로 쓰고, 복원 손실 식과 PAN 입력 부분만 정확히 정한 위치에서 바꿉니다([train.py](train.py)). 선택은 C2MS와 같이 validation MSE 최소입니다.
- **시험:** WV3 RR·FR test 20장과 WV2 전이(WV3 모델을 그대로 적용) 20장. v2.3 항목에 더해, RR에서 세부 에너지(MS 나이퀴스트 위 대역의 출력/정답), 넣은 세부 비율 `std(y − LMS) / std(정답 − LMS)`, 밴드 밝기 치우침을 미리 정한 보조 지표로 함께 기록합니다([evaluate.py](evaluate.py)).

### 단계와 관문 (학습 전에 고정)

| 단계 | 내용 | 다음으로 가는 조건 |
|---|---|---|
| 0 진단 | WV3 validation으로 PAN 밝기·흐림 민감도 확인 | 완료 |
| 1 선별 | WV3 seed 42, 6개 갈래(갈래당 약 2.7시간, 모두 약 16-17시간). 순서: L1S_GB, C2MS_GB, L1S_G, L1S_B, C2MS_G, C2MS_B. 각 실행은 학습 직후 평가 | WV3 test가 C2MS보다 나쁘지 않고(RR PSNR −0.06 dB 이상, SAM +0.04° 이하), WV2에서 RR PSNR과 세부 에너지가 C2MS 이상, L1S 갈래는 WV3 SAM −0.05° 이하. 사용자와 함께 결정 |
| 2 반복·두 번째 전이 | 통과한 갈래: WV3 seed 43·44, QB seed 42-44(QB → GF2 전이 포함) | 사용자 승인 후 |


1단계는 2026-10-28 16:17에 끝났고, 관문을 통과한 갈래가 없어 2단계는 시작하지 않았습니다.

<a id="results"></a>

## 3. 정량 결과

모두 WV3 seed 42 단일 시드이고, 대조군은 같은 시드의 54번 C2MS입니다. test RR·FR 20장 평균이며 단일 시드 잡음 문턱은 0.06 dB입니다. 선택은 C2MS와 같은 원래 축척 validation MSE 최소이며 선택 에폭은 149-150입니다. 원본 값은 `results/<갈래>_WV3_s42/evaluation/<WV3|WV2>_v2_3.json`에 있습니다.

### WV3 (학습한 센서)

| 갈래 | RR PSNR / SAM | RR ΔPSNR / ΔSAM / ΔERGAS | RR2 ΔPSNR / ΔSAM | FR_WALD ΔPSNR / ΔSAM | 관측 비율 | PSNR 상승 장면 | v2.3 판정 |
|---|---|---|---|---|---:|---:|---|
| C2MS (대조군) | 38.493 / 3.090° | | | | | | 비교 기준 |
| 65번 C2MS_L1S (참조) | 38.586 / 3.001° | +0.093 / −0.089° / −0.008 | | | | | |
| C2MS_G | 38.491 / 3.097° | −0.002 / +0.007° / +0.002 | +0.049 / +0.003° | +0.020 / −0.005° | 0.876 | 11/20 | 개선 없음 |
| C2MS_B | 38.289 / 3.157° | −0.204 / +0.066° / +0.056 | +0.996 / −0.208° | −0.115 ✗ / +0.052° ✗ | 0.961 | 0/20 | 개선 없음 |
| C2MS_GB | 38.253 / 3.160° | −0.240 / +0.070° / +0.065 | +0.856 / −0.199° | −0.142 ✗ / +0.052° ✗ | 0.988 | 0/20 | 개선 없음 |
| L1S_G | 38.522 / 3.014° | +0.030 / **−0.076°** / +0.012 | +0.453 / −0.270° | −0.079 ✗ / −0.044° | 1.142 ✗ | 7/20 | 부분 개선 |
| L1S_B | 38.418 / 3.051° | −0.075 / −0.039° / +0.041 | +1.358 / −0.344° | −0.144 ✗ / −0.013° | 1.222 ✗ | 4/20 | 부분 개선 |
| L1S_GB | 38.325 / 3.078° | −0.167 / −0.012° / +0.067 | +0.930 / −0.310° | −0.200 ✗ / +0.008° | 1.279 ✗ | 4/20 | 부분 개선 |

### WV2 전이 (WV3 모델을 그대로 적용)

| 갈래 | RR PSNR / SAM | RR ΔPSNR / ΔSAM / ΔERGAS | RR2 ΔPSNR / ΔSAM | FR_WALD ΔPSNR / ΔSAM | 관측 비율 | PSNR 상승 장면 | v2.3 판정 |
|---|---|---|---|---|---:|---:|---|
| C2MS (대조군) | 29.523 / 5.854° | | | | | | 비교 기준 |
| 65번 C2MS_L1S (참조) | 28.445 / 5.988° | −1.078 / +0.134° / +0.545 | | | | | |
| C2MS_G | 29.261 / 6.111° | −0.262 / +0.257° / +0.151 | −0.385 ✗ / +0.296° ✗ | −0.342 ✗ / +0.231° ✗ | 0.994 | 2/20 | 개선 없음 |
| **C2MS_B** | **30.549 / 5.785°** | **+1.025 / −0.070° / −0.465** | +0.679 / +0.148° ✗ | +0.736 / −0.004° | 0.931 | **19/20** | 부분 개선 |
| C2MS_GB | 29.245 / 6.181° | −0.278 / +0.327° / +0.179 | −0.329 ✗ / +0.382° ✗ | −0.049 / +0.177° ✗ | 0.988 | 1/20 | 개선 없음 |
| L1S_G | 28.991 / 5.806° | −0.533 / −0.048° / +0.297 | −0.305 ✗ / −0.044° | −0.593 ✗ / +0.061° ✗ | 1.393 ✗ | 1/20 | 부분 개선 |
| L1S_B | 29.660 / 5.851° | +0.137 / −0.003° / −0.101 | +0.172 / −0.036° | +0.216 / −0.031° | 1.686 ✗ | 16/20 | 부분 개선 |
| **L1S_GB** | **30.350 / 5.793°** | **+0.827 / −0.061° / −0.384** | +1.176 / −0.110° | +0.831 / −0.079° | 1.345 ✗ | **19/20** | 부분 개선 |

### 세부 지표 (RR, 미리 정한 보조 지표)

넣은 세부는 `std(y − LMS) / std(정답 − LMS)`, 세부 에너지는 MS 나이퀴스트 위 대역의 출력/정답 에너지 비율이며 둘 다 1이 정답입니다. 밴드 밝기 치우침은 밴드별 평균 차이의 절댓값 중 가장 큰 값입니다.

| 갈래 | WV3 세부 에너지 / 넣은 세부 / 치우침 | WV2 세부 에너지 / 넣은 세부 / 치우침 |
|---|---|---|
| C2MS | 0.947 / 0.956 / 0.17 DN | 0.585 / 0.631 / 5.56 DN |
| 65번 C2MS_L1S | 0.937 / 0.948 / 0.27 DN | 0.452 / 0.523 / 23.01 DN |
| C2MS_G | 0.944 / 0.952 / 0.20 DN | 0.563 / 0.626 / 6.43 DN |
| C2MS_B | 0.943 / 0.952 / 0.27 DN | **0.702 / 0.731** / 5.43 DN |
| C2MS_GB | 0.939 / 0.948 / 0.25 DN | 0.643 / 0.669 / 4.53 DN |
| L1S_G | 0.936 / 0.946 / 0.31 DN | 0.456 / 0.517 / 11.42 DN |
| L1S_B | 0.938 / 0.948 / 0.29 DN | 0.621 / 0.653 / 16.24 DN |
| L1S_GB | 0.940 / 0.950 / 0.37 DN | **0.700 / 0.732** / 7.76 DN |

## 4. 기준선 대비 차이와 해석

**흐림 증강이 WV2 흐려짐을 줄였습니다.** 흐림 증강을 넣은 네 갈래 모두 WV2의 넣은 세부와 세부 에너지가 같은 손실의 증강 없는 모델보다 늘었습니다(세부 에너지 C2MS 0.585 → C2MS_B 0.702·C2MS_GB 0.643, 65번 0.452 → L1S_B 0.621·L1S_GB 0.700). 다만 C2MS_GB는 세부가 늘었는데도 PSNR이 낮아졌습니다(아래 밝기 증강 참고). validation 진단에서 본 "흐린 PAN에서는 세부를 덜 넣는다"는 원인과 맞습니다. 같은 갈래들은 WV3 RR2(2배 거친 시험)도 +0.86에서 +1.36 dB 올랐습니다.

**그 대가로 학습한 WV3에서 PSNR이 낮아졌습니다.** 흐림 증강 갈래는 WV3 RR PSNR이 0.075-0.240 dB 낮고, MSE 손실의 C2MS_B·C2MS_GB는 20장 모두에서 낮았습니다. 세부 지표는 C2MS와 거의 같아서(0.94 안팎), 세부의 양보다 세부의 정확도가 떨어진 것으로 봅니다. 학습 중 절반의 표본에서 PAN이 정답보다 흐려져, PAN과 정답의 관계가 덜 정확하게 학습되었다는 해석이 가능하지만 확인하지는 않았습니다.

**밝기 증강의 효과는 손실에 따라 달랐습니다.**
- 65번 손실에서는 밝기 치우침을 줄였습니다(WV2 23.0 → L1S_G 11.4 DN, 흐림과 함께 쓰면 16.2 → 7.8 DN). L1S_GB가 L1S_B보다 WV2에서 0.69 dB 높습니다.
- MSE 손실에서는 C2MS의 치우침이 이미 작아서(5.6 DN) 줄일 것이 거의 없었고, 오히려 WV2 SAM을 +0.26에서 +0.33° 키웠습니다. C2MS_GB는 C2MS_B보다 WV2에서 1.30 dB 낮았습니다. 밝기를 바꾼 PAN과 바꾸지 않은 MS 사이의 밴드 비율이 달라져, MSE 모델이 분광 관계를 덜 정확하게 배웠을 수 있습니다.
- WV3에서 C2MS_G는 C2MS와 차이가 없었고(−0.002 dB), L1S_G는 SAM −0.076°로 65번(−0.089°)에 가까운 SAM 이득을 유지한 유일한 갈래입니다.

**65번 손실의 SAM 이득은 흐림 증강과 함께 줄었습니다.** WV3 SAM 감소가 L1S_G −0.076°, L1S_B −0.039°, L1S_GB −0.012°로 흐림이 들어갈수록 작아졌습니다. 관측 비율은 세 갈래 모두 65번과 같이 1.10을 넘었습니다.

**1단계 관문 결과:**

| 갈래 | WV3 RR PSNR ≥ −0.06 dB | WV3 SAM ≤ +0.04° | WV2 PSNR·세부 에너지 ≥ C2MS | L1S: WV3 SAM ≤ −0.05° | 통과 |
|---|:---:|:---:|:---:|:---:|:---:|
| C2MS_G | ✓ | ✓ | ✗ | | ✗ |
| C2MS_B | ✗ (−0.204) | ✗ (+0.066) | ✓ | | ✗ |
| C2MS_GB | ✗ | ✗ | ✗ (PSNR) | | ✗ |
| L1S_G | ✓ | ✓ | ✗ | ✓ | ✗ |
| L1S_B | ✗ (−0.075) | ✓ | ✓ | ✗ (−0.039) | ✗ |
| L1S_GB | ✗ (−0.167) | ✓ | ✓ | ✗ (−0.012) | ✗ |

같은 센서 성능과 전이 성능이 서로 맞바뀌어, 두 조건을 함께 만족한 갈래는 없었습니다.

**주의할 점:** 모두 단일 시드입니다. 65번의 WV2 전이는 시드에 따라 −1.08에서 +0.41 dB까지 흔들렸고 seed 42가 가장 나쁜 시드였으므로, 65번 대비 차이(L1S_GB +1.91 dB)는 과장되었을 수 있습니다. 대조군 C2MS의 WV2 결과도 시드에 따라 달라 C2MS 대비 +1.0 dB 역시 1시드로는 확정할 수 없습니다. 증강 강도(σ 최대 1.2, 밝기 ±0.35)는 학습 센서와 공개 센서 MTF로 정했으며 WV2 test 결과를 보고 고르지 않았습니다. WV2 test를 보고 강도를 다시 고르면 전이 결과가 test에 맞춰질 수 있습니다.

<a id="graphs"></a>

## 5. 학습·평가 그래프

각 갈래의 validation 곡선(`learning.png`)과 C2MS 대비 장면별 PSNR 차이(`comparison.png`)입니다. WV2 전이에서 가장 크게 좋아진 두 갈래를 표시하고, 나머지는 `results/<갈래>_WV3_s42/evaluation/`에 있습니다.

**C2MS_B (MSE + 흐림)**

![C2MS_B 학습 곡선](results/C2MS_B_WV3_s42/evaluation/learning.png)

![C2MS_B test 비교](results/C2MS_B_WV3_s42/evaluation/comparison.png)

**L1S_GB (65번 손실 + 밝기 + 흐림)**

![L1S_GB 학습 곡선](results/L1S_GB_WV3_s42/evaluation/learning.png)

![L1S_GB test 비교](results/L1S_GB_WV3_s42/evaluation/comparison.png)

<a id="images"></a>

## 6. 결과 이미지

고정 6장면(0-based 0·1·3·7·14·17)의 C2MS 대 각 갈래 이미지를 `common/result_images.py`로 만들었습니다(seed 42, RGB 밴드 [4,2,1]). 장면 번호는 결과를 보기 전에 정한 고정 번호입니다.

- **원본 크기 비교:** 위 줄은 LR MS(최근접 확대) · PAN 입력 · C2MS · 후보 · 정답, 아래 줄은 C2MS와 후보의 |예측 − 정답| 지도입니다(같은 색 척도, 밝을수록 오차가 큼).
- **확대 비교:** 후보의 개선(C2MS 오차 − 후보 오차)이 가장 큰 64×64 영역을 4배 확대한 정답 전체(노란 상자) · C2MS · 후보 · 정답 · C2MS 오차 · 후보 오차입니다. 후보에 유리하게 고른 영역이므로 전체 차이는 아래 장면 PSNR로 확인합니다.

### 장면 PSNR (C2MS → 갈래, dB)

| 대상 | 갈래 | 0 | 1 | 3 | 7 | 14 | 17 |
|---|---|---|---|---|---|---|---|
| WV2 | C2MS | 28.69 | 26.82 | 34.94 | 29.18 | 28.93 | 29.39 |
| | C2MS_G | 28.40 | 26.59 | 34.64 | 28.73 | 28.66 | 29.15 |
| | **C2MS_B** | **30.01** | **28.29** | **35.85** | **30.07** | **29.92** | **30.18** |
| | C2MS_GB | 28.37 | 26.58 | 34.78 | 28.74 | 28.67 | 29.20 |
| | L1S_G | 28.11 | 26.29 | 34.57 | 28.49 | 28.13 | 28.81 |
| | L1S_B | 28.93 | 27.13 | 35.10 | 29.21 | 29.27 | 29.76 |
| | **L1S_GB** | 29.66 | 27.88 | 35.66 | 29.83 | 29.75 | 30.19 |
| WV3 | C2MS | 39.81 | 37.60 | 41.25 | 38.39 | 33.56 | 36.01 |
| | C2MS_G | 39.85 | 37.61 | 41.13 | 38.35 | 33.62 | 36.02 |
| | C2MS_B | 39.33 | 37.05 | 41.03 | 38.27 | 33.44 | 35.96 |
| | C2MS_GB | 39.32 | 37.05 | 40.84 | 38.26 | 33.44 | 35.91 |
| | L1S_G | 39.59 | 37.40 | 41.53 | 38.28 | 33.41 | 36.00 |
| | L1S_B | 39.36 | 37.14 | 41.51 | 38.27 | 33.34 | 35.93 |
| | L1S_GB | 39.22 | 37.03 | 41.34 | 38.17 | 33.28 | 35.85 |

WV2에서는 C2MS_B와 L1S_GB가 여섯 장면 모두 C2MS보다 높았고, WV3에서는 두 갈래 모두 장면 3(L1S_GB)을 빼고 낮았습니다.

### WV2 전이 · C2MS_B (MSE + 흐림)

C2MS보다 건물 경계와 도로 표시가 선명하고, 오차 지도에서 경계의 밝은 오차가 줄었습니다. 색은 C2MS와 비슷합니다.

![WV2 C2MS_B 원본 크기 비교 · 장면 0](results/C2MS_B_WV3_s42/evaluation/WV2/full_scene_00.png)

![WV2 C2MS_B 확대 비교 · 장면 0](results/C2MS_B_WV3_s42/evaluation/WV2/compare_scene_00.png)

<details>
<summary><b>WV2 C2MS_B 나머지 5장면 펼치기</b> (1·3·7·14·17)</summary>

**장면 1**

![WV2 C2MS_B 원본 크기 비교 · 장면 1](results/C2MS_B_WV3_s42/evaluation/WV2/full_scene_01.png)

![WV2 C2MS_B 확대 비교 · 장면 1](results/C2MS_B_WV3_s42/evaluation/WV2/compare_scene_01.png)

**장면 3**

![WV2 C2MS_B 원본 크기 비교 · 장면 3](results/C2MS_B_WV3_s42/evaluation/WV2/full_scene_03.png)

![WV2 C2MS_B 확대 비교 · 장면 3](results/C2MS_B_WV3_s42/evaluation/WV2/compare_scene_03.png)

**장면 7**

![WV2 C2MS_B 원본 크기 비교 · 장면 7](results/C2MS_B_WV3_s42/evaluation/WV2/full_scene_07.png)

![WV2 C2MS_B 확대 비교 · 장면 7](results/C2MS_B_WV3_s42/evaluation/WV2/compare_scene_07.png)

**장면 14**

![WV2 C2MS_B 원본 크기 비교 · 장면 14](results/C2MS_B_WV3_s42/evaluation/WV2/full_scene_14.png)

![WV2 C2MS_B 확대 비교 · 장면 14](results/C2MS_B_WV3_s42/evaluation/WV2/compare_scene_14.png)

**장면 17**

![WV2 C2MS_B 원본 크기 비교 · 장면 17](results/C2MS_B_WV3_s42/evaluation/WV2/full_scene_17.png)

![WV2 C2MS_B 확대 비교 · 장면 17](results/C2MS_B_WV3_s42/evaluation/WV2/compare_scene_17.png)

</details>

### WV2 전이 · L1S_GB (65번 손실 + 밝기 + 흐림)

65번 seed 42에서 보였던 흐리고 어두운 결과가 사라졌고, C2MS보다 선명합니다.

![WV2 L1S_GB 원본 크기 비교 · 장면 0](results/L1S_GB_WV3_s42/evaluation/WV2/full_scene_00.png)

![WV2 L1S_GB 확대 비교 · 장면 0](results/L1S_GB_WV3_s42/evaluation/WV2/compare_scene_00.png)

<details>
<summary><b>WV2 L1S_GB 나머지 5장면 펼치기</b> (1·3·7·14·17)</summary>

**장면 1**

![WV2 L1S_GB 원본 크기 비교 · 장면 1](results/L1S_GB_WV3_s42/evaluation/WV2/full_scene_01.png)

![WV2 L1S_GB 확대 비교 · 장면 1](results/L1S_GB_WV3_s42/evaluation/WV2/compare_scene_01.png)

**장면 3**

![WV2 L1S_GB 원본 크기 비교 · 장면 3](results/L1S_GB_WV3_s42/evaluation/WV2/full_scene_03.png)

![WV2 L1S_GB 확대 비교 · 장면 3](results/L1S_GB_WV3_s42/evaluation/WV2/compare_scene_03.png)

**장면 7**

![WV2 L1S_GB 원본 크기 비교 · 장면 7](results/L1S_GB_WV3_s42/evaluation/WV2/full_scene_07.png)

![WV2 L1S_GB 확대 비교 · 장면 7](results/L1S_GB_WV3_s42/evaluation/WV2/compare_scene_07.png)

**장면 14**

![WV2 L1S_GB 원본 크기 비교 · 장면 14](results/L1S_GB_WV3_s42/evaluation/WV2/full_scene_14.png)

![WV2 L1S_GB 확대 비교 · 장면 14](results/L1S_GB_WV3_s42/evaluation/WV2/compare_scene_14.png)

**장면 17**

![WV2 L1S_GB 원본 크기 비교 · 장면 17](results/L1S_GB_WV3_s42/evaluation/WV2/full_scene_17.png)

![WV2 L1S_GB 확대 비교 · 장면 17](results/L1S_GB_WV3_s42/evaluation/WV2/compare_scene_17.png)

</details>

### WV3 (학습한 센서)

WV3에서는 두 갈래 모두 장면 PSNR이 C2MS보다 조금 낮습니다. 확대 비교는 후보에 유리한 영역을 고른 것이므로 전체 차이는 위 장면 PSNR 표로 확인합니다.

<details>
<summary><b>WV3 C2MS_B·L1S_GB 6장면 펼치기</b></summary>

**C2MS_B · 장면 0**

![WV3 C2MS_B 원본 크기 비교 · 장면 0](results/C2MS_B_WV3_s42/evaluation/WV3/full_scene_00.png)

![WV3 C2MS_B 확대 비교 · 장면 0](results/C2MS_B_WV3_s42/evaluation/WV3/compare_scene_00.png)

**C2MS_B · 장면 1**

![WV3 C2MS_B 원본 크기 비교 · 장면 1](results/C2MS_B_WV3_s42/evaluation/WV3/full_scene_01.png)

![WV3 C2MS_B 확대 비교 · 장면 1](results/C2MS_B_WV3_s42/evaluation/WV3/compare_scene_01.png)

**C2MS_B · 장면 3**

![WV3 C2MS_B 원본 크기 비교 · 장면 3](results/C2MS_B_WV3_s42/evaluation/WV3/full_scene_03.png)

![WV3 C2MS_B 확대 비교 · 장면 3](results/C2MS_B_WV3_s42/evaluation/WV3/compare_scene_03.png)

**C2MS_B · 장면 7**

![WV3 C2MS_B 원본 크기 비교 · 장면 7](results/C2MS_B_WV3_s42/evaluation/WV3/full_scene_07.png)

![WV3 C2MS_B 확대 비교 · 장면 7](results/C2MS_B_WV3_s42/evaluation/WV3/compare_scene_07.png)

**C2MS_B · 장면 14**

![WV3 C2MS_B 원본 크기 비교 · 장면 14](results/C2MS_B_WV3_s42/evaluation/WV3/full_scene_14.png)

![WV3 C2MS_B 확대 비교 · 장면 14](results/C2MS_B_WV3_s42/evaluation/WV3/compare_scene_14.png)

**C2MS_B · 장면 17**

![WV3 C2MS_B 원본 크기 비교 · 장면 17](results/C2MS_B_WV3_s42/evaluation/WV3/full_scene_17.png)

![WV3 C2MS_B 확대 비교 · 장면 17](results/C2MS_B_WV3_s42/evaluation/WV3/compare_scene_17.png)

**L1S_GB · 장면 0**

![WV3 L1S_GB 원본 크기 비교 · 장면 0](results/L1S_GB_WV3_s42/evaluation/WV3/full_scene_00.png)

![WV3 L1S_GB 확대 비교 · 장면 0](results/L1S_GB_WV3_s42/evaluation/WV3/compare_scene_00.png)

**L1S_GB · 장면 1**

![WV3 L1S_GB 원본 크기 비교 · 장면 1](results/L1S_GB_WV3_s42/evaluation/WV3/full_scene_01.png)

![WV3 L1S_GB 확대 비교 · 장면 1](results/L1S_GB_WV3_s42/evaluation/WV3/compare_scene_01.png)

**L1S_GB · 장면 3**

![WV3 L1S_GB 원본 크기 비교 · 장면 3](results/L1S_GB_WV3_s42/evaluation/WV3/full_scene_03.png)

![WV3 L1S_GB 확대 비교 · 장면 3](results/L1S_GB_WV3_s42/evaluation/WV3/compare_scene_03.png)

**L1S_GB · 장면 7**

![WV3 L1S_GB 원본 크기 비교 · 장면 7](results/L1S_GB_WV3_s42/evaluation/WV3/full_scene_07.png)

![WV3 L1S_GB 확대 비교 · 장면 7](results/L1S_GB_WV3_s42/evaluation/WV3/compare_scene_07.png)

**L1S_GB · 장면 14**

![WV3 L1S_GB 원본 크기 비교 · 장면 14](results/L1S_GB_WV3_s42/evaluation/WV3/full_scene_14.png)

![WV3 L1S_GB 확대 비교 · 장면 14](results/L1S_GB_WV3_s42/evaluation/WV3/compare_scene_14.png)

**L1S_GB · 장면 17**

![WV3 L1S_GB 원본 크기 비교 · 장면 17](results/L1S_GB_WV3_s42/evaluation/WV3/full_scene_17.png)

![WV3 L1S_GB 확대 비교 · 장면 17](results/L1S_GB_WV3_s42/evaluation/WV3/compare_scene_17.png)

</details>

나머지 네 갈래(C2MS_G·C2MS_GB·L1S_G·L1S_B)의 같은 형식 이미지는 `results/<갈래>_WV3_s42/evaluation/<WV3|WV2>/`에 있습니다.

<a id="evidence"></a>

## 7. 재현 자료

- [train.py](train.py): 58번 학습기에 갈래별 복원 손실과 PAN 증강을 넣은 학습 진입점. 바꾸는 위치는 정확히 정한 횟수만 일치해야 하며, 다르면 실행을 거부합니다. preflight에서 증강 범위·확률·재현성·조합을 확인합니다([augment_checks.json](provenance/run_records/preflight/augment_checks.json)).
- [model.py](model.py)·[prepare.py](prepare.py): 54번 C2MS 구조와 센서 연산을 그대로 씁니다.
- [run.py](run.py)·[evaluate.py](evaluate.py): 65번 컨트롤러와 평가(54번 대조군·65번·58번 참조 대비 v2.3, 세부 지표, 결과 이미지).
- [config.json](config.json): 학습 전에 고정한 갈래·증강 범위와 근거·관문. 서버에 봉인된 설정·소스 해시는 [runtime_config.json](provenance/run_records/runtime_config.json)에 있습니다.
- 가중치: `checkpoints/<갈래>_WV3_s42/best.pt`(선택 에폭 C2MS_B·C2MS_GB 149, 나머지 150)와 `best_ema.pt`는 Git에 있습니다. 재개용 `latest.pt`와 초기 `initial.pt`는 GitHub Release `weights-experiments-performance-67`에 있고 [manifest.json](manifest.json)의 `release_assets`에 크기·SHA-256이 있습니다.
- 결과: `results/<갈래>_WV3_s42/`의 `history.json`, `training_complete.json`, `evaluation_complete.json`, `evaluation/<WV3|WV2>_v2_3.json`(장면별 수치), 그래프와 결과 이미지.
- 실행 기록: `provenance/run_records/`(준비·벤치마크·preflight·컨트롤러 기록, 캐시 manifest, 환경), 서버 소스 manifest·검증([ctrs67-source-manifest.json](provenance/ctrs67-source-manifest.json), [source_verification.json](provenance/source_verification.json)), 실행 스크립트(`stage67.ps1`, `launch67.cmd`, `start67.ps1`), 재부팅 전 안전 종료·재개 스크립트(`stop67.ps1`, `ctrlc.py`, `resume67.ps1`)와 기록. 상세 학습 로그는 Git에 넣지 않았습니다.
- `provenance/stage0/`: validation 진단 스크립트와 결과, 학습 데이터 PAN/MS 밝기 비율 측정 기록.

## 8. 개선 여부와 다음 단계

**부분 개선**입니다(seed 42 단일 시드). 학습 전에 정한 1단계 관문을 통과한 갈래는 없으며, 기준 모델은 C2MS로 유지합니다.

| 갈래 | WV3 판정 (RR ΔPSNR / ΔSAM) | WV2 전이 판정 (RR ΔPSNR / ΔSAM) |
|---|---|---|
| C2MS_G | 개선 없음 (−0.002 / +0.007°) | 개선 없음 (−0.262 / +0.257°) |
| C2MS_B | 개선 없음 (−0.204 / +0.066°) | 부분 개선 (+1.025 / −0.070°, RR2 SAM 미달) |
| C2MS_GB | 개선 없음 (−0.240 / +0.070°) | 개선 없음 (−0.278 / +0.327°) |
| L1S_G | 부분 개선 (+0.030 / −0.076°) | 부분 개선 (−0.533 / −0.048°) |
| L1S_B | 부분 개선 (−0.075 / −0.039°) | 부분 개선 (+0.137 / −0.003°, 관측 비율 미달) |
| L1S_GB | 부분 개선 (−0.167 / −0.012°) | 부분 개선 (+0.827 / −0.061°, 관측 비율 미달) |

- PAN 흐림 증강은 WV2 전이의 흐려짐을 줄이는 데 효과가 있었고(C2MS_B +1.025 dB, 세부 에너지 0.585 → 0.702), 밝기 증강은 65번 손실의 밝기 치우침을 줄였습니다.
- 그러나 흐림 증강은 학습한 WV3의 PSNR을 0.08-0.24 dB 낮췄고, 65번 손실의 SAM 이득도 줄였습니다.
- 다음 단계는 사용자와 함께 정합니다. 가능한 방향은 (1) C2MS_B·L1S_GB를 seed 43·44로 반복해 전이 이득이 시드 잡음보다 큰지 확인, (2) 흐림 적용 확률이나 σ 최대값을 낮춰 WV3 손해를 줄이는 갈래, (3) 흐림 증강을 거친 쌍에만 적용하는 갈래입니다. (2)·(3)의 강도는 WV2 test가 아닌 validation이나 사전 근거로 정해야 합니다.
