# 71. 거친 학습 쌍에 집중한 PAN 흐림 증강

**연구 분야:** 성능 개선 · **실험자:** 이병현

**변경 내용:** PAN Gaussian 흐림을 원래·거친 쌍 각각 확률 0.25로 적용하는 군과 거친 쌍만 확률 0.50으로 적용하는 군을 비교했습니다.

| 확인된 결과 | 상태 | 개선 여부 |
|---|:---:|:---:|
| 거친 쌍 전용 흐림은 WV2 RR +0.728 dB·SAM −0.006°였지만 WV3에서는 −0.026 dB·SAM +0.016°였고, WV2 RR2·FR_WALD SAM에 미달했습니다. 양쪽 쌍 흐림은 WV2 PSNR만 개선하고 SAM이 +0.703° 악화했습니다. | 평가&nbsp;완료<br>seed42 | 부분&nbsp;개선 |

[실험 목록](../../../research/performance/README.md) · [전체 수치](results/summary.json)

## 1. 목적과 상태

PSNR·SAM 동시 개선과 WV2 전이 개선을 검증했습니다. 사용자 지시로 seed 42만 실행했습니다. 70번 4회와 71번 2회 모두 150에폭 학습 및 전체 RR·RR2·FR_WALD·FR 관측 일치 평가를 마쳤으며, 전체 실행은 2026-10-30 17:43 KST에 완료됐습니다. validation으로 체크포인트를 고르고 모든 사전 등록 군을 test에서 보고했습니다. 추가 시드와 결합 실험은 실행하지 않았습니다.

## 2. 변경한 구조와 비교 조건

| 항목 | 조건 |
|---|---|
| 구조 | 원본 SSA-MRN K4 + W2인 C2MS |
| 학습 | Adam, batch32, AMP FP16, 150에폭 cosine 1e-4에서 1e-6 |
| 손실 | 원래 쌍 MSE + 가중치 1의 2배 거친 쌍 MSE |
| 선택 | 원래 축척 validation MSE 최소, 정확한 동률은 먼저 나온 checkpoint |
| 추론 | 단일 모델 한 번, 사영·앙상블·후처리 없음 |
| 공식 비교 | WV3 및 WV2 전이 54번, 같은 seed42 |
| 추가 비교 | 이번 실행기의 C2MS_R 재학습 대조군 |
| 장치 | RTX 3060 Ti, 한 번에 한 학습 |

BLUR_BOTH는 원래·거친 쌍 각각 확률 0.25, BLUR_COARSE는 원래 쌍 0·거친 쌍 0.50입니다. σ는 0-1.2픽셀이며 PAN만 바꾸고 MS·LMS·정답은 유지했습니다. 두 군은 손실 항 기준 기대 증강 노출량이 같지만 픽셀 수가 같지는 않습니다. 70번 C2MS_R_WV3_s42를 재학습 대조군으로 공유했습니다.

## 3. 정량 결과

각 test는 20장 전체 평균입니다. Δ는 후보 − 대조군이며 PSNR은 양수, SAM·ERGAS는 음수가 좋습니다. **단일시드 PSNR 잡음 문턱은 0.06 dB**입니다. 공식 기준과 이번 재학습 대조군 비교를 구분합니다. QB RR2 공식 판정은 57번 고정값 PSNR 39.1148 dB·SAM 4.2410°를 유지합니다.

| 대상 | 군 | best epoch | RR PSNR / SAM | 공식 ΔPSNR / ΔSAM / ΔERGAS | 재학습 대조군 ΔPSNR / ΔSAM | PSNR 상승 장면 |
|---|---|---:|---|---|---|---:|
| WV2 | BLUR_BOTH | 149 | 29.6669 / 6.5572° | +0.1437 / +0.7031° / -0.0374 | +0.5695 / +0.4062° | 13/20 |
| WV3 | BLUR_BOTH | 149 | 38.4211 / 3.1152° | -0.0716 / +0.0249° / +0.0220 | -0.1016 / +0.0211° | 3/20 |
| WV2 | BLUR_COARSE | 150 | 30.2509 / 5.8481° | +0.7277 / -0.0060° / -0.3427 | +1.1535 / -0.3028° | 18/20 |
| WV3 | BLUR_COARSE | 150 | 38.4662 / 3.1062° | -0.0265 / +0.0160° / +0.0080 | -0.0565 / +0.0121° | 7/20 |

| 대상 | 후보 | RR2 ΔPSNR / ΔSAM | FR_WALD ΔPSNR / ΔSAM | FR 관측 RMSE 비율 | 공식 v2.3 |
|---|---|---|---|---:|---|
| WV2 | BLUR_BOTH | -0.7896 / +0.8768° | -0.1458 / +0.7829° | 1.0420 | 부분 개선 |
| WV3 | BLUR_BOTH | +1.0652 / -0.2125° | -0.0431 / +0.0279° | 0.9915 | 개선 없음 |
| WV2 | BLUR_COARSE | +0.4911 / +0.0660° | +0.4855 / +0.0578° | 0.9374 | 부분 개선 |
| WV3 | BLUR_COARSE | +0.9278 / -0.1737° | +0.0132 / +0.0153° | 0.9224 | 개선 없음 |

**학습하지 않은 센서 WV2는 WV3 가중치를 그대로 적용한 전이입니다.** FR에는 정답이 없으며 FR_WALD는 더 열화한 별도 정답 기반 시험입니다. RR2는 RR 장면을 추가 4배 Wald 열화하는 시험이며 2배 거친 학습 쌍과 구분합니다. QNR·Dλ·Ds는 원시 JSON에 참고값으로 보존하며 판정에 쓰지 않습니다.

**사전 고정 validation 관문**

| 센서 | 후보 | clean ΔPSNR / ΔSAM | 고정 blur 평균 ΔPSNR / ΔSAM | 관문 |
|---|---|---|---|---|
| WV3 | BLUR_BOTH | -0.0866 / +0.0284° | +2.3096 / -0.3645° | 미달 |
| WV3 | BLUR_COARSE | -0.0604 / +0.0236° | +1.5552 / -0.2004° | 미달 |

blur validation은 σ 0.4·0.8·1.2의 산술평균이며 전이의 대리 진단입니다. 70번은 clean PSNR·SAM 동시 개선을 요구했고 QB만 통과했습니다. 71번은 clean PSNR ≥ −0.03 dB·SAM ≤ +0.02° 및 blur 평균 PSNR 상승·SAM 감소를 요구했으나 두 군 모두 clean 항목에서 미달했습니다. test를 보고 후보나 강도를 고르지 않았습니다.


**실제 업데이트 기록** (AMP overflow로 건너뛴 횟수를 포함합니다. 센서별 표본 순서 해시는 모든 군에서 같았습니다.)

| 실행 | 시도 | 적용 | 건너뜀 |
|---|---:|---:|---:|
| WV3_BLUR_BOTH_s42 | 45600 | 45595 | 5 |
| WV3_BLUR_COARSE_s42 | 45600 | 45596 | 4 |

## 4. 기준선 대비 차이와 해석

거친 쌍 전용 흐림은 WV2 RR +0.728 dB·SAM −0.006°였지만 WV3에서는 −0.026 dB·SAM +0.016°였고, WV2 RR2·FR_WALD SAM에 미달했습니다. 양쪽 쌍 흐림은 WV2 PSNR만 개선하고 SAM이 +0.703° 악화했습니다.

거친 쌍 전용은 양쪽 쌍 증강보다 WV3·WV2의 PSNR·SAM 모두 낫습니다. 다만 공식 기준 대비 WV3 손해를 없애지 못했고 WV2 SAM 감소는 매우 작습니다. 같은 실행기의 재학습 대조군 대비 WV2 +1.154 dB·SAM −0.303°이고 보조 항목도 통과하지만, 이 대조군 자체가 보관된 54번보다 WV2 −0.426 dB·SAM +0.297° 낮습니다. 재학습 대조군 대비 결과만으로 공식 기준 교체를 주장하지 않습니다.

## 5. 학습·평가 그래프

실제 history의 학습 MSE·validation PSNR·SAM과 공식 대조군 대비 평가 비교입니다. EMA는 기존 학습기의 기록으로만 보존하며 주 결과는 일반 best.pt입니다.

**WV3_BLUR_BOTH_s42**

![학습 곡선](results/WV3_BLUR_BOTH_s42/evaluation/learning.png)

![공식 대조군 비교](results/WV3_BLUR_BOTH_s42/evaluation/comparison.png)

**WV3_BLUR_COARSE_s42**

![학습 곡선](results/WV3_BLUR_COARSE_s42/evaluation/learning.png)

![공식 대조군 비교](results/WV3_BLUR_COARSE_s42/evaluation/comparison.png)

## 6. 결과 이미지

고정 장면 0·1·3·7·14·17에 대해 원본 크기 비교와 확대 비교를 모두 보존했습니다. 원본 형식은 LR MS·PAN·현재 기준 C2MS·후보·정답 및 공통 척도의 두 오차 지도입니다. 확대 형식은 후보 개선이 가장 큰 64×64 창을 4배 확대하므로 **후보에 유리하게 고른 창**입니다. 전체 장면 PSNR과 함께 봅니다. QB RGB는 [2,1,0], WV3·WV2는 [4,2,1]입니다. 71번은 WV3 학습 전이 실험으로 QB 학습·이미지를 생성하지 않았습니다.

**WV3_BLUR_BOTH_s42 · WV2 · 장면 0**

![원본 크기](results/WV3_BLUR_BOTH_s42/evaluation/WV2/full_scene_00.png)

![확대](results/WV3_BLUR_BOTH_s42/evaluation/WV2/compare_scene_00.png)

<details>
<summary>고정 나머지 다섯 장면과 전체 장면 PSNR</summary>

| 장면 | 기준 PSNR | 후보 PSNR |
|---:|---:|---:|
| 0 | 28.691 | 29.015 |
| 1 | 26.815 | 27.448 |
| 3 | 34.937 | 34.247 |
| 7 | 29.176 | 29.185 |
| 14 | 28.932 | 29.144 |
| 17 | 29.386 | 29.123 |

![장면 1 원본](results/WV3_BLUR_BOTH_s42/evaluation/WV2/full_scene_01.png)

![장면 1 확대](results/WV3_BLUR_BOTH_s42/evaluation/WV2/compare_scene_01.png)

![장면 3 원본](results/WV3_BLUR_BOTH_s42/evaluation/WV2/full_scene_03.png)

![장면 3 확대](results/WV3_BLUR_BOTH_s42/evaluation/WV2/compare_scene_03.png)

![장면 7 원본](results/WV3_BLUR_BOTH_s42/evaluation/WV2/full_scene_07.png)

![장면 7 확대](results/WV3_BLUR_BOTH_s42/evaluation/WV2/compare_scene_07.png)

![장면 14 원본](results/WV3_BLUR_BOTH_s42/evaluation/WV2/full_scene_14.png)

![장면 14 확대](results/WV3_BLUR_BOTH_s42/evaluation/WV2/compare_scene_14.png)

![장면 17 원본](results/WV3_BLUR_BOTH_s42/evaluation/WV2/full_scene_17.png)

![장면 17 확대](results/WV3_BLUR_BOTH_s42/evaluation/WV2/compare_scene_17.png)

</details>

**WV3_BLUR_BOTH_s42 · WV3 · 장면 0**

![원본 크기](results/WV3_BLUR_BOTH_s42/evaluation/WV3/full_scene_00.png)

![확대](results/WV3_BLUR_BOTH_s42/evaluation/WV3/compare_scene_00.png)

<details>
<summary>고정 나머지 다섯 장면과 전체 장면 PSNR</summary>

| 장면 | 기준 PSNR | 후보 PSNR |
|---:|---:|---:|
| 0 | 39.811 | 39.500 |
| 1 | 37.602 | 37.281 |
| 3 | 41.255 | 41.163 |
| 7 | 38.386 | 38.417 |
| 14 | 33.564 | 33.560 |
| 17 | 36.009 | 35.989 |

![장면 1 원본](results/WV3_BLUR_BOTH_s42/evaluation/WV3/full_scene_01.png)

![장면 1 확대](results/WV3_BLUR_BOTH_s42/evaluation/WV3/compare_scene_01.png)

![장면 3 원본](results/WV3_BLUR_BOTH_s42/evaluation/WV3/full_scene_03.png)

![장면 3 확대](results/WV3_BLUR_BOTH_s42/evaluation/WV3/compare_scene_03.png)

![장면 7 원본](results/WV3_BLUR_BOTH_s42/evaluation/WV3/full_scene_07.png)

![장면 7 확대](results/WV3_BLUR_BOTH_s42/evaluation/WV3/compare_scene_07.png)

![장면 14 원본](results/WV3_BLUR_BOTH_s42/evaluation/WV3/full_scene_14.png)

![장면 14 확대](results/WV3_BLUR_BOTH_s42/evaluation/WV3/compare_scene_14.png)

![장면 17 원본](results/WV3_BLUR_BOTH_s42/evaluation/WV3/full_scene_17.png)

![장면 17 확대](results/WV3_BLUR_BOTH_s42/evaluation/WV3/compare_scene_17.png)

</details>

**WV3_BLUR_COARSE_s42 · WV2 · 장면 0**

![원본 크기](results/WV3_BLUR_COARSE_s42/evaluation/WV2/full_scene_00.png)

![확대](results/WV3_BLUR_COARSE_s42/evaluation/WV2/compare_scene_00.png)

<details>
<summary>고정 나머지 다섯 장면과 전체 장면 PSNR</summary>

| 장면 | 기준 PSNR | 후보 PSNR |
|---:|---:|---:|
| 0 | 28.691 | 29.633 |
| 1 | 26.815 | 27.855 |
| 3 | 34.937 | 35.431 |
| 7 | 29.176 | 29.969 |
| 14 | 28.932 | 29.657 |
| 17 | 29.386 | 30.085 |

![장면 1 원본](results/WV3_BLUR_COARSE_s42/evaluation/WV2/full_scene_01.png)

![장면 1 확대](results/WV3_BLUR_COARSE_s42/evaluation/WV2/compare_scene_01.png)

![장면 3 원본](results/WV3_BLUR_COARSE_s42/evaluation/WV2/full_scene_03.png)

![장면 3 확대](results/WV3_BLUR_COARSE_s42/evaluation/WV2/compare_scene_03.png)

![장면 7 원본](results/WV3_BLUR_COARSE_s42/evaluation/WV2/full_scene_07.png)

![장면 7 확대](results/WV3_BLUR_COARSE_s42/evaluation/WV2/compare_scene_07.png)

![장면 14 원본](results/WV3_BLUR_COARSE_s42/evaluation/WV2/full_scene_14.png)

![장면 14 확대](results/WV3_BLUR_COARSE_s42/evaluation/WV2/compare_scene_14.png)

![장면 17 원본](results/WV3_BLUR_COARSE_s42/evaluation/WV2/full_scene_17.png)

![장면 17 확대](results/WV3_BLUR_COARSE_s42/evaluation/WV2/compare_scene_17.png)

</details>

**WV3_BLUR_COARSE_s42 · WV3 · 장면 0**

![원본 크기](results/WV3_BLUR_COARSE_s42/evaluation/WV3/full_scene_00.png)

![확대](results/WV3_BLUR_COARSE_s42/evaluation/WV3/compare_scene_00.png)

<details>
<summary>고정 나머지 다섯 장면과 전체 장면 PSNR</summary>

| 장면 | 기준 PSNR | 후보 PSNR |
|---:|---:|---:|
| 0 | 39.811 | 39.689 |
| 1 | 37.602 | 37.403 |
| 3 | 41.255 | 41.194 |
| 7 | 38.386 | 38.441 |
| 14 | 33.564 | 33.633 |
| 17 | 36.009 | 36.024 |

![장면 1 원본](results/WV3_BLUR_COARSE_s42/evaluation/WV3/full_scene_01.png)

![장면 1 확대](results/WV3_BLUR_COARSE_s42/evaluation/WV3/compare_scene_01.png)

![장면 3 원본](results/WV3_BLUR_COARSE_s42/evaluation/WV3/full_scene_03.png)

![장면 3 확대](results/WV3_BLUR_COARSE_s42/evaluation/WV3/compare_scene_03.png)

![장면 7 원본](results/WV3_BLUR_COARSE_s42/evaluation/WV3/full_scene_07.png)

![장면 7 확대](results/WV3_BLUR_COARSE_s42/evaluation/WV3/compare_scene_07.png)

![장면 14 원본](results/WV3_BLUR_COARSE_s42/evaluation/WV3/full_scene_14.png)

![장면 14 확대](results/WV3_BLUR_COARSE_s42/evaluation/WV3/compare_scene_14.png)

![장면 17 원본](results/WV3_BLUR_COARSE_s42/evaluation/WV3/full_scene_17.png)

![장면 17 확대](results/WV3_BLUR_COARSE_s42/evaluation/WV3/compare_scene_17.png)

</details>

## 7. 재현 자료

[설정](config.json) · [전체 수치 요약](results/summary.json) · [manifest](manifest.json)

학습 당시 봉인 설정·환경·캐시 manifest·상태·validation 선별 결과는 provenance/run_records에, 실제 best·best_ema는 checkpoints에, 학습 compact history·완료 기록·장면별 지표·그래프·이미지는 results에 보존합니다. latest·initial은 실험별 GitHub Release에 저장하며 manifest의 release_assets에 실제 크기·SHA-256·복구 URL을 기록합니다.

70번 학습 완료 후 validation 공통 모듈 검색 경로 누락으로 중단됐다가 PYTHONPATH를 지정한 외부 실행기로 검증부터 재개했습니다. 서버의 학습 당시 소스·가중치를 바꾸지 않았고 네 학습의 best·history 해시 불변을 확인했습니다. 원본 validation과 복구 근거는 각 provenance/recovery_20261030 및 70번 복구 기록에 있습니다. 재현 시 run_suite 실행 환경의 PYTHONPATH에 SSA-MRN을 지정합니다. 현행 validation.py에는 같은 경로 등록을 명시했습니다.

## 8. 개선 여부와 다음 단계

**부분 개선**입니다(seed42 한 번). 거친 쌍 전용 흐림은 WV2 RR +0.728 dB·SAM −0.006°였지만 WV3에서는 −0.026 dB·SAM +0.016°였고, WV2 RR2·FR_WALD SAM에 미달했습니다. 양쪽 쌍 흐림은 WV2 PSNR만 개선하고 SAM이 +0.703° 악화했습니다. 현재 기준 모델은 C2MS로 유지합니다. 원래 센서 PSNR·SAM 동시 개선과 WV2 전이 개선을 함께 달성하지 못했고 사전 관문도 전 센서에서 통과하지 못했습니다. 추가 시드·결합 실험은 시작하지 않습니다.
