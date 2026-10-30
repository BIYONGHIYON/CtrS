# 70. C2MS 원래 해상도 가지의 합성곱 참조 범위 확장

**연구 분야:** 성능 개선 · **실험자:** 이병현

**변경 내용:** C2MS 원래 해상도 가지의 블록 2·4에서 두 번째 3×3 합성곱 두 개만 dilation·padding 1에서 2로 바꿨습니다.

| 확인된 결과 | 상태 | 개선 여부 |
|---|:---:|:---:|
| dilation은 QB RR에서 공식 기준 대비 +0.009 dB·SAM −0.004°였지만 PSNR 잡음 문턱과 RR2 PSNR에 미달했습니다. WV3·WV2는 PSNR·SAM 모두 악화했습니다. | 평가&nbsp;완료<br>seed42 | 부분&nbsp;개선 |

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
| 공식 비교 | QB 53번, WV3 및 WV2 전이 54번, 같은 seed42 |
| 추가 비교 | 이번 실행기의 C2MS_R 재학습 대조군 |
| 장치 | RTX 3060 Ti, 한 번에 한 학습 |

C2MS_D는 `core.backbone.original[1].conv21`과 `[3].conv21`의 dilation/padding만 2로 바꿉니다. W2는 그대로이며 파라미터 수·초기 가중치·표본 순서를 맞췄습니다. dilation은 state_dict에 저장되지 않으므로 평가 모델 생성 때 실제 geometry를 확인했습니다.

## 3. 정량 결과

각 test는 20장 전체 평균입니다. Δ는 후보 − 대조군이며 PSNR은 양수, SAM·ERGAS는 음수가 좋습니다. **단일시드 PSNR 잡음 문턱은 0.06 dB**입니다. 공식 기준과 이번 재학습 대조군 비교를 구분합니다. QB RR2 공식 판정은 57번 고정값 PSNR 39.1148 dB·SAM 4.2410°를 유지합니다.

| 대상 | 군 | best epoch | RR PSNR / SAM | 공식 ΔPSNR / ΔSAM / ΔERGAS | 재학습 대조군 ΔPSNR / ΔSAM | PSNR 상승 장면 |
|---|---|---:|---|---|---|---:|
| QB | C2MS_D | 143 | 38.2827 / 4.5730° | +0.0090 / -0.0040° / -0.0007 | +0.0223 / -0.0075° | 10/20 |
| QB | C2MS_R | 149 | 38.2604 / 4.5805° | -0.0134 / +0.0034° / +0.0064 | 대조군 자체 | 9/20 |
| WV2 | C2MS_D | 145 | 28.8388 / 6.2449° | -0.6845 / +0.3908° / +0.4060 | -0.2586 / +0.0939° | 0/20 |
| WV3 | C2MS_D | 145 | 38.4376 / 3.1101° | -0.0551 / +0.0199° / +0.0162 | -0.0851 / +0.0160° | 6/20 |
| WV2 | C2MS_R | 150 | 29.0974 / 6.1510° | -0.4259 / +0.2969° / +0.2587 | 대조군 자체 | 1/20 |
| WV3 | C2MS_R | 150 | 38.5227 / 3.0941° | +0.0300 / +0.0039° / -0.0088 | 대조군 자체 | 13/20 |

| 대상 | 후보 | RR2 ΔPSNR / ΔSAM | FR_WALD ΔPSNR / ΔSAM | FR 관측 RMSE 비율 | 공식 v2.3 |
|---|---|---|---|---:|---|
| QB | C2MS_D | -0.1357 / +0.0399° | +0.0195 / -0.0204° | 0.9613 | 부분 개선 |
| WV2 | C2MS_D | -0.8441 / +0.4295° | -0.8499 / +0.4396° | 1.0401 | 개선 없음 |
| WV3 | C2MS_D | +0.3773 / -0.0864° | -0.0163 / +0.0108° | 0.9536 | 개선 없음 |

**학습하지 않은 센서 WV2는 WV3 가중치를 그대로 적용한 전이입니다.** FR에는 정답이 없으며 FR_WALD는 더 열화한 별도 정답 기반 시험입니다. RR2는 RR 장면을 추가 4배 Wald 열화하는 시험이며 2배 거친 학습 쌍과 구분합니다. QNR·Dλ·Ds는 원시 JSON에 참고값으로 보존하며 판정에 쓰지 않습니다.

**사전 고정 validation 관문**

| 센서 | 후보 | clean ΔPSNR / ΔSAM | 고정 blur 평균 ΔPSNR / ΔSAM | 관문 |
|---|---|---|---|---|
| QB | C2MS_D | +0.0078 / -0.0065° | +0.0090 / -0.0068° | 통과 |
| WV3 | C2MS_D | -0.0273 / +0.0054° | -0.0608 / +0.0042° | 미달 |

blur validation은 σ 0.4·0.8·1.2의 산술평균이며 전이의 대리 진단입니다. 70번은 clean PSNR·SAM 동시 개선을 요구했고 QB만 통과했습니다. 71번은 clean PSNR ≥ −0.03 dB·SAM ≤ +0.02° 및 blur 평균 PSNR 상승·SAM 감소를 요구했으나 두 군 모두 clean 항목에서 미달했습니다. test를 보고 후보나 강도를 고르지 않았습니다.


**실제 업데이트 기록** (AMP overflow로 건너뛴 횟수를 포함합니다. 센서별 표본 순서 해시는 모든 군에서 같았습니다.)

| 실행 | 시도 | 적용 | 건너뜀 |
|---|---:|---:|---:|
| QB_C2MS_D_s42 | 80400 | 80380 | 20 |
| QB_C2MS_R_s42 | 80400 | 80381 | 19 |
| WV3_C2MS_D_s42 | 45600 | 45595 | 5 |
| WV3_C2MS_R_s42 | 45600 | 45596 | 4 |

## 4. 기준선 대비 차이와 해석

dilation은 QB RR에서 공식 기준 대비 +0.009 dB·SAM −0.004°였지만 PSNR 잡음 문턱과 RR2 PSNR에 미달했습니다. WV3·WV2는 PSNR·SAM 모두 악화했습니다.

같은 실행기의 대조군 대비 dilation은 QB +0.022 dB·SAM −0.0075°, WV3 −0.085 dB·+0.016°, WV2 −0.259 dB·+0.094°입니다. QB의 작은 이득은 잡음 문턱 미만이며 RR2가 악화했습니다. 원본 42번의 이득을 C2MS에서 재현했다는 근거는 없습니다. W2와 기능이 겹쳤다는 설명은 가설입니다.

## 5. 학습·평가 그래프

실제 history의 학습 MSE·validation PSNR·SAM과 공식 대조군 대비 평가 비교입니다. EMA는 기존 학습기의 기록으로만 보존하며 주 결과는 일반 best.pt입니다.

**QB_C2MS_D_s42**

![학습 곡선](results/QB_C2MS_D_s42/evaluation/learning.png)

![공식 대조군 비교](results/QB_C2MS_D_s42/evaluation/comparison.png)

**QB_C2MS_R_s42**

![학습 곡선](results/QB_C2MS_R_s42/evaluation/learning.png)

![공식 대조군 비교](results/QB_C2MS_R_s42/evaluation/comparison.png)

**WV3_C2MS_D_s42**

![학습 곡선](results/WV3_C2MS_D_s42/evaluation/learning.png)

![공식 대조군 비교](results/WV3_C2MS_D_s42/evaluation/comparison.png)

**WV3_C2MS_R_s42**

![학습 곡선](results/WV3_C2MS_R_s42/evaluation/learning.png)

![공식 대조군 비교](results/WV3_C2MS_R_s42/evaluation/comparison.png)

## 6. 결과 이미지

고정 장면 0·1·3·7·14·17에 대해 원본 크기 비교와 확대 비교를 모두 보존했습니다. 원본 형식은 LR MS·PAN·현재 기준 C2MS·후보·정답 및 공통 척도의 두 오차 지도입니다. 확대 형식은 후보 개선이 가장 큰 64×64 창을 4배 확대하므로 **후보에 유리하게 고른 창**입니다. 전체 장면 PSNR과 함께 봅니다. QB RGB는 [2,1,0], WV3·WV2는 [4,2,1]입니다.

**QB_C2MS_D_s42 · QB · 장면 0**

![원본 크기](results/QB_C2MS_D_s42/evaluation/QB/full_scene_00.png)

![확대](results/QB_C2MS_D_s42/evaluation/QB/compare_scene_00.png)

<details>
<summary>고정 나머지 다섯 장면과 전체 장면 PSNR</summary>

| 장면 | 기준 PSNR | 후보 PSNR |
|---:|---:|---:|
| 0 | 42.812 | 42.752 |
| 1 | 43.179 | 43.164 |
| 3 | 37.964 | 38.137 |
| 7 | 39.405 | 39.450 |
| 14 | 38.473 | 38.439 |
| 17 | 39.396 | 39.445 |

![장면 1 원본](results/QB_C2MS_D_s42/evaluation/QB/full_scene_01.png)

![장면 1 확대](results/QB_C2MS_D_s42/evaluation/QB/compare_scene_01.png)

![장면 3 원본](results/QB_C2MS_D_s42/evaluation/QB/full_scene_03.png)

![장면 3 확대](results/QB_C2MS_D_s42/evaluation/QB/compare_scene_03.png)

![장면 7 원본](results/QB_C2MS_D_s42/evaluation/QB/full_scene_07.png)

![장면 7 확대](results/QB_C2MS_D_s42/evaluation/QB/compare_scene_07.png)

![장면 14 원본](results/QB_C2MS_D_s42/evaluation/QB/full_scene_14.png)

![장면 14 확대](results/QB_C2MS_D_s42/evaluation/QB/compare_scene_14.png)

![장면 17 원본](results/QB_C2MS_D_s42/evaluation/QB/full_scene_17.png)

![장면 17 확대](results/QB_C2MS_D_s42/evaluation/QB/compare_scene_17.png)

</details>

**WV3_C2MS_D_s42 · WV2 · 장면 0**

![원본 크기](results/WV3_C2MS_D_s42/evaluation/WV2/full_scene_00.png)

![확대](results/WV3_C2MS_D_s42/evaluation/WV2/compare_scene_00.png)

<details>
<summary>고정 나머지 다섯 장면과 전체 장면 PSNR</summary>

| 장면 | 기준 PSNR | 후보 PSNR |
|---:|---:|---:|
| 0 | 28.691 | 27.890 |
| 1 | 26.815 | 26.204 |
| 3 | 34.937 | 34.254 |
| 7 | 29.176 | 28.446 |
| 14 | 28.932 | 28.056 |
| 17 | 29.386 | 28.633 |

![장면 1 원본](results/WV3_C2MS_D_s42/evaluation/WV2/full_scene_01.png)

![장면 1 확대](results/WV3_C2MS_D_s42/evaluation/WV2/compare_scene_01.png)

![장면 3 원본](results/WV3_C2MS_D_s42/evaluation/WV2/full_scene_03.png)

![장면 3 확대](results/WV3_C2MS_D_s42/evaluation/WV2/compare_scene_03.png)

![장면 7 원본](results/WV3_C2MS_D_s42/evaluation/WV2/full_scene_07.png)

![장면 7 확대](results/WV3_C2MS_D_s42/evaluation/WV2/compare_scene_07.png)

![장면 14 원본](results/WV3_C2MS_D_s42/evaluation/WV2/full_scene_14.png)

![장면 14 확대](results/WV3_C2MS_D_s42/evaluation/WV2/compare_scene_14.png)

![장면 17 원본](results/WV3_C2MS_D_s42/evaluation/WV2/full_scene_17.png)

![장면 17 확대](results/WV3_C2MS_D_s42/evaluation/WV2/compare_scene_17.png)

</details>

**WV3_C2MS_D_s42 · WV3 · 장면 0**

![원본 크기](results/WV3_C2MS_D_s42/evaluation/WV3/full_scene_00.png)

![확대](results/WV3_C2MS_D_s42/evaluation/WV3/compare_scene_00.png)

<details>
<summary>고정 나머지 다섯 장면과 전체 장면 PSNR</summary>

| 장면 | 기준 PSNR | 후보 PSNR |
|---:|---:|---:|
| 0 | 39.811 | 39.681 |
| 1 | 37.602 | 37.422 |
| 3 | 41.255 | 41.098 |
| 7 | 38.386 | 38.379 |
| 14 | 33.564 | 33.572 |
| 17 | 36.009 | 36.016 |

![장면 1 원본](results/WV3_C2MS_D_s42/evaluation/WV3/full_scene_01.png)

![장면 1 확대](results/WV3_C2MS_D_s42/evaluation/WV3/compare_scene_01.png)

![장면 3 원본](results/WV3_C2MS_D_s42/evaluation/WV3/full_scene_03.png)

![장면 3 확대](results/WV3_C2MS_D_s42/evaluation/WV3/compare_scene_03.png)

![장면 7 원본](results/WV3_C2MS_D_s42/evaluation/WV3/full_scene_07.png)

![장면 7 확대](results/WV3_C2MS_D_s42/evaluation/WV3/compare_scene_07.png)

![장면 14 원본](results/WV3_C2MS_D_s42/evaluation/WV3/full_scene_14.png)

![장면 14 확대](results/WV3_C2MS_D_s42/evaluation/WV3/compare_scene_14.png)

![장면 17 원본](results/WV3_C2MS_D_s42/evaluation/WV3/full_scene_17.png)

![장면 17 확대](results/WV3_C2MS_D_s42/evaluation/WV3/compare_scene_17.png)

</details>

## 7. 재현 자료

[설정](config.json) · [전체 수치 요약](results/summary.json) · [manifest](manifest.json)

학습 당시 봉인 설정·환경·캐시 manifest·상태·validation 선별 결과는 provenance/run_records에, 실제 best·best_ema는 checkpoints에, 학습 compact history·완료 기록·장면별 지표·그래프·이미지는 results에 보존합니다. latest·initial은 실험별 GitHub Release에 저장하며 manifest의 release_assets에 실제 크기·SHA-256·복구 URL을 기록합니다.

70번 학습 완료 후 validation 공통 모듈 검색 경로 누락으로 중단됐다가 PYTHONPATH를 지정한 외부 실행기로 검증부터 재개했습니다. 서버의 학습 당시 소스·가중치를 바꾸지 않았고 네 학습의 best·history 해시 불변을 확인했습니다. 원본 validation과 복구 근거는 각 provenance/recovery_20261030 및 70번 복구 기록에 있습니다. 재현 시 run_suite 실행 환경의 PYTHONPATH에 SSA-MRN을 지정합니다. 현행 validation.py에는 같은 경로 등록을 명시했습니다.

## 8. 개선 여부와 다음 단계

**부분 개선**입니다(seed42 한 번). dilation은 QB RR에서 공식 기준 대비 +0.009 dB·SAM −0.004°였지만 PSNR 잡음 문턱과 RR2 PSNR에 미달했습니다. WV3·WV2는 PSNR·SAM 모두 악화했습니다. 현재 기준 모델은 C2MS로 유지합니다. 원래 센서 PSNR·SAM 동시 개선과 WV2 전이 개선을 함께 달성하지 못했고 사전 관문도 전 센서에서 통과하지 못했습니다. 추가 시드·결합 실험은 시작하지 않습니다.
