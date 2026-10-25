# 성능 59. 기준 모델 C2MS의 cosine 학습 일정 연장

연구 분야: 기존 성능 개선 · 실제 실험자: 이병현

C2MS 구조와 축척 혼합 손실은 그대로 두고, 학습 일정만 150에폭에서 300에폭 cosine으로 늘려 처음부터 다시 학습했습니다. 성능 58번의 분광 손실은 쓰지 않았습니다. QB·GF2·WV3 seed 42를 학습하고 WV3 모델은 WV2에 전이 평가했습니다.

| 확인된 결과 | 상태 | 개선 여부 |
| --- | :---: | :---: |
| 같은 시드의 C2MS 150에폭 대비 RR PSNR이 GF2 +0.205 dB, WV3 +0.366 dB(모두 20/20장)로 v2.3 개선입니다. QB는 −0.284 dB(1/20장), WV2 전이는 −1.760 dB(0/20장)로 개선 없음입니다. | 평가&nbsp;완료<br>seed42 | 부분&nbsp;개선 |

## 1. 실험 목적과 현재 상태

C2MS의 validation 곡선이 150에폭 끝까지 조금씩 오르고 있어서, 학습을 더 길게 하면 test 수치도 오르는지 확인했습니다. 2026-10-25 08:34 KST에 QB → GF2 → WV3 순서로 각 300에폭을 직렬 학습했고, 2026-10-26 03:25 KST에 세 센서의 학습과 test 평가를 마쳤습니다. 각 센서는 새 초기화에서 시작한 독립 모델입니다. WV2는 학습 분할이 없어 WV3 학습 모델로 전이 평가했습니다. 기준 모델은 C2MS(150에폭)로 유지합니다.

## 2. 변경한 구조와 비교 조건

| 조건 | 설정 |
| --- | --- |
| 구조·입력·출력 | 기존 C2MS(SSA-MRN K4 + W2 병렬 가지), PAN·LMS·MS → 복원 MS, 파라미터 수 같음 |
| 손실 | 원래 축척 MSE + 2배 거친 쌍 MSE(가중치 1.0), 분광 손실 없음 |
| 바꾼 것 | 학습 일정만 150에폭 → 300에폭, 새 초기화 |
| optimizer·학습률 | Adam, cosine 0.0001 → 0.000001(300에폭에 맞춤) |
| 데이터·시드 | 기존 train/validation 분할·정규화·센서 연산 유지, seed 42 |
| 배치·정밀도 | batch 32, FP16 AMP 학습, 손실·validation·평가 FP32, TF32 끔 |
| 선택 | 원래 축척 validation MSE 최소(0에폭 포함). test는 선택에 쓰지 않음 |
| 대조군 | 같은 seed 42의 C2MS 150에폭 best(QB는 성능 53번, GF2·WV3는 성능 54번) |
| 평가 | test 20장 전체, RR·RR2·FR_WALD·FR 관측 일치, 판정 기준 v2.3(단일 시드 잡음 문턱 0.06 dB) |

150에폭 가중치에 150에폭을 덧붙인 실험이 아닙니다. 학습률 일정이 첫 에폭 이후부터 달라지므로, 이 비교는 "추가 150에폭"이 아니라 "300에폭에 맞춘 전체 일정"의 효과입니다.

## 3. 정량 결과

C2MS 300에폭 − C2MS 150에폭, seed 42, test 20장 평균입니다.

| 데이터셋 | best 에폭 150 → 300 | RR PSNR | RR SAM | RR ERGAS | RR2 PSNR | FR_WALD PSNR | FR 관측 오차 | 장면 PSNR·SAM 우세 | 판정 |
|---|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| QB | 149 → 198 | −0.284 dB | +0.015° | +0.119 | −0.385 dB | −0.257 dB | 1.04배 | 1/20 · 3/20 | 개선&nbsp;없음 |
| GF2 | 148 → 283 | +0.205 dB | −0.023° | −0.018 | +0.032 dB | +0.185 dB | 0.96배 | 20/20 · 20/20 | 개선 |
| WV3 | 148 → 255 | +0.366 dB | −0.107° | −0.086 | +0.991 dB | +0.216 dB | 0.83배 | 20/20 · 20/20 | 개선 |
| WV2 전이 | 148 → 255 | −1.760 dB | +0.948° | +1.045 | −2.739 dB | −1.517 dB | 1.01배 | 0/20 · 0/20 | 개선&nbsp;없음 |

절대 수치(RR PSNR / SAM / ERGAS)는 다음과 같습니다.

| 데이터셋 | C2MS 150에폭 | C2MS 300에폭 |
|---|---|---|
| QB | 38.274 / 4.577 / 3.723 | 37.990 / 4.592 / 3.842 |
| GF2 | 48.507 / 0.797 / 0.715 | 48.713 / 0.774 / 0.697 |
| WV3 | 38.493 / 3.090 / 2.267 | 38.858 / 2.983 / 2.181 |
| WV2 전이 | 29.523 / 5.854 / 4.537 | 27.763 / 6.803 / 5.582 |

FR QNR(참고)은 QB 0.826 → 0.818, GF2 0.939 → 0.944, WV3 0.953 → 0.930, WV2 0.925 → 0.807입니다. 이 값은 판정에 쓰지 않습니다.

validation(원래 축척, best 기준)은 세 센서 모두 올랐습니다. QB 40.283 → 40.378 dB, GF2 44.340 → 44.651 dB, WV3 38.034 → 38.481 dB입니다.

WV2 전이에서 무엇이 달라졌는지 성능 54번과 같은 방법으로 측정했습니다([transfer_detail.json](results/transfer_detail.json)).

| 정답 대비 (test 20장) | WV3 150 | WV3 300 | WV2 150 | WV2 300 |
|---|---:|---:|---:|---:|
| 선명도 (출력 고주파 세기 ÷ 정답) | 0.880 | 0.876 | 0.456 | **0.651** |
| 넣은 세부와 실제 세부의 상관 | 0.948 | 0.952 | 0.749 | **0.619** |
| 넣은 세부의 세기 ÷ 실제 세부 | 0.956 | 0.956 | 0.631 | 0.832 |
| 밴드별 평균 밝기 치우침 (가장 큰 밴드) | 0.08% | 0.10% | 3.0% | 2.1% |

## 4. 기준선 대비 차이와 해석

- **GF2·WV3:** 300에폭이 세 RR 지표와 모든 보조 판정을 통과했고 20장 모두에서 좋아졌습니다. 학습 데이터가 적은 WV3(9,714쌍)에서 이득이 가장 컸습니다. 150에폭 일정이 이 두 센서에는 짧았던 것으로 보입니다.
- **QB:** validation은 +0.096 dB 올랐지만 test는 −0.284 dB로 거의 모든 장면(19/20)에서 내려갔습니다. validation 곡선은 198에폭 이후 평평했고(300에폭 40.373 dB), validation 향상이 test로 이어지지 않았습니다. 단일 시드라서 학습 실행 간 변동인지 QB test 장면과 validation의 차이인지는 이 실험만으로 구분할 수 없습니다.
- **WV2 전이:** 가장 크게 나빠졌습니다. 300에폭 모델은 WV2에서 세부를 더 세게 넣어(선명도 0.46 → 0.65) 더 선명해졌지만, 넣은 세부와 실제 세부의 상관이 0.75 → 0.62로 떨어졌습니다. 학습한 WV3에서는 두 값이 거의 같습니다. 학습 센서의 세부 주입 방식에 더 깊게 맞춰져, 처음 보는 센서에서는 틀린 세부를 더 강하게 넣은 것으로 해석합니다. 색 치우침은 오히려 줄었으므로 하락의 주된 원인은 세부입니다.
- 학습 시간이 두 배인 데 비해 이득은 센서에 따라 엇갈립니다. 기준 모델(C2MS 150에폭)은 바꾸지 않습니다.

## 5. 학습·평가 그래프

학습 MSE(원래 축척)와 validation PSNR·SAM 곡선입니다(best는 validation MSE로 선택).

![QB 학습 곡선](results/C2MS_QB_s42/evaluation/learning.png)

![GF2 학습 곡선](results/C2MS_GF2_s42/evaluation/learning.png)

![WV3 학습 곡선](results/C2MS_WV3_s42/evaluation/learning.png)

test PSNR 차이(300에폭 − 150에폭)입니다.

![QB test 비교](results/C2MS_QB_s42/evaluation/comparison.png)

![GF2 test 비교](results/C2MS_GF2_s42/evaluation/comparison.png)

![WV3·WV2 test 비교](results/C2MS_WV3_s42/evaluation/comparison.png)

## 6. 결과 이미지

고정 RR 장면 0·1·3·7·14·17, seed 42입니다. 장면 번호는 QB에서 정답 영상만 보고 정한 것이며, 다른 센서의 같은 번호는 QB와 다른 장소입니다.

- **원본 크기 비교:** 위 줄은 LR MS(최근접 확대) · PAN 입력 · 현재 기준 C2MS(150에폭) · C2MS(300에폭) · 정답, 아래 줄은 두 모델의 |예측 − 정답| 지도입니다(같은 색 척도).
- **확대 비교:** 300에폭의 개선(150 오차 − 300 오차)이 가장 큰 64×64 창을 4배 확대한 정답 전체(노란 상자) · 150에폭 · 300에폭 · 정답 · 150 오차 · 300 오차입니다. 300에폭에 유리하게 고른 창이므로 전체 차이는 장면 PSNR로 확인합니다. QB·WV2처럼 장면 전체가 나빠진 경우에도 창 안은 300에폭이 나을 수 있습니다.

| 장면 PSNR 150 → 300 (dB) | 0 | 1 | 3 | 7 | 14 | 17 |
|---|---:|---:|---:|---:|---:|---:|
| QB | 42.81 → 42.54 | 43.18 → 43.18 | 37.96 → 37.85 | 39.41 → 38.99 | 38.47 → 38.22 | 39.40 → 39.36 |
| GF2 | 47.62 → 47.92 | 49.16 → 49.46 | 49.67 → 49.79 | 50.65 → 50.81 | 47.87 → 47.98 | 49.23 → 49.40 |
| WV3 | 39.81 → 40.37 | 37.60 → 38.22 | 41.25 → 41.93 | 38.39 → 38.72 | 33.56 → 33.58 | 36.01 → 36.23 |
| WV2 전이 | 28.69 → 27.10 | 26.82 → 25.09 | 34.94 → 33.60 | 29.18 → 27.51 | 28.93 → 27.24 | 29.39 → 27.15 |

장면별 PSNR·확대 창 위치는 각 `results/C2MS_<센서>_s42/evaluation/<데이터셋>/result_images.json`에 있습니다.

### QB 장면 7

![QB 원본 크기 비교 · 장면 7](results/C2MS_QB_s42/evaluation/QB/full_scene_07.png)

![QB 확대 비교 · 장면 7](results/C2MS_QB_s42/evaluation/QB/compare_scene_07.png)

<details>
<summary><b>QB 나머지 5장면</b> (0·1·3·14·17)</summary>

![QB 원본 크기 비교 · 장면 0](results/C2MS_QB_s42/evaluation/QB/full_scene_00.png)
![QB 확대 비교 · 장면 0](results/C2MS_QB_s42/evaluation/QB/compare_scene_00.png)
![QB 원본 크기 비교 · 장면 1](results/C2MS_QB_s42/evaluation/QB/full_scene_01.png)
![QB 확대 비교 · 장면 1](results/C2MS_QB_s42/evaluation/QB/compare_scene_01.png)
![QB 원본 크기 비교 · 장면 3](results/C2MS_QB_s42/evaluation/QB/full_scene_03.png)
![QB 확대 비교 · 장면 3](results/C2MS_QB_s42/evaluation/QB/compare_scene_03.png)
![QB 원본 크기 비교 · 장면 14](results/C2MS_QB_s42/evaluation/QB/full_scene_14.png)
![QB 확대 비교 · 장면 14](results/C2MS_QB_s42/evaluation/QB/compare_scene_14.png)
![QB 원본 크기 비교 · 장면 17](results/C2MS_QB_s42/evaluation/QB/full_scene_17.png)
![QB 확대 비교 · 장면 17](results/C2MS_QB_s42/evaluation/QB/compare_scene_17.png)

</details>

### GF2 장면 7

![GF2 원본 크기 비교 · 장면 7](results/C2MS_GF2_s42/evaluation/GF2/full_scene_07.png)

![GF2 확대 비교 · 장면 7](results/C2MS_GF2_s42/evaluation/GF2/compare_scene_07.png)

<details>
<summary><b>GF2 나머지 5장면</b> (0·1·3·14·17)</summary>

![GF2 원본 크기 비교 · 장면 0](results/C2MS_GF2_s42/evaluation/GF2/full_scene_00.png)
![GF2 확대 비교 · 장면 0](results/C2MS_GF2_s42/evaluation/GF2/compare_scene_00.png)
![GF2 원본 크기 비교 · 장면 1](results/C2MS_GF2_s42/evaluation/GF2/full_scene_01.png)
![GF2 확대 비교 · 장면 1](results/C2MS_GF2_s42/evaluation/GF2/compare_scene_01.png)
![GF2 원본 크기 비교 · 장면 3](results/C2MS_GF2_s42/evaluation/GF2/full_scene_03.png)
![GF2 확대 비교 · 장면 3](results/C2MS_GF2_s42/evaluation/GF2/compare_scene_03.png)
![GF2 원본 크기 비교 · 장면 14](results/C2MS_GF2_s42/evaluation/GF2/full_scene_14.png)
![GF2 확대 비교 · 장면 14](results/C2MS_GF2_s42/evaluation/GF2/compare_scene_14.png)
![GF2 원본 크기 비교 · 장면 17](results/C2MS_GF2_s42/evaluation/GF2/full_scene_17.png)
![GF2 확대 비교 · 장면 17](results/C2MS_GF2_s42/evaluation/GF2/compare_scene_17.png)

</details>

### WV3 장면 7

![WV3 원본 크기 비교 · 장면 7](results/C2MS_WV3_s42/evaluation/WV3/full_scene_07.png)

![WV3 확대 비교 · 장면 7](results/C2MS_WV3_s42/evaluation/WV3/compare_scene_07.png)

<details>
<summary><b>WV3 나머지 5장면</b> (0·1·3·14·17)</summary>

![WV3 원본 크기 비교 · 장면 0](results/C2MS_WV3_s42/evaluation/WV3/full_scene_00.png)
![WV3 확대 비교 · 장면 0](results/C2MS_WV3_s42/evaluation/WV3/compare_scene_00.png)
![WV3 원본 크기 비교 · 장면 1](results/C2MS_WV3_s42/evaluation/WV3/full_scene_01.png)
![WV3 확대 비교 · 장면 1](results/C2MS_WV3_s42/evaluation/WV3/compare_scene_01.png)
![WV3 원본 크기 비교 · 장면 3](results/C2MS_WV3_s42/evaluation/WV3/full_scene_03.png)
![WV3 확대 비교 · 장면 3](results/C2MS_WV3_s42/evaluation/WV3/compare_scene_03.png)
![WV3 원본 크기 비교 · 장면 14](results/C2MS_WV3_s42/evaluation/WV3/full_scene_14.png)
![WV3 확대 비교 · 장면 14](results/C2MS_WV3_s42/evaluation/WV3/compare_scene_14.png)
![WV3 원본 크기 비교 · 장면 17](results/C2MS_WV3_s42/evaluation/WV3/full_scene_17.png)
![WV3 확대 비교 · 장면 17](results/C2MS_WV3_s42/evaluation/WV3/compare_scene_17.png)

</details>

### WV2 전이 장면 7

![WV2 원본 크기 비교 · 장면 7](results/C2MS_WV3_s42/evaluation/WV2/full_scene_07.png)

![WV2 확대 비교 · 장면 7](results/C2MS_WV3_s42/evaluation/WV2/compare_scene_07.png)

<details>
<summary><b>WV2 나머지 5장면</b> (0·1·3·14·17)</summary>

![WV2 원본 크기 비교 · 장면 0](results/C2MS_WV3_s42/evaluation/WV2/full_scene_00.png)
![WV2 확대 비교 · 장면 0](results/C2MS_WV3_s42/evaluation/WV2/compare_scene_00.png)
![WV2 원본 크기 비교 · 장면 1](results/C2MS_WV3_s42/evaluation/WV2/full_scene_01.png)
![WV2 확대 비교 · 장면 1](results/C2MS_WV3_s42/evaluation/WV2/compare_scene_01.png)
![WV2 원본 크기 비교 · 장면 3](results/C2MS_WV3_s42/evaluation/WV2/full_scene_03.png)
![WV2 확대 비교 · 장면 3](results/C2MS_WV3_s42/evaluation/WV2/compare_scene_03.png)
![WV2 원본 크기 비교 · 장면 14](results/C2MS_WV3_s42/evaluation/WV2/full_scene_14.png)
![WV2 확대 비교 · 장면 14](results/C2MS_WV3_s42/evaluation/WV2/compare_scene_14.png)
![WV2 원본 크기 비교 · 장면 17](results/C2MS_WV3_s42/evaluation/WV2/full_scene_17.png)
![WV2 확대 비교 · 장면 17](results/C2MS_WV3_s42/evaluation/WV2/compare_scene_17.png)

</details>

FR에는 정답이 없으므로 FR 이미지는 만들지 않았습니다.

## 7. 재현 자료

- [설정](config.json), [모델](model.py), [센서 연산](prepare.py), [학습](train.py), [실행 제어](run.py), [평가](evaluate.py)
- 학습 당시 소스·설정은 [provenance/code](provenance/code/)에, 시작 시점 검증 기록은 [provenance/startup](provenance/startup/)에, 학습 종료까지의 실행 기록(실행 설정·데이터 manifest·환경·사전 검증·완료 기록)은 [provenance/run_records](provenance/run_records/)에 보존했습니다. 세 센서 모두 AMP 2업데이트 저장·재개 후 최대 가중치 차이는 0이었습니다.
- WV2 선명도·세부 분석은 학습·평가가 끝난 뒤 [provenance/detail59.py](provenance/detail59.py)로 계산했습니다.
- 가중치: `checkpoints/C2MS_<센서>_s42/best.pt`(Git). 재개용 `latest.pt`와 초기 `initial.pt`는 GitHub Release `weights-experiments-performance-59`에 있습니다.
- 센서별 test 수치·장면별 값·판정: `results/C2MS_<센서>_s42/evaluation/<데이터셋>_v2_3.json`, 에폭별 기록: `results/C2MS_<센서>_s42/history.json`

```bash
python SSA-MRN/experiments/performance/59_extended_cosine_training/run.py --mode prepare
python SSA-MRN/experiments/performance/59_extended_cosine_training/run.py --mode optimize
python SSA-MRN/experiments/performance/59_extended_cosine_training/run.py --mode run
python SSA-MRN/common/verify.py --experiment 59 --research-track performance --releases
```

## 8. 개선 여부와 다음 단계

**부분 개선**입니다. GF2·WV3는 판정 기준 v2.3의 모든 항목을 통과한 개선이고, QB와 WV2 전이는 개선 없음입니다. 단일 시드이며 기준 모델 교체 조건(3시드 개선)에 해당하지 않으므로 기준 모델은 C2MS(150에폭)로 유지합니다.

- QB 하락이 시드 변동인지 확인하려면 seed 43·44의 300에폭이 필요합니다(QB 1시드 약 6.5시간).
- 300에폭은 학습 센서의 수치를 올리는 대신 다른 센서로의 전이를 크게 해쳤습니다. 300에폭을 쓰려면 전이 성능을 지키는 방법(예: 학습 후반의 세부 주입 세기 제어)이 함께 필요합니다.
