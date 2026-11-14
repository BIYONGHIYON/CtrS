# 기준 모델 C2MS-SP 결과 이미지

원본 SSA-MRN, 기준 모델 C2MS-SP([성능 81번](../experiments/performance/81_l1_spectral_loss_projection/README.md)), 경량판 C2MS-SP-Lite([성능 83번](../experiments/performance/83_factorized_c2ms_finetuning/README.md))를 네 데이터셋의 RR test 고정 6장면(0-based 0·1·3·7·14·17, seed 42)에서 비교한 이미지입니다. 장면 번호는 QB에서 정답 영상만 보고 풍경 종류로 고른 표준 번호이며 다른 데이터셋에도 같은 번호를 썼습니다(결과를 보고 고르지 않았습니다). 장면별 정답 1-99% 공통 대비로 표시했고 수치는 원래 값으로 계산했습니다.

- **모델:** 원본은 발표된 방식 그대로(QB 성능 43번, GF2·WV3 성능 54번 C0 가중치)입니다. C2MS-SP는 C2MS 구조를 L1·분광 손실로 학습한 성능 65번 가중치에 관측 일치 사영(공액 기울기법 10회)을 붙였고, C2MS-SP-Lite는 그 가중치를 분해·미세조정한 성능 83번 K2에 같은 사영을 붙였습니다. 코드는 `common/baseline_models.py`, 이미지는 `common/baseline_images.py`(`result_images.render_multi`)로 만들었습니다.
- **원본 크기 비교:** 위 줄은 LR MS(최근접 확대) · PAN 입력 · 원본 · C2MS-SP · C2MS-SP-Lite · 정답, 아래 줄은 세 모델의 |예측 − 정답| 지도입니다. 오차 지도는 원본 오차의 99퍼센타일을 공통 색 척도로 쓰며 밝을수록 오차가 큽니다.
- **확대 비교:** C2MS-SP의 개선(원본 오차 − C2MS-SP 오차)이 가장 큰 64×64 영역을 4배 확대했습니다. 왼쪽부터 정답 전체(노란 상자) · 원본 · C2MS-SP · C2MS-SP-Lite · 정답 · 세 모델의 오차입니다. C2MS-SP에 유리하게 고른 영역이므로 전체 차이는 표의 장면 PSNR로 확인합니다.
- 3시드 평균 수치는 [루트 README](../../README.md#원본-대비-개선)와 [baseline_metrics.json](images/c2ms_sp/baseline_metrics.json)에 있습니다. 이전 기준 모델 C2MS의 이미지는 [C2MS 결과 이미지](c2ms_result_images.md)에 기록으로 남겼습니다.

[QB](#qb) · [GF2](#gf2) · [WV3](#wv3) · [WV2 전이](#wv2) · [루트 README](../../README.md)

<a id="qb"></a>

## QB

RGB 밴드 [2,1,0]. QB로 학습·평가했습니다.

| 장면 | 풍경 | 장면 PSNR 원본 / SP / SP-Lite | 확대 영역 PSNR 원본 / SP / SP-Lite |
|---:|---|---|---|
| 0 | 바다·항구 방파제 | 41.45 / 43.04 / 43.03 dB (SP +1.59) | 35.13 / 37.51 / 37.53 dB |
| 1 | 부두·곡선 방파제·수목 | 42.44 / 43.31 / 43.30 dB (SP +0.87) | 39.45 / 40.82 / 40.87 dB |
| 3 | 요트 마리나·운동장 | 36.22 / 38.10 / 38.05 dB (SP +1.88) | 33.60 / 35.95 / 35.93 dB |
| 7 | 숲 공원 | 38.83 / 39.71 / 39.69 dB (SP +0.87) | 35.90 / 37.05 / 37.05 dB |
| 14 | 언덕 공원·곡선 주택가 | 37.62 / 38.56 / 38.53 dB (SP +0.94) | 35.21 / 36.74 / 36.72 dB |
| 17 | 산지 녹지·산책로 | 38.73 / 39.62 / 39.58 dB (SP +0.89) | 35.30 / 36.25 / 36.19 dB |

**장면 0 · 바다·항구 방파제**

![QB 원본 크기 비교 · 장면 0 · 바다·항구 방파제](images/c2ms_sp/QB/full_scene_00.png)

![QB 확대 비교 · 장면 0 · 바다·항구 방파제](images/c2ms_sp/QB/compare_scene_00.png)

**장면 1 · 부두·곡선 방파제·수목**

![QB 원본 크기 비교 · 장면 1 · 부두·곡선 방파제·수목](images/c2ms_sp/QB/full_scene_01.png)

![QB 확대 비교 · 장면 1 · 부두·곡선 방파제·수목](images/c2ms_sp/QB/compare_scene_01.png)

**장면 3 · 요트 마리나·운동장**

![QB 원본 크기 비교 · 장면 3 · 요트 마리나·운동장](images/c2ms_sp/QB/full_scene_03.png)

![QB 확대 비교 · 장면 3 · 요트 마리나·운동장](images/c2ms_sp/QB/compare_scene_03.png)

**장면 7 · 숲 공원**

![QB 원본 크기 비교 · 장면 7 · 숲 공원](images/c2ms_sp/QB/full_scene_07.png)

![QB 확대 비교 · 장면 7 · 숲 공원](images/c2ms_sp/QB/compare_scene_07.png)

**장면 14 · 언덕 공원·곡선 주택가**

![QB 원본 크기 비교 · 장면 14 · 언덕 공원·곡선 주택가](images/c2ms_sp/QB/full_scene_14.png)

![QB 확대 비교 · 장면 14 · 언덕 공원·곡선 주택가](images/c2ms_sp/QB/compare_scene_14.png)

**장면 17 · 산지 녹지·산책로**

![QB 원본 크기 비교 · 장면 17 · 산지 녹지·산책로](images/c2ms_sp/QB/full_scene_17.png)

![QB 확대 비교 · 장면 17 · 산지 녹지·산책로](images/c2ms_sp/QB/compare_scene_17.png)

<a id="gf2"></a>

## GF2

RGB 밴드 [2,1,0]. GF2로 학습·평가했습니다(10bit 영상이라 PSNR이 약 6 dB 높게 나옵니다).

| 장면 | 장면 PSNR 원본 / SP / SP-Lite | 확대 영역 PSNR 원본 / SP / SP-Lite |
|---:|---|---|
| 0 | 45.00 / 47.79 / 47.72 dB (SP +2.79) | 42.29 / 46.41 / 46.20 dB |
| 1 | 46.89 / 49.43 / 49.39 dB (SP +2.54) | 41.44 / 44.93 / 44.90 dB |
| 3 | 48.08 / 49.85 / 49.81 dB (SP +1.77) | 47.01 / 50.13 / 50.08 dB |
| 7 | 48.45 / 50.78 / 50.74 dB (SP +2.33) | 44.64 / 47.62 / 47.57 dB |
| 14 | 45.72 / 48.04 / 48.00 dB (SP +2.32) | 42.93 / 45.40 / 45.31 dB |
| 17 | 47.18 / 49.63 / 49.57 dB (SP +2.45) | 46.26 / 49.69 / 49.65 dB |

**장면 0**

![GF2 원본 크기 비교 · 장면 0](images/c2ms_sp/GF2/full_scene_00.png)

![GF2 확대 비교 · 장면 0](images/c2ms_sp/GF2/compare_scene_00.png)

**장면 1**

![GF2 원본 크기 비교 · 장면 1](images/c2ms_sp/GF2/full_scene_01.png)

![GF2 확대 비교 · 장면 1](images/c2ms_sp/GF2/compare_scene_01.png)

**장면 3**

![GF2 원본 크기 비교 · 장면 3](images/c2ms_sp/GF2/full_scene_03.png)

![GF2 확대 비교 · 장면 3](images/c2ms_sp/GF2/compare_scene_03.png)

**장면 7**

![GF2 원본 크기 비교 · 장면 7](images/c2ms_sp/GF2/full_scene_07.png)

![GF2 확대 비교 · 장면 7](images/c2ms_sp/GF2/compare_scene_07.png)

**장면 14**

![GF2 원본 크기 비교 · 장면 14](images/c2ms_sp/GF2/full_scene_14.png)

![GF2 확대 비교 · 장면 14](images/c2ms_sp/GF2/compare_scene_14.png)

**장면 17**

![GF2 원본 크기 비교 · 장면 17](images/c2ms_sp/GF2/full_scene_17.png)

![GF2 확대 비교 · 장면 17](images/c2ms_sp/GF2/compare_scene_17.png)

<a id="wv3"></a>

## WV3

RGB 밴드 [4,2,1]. WV3로 학습·평가했습니다.

| 장면 | 장면 PSNR 원본 / SP / SP-Lite | 확대 영역 PSNR 원본 / SP / SP-Lite |
|---:|---|---|
| 0 | 38.53 / 39.76 / 39.76 dB (SP +1.23) | 32.24 / 33.70 / 33.73 dB |
| 1 | 36.24 / 37.49 / 37.50 dB (SP +1.25) | 30.52 / 31.85 / 31.85 dB |
| 3 | 40.61 / 41.78 / 41.78 dB (SP +1.17) | 34.84 / 35.93 / 35.95 dB |
| 7 | 37.22 / 38.46 / 38.46 dB (SP +1.25) | 30.18 / 31.35 / 31.32 dB |
| 14 | 32.78 / 33.52 / 33.51 dB (SP +0.74) | 35.80 / 37.59 / 37.59 dB |
| 17 | 34.90 / 36.12 / 36.10 dB (SP +1.22) | 34.56 / 36.08 / 36.06 dB |

**장면 0**

![WV3 원본 크기 비교 · 장면 0](images/c2ms_sp/WV3/full_scene_00.png)

![WV3 확대 비교 · 장면 0](images/c2ms_sp/WV3/compare_scene_00.png)

**장면 1**

![WV3 원본 크기 비교 · 장면 1](images/c2ms_sp/WV3/full_scene_01.png)

![WV3 확대 비교 · 장면 1](images/c2ms_sp/WV3/compare_scene_01.png)

**장면 3**

![WV3 원본 크기 비교 · 장면 3](images/c2ms_sp/WV3/full_scene_03.png)

![WV3 확대 비교 · 장면 3](images/c2ms_sp/WV3/compare_scene_03.png)

**장면 7**

![WV3 원본 크기 비교 · 장면 7](images/c2ms_sp/WV3/full_scene_07.png)

![WV3 확대 비교 · 장면 7](images/c2ms_sp/WV3/compare_scene_07.png)

**장면 14**

![WV3 원본 크기 비교 · 장면 14](images/c2ms_sp/WV3/full_scene_14.png)

![WV3 확대 비교 · 장면 14](images/c2ms_sp/WV3/compare_scene_14.png)

**장면 17**

![WV3 원본 크기 비교 · 장면 17](images/c2ms_sp/WV3/full_scene_17.png)

![WV3 확대 비교 · 장면 17](images/c2ms_sp/WV3/compare_scene_17.png)

<a id="wv2"></a>

## WV2 전이

RGB 밴드 [4,2,1]. WV3로 학습한 모델을 학습하지 않은 WV2에 적용한 전이 결과입니다(참고). C2MS-SP·SP-Lite에는 [성능 82번](../experiments/performance/82_cross_sensor_band_calibration/README.md)의 밴드별 센서 보정을 함께 적용했고, 원본은 발표된 방식 그대로입니다.

| 장면 | 장면 PSNR 원본 / SP / SP-Lite | 확대 영역 PSNR 원본 / SP / SP-Lite |
|---:|---|---|
| 0 | 27.80 / 30.90 / 30.92 dB (SP +3.11) | 24.19 / 27.56 / 27.62 dB |
| 1 | 26.02 / 29.44 / 29.42 dB (SP +3.41) | 23.00 / 27.77 / 27.74 dB |
| 3 | 33.71 / 36.91 / 36.84 dB (SP +3.20) | 27.19 / 30.70 / 30.67 dB |
| 7 | 28.38 / 30.67 / 30.69 dB (SP +2.29) | 24.03 / 26.51 / 26.51 dB |
| 14 | 27.91 / 30.55 / 30.57 dB (SP +2.64) | 25.65 / 29.38 / 29.40 dB |
| 17 | 28.43 / 30.95 / 30.98 dB (SP +2.52) | 27.33 / 30.43 / 30.40 dB |

**장면 0**

![WV2 원본 크기 비교 · 장면 0](images/c2ms_sp/WV2/full_scene_00.png)

![WV2 확대 비교 · 장면 0](images/c2ms_sp/WV2/compare_scene_00.png)

**장면 1**

![WV2 원본 크기 비교 · 장면 1](images/c2ms_sp/WV2/full_scene_01.png)

![WV2 확대 비교 · 장면 1](images/c2ms_sp/WV2/compare_scene_01.png)

**장면 3**

![WV2 원본 크기 비교 · 장면 3](images/c2ms_sp/WV2/full_scene_03.png)

![WV2 확대 비교 · 장면 3](images/c2ms_sp/WV2/compare_scene_03.png)

**장면 7**

![WV2 원본 크기 비교 · 장면 7](images/c2ms_sp/WV2/full_scene_07.png)

![WV2 확대 비교 · 장면 7](images/c2ms_sp/WV2/compare_scene_07.png)

**장면 14**

![WV2 원본 크기 비교 · 장면 14](images/c2ms_sp/WV2/full_scene_14.png)

![WV2 확대 비교 · 장면 14](images/c2ms_sp/WV2/compare_scene_14.png)

**장면 17**

![WV2 원본 크기 비교 · 장면 17](images/c2ms_sp/WV2/full_scene_17.png)

![WV2 확대 비교 · 장면 17](images/c2ms_sp/WV2/compare_scene_17.png)
