# RGB 04 · RGB별 12특징 23탭 타일

[이전 실험 목록](../previous_experiments.md) · [현재 연구](../../README.md)

## 1. 목적과 상태

완료. RGB 유도 HSI의 공간 해상도 ×4 복원. 실제 센서 HR-HSI 정답이 있는 실험이 아니라, 관측 HSI를 합성 축소한 정량 평가입니다.

## 2. 변경 사항과 평가 조건

K=4 · 17밴드씩 204→12 · R/G/B 각각 독립 12특징 core · 3개 decoder와 밴드별 학습 fusion · 내부 평균 축소/23탭 확대 · area ×4 · quadrants 학습 / tiles 평가

- LIB-HSI 204밴드, train/validation/test = 393/45/75 장면. 패치는 독립 장면으로 세지 않습니다.
- 전체 test 장면별 지표의 평균. MSE는 204밴드, PSNR data range=1, SAM은 유효 스펙트럼 대상.
- 정합 적용 실험은 유효 마스크를 사용합니다. 정합 보정은 HR-HSI 참조를 활용하므로 실제 배포 전처리 성능과 구분합니다.

## 3. 정량 결과

| 방법 | MSE ↓ | PSNR dB ↑ | SAM ° ↓ |
|---|---:|---:|---:|
| 23tap | 0.0013944695 | 29.2259 | 2.4790 |
| SSA-MRN 확장 | 0.0004733597 | 34.0530 | 2.2057 |

## 4. 직전 연구와 수치 차이

| 지표 | 직전 연구 | 이번 연구 | 차이 (이번−직전) |
|---|---:|---:|---:|
| MSE ↓ | 0.00104667 | 0.00047336 | -0.00057331 |
| PSNR dB ↑ | 30.75741554 | 34.05302682 | +3.29561128 |
| SAM ° ↓ | 2.08926348 | 2.20570832 | +0.11644483 |


**비교 해석:** RGB별 독립 core와 평가 full256→tiles가 동시에 바뀌었습니다. baseline도 26.3846→29.2259 dB로 바뀌므로 PSNR 차이를 모델만의 개선으로 해석할 수 없습니다.

## 5. 그래프

![학습 MSE와 검증 PSNR·SAM](../assets/rgb04_triple12/learning.png)

원본 기록에서 추출한 [epoch 수치](../../experiments/results/rgb04_triple12/curves.json). 기록이 없는 epoch를 보간하거나 만들어 넣지 않았습니다. test 결과를 epoch 선택에 사용하지 않습니다.

![test 수치 비교](../assets/rgb04_triple12/test_metrics.png)

직전 수치 비교 그래프의 `previous*`는 평가 조건 확인이 필요한 관측값입니다.

## 6. 결과 이미지 예시

왼쪽부터 LR HSI, RGB guide, 예측 HSI, HSI 정답입니다. HSI는 69/52/18 밴드의 GT 1–99% 범위를 공통 적용한 시각화이며, 지표 계산에는 대비 조정을 적용하지 않습니다. LR 표시는 최근접 확대로 픽셀을 보여줍니다.

### 예시 1 · 2020-11-24_018 · tile 0, sample index 12

![입력 HSI, RGB, 예측, 정답](../assets/rgb04_triple12/sample_01.png)

### 예시 2 · 2020-11-20_017 · tile 0, sample index 0

![입력 HSI, RGB, 예측, 정답](../assets/rgb04_triple12/sample_02.png)

### 예시 3 · 2020-12-17_010 · tile 0, sample index 172

![입력 HSI, RGB, 예측, 정답](../assets/rgb04_triple12/sample_03.png)

### 예시 4 · 2020-11-26_038 · tile 0, sample index 72

![입력 HSI, RGB, 예측, 정답](../assets/rgb04_triple12/sample_04.png)

### 예시 5 · 2021-01-07_021 · tile 0, sample index 252

![입력 HSI, RGB, 예측, 정답](../assets/rgb04_triple12/sample_05.png)

[사전 고정한 선택 기록](../../experiments/results/rgb04_triple12/selection.json)

## 7. 가중치와 검증 근거

- [전체 test 수치](../../experiments/results/rgb04_triple12/test/metrics.json)
- [가중치 epoch·SHA-256](../../experiments/results/rgb04_triple12/checkpoint_metadata.json)
- 서버 가중치: `C:\CtrS-triple\SSA-MRN\experiments\checkpoints\remote-runs\20261003-162417-e46b8079c798` (best 99 / latest 100).
- 학습된 `.pt`는 보존합니다. 구조/평가 조건은 체크포인트 내 `config`를 읽어 평가할 수 있으므로 과거 별도 학습 JSON은 폐기했습니다.

## 8. 한계와 다음 판단

합성 축소 결과는 실제 센서 쌍의 초해상도 정답 성능을 증명하지 않습니다. RGB–HSI 정합 잔차, 축소 커널 및 타일 경계의 영향을 별도 검증해야 합니다.
