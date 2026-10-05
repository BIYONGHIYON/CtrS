# 겨울 두 번째 부분 · 미세조정 전후

[현재 연구](../../README.md) · [실험 이력](../previous_experiments.md)

## 1. 목적과 상태

첫 번째 부분 모델을 두 번째 부분에 미세조정했을 때 같은 test에서 성능이 달라지는지 확인한 완료 평가입니다. 이번에는 기존 JSON·CSV·그림만 정리했습니다. 실행 당시 코드 commit/run ID와 가중치 SHA-256은 미확보입니다.

## 2. 변경 사항과 조건

ECRformer, SAR 2 + cloudy 13 → optical 13. 첫 번째 부분 가중치에서 새 optimizer로 미세조정했습니다. 두 번째 부분 train 7,162 / validation 1,385 / test 784패치, lr 1e-4, seed 42, 유효 배치 16, 실행 13 epoch, best epoch 2(0-based). [학습 설정](../../reproduction/winter_half2/hparams.yaml).

각 부분은 한 test ROI의 패치이며 독립 장면 수가 아닙니다. 첫 번째 부분 전체 train 명세가 없어 두 부분 전체의 중복 여부를 확정하지 못했습니다.

## 3. 정량 결과

| 지표 | 전 | 후 | 차이(후 − 전) |
| --- | ---: | ---: | ---: |
| RMSE ↓ | 0.05518 | 0.05146 | −0.00372 |
| MAE ↓ | 0.03932 | 0.03661 | −0.00271 |
| PSNR dB ↑ | 25.245 | 25.888 | +0.643 |
| SAM ° ↓ | 11.478 | 10.779 | −0.698 |
| SSIM ↑ | 0.81873 | 0.82994 | +0.01121 |
| LPIPS ↓ | 0.48422 | 0.47428 | −0.00994 |

[전 summary](../../reproduction/winter_half2/before_summary.json) · [후 summary](../../reproduction/winter_half2/after_summary.json) · [전 패치 CSV](../../reproduction/winter_half2/before_metrics.csv) · [후 패치 CSV](../../reproduction/winter_half2/after_metrics.csv). 표는 반올림이며 원본 JSON을 기준으로 합니다. LPIPS는 구현의 RGB 변환 기반 지표로 13밴드 분광 지표와 구분합니다.

## 4. 직전 실험과 비교

위 차이는 같은 두 번째 부분 test 784패치의 전후입니다. 첫 번째 부분 test 783패치나 spring validation과 직접 비교하지 않습니다. SAM 개선은 관측됐지만 분광 손실 추가의 효과가 아닙니다.

## 5. 곡선

epoch별 과거 학습 곡선은 미확보입니다. 평가 요약·CSV로 학습 곡선을 만들어 채우지 않습니다. 미래 실행은 loss/MAE와 validation PSNR/SAM/SSIM 및 학습률을 보존합니다.

## 6. 이미지

SAR · cloudy · before · after · target 순서입니다. 4행은 test 인덱스 0~3이고 한 ROI의 패치입니다. 사전 선정 seed/manifest는 미확보입니다.

![같은 겨울 test 패치 0~3의 미세조정 전후](../../reproduction/winter_half2/comparisons/before_after_0000_0003.png)

RGB 밴드 `(3,2,1)`(0-based), 밝기 3.0, SAR 첫 밴드 회색조. 잔여 구름·흐림도 확인합니다. 표시용 RGB와 원래 13밴드 수치 평가를 구분합니다.

## 7. 가중치와 증거

전: [winter_half1/model_weights.pt](../../reproduction/winter_half1/model_weights.pt), 후: [winter_half2/model_weights.pt](../../reproduction/winter_half2/model_weights.pt). 기존 파일을 이동·수정하지 않았습니다. 평가용 가중치는 optimizer 상태가 없으며 학습 재개 checkpoint와 다릅니다. summary의 원본 checkpoint 경로는 Git 다운로드 링크가 아닙니다.

## 8. 한계와 다음 판단

전체 SEN12MS-CR test 재현이 아닌 겨울 부분 평가입니다. 평균 개선이 모든 패치·구름 비율의 개선을 보장하지 않습니다. 다음 실험은 사전 고정 사례·밴드별 오차·ROI/구름 비율별 분석·곡선을 추가합니다.
