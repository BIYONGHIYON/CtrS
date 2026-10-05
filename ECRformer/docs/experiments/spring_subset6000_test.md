# Spring 고정 6,000개 학습 · 전체 test 평가

[현재 연구](../../README.md) · [학습 발산 분석](spring_subset6000.md) · [실험 이력](../previous_experiments.md)

## 1. 목적과 상태

2026-10-04 가장 최근 ECRformer 학습의 best(epoch 8)를 2026-10-05에 전체 spring test로 평가했습니다. 같은 학습의 발산한 last(epoch 18)는 사전 선정 5사례에만 추가 평가했습니다. **새 학습·학습 재개 없이 추론과 보고 자료 생성만 수행했습니다.** 안정화 `train_stable.py`의 성능 검증이 아닙니다.

## 2. 평가 조건

| 항목 | 조건 |
| --- | --- |
| 모델 / checkpoint | 전체 ECRformer 약 11.37M / best epoch 8, step 3,375 |
| 기존 학습 | spring train 24,378개 중 고정 6,000개, validation 756개, seed 42 |
| 이번 test | 5 ROI, 전체 3,983패치. 패치 수는 독립 위성 장면 수가 아님 |
| 입력 / 정답 / 크기 | SAR 2 + cloudy optical 13 / optical 13 / 전체 256×256 |
| 전처리 | 기존 default 로더 유지: optical 0~10,000, SAR -25~0 clip 후 0~1 정규화 |
| 장치 / 평가 정밀도 | RTX 3060 Ti / FP32, batch 1, workers 0, TTA 없음, crop 없음 |
| metric | 기존 compute_metric 유지. 예측은 clip하지 않음. LPIPS는 RGB 기반 |
| 사례 선택 | 평가 지표 확인 전 seed 42로 서로 다른 ROI에서 각 1개 선택 |
| 중복 점검 | 저장된 6,000개 train subset과 test의 동일 S1 경로 중복 0개 |

정확한 파일 중복이 없다는 확인은 공간·시간의 완전한 독립성 검증과 다릅니다. 구름 비율 마스크가 없어 비율별 평가를 임의로 생성하지 않았습니다. 평가 설정은 checkpoint의 저장된 모델 설정을 사용했습니다.

## 3. 전체 test 정량 결과

각 패치에서 구한 지표의 평균입니다. 구름 입력을 그대로 정답과 비교한 baseline과 best 예측을 **같은 3,983개 패치**에 평가했습니다.

| 지표 | 구름 입력 baseline | best epoch 8 | 차이(best − baseline) |
| --- | ---: | ---: | ---: |
| RMSE ↓ | 0.159714 | 0.047767 | −0.111947 |
| MAE ↓ | 0.122987 | 0.033674 | −0.089313 |
| PSNR dB ↑ | 18.722031 | 27.150506 | +8.428475 |
| SAM ° ↓ | 13.709208 | 9.298802 | −4.410406 |
| SSIM ↑ | 0.680919 | 0.863298 | +0.182380 |
| LPIPS ↓ | 0.420266 | 0.434178 | **+0.013912 (악화)** |

RMSE·MAE·PSNR·SAM·SSIM 평균은 개선됐으나 LPIPS는 악화했습니다. 모든 패치나 모든 품질 측면이 개선됐다는 뜻은 아닙니다. [원본 평균](../../reproduction/spring_subset6000_test_20261005/summary.json)과 [3,983개 패치별 수치](../../reproduction/spring_subset6000_test_20261005/metrics.csv)를 보존했습니다.

구름 입력 대비 PSNR이 높아진 패치는 3,673/3,983개입니다. LPIPS가 낮아진 패치는 1,967개에 그쳤습니다. 평균 지표와 패치별 개선 여부를 함께 보아야 합니다.

## 4. 이전 상태와 비교

위 baseline은 **이전 학습 모델이 아니라 구름 입력 자체**입니다. 겨울 test와는 계절·ROI·패치 목록이 달라 직접 개선량을 계산하지 않습니다. 기존 spring best validation(PSNR 30.0305)과 이번 test(27.1505)도 다른 분할이므로 학습 전후 차이로 해석하지 않습니다.

last는 같은 고정 5사례에서만 다음과 같이 평가됐습니다. last 전체 test 평균이 아닙니다.

| ROI / patch | best PSNR dB | last PSNR dB |
| --- | ---: | ---: |
| 31 / p288 | 27.1082 | −36.4234 |
| 123 / p268 | 35.8812 | −35.7668 |
| 140 / p182 | 25.8198 | −36.3819 |
| 44 / p810 | 25.6312 | −36.4263 |
| 106 / p137 | 29.1486 | −36.4139 |

[5사례의 전체 지표와 출력 범위](../../reproduction/spring_subset6000_test_20261005/selected_metrics.json). last는 이 5사례에서 FP32 추론으로도 실패했습니다. 저장된 모델 상태의 문제가 지속된다는 근거지만 최초 발산의 원인을 FP16으로 확정하는 실험은 아닙니다.

## 5. 학습·검증 및 test 그래프

![실제 기록된 epoch 7~18 학습·검증 곡선](../../reproduction/spring_subset6000_test_20261005/learning.png)

재개 구간 **epoch 7~18만** 그렸습니다. 초록 점선은 best 8, 빨강 점선은 발산 16입니다. MAE는 로그 축이며 기록이 없는 0~6은 채우지 않았습니다. [compact epoch 기록](../../reproduction/spring_subset6000_test_20261005/curves.json)에 lr 0.0004 등의 원래 관측값도 남겼습니다. 주 학습 loss 전체 곡선은 당시 기록에 없어 재구성하지 않았습니다.

![같은 전체 spring test의 baseline과 best 모델 비교](../../reproduction/spring_subset6000_test_20261005/test_metrics.png)

그래프는 PSNR·SAM·SSIM 3지표를 보여줍니다. 이 그림에 없는 LPIPS는 위 표처럼 악화했으므로 함께 읽어야 합니다.

## 6. 사전 고정 5사례 이미지

왼쪽부터 **SAR(VV) · 구름 입력 · best epoch 8 · last epoch 18 · 정답**입니다. RGB는 0-based `(3,2,1)`과 동일 밝기 3.0을 사용했습니다. SAR는 정규화된 VV의 공통 0~1 회색조입니다. RGB 표시만 clip하며 metric은 clip하지 않았습니다.

last의 5사례 출력 범위는 전체 밴드 기준 대략 **35.96~108.37**로 정답의 0~1 범위를 크게 벗어났습니다. 따라서 last 열이 흰색으로 포화되는 것은 파일 누락이나 잘못된 이미지 저장이 아니라 실제 발산 출력의 표시 결과입니다.

![ROI31 p288](../../reproduction/spring_subset6000_test_20261005/sample_01.png)

![ROI123 p268](../../reproduction/spring_subset6000_test_20261005/sample_02.png)

![ROI140 p182](../../reproduction/spring_subset6000_test_20261005/sample_03.png)

![ROI44 p810](../../reproduction/spring_subset6000_test_20261005/sample_04.png)

![ROI106 p137](../../reproduction/spring_subset6000_test_20261005/sample_05.png)

[선정 목록·상대 TIFF 경로·seed·순서](../../reproduction/spring_subset6000_test_20261005/selection.json). 선택 인덱스는 `[250, 2580, 3275, 1538, 1672]`이며 결과 확인 후 교체하지 않았습니다. 5 ROI가 서로 다르다는 확인을 모든 공간·시간 독립성이 보장된 5개 장면이라고 확대 해석하지 않습니다.

그림에서 best는 구름을 줄이지만 농경지 경계·산지 질감 등 세부가 흐려지는 사례가 있습니다. 구름이 적어 보이는 ROI123 사례는 PSNR/SAM이 입력 baseline보다 악화했습니다. 좋은 사례만으로 결과를 포장하지 않고 해당 한계를 그대로 남깁니다.

## 7. 가중치·코드·검증 근거

- 서버 원본 결과: `C:\CtrS\ECRformer\Official_ECRformer\results\spring_subset6000_test_20261005`.
- best/last는 기존 `experiments\ecrformer_spring_spring_subset6000\version_0\checkpoints`에 유지합니다. Git에 가중치·원시 로그·TIFF·전체 NPZ를 추가하지 않았습니다.
- best SHA-256: `f365d3384b202eb6580327069cd6e10a5f3c7ee764d0dec9678d22e9eccb88dd`.
- last SHA-256: `fc846942092451b4bcb1d03571f1f9295f73db57e0f59b75656e9ce2d6647551`.
- [provenance](../../reproduction/spring_subset6000_test_20261005/provenance.json)에 평가 당시 Git HEAD와 실제 서버 source SHA-256·정밀도·checkpoint epoch를 기록했습니다. 서버는 미커밋 변경이 있어 HEAD만으로 실행 코드를 완전히 특정할 수 없습니다. 과거 학습 당시 정확한 commit은 미확보이며 평가 HEAD를 학습 commit으로 대신 쓰지 않았습니다.
- [artifact manifest](../../reproduction/spring_subset6000_test_20261005/report_manifest.json) 해시, 3,983개 행·전체 평균·선정 사례 대응을 검증했습니다. 순수 helper 테스트 4개도 통과했습니다.
- 보고 자료 폴더의 `.gitattributes`로 JSON/CSV 줄바꿈 자동 변환을 막아 Windows/Linux clone에서도 원본 artifact 해시가 유지되도록 했습니다.
- [재실행·검증 안내](../test_report_export.md). 모델·로더·학습 설정은 수정하지 않았습니다.

## 8. 한계와 다음 판단

spring 일부 학습과 5 test ROI의 결과이지 모든 계절의 SEN12MS-CR 재현이 아닙니다. 평균 복원 오차는 감소했지만 질감과 RGB 지각 품질의 개선은 보장되지 않습니다. last 대신 best를 평가 후보로 유지하고, 다음 안정화 학습은 별도 사용자 승인 후 실행합니다. 향후 개선은 validation에서 조정하고 동일 test 프로토콜로 최종 확인하되, 이번 test를 보고 유리한 사례나 조건으로 바꾸지 않습니다.
