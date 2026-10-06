# SSA-MRN · RGB 유도 HSI 초해상도

고해상도 RGB와 저해상도 HSI로 ×4 HSI 복원을 연구합니다. 아래 수치는 LIB-HSI 관측 HSI를 합성 area 축소한 평가입니다.

[연구 설명](docs/guide/research_overview.md) · [PAN–MS 재현](docs/reproduction.md) · [이전 실험](docs/previous_experiments.md)

## 최신 완료 · RGB11 HSI 경계 손실 비교

RGB07 저학습률 모델의 같은 best 가중치에서 대조군과 경계 손실 4개 강도를 각각 **1e-5 × 10에폭** 학습했습니다. 모두 정상 종료됐고, 검증 MSE로 초기 기여 20% 설정의 best 추가 4에폭을 선택했습니다. 구조는 RGB07의 17특징·공동 디코더이며 23탭·분광 손실·LR 보정을 유지했습니다.

경계 손실은 204밴드 HSI의 가로·세로 밝기 변화량을 정답과 맞춥니다. 양쪽 픽셀이 모두 유효한 경계만 사용합니다. 아래 비율은 학습 장면 8개에서 보정한 초기 gradient/MSE 손실 기여입니다.

| 초기 경계 손실 기여 | best 추가 에폭 | MSE ↓ | PSNR dB ↑ | SAM ° ↓ | gradient RMSE ↓ |
|---|---:|---:|---:|---:|---:|
| 0% · 대조군 | 4 | 0.0004837964 | 34.207613 | 2.245079 | 0.02140527 |
| 2.5% | 4 | 0.0004837661 | 34.208050 | 2.244954 | 0.02140385 |
| 5% | 4 | 0.0004837415 | 34.208456 | 2.244829 | 0.02140241 |
| 10% | 4 | 0.0004836854 | 34.209241 | 2.244597 | 0.02139967 |
| 20% · 선택 | 4 | 0.0004836154 | 34.210515 | 2.244183 | 0.02139492 |

위 표는 **validation 45장면**입니다. 선택한 모델과 같은 학습량의 대조군만 test를 평가했습니다.

| test 75장면 · 300타일 | MSE ↓ | PSNR dB ↑ | SAM ° ↓ |
|---|---:|---:|---:|
| 23탭 + LR 보정 | 0.0011539606 | 30.0649 | 2.4450 |
| 같은 학습량 대조군 | 0.0004378418 | 34.3916 | 2.15071 |
| **경계 손실 20%** | **0.0004376169** | **34.3948** | **2.14986** |

대조군 대비 **PSNR +0.0032 dB, SAM −0.00085°**로 개선 폭은 작습니다. PSNR은 56/75장면, SAM은 70/75장면에서 개선됐습니다. 눈에 띄는 선명도 개선이 확인됐다고 해석하지 않습니다. 직전 RGB09 대비 PSNR·SAM은 조금 나아졌지만 MSE는 약 0.105% 증가해 모든 지표에서 우세한 모델은 아닙니다. [조건·전체 차이·가중치 근거](docs/experiments/rgb11_gradient_suite.md).

![다섯 실험의 학습·검증 곡선](docs/assets/rgb11_gradient_suite/suite_curves.png)

![대조군 대비 검증 차이](docs/assets/rgb11_gradient_suite/suite_comparison.png)

![선택 모델·대조군·23탭 test 비교](docs/assets/rgb11_gradient_suite/test_metrics.png)

## 고정 test 이미지 5종

왼쪽부터 **LR HSI · RGB 입력 · 예측 · 정답**입니다. 표시 대비는 HSI 패널끼리 같으며 지표는 전체 204밴드로 계산합니다.

![test 예시 1](docs/assets/rgb11_gradient_suite/sample_01.png)

![test 예시 2](docs/assets/rgb11_gradient_suite/sample_02.png)

![test 예시 3](docs/assets/rgb11_gradient_suite/sample_03.png)

![test 예시 4](docs/assets/rgb11_gradient_suite/sample_04.png)

![test 예시 5](docs/assets/rgb11_gradient_suite/sample_05.png)

[204밴드 오프라인 HTML](docs/assets/rgb11_gradient_suite/band_viewer/index.html)을 다운로드해 브라우저에서 열 수 있습니다. 기존 [공개 웹뷰어](https://biyonghiyon.github.io/CtrS/ssa-mrn/)는 이전 RGB07 결과입니다.

## 실행 상태와 보관

다섯 학습은 **2026-10-06 19:21:59(KST), 모두 exit 0**으로 완료됐습니다. 이번 실험의 학습은 종료됐고 각 best/latest·설정·해시는 서버에 보존했습니다. [전체 실험 기록](experiments/results/rgb11_gradient_suite/suite_results.json)에 가중치 경로와 실제 50개 에폭·장면별 검증 수치를 보관합니다. 원본 데이터는 Git에 포함하지 않습니다.

[보고서 양식](docs/experiments/template.md) · [작업 규칙](AGENTS.md) · [데이터셋 감사 자료](references/rgb_hsi_datasets.md)
