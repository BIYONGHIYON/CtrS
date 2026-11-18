# FusionNet

| 평가자 | 설정 | 상태 |
|:---:|:---:|:---:|
| 김윤지 | 저자 가중치 QB·WV3, WV2 전이 / GF2 직접 학습 seed 42 | 학습 완료<br>일부 평가 |

**출처:** Deng 외, Detail Injection-based Deep Convolutional Neural Networks for Pansharpening · [DLPan-Toolbox 공식 코드](https://github.com/liangjiandeng/DLPan-Toolbox). 공식 코드의 라이선스와 저자 가중치의 별도 재배포 조건은 구분합니다.

**확인된 결과:** 네 센서 RR 각 20장의 실제 MATLAB 채점과 FR 각 20장의 Python 평가를 마쳤습니다. GF2는 400에폭 학습하고 validation MSE 최저인 382에폭을 선택했습니다. 저자 가중치의 학습 데이터 분할은 확인되지 않았으며 논문 조건을 완전히 재현했다고 판단할 수 없습니다. 고정 장면 비교 이미지는 작성 전입니다.

[평가 표](../README.md#models) · [결과](#results) · [해석](#verdict) · [방법](#method) · [그림](#figures) · [재현 자료](#evidence)

<a id="results"></a>

## 결과

### RR 실제 MATLAB 채점

| 센서 | 설정 | PSNR (dB) | SAM | ERGAS | Q2n |
|---|---|---:|---:|---:|---:|
| QB | 저자 | 31.485 | 7.3719 | 8.2911 | 0.8047 |
| GF2 | 직접 학습 seed 42 | 45.547 | 0.9847 | 1.0072 | 0.9623 |
| WV3 | 저자 | 32.519 | 4.9298 | 4.6888 | 0.8058 |
| WV2 | WV3 가중치 전이 | 27.058 | 7.4858 | 6.0711 | 0.7099 |

각 센서 RR test 20장입니다. PSNR은 band mean과 peak 2047을 사용했습니다. GF2의 원래 범위 peak 1023을 쓰면 PSNR은 39.522 dB입니다. 같은 배열의 Python과 MATLAB 결과에서 PSNR·SAM·ERGAS는 표시 정밀도에서 일치했습니다. Q2n은 구현에 따른 작은 차이가 있습니다. [저자 설정 장면별 채점](results/matlab_author.json) · [GF2 장면별 채점](results/matlab_retrain.json).

### 기준 모델과 비교

사용자가 제공한 작업 규칙에 따라 기준 모델은 C2MS입니다. QB에서 C2MS seed 42의 실제 MATLAB RR PSNR은 [38.274 dB](../../experiments/performance/53_scale_mixed_training/results/matlab_reevaluation/40202998202/e53_QB_C2MS_s42/summary.json), FusionNet 저자 가중치는 31.485 dB로 차이는 -6.788 dB입니다. FusionNet 저자 가중치의 학습 시드는 확인되지 않아 같은 시드 쌍 비교가 아닙니다. 다른 센서는 C2MS 대조 결과의 조건을 확인한 뒤 비교합니다.

### FR 참고 결과

FR에는 정답이 없으며 Python 공통 평가기로 각 20장을 평가했습니다. GF2 QNR은 0.8708입니다. QNR·Dλ·Ds는 참고 지표이며 개선 판정에 사용하지 않습니다. 다른 센서와 장면별 수치는 [실행 기록](provenance/progress_20261119.md)과 Kaggle 실행 결과에서 확인할 수 있습니다.

<a id="verdict"></a>

## 해석과 정리

QB 결과는 지정한 C2MS 기준보다 낮지만, 저자 가중치의 학습 데이터와 분할이 불명확해 차이의 원인을 단정하지 않습니다. WV2는 WV3 가중치로 평가한 전이 결과입니다. 서로 다른 정규화 범위와 평가 설정을 섞어 논문 보고값 재현이나 모델의 우열을 확정하지 않습니다.

<a id="method"></a>

## 방법

공식 FusionNet은 PAN을 밴드 수만큼 복제한 뒤 LMS와의 차이에서 세부 성분을 예측하고 LMS에 더합니다. QB·WV3·WV2 입력은 DN/2047, GF2 입력은 DN/1023입니다. 출력은 별도로 클리핑하지 않았습니다. GF2 학습은 Adam, 학습률 0.0003, 배치 32, MSE, 400에폭, seed 42를 사용했습니다. 400에폭은 DLPan 공식 학습 설정의 기본값으로, 논문의 모든 실험 설정과 일치한다고 확인된 것은 아닙니다.

저자 가중치는 QB·WV3에 적용했고 WV2에는 WV3 가중치를 사용했습니다. GF2는 우리 데이터의 학습 분할로 학습했으며 validation MSE를 이용해 best를 선택하고 test로 재선택하지 않았습니다. 학습 데이터 분할과 정규화 조건이 다른 기준 모델 간 직접 비교에는 한계가 있습니다.

<a id="figures"></a>

## 그림

C2MS seed 42와 QB RR 고정 장면 0·1·3·7·14·17의 원본 크기 및 확대 비교 이미지는 아직 만들지 않았습니다. 확대 창은 후보의 오차 감소가 가장 큰 영역을 선택하므로 후보에 유리하게 선택된 영역임을 함께 표시할 예정입니다.

<a id="evidence"></a>

## 재현 자료

[설정](config.json) · [GF2 best 가중치](checkpoints/retrain_GF2_seed42/best.pt) · [manifest](manifest.json) · [진행 기록](provenance/progress_20261119.md) · [Kaggle 학습](https://www.kaggle.com/code/kimyunjiji/fusionnet-gf2-training-and-dlpan-evaluation) · [Kaggle 체크포인트 검증과 평가](https://www.kaggle.com/code/kimyunjiji/fusionnet-gf2-checkpoint-validation-and-evaluation?scriptVersionId=371079856).

[GF2 initial·latest Release](https://github.com/BIYONGHIYON/CtrS/releases/tag/weights-benchmarks-fusionnet-retrain) · [저자 예측 Release](https://github.com/BIYONGHIYON/CtrS/releases/tag/predictions-benchmarks-fusionnet-author) · [GF2 예측 Release](https://github.com/BIYONGHIYON/CtrS/releases/tag/predictions-benchmarks-fusionnet-retrain).

[저자 MATLAB 실행](https://github.com/BIYONGHIYON/CtrS/actions/runs/44337025831) · [GF2 MATLAB 실행](https://github.com/BIYONGHIYON/CtrS/actions/runs/44337153698). 저자 가중치의 재배포 허용 여부는 별도 확인되지 않았으며 출처 확인이 필요합니다.
