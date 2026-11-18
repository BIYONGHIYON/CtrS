# FusionNet

| 평가자 | 설정 | 상태 |
|:---:|:---:|:---:|
| 김윤지 | `author` QB·WV3, WV2 전이 / `retrain` GF2 seed 42 | 학습 완료 · MATLAB 채점 중 |

**출처:** Deng et al., Detail Injection-based Deep Convolutional Neural Networks for Pansharpening, IEEE TGRS · [공식 DLPan-Toolbox](https://github.com/liangjiandeng/DLPan-Toolbox) · GPL-3.0.

**상태:** QB·WV3 저자 가중치를 공식 PyTorch 구조에 strict load했습니다. 세 센서의 RR·FR 예비 평가는 Python으로 실행했으나 저자 학습 데이터 분할이 확인되지 않았고 결과 차이 검토 및 MATLAB 채점이 남아 있습니다. GF2는 저자 가중치가 없어 400에폭 직접 학습했고 validation MSE 최저인 382에폭을 선택했습니다. 수치는 아직 최종 비교표에 넣지 않습니다.

## 결과

저자 PyTorch 가중치, 공통 Python 지표, RR 20장 기준 예비 평균 PSNR은 QB 31.485 dB, WV3 32.519 dB, WV2 전이 27.058 dB였습니다. 공식 `val_step`처럼 `lms + net(lms, pan)`을 사용했으며 별도 클리핑은 하지 않았습니다. 이 수치는 MATLAB 채점값이 아닙니다. 공개 가중치의 정확한 학습 데이터·분할은 확인되지 않았습니다. 원본 TensorFlow 저장소 가중치로 별도 실행한 참고 수치와 섞지 않습니다.

GF2 재학습의 Python 예비 결과(각 20장)는 RR PSNR 45.547 dB · SAM 0.985 · ERGAS 1.007, FR QNR 0.8708입니다. 실제 MATLAB 채점 결과가 아니며 비교표에 넣기 전입니다. 체크포인트와 평가 파일, 실행 링크는 [진행 기록](provenance/progress_20261119.md)에 있습니다.

### 실제 MATLAB 채점 (저자 가중치)

| 센서 | PSNR (band mean, peak 2047) | SAM | ERGAS | Q2n |
|---|---:|---:|---:|---:|
| QB | 31.485 | 7.3719 | 8.2911 | 0.8047 |
| WV3 | 32.519 | 4.9298 | 4.6888 | 0.8058 |
| WV2 전이 | 27.058 | 7.4858 | 6.0711 | 0.7099 |

RR 각 20장, DLPan MATLAB 채점입니다. Python과 PSNR·SAM·ERGAS는 표시 정밀도에서 같고 Q2n은 구현 차이가 있습니다. 논문값 재현과 정확한 저자 학습 분할은 확인되지 않았습니다. [장면별 결과](results/matlab_author.json). GF2 MATLAB 채점은 진행 중입니다.

## 해석과 정리

공개 가중치에 대응하는 센서는 학습 없이 평가합니다. 가중치의 학습 데이터·분할과 예비 결과 차이를 추가 조사합니다. 센서 전이는 참고로 표시합니다.

<details><summary>방법</summary>

저자 네트워크는 PAN을 밴드 수만큼 복제해 LMS를 뺀 뒤 합성곱과 네 개의 잔차 블록으로 세부 성분을 예측합니다. 어댑터는 이 성분을 LMS에 더합니다. QB·WV3의 입력은 DN / 2047이며 WV2는 WV3 가중치를 적용합니다. GF2는 DN / 1023, 400에폭, 배치 32, Adam 학습률 0.0003, MSE, seed 42로 학습하고 validation MSE로 가중치를 고릅니다. 공식 학습 설정은 DLPan `option_fusionnet.py`와 `fusionnet_main.py`를 따릅니다.

</details>

<details><summary>그림</summary>

네 센서 채점 후 고정 장면 비교를 작성합니다.

</details>

<details><summary>재현 자료</summary>

공식 코드와 설정은 `upstream/` 및 `config.json`을 참고합니다. [Kaggle 실행](https://www.kaggle.com/code/kimyunjiji/fusionnet-gf2-training-and-dlpan-evaluation)에서 저자 가중치 로드, 예비 평가, GF2 학습을 수행합니다. 코드와 가중치·데이터 해시 및 평가 산출물의 보관이 완료되면 manifest를 갱신합니다.

</details>
