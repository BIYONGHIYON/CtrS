# ARConv

> 현재 비교 기준은 SMOC입니다([최신 평가 표](../README.md#models)). 이 보고서의 C2MS-SP 수치·이미지는 당시 비교 기록이며 그대로 보존합니다.

| 평가자 | 설정 | 상태 |
|:---:|:---:|:---:|
| 김윤지 | `author` (저자 가중치, QB 200 · GF2 630 · WV3 600에폭, 저자 제공) | 평가 완료 |

**출처:** Wang 외, Adaptive Rectangular Convolution for Remote Sensing Pansharpening, CVPR, 2025, [arXiv 2503.00467](https://arxiv.org/abs/2503.00467) · 공식 코드 [Xueyangwang-will/ARConv](https://github.com/Xueyangwang-will/ARConv) `f568bef` · 코드 GPL-3.0, 저자 가중치는 재배포 조건이 확인되지 않아 재배포하지 않음

**결과:** 저자 가중치의 RR test PSNR(실제 MATLAB)은 QB 38.425 · GF2 49.385 · WV3 39.167 dB이고, C2MS-SP 3시드 평균 대비 +0.052 · +0.600 · +0.493 dB입니다. SAM·ERGAS도 세 센서 모두 C2MS-SP 평균보다 낮습니다. WV2 전이(참고)는 29.853 dB로 −1.682 dB입니다. 판정은 붙이지 않습니다.

[평가 표](../README.md#models) · [결과](#results) · [해석](#verdict) · [방법](#method) · [그림](#figures) · [재현 자료](#evidence)

<a id="results"></a>

## 결과

각 센서의 RR test 20장을 MATLAB R2025b와 DLPan-Toolbox(`a34f884`) `score_suite.m`으로 채점했습니다([matlab.json](results/matlab.json)). PSNR은 팀 공통 비교와 같은 밴드별 평균, peak 2047입니다(GF2의 peak 1023 기준 밴드 평균은 43.360 dB). 저자 가중치에는 시드가 없어 같은 시드 짝 비교 대신 C2MS-SP 3시드(42·43·44) 평균과의 차이를 적고, 참고로 seed 42와의 PSNR 차이도 적습니다. C2MS-SP 값은 [기준 비교 수치](../../docs/images/c2ms_sp/baseline_metrics.json)입니다.

| 센서 | ARConv PSNR | SAM | ERGAS | SCC | Q2n | C2MS-SP 3시드 평균 PSNR / SAM / ERGAS | 차이 PSNR / SAM / ERGAS | seed 42 대비 PSNR |
|---|---:|---:|---:|---:|---:|---|---|---:|
| QB | 38.425 | 4.469 | 3.661 | 0.9827 | 0.9366 | 38.373 / 4.474 / 3.685 | +0.052 / −0.005 / −0.024 | +0.029 |
| GF2 | 49.385 | 0.710 | 0.645 | 0.9906 | 0.9824 | 48.785 / 0.763 / 0.691 | +0.600 / −0.053 / −0.046 | +0.652 |
| WV3 | 39.167 | 2.858 | 2.118 | 0.9863 | 0.9165 | 38.673 / 2.984 / 2.241 | +0.493 / −0.126 / −0.123 | +0.504 |
| WV2 (WV3 가중치 전이, 참고) | 29.853 | 5.452 | 4.418 | 0.9123 | 0.8357 | 31.535 / 5.063 / 3.664 | −1.682 / +0.388 / +0.754 | −1.482 |

C2MS-SP 시드별 PSNR: QB 38.396 · 38.344 · 38.379, GF2 48.732 · 48.870 · 48.752, WV3 38.663 · 38.720 · 38.636, WV2 31.335 · 31.668 · 31.604 dB(42 · 43 · 44). WV2의 C2MS-SP에는 성능 82번 밴드별 센서 보정이 붙어 있고 ARConv에는 없습니다. MATLAB과 같은 배열의 Python 공통 지표는 PSNR·SAM·ERGAS·SCC가 표시 정밀도에서 같고, Q2n은 구현 차이로 최대 약 0.000039 다릅니다.

**논문 보고값과 비교:** 논문은 RR PSNR을 보고하지 않습니다. 보고된 SAM / ERGAS / Q(논문 표 1-3)와 우리 재현값은 아래와 같습니다.

| 센서 | 논문 SAM | 재현 SAM | 논문 ERGAS | 재현 ERGAS | 논문 Q4/Q8 | 재현 Q2n |
|---|---:|---:|---:|---:|---:|---:|
| QB | 4.430 | 4.469 | 3.633 | 3.661 | 0.939 | 0.937 |
| GF2 | 0.698 | 0.710 | 0.626 | 0.645 | 0.983 | 0.982 |
| WV3 | 2.885 | 2.858 | 2.139 | 2.118 | 0.921 | 0.917 |

**FR (참고, Python 공통 지표, 정답 없음):**

| 센서 | QNR | Dλ | Ds |
|---|---:|---:|---:|
| QB | 0.9412 | 0.0243 | 0.0355 |
| GF2 | 0.9414 | 0.0268 | 0.0328 |
| WV3 | 0.9677 | 0.0127 | 0.0199 |
| WV2 | 0.9545 | 0.0232 | 0.0230 |

장면별 FR 값은 [summary.json](results/summary.json)에 있습니다. 논문 FR은 HQNR이라 QNR과 직접 비교하지 않습니다.

<a id="verdict"></a>

## 해석과 정리

- **재현:** WV3는 SAM·ERGAS가 논문보다 조금 낮게(좋게) 나왔고, QB·GF2는 SAM 0.01-0.04, ERGAS 0.02-0.03 높게 나왔습니다. 원인 후보는 배포 체크포인트가 논문 실행과 같은 것인지 확인되지 않은 점(QB 2번째, GF2 3·4·7·8번째 고정 커널 크기가 보충자료 표 9와 다름)과 평가 코드 차이입니다. 숫자는 보정하지 않았습니다.
- **C2MS-SP와의 차이가 나는 곳:** seed 42 장면 PSNR로 보면 ARConv가 GF2 20/20장(+0.09에서 +1.11 dB), WV3 20/20장(+0.15에서 +0.77 dB)에서 높고, QB는 10/20장(−0.12에서 +0.30 dB)으로 비슷합니다. 학습하지 않은 WV2에서는 2/20장만 높습니다. C2MS-SP의 WV2 결과에는 밴드별 센서 보정이 들어 있어 전이 조건이 같지 않습니다.
- **시드와 예산:** 저자 가중치 한 벌(시드·선택 절차 미공개)의 결과이고, 학습 에폭은 저자가 제공한 체크포인트의 에폭입니다. ARConv 파라미터는 약 1,592만 개(QB 15,921,412, WV3 15,923,720)입니다.
- **가져올 점:** 특징에 따라 커널의 높이·너비를 정하는 적응 직사각형 합성곱이 GF2·WV3에서 C2MS-SP보다 높은 수치를 보였습니다. 가져올지는 성능 개선 연구에 새 실험으로 등록해 판단합니다. 현재 등록된 실험은 없습니다.
- 판정(개선 여부, C2MS-SP 대비 우열)은 붙이지 않고, 이 결과로 기준 모델을 바꾸지 않습니다.

<a id="method"></a>

<details>
<summary><b>방법</b> · 목적과 상태 · 모델과 비교 조건</summary>

### 목적과 상태

ARConv는 CVPR 2025에서 발표된 팬샤프닝 모델로, 저자가 QB·GF2·WV3 가중치를 공개해 학습 없이 우리 test 장면·지표로 잴 수 있습니다. 확인하려는 질문은 ① 저자 가중치로 논문 보고값이 재현되는가, ② 같은 test·지표에서 C2MS-SP와 수치가 어떻게 다른가입니다.

| 설정 | 센서 | 상태 (2026-11-18 KST) |
|---|---|---|
| `author` | QB·GF2·WV3, WV2(WV3 가중치 전이, 참고) | 추론·실제 MATLAB 채점·Python FR·고정 6장면 이미지 완료 |

### 모델과 비교 조건

공식 `ARNet`에 PAN과 보간 MS를 넣고 `epoch=1000`으로 고정 커널을 사용합니다. 모든 센서 입력을 2047로 나누고 출력을 2047로 복원했습니다. CtrS GF2 표준 정규화는 1023이므로 공통 어댑터에서는 입력에 1023/2047, 출력에 2047/1023 변환이 필요합니다. 체크포인트 세 개는 공식 네트워크에 strict load를 통과했습니다.

[논문 보충자료 표 9](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Wang_Adaptive_Rectangular_Convolution_CVPR_2025_supplemental.pdf)에 QB `[1,9]`, GF2 `[1,18]`, WV3 `[1,18]`이 명시되어 있습니다. 앞서 장면 0에 대해 상한 3·9·18·27을 저자 제공 출력과 대조한 결과와 일치합니다. 전체 20장 평균 절대 차이는 QB 약 0.0195 DN, GF2 약 0.0167 DN, WV3 약 0.0573 DN입니다. WV3의 체크포인트 내 고정 커널 크기 10개는 표 9와 일치했습니다. QB는 2번째, GF2는 3·4·7·8번째 커널이 표와 다릅니다. 제공된 출력과 근접하더라도 이 체크포인트가 표의 실행과 완전히 같은 것인지는 확인되지 않았습니다. 보충자료 7.1절은 PanCollection 학습·평가 분할을 명시하지만 배포 체크포인트 자체의 생성 이력은 별도로 확인되지 않았습니다.

| 항목 | C2MS-SP | ARConv (`author`) |
|---|---|---|
| 학습 데이터·분할 | PanCollection 학습·validation, 3시드(42·43·44) | 저자 학습(보충자료 7.1절 PanCollection 분할), 학습하지 않음 |
| test 장면·정규화 | RR·FR 각 20장, QB·WV3·WV2 2047, GF2 1023 | 같은 test 파일(SHA-256 확인), 저자 방식은 모든 센서 2047(어댑터에서 GF2 변환) |
| 에폭·가중치 선택 | 150에폭, validation MSE 최소 | 저자 제공 체크포인트(QB 200 · GF2 630 · WV3 600에폭), 선택 절차 미공개 |
| 후처리 | 관측 일치 사영(K10), WV2 밴드별 보정 | 없음 |
| 시드 | 42·43·44 | 미공개 |
| 파라미터 수 | | 15,921,412 (QB·GF2), 15,923,720 (WV3·WV2) |

</details>

<a id="figures"></a>

<details>
<summary><b>그림</b> · 학습·평가 그래프 · 결과 이미지</summary>

### 학습·평가 그래프

`author` 설정만 평가해서 학습 곡선은 없습니다.

### 결과 이미지

네 데이터셋의 RR test 고정 6장면(0·1·3·7·14·17)입니다. 원본 크기 비교는 LR MS · PAN · C2MS-SP(seed 42) · ARConv · 정답과 같은 색 척도의 두 오차 지도이고, 확대 비교는 C2MS-SP 오차에서 ARConv 오차를 뺀 값이 가장 큰 64×64 영역을 4배 확대한 것입니다. 확대 영역은 ARConv에 유리하게 고른 것이므로 아래 장면 PSNR(C2MS-SP / ARConv, dB)을 함께 봅니다. WV2는 두 모델 모두 WV3 가중치를 적용한 전이이고, C2MS-SP에는 성능 82번 밴드별 센서 보정이 붙어 있습니다.

ARConv 예측은 가중치를 다시 실행하지 않고 [예측 Release](https://github.com/BIYONGHIYON/CtrS/releases/tag/predictions-benchmarks-arconv-author)의 zip(SHA-256은 [manifest](manifest.json))에서 읽었습니다(`python SSA-MRN/common/benchmark_evaluate.py --model arconv --setting author --sensor <센서> --figures --release`). 장면별 수치는 `results/images/author_<센서>/result_images.json`에 있습니다.

| 데이터셋 | 장면 0 | 장면 1 | 장면 3 | 장면 7 | 장면 14 | 장면 17 |
|---|---:|---:|---:|---:|---:|---:|
| QB | 43.04 / 43.22 | 43.31 / 43.43 | 38.10 / 38.40 | 39.71 / 39.65 | 38.56 / 38.58 | 39.62 / 39.59 |
| GF2 | 47.79 / 48.91 | 49.43 / 50.23 | 49.85 / 50.97 | 50.78 / 51.40 | 48.04 / 48.67 | 49.63 / 50.24 |
| WV3 | 39.76 / 40.53 | 37.49 / 38.23 | 41.78 / 42.47 | 38.46 / 38.86 | 33.52 / 34.04 | 36.12 / 36.40 |
| WV2 전이 (WV3 가중치) | 30.90 / 29.13 | 29.44 / 27.34 | 36.91 / 35.83 | 30.67 / 29.37 | 30.55 / 29.13 | 30.95 / 29.69 |

#### 장면 0

![QB 장면 0 원본 크기 비교](results/images/author_QB/full_scene_00.png)

![QB 장면 0 확대 비교](results/images/author_QB/compare_scene_00.png)

#### 장면 1

![QB 장면 1 원본 크기 비교](results/images/author_QB/full_scene_01.png)

![QB 장면 1 확대 비교](results/images/author_QB/compare_scene_01.png)

#### 장면 3

![QB 장면 3 원본 크기 비교](results/images/author_QB/full_scene_03.png)

![QB 장면 3 확대 비교](results/images/author_QB/compare_scene_03.png)

#### 장면 7

![QB 장면 7 원본 크기 비교](results/images/author_QB/full_scene_07.png)

![QB 장면 7 확대 비교](results/images/author_QB/compare_scene_07.png)

#### 장면 14

![QB 장면 14 원본 크기 비교](results/images/author_QB/full_scene_14.png)

![QB 장면 14 확대 비교](results/images/author_QB/compare_scene_14.png)

#### 장면 17

![QB 장면 17 원본 크기 비교](results/images/author_QB/full_scene_17.png)

![QB 장면 17 확대 비교](results/images/author_QB/compare_scene_17.png)

<details><summary>GF2 6장면</summary>

#### GF2 장면 0

![GF2 장면 0 원본 크기 비교](results/images/author_GF2/full_scene_00.png)

![GF2 장면 0 확대 비교](results/images/author_GF2/compare_scene_00.png)

#### GF2 장면 1

![GF2 장면 1 원본 크기 비교](results/images/author_GF2/full_scene_01.png)

![GF2 장면 1 확대 비교](results/images/author_GF2/compare_scene_01.png)

#### GF2 장면 3

![GF2 장면 3 원본 크기 비교](results/images/author_GF2/full_scene_03.png)

![GF2 장면 3 확대 비교](results/images/author_GF2/compare_scene_03.png)

#### GF2 장면 7

![GF2 장면 7 원본 크기 비교](results/images/author_GF2/full_scene_07.png)

![GF2 장면 7 확대 비교](results/images/author_GF2/compare_scene_07.png)

#### GF2 장면 14

![GF2 장면 14 원본 크기 비교](results/images/author_GF2/full_scene_14.png)

![GF2 장면 14 확대 비교](results/images/author_GF2/compare_scene_14.png)

#### GF2 장면 17

![GF2 장면 17 원본 크기 비교](results/images/author_GF2/full_scene_17.png)

![GF2 장면 17 확대 비교](results/images/author_GF2/compare_scene_17.png)

</details>

<details><summary>WV3 6장면</summary>

#### WV3 장면 0

![WV3 장면 0 원본 크기 비교](results/images/author_WV3/full_scene_00.png)

![WV3 장면 0 확대 비교](results/images/author_WV3/compare_scene_00.png)

#### WV3 장면 1

![WV3 장면 1 원본 크기 비교](results/images/author_WV3/full_scene_01.png)

![WV3 장면 1 확대 비교](results/images/author_WV3/compare_scene_01.png)

#### WV3 장면 3

![WV3 장면 3 원본 크기 비교](results/images/author_WV3/full_scene_03.png)

![WV3 장면 3 확대 비교](results/images/author_WV3/compare_scene_03.png)

#### WV3 장면 7

![WV3 장면 7 원본 크기 비교](results/images/author_WV3/full_scene_07.png)

![WV3 장면 7 확대 비교](results/images/author_WV3/compare_scene_07.png)

#### WV3 장면 14

![WV3 장면 14 원본 크기 비교](results/images/author_WV3/full_scene_14.png)

![WV3 장면 14 확대 비교](results/images/author_WV3/compare_scene_14.png)

#### WV3 장면 17

![WV3 장면 17 원본 크기 비교](results/images/author_WV3/full_scene_17.png)

![WV3 장면 17 확대 비교](results/images/author_WV3/compare_scene_17.png)

</details>

<details><summary>WV2 전이 (WV3 가중치) 6장면</summary>

#### WV2 장면 0

![WV2 장면 0 원본 크기 비교](results/images/author_WV2/full_scene_00.png)

![WV2 장면 0 확대 비교](results/images/author_WV2/compare_scene_00.png)

#### WV2 장면 1

![WV2 장면 1 원본 크기 비교](results/images/author_WV2/full_scene_01.png)

![WV2 장면 1 확대 비교](results/images/author_WV2/compare_scene_01.png)

#### WV2 장면 3

![WV2 장면 3 원본 크기 비교](results/images/author_WV2/full_scene_03.png)

![WV2 장면 3 확대 비교](results/images/author_WV2/compare_scene_03.png)

#### WV2 장면 7

![WV2 장면 7 원본 크기 비교](results/images/author_WV2/full_scene_07.png)

![WV2 장면 7 확대 비교](results/images/author_WV2/compare_scene_07.png)

#### WV2 장면 14

![WV2 장면 14 원본 크기 비교](results/images/author_WV2/full_scene_14.png)

![WV2 장면 14 확대 비교](results/images/author_WV2/compare_scene_14.png)

#### WV2 장면 17

![WV2 장면 17 원본 크기 비교](results/images/author_WV2/full_scene_17.png)

![WV2 장면 17 확대 비교](results/images/author_WV2/compare_scene_17.png)

</details>

</details>

<a id="evidence"></a>

<details>
<summary><b>재현 자료</b> · 코드, 가중치, 설정, 환경</summary>

- [MATLAB 상세 결과](results/matlab.json), [FR 장면별 값](results/summary.json), [입력·가중치 해시](provenance/metadata.json).
- [RR 예측 Release](https://github.com/BIYONGHIYON/CtrS/releases/tag/predictions-benchmarks-arconv-author), [MATLAB 실행 기록](https://github.com/BIYONGHIYON/CtrS/actions/runs/44121444016).
- GF2 실제 RR 장면 0·1·2를 CPU에서 저자 정규화와 어댑터 정규화로 비교했습니다. 평균 절대 출력 차이 0.000036–0.000048 DN, 최대 차이 0.002167 DN, 전체 화소 PSNR 최대 차이 0.000005 dB입니다. 전체 20장 재실행이나 원래 GPU 출력과 배열 대조를 대신하지는 않습니다.
- 저자 가중치는 가중치 자체의 재배포 조건이 확인되지 않아 저장소와 Release에 두지 않습니다. 원 배포처 주소와 세 체크포인트의 SHA-256은 [config.json](config.json)에 있습니다. [추론 실행기](run_author.py)는 노트북의 계산 과정을 재구성했으며 GF2 장면 0에서 CPU 직접 추론과 PSNR이 일치했습니다.
- 원본 코드 저장소는 GPL-3.0입니다. 가중치에 별도로 적용되는 조건은 명시적으로 확인되지 않았습니다.
- 보고서·MATLAB 수치는 저자 방식 추론기(`run_author.py`)로 만든 예측에서 나왔습니다. 공통 어댑터(`model.py`)의 20장 전체 출력과 이 예측의 배열 대조는 하지 않았습니다(위 GF2 3장면 CPU 대조만 있음).

</details>
