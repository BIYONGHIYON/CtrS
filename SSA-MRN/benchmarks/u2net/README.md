# U2Net: Spatial-Spectral-Integrated Double U-Net

| 평가자 | 설정 | 상태 |
|:---:|:---:|:---:|
| 송준현 | `retrain`: QB·GF2 seed42, 360에폭<br>`author`: WV3 공개 가중치·WV2 전이 | 4센서 RR 실제 MATLAB 평가·이미지·가중치 보관 완료 |

**출처:** Peng et al., ACM MM 2023. [논문](https://doi.org/10.1145/3581783.3612084) · [공식 코드](https://github.com/PSRben/U2Net). 공식 소스는 GPL-3.0 배포본이며 고정 버전·가중치 해시는 [설정](config.json)과 [manifest](manifest.json)에 있습니다.

**결과:** 네 센서 RR test 각20장을 실제 MATLAB으로 채점했습니다. SMOC 대비 RR PSNR 차이는 QB −1.868, GF2 −0.513, WV3 −0.292, WV2 전이 −1.515 dB입니다. QB·GF2는 같은 seed42, 저자 seed 미상의 WV3·WV2는 SMOC 3시드 평균과 비교합니다. **재학습은 micro4 × 누적4를 사용한 보관 실험으로, 현재 규칙의 엄격한 논문 학습법 재현에는 해당하지 않습니다.**

[평가 표](../README.md#models) · [결과](#current-results) · [해석](#current-interpretation) · [방법](#current-method) · [그림](#current-figures) · [재현 자료](#current-evidence)

<a id="current-results"></a>

## 결과

MATLAB R2025b·DLPan 공통 프로필로 같은 예측 배열을 채점했습니다. Δ는 U2Net − SMOC입니다. PSNR은 밴드별 PSNR의 평균, peak2047입니다. 경계 자르기와 평가기 추가 clipping은 끄고 Q2n block32·배율4를 사용했습니다. U2Net 자체의 원래 추론 clipping은 유지했습니다. [실제 결과](results/matlab.json) · [장면별 차이](results/scene_deltas_matlab.json) · [집계](results/summary_smoc.json)

| 센서·설정 | 모델 | PSNR ↑ | SAM ↓ | ERGAS ↓ | SCC ↑ | Q2n ↑ | Q ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| QB · retrain | SMOC seed42 | 38.367643 | 4.518059 | 3.691220 | 0.982587 | 0.935467 | 0.935517 |
| QB · retrain | U2Net seed42 | 36.500012 | 4.807527 | 4.662338 | 0.971718 | 0.920406 | 0.923191 |
| QB · retrain | Δ | -1.867630 | +0.289468 | +0.971118 | -0.010868 | -0.015062 | -0.012326 |
| GF2 · retrain | SMOC seed42 | 49.419027 | 0.712302 | 0.643031 | 0.990701 | 0.982581 | 0.983443 |
| GF2 · retrain | U2Net seed42 | 48.906404 | 0.728360 | 0.675263 | 0.990494 | 0.980906 | 0.983312 |
| GF2 · retrain | Δ | -0.512623 | +0.016058 | +0.032232 | -0.000207 | -0.001675 | -0.000130 |
| WV3 · author | SMOC 3seed 평균 | 39.135979 | 2.859080 | 2.120891 | 0.986482 | 0.916012 | 0.919701 |
| WV3 · author | U2Net author | 38.844178 | 2.896916 | 2.199791 | 0.985843 | 0.915956 | 0.919474 |
| WV3 · author | Δ | -0.291802 | +0.037837 | +0.078900 | -0.000639 | -0.000056 | -0.000228 |
| WV2 · author · 전이 | SMOC 3seed 평균 | 32.011344 | 4.950737 | 3.456145 | 0.945646 | 0.884924 | 0.883396 |
| WV2 · author · 전이 | U2Net author | 30.496774 | 5.258538 | 4.076282 | 0.933318 | 0.846967 | 0.853769 |
| WV2 · author · 전이 | Δ | -1.514570 | +0.307802 | +0.620137 | -0.012327 | -0.037958 | -0.029627 |

GF2의 센서 peak1023 PSNR은 U2Net 42.881559, SMOC seed42 43.394183 dB입니다. 표의 peak2047과 절대값을 섞지 않으며 두 방식의 Δ는 동일합니다.

### 시드 범위와 Python 교차 확인

QB·GF2의 U2Net은 각1시드이므로 시드 표준편차를 추정하지 않습니다. WV3·WV2의 U2Net 역시 저자 가중치1개이며 seed 미상입니다. SMOC의 3시드 값과 표준편차는 아래와 같습니다(표본 표준편차).

| 센서 | SMOC seed42 PSNR | seed43 | seed44 | 평균 ± 표준편차 |
|---|---:|---:|---:|---:|
| WV3 | 39.139585 | 39.143444 | 39.124909 | 39.135979 ± 0.009780 |
| WV2 | 32.116747 | 31.870374 | 32.046910 | 32.011344 ± 0.126979 |

| 센서 | MATLAB − Python PSNR | SAM | ERGAS | SCC | Q2n |
|---|---:|---:|---:|---:|---:|
| QB | +0 | +1.78e-15 | +0 | +2.22e-16 | +1.32e-06 |
| GF2 | +0 | +2.22e-15 | +3.33e-16 | -2.22e-16 | +1.59e-06 |
| WV3 | +0 | +4.44e-16 | +4.44e-16 | -2.22e-16 | -1.12e-05 |
| WV2 | +0 | +8.88e-16 | +8.88e-16 | +2.22e-16 | +0.000118 |

Python 수치는 같은 CPU 예측 배열의 공통 평가기 결과입니다. MATLAB과 Q2n의 작은 구현 차이를 보정하지 않고 각각 보관했습니다. 기존 Kaggle GPU 추론과 CPU 재추론의 장면별 PSNR 최대 차이는 약 0.000011 dB였습니다.

### FR: 기존 Python 참고 지표

| 센서 | U2Net QNR ↑ | Dλ ↓ | Ds ↓ |
|---|---:|---:|---:|
| QB | 0.821972 | 0.043294 | 0.141137 |
| GF2 | 0.941227 | 0.025440 | 0.034266 |
| WV3 | 0.960572 | 0.017416 | 0.022463 |
| WV2 전이 | 0.932447 | 0.033710 | 0.035099 |

FR은 원래 Kaggle 평가에서 보존한 Python 값이며 이번 실제 MATLAB RR 평가와 구분합니다. 고해상도 GT가 없으므로 FR PSNR·SAM을 만들지 않습니다.

<a id="current-interpretation"></a>

## 해석과 정리

현재 SMOC와 비교한 네 센서 전체 평균에서 U2Net의 PSNR은 낮고 SAM·ERGAS는 큽니다. 이전 C2MS-SP와의 GF2 차이(+0.174 dB)를 현재 SMOC 대비 결과로 사용하면 안 됩니다. 비교 기준을 맞추면 GF2 차이는 −0.513 dB입니다. 다른 모델 평가이므로 개선 판정이나 기준 모델 교체 결론은 붙이지 않습니다.

QB·GF2는 validation MSE로 선택한 epoch201·301의 best이며 test로 에폭을 다시 고르지 않았습니다. WV3·WV2는 같은 저자 WV3 가중치를 사용합니다. WV2는 학습하지 않은 센서에 대한 전이 참고 결과이며 SMOC에만 정의된 밴드 보정·사영이 포함됩니다.

장면별로 PSNR이 높은 장면 수는 QB 0/20, GF2 1/20, WV3 3/20, WV2 1/20입니다. SAM이 낮은 장면은 각각 0·4·3·3장, ERGAS가 낮은 장면은 0·3·2·1장입니다. WV3·WV2는 각 장면의 SMOC 3시드 지표 평균과 비교한 횟수이며, 시드 반복의 통계적 검정이나 확대 창의 개선 횟수는 아닙니다.

논문 보고값과 재학습 결과의 차이 원인으로는 4밴드 대응, GF2 정규화, 기울기 누적, full-validation MSE 선택, 데이터·평가 관례가 남아 있습니다. 이번 자료만으로 한 원인을 확정하거나 공식 구현의 한계로 일반화하지 않습니다. U2Net의 두 경로 결합 구조는 참고할 수 있지만, 새 성능 실험을 이 보고서에 혼합하지 않았습니다.

<a id="current-method"></a>

<details>
<summary><b>방법</b> · 학습·추론·연산량·하드웨어</summary>

공식 U2Net은 PAN 공간 경로와 MS 분광 경로를 두 U-Net으로 처리하고 S2Block에서 결합합니다. 제공 LMS를 사용하지 않으며 MS를 내부에서 학습된 연산으로 확대합니다. 4밴드는 공식 생성자의 `ms_dim=8` 상수 한 곳을 어댑터에서 4로 바꾸며 공식 하위 모듈 자체는 고치지 않았습니다.

| 항목 | SMOC | U2Net |
|---|---|---|
| 학습 | 원래 크기·거친 쌍 혼합 150에폭 | QB·GF2 360에폭, 저자 WV3 학습 이력 미상 |
| 배치·정밀도 | 32, 기준 구현 유지 | micro4 × 누적4 = 유효16, FP32·AMP/TF32 끔 |
| 손실·optimizer | 기준 L1·분광·거친 쌍 손실 | L1, Adam 1e-3, StepLR 100에폭마다 ×0.5 |
| 가중치 선택 | validation MSE 최소 | full-validation MSE 최소, epoch0 포함·동률 최초 |
| 입력 | PAN·MS·제공 LMS | PAN·MS, LMS 미사용 |
| 정규화 | QB·WV3·WV2 2047, GF2 1023 | 같음; 저자 원래 GF2 hardcode2047과 차이 공개 |
| 추론 | 전체 장면 + K10 사영, WV2 밴드 보정 | 공식 64px core·halo4, 입력72px·반사 패딩·[0,1] clipping |
| 추가 방법 | 기준 모델 정의 유지 | EMA·분광 손실·사영·센서 보정 없음 |
| 파라미터 수 | 실사용 2,735,418 | 4밴드 632,788; 8밴드 642,008 |
| 연산량 | 256² 59.7 GMAC, 고정 필터 제외 | 전체 attention 연산 및 타일 중복을 포함한 MAC/FLOPs 미측정 |

학습은 Kaggle Tesla T4 두 장에서 GPU0 QB·GPU1 GF2를 각각 독립 실행했습니다(각14.56 GiB). 원본 기록의 누적 학습 시간은 QB 약66.25시간, GF2 약76.76시간이며 중단·재개가 포함된 기록 방식입니다. 동일 하드웨어에서 SMOC와 시간을 통제한 실험은 아닙니다. QB의 원본 GPU peak 0.0063 GiB는 측정 장치 지정 문제로 유효하지 않아 하드웨어 요구량으로 쓰지 않습니다. GF2의 기록2.84 GiB도 최소 요구 메모리를 뜻하지 않습니다.

원래 T4 평가 기록의 장면당 평균 추론 시간은 QB RR 0.783초·FR 1.629초, GF2 RR 0.785초·FR 1.636초입니다. forward와 출력 CPU 이동을 포함하고 입력 이동·지표 계산을 제외하며, 첫 장면 warm-up과 동시 실행의 영향을 포함한 평균입니다. SMOC와 시간을 통제한 비교가 아니므로 처리량 우열을 결론내리지 않습니다.

새 MATLAB 채점은 저장한 best의 CPU FP32 재추론 배열을 사용했습니다. GPU 처리량 비교를 CPU 추론 시간으로 대체하지 않았습니다. 원래 Kaggle 장면별 추론 시간·환경·history는 원본 지표 및 provenance에 있습니다.

smoke는 별도 폴더의2에폭 점검이며 본 학습의1-360 연속 history와 분리했습니다. 원본 `smoke_excluded=false`는 코드에서 smoke_mode 여부를 기록한 이름이 모호한 필드로, 본 학습 기록을 뜻합니다. 원본 필드나 checkpoint 내부 설정은 수정하지 않았습니다.

</details>

<a id="current-figures"></a>

<details>
<summary><b>그림</b> · 학습 곡선과 네 센서 고정6장면</summary>

![QB·GF2 학습·validation 곡선](results/learning.png)

각 센서 0-based 장면0·1·3·7·14·17을 동일하게 사용했습니다. 이미지의 대조군은 모두 **SMOC seed42**입니다. 저자 WV3·WV2의 전체 수치표 대조군은 **3시드 평균**이므로 이미지와 범위가 다릅니다. 확대 창은 **U2Net에 유리한 64×64 창**을 stride16으로 고른 예시입니다. 전체 영상·공통 대비·전체 장면 수치를 함께 보며 확대 창만으로 전체 성능을 결론내리지 않습니다. 이미지의 PSNR 표시는 Python 값이며 공식 RR 표는 MATLAB 값입니다.

### QB · retrain

장면 0
![QB 장면0 전체](results/images_smoc/retrain_QB/full_scene_00.png)
![QB 장면0 확대](results/images_smoc/retrain_QB/compare_scene_00.png)

장면 1
![QB 장면1 전체](results/images_smoc/retrain_QB/full_scene_01.png)
![QB 장면1 확대](results/images_smoc/retrain_QB/compare_scene_01.png)

장면 3
![QB 장면3 전체](results/images_smoc/retrain_QB/full_scene_03.png)
![QB 장면3 확대](results/images_smoc/retrain_QB/compare_scene_03.png)

장면 7
![QB 장면7 전체](results/images_smoc/retrain_QB/full_scene_07.png)
![QB 장면7 확대](results/images_smoc/retrain_QB/compare_scene_07.png)

장면 14
![QB 장면14 전체](results/images_smoc/retrain_QB/full_scene_14.png)
![QB 장면14 확대](results/images_smoc/retrain_QB/compare_scene_14.png)

장면 17
![QB 장면17 전체](results/images_smoc/retrain_QB/full_scene_17.png)
![QB 장면17 확대](results/images_smoc/retrain_QB/compare_scene_17.png)

### GF2 · retrain

장면 0
![GF2 장면0 전체](results/images_smoc/retrain_GF2/full_scene_00.png)
![GF2 장면0 확대](results/images_smoc/retrain_GF2/compare_scene_00.png)

장면 1
![GF2 장면1 전체](results/images_smoc/retrain_GF2/full_scene_01.png)
![GF2 장면1 확대](results/images_smoc/retrain_GF2/compare_scene_01.png)

장면 3
![GF2 장면3 전체](results/images_smoc/retrain_GF2/full_scene_03.png)
![GF2 장면3 확대](results/images_smoc/retrain_GF2/compare_scene_03.png)

장면 7
![GF2 장면7 전체](results/images_smoc/retrain_GF2/full_scene_07.png)
![GF2 장면7 확대](results/images_smoc/retrain_GF2/compare_scene_07.png)

장면 14
![GF2 장면14 전체](results/images_smoc/retrain_GF2/full_scene_14.png)
![GF2 장면14 확대](results/images_smoc/retrain_GF2/compare_scene_14.png)

장면 17
![GF2 장면17 전체](results/images_smoc/retrain_GF2/full_scene_17.png)
![GF2 장면17 확대](results/images_smoc/retrain_GF2/compare_scene_17.png)

### WV3 · author

장면 0
![WV3 장면0 전체](results/images_smoc/author_WV3/full_scene_00.png)
![WV3 장면0 확대](results/images_smoc/author_WV3/compare_scene_00.png)

장면 1
![WV3 장면1 전체](results/images_smoc/author_WV3/full_scene_01.png)
![WV3 장면1 확대](results/images_smoc/author_WV3/compare_scene_01.png)

장면 3
![WV3 장면3 전체](results/images_smoc/author_WV3/full_scene_03.png)
![WV3 장면3 확대](results/images_smoc/author_WV3/compare_scene_03.png)

장면 7
![WV3 장면7 전체](results/images_smoc/author_WV3/full_scene_07.png)
![WV3 장면7 확대](results/images_smoc/author_WV3/compare_scene_07.png)

장면 14
![WV3 장면14 전체](results/images_smoc/author_WV3/full_scene_14.png)
![WV3 장면14 확대](results/images_smoc/author_WV3/compare_scene_14.png)

장면 17
![WV3 장면17 전체](results/images_smoc/author_WV3/full_scene_17.png)
![WV3 장면17 확대](results/images_smoc/author_WV3/compare_scene_17.png)

### WV2 · author

장면 0
![WV2 장면0 전체](results/images_smoc/author_WV2/full_scene_00.png)
![WV2 장면0 확대](results/images_smoc/author_WV2/compare_scene_00.png)

장면 1
![WV2 장면1 전체](results/images_smoc/author_WV2/full_scene_01.png)
![WV2 장면1 확대](results/images_smoc/author_WV2/compare_scene_01.png)

장면 3
![WV2 장면3 전체](results/images_smoc/author_WV2/full_scene_03.png)
![WV2 장면3 확대](results/images_smoc/author_WV2/compare_scene_03.png)

장면 7
![WV2 장면7 전체](results/images_smoc/author_WV2/full_scene_07.png)
![WV2 장면7 확대](results/images_smoc/author_WV2/compare_scene_07.png)

장면 14
![WV2 장면14 전체](results/images_smoc/author_WV2/full_scene_14.png)
![WV2 장면14 확대](results/images_smoc/author_WV2/compare_scene_14.png)

장면 17
![WV2 장면17 전체](results/images_smoc/author_WV2/full_scene_17.png)
![WV2 장면17 확대](results/images_smoc/author_WV2/compare_scene_17.png)

</details>

<a id="current-evidence"></a>

<details>
<summary><b>재현 자료</b> · 코드·가중치·출처·검증</summary>

- [어댑터](model.py) · [설정](config.json) · [공식 하위 모듈](upstream) · [원본 학습 소스](provenance/retrain_source) · [원본 학습 환경](provenance/original_training_environment.json)
- [QB best](checkpoints/retrain_QB_s42/best.pt) · [GF2 best](checkpoints/retrain_GF2_s42/best.pt) · [기존 저자 WV3 best](checkpoints/author_WV3/best.pt)
- [재개 latest·초기 initial·원본 실행 자료 Release](https://github.com/BIYONGHIYON/CtrS/releases/tag/weights-benchmarks-u2net)
- [QB·GF2 RR 예측 Release](https://github.com/BIYONGHIYON/CtrS/releases/tag/predictions-benchmarks-u2net-retrain) · [WV3·WV2 RR 예측 Release](https://github.com/BIYONGHIYON/CtrS/releases/tag/predictions-benchmarks-u2net-author)
- [실제 MATLAB 실행](https://github.com/BIYONGHIYON/CtrS/actions/runs/45298459042) · [집계](results/summary_smoc.json) · [MATLAB 장면별 원본](results/matlab.json) · [MATLAB−Python 차이](results/matlab)
- [QB 학습 기록](results/retrain_QB_s42/history.json) · [GF2 학습 기록](results/retrain_GF2_s42/history.json) · [해시 manifest](manifest.json) · [검증](results/verification_smoc.json)

공식 소스와 원본 학습 소스는 고정·보존했습니다. 저자 가중치를 새 Release로 다시 배포하지 않았으며 기존 저장소에 보관된 파일과 공식 배포 출처를 유지했습니다. 재개용 원본 ZIP은 데이터 H5·입력·GT를 포함하지 않습니다.

```bash
git submodule update --init SSA-MRN/benchmarks/u2net/upstream
python SSA-MRN/common/benchmark_evaluate.py --model u2net --setting retrain --sensor QB --seed 42 --device cuda --data /path/to/test
python SSA-MRN/common/benchmark_evaluate.py --model u2net --setting retrain --sensor GF2 --seed 42 --device cuda --data /path/to/test
python SSA-MRN/common/benchmark_evaluate.py --model u2net --setting author --sensor WV3 --device cuda --data /path/to/test
python SSA-MRN/common/benchmark_evaluate.py --model u2net --setting author --sensor WV2 --device cuda --data /path/to/test
python SSA-MRN/common/verify.py --benchmark u2net --releases
python SSA-MRN/common/index.py --check
```

공통 Python 재평가 명령은 RR·FR H5를 모두 요구합니다. 실제 MATLAB 재채점은 전용 워크플로와 Release 예측을 사용합니다. 원본 학습 재개는 보관된 source/run.py와 원본 config의360에폭 설정을 그대로 사용하며, 이번 정리를 이유로 학습을 다시 실행하지 않았습니다.

</details>

<details>
<summary><b>이전 C2MS-SP 비교 기록</b> · 당시 보고서 원문 보존</summary>

# U2Net: Spatial-Spectral-Integrated Double U-Net

> 현재 비교 기준은 SMOC입니다([최신 평가 표](../README.md#models)). 이 보고서의 C2MS-SP 수치·이미지는 당시 비교 기록이며 그대로 보존합니다.

| 평가자 | 설정 | 상태 |
|:---:|:---:|:---:|
| 송준현 | `author` · 저자 WV3 공개 가중치, 재학습 없음 | WV3·WV2 평가 완료<br>QB·GF2 미평가 |

**출처:** Siran Peng, Chenhao Guo, Xiao Wu, Liang-Jian Deng. *U2Net: A General Framework with Spatial-Spectral-Integrated Double U-Net for Image Fusion*. ACM MM 2023. [논문](https://doi.org/10.1145/3581783.3612084) · [공식 코드](https://github.com/PSRben/U2Net) · 코드와 공개 가중치는 공식 저장소의 GPL-3.0 배포본을 사용했습니다. 고정 버전과 해시는 [config.json](config.json)에 있습니다.

**결과:** PanCollection WV3·WV2의 RR·FR 각20장을 Python 공통 평가기로 채점했습니다. U2Net author의 RR PSNR은 WV3 38.844dB, WV2 전이 30.497dB이며, C2MS-SP **3시드 평균**과의 차이는 각각 +0.171dB, −1.039dB입니다. 고정6장면의 RR 비교 이미지는 **C2MS-SP seed42**와 비교했습니다. QB·GF2와 실제 MATLAB 채점은 수행하지 않았으며, 4개 데이터셋 평가 완료로 집계하지 않습니다.

[평가 목록](../README.md#models) · [수치](#results) · [해석](#verdict) · [방법](#method) · [그림](#figures) · [재현 자료](#evidence)

<a id="results"></a>

## 결과

### RR 전체20장: Python 공통 지표

저자 가중치의 학습 seed는 알려져 있지 않습니다. 저자 가중치 한 개와 C2MS-SP seed42·43·44의 산술평균을 비교하며, 동일 seed로 짝지은 실험이 아닙니다. Δ는 U2Net − C2MS-SP입니다. 수치는 [summary.json](results/summary.json)과 같습니다. Q2n·SCC를 실제 MATLAB으로 재검증하지 않았습니다.

| 센서 | 모델 | PSNR ↑ | SAM ↓ | ERGAS ↓ | SCC ↑ | Q2n ↑ |
|---|---|---:|---:|---:|---:|---:|
| WV3 | C2MS-SP, 3seed 평균 | 38.673246 | 2.984351 | 2.240736 | 0.984305 | 0.910665 |
| WV3 | U2Net, author | 38.844178 | 2.896916 | 2.199791 | 0.985843 | 0.915967 |
| WV3 | Δ | +0.170932 | −0.087434 | −0.040945 | +0.001538 | +0.005302 |
| WV2 전이 | C2MS-SP, 3seed 평균 | 31.535461 | 5.063285 | 3.663853 | 0.938174 | 0.877077 |
| WV2 전이 | U2Net, author | 30.496774 | 5.258539 | 4.076282 | 0.933318 | 0.846848 |
| WV2 전이 | Δ | −1.038687 | +0.195254 | +0.412429 | −0.004855 | −0.030229 |

WV2는 두 모델 모두 WV3 학습 가중치를 사용한 전이 평가입니다. C2MS-SP에는 당시 기준 모델 정의에 따른 밴드별 센서 보정과 관측 일치 사영이 포함되며, U2Net에는 우리 방법을 붙이지 않았습니다.

### 원 논문 보고값과의 차이

| 센서 | 원 논문 U2Net PSNR / SAM / ERGAS / Q8 | 이번 author 평가 | PSNR 차이 |
|---|---|---|---:|
| WV3 | 39.117 / 2.888 / 2.149 / 0.920 | 38.844 / 2.897 / 2.200 / 0.916 | −0.273dB |
| WV2 전이 | 30.740 / 5.250 / 4.070 / 0.849 | 30.497 / 5.259 / 4.076 / 0.847 | −0.243dB |

출처는 [원 논문 Table1·2](https://arxiv.org/html/2212.06466)입니다. 원 논문 수치와 이번 Python 수치가 완전히 일치했다고 쓰지 않습니다. PSNR 집계·경계 및 Toolbox 구현 차이, 공개 가중치의 실제 학습 이력 등이 확인할 원인 후보이며, 현재 원인을 확정하지 않았습니다. 수치를 맞추기 위해 checkpoint나 후처리를 다시 선택하지 않았습니다.

### FR 전체20장: 참고 지표

| 센서 | U2Net QNR ↑ | Dλ ↓ | Ds ↓ |
|---|---:|---:|---:|
| WV3 | 0.960572 | 0.017416 | 0.022463 |
| WV2 전이 | 0.932447 | 0.033710 | 0.035099 |

이는 Python 공통 평가기의 QNR 계열이며 실제 MATLAB 채점 결과가 아닙니다. 고해상도 GT가 없는 FR의 PSNR·SAM이나 정답 이미지는 만들지 않았습니다.

<a id="verdict"></a>

## 해석과 정리

WV3에서 U2Net의 RR PSNR·SAM·ERGAS 수치는 C2MS-SP 3시드 평균보다 높거나 낮은 방향으로 각각 +0.171dB·−0.087°·−0.041의 차이를 보였습니다. 반면 학습하지 않은 WV2에서는 −1.039dB·+0.195°·+0.412였습니다. 이 결과는 공개 가중치·원래 타일 추론을 사용한 한 번의 author 평가이며, 같은 학습 조건의 재학습 비교나 여러 seed 반복 결과가 아닙니다. 다른 모델 평가이므로 개선 판정이나 기준 모델 교체 결론은 붙이지 않습니다.

**수치표와 이미지의 기준 모델 범위가 다릅니다.** 위 전체20장 표는 C2MS-SP **3시드 평균**, 아래 이미지는 C2MS-SP **seed42 단일 모델**입니다. U2Net은 같은 저자 공개 가중치이며 학습 seed 미상입니다. 이미지 여섯 장면의 차이를 전체20장 평균으로 대체하지 않습니다. WV3·WV2의 각6장면에서 U2Net의 PSNR은 기존 전체 평가의 동일 장면 값과 정확히 일치했습니다.

WV3 장면3에서는 건물·수목 주변 차이를, WV2 장면7에서는 전체 영상 차이와 확대 창의 차이를 함께 볼 수 있습니다. WV2 장면7의 전체 PSNR은 C2MS-SP 30.671dB, U2Net 30.134dB이지만, **U2Net에 유리하게 고른 확대 창**의 PSNR은 29.03dB와 29.38dB입니다. 일부 확대 창만 보고 전체 결과를 설명하지 않도록 전체 장면과 장면 수치를 함께 제시합니다. 이 예시의 관찰을 모든 경계·세부 영역의 통계적 결론으로 확대하지 않습니다.

다음 작업은 두 가지입니다. 첫째, 저자 가중치가 없는 QB·GF2를 공식 코드와 논문 설정으로 우리 학습 데이터에서 직접 학습해(`retrain`, seed42, validation 선택) 가중치·예측·결과를 함께 올립니다. 공식 클래스는 MS 8밴드를 고정하고 공개 가중치도 WV3용입니다. 그래서 이 가중치를 잘라 QB·GF2 결과로 쓰지 않고, 4밴드 대응 변경점은 어댑터에 기록합니다. 둘째, WV3·WV2 예측과 새로 학습한 QB·GF2 예측을 Release에 올리고 실제 MATLAB 워크플로로 채점합니다. 그 값을 보고서의 공식 RR 수치로 씁니다. 현재 자료만으로 주간 목표인 한 모델의 4센서 완료를 충족하지 않습니다.

<a id="method"></a>

<details>
<summary><b>방법</b> · 모델과 평가 조건</summary>

U2Net은 PAN의 공간 경로와 MS의 분광 경로를 두 U-Net으로 처리하고, 각 단계의 S2Block에서 결합합니다. 공식 모델 파일과 가중치는 수정하지 않았습니다. [model.py](model.py) 어댑터에서 입력 순서를 `(PAN,LMS,MS)`에서 공식 `(MS,PAN)`으로 맞추며 제공 LMS는 사용하지 않습니다. 내부 학습된 4배 확대와 네트워크 연산은 공식 구현을 그대로 사용합니다.

| 항목 | C2MS-SP | U2Net author |
|---|---|---|
| 학습 출처 | 65번 WV3 seed42·43·44 보관 가중치 | 저자 공개 WV3 `200000_iter.pth` |
| 이번 학습 | 없음 | 없음 |
| 입력 | PAN·제공 LMS·MS | PAN·MS; 제공 LMS 미사용 |
| 추론 | 전체 장면, K10 관측 일치 사영 | 공식 test.py 방식 64px 타일·halo4, 72px 입력 |
| clipping | 기준 모델 구현 유지 | 공식 test.py의 [0,1] clipping 유지 |
| WV2 | WV3 가중치 + 장면별 밴드 보정 + WV2 관측 사영 | WV3 가중치 그대로, 보정·사영 미적용 |
| 정규화 | 두 센서 DN/2047 | 같음 |
| 전체 수치 비교 | 3seed 평균 | 저자 모델1개, seed 미상 |
| 이미지 비교 | seed42 | 같은 저자 가중치 |
| 실행 장치 | 이미지 비교 Kaggle Tesla T4 | 수치·이미지 Kaggle Tesla T4 |

저자 가중치는 validation으로 우리가 다시 선택한 checkpoint가 아닙니다. 파일명은 200000iter이며 실제 학습 로그와 epoch가 제공되지 않아 확정하지 않습니다. 원 논문의 360epoch·batch16·Adam·L1·초기 lr0.001·100epoch마다0.5 감소, 공식 train.py 기본500epoch와 구분합니다. 논문 학습 장치는 GPU-3090으로 기술돼 있으나 이번에 재학습하거나 필요한 최소 VRAM·학습시간을 측정하지 않았습니다.

U2Net 등록 파라미터는 **642,008개**입니다(미사용 등록층 포함). 전체 MAC/FLOPs와 순수 GPU 추론시간·peak VRAM은 아직 측정하지 않았습니다. 합성72² 타일의 Conv·Linear·ConvTranspose 부분합0.694GMAC는 attention 행렬곱과 타일 중복을 제외하므로 총 연산량으로 보고하지 않습니다. 두 모델의 입력 처리·학습·추론 조건이 다르다는 한계를 유지합니다.

</details>

<a id="figures"></a>

<details>
<summary><b>그림</b> · 전체 수치 비교와 고정 장면</summary>

재학습하지 않았으므로 이번 학습 곡선은 없습니다. 아래 첫 그래프는 전체20장 U2Net과 C2MS-SP 3seed 평균의 비교입니다. 축을 확대했으며, 두 번째는 고정6장면에서 C2MS-SP seed42와의 차이입니다.

![전체20장 수치 비교](results/numeric_comparison.png)
![고정6장면 seed42 비교](results/fixed_scene_deltas.png)

장면 번호는 두 센서 모두 0-based **0·1·3·7·14·17**로 고정했습니다. QB에서 풍경별로 정한 번호를 그대로 적용했으나, QB 풍경 이름을 WV3·WV2에 붙이지 않습니다. 원본 패널은 **LR MS · PAN · C2MS-SP · U2Net · GT**, 아래는 같은 색 척도의 두 오차 지도입니다. RGB[4,2,1], 장면 GT 1-99% 공통 대비이며 PAN은 별도 흑백 대비입니다.

확대 패널은 GT 전체의 노란 상자 · C2MS-SP · U2Net · GT · 두 오차 지도입니다. **U2Net의 오차 감소가 가장 큰 64×64 창을 사후 선택하고 4배 확대**했으므로 U2Net에 유리한 예시입니다. 모든 장면의 전체 PSNR을 함께 표시합니다.

### WV3 고정 장면0

![WV3 장면0 전체](results/images/author_WV3/full_scene_00.png)
![WV3 장면0 확대](results/images/author_WV3/compare_scene_00.png)

### WV2 전이 고정 장면0

![WV2 장면0 전체](results/images/author_WV2/full_scene_00.png)
![WV2 장면0 확대](results/images/author_WV2/compare_scene_00.png)

나머지 장면의 원본·확대 이미지는 아래에 모두 표시합니다.

#### WV3 장면1

![WV3 장면1 전체](results/images/author_WV3/full_scene_01.png)
![WV3 장면1 확대](results/images/author_WV3/compare_scene_01.png)

#### WV3 장면3

![WV3 장면3 전체](results/images/author_WV3/full_scene_03.png)
![WV3 장면3 확대](results/images/author_WV3/compare_scene_03.png)

#### WV3 장면7

![WV3 장면7 전체](results/images/author_WV3/full_scene_07.png)
![WV3 장면7 확대](results/images/author_WV3/compare_scene_07.png)

#### WV3 장면14

![WV3 장면14 전체](results/images/author_WV3/full_scene_14.png)
![WV3 장면14 확대](results/images/author_WV3/compare_scene_14.png)

#### WV3 장면17

![WV3 장면17 전체](results/images/author_WV3/full_scene_17.png)
![WV3 장면17 확대](results/images/author_WV3/compare_scene_17.png)

#### WV2 장면1

![WV2 장면1 전체](results/images/author_WV2/full_scene_01.png)
![WV2 장면1 확대](results/images/author_WV2/compare_scene_01.png)

#### WV2 장면3

![WV2 장면3 전체](results/images/author_WV2/full_scene_03.png)
![WV2 장면3 확대](results/images/author_WV2/compare_scene_03.png)

#### WV2 장면7

![WV2 장면7 전체](results/images/author_WV2/full_scene_07.png)
![WV2 장면7 확대](results/images/author_WV2/compare_scene_07.png)

#### WV2 장면14

![WV2 장면14 전체](results/images/author_WV2/full_scene_14.png)
![WV2 장면14 확대](results/images/author_WV2/compare_scene_14.png)

#### WV2 장면17

![WV2 장면17 전체](results/images/author_WV2/full_scene_17.png)
![WV2 장면17 확대](results/images/author_WV2/compare_scene_17.png)


</details>

<a id="evidence"></a>

<details>
<summary><b>재현 자료</b> · 코드·가중치·수치·검증</summary>

- [공식 upstream](upstream) · [어댑터](model.py) · [설정](config.json)
- [저자 WV3 best 원본](checkpoints/author_WV3/best.pt): WV2도 이 가중치를 사용합니다. 우리가 학습한 latest·initial은 없으므로 Release 자산을 만들지 않았습니다.
- [WV3 전체20장 수치](results/author_WV3/metrics.json) · [WV2 전체20장 수치](results/author_WV2/metrics.json) · [집계](results/summary.json)
- [WV3 이미지·장면 수치](results/images/author_WV3/comparison_summary.json) · [WV2 이미지·장면 수치](results/images/author_WV2/comparison_summary.json)
- [평가 소스](provenance/evaluation_source) · [이미지 생성 당시 원본 소스](provenance/image_source) · [이미지 실행 환경](provenance/image_environment.json)
- [검증](results/verification.json) · [보관 manifest](manifest.json) · [원본 ZIP 출처와 해시](provenance/archive_sources.json)

원본 이미지 ZIP의 manifest75개와 이미지24개의 해시·PNG 형식, RR·FR 전체20장 평균을 확인했습니다. U2Net 동일 장면 PSNR 재현 오차는12장 모두0dB입니다. 보관 과정에서 실제 H5 재추론·재학습·MATLAB 평가를 수행하지 않았습니다.

실제 재평가에는 공식 하위 모듈과 H5 데이터가 필요합니다.

```bash
git submodule update --init SSA-MRN/benchmarks/u2net/upstream
python SSA-MRN/common/benchmark_evaluate.py --model u2net --setting author --sensor WV3 --device cuda
python SSA-MRN/common/benchmark_evaluate.py --model u2net --setting author --sensor WV2 --device cuda
python SSA-MRN/common/benchmark_evaluate.py --model u2net --summary
python SSA-MRN/common/verify.py --benchmark u2net
```

이미지의 기존 생성 조건은 보존된 `provenance/image_source/make_comparisons.py`와 package manifest에 있습니다. 이 원본 소스는 학습·평가 당시 경로를 포함하므로 현행 폴더명에 맞추려고 내용을 바꾸지 않았습니다. 기준 모델 가중치는 기존65번 경로를 참조하며 이 보고서 폴더에 중복 보관하지 않습니다.

</details>


</details>
