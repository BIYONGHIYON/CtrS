# CANNet

> 현재 비교 기준은 SMOC입니다([최신 평가 표](../README.md#models)). 이 보고서의 C2MS-SP 수치·이미지는 당시 비교 기록이며 그대로 보존합니다.

**출처:** Yule Duan, Xiao Wu, Haoyu Deng, Liang-Jian Deng, Content-Adaptive Non-Local Convolution for Remote Sensing Pansharpening, CVPR, 2024, [arXiv 2404.07543](https://arxiv.org/abs/2404.07543)

**공식 코드:** [Duanyll/CANConv](https://github.com/Duanyll/CANConv) · `0b7cc6f`(kmcuda 하위 모듈 `6906cc3`) · GPL-3.0(코드·저자 가중치 재배포 가능)

**평가자:** 이병현

**평가 설정:** `author`만 평가했습니다. 2026-11-17 사용자 결정으로, 저자 가중치가 공개된 모델은 재학습하지 않고 저자 가중치로 저자 평가를 재현합니다.

| 확인된 결과 | 상태 |
|---|---|
| 저자 가중치 QB RR PSNR 38.261 · SAM 4.552 · ERGAS 3.736. WV3는 논문 SAM·ERGAS와 소수점 셋째 자리 수준으로 일치했고, QB·GF2는 SAM·ERGAS가 논문보다 조금 나쁩니다. C2MS-SP 3시드 평균 대비 QB −0.112dB, GF2 +0.489dB, WV3 +0.304dB입니다. | 평가 완료 |

[평가 표](../README.md#models) · [수치](#results) · [그래프](#graphs) · [이미지](#images) · [재현 자료](#evidence)

## 1. 목적과 상태

CANNet은 특징 지도의 화소를 k-means로 묶고, 묶음마다 다른 합성곱 커널을 적용하는 내용 적응 비국소 합성곱(CANConv)을 U자 구조에 넣은 모델입니다. 저자가 QB·GF2·WV3 가중치를 모두 공개해서, 학습 없이 우리 test 장면·지표로 잴 수 있습니다. 확인하려는 질문은 두 가지입니다. ① 저자 가중치로 논문 보고값이 재현되는가, ② 같은 test·지표에서 C2MS-SP와 수치가 어떻게 다른가입니다.

| 설정 | 센서 | 상태 (2026-11-17 KST) |
|---|---|---|
| `author` | QB·GF2·WV3, WV2(WV3 가중치 전이, 참고) | Kaggle 추론·Python 지표·실제 MATLAB 채점 완료 |

재학습(`retrain`)은 하지 않습니다. 2026-11-16 Kaggle에서 공식 설정(500에폭)으로 시작했다가 시드 하나에 GPU 약 37시간이 걸려 사용자 결정으로 중단했습니다.

## 2. 모델과 비교 조건

입력 PAN과 4배 확대 MS(LMS)를 이어 붙여 3×3 합성곱으로 32채널 특징을 만듭니다. CAN 잔차 블록 → 내림 → CAN 잔차 블록 → 내림 → CAN 잔차 블록 → 올림 → CAN 잔차 블록 → 올림 → CAN 잔차 블록을 거친 뒤 3×3 합성곱으로 잔차를 만들어 LMS에 더합니다. CAN 잔차 블록의 합성곱은 화소 특징을 32개 묶음으로 k-means 군집화하고 묶음별 커널로 계산합니다. 추론할 때는 장면마다 k-means를 새로 돌리므로 같은 가중치라도 결과가 아주 조금 달라질 수 있습니다.

어댑터(`model.py`)는 공식 `CANNet`을 수정 없이 감싸며 `forward(pan, lms, ms)`에서 `ms`를 쓰지 않습니다. 입력과 출력은 DN을 정규화 값으로 나눈 값입니다. 공식 코드는 고치지 않았습니다. 네이티브 연산(`pwacnative`, `libKMCUDA`)은 CUDA 전용이라 Kaggle T4에서 빌드해 실행했고(`CUDA_ARCH=75`), kmcuda가 여러 GPU 사이 peer access를 요구해 프로세스마다 GPU 하나만 보이게 했습니다.

| 항목 | C2MS-SP | CANNet (`author`) |
|---|---|---|
| 학습 데이터·분할 | PanCollection 학습·validation, 3시드(42·43·44) | 저자 학습(분할·시드 미공개), 학습하지 않음 |
| test 장면·정규화 | RR·FR 각 20장, QB·WV3·WV2 2047, GF2 1023 | 같음(RR test 파일 SHA-256이 공개 PanCollection과 같음) |
| 지표 코드 | `common/metrics`(DLPan 정의) | 같음 + 실제 MATLAB(DLPan) |
| 후처리 | 관측 일치 사영(K10) | 없음 |
| 파라미터 수 | | 785,118 (QB·GF2) |

<a id="results"></a>

## 3. 정량 결과

### 3.1 저자 가중치 RR test 20장 (CtrS 어댑터 경로, Python 지표)

| 센서 | PSNR | SAM | ERGAS | SCC | Q2n | C2MS-SP 3시드 평균과의 차이 (PSNR / SAM / ERGAS) |
|---|---:|---:|---:|---:|---:|---|
| QB | 38.261 | 4.552 | 3.736 | 0.9821 | 0.9350 | −0.112 / +0.078 / +0.051 |
| GF2 | 49.274 | 0.721 | 0.651 | 0.9906 | 0.9822 | +0.489 / −0.042 / −0.040 |
| WV3 | 38.977 | 2.925 | 2.160 | 0.9853 | 0.9145 | +0.304 / −0.060 / −0.081 |
| WV2 (WV3 가중치 전이, 참고) | 29.957 | 5.492 | 4.347 | 0.9159 | 0.8392 | −1.578 / +0.429 / +0.683 (C2MS-SP는 밴드별 보정 포함) |

저자 가중치는 시드가 없어 같은 시드 짝 비교가 아니라 C2MS-SP 3시드 평균과의 차이입니다. 판정은 붙이지 않습니다. FR(참고): QB QNR 0.900, GF2 0.932, WV3 0.962, WV2 0.907([summary.json](results/summary.json)).

### 3.2 논문 보고값 재현 (공식 test 스크립트, 3회 반복)

| 센서 | PSNR (3회 평균 [최소, 최대]) | SAM (논문) | ERGAS (논문) | Q2n (논문 Q4/Q8) |
|---|---|---|---|---|
| QB | 38.266 [38.266, 38.267] (비교표 38.488) | 4.552 (4.507) | 3.734 (3.652) | 0.935 (0.937) |
| GF2 | 49.278 [49.276, 49.279] (비교표 49.52) | 0.721 (0.707) | 0.651 (0.630) | 0.982 (0.983) |
| WV3 | 38.978 [38.977, 38.978] (비교표 39.003) | 2.925 (2.930) | 2.160 (2.158) | 0.915 (0.920) |
| WV2 (참고) | 29.964 [29.963, 29.965] | 5.492 | 4.344 | 0.839 |

CANNet 논문은 PSNR을 보고하지 않아, PSNR은 다른 논문(S2WMamba·FusionMamba)의 CANNet 비교값과 나란히 적었습니다. 공식 경로(`canconv.scripts.test.run_test`)와 CtrS 어댑터 경로의 PSNR 차이는 0.005dB 이하입니다.

### 3.3 실제 MATLAB 채점 (같은 예측 배열)

공식 test 스크립트 1회차 예측을 GitHub Actions에서 실제 MATLAB R2025b와 DLPan-Toolbox(`a34f884`, 실험 53·54 재평가와 같은 `score_suite.m`)으로 채점했습니다. 원본 test 파일은 SHA-256을 확인하고 받았으며, 자르기·값 자르기는 쓰지 않았습니다([작업 실행](https://github.com/BIYONGHIYON/CtrS/actions/runs/43897177708)).

| 센서 | PSNR (밴드 평균, peak 2047) | SAM | ERGAS | SCC | Q2n (Q4/Q8) | Q (밴드 평균) | 논문 (SAM / ERGAS / Q) |
|---|---:|---:|---:|---:|---:|---:|---|
| QB | 38.266 | 4.5520 | 3.7342 | 0.9822 | 0.9350 | 0.9348 | 4.507 / 3.652 / 0.937 |
| GF2 | 49.278 | 0.7212 | 0.6511 | 0.9906 | 0.9822 | 0.9830 | 0.707 / 0.630 / 0.983 |
| WV3 | 38.978 | 2.9249 | 2.1602 | 0.9853 | 0.9145 | 0.9185 | 2.930 / 2.158 / 0.920 |
| WV2 (참고) | 29.965 | 5.4916 | 4.3433 | 0.9160 | 0.8392 | 0.8408 | — |

같은 배열에서 MATLAB과 Python의 차이는 모든 센서·지표에서 0.00003 이하입니다. GF2 PSNR은 센서 peak 1023 기준으로 43.254dB이며, Python과 비교표처럼 peak 2047로 환산하면 49.278dB입니다. 논문의 Q4/Q8 값은 DLPan의 밴드 평균 Q와 더 가깝습니다(GF2 0.9830 vs 0.983, WV3 0.9185 vs 0.920).

<a id="interpretation"></a>

## 4. 논문 보고값·C2MS-SP와의 차이와 해석

- **WV3:** SAM 2.925(논문 2.930), ERGAS 2.160(2.158)으로 재현됩니다.
- **QB·GF2:** SAM·ERGAS가 논문보다 조금 나쁘고(QB ERGAS +0.082, GF2 +0.021), PSNR도 비교표보다 약 0.23dB 낮습니다. 다음 원인은 확인으로 배제했습니다.
  - **test 파일:** RR test 파일의 SHA-256이 공개 PanCollection(HuggingFace `elsting/PanCollection` main)과 같습니다.
  - **추론 경로:** 저자의 test 스크립트로 돌려도 어댑터 경로와 0.005dB 이내로 같습니다.
  - **k-means 비결정성:** 3회 반복의 흔들림이 0.003dB 이하입니다.
  - **지표 코드:** 실제 MATLAB(DLPan)과 Python이 같은 배열에서 0.00003 이내로 일치합니다(3.3절).
  남은 원인 후보는 저장소에 공개된 QB·GF2 가중치가 논문 표를 만든 가중치와 다르거나, 논문의 평가 설정(경계 자르기·값 자르기 등)이 다른 경우입니다. 숫자는 보정하지 않습니다.
- **C2MS-SP와의 차이:** 같은 test·지표에서 CANNet 저자 가중치는 QB에서 C2MS-SP보다 PSNR이 0.112dB 낮고, GF2·WV3에서는 0.489dB·0.304dB 높습니다. 저자 학습 조건(에폭 500, 시드)과 우리 조건이 달라 같은 조건 비교는 아닙니다. 가져올 아이디어(내용 적응 합성곱 등)는 성능 개선 연구의 새 실험으로 등록합니다.

<a id="graphs"></a>

## 5. 학습·평가 그래프

`author` 설정만 평가해 학습 곡선이 없습니다.

<a id="images"></a>

## 6. 결과 이미지

이번 정리에서는 만들지 않았습니다. 고정 6장면 이미지는 `common/benchmark_evaluate.py --figures`(GPU 필요)로 추가합니다.

<a id="evidence"></a>

## 7. 재현 자료

- 공식 코드: `upstream/`(하위 모듈, 고정 커밋). 저자 가중치 `upstream/weights/cannet_{qb,gf2,wv3}.pth`, SHA-256은 [config.json](config.json)에 있습니다.
- Kaggle 실행 원본: [provenance/kaggle_cannet_repro.py](provenance/kaggle_cannet_repro.py)(percent 형식, `# %%`가 한 셀). Kaggle 노트북 `ctrs-cannet-author-repro` 버전 1, GPU T4, torch 2.11.0+cu130, 14.6분.
- 장면별 지표: `results/author_<센서>/metrics.json`(RR·FR, 가중치·데이터 해시), 재현 요약 [results/reproduction.json](results/reproduction.json)(데이터 해시, 3회 반복 값, 논문값), [results/summary.json](results/summary.json).
- 예측 배열: GitHub Release [`predictions-benchmarks-cannet-author`](https://github.com/BIYONGHIYON/CtrS/releases/tag/predictions-benchmarks-cannet-author)(공식 test 스크립트 1회차 RR 예측, 센서별 zip).
- MATLAB 채점: [.github/workflows/matlab-benchmark-author.yml](../../../.github/workflows/matlab-benchmark-author.yml), [results/matlab.json](results/matlab.json).
- 보관 파일 해시: [manifest.json](manifest.json), `python SSA-MRN/common/verify.py --benchmark cannet`.

## 8. 정리와 다음 단계

CANNet 저자 가중치는 WV3에서 논문값을 재현했고, QB·GF2에서는 데이터·추론 경로·지표(실제 MATLAB 포함)를 맞춰도 논문보다 조금 낮게 나왔습니다. 같은 test·지표에서 C2MS-SP 3시드 평균과의 RR PSNR 차이는 QB −0.112dB, GF2 +0.489dB, WV3 +0.304dB입니다. 판정은 붙이지 않으며 이 결과로 기준 모델을 바꾸지 않습니다.
