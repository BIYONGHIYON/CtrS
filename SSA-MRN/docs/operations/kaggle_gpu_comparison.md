# Kaggle 원본·변경 모델 비교

> **과거 K6 실험 재현 안내입니다.** 현재 신규 연구 기본값은 [K4](../../README.md#decision)입니다. 아래 고정 계획은 당시 실험 재현용이며 신규 K4 계획으로 혼용하지 않습니다.

[한 셀 노트북](../../scripts/kaggle_band_gated_hf.ipynb) · [동일한 Python 셀](../../scripts/kaggle_one_cell.py) · [완료 결과](../experiments/kaggle_band_gated_hf.md)

Kaggle에서 GPU T4 x2와 Internet을 켜고 PanCollection H5 데이터셋을 연결한 후 노트북의 한 셀을 실행한다. 이전 학습을 중지하고 커널을 재시작한 뒤 실행한다. GPU가 하나면 두 실험은 순차 실행한다.

## 비교 대상

GPU 0은 GitHub의 `RestoredPansharpeningNet` 원본 baseline, GPU 1은 [지정된 실험 계획](https://github.com/BIYONGHIYON/CtrS/blob/ssa-band-gated-hf-plan/SSA-MRN/docs/experiments/kaggle_band_gated_hf.md)의 `band_gated_hf`를 실행한다. 사용자 요청에 따라 기존 고정 `high_frequency`는 실행 대상에서 제외한다. `SWAP_GPUS=True`로 새 출력 폴더에서 반복하면 두 물리 GPU 배정을 바꿀 수 있다.

원본은 be02b7cbf03fe00681194ae9fc9419661f24cd56, 후보 정의는 계획 브랜치의 1b56efb54e9c200d5b497e719bb0857b25f750bf에 고정한다. 후보 클래스와 builder/config 함수는 해당 노트북의 AST에서 정의만 추출한다. 노트북의 학습 셀을 실행하거나 임의의 gate를 만들어 대체하지 않는다. SSA-MRN adapter는 공식 upstream의 미정의 `ms`를 복구한 저장소 로직이며, 공식 submodule은 수정하지 않는다.

후보는 PAN의 5×5 binomial 고주파와 기준 출력의 밴드별 `|dx|·|dy|`를 함께 입력받는다. 12→32→4 gate가 밴드·픽셀별 sigmoid 주입량을 결정하고, PAN 1→16→4 잔차를 곱해 더한다. 기준 출력의 기울기는 원본 정의대로 detach한다. 잔차 마지막 Conv는 0, gate 마지막 weight는 0/bias는 −2로 초기화한다.

## 공통 조건과 속도

- QB, K6, 100에폭, seed42, effective batch32, micro batch32, Adam LR1e-4.
- 두 모델군에 동일한 AMP FP16/GradScaler, channels-last, GPU 캐시 또는 mmap 배치 공급을 적용한다. validation과 test는 FP32다.
- `CONTROLLED_COMPARISON=True`에서는 micro batch32, AMP, layout을 동일하게 고정하고 deterministic 연산과 TF32 비활성화를 두 쪽에 적용한다. `False`는 모델별 micro batch 속도를 측정하고 cuDNN autotuning을 사용하므로 수치 조건이 더 달라질 수 있다.
- 계획의 CUDA FP32·micro batch4에서 공통 AMP·micro32로 변경한 실행이다. 계획의 모델 로직·MSE·effective batch32·Adam LR1e-4는 유지한다. checkpoint의 `reference_plan_config`는 원래 선언이며 실제 실행 조건은 `amp`, `micro_batch` 등 별도 필드에 기록한다.
- 본체 초기 가중치 SHA-256와 epoch별 데이터 순서 SHA-256가 두 모델군에서 같은지 검사한다. seed42가 기본이고 `[42,43,44]`로 늘리면 각 seed에서 두 모델을 비교한다.
- 원래 FP32 코드와 bitwise 일치를 주장하지 않는다. 성능 차이는 같은 새 환경에서 학습한 원본 기준선과 비교한다.
- H5는 한 번 검증·정규화한 뒤 디스크 캐시를 만들고, GPU 여유 메모리의 45% 안에 들어가면 train/val 전체를 각 GPU에 적재한다. 들어가지 않으면 mmap, worker2, pinned memory, 비동기 CUDA 전송으로 전환한다.
- micro32 메모리 검사가 실패하면 `GPU_CACHE=False`로 다시 실행하거나 두 모델군의 micro batch를 동일하게 줄여 새 출력 폴더에서 실행한다. 임의로 한쪽 배치만 줄이지 않는다.
- 학습 시작 전 H5 SHA-256 계산과 캐시 구축 비용이 있다. 원본 데이터는 변경하지 않는다. 과거 결과 폴더도 삭제하지 않는다.

H5 파일명이 유일하면 `/kaggle/input`에서 자동 검색한다. `QuickBird__Training_Dataset__train_qb.h5`처럼 폴더명이 파일명에 합쳐진 경우 마지막 `__` 이후의 이름으로 비교하며 `._train_qb.h5` 메타데이터는 제외한다. 검색 실패나 중복이면 `TRAIN_H5`, `VAL_H5`, `RR_H5`, `FR_H5`를 실제 경로로 지정한다. 테스트 데이터까지 준비됐는지 학습 전에 검사한다.

## 결과와 재개

출력은 `/kaggle/working/ssa_kaggle_model_comparison_v1`이며 기존 학습 폴더와 분리된다. 각 run의 `latest.pt`와 validation MSE가 개선된 `best.pt`에 실제 model, optimizer, scaler, RNG, epoch 기록을 저장한다. 같은 데이터·코드·설정으로 같은 출력 폴더에서 재실행하면 완료 epoch부터 이어간다. 예전 노트북의 checkpoint를 자동으로 가져오지 않는다.

best로 RR/FR 전체 장면을 평가한 뒤 다음 산출물을 저장한다.

- 장면별 RR/FR JSON, 전체 평균 `results.json`, 비교 `summary.csv`, baseline 대비 `paired_comparison.json`.
- 실제 학습 MSE·validation PSNR/SAM 곡선, test 기준선 비교 그래프.
- 점수 계산 전에 seed42로 고정한 서로 다른 RR 5장면의 LR MS · PAN · 예측 · GT 4패널과 selection manifest.
- 기본 설정에서는 모든 RR/FR 장면의 원해상도 다중밴드 예측 배열 `.npz`도 보존한다. FR에는 GT를 생성하지 않는다.
- 코드 revision·소스·데이터·best/latest와 결과 파일 SHA-256 manifest, 보고서, 결과 ZIP 및 ZIP SHA-256.

캐시와 임시 모니터 로그는 ZIP에서 제외하며 실제 best/latest, 재현 소스와 설정은 포함한다. ZIP의 CRC도 검사한다. Kaggle 출력은 세션 밖의 영구 보관을 자동으로 보장하지 않으므로 노트북 출력으로 저장해야 한다. GitHub 푸시나 원격 보관이 이루어진 것으로 표시하지 않는다.

FR QNR/D_s 및 RR Q2n/SCC는 저장소의 MATLAB 정합성 미검증 제한을 유지한다. 저장소 PSNR은 센서에 관계없이 peak2047을 쓰며 센서 peak 기준 PSNR은 별도 필드로 기록한다. validation PSNR은 peak1 기반 밴드별 PSNR 평균이고 SAM은 각 이미지의 유효 픽셀 평균을 이미지별로 평균한다. 전체 평가 장면 수를 결과에 기록하며 계획의 RR/FR 각 20장과 다르면 보고서에 차이를 명시한다.

epoch 시간 비교는 학습+validation만 포함한다. 캐시 구축·checkpoint·평가를 포함한 이번 실행 wall time은 `session_timing.json`에 따로 저장하며, 재개한 세션 시간을 전체 실험 시간으로 표시하지 않는다. 모니터는 실제 `nvidia-smi` GPU 사용률·메모리와 PyTorch allocated/peak 메모리를 표시한다. 고정 Stage2 138분이나 검증하지 않은 총 완료 시각은 표시하지 않는다.

## 현재 검증 범위

Kaggle에서 두 모델 모두 100에폭 학습 및 RR/FR 각 20장 평가를 완료했다. 다운로드한 149개 산출물의 해시·checkpoint epoch·장면별 숫자·예측 80개·본체 초기화와 데이터 순서를 로컬에서 검증했다. 이 환경에는 PyTorch/CUDA와 원본 H5가 없어 로컬 재학습·GT 지표 재산출은 하지 않았다.

```bash
python SSA-MRN/scripts/verify_kaggle_result.py --input SSA-MRN/docs/assets/kaggle_band_gated_hf --cell SSA-MRN/scripts/kaggle_one_cell.py
```
