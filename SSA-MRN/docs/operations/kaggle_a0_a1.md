# Kaggle QB A0·A1 한 셀 실행

[Python 셀](../../scripts/kaggle_a0_a1_one_cell.py) · [한 셀 노트북](../../notebooks/kaggle_a0_a1_one_cell.ipynb)

Kaggle에서 T4 x2와 Internet을 켜고 기존 QB H5 데이터셋을 연결한다. 새 커널에서 노트북의 한 셀을 실행하거나 Python 파일 전체를 한 셀에 붙여 넣는다. 기본 출력은 `/kaggle/working/qb_wavelet_A0_A1_v1`이다. 경로가 달라지면 파일명으로 유일한 H5를 검색하며 중복이면 셀 상단 경로를 지정한다. 원본 데이터와 기존 결과는 수정·삭제하지 않는다.

GPU 0은 A0, GPU 1은 A1을 독립 프로세스로 동시에 학습한다. A0은 gate=1, A1은 밴드×Haar 방향×위치별 sigmoid gate다. 원본 LR MS에서 시작해 서로 독립적인 두 ×2 단계가 MS 기반 LL·고주파를 예측하고 MS/PAN 공동 조건의 PAN 잔차를 주입한다. PAN은 두 단계의 고정 Haar DWT로 분해하며 합성은 IWT다. LL은 `2×현재 MS+학습 잔차`이며 GT의 LL로 가정하지 않는다. 모든 밴드를 일반 Conv로 함께 처리한다. PAN 잔차 head는 0, A1 gate는 weight=0/bias=−2로 초기화한다. 분광 보조 손실·정합·관측 보정·23탭은 추가하지 않는다.

기본값은 현재 저장소의 원본 Kaggle 비교 셀에 맞춘 QB·100에폭·seed42·effective/micro batch32·Adam foreach LR1e-4·betas(0.9,0.999)·eps1e-8·weight_decay0·스케줄러 없음·final MSE·AMP FP16·channels-last·결정론 설정·TF32 비활성화다. validation MSE로 best를 선택하고 전체 RR/FR을 FP32로 평가한다. RR PSNR은 peak2047 밴드별 평균이며 예측을 클리핑하지 않는다. FR에는 GT가 없다. 제공 LMS는 A 학습 입력에 쓰지 않고 FR 지표에만 사용한다. 원본 MS 입력 경로와 모델 용량의 차이는 구조 변경의 일부다. A에 K6를 붙여 원 논문 구조라고 표시하지 않는다.

CPU와 CUDA의 `randperm`은 동일 seed라도 순서가 다르다. 기본값은 기존 B0의 CPU 순서를 시작 전에 전 에폭에 대해 한 번 생성해 GPU에 올린다. 실제 배치는 GPU 인덱싱으로 생성하고 학습 루프에 CPU 샘플러나 전송이 없다. 원본 B0와 순서까지 맞출 필요가 없는 별도 실험은 `ORDER_MODE="gpu"`로 순서 생성도 GPU에서 수행할 수 있다. train/val의 PAN·원본 MS·GT는 각 GPU에 FP32로 상주시키고 사용하지 않는 LMS는 적재하지 않는다. batch마다 `.item()`·CPU 전송·명시적 CUDA 동기화를 하지 않는다. AMP와 channels-last로 Tensor Core 사용을 유도하며 실제 이용률이나 속도 향상은 Kaggle 실행 후 확인해야 한다.

`COMPILE=False`와 `ADAM_IMPL="foreach"`는 기존 B0의 수치 조건을 유지하기 위한 기본값이다. `COMPILE=True`는 reduce-overhead 컴파일을, `ADAM_IMPL="fused"`는 fused Adam을 사용하는 별도 속도 프로필이다. preflight는 forward/backward와 optimizer를 실제 수행하고 초기 모델·optimizer·scaler를 복원한 뒤 본 학습을 시작한다. 컴파일 비용과 warm-up 시간을 기록한다. 실패 시 임의로 한쪽만 실행 모드를 바꾸지 않는다. OOM이면 두 모델의 MICRO_BATCH를 함께 낮추고 새 출력 폴더에서 시작한다. 기존 B0의 micro batch와 달라지면 새 B0 대조군이 필요하다.

## 기존 원 논문 기준선과 통제

기존 Kaggle의 복구 SSA-MRN K6 baseline 산출물을 Kaggle 데이터셋으로 연결하고, 셀 상단에 `BASELINE_RUNS={42: "/kaggle/input/.../runs/github_QB_baseline_k6_s42"}`를 지정한다. best/latest만이 아니라 원본 결과 디렉터리의 `code`, `environment.json`, `artifact_manifest.json`, run별 `results.json`, `history.json`을 함께 유지한다. 시드를 늘리면 같은 시드의 B0 결과가 각각 필요하다.

셀은 train/val/RR/FR SHA256·shape, 설정, 실제 optimizer 상태, 완료 epoch, best epoch·해시, 학습 이력, 원본 trainer 해시, metric 소스 해시, PyTorch/CUDA·GPU 종류를 검증한다. 기존 원본 비교 셀의 worker만 현재 감사 대상으로 허용하며 다른 실행 셀은 설정·소스 검토 후 감사 코드를 추가해야 한다. A 학습 전에 설정 불일치를 차단하고, 학습 시 기존 baseline의 epoch별 실제 순서 해시까지 확인한다. 감사된 B0에 대해서만 A0/B0·A1/B0 차이와 B0 포함 그래프를 만든다. B0의 과거 시간과 A의 현재 GPU 시간을 성능 속도 비교에 섞지 않는다.

`BASELINE_RUNS={}`이면 A0/A1은 실행하지만 `baseline_control.json`에 B0 미검증을 명시하며 B0 개선량을 계산하지 않는다. 저장소 기본 설정과 사용자가 실제 실행한 설정이 같다는 확인 없이 원 논문 대비 변인통제가 완료됐다고 쓰지 않는다. 논문의 출판 표 숫자는 외부 참고값이며 동일 분할·실행·지표의 paired baseline과 다르다. FR QNR/Ds, RR Q2n/SCC의 MATLAB 정합성 미검증 제한도 유지한다. 새 구조의 공통 초기화는 A0/A1끼리 확인하며, 다른 구조인 B0와 동일 초기 가중치를 주장하지 않는다.

## 저장과 검증 범위

best는 validation 개선 시 즉시, latest는 기본 5에폭마다 및 마지막 에폭에 저장한다. 같은 출력 폴더와 동일 설정으로 재실행하면 latest부터 재개하며 최대 4에폭을 재실행할 수 있다. 별도 저장된 더 좋은 best는 보존한다. checkpoint에는 실제 model·optimizer·scaler·RNG·config·이력이 들어간다.

전체 RR/FR 장면별 JSON·평균·CSV, 실제 학습/validation 곡선, 비교 그래프, 학습 전에 고정한 RR 5장면의 LR MS/PAN/예측/GT 패널, 모든 다중밴드 예측, stage/방향/밴드별 gate 평균·표준편차·포화율·PAN 잔차/주입/MS 고주파 에너지, 코드·데이터·가중치·산출물 SHA256과 ZIP을 저장한다. ZIP CRC와 SHA256을 확인하지만 Kaggle 밖 원격 보관을 자동으로 완료하지는 않는다.

로컬에서는 셀/모델/worker의 문법과 한 셀 노트북 일치를 확인했다. 실제 DWT/IWT 함수에 NumPy tensor shim을 사용해 세 shape에서 round-trip·에너지 보존·두 단계 합성·상수 LL 계수 배율도 검증했다. 이는 PyTorch 실행 검증을 대체하지 않는다. 현재 환경에 PyTorch/CUDA와 Kaggle H5가 없어 실제 학습·메모리 적합성·성능 향상은 검증 전이다. Kaggle preflight에서 Haar round-trip, 초기 PAN 주입=0, 출력 shape, gradient, warm-up 초기화 복구를 확인하고 실행 후 A0/A1 공통 가중치와 epoch 순서를 검사한다.

## 완료 결과의 보관 검증

2026-10-09에는 다운로드한 실제 A0/A1의100에폭·RR/FR20장씩 결과를 검증해 [완료 보고서](../experiments/kaggle_wavelet_a0_a1.md)에 보관했다. 위의 CUDA 실행 검증 전 설명은 최초 셀 작성 시점의 범위이며, 실제 Kaggle preflight 통과와 완료 산출물은 보고서에 별도 기록했다. 원 실행의 B0 미검증 기록을 유지하고 기존 Kaggle B0와의 사후 검증을 `verification.json`·`posthoc_comparison.json`으로 구분한다.

저장소 루트에서 다음 명령으로 보관 파일의 해시·checkpoint 내부 metadata/tensor·전체 예측·순서를 재검증하고 그래프를 다시 생성할 수 있다. verifier에는 NumPy, 그래프에는 NumPy·Matplotlib이 필요하며 CUDA/PyTorch는 필요하지 않다. checkpoint reader는 제한된 torch.save ZIP tensor 형식만 허용하고 임의 pickle global을 실행하지 않는다. 새 inference를 했다는 뜻은 아니다.

```bash
python SSA-MRN/scripts/verify_wavelet_archive.py SSA-MRN/docs/assets/kaggle_wavelet_a0_a1 --baseline SSA-MRN/docs/assets/kaggle_band_gated_hf --cell SSA-MRN/scripts/kaggle_a0_a1_one_cell.py --output SSA-MRN/docs/assets/kaggle_wavelet_a0_a1/verification.json
python SSA-MRN/scripts/summarize_wavelet_results.py SSA-MRN/docs/assets/kaggle_wavelet_a0_a1 --baseline SSA-MRN/docs/assets/kaggle_band_gated_hf
```

실제 optimizer 업데이트는 B0/A0/A1이53,585/53,588/53,586으로 AMP skip 차이가 있다. nominal 학습 설정을 맞춘 비교이며 같은 실제 업데이트 수·동일 용량·동일 MS/LMS 입력 경로라고 표시하지 않는다. 결과를 재생성하면 `repository_manifest.json`도 새 해시로 기록해야 한다.
