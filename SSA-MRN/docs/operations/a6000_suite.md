# 학교 A6000: SSA-MRN 구조 개선 4개 비교

> **과거 K6 실험 재현 안내입니다.** 현재 신규 연구 기본값은 [K4](../../README.md#decision)입니다. 아래 고정 계획은 당시 실험 재현용이며 신규 K4 계획으로 혼용하지 않습니다.

## 목적과 상태

Windows에서는 K 비교와 보조 손실을 탐색하고, 학교 서버에서는 SSA-MRN의 **구조 변경을 하나씩** 비교합니다.

2026-10-08: 네 모델의 100에폭 학습과 QB RR20/FR20 전체 평가를 완료했습니다. [결과 보고서](../experiments/a6000_architecture_qb.md)에 수치·5장면 이미지·가중치를 보존합니다. 단일 시드 탐색이므로 반복 검증은 후속 작업입니다.

## 네 실험

| 실험 | 기준선에서 바꾼 것 | 고정 사항 |
|---|---|---|
| `baseline` | 없음: 복구한 공식 SSA-MRN | 내부 bilinear, 제공 LMS, MSE |
| `interp23` | 사용되는 내부 확대 4곳만 23탭으로 교체 | 축소 bilinear·제공 LMS 유지 |
| `lr_correction` | 출력의 LR 관측 오차를 공간 보정으로 반영 | 기존 내부 bilinear·MSE 유지 |
| `high_frequency` | PAN 고주파에서 밴드별 잔차를 만드는 경로 추가 | 기존 내부 bilinear·MSE 유지 |

세 변형을 결합하지 않습니다. 모델 공통 계층은 같은 초기 가중치로 시작합니다. 변형별 조건과 checkpoint의 config를 보존합니다.

### 23탭

`upsample1`(×4), `upsample100`·`upsample101`·`upsample102`(각 ×2)를 교체합니다. 기존 toolbox 계열 23탭 계수·주기 경계·단계별 삽입 위상을 사용합니다. 제공 LMS나 네트워크 축소 연산, SSA 블록 내부 연산은 바꾸지 않습니다. SciPy의 separable wrap convolution 기준과 ×2·×4 수치 일치를 검증합니다. 경계와 위상이 baseline bilinear와 다른 것은 이 실험의 변경점입니다.

### LR 보정

QB의 41×41 MTF 관측 연산자를 사용합니다. 관측 계수와 커널은 기존 MATLAB 프로토콜 검증 코드에 근거합니다. 패치마다 입력 LMS를 관측한 값과 입력 MS의 내부 MSE가 가장 작은 16개 위상 중 하나를 고릅니다. **정답 GT와 예측 결과를 위상 선택에 사용하지 않으며 train·validation·test 모두 같은 입력 기반 규칙을 사용합니다.**

`출력 = 예측 + 0.1 × bilinear 확대(입력 MS − 관측(예측))`입니다. LR 경계 5픽셀의 보정 오차를 0으로 마스킹하여 알 수 없는 패치 halo의 직접 영향을 제외합니다. 보정 경로는 미분 가능하고 전체 출력에 기본 GT MSE를 적용합니다. Windows의 보조 관측 손실과 달리 출력 자체를 바꿉니다.

이 입력 기반 위상 선택 규칙은 새로운 연구 가정입니다. 기존 TRAIN의 GT 기반 사전 위상 감사와 같다고 주장하지 않습니다. 보정 gain 0.1은 사전 고정한 탐색 값이며, 정확한 역투영이나 모든 지표 개선을 보장하지 않습니다. 필요하면 후속 위상 진단과 gain 비교를 별도 실험으로 수행합니다.

### 고주파 경로

PAN에서 5×5 separable binomial 저주파를 빼고 `Conv(1→16) → ReLU → Conv(16→4)`로 잔차를 만듭니다. 마지막 Conv는 0으로 초기화하여 초기 출력이 기준선과 정확히 같습니다. 학습하며 잔차 경로를 활성화합니다. GT·별도 손실·23탭·LR 보정은 추가하지 않습니다.

## 공통 조건

QB 전체 train/validation, K=6, 시드 42, 100에폭, Adam·lr 1e-4, 기본 MSE, effective batch 32·micro batch 32, CUDA FP32·결정론 설정입니다. 최대 4개 병렬이며 GPU는 A6000 한 장입니다. 속도는 실제 측정으로 판단합니다. CPU 스레드는 자식별 2개로 제한합니다.

Windows의 micro batch 4·GPU·PyTorch 조건과 달라 서버 간 수치를 동일 조건의 시드 평균으로 합치지 않습니다. 개선 효과는 학교 서버의 같은 기준선과 비교합니다. 학습 도중 코드·계획·데이터 변경은 큐를 중단시킵니다.

## 경로와 데이터

| 항목 | 위치 |
|---|---|
| SSH | `dongguk-gpu04` (`cs.dongguk.edu:102`, `gpu_04`) |
| 최신 코드 | `/home/gpu_04/CtrS-a6000` |
| 브랜치 | `a6000-pan-experiments` |
| 기존 환경 | `/home/gpu_04/CtrS_old/SSA-MRN/.conda-env/bin/python` |
| 데이터 | `/home/gpu_04/CtrS-a6000/SSA-MRN/data/dataset` |
| 새 결과 | `SSA-MRN/experiments/a6000_architecture_qb` |

QB의 `Training Dataset/train_qb.h5`·`valid_qb.h5`를 사용합니다. 업로드 시 Windows의 폴더 구조를 유지합니다. GF2·WV3·WV2 업로드 데이터는 보존하며 이번 학습에는 사용하지 않습니다. QB train 5,194,901,672바이트·valid 577,415,336바이트와 H5 입력 shape·첫/마지막 패치 읽기를 확인합니다. 파일 크기 확인은 전체 SHA-256 검증의 대체가 아니므로 전송 후 원본 해시를 대조합니다.

계정은 50GiB 한도입니다. `quota -s`로 확인하며 환경·기존 결과는 삭제하지 않습니다. 비밀번호는 설정·문서에 저장하지 않습니다.

## 실행

```bash
cd /home/gpu_04/CtrS-a6000
./SSA-MRN/scripts/run_a6000_suite.sh check
/home/gpu_04/CtrS_old/SSA-MRN/.conda-env/bin/python SSA-MRN/scripts/verify_a6000_variants.py --cuda
./SSA-MRN/scripts/run_a6000_suite.sh status
```

본 학습 시작:

```bash
./SSA-MRN/scripts/run_a6000_suite.sh start
./SSA-MRN/scripts/run_a6000_suite.sh logs
watch -n 2 './SSA-MRN/scripts/run_a6000_suite.sh status'
```

중단 후 `resume-latest`를 사용합니다. 마지막 완료 epoch의 모델·Adam·RNG를 복원합니다. SSH 종료와 상태 화면 Ctrl+C는 학습을 종료하지 않습니다. 재부팅 후 자동 재개는 없습니다. 중복 큐는 잠금으로 차단하고 다른 GPU 작업이 있으면 새 시작을 거부합니다. 실패 시 이 큐의 자식만 중단합니다.

## 판단과 결과 보관

best validation MSE로 후보를 판단하고 test를 선택에 사용하지 않습니다. 효과가 있는 후보만 반복 시드·센서 확장·결합으로 진행합니다. 독립 RR/FR 전체 평가, PSNR·SAM·ERGAS 등 지표, 가중치 해시·곡선·5장면 이미지는 후속 결과 보고에서 정리합니다. 코드 사전 검증과 학습 완료는 연구 성능 개선의 증거가 아닙니다.

## 평가 재현

```bash
cd /home/gpu_04/CtrS-a6000
/home/gpu_04/CtrS_old/SSA-MRN/.conda-env/bin/python SSA-MRN/scripts/evaluate_a6000.py
```

best 가중치의 RR/FR 전체를 평가하고 `docs/assets/a6000_architecture_qb`에 지표·이미지·곡선·추론 가중치·해시 manifest를 생성합니다. 원본 재개 가중치는 삭제하지 않습니다.
