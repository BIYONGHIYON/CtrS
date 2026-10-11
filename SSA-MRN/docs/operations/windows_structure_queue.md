# Windows 두 작업 병렬 · SSA-MRN 구조 선별 계획

**전체 완료: 2026-10-11 20:11:14 KST.** [9개 구조 결과·가중치](../experiments/windows_structure_qb_50.md) Windows11·Ryzen5600X·RAM15GB·RTX3060Ti8GB를 대상으로 한다. [구조별 설계](../../README.md#structure-plan)의 다음 실행 계획이며, 과거 Windows 손실 연구와 별도 run으로 보관한다.

## 목적과 시간 예산

QB K4 구조9개를 **각각50에폭** 비교한다. 최상위 후보의 추가 연장은 하지 않는다. 9개 ×50에폭 = **합계450모델에폭**이다. GPU 한 장에서 두 프로세스가 자원을 공유하므로 시간이 반드시 절반이 되지는 않는다. 실제 속도를 측정해 전체 ETA를 갱신한다.

이 예산은 기존100에폭 비교 계획을 대체하는 **빠른 선별 단계**다. 50에폭 결과로 잠재적 성능을 부정하거나 최종 채택을 확정하지 않는다. 기존100/200에폭 수치와 직접 개선량으로 비교하지 않는다. 사용자 실행 요청으로 독립 checkout에 구현했다.

## 공통 조건

| 항목 | 설계 |
|---|---|
| 데이터 | 기존 QB train17139/validation1905 전체, H5 원본·분할 유지 |
| 입력 | PAN/LMS/GT64×64, MS16×16, peak2047 |
| 모델·손실 | K4, GT MSE, 기존 확대·축소법 유지 |
| 학습 | seed42, Adam lr1e-4, effective batch256, 일정한 학습률 |
| 정밀도 | train AMP FP16/GradScaler, validation/test FP32, TF32 해제 |
| 실제 micro batch | 256 고정. 32/64/128/256 실데이터 병렬 측정 후 사용자 승인으로 확대 |
| 누적 | effective256 / micro 횟수. 마지막 불완전 배치의 실제 샘플 수로 정규화 |
| 공통 초기화·순서 | 공통 계층 초기값·에폭별 샘플 순서 일치, 신규 계층 seed/초기화 기록 |
| checkpoint 선택 | 동일 예산 안에서 best validation MSE, latest는 재개용 |

micro batch는 BatchNorm 동작을 바꾸므로 run별 자동 확대/축소하지 않는다. OOM으로 공통 micro를 바꿔야 하면 영향을 받는 비교를 새 run ID로 다시 시작한다. 실행 도중 패치 크기·정밀도·optimizer를 바꾸지 않는다.

## 실행 대기열

동시에 활성 학습은 최대2개다. 슬롯이 비면 **다른 학습의 종료를 기다리지 않고** 의존성이 충족된 다음 항목을 즉시 시작한다. 아래 순서는 대기열 우선순위이며 실제 시작 순서는 완료 시점에 따라 바뀐다.

| 대기 순서 | run | 변경 | 총 학습 에폭 |
|---|---|---|---|
| 1 | S0 | 원본 K4 기준선 | 50 |
| 2 | S1 | 전체 SSAI 최종 기여0 | 50 |
| 3 | S2 | 중간 SSAI 최종 기여0 | 50 |
| 4 | S3 | 작은 SSAI 최종 기여0, MS 입력 유지 | 50 |
| 5 | R1 | 마지막 단계만 LMS+잔차 | 50 |
| 6 | R2 | 작은/중간/전체 단계 잔차 복원 | 50 |
| 7 | M1 | 작은 SSAI를 MS 전용 밴드 혼합으로 교체 | 50 |
| 8 | I1 | 초기 LRPAN 제거, 이후 PAN 경로 유지 | 50 |
| 9 | I2 | 초기 PAN/LRPAN·LMS 별도 특징 추출 후 융합 | 50 |

M1은 S3 완료 후 시작한다. 나머지 후보의 정확한 텐서 연결은 구조 설계에 따른다. 모든 후보가50에폭을 완료해야 TOP을 결정한다. 조건 결합은 이번 자동 큐에서 제외하고 별도 예산·설정으로 설계한다.

예상 흐름: **S0+S1 → 빈 슬롯에 S2/S3 → R1/R2 → M1/I1 → I2 → 선별 → 최종 평가**. 평가/보관 때문에 슬롯이 오래 막히지 않도록 큰 ZIP 생성과 전체 test는 학습 선별 종료 후 수행한다.

## 메모리·처리 속도 최적화

1. **작은 예비 실행:** 가장 무거운 후보를 포함해 forward/backward가 안정적인지 검증한다. micro32/64/128/256의 두 작업 병렬 처리량과 합산 peak VRAM을 측정한다. warmup 후 단독/병렬을 같은 샘플 수로 비교한다. 측정용 가중치·BN 상태는 본학습에 재사용하지 않는다.
2. **VRAM:** 합산 전체 GPU 메모리 약6.5GiB 이하를 초기 상한으로 둬 Windows 화면과 CUDA workspace 여유를 남긴다. 고정 사용률 목표 대신 OOM 없는 최대 합산 패치/초를 선택한다. 실제 여유·변동폭에 따라 상한을 조정하고 기록한다.
3. **데이터:** 정규화 FP32 캐시를 디스크 mmap으로 한 번 생성하고 두 프로세스가 같은 파일을 읽는다. GPU별/프로세스별 전체 데이터 복제는 피한다. RAM15GB에서는 OS·두 Python·파일 캐시 여유를 우선 확보한다. 캐시는 데이터 해시로 검증한다.
4. **CPU:** 프로세스당 DataLoader workers2로 시작한다(총4). Windows spawn에 맞게 main guard와 최상위 Dataset을 사용한다. OMP/MKL 및 torch CPU threads는 각각1로 시작한다. worker4보다 worker1이 빠르면 전체 run을 공통 설정으로 줄인다.
5. **전송:** pin_memory·non_blocking 전송, persistent_workers, 낮은 prefetch부터 측정한다. H5 handle은 worker별로 열거나 mmap 캐시만 읽는다. 배치마다 전체 H5 재탐색·NumPy 중복 복사를 피한다.
6. **연산:** channels_last는 실제 SSA reshape를 포함한 정확도·속도 검증 후 두 모델 모두 동일하게 적용한다. cuDNN benchmark를 사용하면 비트 단위 재현 불가를 기록한다. torch.compile은 초기 컴파일 비용과 shape 호환성 때문에 최초 큐의 기본 최적화에서 제외한다.
7. **로그:** batch별 loss.item()/전체 GPU 동기화를 줄이고2~5초 간격으로 갱신한다. 슬롯·run·epoch·진행률·patch/s·합산VRAM·validation·ETA를 표시한다. 체크포인트 시점에 필요한 값을 정확하게 저장한다.

두 프로세스가 GPU 한 장을 공유하므로 메모리 여유만으로 병렬 이득을 판단하지 않는다. 병렬 합산 처리량이 단독보다 낮으면 그 측정 결과를 먼저 보고한다. 각 run의 처리 속도·CPU/RAM·VRAM을 함께 남긴다. RAM 압박·paging이 나타나면 worker/prefetch를 줄인다.

## 동일 예산 선별과 평가

- 모든 후보50에폭을 완료하고 최소 validation MSE로 정렬한다. 동률은 SAM, 그다음 사전 고정 대기열 순으로 고정한다. test는 이 선택에 사용하지 않는다.
- 기준선과 모든 후보의 학습 종료 조건은 총50에폭이다. 최상위 후보도 추가 학습하지 않는다. `resume-latest`는 중단된50에폭 실행의 복구용이며 목표 에폭을 늘리는 기능과 구분한다.
- 50에폭도 기준선과 후보가 혼합 지표를 보이면 채택 보류한다. 승자만 보고하지 않고9개 선별 곡선과 최종 두 모델을 모두 기록한다.
- 최종 두 모델만 best로 RR/FR 전체 각20장을 평가한다. PSNR/MSE/ERGAS·SAM·밴드별 오차/평균편향·고정 경계 영역 오차, FR 입력MS 일치도·잠정 QNR 및 추론 비용을 기록한다. 마스크와 고정5장면은 test 전에 정한다.
- 결과로 다음3시드·센서 확장·결합 여부를 결정한다. 이번 큐는 빠른 단일시드 선별로 명시한다.

## 연속 실행과 복구

클라이언트 SSH와 독립적인 Windows 컨트롤러가 run 상태를 디스크에 기록한다. 시작 전 기존 학습/전송 PID·GPU 사용을 확인하고 별도 checkout/runtime에서 실행한다. 정상 종료·저장 확인 후 슬롯을 반환하고 다음 run을 시작한다. 완료 run은 건너뛴다.

latest는5분 간격 또는 정한 optimizer step 경계와 매에폭 저장한다. model/Adam/GradScaler/RNG/epoch/next batch/partial epoch 집계/설정 해시를 원자적으로 저장하고 이전 latest도 보관한다. 중단된 gradient 누적 구간은 마지막 완전한 optimizer 경계부터 재개한다. controller 재시작 시 기존 PID·lock을 확인해 중복 실행을 막는다.

오류 run은 failed로 남기고 독립 큐를 계속 진행하되, S0 실패·공통 데이터 손상·디스크 부족·최종 후보 선별 불가이면 관련 단계는 중단한다. 설정을 바꾸는 OOM 자동 재실행은 하지 않는다. 전용 `windows_structure_suite.py`가 두 슬롯과 mid-epoch 복구를 제공한다.

## 결과 보관·진행표

각 run의 설정·코드/데이터 SHA, best/latest 원본·해시·compact epoch 기록·곡선을 보관한다. 평가한 run은 장면별 JSON·고정5장면·비교 그래프도 보관한다. 이전 폴더는 원격 보관·복구 검증 후 공통 규칙에 따라 정리한다.

| 단계 | 상태 | 남은 일 |
|---|---|---|
| 두 슬롯·공통 캐시·후보 구현 | 완료 | 자동 큐 실행 확인 |
| micro/병렬 처리량·복구 예비 검증 | 완료 | 실제 학습 처리량 확인 |
| 9개 ×50에폭 | 완료 | 실패0개, 총450에폭 |
| validation 후보 선별 | 완료 | I1 선정, 추가 학습 없음 |
| 최종 RR/FR·보관 | 평가·원본 해시 완료 | S0/I1 각RR20·FR20, 원격 보관·복구 검증은 전달 시 확인 |

예상 종료 시각은 예비 측정 후 남은 run별 실측 처리 속도와 평가 시간을 합산해 제시한다. 현재 문서의450모델에폭은 학습 예산이며 시간 보장이 아니다.

## 서버 실행 위치와 검증 기록

- checkout: `C:\CtrS-structure-suite`, branch: `ssa-windows-structure-suite`
- 원본 base: `128e28470ec094988633fe55fd5d242770ec7a57`; 실행 소스 SHA는 `runtime/qb_structure_50/plan.json`에 고정
- 원본 데이터: `C:\CtrS\SSA-MRN\data\dataset\QuickBird` (수정하지 않음)
- runtime: `C:\CtrS-structure-suite\runtime\qb_structure_50`
- S0 원본 출력 정확 일치, 모든 후보 forward/backward, optimizer·AMP 저장/복원 동일성 통과
- Windows worker2·AMP·FP32 validation·best/latest 저장 경로를 별도64패치 smoke로 검증
- I2 초기 stem은 3520→3744 parameters(+6.4%); 전체 모델은 426654→426878(+0.05%)
- 사전 고정 RR 예시: index1/8/11/13/19, 전체 장면, seed42 표기. test 결과로 변경하지 않음

| 공통 배치 후보 | 실데이터 두 작업 합산 patch/s | GPU 전체 최고 MiB |
|---|---:|---:|
|32|858|1390|
|64|940|1980|
|128|896|3034|
|256|965|4908|

짧은 예비 측정이며 실제 epoch 속도·종료시각을 보장하지 않는다. 배치256은32 대비 약12% 빠르지만 epoch당 optimizer update가536→67회로 줄어든다. 모든 후보에 동일하게 적용하고 기존배치32 결과와 직접 비교하지 않는다. 단독 R2 측정1387patch/s보다 두 작업 합산 속도가 낮았다. GPU 한 장의 Windows 프로세스 공유 비용이 있어 병렬이 전체 작업량의 처리시간을 줄인다고 단정할 수 없다.

### PowerShell 명령

```powershell
# 시작 또는 중단 후 latest에서 큐 재개 (실행 중이면 중복 시작 차단)
& C:\CtrS\.venv\Scripts\python.exe C:\CtrS-structure-suite\SSA-MRN\scripts\windows_structure_suite.py start --runtime C:\CtrS-structure-suite\runtime\qb_structure_50
# 현황
& C:\CtrS\.venv\Scripts\python.exe C:\CtrS-structure-suite\SSA-MRN\scripts\windows_structure_suite.py status --runtime C:\CtrS-structure-suite\runtime\qb_structure_50
# 한 줄 진행률, Ctrl+C는 조회만 종료
& C:\CtrS\.venv\Scripts\python.exe C:\CtrS-structure-suite\SSA-MRN\scripts\windows_structure_suite.py logs --runtime C:\CtrS-structure-suite\runtime\qb_structure_50
```

`start`는 기존50에폭 목표를 늘리지 않는다. 실패 run이 있으면 최종 선별은 보류하고 `failed.json`/`train.log`에서 원인을 확인한다. 완료된 과거 실행 폴더와 원본 데이터는 이번 작업에서 삭제하지 않았다.
