# 6단계 예비 실험 환경

[현재 완료 연구](../../README.md) · [실험 목록](../../configs/pilot/suite.json)

이 환경은 RGB07 개선 후보를 짧게 선별하기 위한 것입니다. 서버 checkout은 `C:\CtrS-pilot-suite`, Python은 `C:\CtrS\.venv\Scripts\python.exe`를 사용합니다. 원본 데이터·RGB07 가중치는 기존 위치에서 읽고, 실행마다 새로운 출력 폴더를 만듭니다. 환경 구성만으로 학습을 시작하지 않습니다.

## 비교 설계

전체 393개 학습·45개 검증 장면, HR256/LR64, 204밴드, 17특징, K4, area 합성 축소, 23탭, 기존 정합 manifest와 유효 마스크, seed42를 사용합니다. 예비 후보 선택에는 test75를 사용하지 않습니다. 10에폭 시간은 RGB07의 에폭당 약 184초에서 약 31분으로 추정하며 모델·디스크 상태에 따라 다시 측정합니다.

| 단계 | 실행 이름 | 목적 | 비교 기준 |
|---|---|---|---|
| 공통 기준 | `00_baseline10` | RGB07 구조를 새로 10에폭 학습 | 구조 후보의 공통 대조군 |
| 1 | `01_rgb_diagnostic` | 고정된 RGB07 best99에서 RGB 흐림·±1px 이동·평균/0 입력 진단 | 동일 가중치의 original RGB |
| 2 | `02_warm_control10` | best99에서 LR1e-4로 추가 10에폭 | 낮은 학습률 실험의 대조군 |
| 2 | `02_warm_low10` | 같은 best99에서 LR1e-5로 추가 10에폭 | `02_warm_control10` |
| 3 | `03_aligned_attention10` | 가로·세로 전치만 제거; spatial softmax 유지 | `00_baseline10` |
| 4 | `04_gated_detail10` | 3채널 RGB 고주파 특징과 HSI 조건 gate를 통한 추가 보정 | `00_baseline10` |
| 5 | `05_global_encoder10` | LR에서 그룹17·전역17 특징을 섞어 17특징 생성 | `00_baseline10` |
| 6 | `06_local_alignment10` | LR 특징 조건 HR ±2px offset과 세부 gain 학습 | `00_baseline10` |

3~6은 각각 **독립된 한 요소 비교**입니다. 이전 단계가 개선됐다고 가정해 누적하지 않습니다. 각 요소의 이득이 확인된 뒤 조합 실험을 별도로 만듭니다. 12→17 확대와 분광 손실을 함께 바꾼 과거 실험처럼 효과를 혼동하지 않도록 하기 위한 조건입니다.

2단계는 모델 가중치만 로드하고 optimizer·epoch·best 기록을 새로 시작합니다. 두 run 모두 같은 가중치 해시에서 출발합니다. `resume`은 같은 run의 중단 복구용이며, 학습률 변경 실험을 `resume`으로 시작하지 않습니다.

1단계의 모든 RGB 변형은 동일한 보수적 마스크(4px erosion)에서 비교합니다. 평균/0 RGB는 학습 분포 밖 입력이므로 민감도 진단이며, 재학습한 HSI-only 모델의 성능으로 해석하지 않습니다. 6단계는 HSI-RefSR/SSC-HSR의 정합·융합 원리를 참고한 작은 실험 모듈이며 해당 논문의 재현이 아닙니다. 세부 gain은 0~2이며 초기값 1입니다. RGB를 더 부드럽게 하거나 세부를 강조할 수 있지만 확률로 해석하지 않습니다. offset은 HR 픽셀 단위이며 ±2px로 제한합니다.

## 원격 실행

VS Code Remote-SSH로 서버에 접속한 PowerShell에서 실행합니다. 다른 사람이 GPU를 사용 중인지 팀 내 확인 후 한 작업씩 시작합니다. 컨트롤러는 기존 SSA-MRN/ECRformer `train.py`, `train_lib.py` 및 진단 실행을 발견하면 시작을 거절하며 기존 프로세스를 중단하지 않습니다.

```powershell
cd C:\CtrS-pilot-suite
$py = 'C:\CtrS\.venv\Scripts\python.exe'
$pilot = 'C:\CtrS-pilot-suite\SSA-MRN\scripts\pilot.py'

& $py $pilot list
& $py $pilot status

# 1단계: 학습 없이 RGB 사용 정도 진단
& $py $pilot start 01_rgb_diagnostic

# 로그 확인: Ctrl+C는 로그 보기만 종료
& $py $pilot logs --follow
```

완료를 확인한 뒤 원하는 다음 작업을 한 개씩 시작합니다.

```powershell
& $py $pilot start 02_warm_control10
& $py $pilot start 02_warm_low10
& $py $pilot start 00_baseline10
& $py $pilot start 03_aligned_attention10
& $py $pilot start 04_gated_detail10
& $py $pilot start 05_global_encoder10
& $py $pilot start 06_local_alignment10
```

위 블록은 선택 가능한 명령 목록입니다. 앞선 run이 끝난 뒤 다음 줄을 실행합니다. 한 번에 여러 작업을 큐에 넣거나 자동 시작하지 않습니다.

서버의 `CtrS-Pilot-Control` 예약 작업이 실행을 소유하므로 Mac이나 SSH 연결을 종료해도 학습이 지속됩니다. Windows의 trainer 계정은 로그인 상태를 유지하고 서버 절전·재부팅·전원 종료를 피해야 합니다. 서버가 재시작되면 컨트롤러는 로그인 때 켜지지만 학습을 자동 재개하지 않습니다.

```powershell
# 중단된 해당 run의 latest 가중치로 복구. 실제 run 경로를 넣습니다.
& $py $pilot resume-latest 03_aligned_attention10 --from-dir 'C:\CtrS-pilot-suite\SSA-MRN\experiments\checkpoints\remote-runs-pilot\실제-run-ID'
```

실행마다 `remote-runs-pilot/<run-ID>`에 best/latest, 설정과 검증 결과를 저장하고, `remote-control-pilot/runs/<run-ID>/train.log`에 로그를 남깁니다. 기존 100에폭 가중치·실험 결과는 별도 경로에 보존합니다.

## 판단 및 결과 보관

검증 PSNR·SAM, 204밴드 공간 gradient RMSE, 장면별 변화와 실제 시간을 함께 봅니다. gradient RMSE는 예측과 정답의 수평·수직 차분 오차이며 양쪽 픽셀이 유효한 경우만 포함합니다. 선명도 하나로 판단하지 않습니다. best는 기존처럼 validation MSE로 고릅니다.

학습곡선은 `history.jsonl`, best의 장면별 수치는 `best_validation_metrics.json`에 저장합니다. 완료 run을 모아 다음 명령으로 검증 비교표와 그래프를 만듭니다.

```powershell
& $py SSA-MRN\scripts\summarize_pilot.py --runs '기준-run-폴더' '후보-run-폴더' --output SSA-MRN\experiments\results\pilot_comparison
```

summary는 scratch와 기존 가중치 초기화를 분리합니다. 두 그룹의 숫자를 한 순위로 해석하지 않습니다. 10에폭에서 유망한 후보는 같은 기준 모델과 함께 20~30에폭으로 재확인합니다. 최종 구조를 고른 뒤 전체 학습과 보류된 test75 평가를 수행하고, 정식 실험 문서에는 그래프와 사전 고정된 서로 다른 test 장면 5개의 4패널 이미지·204밴드 HTML을 포함합니다. 예비 실험은 기존 README의 최신 완료 연구를 대체하지 않습니다.

## 환경 재설치

아래 명령은 idle 컨트롤러만 등록·시작하며 학습을 시작하지 않습니다. 현재 작업이 실행 중일 때 소스 파일이나 컨트롤러를 덮어쓰지 않습니다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup-remote-control.ps1 -TaskName CtrS-Pilot-Control -ControllerName remote-training-pilot.py
& $py SSA-MRN\scripts\check_pilot_environment.py
```

근거: [PanNet](https://xueyangfu.github.io/projects/iccv2017.html), [MHF-Net](https://arxiv.org/abs/1901.03281), [HSI-RefSR](https://arxiv.org/abs/2302.06298), [SSC-HSR](https://arxiv.org/abs/2505.02109).

## 준비 검증

서버에서 CPU 단위 검증 9개를 통과했고, 두 warm 설정 모두 동일한 best99 해시·next epoch1·빈 optimizer 상태를 확인했습니다. 5개 모델은 실제 HR256·배치4·AMP 조건의 합성 CUDA forward/backward에서 유한 손실·gradient를 확인했습니다. 최대 할당은 약 4.1GiB, 예약은 약 4.6GiB였습니다. 이 검증에는 Adam 상태와 DataLoader prefetch가 포함되지 않으므로 실제 학습 메모리는 첫 run에서 다시 기록합니다. optimizer step은 0회입니다.

컨트롤러 heartbeat가 online인 상태를 확인했습니다. 체크포인트·정합·데이터 분할과 CPU/CUDA 검증 근거는 [환경 기록](../../references/pilot_setup_20261005.json)에 보관합니다. trainer 계정의 ScheduledTask CIM 조회는 권한 제한이 있어 작업 등록 결과와 컨트롤러 heartbeat로 동작을 확인했습니다.
