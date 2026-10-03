# RGB별 34특징 · 6밴드 그룹 실험

## 구성

```text
HSI 204밴드 → 연속 6밴드 × 34그룹 → 그룹당 특징 1개
                                  ↓
              R / G / B 각각 독립 코어에서 34특징 처리
                                  ↓
              각 코어의 34특징 → 그룹별 6밴드 복원
                                  ↓
              204밴드 잔차 3개를 밴드별 결합 + 23탭 baseline
```

- 0-based HSI 밴드 0–5 → 특징 0, 6–11 → 특징 1, …, 198–203 → 특징 33.
- encoder는 `Conv2d(204,34,1,groups=34)`, 각 decoder는 `Conv2d(34,204,1,groups=34)`입니다.
- 원본 HSI 밴드를 버리거나 특정 밴드만 선택하지 않습니다. 여섯 밴드의 학습된 선형 조합이 특징 하나가 됩니다.
- R·G·B 각각 scalar guide가 34개 SSA 분기를 모두 유도합니다. K=4와 3단계 공간 처리는 유지합니다.
- 34개 잔차 특징을 204밴드로 복원한 뒤 RGB별 추정치를 각 출력 밴드별로 결합합니다.
- 파라미터: **6,157,180개**. 기존 triple12는 1,950,186개입니다.

기존 12특징 모델과 설정을 보존하고, 새 모델 종류 `rgb_triple_grouped34_23tap`과
[별도 설정](../configs/lib_rgb_hsi_triple34_k4_23tap_tiles.json)을 추가했습니다.
12특징 체크포인트는 34특징 모델에 불러오거나 이어서 학습할 수 없습니다.

## 연산과 메모리

34분기의 `conv(cat(34특징, 해당 특징))`를 공통 특징 합성곱과 분기별 단일 특징 합성곱으로 나눠 더합니다.
가중치와 분기 순서, 원래 elementwise attention·공간 softmax는 유지합니다.
이렇게 하면 고해상도에서 `34×35=1,190`채널의 반복 입력을 직접 만들지 않습니다.
출력·입력 기울기·투영 가중치 기울기를 원래 개별 분기 연산과 비교하는 테스트가 있습니다.

초기 설정은 물리 배치 1 / 누적 4 / 검증 배치 1입니다. 일반 학습의 유효 배치는 4입니다.
`--smoke`는 코드의 기존 동작에 따라 누적을 1로 설정하여 두 단계만 실행합니다.
RTX 3060 Ti / PyTorch 2.7.1+cu128 / AMP에서 실제 데이터 smoke가 통과했습니다.
두 학습 단계와 부분 검증은 약 11.08초, 최대 할당 1,425.67MiB / 최대 예약 1,822MiB였습니다.
물리 배치 1의 제한된 smoke 결과이며 전체 학습 속도·메모리 상한을 보장하지 않습니다.
기존 배치 4와 달리 BatchNorm은 배치 1의 통계를 사용하므로, 누적 4가 기존 학습과 완전히 같은 것은 아닙니다.

## 전처리와 평가

기존 triple12와 동일하게 `train_layout=quadrants`, `eval_layout=tiles`, 패치 256,
area x4 축소, 23탭 보간, 정합 manifest와 유효 마스크를 사용합니다.
제공 분할 train 393 / validation 45 / test 75장면을 그대로 사용합니다.
시험은 300타일의 오차를 75장면으로 합산한 뒤 장면 평균을 보고합니다.

34특징의 정식 시험 결과는 아직 없습니다. 새 구조가 더 선명하거나 스펙트럼 보존에 유리하다는 주장은
같은 시험 장면·타일·정합 마스크의 triple12 결과와 비교한 뒤 판단해야 합니다.
HR HSI를 이용한 정합 및 합성 축소 평가의 한계는 기존 연구와 같습니다.

## 클라이언트에서 실행

이 변경이 설치된 서버 저장소를 VS Code Remote SSH로 열고 PowerShell에서 실행합니다.
아래 상대 경로는 그 저장소의 루트 기준입니다. 기존 서버의 데이터와 Python을 사용합니다.

```powershell
& C:\CtrS\.venv\Scripts\python.exe SSA-MRN\scripts\train_lib.py `
  --config SSA-MRN\configs\lib_rgb_hsi_triple34_k4_23tap_tiles.json `
  --data-root 'C:\CtrS\SSA-MRN\data\dataset\RGB-HSI\LIB-HSI' `
  --smoke
```

smoke 결과의 유한 지표·메모리·완료 여부를 확인한 후 `--smoke`를 제거하면 새 학습입니다.
별도 출력 폴더 `SSA-MRN/experiments/checkpoints/lib_rgb_hsi_triple34_k4_23tap_tiles`를 사용합니다.
VS Code 실행 구성에는 `LIB: triple34 tiles smoke / train (new run) / evaluate best`를 추가했습니다.
해당 구성은 저장소 안의 `.venv`와 기본 데이터 경로를 가정하므로 별도 worktree에서는 위 명령으로 공용 경로를 지정하세요.

## SSH 연결과 독립적인 백그라운드 실행

서버에 별도 작업 폴더 `C:\CtrS-triple34`와 예약 작업 `CtrS-Triple34-LIB-Control`을 등록했습니다.
기존 12특징 실행기·가중치·결과를 보존했습니다. 서버의 기존 controller를 새 폴더로 복사하여
34특징 설정과 공용 Python·데이터 경로를 지정한 서버 로컬 설정입니다.
`C:\CtrS-triple34\scripts`의 controller와 `triple34-remote-config.json`은 서버 로컬 파일이며 이 PR에 포함되지 않습니다.

2026-10-03 21:44:41부터 새로운 가중치로 100 epoch 학습을 시작했습니다.
실행 ID는 `20261003-214440-953e9b4298e3`입니다. smoke 가중치나 기존 12특징 가중치를 재개하지 않았습니다.

SSH/VS Code Remote SSH로 접속한 PowerShell에서:

```powershell
$py = 'C:\CtrS\.venv\Scripts\python.exe'
$ctl = 'C:\CtrS-triple34\scripts\remote-training.py'
& $py $ctl status
& $py $ctl logs --follow
```

실행기는 예약 작업 아래에서 학습 프로세스를 생성합니다. 클라이언트 연결 종료나 Mac 절전과 독립적입니다.
실제 SSH 세션을 종료한 뒤 새 세션으로 재접속해 동일 PID(25272)와 실행 ID가 유지되고,
epoch 1 진행이 81/1,572 → 427/1,572 단계로 증가한 것을 확인했습니다.
로그 보기에서 Ctrl+C를 눌러도 학습은 중단되지 않습니다. 서버 `trainer` 계정은 로그인 상태를 유지해야 하며,
화면 잠금은 가능합니다. 서버 전원·로그아웃·재부팅 시 학습이 계속된다는 뜻은 아닙니다.
서버 AC 전원의 자동 절전 대기 시간은 0(사용 안 함)으로 확인했습니다.

학습을 명시적으로 중단한 뒤 저장 지점에서 재개하려면:

```powershell
# 학습 중단을 원할 때만 실행. 저장되지 않은 현재 epoch 진행분은 잃을 수 있습니다.
& $py $ctl stop
$run = 'C:\CtrS-triple34\SSA-MRN\experiments\checkpoints\remote-runs\20261003-214440-953e9b4298e3'
& $py $ctl resume-latest --from-dir $run
```

최소 한 epoch가 완료되어 `latest.pt`가 있어야 재개할 수 있습니다.
재개는 원본 체크포인트를 복사해 새 고유 실행 폴더에 저장하며 기존 결과를 덮어쓰지 않습니다.
학습이 이미 진행 중이면 `start`나 재개 명령을 추가로 실행하지 마세요.

현재 실행 결과 폴더:

```text
C:\CtrS-triple34\SSA-MRN\experiments\checkpoints\remote-runs\20261003-214440-953e9b4298e3
```
