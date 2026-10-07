# 1~4단계 직렬 학습 실행

[실험 계획](../improvement_plan.md) · [관측 연산자 검증](../experiments/improvements/observation_validation.md)

**코드·사전 검증 완료 / 본 학습 시작 보류.** 총 45회이며 하룻밤 이상 걸릴 수 있습니다.

## 실행 위치

| 항목 | 위치 |
|---|---|
| 서버 실행 폴더 | `C:\CtrS-controlled-suite\SSA-MRN` |
| Python | `C:\CtrS\.venv\Scripts\python.exe` |
| 원본 데이터 | `C:\CtrS\SSA-MRN\data\dataset` — 읽기 전용 |
| 설정 | `scripts/controlled_suite_plan.json` |
| 새 결과 | `experiments/controlled_suite_v2/` |

## 시작과 점검

서버 PowerShell에서 실행합니다. 지금은 `check`와 `status`까지만 사용합니다.

```powershell
cd C:\CtrS-controlled-suite\SSA-MRN\scripts
.\run_controlled_suite.ps1 -Command check
.\run_controlled_suite.ps1 -Command status
```

학습 시작이 결정되면:

```powershell
.\run_controlled_suite.ps1 -Command start
.\run_controlled_suite.ps1 -Command logs -Follow
```

| 명령 | 기능 |
|---|---|
| `check` | 데이터·CUDA·관측 프로파일 검증. 학습 없음 |
| `status` | 컨트롤러·학습 PID, 생존 여부, 현재 run, heartbeat |
| `start` | 1→2→3→4 순서로 새 큐 실행 |
| `resume-latest` | 미완료 run의 마지막 완료 epoch에서 재개 |
| `resume-best` | 미완료 run의 최저 검증 MSE epoch로 되돌아가 재개 |
| `inspect` | 기록된 best epoch와 검증 MSE 확인 |
| `logs -Follow` | 실시간 로그 보기 |

일반 재개에는 `resume-latest`를 사용합니다. 모델·Adam·CPU/CUDA RNG·데이터 순서를 복원하고, 중단된 epoch는 다시 계산합니다. 완료 run은 건너뜁니다. 오류가 나면 큐를 중단합니다.

## 접속 종료와 중복 실행

Windows WMI로 SSH 세션과 독립된 프로세스를 만듭니다. 로그 뷰어의 Ctrl+C와 클라이언트 접속 종료는 학습을 종료하지 않습니다. 서버 절전·재부팅은 학습을 중단하며 부팅 후 자동 재개는 없습니다.

기존 학습이 살아 있거나 코드·계획·관측 프로파일·원본 파일 정보가 달라지면 시작/재개를 거부합니다. 실행 중인 checkout에는 pull이나 코드 덮어쓰기를 하지 않습니다.

## 결과 보관

각 run은 `config.json`, `history.jsonl`, `train.log`, `best.pt`, `latest.pt`, `complete.json`을 저장합니다. 코드 해시와 계획도 보존합니다. `complete.json`은 **학습 완료**이며 독립 test 평가 완료를 뜻하지 않습니다.

이전에 잠시 실행했다 중지한 기록은 `experiments/controlled_suite/`에 남겨 두었습니다. 새 코드는 별도 `controlled_suite_v2/`를 사용합니다. 다음 실험 전에 [용량 정리 규칙](../../../AGENTS.md)을 적용합니다.
