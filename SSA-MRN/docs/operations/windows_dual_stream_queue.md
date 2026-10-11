# Windows · QB C0~C5 연속 학습

**2026-10-12: 6개 학습과 자동 평가 완료.** [완료 결과·가중치](../experiments/windows_dual_stream_qb_50.md)를 먼저 확인한다. 완료 큐를 새 학습에 재사용하지 않는다.

## 실행 조건

각 50에포크, 공통 배치 96, 최대 2개 병렬이다. C0→C1→C2→C3→C4→C5 순서로 빈 슬롯이 생기면 다음 작업을 시작한다. 하나의 RTX 3060 Ti에서 두 프로세스가 함께 학습한다.

| 용도 | Windows 경로 |
|---|---|
| 별도 checkout | `C:\CtrS-dual-stream-suite` |
| 원본 데이터 | `C:\CtrS\SSA-MRN\data\dataset` |
| 공용 Python | `C:\CtrS\.venv\Scripts\python.exe` |
| 실제 실행 기록 | `C:\CtrS-dual-stream-suite\runtime\qb_dual_stream_50` |

## 조회 명령

서버 PowerShell에서 실행한다. 로그 조회의 Ctrl+C는 학습을 종료하지 않는다. 완료 뒤에는 status와 inspect를 사용한다.

```powershell
cd C:\CtrS-dual-stream-suite
# 상태·PID
& C:\CtrS\.venv\Scripts\python.exe SSA-MRN\scripts\windows_dual_stream_suite.py status --runtime runtime\qb_dual_stream_50
# 학습 중 한 줄 진행률
& C:\CtrS\.venv\Scripts\python.exe SSA-MRN\scripts\windows_dual_stream_suite.py logs --follow --runtime runtime\qb_dual_stream_50
# 체크포인트·epoch·SHA-256
& C:\CtrS\.venv\Scripts\python.exe SSA-MRN\scripts\windows_dual_stream_suite.py inspect --runtime runtime\qb_dual_stream_50
```

## 중단된 실행만 재개

```powershell
# 활성 controller가 없고 학습이 미완료인 경우에만 실행
& C:\CtrS\.venv\Scripts\python.exe SSA-MRN\scripts\windows_dual_stream_suite.py resume-latest --runtime runtime\qb_dual_stream_50
```

controller는 WMI로 분리 실행해 SSH 연결 종료와 독립적으로 작동한다. 중복 controller/run은 거부하며 학습 도중 코드·배치·source hash를 변경하면 재개가 거부된다. 강제 중단 시 마지막 저장 이후 최대 5분의 진행을 다시 계산할 수 있다. CUDA 비트 단위 재현은 보장하지 않는다.

종료 후 validation 순위를 고정하고 C0/C1/최상위 신규 후보만 FP32 전체 RR/FR를 평가한다. 이번 신규 후보는 C5, 전체 validation 1위는 C1이다. 추가 실험은 별도 실행 폴더와 승인된 계획을 사용한다.

## 보관·복구

원본 best/latest 12개, 설정·소스 snapshot·해시·compact history·전체 평가 JSON·그래프·고정 예시는 보고서 7절에 있다.

```powershell
& C:\CtrS\.venv\Scripts\python.exe SSA-MRN\scripts\verify_dual_stream_results.py --check-torch
```

원격 보관과 복구를 확인한 뒤에만 [공통 정리 규칙](../../../AGENTS.md)에 따라 사용하지 않는 실행 폴더를 정리한다. 이번 결과 전달에서는 서버 실행 폴더·원본 데이터·다른 연구의 checkout을 삭제하지 않았다.
