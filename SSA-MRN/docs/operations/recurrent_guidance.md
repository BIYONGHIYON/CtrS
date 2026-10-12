# Windows · 반복 복원 특징 전달 실행·복구

[완료 결과](../experiments/recurrent_guidance.md)

7개 각각 50에포크, 최대 2개 병렬로 실행했습니다. D0→D1→D2→D3→D4→D5→D6 순서로 빈 슬롯이 생기면 다음 작업을 시작한다. 한 개 RTX3060Ti를 공유했습니다. 학습은 WMI controller로 분리되어 SSH 연결 종료와 독립적으로 진행한다.

**2026-10-12 08:32:09 KST 학습·평가 완료.** 이 문서는 운영·복구 명령이며 새 학습은 사용자 요청 후 별도 runtime에서 시작합니다.

Git 공개 실행기는 `scripts/recurrent_guidance_suite.py`입니다. 아래 명령은 현재 서버의 원본 파일명이며 서버 소스는 변경하지 않습니다. 원본 runtime 재개에는 원본 소스·설정을 사용합니다.

## 상태·로그·체크포인트

서버 PowerShell에서 실행한다. Ctrl+C는 로그 조회만 종료한다.

```powershell
cd C:\CtrS-recurrent-guidance
& C:\CtrS\.venv\Scripts\python.exe SSA-MRN\scripts\windows_recurrent_guidance_suite.py status --runtime runtime\qb_recurrent_guidance_50
& C:\CtrS\.venv\Scripts\python.exe SSA-MRN\scripts\windows_recurrent_guidance_suite.py logs --follow --runtime runtime\qb_recurrent_guidance_50
& C:\CtrS\.venv\Scripts\python.exe SSA-MRN\scripts\windows_recurrent_guidance_suite.py inspect --runtime runtime\qb_recurrent_guidance_50
```

## 중단된 실행 재개

활성 controller·학습 PID가 없고 미완료 작업이 있을 때 사용한다. 완료한 작업은 다시 학습하지 않는다. 최종 평가만 실패했다면 로그를 확인한 뒤 `evaluate`로 재평가한다.

```powershell
& C:\CtrS\.venv\Scripts\python.exe SSA-MRN\scripts\windows_recurrent_guidance_suite.py resume-latest --runtime runtime\qb_recurrent_guidance_50
```

latest가 손상되면 previous로 복구한다. 마지막 저장 이후 최대 5분의 진행을 다시 계산할 수 있다. 학습 도중 코드·배치·설정을 바꾸면 source/config fingerprint 검사가 재개를 거부한다.

## 새 실행 준비

새 runtime에서만 사용한다. 전체 데이터 해시·mmap cache·모델 검증·실제 병렬 VRAM·AMP 1에포크 smoke를 확인한 후 WMI controller를 시작한다. 기존 cache·smoke 폴더가 있는 준비 작업을 무작정 재실행하지 않는다.

```powershell
& C:\CtrS\.venv\Scripts\python.exe SSA-MRN\scripts\prepare_recurrent_guidance.py --runtime runtime\qb_recurrent_guidance_new --data C:\CtrS\SSA-MRN\data\dataset --base-commit ce74e0380a99ba449f970c420eb19ae10008a032 --launch
```

완료 후 validation으로 후보를 고정하여 D0/D1/신규 최상위 후보의 RR20·FR20과 고정 5장면을 평가한다. server 원본 best/latest·해시·설정·결과·snapshot의 원격 보관과 복구를 확인한 뒤 [공통 정리 규칙](../../../AGENTS.md)을 적용한다.
