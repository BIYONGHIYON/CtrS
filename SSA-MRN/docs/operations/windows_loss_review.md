# Windows 완료 실험 · 평가와 원본 보관

[완료 보고서](../experiments/windows_controlled.md) · [학습 관리](controlled_suite.md)

이번 평가는 완료된 Windows 실행 폴더를 읽어 **QB K4 기준선과 validation 선별 최종 후보**의 best를 비교합니다. 새로운 학습은 시작하지 않습니다. 9개 손실 후보와 최종 후보의 원본 best/latest를 복사하고 SHA-256·설정·에폭·Adam/RNG 상태를 확인합니다.

## 1. SSH와 독립적으로 실행

평가 코드는 별도 실행 안내 폴더에 준비하고 기존 학습 checkout을 pull하거나 덮어쓰지 않습니다. `scripts/evaluate_windows_losses.py`와 `scripts/run_windows_loss_review.ps1` 두 파일을 같은 새 폴더에 둡니다.

```powershell
powershell -ExecutionPolicy Bypass -File C:\CtrS-loss-review\run_windows_loss_review.ps1 -Action start -Output C:\Users\trainer\windows_losses_new
```

관리 스크립트는 WMI 프로세스를 생성하므로 SSH·로그 창을 닫아도 평가가 이어집니다. 기존 산출물 폴더가 있거나 학습 프로세스가 있으면 시작을 거부합니다. 상태·로그 확인은 동일한 `-Output`으로 실행합니다.

```powershell
powershell -ExecutionPolicy Bypass -File C:\CtrS-loss-review\run_windows_loss_review.ps1 -Action status -Output C:\Users\trainer\windows_losses_new
powershell -ExecutionPolicy Bypass -File C:\CtrS-loss-review\run_windows_loss_review.ps1 -Action logs -Output C:\Users\trainer\windows_losses_new
```

이번 실제 실행은 `2026-10-10`에 WMI로 생성한 PID 14020의 자식 프로세스에서 `C:\Users\trainer\evaluate_windows_losses.py`를 사용했습니다. 산출물은 `C:\Users\trainer\windows_losses_20261010`, 로그는 같은 이름의 `.log`, 전송 파일은 같은 이름의 `.zip`입니다. 평가 시작 전 학습 Python 프로세스가 없고 기존 컨트롤러가 `finished`임을 확인했습니다.

## 2. 실행 범위와 체크포인트

- 학습 원본: `C:\CtrS-budget-suite\SSA-MRN\experiments\controlled_suite_v2`
- 선택 원본: `screening_selection.json`; 후보와 계수는 test 관측 전에 고정
- 평가 가중치: QB K4 기준선 best와 `05_final_candidate\best.pt`; 둘 다 best epoch 98
- 평가 데이터: `C:\CtrS\SSA-MRN\data\dataset\QuickBird` 아래 RR·FR H5 전체 각20장
- 시각화: seed42로 RR index1·8·11·13·19를 추론·지표 계산 전에 고정

`train_controlled.py` 원본은 메타데이터가 `config` 안에 있습니다. 기존 `evaluate_paper.py`의 최상위 `sensor/channels` 형식으로 파일 이름만 바꾸어 실행하지 않습니다. 이번 평가기는 원본을 변환하지 않고 `config.k`와 H5 채널 수로 모델을 구성하고 strict 로딩합니다.

## 3. 보관·복구 검증

```bash
python3 SSA-MRN/scripts/verify_windows_k_baselines.py
python3 SSA-MRN/scripts/verify_windows_losses.py
```

두 번째 검증은 원본 20개 체크포인트 해시, 9개 선별 이력, 30→100에폭 연속 이력, 기존 보관 QB 기준선 해시, RR/FR 전체 장면과 평균·차이·이미지 목록을 검사합니다. CUDA 학습이나 데이터 접근 없이 실행합니다.

원격 브랜치 푸시 뒤 새 sparse clone에서 같은 검증을 다시 수행합니다. 원본 데이터·공용 환경·기존 실행 폴더는 유지합니다. 이번 작업에서는 폴더 삭제나 새 학습을 진행하지 않습니다.
