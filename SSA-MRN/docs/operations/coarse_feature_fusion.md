# Windows · QB 작은 특징 보존 대기열

[실험 설계·결과](../experiments/coarse_feature_fusion.md)

- checkout: `C:\CtrS-coarse-anchor-suite`, branch `ssa-coarse-anchor-suite`
- 최신 main base: `423c8431e1954e13eb928680014b051a68f70ed8`
- runtime: `C:\CtrS-coarse-anchor-suite\runtime\qb_coarse_anchor_50`
- Python: 기존 공용 환경 `C:\CtrS\.venv\Scripts\python.exe`
- 데이터: `C:\CtrS\SSA-MRN\data\dataset\QuickBird`, 원본 수정 없음
- 조건: QB K4·seed42·각50에폭·배치64·Adam1e-4·MSE·train AMP·validation/test FP32
- 순서: N0/N1/N2/N3/N4, 동시 최대2개. 슬롯이 비면 다음 실험 시작
- N5: N3 validation MSE<N2, N4<N0가 모두 만족될 때만50에폭 실행. test로 결정하지 않음
- 완료 후 validation 후보 선택→N0/후보 전체 RR20·FR20 평가→곡선·게이트·고정5장면 저장

## 최적화·검증

배치 인덱스를 한꺼번에 mmap에서 읽고 DataLoader worker를 에폭마다 새로 띄우지 않는다. 각 학습 프로세스 CPU thread1, worker1, pin_memory/non_blocking, prefetch1. 합산 실데이터 예비 측정은 worker0 약906, worker1 약912, worker2 약900패치/초였다. worker1을 공통 선택했다. 합산 최고 GPU 메모리는 worker1 약1856MiB다. 짧은 측정이므로 시간 보장은 아니다.

N3=N2, N4=N0, N5=N2의 초기 출력 정확 일치, 게이트 역전파, AMP/Adam 복구(rtol1e-6/atol1e-8), Windows persistent worker의 재개 샘플 순서, 조건부 N5 실행/생략 분기를 검사했다. 독립128train/64validation패치·1에폭 smoke 두 개에서 학습·FP32 검증·best/latest 저장·완료 run 재개를 확인했다. smoke 가중치는 본 학습에 사용하지 않는다.

## PowerShell 명령

```powershell
# 상태
& C:\CtrS\.venv\Scripts\python.exe C:\CtrS-coarse-anchor-suite\SSA-MRN\scripts\coarse_feature_fusion_suite.py status --runtime C:\CtrS-coarse-anchor-suite\runtime\qb_coarse_anchor_50

# 한 줄 진행률. Ctrl+C는 조회만 종료
& C:\CtrS\.venv\Scripts\python.exe C:\CtrS-coarse-anchor-suite\SSA-MRN\scripts\coarse_feature_fusion_suite.py logs --runtime C:\CtrS-coarse-anchor-suite\runtime\qb_coarse_anchor_50

# 중단 후 latest에서 남은 대기열 재개. 살아 있는 컨트롤러·학습의 중복 실행 차단
& C:\CtrS\.venv\Scripts\python.exe C:\CtrS-coarse-anchor-suite\SSA-MRN\scripts\coarse_feature_fusion_suite.py resume-latest --runtime C:\CtrS-coarse-anchor-suite\runtime\qb_coarse_anchor_50

# 체크포인트 epoch/지표/해시
& C:\CtrS\.venv\Scripts\python.exe C:\CtrS-coarse-anchor-suite\SSA-MRN\scripts\coarse_feature_fusion_suite.py inspect --runtime C:\CtrS-coarse-anchor-suite\runtime\qb_coarse_anchor_50
```

WMI 컨트롤러로 실행되어 SSH 종료와 독립적이다. latest는 매에폭·5분 간격, best는 validation 선별용이다. latest가 없다면 previous를 복구한다. 목표는 총50에폭이며 resume으로 늘어나지 않는다. 실패 run은 failed로 기록하고 최종 평가를 보류한다.

## 이전 실험 정리

이전 `C:\CtrS-structure-suite\runtime\qb_structure_50`의68개 보관 파일 해시를 원격 보관 inventory와 대조하고 독립 clone 복구 verifier를 재통과한 뒤 완료 runtime을 삭제했다(3,133,207,393bytes, 약2.92GiB). 새 캐시는 별도 경로로 옮겨 원본 H5·캐시 SHA를 재검증했다. 복구 위치는 main의 `SSA-MRN/docs/assets/windows_structure_qb_50`, 원본 증거 commit3432232다. 원본 데이터·공용 환경·이전 source checkout은 유지한다. 새 학습은 이전 runtime 경로를 참조하지 않는다.

## 실행 기록

**전체 완료: 2026-10-11 22:29:45 KST.** N0~N5 각50에폭·합계300에폭·실패0개, N3 선정 후 N0/N3 전체 RR20·FR20 평가 완료. 현재 학습·전송 프로세스가 없음을 결과 전달 시 확인했다. 아래 시작 snapshot은 과거 기록이다.

본 학습 시작: **2026-10-11T20:56:57+09:00**. 확인: **2026-10-11T20:58:39+09:00**. 컨트롤러 PID17240; N0 실제 학습 PID10804/N1 PID14628, 두 run 모두3/50에폭·오류0개다. GPU100%·1928MiB, 당시 여유RAM 약4.9GiB였다. 첫2에폭의 validation 및best/latest 저장, SSH가 종료된 뒤 독립 실행을 확인했다. 이 수치는 해당 시각의 snapshot이며 완료 결과가 아니다. 서버 runtime은 현재 실행·재개용이며 원격 영구 보관 완료를 뜻하지 않는다.

[소스·설정](../assets/windows_coarse_anchor_preflight/plan.json) · [구조/복구 검사](../assets/windows_coarse_anchor_preflight/preflight.json) · [병렬 측정](../assets/windows_coarse_anchor_preflight/benchmark.json) · [실제 시작 확인](../assets/windows_coarse_anchor_preflight/launch_verification.json) · [정리 기록](../assets/windows_coarse_anchor_preflight/cleanup.json)

## 원격 보관·복구 확인

원격 커밋 `6221914`를 독립 `C:\CtrS-coarse-recovery-20261011`에 clone하여 원본79개 해시·12개 strict load·6개 best 추론·N0/N3 실제 RR 장면0 CPU 재추론을 검증했다(rtol1e-4). [복구 기록](windows_coarse_anchor_recovery.json). 원본 데이터와 완료 runtime은 결과 전달 후 자동 삭제하지 않았다.
