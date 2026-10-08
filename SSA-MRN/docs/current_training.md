# 현재 학습 현황 · SSA-MRN 성능 개선

**Linux 최신 확인: 2026-10-08 18:07 KST.** 후속10개 모두100에폭 완료, 큐 `finished`, active 없음, GPU 학습 프로세스 없음입니다. 이후 best12개 모델의 전체 RR/FR 평가·곡선·예시·가중치 해시를 정리했습니다. [후속 결과](experiments/improvements/a6000_followup_123.md)

**Windows 기록은 2026-10-08 11:39 KST의 과거 확인값**이며 이번 Linux 작업에서 현재 상태를 확인하지 않았습니다. 아래 Windows 초기 상세 진행률은 03:26 KST의 과거 기록입니다.

## 두 서버의 역할

| 항목 | Windows · RTX 3060 Ti | 학교 · RTX A6000 48GB |
|---|---|---|
| 목적 | 동일 환경 K 비교와 보조 손실 후보 탐색 | QB 반복시드·확대 위치·GF2/WV3 후속 검증 완료 |
| 데이터 | QB·GF2·WV3 전체 학습/검증 | QB·GF2·WV3 전체 학습/검증 |
| 실행 | 1개씩 직렬 | 4개 병렬 |
| 시드 | 42 | QB 42·43·44, GF2/WV3 42 |
| 공통 최적화 | Adam·lr 1e-4·effective batch 32·CUDA FP32·결정론 | 동일 |
| micro batch | 4, gradient 누적 | 32, 한 번에 계산 |
| 결과 폴더 | `C:\CtrS-budget-suite\SSA-MRN\experiments\controlled_suite_v2` | `/home/gpu_04/CtrS-a6000-followup/SSA-MRN/experiments/a6000_followup_123` |

GPU·PyTorch·gradient 합산 순서가 달라 두 서버의 값을 동일 조건의 반복 시드 평균으로 묶지 않습니다. 개선 효과는 각 서버의 자체 기준선과 비교합니다.

## 1. Windows: K 비교 → 손실 탐색

| 순서 | 조건 | 학습량 |
|---|---|---|
| 기준선 | QB·GF2·WV3 × K=4·6 × 시드 42 | 6회 × 100에폭 |
| 손실 후보 선별 | 선정 K의 QB에서 스펙트럼·MTF 관측·경계 손실 각각 계수 0.001·0.01·0.1 | 9회 × 30에폭 |
| 최종 후보 | 30에폭 best validation MSE가 가장 낮은 후보를 latest에서 이어 학습 | 추가 70에폭, 총 100에폭 |

총 **940에폭**입니다. 각 보조 손실은 기본 MSE에 하나씩 추가하며 서로 결합하지 않습니다. K는 센서별 검증 MSE 승리 수로 고르고 동률은 K=4입니다. test는 선택에 사용하지 않습니다.

**확인한 진행:** 첫 기준선 `01_QB_k4_s42`, 22/100에폭의 451/536배치(84.1%). 나머지 기준선과 손실 후보는 아직 완료되지 않았습니다. 이전 첫 학습을 중단하지 않고 축소 큐로 인계해 원래 실행 폴더에서 계속 학습 중입니다.

초기 203초/에폭으로 계산한 전체 약 53시간은 잠정 추정입니다. 센서·K·손실별 속도와 후속 평가 시간은 포함 여부가 다르므로 3일 완료를 보장하지 않습니다.

## 2. 학교: 후속 1~3단계 완료

| 단계 | 학습 | 완료 상태 |
|---|---|---|
| 1 | QB 기준선/전체23탭 × 시드43·44 | 4개 ×100에폭 완료 |
| 2 | QB 입력만/출력만23탭 × 시드42 | 2개 ×100에폭 완료, 기존 기준선/전체23탭 시드42 재사용 |
| 3 | GF2·WV3 기준선/전체23탭 × 시드42 | 4개 ×100에폭 완료 |

코드 `/home/gpu_04/CtrS-a6000-followup`, 결과 `SSA-MRN/experiments/a6000_followup_123`, 브랜치 `a6000-followup-suite`입니다. 최대4개 병렬, 이전 단계 전체 완료 뒤 다음 단계를 실행했습니다. 4단계 Windows 손실 결합은 제외했습니다. 공통조건 K6·Adam lr1e-4·batch=micro batch32·MSE·CUDA FP32·결정론입니다. Python·데이터·기존 결과를 참조하는 `CtrS_old`와 `CtrS-a6000`은 유지합니다.

현재 새 학습은 없습니다. 결과 판단은 [보고서](experiments/improvements/a6000_followup_123.md)를 기준으로 하며 모든 센서의 개선을 주장하지 않습니다.

## 3. 코드 사전 검증과 제한

- 23탭 ×2·×4가 SciPy separable wrap convolution 기준과 수치 일치
- 공통 계층 초기화, 출력 크기, 유한 gradient, 결정론 CUDA 역전파 확인
- 고주파 경로 초기 출력이 기준선과 정확히 같음
- LR 관측 오차가 0일 때 보정량이 0임
- 실제 QB 32개 train/validation 패치로 4개 동시 학습·검증·체크포인트 저장 통과: 최고 GPU 메모리 3,580MiB
- 학습 H5는 예상 바이트·shape·첫/마지막 패치와 inventory 메타데이터 확인. 전체 학습 H5 SHA-256 대조를 완료했다고 주장하지 않음

이 검증은 구현 실행 가능성을 확인하며 연구 성능 개선의 증거가 아닙니다. 이 절의 초기 smoke 검증은 성능 근거가 아닙니다. 후속 본 학습·test 완료 근거는 이번 보고서에 있습니다.

## 4. 상태와 로그 확인

Windows PowerShell:

```powershell
& C:\CtrS-budget-suite\SSA-MRN\scripts\run_controlled_suite.ps1 -Command status
& C:\CtrS-budget-suite\SSA-MRN\scripts\run_controlled_suite.ps1 -Command logs -Follow
```

학교 서버 터미널:

```bash
cd /home/gpu_04/CtrS-a6000-followup
./SSA-MRN/scripts/run_a6000_followup.sh status
./SSA-MRN/scripts/run_a6000_followup.sh logs
```

접속 종료·로그 화면 Ctrl+C는 학습을 중단하지 않습니다. 재부팅·절전은 피하고 실행 중 코드·계획·데이터를 덮어쓰지 않습니다.

## 5. 학습 이후 할 일 · 상태 구분

학교 후속10개 학습과 재사용2개를 포함한 전체 RR/FR 평가·35개 예시·실측 곡선·원본24개 가중치/해시 정리는 완료했습니다. 다음 판단은 GF2 반복시드와 FR 평가 구현 검증이며 새 학습은 자동 시작하지 않습니다. 원격 가중치 복구 검증 전에는 서버 결과를 삭제하지 않습니다. Windows 진행과 후속 평가는 별도 관리합니다.

[Windows 실행 가이드](operations/controlled_suite.md) · [Linux 후속 실행·평가](operations/a6000_followup.md)
