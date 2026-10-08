# Linux GF2 반복·QB 입력 23탭 실행과 복구

상태: **2026-10-08 23:17:31 KST 학습·전체 평가·검증 완료**. 19:05:47 시작, 약 4시간 12분. 6개 새 학습 모두 100에폭이고 시간 예산으로 미룬 run은 없습니다.

[결과 보고서](../experiments/a6000_gf2_repeat_qb_input.md) · [사전 설계 근거](a6000_next_6h_evidence.json) · [실행 계획](../../scripts/a6000_next_6h_plan.json)

## 실행 구성

1. GF2 기준선/전체 23탭 × 시드 43·44, 최대 4개 동시.
2. 이전 4개 모두 끝나면 QB 입력 23탭 × 시드 43·44, 2개 동시.
3. 기존 9개 모델 포함 15개 모델의 RR 20장·FR 20장 전체 평가, 가중치·곡선·고정 이미지·해시 검증.

K=6, Adam lr=1e-4, batch=micro batch=32, 100에폭, MSE, CUDA FP32·결정론입니다. best는 validation MSE로 선택합니다. 학습 코드 기준 commit은 be02b7cbf03fe00681194ae9fc9419661f24cd56이며 새 관리 코드의 파일별 해시는 결과 manifest에 있습니다.

## 서버 경로

| 용도 | 위치 |
|---|---|
| checkout / branch | `/home/gpu_04/CtrS-a6000-next6h` / `a6000-next6h` |
| Python | `/home/gpu_04/CtrS_old/SSA-MRN/.conda-env/bin/python` |
| 데이터 | `/home/gpu_04/CtrS-a6000/SSA-MRN/data/dataset` |
| 실행 결과 | `SSA-MRN/experiments/a6000_gf2_repeat_qb_input` |
| 기존 대조군 9개 | `SSA-MRN/references/next6h_runs` |
| 기존 WV3 진단 best | `SSA-MRN/references/next6h_diagnostics` |
| 전달 산출물 | `SSA-MRN/docs/assets/a6000_gf2_repeat_qb_input` |

## 상태 확인

```bash
cd /home/gpu_04/CtrS-a6000-next6h
bash SSA-MRN/scripts/run_a6000_next6h.sh status
bash SSA-MRN/scripts/run_a6000_next6h.sh logs
watch -n 10 'bash SSA-MRN/scripts/run_a6000_next6h.sh status'
```

`timing_projection`은 1단계에서만 존재하는 전망값입니다. 2단계·평가·종료 상태에서 직접 인덱싱하면 KeyError가 나므로 위 status 명령을 사용합니다. 상태가 finished이고 evaluation_verified가 true인 경우 전체 평가 검증까지 완료입니다. 종료 후 controller_alive=false는 정상입니다.

## 시간 예산·재개

1단계 10에폭 후 최근 5개 에폭 중앙값 전망을 표시합니다. 실제 1단계 완료 경과 시간 + QB 학습 108분 + 평가·정리 30분이 360분 이내일 때만 2단계 두 run을 함께 시작합니다. 이번에는 두 run 모두 수행했습니다. 시간 판단으로만 제외하며 성능으로 시드를 제외하지 않습니다.

controller는 SSH와 독립적으로 실행되며 코드·데이터 불변성을 검사합니다. 실행 중 학습 코드나 설정을 수정하지 않습니다. 부팅 후 자동 재개는 없습니다. 실패 시 상태·PID·GPU와 checkpoint를 확인하고 관리 스크립트의 resume-latest로 Adam/RNG까지 복구합니다. 터미널에서 train.py를 직접 실행하지 않습니다. 이번 완료 실험을 다시 시작하거나 resume할 필요는 없습니다.

## 사전 검증·용량 정리

[4개 실제 CUDA 동시 검증](a6000_next6h_hardware_verification.json), [실행 기록](a6000_next6h_runtime.json), [정리 경로·복구 위치·용량 기록](a6000_next6h_cleanup.json)을 보존했습니다. 이전 원격 Git 가중치를 다시 받아 해시를 확인하고 필요한 대조군을 새 reference로 이관한 뒤 이전 중복 실행 폴더를 정리했습니다. 약467MiB를 회수했습니다. prepare_next6h_references.py는 이 일회성 절차의 기록이며 삭제된 원본을 전제로 하므로 다시 실행하지 않습니다.

CtrS_old 환경과 CtrS-a6000 데이터 루트는 유지합니다. 원본 full checkpoint가 보관되지 않은 LR 보정·고주파 실험은 유지했습니다. 새 결과는 원격 가중치 복구가 검증될 때까지 서버 실행 폴더를 유지합니다. 커밋만으로 무시된 .pt가 보관됐다고 판단하지 않습니다.
