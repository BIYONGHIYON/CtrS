# A6000 후속 1~3단계 실험

[첫 구조 비교 결과](../experiments/improvements/a6000_architecture_qb.md)

## 목적과 상태

23탭의 작은 개선이 시드를 바꿔도 유지되는지 확인하고, 확대 위치와 센서 의존성을 분석합니다. Windows 손실 탐색은 그대로 유지하며 **4단계 손실 결합은 포함하지 않습니다.**

2026-10-08: **후속10개 본 학습 모두100에폭 완료**, 기존 QB 시드42 두 모델 재사용, best12개 모델 각각 RR20/FR20 전체 평가 완료. [결과 보고서](../experiments/improvements/a6000_followup_123.md)

## 큐 순서 · 최대 4개 병렬

| 단계 | 새 학습 | 재사용 | 새 학습 수 |
|---|---|---|---:|
| 1. 반복 검증 | QB K6 기준선/전체 23탭 × 시드 43·44 | 기존 시드 42 쌍은 분석 때 함께 비교 | 4 |
| 2. 위치 분석 | QB K6 입력 쪽만/출력 쪽만 23탭 × 시드 42 | 동일 조건의 기존 기준선/전체 23탭 | 2 |
| 3. 센서 확장 | GF2·WV3 K6 기준선/전체 23탭 × 시드 42 | 없음 | 4 |

새 학습은 **10회 × 100에폭 = 1,000에폭**입니다. 단계는 1→2→3으로 진행하며, 이전 단계의 모든 작업이 끝난 뒤 다음 단계가 시작됩니다. 2단계는 완료된 두 결과를 재사용하므로 실제 실행은 2개입니다. GPU는 한 장이며 프로세스 수만으로 처리량 증가를 가정하지 않습니다.

위치 구분:

- 입력 쪽: `upsample1` ×4(PAN 복원), `upsample100` ×2(MS 중간 확대)
- 출력 쪽: `upsample101`·`upsample102` ×2(최종 복원)
- 전체: 위 네 곳 모두
- 제공 LMS·기존 축소 bilinear·SSA 내부 연산은 유지

## 공통 통제와 새 기록

K=6, Adam·lr 1e-4, batch=micro batch=32, 100에폭, 기본 MSE, CUDA FP32·결정론 설정입니다. 같은 센서·시드의 기준선과 변형은 동일 공통 계층 초기화와 데이터 순서를 사용합니다. GF2는 4밴드, WV3는 8밴드이며 센서별 정규화와 데이터 분할을 유지합니다.

새 학습은 매 에폭 다음을 기록합니다.

- train MSE / validation MSE
- **실제 validation band-mean PSNR(peak=1)**: 이미지마다 밴드별 PSNR 평균 후 이미지 평균
- **실제 validation SAM(°)**: 이미지마다 유효 픽셀 SAM 평균 후 이미지 평균

best는 validation MSE로만 선택합니다. 기존 완료 실험의 validation SAM 곡선은 없으므로 재사용 결과에 새 곡선이 있다고 쓰지 않습니다. 새 validation PSNR은 이전 파생 global PSNR과 정의가 다르며 비교 그래프에서 구분합니다. 특히 GF2는 RR 보고서의 고정 peak2047 PSNR과 같은 수치가 아닙니다.

위치·센서 실험은 사전 고정한 가설 검증입니다. 같은 QB test를 이미 탐색에 사용했으므로 반복 test 분석을 새로운 독립 검증으로 포장하지 않습니다. 최종 모델 선택은 validation을 기준으로 하고, 추후 test 보고에서 사용 이력을 명시합니다.

## 실행 위치와 보존

| 항목 | 위치 |
|---|---|
| 새 코드 | `/home/gpu_04/CtrS-a6000-followup` |
| 브랜치 | `a6000-followup-suite` |
| 설정 | `SSA-MRN/scripts/a6000_followup_plan.json` |
| 새 결과 | `SSA-MRN/experiments/a6000_followup_123` |
| 기존 Python | `/home/gpu_04/CtrS_old/SSA-MRN/.conda-env/bin/python` |
| 원본 데이터 | `/home/gpu_04/CtrS-a6000/SSA-MRN/data/dataset` |
| 기존 비교 결과 | `/home/gpu_04/CtrS-a6000/SSA-MRN/experiments/a6000_architecture_qb` |

**`CtrS-a6000`와 `CtrS_old`를 삭제하지 않습니다.** 새 실행은 이 폴더의 데이터·재사용 가중치·환경을 참조합니다. 재사용 조건은 기존 complete config·100에폭 완료, 공통 코어/23탭 코드 해시로 검증합니다. 원본 파일은 복사·변경하지 않고 경로로 참조합니다. `references.json`과 inventory에 참조 위치·가중치/기록 해시를 보존합니다.

큐는 중복 컨트롤러를 잠금으로 차단하고 다른 GPU 작업이 있으면 시작을 거부합니다. 실행 중 코드·데이터·참조 결과가 바뀌면 중단합니다. 한 학습이 실패하면 이 큐의 자식만 종료하고 체크포인트를 보존합니다.

## 실행 명령

학교 서버에서:

```bash
cd /home/gpu_04/CtrS-a6000-followup
./SSA-MRN/scripts/run_a6000_followup.sh check
./SSA-MRN/scripts/run_a6000_followup.sh status
```

본 학습 시작:

```bash
./SSA-MRN/scripts/run_a6000_followup.sh start
```

확인:

```bash
./SSA-MRN/scripts/run_a6000_followup.sh logs
watch -n 2 './SSA-MRN/scripts/run_a6000_followup.sh status'
tail -n 5 -F SSA-MRN/experiments/a6000_followup_123/*/train.log
```

중단 후 `resume-latest`로 재개합니다. 완료 run은 건너뛰고 마지막 완료 epoch의 모델·Adam·RNG·데이터 순서를 복원합니다. SSH/로그 화면 종료는 학습을 중단하지 않습니다. 서버 재부팅 후 자동 재개는 없습니다.

## 사전 검증과 학습 이후

단계 4/2/4회, 단계 간 겹침 없음, 최대 동시 4개, 재사용 2개를 큐 모의 실행으로 확인했습니다. CUDA에서 세 센서의 밴드 수·공통 초기화·확대 위치·유한 gradient와 새 validation MSE/PSNR/SAM의 기준 계산 일치를 검증했습니다. GF2/WV3의 실제 32패치 병렬 검증은 본 연구 결과와 구분합니다.

학습 이후 단계별 validation 차이, 시드별 차이와 평균/표준편차, 독립 RR/FR 지표, 사전 고정한 5장면, 곡선·가중치·해시를 정리합니다. 성능 차이가 일관적인지 확인하기 전에는 23탭을 최종 채택하지 않습니다. LR 보정·고주파·Windows 손실 결합은 이번 큐에서 실행하지 않습니다.


## 완료 후 평가·가중치 검증

학습 큐 `finished`와 GPU가 비어 있음을 확인한 뒤 아래 평가를 SSH와 독립적으로 실행합니다. 학습 재시작 명령이 아닙니다.

```bash
cd /home/gpu_04/CtrS-a6000-followup
nohup /home/gpu_04/CtrS_old/SSA-MRN/.conda-env/bin/python SSA-MRN/scripts/evaluate_a6000_followup.py --data-root /home/gpu_04/CtrS-a6000/SSA-MRN/data/dataset > /home/gpu_04/a6000_followup_eval.log 2>&1 < /dev/null &
```

산출물은 `SSA-MRN/docs/assets/a6000_followup_123`입니다. `selection.json`을 먼저 고정하고, best validation MSE 가중치만 평가합니다. 원본 best/latest를 변환 없이 `weights/<run ID>/`로 복사하며 source/복사본 해시를 확인합니다. H5 데이터는 읽기 전용입니다.

기록 보관에는 학습 폴더의 `inventory.json`, `plan.json`, `references.json`, `validation_scores.json`을 각각 `training_inventory.json`, `training_plan.json`, `training_references.json`, `validation_scores.json`으로 산출물 폴더에 복사하고 `report_manifest.json`의 artifact 목록/해시에 포함합니다. 실행 기록을 삭제하지 않고 평가와 분리해 보관합니다.

```bash
python3 SSA-MRN/scripts/verify_a6000_followup_report.py
```

Git의 원본24개 가중치에는 Adam·RNG·checkpoint config가 있으며 전체 환경·데이터를 포함하지 않습니다. 재개할 때 config에 기록된 서버 경로와 Python/의존성을 준비하고 저장된 코드 해시를 맞춰야 합니다. GitHub에서 파일을 실제 복구해 SHA-256 일치와 checkpoint 구조가 확인되기 전에는 보관 검증을 완료했다고 쓰지 않습니다. `CtrS_old`·`CtrS-a6000`은 참조 환경·데이터·기존 결과이므로 그대로 유지합니다.


2026-10-08 원격 검증 완료: GitHub의 결과 commit `182e2fa01b39bccd10241810d0cee8faa9c84646`을 fresh clone해 원본24개 가중치와 산출물71개를 실제로 읽어 SHA-256을 대조했습니다. [파일별 검증 기록](a6000_followup_remote_verification.json). 원본 서버 폴더는 그대로 유지합니다.
