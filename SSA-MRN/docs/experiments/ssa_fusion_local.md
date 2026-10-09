# QB SSA 핵심 융합 A0–A3 · 100에폭 비교

2026-10-09, Kaggle T4×2에서 QB K6 seed46 네 조건을 두 차례에 걸쳐 각 100에폭 학습했다. A0/A1, A2/A3의 원본 Output ZIP을 각각 내려받고 파일 크기·SHA-256 manifest 및 ZIP 무결성을 검증했다. **단일 시드 validation에서는 기존 A0의 최저 MSE가 가장 낮다.** RR/FR 성능은 아직 평가하지 않았으므로 새 구조의 채택을 보류한다.

## 조건과 구조

train 17,139 / validation 1,905, PAN 1×64×64, MS 4×16×16, LMS·GT 4×64×64, seed46, K6, 100에폭, effective batch32, micro16, AMP, Adam lr1e-4, 증강 없음. A0/A1은 GPU0/GPU1, 이어서 A2/A3은 GPU0/GPU1에서 실행했다. 데이터 SHA-256은 두 실행의 설정 JSON에서 동일하다. 동일한 backbone·다중 해상도 경로·손실·분할을 사용한다. T4 실행의 `deterministic=False`이므로 비트 단위 재현은 보장하지 않는다.

| 조건 | SSA 변경 | 파라미터 |
|---|---|---:|
| A0 | 공개 구현의 가로·세로 전치 및 전체 H×W softmax 유지 | 431,046 |
| A1 | 관련 두 H/W permute를 함께 제거 | 431,046 |
| A2 | A1의 정렬을 유지하고 전체 공간 softmax를 sigmoid 게이트로 교체 | 431,046 |
| A3 | MS Query·PAN Key/Value의 5×5 국소 교차 어텐션으로 교체 | 431,550 |

A3은 head 1개, 채널 K=6, 위치당 최대25개 이웃을 사용한다. 경계 밖은 마스킹하여 유효 이웃만 정규화하고, 출력은 B×6×H×W다. 위치별 계산은 행 단위로 나눠 메모리를 제한한다. A2는 정규화 범위뿐 아니라 출력 진폭도 바꾼다. A3은 Value projection과 채널 내적까지 바꾸므로 A2→A3을 국소성 하나의 효과로 해석하지 않는다. [구현](../../src/ssamrn/models/ssa_fusion.py)과 [합성 사전 검증](../assets/ssa_fusion_preflight/verification.json)을 참조한다.

## Validation 결과

각 행은 **validation MSE가 최소인 checkpoint**의 PSNR·SAM이다. PSNR은 band mean, peak=1이며 SAM은 도 단위다.

| 조건 | 최적 에폭 | MSE ↓ | A0 대비 MSE | PSNR dB ↑ | SAM ° ↓ |
|---|---:|---:|---:|---:|---:|
| A0 기존 SSA | 99 | 0.000171330678 | 기준 | 39.312961 | 4.555021 |
| A1 전치 제거 | 99 | 0.000172062520 | +0.427% | 39.310067 | 4.552727 |
| A2 sigmoid | 99 | 0.000171955203 | +0.365% | 39.309274 | 4.560935 |
| A3 5×5 교차 어텐션 | 99 | 0.000171885232 | +0.324% | 39.299238 | 4.570770 |

A1의 SAM은 A0보다 약 0.0023° 낮지만 MSE와 PSNR은 나쁘다. A2·A3의 MSE·PSNR·SAM은 이 validation 조건에서 모두 A0보다 나쁘다. A3의 최저 MSE는 A2보다 약 0.000000070 낮지만, PSNR·SAM은 더 나쁘다. 단일 시드의 작은 차이로 일반화 또는 통계적 우위를 주장하지 않는다.

![A0/A1 학습·validation 곡선](../assets/ssa_fusion_A0_A1_qb_k6_s46/validation_curves.png)

![A2/A3 학습·validation 곡선](../assets/ssa_fusion_A2_A3_qb_k6_s46/validation_curves.png)

## 산출물과 복구 상태

[A0/A1 원본](../assets/ssa_fusion_A0_A1_qb_k6_s46/)과 [A2/A3 원본](../assets/ssa_fusion_A2_A3_qb_k6_s46/)에 모델별 best/latest 및 이전 백업, 100에폭 history, 로그, complete/status/provenance, 설정, 실행 소스 snapshot, 데이터·코드·파일 해시를 보관했다. 두 ZIP 모두 CRC 검사를 통과했고 각 artifact manifest의 28개 파일이 크기·SHA-256과 일치한다. 네 모델의 체크포인트 ZIP 컨테이너는 로컬에서 CRC 검사했다. **A0/A1은 CPU `torch.load(weights_only=True)`를 확인했지만 A2/A3의 PyTorch 로드 및 실제 재개는 아직 확인하지 못했다.** A2/A3는 별도로 [다운로드 검증 기록](../assets/ssa_fusion_A2_A3_qb_k6_s46/download_verification.json)을 남겼다.

Kaggle Draft나 셀의 ZIP 링크만으로 원격 보관됐다고 판단하지 않는다. 이번 보존 근거는 실제 내려받은 ZIP과 로컬 manifest 대조다. [Kaggle 한 셀](../../scripts/kaggle_ssa_fusion_cell.py), [동일 ipynb](../../scripts/kaggle_ssa_fusion.ipynb), [생성기](../../scripts/build_kaggle_ssa_fusion.py), [합성 검증 코드](../../scripts/verify_ssa_fusion.py)를 함께 둔다. A2/A3의 실제 실행 셀 수정본도 산출물 폴더에 기록했다.

## 판단과 다음 검증

현재 결과는 QB validation 하나의 구조 진단이다. A0/A1/A2/A3의 RR PSNR·SAM·ERGAS와 장면별 차이, FR 지표·시각 결과, 추론 시간·GPU 메모리를 같은 평가 코드로 확인해야 한다. FR QNR은 구현 검증 전 잠정 수치로 표시한다. 기존 test는 이미 탐색에 쓰였으므로 독립 최종 holdout이라 부르지 않고, test로 에폭이나 모델을 선택하지 않는다. 후보가 생기면 QB 3시드와 GF2·WV3로 확장한다. 30에폭은 동작·발산 점검용이며 본 비교 판단은 100에폭과 validation 선택을 기준으로 한다. 이번 작업에서 새 학습이나 RR/FR 평가는 시작하지 않았다.
