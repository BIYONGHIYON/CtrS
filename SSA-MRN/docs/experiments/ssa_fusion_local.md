# QB SSA 핵심 융합 A0–A3 · 100에폭 구조 진단

[실험 목록](../../README.md#experiments) · [수치](#results) · [그래프](#graphs) · [이미지](#images) · [가중치](#evidence)

| 비교 기준 | 이번에 바꾼 점 | 관측된 차이 | 판단 |
|---|---|---|---|
| QB K6 seed46 A0, validation 선택 | A1 전치 제거, A2 sigmoid, A3 5×5 국소 교차 어텐션 | A1/A2/A3 최저 validation MSE가 A0보다 각각 0.427/0.365/0.324% 높음 | 100에폭 학습·가중치 보관 완료, RR/FR 평가 전 구조 채택 보류 |

## 1. 목적과 상태

2026-10-09, Kaggle T4×2에서 QB K6 seed46 네 조건을 두 차례에 걸쳐 각100에폭 학습했다. 목적은 SSA의 전치 및 전체 공간 정규화가 영상 크기에 따라 불안정할 수 있다는 **가설**을 진단하고, 국소 PAN–MS 대응 구조를 비교하는 것이다. 현재 연구 기본값 K4와 별개로 시작된 **과거 K6 구조 진단**이며 K4 성능 주장에 사용하지 않는다. A0/A1, A2/A3 원본 Output ZIP을 내려받아 파일 해시를 검증했다. 이 브랜치 코드 commit은 `4bf72ad`이고, 병합 충돌 해결 commit은 후속 기록으로 확인한다.

## 2. 변경 사항과 평가 조건

QB train 17,139 / validation 1,905, PAN 1×64×64, MS 4×16×16, LMS·GT 4×64×64, seed46, K6, 목표100에폭, effective batch32, micro16, AMP, Adam lr1e-4, 증강 없음. A0/A1은 GPU0/GPU1, 이어서 A2/A3은 GPU0/GPU1에서 실행했다. 데이터 SHA-256은 두 실행의 설정 JSON에서 동일하다. 나머지 backbone·다중 해상도 경로·손실·분할은 같다. T4 실행의 `deterministic=False`이므로 비트 단위 재현은 보장하지 않는다. RR/FR의 독립 test 장면 평가는 아직 실행하지 않았다.

| 조건 | SSA 변경 | 파라미터 |
|---|---|---:|
| A0 | 공개 구현의 가로·세로 전치 및 전체 H×W softmax 유지 | 431,046 |
| A1 | 관련 두 H/W permute를 함께 제거 | 431,046 |
| A2 | A1 정렬을 유지하고 전체 공간 softmax를 sigmoid 게이트로 교체 | 431,046 |
| A3 | MS Query·PAN Key/Value의 5×5 국소 교차 어텐션으로 교체 | 431,550 |

A3은 head 1개, 채널 K=6, 위치당 최대25개 이웃을 사용한다. 경계 밖은 마스킹하여 유효 이웃만 정규화하고 출력은 B×6×H×W다. 위치별 계산은 행 단위로 나눠 메모리를 제한한다. A2는 정규화 범위뿐 아니라 출력 진폭도 바꾼다. A3은 Value projection과 채널 내적까지 바꾸므로 A2→A3을 국소성 하나의 효과로 해석하지 않는다. [구현](../../src/ssamrn/models/ssa_fusion.py)과 [합성 사전 검증](../assets/ssa_fusion_preflight/verification.json)을 참조한다.

<a id="results"></a>

## 3. 정량 결과

각 행은 **validation MSE가 최소인 checkpoint**의 PSNR·SAM이다. PSNR은 band mean, peak=1이며 SAM은 도 단위다. test RR/FR 수치는 아직 없다.

| 조건 | 최적 에폭 | MSE ↓ | A0 대비 MSE | PSNR dB ↑ | SAM ° ↓ |
|---|---:|---:|---:|---:|---:|
| A0 기존 SSA | 99 | 0.000171330678 | 기준 | 39.312961 | 4.555021 |
| A1 전치 제거 | 99 | 0.000172062520 | +0.427% | 39.310067 | 4.552727 |
| A2 sigmoid | 99 | 0.000171955203 | +0.365% | 39.309274 | 4.560935 |
| A3 5×5 교차 어텐션 | 99 | 0.000171885232 | +0.324% | 39.299238 | 4.570770 |

## 4. 직전 연구와 수치 차이

이번 통제 비교의 직전 조건은 A0다. A1−A0의 validation PSNR/SAM/MSE는 −0.002894 dB / −0.002294° / +7.31842e−7, A2−A0는 −0.003687 dB / +0.005914° / +6.24525e−7, A3−A0는 −0.013722 dB / +0.015749° / +5.54554e−7이다. A1의 SAM은 조금 낮지만 MSE와 PSNR은 나쁘다. A3의 MSE는 A2보다 약 6.9971e−8 낮지만 PSNR·SAM은 나쁘다. 직전 MTF 증강 연구는 학습 변경점이 달라 이 구조 조건과 인과적으로 직접 비교하지 않는다.

<a id="graphs"></a>

## 5. 그래프

![A0/A1 학습·validation 곡선](../assets/ssa_fusion_A0_A1_qb_k6_s46/validation_curves.png)

![A2/A3 학습·validation 곡선](../assets/ssa_fusion_A2_A3_qb_k6_s46/validation_curves.png)

train MSE 및 validation MSE·PSNR·SAM의 실제 100에폭 기록이다. RR/FR test baseline 비교 그래프는 평가 전이므로 없다.

<a id="images"></a>

## 6. 결과 이미지 예시

RR/FR 예측을 아직 생성하지 않았다. 따라서 사전 고정 RR 5장면의 LR MS·PAN·예측·GT 4패널과 FR 시각 결과는 **없으며**, 이미지 품질에 관한 결론도 보류한다. 향후 장면 ID·타일·seed·순서를 결과 확인 전에 고정해 기록한다.

<a id="evidence"></a>

## 7. 가중치와 검증 근거

[A0/A1 원본](../assets/ssa_fusion_A0_A1_qb_k6_s46/)과 [A2/A3 원본](../assets/ssa_fusion_A2_A3_qb_k6_s46/)에 모델별 best/latest 및 이전 백업, 100에폭 history, 로그, complete/status/provenance, 설정, 실행 소스 snapshot, 데이터·코드·파일 해시를 보관했다. 각 폴더의 `artifact_manifest.json`은 파일별 SHA-256이며, 두 ZIP 모두 CRC와 manifest 28개 파일의 크기·SHA-256 검사를 통과했다. 네 모델의 checkpoint ZIP 컨테이너도 CRC 검사했다. A0/A1은 CPU `torch.load(weights_only=True)`를 확인했지만 A2/A3의 PyTorch 로드와 실제 재개는 아직 확인하지 못했다. A2/A3는 [다운로드 검증 기록](../assets/ssa_fusion_A2_A3_qb_k6_s46/download_verification.json)을 남겼다. 각 best는 99에폭이고 latest는 100에폭이다.

Kaggle Draft나 셀의 ZIP 링크만으로 원격 보관됐다고 판단하지 않는다. 이번 근거는 내려받은 ZIP과 Git 브랜치의 원본 가중치다. [Kaggle 한 셀](../../scripts/kaggle_ssa_fusion_cell.py), [동일 ipynb](../../scripts/kaggle_ssa_fusion.ipynb), [생성기](../../scripts/build_kaggle_ssa_fusion.py), [합성 검증 코드](../../scripts/verify_ssa_fusion.py)를 함께 둔다. A2/A3의 실제 실행 셀 수정본도 산출물 폴더에 기록했다.

## 8. 한계와 다음 판단

현재 결과는 단일 시드 QB validation 구조 진단이다. A0–A3의 RR PSNR·SAM·ERGAS와 장면별 차이, FR 지표·시각 결과, 추론 시간·GPU 메모리를 같은 평가 코드로 확인해야 한다. FR QNR은 구현 검증 전 잠정 수치로 표시한다. 기존 test는 탐색에 쓰였으므로 독립 최종 holdout이라 부르지 않고 test로 에폭이나 모델을 선택하지 않는다. 유망 후보가 생기면 QB 3시드와 GF2·WV3로 확장하고, 현재 기본값 K4에 적용할지는 K4 기준선과 별도로 비교한다. 30에폭은 동작·발산 점검용이며 본 비교는100에폭을 기준으로 한다. 이번 작업에서 새 학습이나 RR/FR 평가는 시작하지 않았다.
