# 반복 복원의 단계 간 특징 전달과 관측·PAN 차이

**변경 내용:** 기존 밴드별 SSAI를 유지하면서 반복 횟수, 단계 간 특징 전달, MS 관측 오차와 PAN 특징 차이를 비교했습니다.

| 확인된 결과 | 상태 | 개선 여부 |
|---|---|---|
| D3: 원본 D0 대비 PSNR +0.0956 dB, SAM −0.1127°, 잠정 QNR +0.0159. D1 대비 PSNR +0.1078 dB | 평가 완료 | **부분 개선** |

[실험 목록](../../README.md#experiments) · [수치](#results) · [그래프](#graphs) · [이미지](#images) · [가중치·근거](#evidence)

## 1. 목적과 상태

**2026-10-12 08:32:09 KST에 7개 × 50에포크 = 350에포크와 자동 평가가 완료됐습니다.** 전체 history·완료 기록·체크포인트를 확인했습니다. 실패·건너뛴 작업이 없고 AMP step 누락도 전 실험 0회였습니다.

직전 공유 복원 C1의 이득이 다른 seed에서도 유지되는지 확인하고, 밴드별 SSAI를 유지한 채 반복 단계에 전달하는 정보를 바꿨습니다. validation으로 D3를 선택한 뒤 D0·D1·D3의 전체 RR·FR test를 평가했습니다. D2·D4·D5·D6의 test 수치는 없습니다.

## 2. 변경한 구조와 비교 조건

| ID | 실제 구조 | 파라미터 | 직접 비교 |
|---|---|---:|---|
| D0 | 원본 SSA-MRN K4 단일 복원 | 426,654 | 같은 조건의 기준선 |
| D1 | 기존 C1: error encoder와 학습 가능한 α를 사용하는 공유 3단계 복원 | 426,803 | D0 대비 반복 효과 |
| D2 | D1의 반복을 2단계로 변경 | 426,803 | D1 대비 반복 횟수 |
| D3 | D1 + 크기별 단계 간 특징 전달 | 426,875 | D1 대비 특징 전달 |
| D4 | D1 + MS 관측 오차 보정 | 426,807 | D1 대비 관측 오차 |
| D5 | D3 + MS 관측 오차 보정 | 426,879 | D3/D4 대비 결합 |
| D6 | D3 + 밴드별 PAN 특징 차이 | 428,663 | D3 대비 PAN 경로 추가 |

모든 모델은 전체·1/2·1/4 크기의 기존 4밴드별 SSAI, 복원 decoder와 **K=4**를 유지합니다. D1에서 D6까지 core·error encoder·α의 초기값이 같으며 α는 0.1, 새 gate는 0으로 시작했습니다. D3에서 D6까지 시작 시 D1과 같은 출력임을 사전 검증했습니다. 활성화 함수 교체 실험은 아닙니다.

- **특징 전달:** 앞 단계의 전체·1/2 크기 융합 특징과 1/4 크기 SSAI 특징을 다음 단계의 같은 위치에 더합니다. 1×1 Conv와 밴드별 tanh gate를 사용합니다. detach 없이 gradient를 전달하며 batch 간 특징을 보존하지 않습니다.
- **MS 관측 오차:** 현재 영상 X에서 `MS−D(X)`를 계산한 뒤 HR 크기로 확대하고 밴드별 gate를 거쳐 error encoder에 넣습니다. D는 bilinear 1/4 축소, align_corners=True입니다. 센서 MTF를 보정한 관측이 아닙니다.
- **PAN 특징 차이:** 각 밴드·크기의 SSAI에서 `PAN_feature−Conv(MS_feature)`를 계산하고 gate를 거쳐 기존 SSAI 출력에 추가합니다. 공식 transpose·softmax 경로를 유지합니다. D6−D3은 추가 경로 전체의 효과이며 subtraction 자체만의 효과를 분리하지 못합니다.

| 공통 조건 | 실제 설정 |
|---|---|
| 데이터 | QB train 17,139 / validation 1,905 패치, RR 20장 / FR 20장 |
| 입력 | HR 64×64 / LR 16×16, DN 2047 정규화 |
| 학습 | 새 학습 50에포크, seed **43**, Adam lr 1e−4 고정, 최종 출력 MSE |
| 배치 | micro = effective **96**, worker 각 1개, 마지막 불완전 배치 포함 |
| 장치 | Windows RTX 3060 Ti 8GB, Ryzen 5600X, RAM 15GB, 최대 2개 병렬 |
| 최적화 | AMP FP16 + GradScaler, 동일 epoch 순서(seed43+epoch), TF32 비활성, cuDNN autotune |
| 평가·선택 | FP32, validation MSE 최소 → 정확한 동률이면 SAM → 먼저 선택한 epoch |

같은 초기 core 해시를 전 모델에서 확인했습니다. test로 후보·epoch를 다시 선택하지 않았습니다. 실측 공통 배치와 실행 조건은 [plan](../assets/windows_recurrent_guidance_qb_50/plan.json), [benchmark](../assets/windows_recurrent_guidance_qb_50/benchmark.json)에 보존했습니다.

<a id="results"></a>

## 3. 정량 결과

### 7개 실험의 validation 비교

| 모델 | best epoch / 50 | Validation MSE ↓ | D0 대비 MSE | Validation SAM ↓ |
|---|---:|---:|---:|---:|
| D3 | 50 | 0.000222498 | -2.053% | 4.926187° |
| D6 | 50 | 0.000223126 | -1.777% | 4.920961° |
| D4 | 50 | 0.000223132 | -1.774% | 4.918860° |
| D5 | 50 | 0.000224296 | -1.262% | 4.921730° |
| D0 | 49 | 0.000227162 | +0.000% | 5.019070° |
| D1 | 50 | 0.000227806 | +0.283% | 4.943834° |
| D2 | 50 | 0.000227998 | +0.368% | 4.939151° |

**D3가 validation 1위입니다.** D6와 D4도 D1보다 낮지만 차이는 작으며 test 평가 대상은 아닙니다. D0 외 모든 후보의 best가 마지막 50에포크여서 장기 수렴은 미확정입니다.

### 전체 RR·FR test 장면 평균

| 모델 | RR PSNR ↑ | RR SAM ↓ | RR ERGAS ↓ | 잠정 FR QNR ↑ | 잠정 FR D_s ↓ |
|---|---:|---:|---:|---:|---:|
| D0 | 36.285404 | 5.492340° | 4.755496 | 0.881538 | 0.073202 |
| D1 | 36.273192 | 5.392476° | 4.775516 | 0.881093 | 0.086080 |
| D3 | 36.380975 | 5.379622° | 4.711349 | 0.897448 | 0.065137 |

- **D3−D0:** PSNR **+0.095571 dB**, SAM **−0.112717°**, ERGAS **−0.044147**, 잠정 QNR **+0.015910**, D_s **−0.008065**입니다. RR MSE는 −1.612%지만 경계 영역 MSE는 +0.586%로 악화했습니다.
- **D3−D1:** PSNR **+0.107783 dB**, SAM **−0.012854°**, ERGAS **−0.064166**, 잠정 QNR **+0.016355**, D_s **−0.020943**입니다. RR MSE는 −2.718%지만 FR bilinear LR MSE는 **+7.181%**로 악화했습니다.
- **D1−D0:** PSNR **−0.012212 dB**, SAM **−0.099864°**, ERGAS **+0.020020**, 잠정 QNR **−0.000445**, D_s **+0.012878**입니다. 반복만 추가한 D1의 이득은 지표별로 엇갈렸습니다.

[전체 장면별 지표](../assets/windows_recurrent_guidance_qb_50/evaluation/metrics.json) · [validation 순위](../assets/windows_recurrent_guidance_qb_50/ranking.json) · [평가 전 후보 선택](../assets/windows_recurrent_guidance_qb_50/evaluation/selected_models.json)

20개 H5 항목 전체를 평가했습니다. 원래 촬영 장면 ID가 확인되지 않아 20개가 서로 독립적인 원본 촬영 장면이라고 단정하지 않습니다. PSNR은 장면별 밴드 PSNR 평균을 다시 평균한 값입니다. QNR·Q2n·SCC는 MATLAB parity 검증이 완료되지 않았습니다.

## 4. 기준선 대비 차이와 해석

D3는 같은 조건의 D1에 특징 전달을 더해 validation MSE를 **2.330%** 낮췄습니다. 파라미터는 72개 증가했습니다. 주요 RR·잠정 FR 평균은 개선됐지만 이득이 작고 모든 오차 지표가 개선된 것은 아닙니다.

D2의 2단계는 D1보다 validation MSE가 **0.084%** 높았습니다. D4와 D6는 개선됐지만 D3를 넘지 못했고, D5 결합은 D3보다 **0.808%** 높았습니다. 관측·PAN 차이의 최종 test 개선은 미평가이므로 확정하지 않습니다.

[직전 seed42 C1](dual_stream_reconstruction.md)은 같은 조건의 C0 대비 PSNR +0.244200 dB·잠정 QNR +0.024991이었습니다. 이번 seed43 D1은 −0.012212 dB·−0.000445로 그 이득을 재현하지 못했습니다. 각 실행의 대조군 대비 차이를 비교하며 서로 다른 기준선의 절대 점수로 구조 효과를 단정하지 않습니다. Linux 구조 실험과 Windows 손실 결과를 결합한 성능도 아직 확정하지 않습니다.

<a id="graphs"></a>

## 5. 학습·평가 그래프

![7개 실험 train MSE·validation MSE·밴드 평균 PSNR·SAM](../assets/windows_recurrent_guidance_qb_50/evaluation/curves.png)

validation 밴드 평균 PSNR과 전체 MSE에서 유도한 global PSNR은 다르며 원본 history에서 구분했습니다.

![반복 갱신 α와 특징·관측·PAN gate 변화](../assets/windows_recurrent_guidance_qb_50/evaluation/gate_curves.png)

![동일 조건 D0 대비 D3 전체 RR·FR 차이](../assets/windows_recurrent_guidance_qb_50/evaluation/test_comparison.png)

<a id="images"></a>

## 6. 결과 이미지

평가 전에 고정한 RR index **1·8·11·13·19**(0-based)의 전체 장면입니다. **LR MS · PAN · 예측 · GT** 순서이며 RGB [2,1,0] 가정, GT 1–99% 공통 대비를 적용했습니다. LR MS만 최근접 확대했고 FR 정답은 만들지 않았습니다.

### D3 · validation으로 선택한 특징 전달 후보

![D3 고정 장면 8](../assets/windows_recurrent_guidance_qb_50/evaluation/D3_scene_08.png)

<details>
<summary>D0·D1·D3의 고정 5장면 전체 이미지 15개</summary>

### D0

![D0 장면 1](../assets/windows_recurrent_guidance_qb_50/evaluation/D0_scene_01.png)

![D0 장면 8](../assets/windows_recurrent_guidance_qb_50/evaluation/D0_scene_08.png)

![D0 장면 11](../assets/windows_recurrent_guidance_qb_50/evaluation/D0_scene_11.png)

![D0 장면 13](../assets/windows_recurrent_guidance_qb_50/evaluation/D0_scene_13.png)

![D0 장면 19](../assets/windows_recurrent_guidance_qb_50/evaluation/D0_scene_19.png)

### D1

![D1 장면 1](../assets/windows_recurrent_guidance_qb_50/evaluation/D1_scene_01.png)

![D1 장면 8](../assets/windows_recurrent_guidance_qb_50/evaluation/D1_scene_08.png)

![D1 장면 11](../assets/windows_recurrent_guidance_qb_50/evaluation/D1_scene_11.png)

![D1 장면 13](../assets/windows_recurrent_guidance_qb_50/evaluation/D1_scene_13.png)

![D1 장면 19](../assets/windows_recurrent_guidance_qb_50/evaluation/D1_scene_19.png)

### D3

![D3 장면 1](../assets/windows_recurrent_guidance_qb_50/evaluation/D3_scene_01.png)

![D3 장면 8](../assets/windows_recurrent_guidance_qb_50/evaluation/D3_scene_08.png)

![D3 장면 11](../assets/windows_recurrent_guidance_qb_50/evaluation/D3_scene_11.png)

![D3 장면 13](../assets/windows_recurrent_guidance_qb_50/evaluation/D3_scene_13.png)

![D3 장면 19](../assets/windows_recurrent_guidance_qb_50/evaluation/D3_scene_19.png)

</details>

<a id="evidence"></a>

## 7. 재현 자료

| 모델 | best epoch | validation best | 마지막 epoch 50 |
|---|---:|---|---|
| D3 | 50 | [best.pt](../assets/windows_recurrent_guidance_qb_50/D3/best.pt) | [latest.pt](../assets/windows_recurrent_guidance_qb_50/D3/latest.pt) |
| D6 | 50 | [best.pt](../assets/windows_recurrent_guidance_qb_50/D6/best.pt) | [latest.pt](../assets/windows_recurrent_guidance_qb_50/D6/latest.pt) |
| D4 | 50 | [best.pt](../assets/windows_recurrent_guidance_qb_50/D4/best.pt) | [latest.pt](../assets/windows_recurrent_guidance_qb_50/D4/latest.pt) |
| D5 | 50 | [best.pt](../assets/windows_recurrent_guidance_qb_50/D5/best.pt) | [latest.pt](../assets/windows_recurrent_guidance_qb_50/D5/latest.pt) |
| D0 | 49 | [best.pt](../assets/windows_recurrent_guidance_qb_50/D0/best.pt) | [latest.pt](../assets/windows_recurrent_guidance_qb_50/D0/latest.pt) |
| D1 | 50 | [best.pt](../assets/windows_recurrent_guidance_qb_50/D1/best.pt) | [latest.pt](../assets/windows_recurrent_guidance_qb_50/D1/latest.pt) |
| D2 | 50 | [best.pt](../assets/windows_recurrent_guidance_qb_50/D2/best.pt) | [latest.pt](../assets/windows_recurrent_guidance_qb_50/D2/latest.pt) |

**원본 full checkpoint 14개**를 보존했습니다. model·Adam·scaler·RNG·epoch cursor·history를 포함하며 inference용으로 축소하지 않았습니다. 필요한 train·validation·test 데이터 해시는 보관했으며 원본 데이터와 재생성 가능한 NPY cache는 Git에 넣지 않았습니다. 이전 epoch 비상 복사본 previous.pt는 별도로 서버에 유지했습니다.

[원본 보관 manifest](../assets/windows_recurrent_guidance_qb_50/preservation.json) · [전달 manifest](../assets/windows_recurrent_guidance_qb_50/report_manifest.json) · [실행 소스 snapshot](../assets/windows_recurrent_guidance_qb_50/source_snapshot/scripts/windows_recurrent_guidance_suite.py) · [데이터 manifest](../assets/windows_recurrent_guidance_qb_50/data_manifest.json)

[모델](../../src/ssamrn/models/recurrent_guidance.py) · [관리 코드](../../scripts/recurrent_guidance_suite.py) · [운영·재개 명령](../operations/recurrent_guidance.md) · [결과·복구 검증](../../scripts/verify_recurrent_guidance_results.py)

검증기는 350개 epoch 기록, validation 선택, 120개 장면별 RR·FR 결과와 평균, 15개 이미지, 14개 best/latest의 해시·strict load·유한한 forward를 확인합니다. 실제 QB RR index 0을 CPU FP32로 다시 추론해 D0·D1·D3의 저장된 GPU MSE와 일치함을 확인했습니다([실제 장면 검증](../assets/windows_recurrent_guidance_qb_50/real_scene_verification.json)). 원격 복구 검증 기록은 푸시 후 이 절에 연결합니다.

```bash
python SSA-MRN/scripts/verify_recurrent_guidance_results.py --check-torch
```

공개 실행기는 파일명을 통일하고 평가 import를 `evaluate_paper`로 바꿨습니다. 학습된 결과의 원본 코드·이름·해시는 snapshot에 그대로 보존했으며 서버 실행 코드는 변경하지 않았습니다.

기존 실험 폴더·데이터·공용 환경을 삭제하거나 학습을 재시작하지 않았습니다.

## 8. 개선 여부와 다음 단계

**판정: 부분 개선. D3를 후속 검증 후보로 유지합니다.** D0·D1 대비 주요 RR·잠정 FR 지표가 좋아졌으나 경계·LR 관측 오차까지 일관되게 줄지 않았고 단일 seed43·50에포크 결과입니다. 목표 +0.3–0.5 dB에는 미달했습니다.

D1은 이전 seed42의 PSNR 이득을 재현하지 못했습니다. D3의 반복 seed·다른 센서·더 긴 학습, FR 지표 parity가 남아 있으며 D4·D6의 test 개선도 미확정입니다. 최종 모델 채택과 구조·손실 결합의 효과는 아직 확정하지 않습니다. 이번 전달은 완료 결과 보관이며 다음 학습은 별도 요청에 따라 진행합니다.
