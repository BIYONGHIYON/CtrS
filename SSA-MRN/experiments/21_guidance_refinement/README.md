# 특징 선택·복원 후 관측 보정·고주파 PAN 차이

**변경 내용:** 특징 전달 gate를 공간·밴드별로 확장하고, 관측 보정을 복원 후에 적용하며, PAN 특징 차이의 저주파를 제거했습니다.

| 확인된 결과 | 상태 | 개선 여부 |
|---|---|---|
| seed44·공통 배치96으로 E0·E1 본학습을 시작했습니다. 평가 결과는 아직 없습니다. | 학습 중 | 판단 보류 |

[실험 목록](../../README.md#experiments) · [수치](#results) · [그래프](#graphs) · [이미지](#images) · [재현 자료](#evidence)

## 1. 목적과 상태

이전 D3는 RR, D5는 관측 일관성, D6는 잠정 FR 공간 지표에서 강점을 보였습니다. 각 구조를 새 seed에서 재학습하고 세 가지 변경의 효과를 비교합니다. **2026-10-12 10:54:34 KST에 E0·E1의 독립 controller와 실제 배치 진행을 확인했습니다.** 8개 실제 데이터 AMP 1에포크 점검과 strict 가중치 복구를 통과했습니다.

## 2. 변경한 구조와 비교 조건

| ID | 변경 내용 | 직접 비교 |
|---|---|---|
| E0 | 원본 SSA-MRN 단일 복원 | 전체 기준선 |
| E1 | D3: 3단계 공유 복원과 특징 전달 | E0 |
| E2 | D4: MS 관측 오차 보정 | E0 |
| E3 | D5: 특징 전달과 관측 보정 | E1·E2 |
| E4 | D6: 특징 전달과 PAN 차이 | E1 |
| E5 | 현재·이전 특징으로 공간·밴드별 전달 gate를 계산했습니다. | E1 |
| E6 | E3의 관측 보정을 단계별 복원 출력 뒤로 이동했습니다. | E3 |
| E7 | E4의 PAN 차이에서 고정 5×5 box 필터 저주파를 제거했습니다. | E4 |

E5는 각 크기에서 현재·이전 4채널 특징을 결합하고 1×1 Conv로 4채널 gate를 계산합니다. 기존 밴드 gate에 logits를 더하고 tanh를 적용합니다. 추가 Conv는 0으로 초기화했습니다.

E6는 `복원 → α 잔차 갱신 → MS 관측 오차 계산 → α로 조절한 관측 보정` 순서입니다. 기존 error encoder를 재사용하며 `encoder(error)−encoder(0)`로 bias 중복을 제거했습니다. bilinear 관측·기존 α·밴드 gate는 유지했습니다. 따라서 효과는 보정 위치와 적용 방식 전체에 대한 비교입니다.

E7은 각 크기의 `PAN_feature−Conv(MS_feature)`에서 reflect padding의 5×5 평균을 뺀 뒤 기존 gate로 주입합니다. 상수 특징의 고주파가 0임을 확인했습니다. 이 필터는 센서 MTF를 대신하지 않습니다.

- QB 전체 train 17,139 / validation 1,905 패치, RR 20개 / FR 20개를 사용합니다.
- K4, seed44, 50에포크 새 학습, Adam lr 1e-4 고정, 최종 출력 MSE를 사용합니다.
- Windows RTX3060Ti 8GB·5600X·RAM15GB에서 최대 2개 병렬로 실행합니다.
- AMP FP16 학습·FP32 validation/test, TF32 비활성·cuDNN autotune을 적용합니다.
- 공통 micro=effective 배치는 **96**으로 정했습니다. 배치96의 동시 실행은 최대 약 6.25GB, 배치128은 약 7.87GB로 여유가 부족해 제외했습니다. 각 worker는 1개입니다.
- 순서는 E0→E1→E5→E3→E6→E4→E7→E2이며 빈 슬롯에 다음 작업이 들어갑니다.
- 전체 모델의 core 초기값과 새 경로가 0일 때 각 원형 출력의 일치를 확인합니다.

<a id="results"></a>

## 3. 정량 결과

학습·평가 결과가 아직 없습니다. validation MSE 최소, 정확한 동률이면 SAM, 먼저 선택한 epoch로 best를 결정합니다. 8개 모두 RR·FR을 평가하며 test로 후보나 epoch를 다시 선택하지 않습니다.

## 4. 기준선 대비 차이와 해석

E5−E1, E6−E3, E7−E4로 변경 효과를 비교하고 E0 대비 성능도 보고합니다. 이전 seed43과 절대 수치를 섞지 않고 각 실행의 대조군 대비 차이를 확인합니다. 경계 MSE를 필수로 보고하며 FR QNR·Q2n·SCC는 parity 미확인 잠정 지표입니다.

<a id="graphs"></a>

## 5. 학습·평가 그래프

완료 후 train·validation 곡선, gate 변화, E0 대비 RR·FR 비교 그래프를 저장합니다.

<a id="images"></a>

## 6. 결과 이미지

평가 전 RR index 1·8·11·13·19를 고정했습니다. LR MS · PAN · 예측 · GT 순서이며 LR MS만 최근접 확대하고 GT 1–99% 공통 대비·RGB [2,1,0] 가정을 적용합니다. FR에 GT를 만들지 않습니다.

<a id="evidence"></a>

## 7. 재현 자료

[공통 실행·평가 안내](../../README.md#evaluate) · [파일 해시](manifest.json) · [실행 코드](run.py) · [모델](model.py) · [준비 코드](prepare.py) · [원본 실행 근거](provenance/)

새 실행 옵션은 `python SSA-MRN/experiments/21_guidance_refinement/run.py --help`에서 확인합니다. 기존 서버 runtime을 새 코드로 재개하지 않습니다.

## 8. 개선 여부와 다음 단계

판정은 판단 보류입니다. 모든 실험의 validation best로 전체 RR·FR을 평가한 뒤 결과를 기록합니다.
