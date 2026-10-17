# MS 기준 평균 제거와 대비 정규화의 기여 분리

**연구 분야:** 기존 성능 개선 / **실험자:** 김경찬(실행 예정)

**변경 내용:** 샘플별 공유 가역 정규화의 평균 제거와 스케일 조절을 2×2 비교로 분리합니다.

| 확인된 결과 | 상태 | 개선 여부 |
|---|:---:|:---:|
| 실행 코드와 로컬 구문·수식·제어 검사를 준비했습니다. 실제 학습·GPU 검사·RR·FR 평가는 미실행입니다. | 계획 | 판단 보류 |

## 1. 목적과 상태

33·38번은 전역 가역 정규화의 세 시드 RR 이득을 확인했으나 seed44 FR은 악화했습니다. 열린 PR98의 39번 국소 정규화는 전역 방식보다 세 시드 평균 RR·잠정 FR이 악화했습니다. 이번에는 평균 제거와 대비 조절을 분리해 더 단순한 방식으로 이득을 유지할 수 있는지 확인합니다. FR 불안정의 원인을 이미 입증했다고 쓰지 않습니다.

실험 번호는 performance 40으로 준비했습니다. 준비 시점 main과 열린 PR98·99에서 중복을 찾지 못했으며 원격 등록·커밋·푸시는 수행하지 않았습니다. 실제 등록 전 번호 사용 여부를 다시 확인합니다. 로컬 검사는 GPU 사전 검증 완료를 의미하지 않습니다.

## 2. 변경한 구조와 비교 조건

F는 33번과 같은 원본 RestoredPansharpeningNet K4입니다. DN/2047 단위의 LR MS 전체 C,H,W에서 샘플별 모집단 평균 μ와 표준편차 σ를 계산하며 s=max(σ,0.01)입니다.

| 군 | 세 입력의 공통 변환 | 출력 역변환 |
|---|---|---|
| A0 원본 | x | 없음 |
| A1 평균 제거 | x−μ | y+μ |
| A2 대비 조절 | x/s | sy |
| A3 전체 정규화 | (x−μ)/s | sy+μ |

PAN·LMS·MS에 같은 스칼라를 적용합니다. GT와 다른 배치 샘플은 통계에 쓰지 않습니다. 추가 학습 파라미터는 0개입니다. 원본 초기 LRPAN·BatchNorm·밴드별 SSAI·다중 해상도 보간과 복원을 유지합니다. 공통 모델·upstream·지표 소스는 38번 학습 원본에서 바이트 그대로 보존했습니다.

| 항목 | 고정 조건 |
|---|---|
| 데이터 | QuickBird train17,139·validation1,905·RR20·FR20 |
| 내부 특징 차원·공간 배율 | K4·4배 |
| 학습 | fresh seed42, 각100 epoch, Adam 고정lr1e-4 |
| 배치 | effective64·micro32, 마지막 불완전 배치도 사용 |
| 손실 | 역변환한 최종 출력의 원래 DN/2047 공간 MSE |
| 정밀도 | 학습FP16 AMP, 통계·변환·역변환·손실·평가FP32, TF32 off |
| 선택 | epoch0 포함 validation MSE 최소, 정확한 동률이면 이른 epoch |
| 실행 | T4×2, 첫 묶음 A0/A3, 다음 묶음 A1/A2 |

각 worker는 자기 GPU에 train/val을 상주시킵니다. 같은 시드의 초기 상태 해시와 실제 epoch별 배치 순서를 비교합니다. 초기 파라미터는 같지만 초기 출력은 다릅니다. BatchNorm 때문에 micro batch도 고정합니다. 입력과 평가 예측을 clipping하지 않습니다.

## 3. 정량 결과

아직 없습니다. 실행 후 각 군의 results/<군>/history.json과 results/<군>/results/metrics.json에 실제 값을 기록합니다. RR PSNR·SAM·ERGAS·SCC·Q2n, FR QNR·D_lambda·D_s를 전체20장 평균과 장면별로 보존합니다. QNR·Q2n·SCC는 공식 MATLAB 수치 대응 미검증입니다. FR에는 고해상도 GT가 없습니다.

best는 validation으로만 선택합니다. latest는 재개용이며 test에서 best와 비교해 유리한 가중치를 다시 고르지 않습니다.

## 4. 기준선 대비 차이와 해석

A1−A0·A2−A0·A3−A0와 A1−A3·A2−A3를 함께 보고합니다. A3−A1−A2+A0의 상호작용 차이도 탐색적으로 기록하며 통계적 유의성이나 엄밀한 인과 증거로 해석하지 않습니다.

seed42 네 군의 학습이 끝나면, A1 또는 A2의 best validation MSE가 A3의 1.005배 이내일 경우 네 군 모두 seed43·44를 반복하는 계획을 유지합니다. 둘 다 기준을 벗어나면 반복 확대를 중단합니다. 이 판단은 test 전에 validation_decision.json으로 저장하며 추가 시드는 자동 실행하지 않습니다.

A0 대비 RR 유지 목표는 ΔPSNR≥−0.03dB·ΔSAM≤+0.02°입니다. 잠정 FR 목표는 ΔQNR≥+0.003·Dλ와 Ds 비악화입니다. 축약군을 A3의 대체 후보로 검토할 때도 A3 대비 같은 목표를 적용합니다. 기준 통과는 유의성이나 자동 채택을 뜻하지 않습니다. 실제 RR 하락이 있으면 RR·FR 동시 개선이라고 쓰지 않습니다. 시드별 결과가 엇갈리면 부분 개선·채택 보류로 기록합니다.

입력 통계 분포·scale 하한 활성·출력의 gain/offset 대응을 진단합니다. A1은 offset 대응, A2는 양의 gain 대응, A3는 둘의 결합이 예상되지만 scale 하한이 작동하면 성질이 제한됩니다. 이 진단은 영상 품질 점수가 아닙니다. 기존 test를 반복 관찰한 탐색이며 독립 holdout·다른 센서 일반화는 검증하지 않습니다.

## 5. 학습·평가 그래프

현재 실제 곡선은 없습니다. 학습 후 curves.png에 네 군의 train MSE·validation MSE·PSNR·SAM을 표시합니다. 평가 후 comparison.png·summary.json·paired_scene_deltas.json을 생성합니다. 없는 성능이나 곡선을 만들지 않습니다.

## 6. 결과 이미지

RR의 0기준 장면1·8·11·13·19를 이 순서의 전체 장면으로 학습 전에 고정합니다. 각 군에서 LR MS nearest 확대·PAN·예측·GT의 4패널을 저장합니다. RGB[2,1,0]과 동일 장면 GT의 1/99백분위 대비를 MS 패널에 공통 적용합니다. PAN은 별도 grayscale 대비입니다. 표시 범위와 평가 clipping은 구분하며 FR에는 GT를 만들지 않습니다.

## 7. 재현 자료

run.py 전체를 Kaggle 단일 셀로 붙여 넣습니다. 인터넷과 기존 가중치 다운로드가 필요하지 않습니다. source/에는 같은 내장 소스의 검토용 사본이 있습니다. 실행하면 provenance/source·run_cell.py·config·data_manifest에 실제 사용 소스·설정·데이터 해시를 남깁니다.

각 군 checkpoints/<군>_QB_s<seed>/initial.pt·best.pt·latest.pt를 보존합니다. latest에는 model·optimizer·scaler·Python/NumPy/Torch CPU/CUDA RNG·배치 generator·현재 순서·다음 batch·누적 loss·history·best_model을 저장합니다. 유효 batch 경계에서 약60초마다, 그리고 epoch validation 후 원자적으로 저장합니다. best는 실제 validation 최적 모델을 유지합니다.

manifest·ZIP CRC·내부 SHA256을 확인합니다. 로컬 ZIP 생성은 원격 보관 검증이 아닙니다. 실행·재개·Output 보관은 실행안내.md를 따릅니다. 실제 로컬 검사 범위는 validation/local_verification.json에 기록합니다. 현재 환경에는 PyTorch·GPU·원본 H5가 없어 실제 forward/backward·GPU 메모리·AMP 재개는 미검증입니다. Kaggle에서 각 군을 사전 검사한 뒤 학습합니다.

## 8. 개선 여부와 다음 단계

현재 판단 보류입니다. seed42 validation으로 반복 여부를 결정한 뒤 설정을 고정한 상태로 best의 RR·FR을 평가합니다. 반복 시 시드별 평균·paired 차이·표본 SD를 보고하며 20장면을 20회 학습으로 해석하지 않습니다. 유망하면 독립 자료·추가 센서·지표 수치 정합을 확인합니다. 최초성이나 이미 입증된 논문 기여를 주장하지 않습니다.
