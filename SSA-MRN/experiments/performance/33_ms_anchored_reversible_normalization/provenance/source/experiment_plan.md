# MS 기준 가역 밝기·대비 정규화

**연구 분야:** 기존 성능 개선

**실험자:** 김경찬(계획 기준이며 실제 학습은 아직 실행하지 않았습니다.)

**변경 내용:** 원본 SSA-MRN K4의 앞뒤에 LR MS 통계로 정규화·역변환을 추가합니다. PAN·LMS·MS와 모든 밴드에 동일한 샘플별 스칼라 변환을 적용합니다.

| 확인된 결과 | 상태 | 개선 여부 |
|---|:---:|:---:|
| CPU 합성 데이터에서 원본 동등성, 유한 forward/backward, 정규화 복원과 체크포인트 재개를 확인했습니다. 실제 학습·RR·FR은 미실행입니다. | 계획 | 판단 보류 |

## 1. 목적과 상태

RR 개선과 FR 악화가 반복된 선행 결과를 바탕으로 입력의 밝기·대비 변화에 대한 민감도를 줄이는 가설을 검증합니다. 분포 차이가 기존 FR 악화의 원인이라고 확정하지 않습니다. CPU 검사는 성능 평가가 아닙니다. CUDA·실제 H5는 Kaggle에서 자동 사전 검사합니다.

## 2. 변경한 구조와 비교 조건

N0는 원본 RestoredPansharpeningNet의 forward(pan,lms,ms)입니다. N1은 LR MS의 C,H,W 전체에서 샘플별 평균 μ와 모집단 표준편차 s를 구합니다. s의 하한은 DN/2047 공간에서 0.01입니다. PAN·LMS·MS 모두 (x-μ)/s로 변환하고 원본 네트워크를 호출한 뒤 출력에 s를 곱하고 μ를 더합니다. 통계 계산과 입력 변환·역변환은 FP32입니다. 손실은 원래 DN/2047 단위로 복원한 최종 출력의 MSE입니다. GT는 입력 통계에 사용하지 않습니다.

원본 LRPAN 경로, 네 차례 잔차 블록의 BatchNorm, SSAI와 보간을 유지합니다. 두 모델의 파라미터 수는 각각 426,654개입니다. 유효 배치만 같더라도 BatchNorm은 micro batch에 영향을 받으므로 두 군 모두 micro32로 고정합니다. 같은 초기 파라미터가 같은 초기 출력을 의미하지는 않습니다.

| 항목 | 사전 고정 조건 |
|---|---|
| 데이터 | QB train17,139·validation1,905·RR20·FR20, 기존 분할 |
| K·공간비 | K4·4배 |
| 초기화 | seed42, 두 군 같은 초기 모델, 과거 가중치 사용 안 함 |
| 학습 | 100에포크, Adam lr1e-4 고정, effective64·micro32 |
| 정밀도 | 학습 FP16 AMP, 통계·역변환·손실 FP32, validation/test FP32, TF32 비활성 |
| 선택 | epoch0 포함 validation MSE 최소, 정확한 동률은 최초 시점 |
| 장치 | GPU0 N0·GPU1 N1, 독립 process, worker별 train/val GPU상주 |
| 평가 | 전체 장면 추론, 원래 DN 단위, 입력·예측 clipping 없이 기존 지표 |
| 예시 | RR 인덱스1·8·11·13·19, 전체 장면, LR MS·PAN·예측·GT |

에포크별 실제 순서 해시와 초기 상태 해시를 비교합니다. AMP 업데이트 누락은 기록하고 비교 완료 판정을 보류합니다. micro batch를 임의로 줄이거나 CPU fallback을 적용하지 않습니다.

## 3. 정량 결과

실제 validation 및 RR·FR 지표는 없습니다. 실행 후 각 군의 results/<군>/history.json과 results/<군>/results/metrics.json에 기록합니다. CPU 합성 손실은 디버그 값이며 연구 성능 표에 넣지 않습니다. RR PSNR·SAM·ERGAS·SCC·Q2n, FR QNR·Dλ·Ds를 전체20장 평균과 장면별로 보존합니다. Dλ·Ds는 낮을수록 좋습니다. QNR·Q2n·SCC의 공식 MATLAB 정합은 미검증입니다.

## 4. 기준선 대비 차이와 해석

28번·31번 G5 미세조정 결과와 직접 비교하지 않습니다. 같은 실행에서 새로 학습한 N0만 직접 대조군입니다. 저주파를 지우거나 보정기를 추가하지 않습니다. UniPAN은 비선형 분포 변환을 연구했으며 이번 공유 affine 가역 변환은 논문의 완전한 재현이 아닙니다. RevIN은 시계열 논문이며 팬샤프닝 개선을 보장하지 않습니다. 세계 최초라고 주장하지 않습니다.

주요 한계는 학습 패치와 전체 추론 장면의 통계 범위 차이입니다. 공유 affine 변환은 모든 센서·밴드별 응답 차이와 정합·공간 열화 차이를 해결하지 않습니다. 표준편차 하한이 활성화되면 엄밀한 affine equivariance는 깨집니다. CUDA 결정성 미지원 연산은 경고를 허용하므로 비트 재현성을 보장하지 않습니다.

## 5. 학습·평가 그래프

현재 실제 곡선은 없습니다. 실행 후 train MSE·validation MSE·PSNR·SAM 및 RR·FR 8지표 비교 그래프를 생성합니다. 없는 곡선을 추정해 만들지 않습니다.

## 6. 결과 이미지

현재 예측 이미지는 없습니다. 사전 고정한 RR 다섯 장면을 각각 LR MS·PAN·예측·GT로 저장합니다. RGB 밴드[2,1,0]이며 GT의1·99백분위 대비를 모든 MS 패널에 공통 적용합니다. LR MS만 최근접 확대로 표시합니다. FR에는 GT를 만들지 않습니다.

## 7. 재현 자료

단일 셀 run.py에는 원본 모델·기존 지표·정규화 모델·worker·controller 소스가 포함됩니다. 외부 다운로드와 부모 체크포인트가 필요하지 않습니다. 실행하면 provenance/source에 실제 사용 소스, config.json에 설정, data_manifest.json에 원본 H5의 해시·shape·범위를 남깁니다. 초기 모델과 best/latest 원본, optimizer·scaler·RNG·GPU배치 생성기·순서·다음 배치 위치·실제 이력을 보존합니다.

local_checks.py와 local_verification.json은 CPU 합성 데이터 검사 자료입니다. Kaggle 사전 검사는 실제 GPU 캐시·AMP backward·두 업데이트 중단재개를 추가 확인하며 모델 상태·optimizer·scaler·RNG를 원복한 뒤 본 학습을 시작합니다. 이 짧은 검사와 장시간 전체 세션 중단재개 검증은 구분합니다.

관련 문헌:
- UniPAN: https://arxiv.org/abs/2510.22217
- RevIN 공식 구현: https://github.com/ts-kim/RevIN
- PAN-Crafter: https://kaist-viclab.github.io/PAN-Crafter_site/
- 기준 저장소: https://github.com/BIYONGHIYON/CtrS

## 8. 개선 여부와 다음 단계

현재 판단 보류입니다. RR 유지 목표는 ΔPSNR≥−0.03dB·ΔSAM≤+0.02°이며 FR 목표는 잠정 ΔQNR≥+0.003·Dλ와Ds 비악화입니다. 이 임계값은 실용적인 screening 목표이며 통계적 유의성을 의미하지 않습니다. 전체 개선은 반복seed의 주요 RR·FR 동시 개선을 확인해야 합니다.

seed42 이후 추가 실험 여부를 validation으로 결정하려면 EVALUATE_AFTER_TRAINING=False로 시작합니다. 학습이 끝난 동일 실행을 RESUME_FROM으로 연결하고 이 값만 True로 바꾸면 추가 학습 없이 best 평가를 수행합니다. test를 본 뒤 설정·학습량·best를 변경하지 않습니다. 다음 seed는 새 RUN_ID로 시작합니다. 기존 benchmark는 이미 반복 관찰했으므로 새 독립 holdout이라고 주장하지 않습니다. 코드·결과의 커밋·푸시는 별도 요청 시 진행합니다.
