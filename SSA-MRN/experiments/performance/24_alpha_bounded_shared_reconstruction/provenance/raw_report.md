# 3단계 공유 복원의 보정 계수 범위 제한

**변경 내용:** 공유 scalar α의 표현만 sigmoid(a)로 바꿨습니다.

| 확인된 결과 | 상태 | 개선 여부 |
|---|---|---|
| RR ΔPSNR -0.084861 dB, ΔSAM +0.010119°; 잠정 FR ΔQNR +0.015907 | 평가 완료 | 부분 개선 |

## 1. 목적과 상태

두 모델을 처음부터 각각100에폭 학습하고 validation best로 RR20장·FR20장 전체를 평가했습니다. 원격 보관·복구는 아직 검증하지 않았습니다.

## 2. 변경한 구조와 비교 조건

QB, K4, seed42, 3단계 공유 core·error encoder·scalar α, error=0이며 encoder bias는 학습합니다. 기준 α는 제한 없는0.1, 후보는 a=log(0.1/0.9)에서 sigmoid(a)를 사용합니다.
최종 출력 MSE, 고정 lr1e-4, effective=micro32, 마지막 불완전 batch 포함, 동일 fused Adam·AMP FP16·channels-last·결정론 설정을 적용했습니다. validation과 test는 FP32이고 출력을 clipping하지 않았습니다.
공통 초기 core·encoder·유효 α·CPU/GPU 초기 출력 및100개 epoch 순서를 확인했습니다. [사전 고정 protocol](provenance/protocol.json)과 [초기 검사](provenance/initialization_pair.json)에 근거를 보존합니다.

## 3. 정량 결과

각 split20장 장면별 지표의 산술평균입니다. RR MSE는 peak1, PSNR은 peak2047 밴드별 평균, SAM은 도 단위입니다. FR에는 GT가 없으며 지표는 잠정 구현입니다.

| 모델 | Best epoch | Validation MSE | Best α | RR PSNR | RR SAM | RR MSE | FR Dλ | FR Ds | FR QNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | 94 | 0.000164554876 | 0.114275217 | 37.671097 | 4.838250 | 0.000202637929 | 0.052136 | 0.081345 | 0.870462 |
| B1_alpha_bounded | 96 | 0.000164345678 | 0.108804181 | 37.586236 | 4.848369 | 0.000206483841 | 0.046150 | 0.070524 | 0.886368 |

장면별 전체 수치는 [결과 JSON](results/results.json)에 보존합니다.

## 4. 대조군과의 수치 차이

현재 실행의 후보−B1 차이입니다. 과거 체크포인트를 현재 대조군으로 재사용하지 않았습니다.

| 지표 | 후보−B1 |
|---|---:|
| RR PSNR | -0.0848613841 |
| RR SAM | +0.0101190871 |
| RR MSE_peak1 | +3.84591242e-06 |
| FR D_lambda | -0.00598545394 |
| FR D_s | -0.0108203622 |
| FR QNR | +0.0159067906 |

[장면별 차이와 판정 기록](results/paired_comparison.json). 서로 반대 방향의 개선을 그대로 기록하며 test로 α 설정이나 epoch를 다시 선택하지 않습니다.

## 5. 실측 곡선

![학습·검증](results/learning.png)

![유효 α와 원시 파라미터](results/alpha.png)

![전체 RR/FR 비교](results/test_comparison.png)

## 6. 고정 RR 5장면

scene ID1,8,11,13,19를 학습 전에 고정했습니다. LR MS·PAN·예측·GT 순서입니다. 장면별 GT1-99% 대비를 MS 패널에 공통 적용하고 LR은 최근접 확대합니다. FR GT를 생성하지 않습니다.

![B1 RR scene 1](results/B1/RR_scene_01.png)

![B1 RR scene 8](results/B1/RR_scene_08.png)

![B1 RR scene 11](results/B1/RR_scene_11.png)

![B1 RR scene 13](results/B1/RR_scene_13.png)

![B1 RR scene 19](results/B1/RR_scene_19.png)

![B1_alpha_bounded RR scene 1](results/B1_alpha_bounded/RR_scene_01.png)

![B1_alpha_bounded RR scene 8](results/B1_alpha_bounded/RR_scene_08.png)

![B1_alpha_bounded RR scene 11](results/B1_alpha_bounded/RR_scene_11.png)

![B1_alpha_bounded RR scene 13](results/B1_alpha_bounded/RR_scene_13.png)

![B1_alpha_bounded RR scene 19](results/B1_alpha_bounded/RR_scene_19.png)

## 7. 가중치와 재현 근거

| 모델 | 원본 | Epoch | SHA-256 |
|---|---|---:|---|
| B1 | [best](checkpoints/B1/best.pt) | 94 | 909bded04a1e1e91b9142429a0381adad55cc4092ab796377f507173ebf61479 |
| B1 | [latest](checkpoints/B1/latest.pt) | 100 | 221f5d7e271862a5f226c93b333e0da488e7a303a728880d488228882cc8e6db |
| B1_alpha_bounded | [best](checkpoints/B1_alpha_bounded/best.pt) | 96 | 3bfe4a2acf3bca349ee0fc133f56c0ce87d93d0bc379354568641fe2766f1107 |
| B1_alpha_bounded | [latest](checkpoints/B1_alpha_bounded/latest.pt) | 100 | a23ed1f460f34474e3b29cc0e35d64d0c88d77924ba2dc1ddf007d0857fbed45 |

원본에는 model·optimizer·scaler·모든 RNG·배치 순서·커서·누적loss·history를 보존했습니다. resume.json은 best/latest 원본을 함께 가리킵니다. [완전성 검사](results/completion_checks.json)와 [manifest](manifest.json)에 파일 크기·해시를 기록합니다.
Saved Output 보존·다운로드 후 ZIP해시, manifest, 두 모델 resume record와 실제 재개를 확인해야 원격 보관 완료로 판단합니다. 삭제된 VM 파일을 이 코드로 복구할 수 없습니다.

## 8. 한계와 다음 판단

α 제한의 효과는 실험 전 가설이었습니다. sigmoid는 범위와 함께 유효 업데이트 크기도 바꿉니다. 초기 dα/da=0.09이므로 관측 차이를 제한 없는 α의 과도한 증가만으로 설명하지 않습니다.
단일 QB seed42·고정 GPU 배정의 기술적 비교이며 RR 유지의 통계적 비열등성 검정은 아닙니다. 모든 주요 RR/FR 지표의 차이를 함께 읽어야 합니다. QNR/Ds의 MATLAB 대응, Q2n/SCC 대응은 미검증이며 FR 수치로 HR 정답 정확도를 보장하지 않습니다.
