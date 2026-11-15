# Kaggle SSA-MRN B0/B0_DS

[수치](#results) · [그래프](#graphs) · [이미지](#images) · [가중치](#evidence)

| 비교 기준 | 바꾼 점 | 관측 차이 | 판단 |
|---|---|---|---|
| same-session B0 k4 | GT-only auxiliary supervision | see measured table | no guaranteed improvement |

## 1. 목적과 상태
Status: full training and RR/FR evaluation completed.
Learning rate: fixed 0.0001 for 100 epochs, then cosine to 1e-06.
Fresh paired experiment; earlier B0/B0_DS scores are context, not a same-session baseline.
## 2. 변경 사항과 평가 조건
QB; K=4; epochs=200; seeds=[42]; effective batch=32; micro=32.
B0 is retrained in this suite with the original core and zero auxiliary weights.
B0: original single core + final MSE. B0_DS: same core/heads + small quarter/half/full GT L1 supervision.
Same core/head initialization, epoch sample order and final MSE; only auxiliary loss weights differ.
GT-only binomial targets with explicit endpoint coordinates; observed LRMS is not wavelet LL. Backbone resampling stays unchanged.
Inference is FP32 on whole scenes. Best selected only by validation MSE. Auxiliary targets are synthetic, not a calibrated sensor model.
FR has no HR GT; no auxiliary GT target is used in FR evaluation. Repository QNR/Q2n/SCC MATLAB parity limitations remain.
GPU-resident FP32 data, GPU sampling/batching, periodic status synchronization; no CPU DataLoader.
Mid-epoch recovery stores the permutation, batch cursor, optimizer/scaler/RNG and loss accumulator.
Best checkpoint uses validation MSE only. best_first100.pt preserves the actual first-100 validation winner.
Fused Adam/AMP/layout/micro batch are matched; historic B1/B2/k6 are not bitwise reproduced.
<a id="results"></a>
## 3. 정량 결과
| Run | Best epoch | RR PSNR | RR SAM | FR QNR (provisional) |
|---|---:|---:|---:|---:|
| B0_QB_B0_k4_s42 | 199 | 37.629230 | 4.834539 | 0.903297 |
| B0_DS_QB_B0_DS_k4_s42 | 199 | 37.586716 | 4.860947 | 0.913427 |

## 4. 직전 연구와 수치 차이
Same-session paired deltas: summary.csv and paired_comparison.json. Historical k6/B2 architectures are not directly comparable.
<a id="graphs"></a>
## 5. 그래프

![Learning](learning_seed42.png)
![Test comparison](test_comparison_seed42.png)
![Auxiliary objectives](auxiliary_seed42.png)
<a id="images"></a>
## 6. 결과 이미지 예시
Five RR full scenes fixed in selection.json before training/test scoring.

## B0_QB_B0_k4_s42 fixed RR scenes
![Scene 1](runs/B0_QB_B0_k4_s42/RR_scene_01.png)
![Scene 8](runs/B0_QB_B0_k4_s42/RR_scene_08.png)
![Scene 11](runs/B0_QB_B0_k4_s42/RR_scene_11.png)
![Scene 13](runs/B0_QB_B0_k4_s42/RR_scene_13.png)
![Scene 19](runs/B0_QB_B0_k4_s42/RR_scene_19.png)

## B0_DS_QB_B0_DS_k4_s42 fixed RR scenes
![Scene 1](runs/B0_DS_QB_B0_DS_k4_s42/RR_scene_01.png)
![Scene 8](runs/B0_DS_QB_B0_DS_k4_s42/RR_scene_08.png)
![Scene 11](runs/B0_DS_QB_B0_DS_k4_s42/RR_scene_11.png)
![Scene 13](runs/B0_DS_QB_B0_DS_k4_s42/RR_scene_13.png)
![Scene 19](runs/B0_DS_QB_B0_DS_k4_s42/RR_scene_19.png)
<a id="evidence"></a>
## 7. 가중치와 검증 근거
Original best/latest epoch, path and SHA256 in each run/results.json; all retained file hashes in artifact_manifest.json.
Checkpoint originals are Saved Output artifacts; no repository commit is implied.
## 8. 한계와 다음 판단
QB only, one seed, provisional FR/Q2n metrics. Auxiliary loss effectiveness is measured, not guaranteed; no sensor-MTF claim.
