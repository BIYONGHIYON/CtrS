# QB E0/E1 K4 controlled experiment

| Reference | Change | Observed difference | Decision |
|---|---|---|---|
| Fresh E0 | E1 backbone channel attention | RR PSNR +0.480302 dB; provisional FR QNR -0.022632 | Single seed; confirm with additional paired seeds |

## 1. Purpose and status

Both models completed 100 epochs and full RR20/FR20 evaluation. Actual optimizer/skipped update counts are in update_counts.json. Remote archival has not been verified.

## 2. Changes and conditions

QB; K4; seed42; batch=micro32; fused Adam; fixed lr1e-4; final output MSE; AMP FP16; channels-last; strict deterministic; TF32 disabled.
E0 is the original K4 adapter. E1 adds CA to the residual branch of its four 64-channel backbone blocks: GAP, 64-to-4-to-64, 2*sigmoid; last projection zero initialized. No change to SSAI, interpolation, loss or recurrence.
Identical initial core hashes and all 100 epoch order hashes were checked. See protocol.json and initialization_pair.json.
Best selected by FP32 validation MSE; test was not used for epoch/configuration selection.

## 3. Quantitative results

| Model | Best epoch | Validation MSE | RR PSNR | RR SAM | RR MSE peak1 | FR QNR provisional |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 97 | 0.000172096363 | 37.371084 | 4.946089 | 0.000218160682 | 0.909722 |
| E1 | 98 | 0.000155570917 | 37.851386 | 4.809228 | 0.000193513442 | 0.887091 |

Scene rows and all metrics: [results](results.json), [paired differences](paired_comparison.json).

## 4. Previous research differences

Historical interpolation, recurrence, output-detail and auxiliary-loss experiments differ in recipe. This comparison estimates the CA-module effect, including 2320 extra parameters; it does not isolate parameter count.
Current E1-minus-E0 values are in paired_comparison.json. They are a single-seed observation.

## 5. Measured curves

![Measured learning](learning.png)

![Test comparison](test_comparison.png)

## 6. Fixed RR panels

Selection was fixed before training: [selection.json](selection.json). MS panels share GT 1-99% contrast; LR uses nearest-neighbor enlargement. FR has no GT.

![E0 scene 1](E0/RR_scene_01.png)

![E0 scene 8](E0/RR_scene_08.png)

![E0 scene 11](E0/RR_scene_11.png)

![E0 scene 13](E0/RR_scene_13.png)

![E0 scene 19](E0/RR_scene_19.png)

![E1 scene 1](E1/RR_scene_01.png)

![E1 scene 8](E1/RR_scene_08.png)

![E1 scene 11](E1/RR_scene_11.png)

![E1 scene 13](E1/RR_scene_13.png)

![E1 scene 19](E1/RR_scene_19.png)

## 7. Weights and evidence

| Model | Weight | Epoch | SHA-256 |
|---|---|---:|---|
| E0 | [best](E0/best.pt) | 97 | f589a60f7291ac3ca42355a59113c94e8bcad124c9795674cec3ed877494aa8c |
| E0 | [latest](E0/latest.pt) | 100 | 466881fe7565ab33cc5d965e358e42d77d2ffc25cff0b898f2286fc77907360f |
| E1 | [best](E1/best.pt) | 98 | 2b1e73b8a28365b4f9b5b96b50654cf29df01573ff6a70875bc8e292d8e334d0 |
| E1 | [latest](E1/latest.pt) | 100 | a16e53cd3eefe6702f6f1b6ad1299517bba37cab622718dcb87e71a3a7f5053d |

Original best/latest include optimizer, scaler, RNG, batch cursor and measured history. artifact_manifest.json records artifact bytes/hashes; ZIP CRC and archive SHA-256 are verified locally. Preserve Saved Output and verify remote re-download/recovery before declaring archival complete.

## 8. Limitations and next decision

FR has no high-resolution GT. QNR/D_s use scikit-image bicubic rather than MATLAB-bit-exact resizing; Q2n/SCC parity also remains limited. Improvements are not guaranteed. Seed43 was not launched; decide after reviewing RR/FR differences and measured cost.
