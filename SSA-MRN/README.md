# SSA-MRN 연구·실행 안내

[연구 소개와 구조도](../README.md) · [알고리즘·지표 설명](docs/research.md)

## 연구 분담

연구 분야를 선택하면 해당 분야의 목적과 실험 표를 바로 확인할 수 있습니다. SSAI 개선은 QB 단일 시드의 세 가설 평가를 마쳤습니다. 가변 배율 확장은2·3·4배 학습과 RR 평가를 마쳤으며 FR·장면 출처 검증이 남았습니다.

| 연구 | 담당 | 목표 | 현재 단계 |
|---|---|---|---|
| [SSAI 개선](research/ssai/README.md) | 이병현 | 밴드별 공간·분광 특징을 결합하는 SSAI의 연산과 구조를 개선합니다. | SSAI 1-3 평가 완료 |
| [가변 배율 확장](research/variable_scale/README.md) | 김윤지 | 기존 고정 4배 복원을 여러 배율에서 동작하도록 확장합니다. | 학습 완료·일부 평가 |
| [기존 성능 개선](research/performance/README.md) | 김경찬·송준현 | 기존 연구의 구조·학습·손실 변경을 이어서 비교합니다. | 기존 결과 보유 · 후속 연구 계속 |

<a id="decision"></a>

## 연구 기준

원본 SSA-MRN의 K=4·MSE 손실을 기준으로 개선 구조를 비교합니다. 실제 K·센서·학습 조건은 각 보고서에 기록합니다. 과거 결과의 설정은 그대로 유지합니다.

<a id="experiments"></a>

## 실험 보고서

실험은 연구별로 구분하며 각 표 안에서는 오래된 순서부터 최근 순서로 추가합니다.

<a id="ssai-experiments"></a>

### SSAI 개선

[연구 페이지](research/ssai/README.md) · 담당: **이병현**

<details>
<summary>SSAI 개선 실험 표 펼치기</summary>

| 실험 | 변경한 내용 | 확인된 결과 | 상태 | 개선 여부 |
|---|---|---|:---:|:---:|
| [1. 공간 좌표 정렬](experiments/ssai/1_spatial_alignment/README.md)<br>[이병현] | 원본의 두 공간 전치를 제거해 같은 위치의 PAN·MS 특징을 결합했습니다. | PSNR +0.0545dB, SAM -0.0351°, 잠정QNR +0.0077 | 평가&nbsp;완료 | 개선 |
| [2. 기준 면적 정규화](experiments/ssai/2_reference_area_normalization/README.md)<br>[이병현] | 원본 SSAI 출력에 현재 면적/학습 기준 면적을 곱했습니다. | PSNR -0.0002dB, SAM +0.0009°, 잠정QNR -0.0000 | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [3. 기준 크기 sigmoid 게이트](experiments/ssai/3_calibrated_sigmoid_gating/README.md)<br>[이병현] | 원본 전치를 유지하고 공간 softmax를 기준 크기 sigmoid 게이트로 교체했습니다. | PSNR +0.0226dB, SAM -0.0250°, 잠정QNR +0.0090 | 평가&nbsp;완료 | 판단&nbsp;보류 |
| [4. 공간 좌표 정렬과 sigmoid 게이트 결합](experiments/ssai/4_aligned_sigmoid_gating/README.md)<br>[이병현] | 좌표 정렬과 기준 크기 sigmoid를 결합하고 네 후보를 동일 조건으로 비교합니다. | A3: PSNR -0.0158dB, SAM +0.0226°, 잠정 QNR +0.0099 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [5. softmax와 sigmoid 혼합 게이트](experiments/ssai/5_mixed_gating/README.md)<br>[이병현] | 원본 좌표에서 softmax와 기준 크기 sigmoid를 고정 비율로 혼합합니다. | M25 개선 없음; M50 PSNR -0.0710dB·잠정 QNR +0.0043 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [6. MS 조건부 공간 세부정보 보정](experiments/ssai/6_ms_conditioned_detail/README.md)<br>[이병현] | MS가 선택한 PAN 세부정보를 원본 출력에 잔차 보정합니다. | PSNR -0.0017dB·SAM +0.0038°·잠정QNR -0.0046 | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [7. MS–PAN 채널 관계 기반 세부정보 보정](experiments/ssai/7_channel_relation_detail/README.md)<br>[이병현] | 4×4 MS–PAN 채널 관계로 PAN 세부정보를 선택·보정합니다. | PSNR -0.0061dB·SAM +0.0158°·잠정QNR -0.0030 | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [8. 공간 가중치 평균 정규화](experiments/ssai/8_spatial_mean_normalization/README.md)<br>[이병현] | 원본 공간 softmax 평균을1로 맞추고 고정·학습 gamma를 비교합니다. | N1 개선 없음; N2 PSNR +0.0104dB·SAM +0.0036°·잠정QNR -0.0019; 두 후보 학습 가능화 성공 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [9. PAN 재구성 오차 기반 보정](experiments/ssai/9_pan_innovation_correction/README.md)<br>[이병현] | PAN 직접 고주파와 PAN 재구성 오차를 같은 크기의 SSAI에서 비교했습니다. | A1 PSNR -0.0314dB·A2 -0.0352dB, 잠정 QNR 모두 악화 | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [10. 국소 창 교차 어텐션 SSAI](experiments/ssai/10_local_window_cross_attention/README.md)<br>[이병현] | 전역 공간 softmax를 7×7 국소 창 어텐션으로 바꾸고 동일 파라미터 convolution 대조군과 비교합니다. | 같은 에폭 A0 대비 validation: L2 −0.81%(21에폭), C −1.15%(45에폭); test 미평가 | 학습&nbsp;중단<br>10-16&nbsp;01:20 | 판단&nbsp;보류 |
| [11. SSAI 경로 제거 분해와 PAN 기저 결합 상한·FR 대리 지표 검증](experiments/ssai/11_ssai_contribution_headroom/README.md)<br>[이병현] | 학습 없이 A0 best에서 SSA 출력·SSAI 단계·coarse 경로·backbone 잔차를 제거하고, PAN 기저 결합 oracle 상한과 validation FR 대리 지표를 검증했습니다. | SSA 출력 제거 시 RR·FR 동일; SSAI 단계 제거 PSNR −1.55dB(상수 보정); oracle null 초과 설명량 11.3%; 대리 Ds 순위 상관 −0.18 | 사전&nbsp;검증&nbsp;완료<br>10-16&nbsp;03:02 | 성능&nbsp;비교&nbsp;아님 |

</details>

<a id="variable_scale-experiments"></a>

### 가변 배율 확장

[연구 페이지](research/variable_scale/README.md) · 담당: **김윤지**

<details>
<summary>가변 배율 확장 실험 표 펼치기</summary>

| 실험 | 변경한 내용 | 확인된 결과 | 상태 | 개선 여부 |
|---|---|---|:---:|:---:|
| [1. 가변 배율 복원과 배율 조건](experiments/variable_scale/1_scale_conditioned_restoration/README.md)<br>[김윤지] | 입력 배율과 피라미드를 분리하고 전용·공동 학습 및 FiLM을 비교했습니다. | 공동 모델은 전용 대비PSNR0.40-0.98dB 하락; FiLM3·4배 소폭 개선 | 학습&nbsp;완료·일부&nbsp;평가 | 판단&nbsp;보류 |
| [2. 배율별 학습 노출량 확장](experiments/variable_scale/2_scale_exposure_extension/README.md)<br>[김윤지] | 공동 C/D를100에서200에포크로 연장 | 모든 배율의 RR 개선, 전용 PSNR 격차 유지 | 학습&nbsp;완료·일부&nbsp;평가 | 판단&nbsp;보류 |

</details>

<a id="performance-experiments"></a>

### 기존 성능 개선

[연구 페이지](research/performance/README.md) · 담당: **김경찬·송준현**

<details>
<summary>기존 성능 개선 실험 표 펼치기</summary>

| 실험 | 변경한 내용 | 확인된 결과 | 상태 | 개선 여부 |
|---|---|---|:---:|:---:|
| [1. 공개 코드 재현](research/performance/1_public_code_reproduction/README.md)<br>[이병현] | 공개 코드 설정을 재현해 초기 기준을 확보했습니다. | RR·FR 재현 결과 확보; 구조 개선 실험이 아님 | 평⁠가⁠ ⁠완⁠료 | 비⁠교⁠ ⁠기⁠준 |
| [2. 논문 설정 재현](research/performance/2_paper_setting_reproduction/README.md)<br>[김경찬] | 논문의 내부 특징 차원 설정으로 재현했습니다. | RR·FR 재현 결과 확보; 공개 코드 재현과 장치 조건이 다름 | 평⁠가⁠ ⁠완⁠료 | 비⁠교⁠ ⁠기⁠준 |
| [3. MS 관측 연산자 구현](research/performance/3_ms_observation/README.md)<br>[이병현] | 센서 MTF·샘플링 위상·경계 처리를 구성했습니다. | 관측 구현 대응 확인; 학습 성능 비교가 아님 | 사⁠전⁠ ⁠검⁠증⁠ ⁠완⁠료 | 성⁠능⁠ ⁠비⁠교⁠ ⁠아⁠님 |
| [4. 23탭 확대·저해상도 보정·고주파 경로](research/performance/4_interpolation_correction_detail/README.md)<br>[이병현] | 확대 연산, 출력 보정, PAN 고주파 경로를 개별 변경했습니다. | 23탭: PSNR +0.0557 dB, SAM −0.0429°, 잠정 QNR +0.0031; 다른 후보 이득 혼재 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [5. 23탭 확대 위치와 센서 확장](research/performance/5_interpolation_scope_sensors/README.md)<br>[이병현] | 23탭 적용 위치를 분리하고 시드·센서를 확장했습니다. | RR 이득이 있으나 FR·센서별 결과가 엇갈림 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [6. 23탭 확대의 반복 시드 비교](research/performance/6_interpolation_seed_repeat/README.md)<br>[이병현] | 전체 확대 및 입력 확대 후보를 추가 시드로 반복했습니다. | 전체 확대의 GF2 PSNR·QNR 개선 반복; SAM 및 QB QNR 혼재 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [7. 밴드별 게이트 고주파 주입](research/performance/7_band_gated_detail/README.md)<br>[김윤지] | 밴드별 게이트로 PAN 고주파 주입량을 조절했습니다. | RR 악화로 후보 채택 보류 | 평⁠가⁠ ⁠완⁠료 | 개⁠선⁠ ⁠없⁠음 |
| [8. MTF 변화량 데이터 증강](research/performance/8_mtf_augmentation/README.md)<br>[이병현] | 센서 열화 특성의 변화량을 학습 데이터에 추가했습니다. | MTF: PSNR -0.0064 dB, SAM -0.0061°; 잠정 QNR -0.0014 | 평가⁠ ⁠완료 | 부분⁠ ⁠개선 |
| [9. SSAI 내부 융합 연산 변경](research/performance/9_ssai_fusion/README.md)<br>[이병현] | SSAI 핵심 융합 연산을 네 구조로 비교했습니다. | A1: SAM -0.0058°, 잠정 QNR +0.0127; 모든 후보의 PSNR은 하락 | 평가⁠ ⁠완료 | 부분⁠ ⁠개선 |
| [10. 반복 복원에 MS 관측 오차 전달](research/performance/10_recurrent_observation/README.md)<br>[김윤지] | 반복 단계에 입력 MS와 관측한 출력의 차이를 전달했습니다. | PSNR +0.1158 dB, SAM −0.0183°, 잠정 QNR −0.0182 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [11. Haar 대역별 단계 복원과 게이트](research/performance/11_haar_reconstruction/README.md)<br>[김윤지] | Haar 대역별 복원에 밴드·방향·위치 게이트를 적용했습니다. | 원본 대비 RR 큰 폭 악화; 학습 게이트의 추가 이득도 없음 | 평⁠가⁠ ⁠완⁠료 | 개⁠선⁠ ⁠없⁠음 |
| [12. SSAI 내부 특징 차원 비교](research/performance/12_feature_dimension/README.md)<br>[이병현] | 같은 환경에서 내부 특징 차원 4와 6을 비교했습니다. | K4: 세 센서 RR PSNR·SAM 우세; WV3 잠정 QNR은 K6 우세 | 평가⁠ ⁠완료 | 비교⁠ ⁠기준 |
| [13. 반복 복원·관측 오차 전달의 재개 학습](research/performance/13_recurrent_observation_resume/README.md)<br>[김윤지] | 반복 복원과 관측 오차 후보를 재개해 기준선과 비교했습니다. | 기준선 대비 RR 개선·잠정 QNR 하락; 별도 세션 비교 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [14. 분광·관측 일치·경계 손실 추가](research/performance/14_auxiliary_losses/README.md)<br>[이병현] | 기본 MSE에 보조 손실을 하나씩 추가했습니다. | edge 0.1: PSNR −0.0661 dB, SAM −0.0119°, 잠정 QNR +0.0098 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [15. 단일 복원과 3단계 공유 복원](research/performance/15_shared_reconstruction/README.md)<br>[김윤지] | 동일 초기값과 학습 조건에서 복원 횟수를 변경했습니다. | 반복 복원: PSNR +0.3000 dB, SAM −0.1078°, 잠정 QNR −0.0393 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [16. 반복 복원의 축소 전 저역통과 필터](research/performance/16_bandlimited_downsampling/README.md)<br>[이병현] | 축소 전에 고정 Gaussian 필터를 추가했습니다. | PSNR −0.0196 dB, SAM +0.0022°, 잠정 QNR −0.0686 | 평⁠가⁠ ⁠완⁠료 | 개⁠선⁠ ⁠없⁠음 |
| [17. SSAI 단계·잔차 복원·초기 입력 변경](research/performance/17_structural_ablation/README.md)<br>[이병현] | SSAI 단계, 복원 잔차, LRPAN 입력을 분리해 비교했습니다. | LRPAN 제거: PSNR +0.1297 dB, SAM −0.0704°, 잠정 QNR −0.0111 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [18. 작은 크기 특징 보존과 PAN 기여량 조절](research/performance/18_coarse_feature_fusion/README.md)<br>[이병현] | 작은 특징을 병렬 전달하고 PAN 기여량을 학습했습니다. | N3: PSNR +0.1875 dB, SAM −0.0134°, 잠정 QNR −0.0147 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [19. 반복 복원과 PAN·MS 분리 경로](research/performance/19_dual_stream_reconstruction/README.md)<br>[이병현] | 3단계 공유 복원과 분리 특징 경로를 비교했습니다. | C1: PSNR +0.2442 dB, SAM −0.1883°, 잠정 QNR +0.0250. 새 분리 경로 C5는 RR 악화 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [20. 반복 복원의 단계 간 특징 전달과 관측·PAN 차이](research/performance/20_recurrent_guidance/README.md)<br>[이병현] | 밴드별 SSAI를 유지하며 단계 간 정보 전달을 비교했습니다. | D3: PSNR +0.0956 dB; D4·D5 관측 오차 감소; D6 잠정 QNR +0.0166, RR 악화 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [21. 특징 선택·복원 후 관측 보정·고주파 PAN 차이](research/performance/21_guidance_refinement/README.md)<br>[이병현] | 공간·밴드별 특징 gate, 복원 후 관측 보정, 고주파 PAN 차이를 비교합니다. | E1: PSNR +0.3345 dB, SAM -0.2219°; E7: 잠정 QNR +0.0437, 지표별 이득 상이 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [22. 단계별 갱신량·특징 변환·마지막 단계 PAN 주입](research/performance/22_stage_aware_restoration/README.md)<br>[이병현] | 공유 복원기의 단계별 조절과 마지막 단계 PAN 차이를 비교했습니다. | F0 기준 F4: PSNR -0.0009 dB, SAM +0.0064°, 잠정 QNR +0.0018; 추가 이득 혼재 | 평⁠가⁠ ⁠완료 | 부⁠분⁠ ⁠개선 |
| [23. 반복 복원·단계 간 특징 전달의 센서별 수렴 비교](research/performance/23_multi_sensor_comparison/README.md)<br>[이병현] | 원본·3단계 공유 복원·특징 전달을 세 센서에서 150에포크 비교했습니다. | F0: QB PSNR +0.1271 dB, GF2 +0.8291 dB; WV3 잠정 QNR 하락. QB·GF2 RR은 repeat 우세 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [24. 3단계 공유 복원의 보정 계수 범위 제한](research/performance/24_alpha_bounded_shared_reconstruction/README.md)<br>[김윤지] | 공유 scalar α의 표현만 sigmoid(a)로 바꿨습니다. | PSNR −0.0849 dB, SAM +0.0101°; 잠정 QNR +0.0159, RR 유지 목표 미달 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [25. 중간 해상도 복원 목표와 보조 손실](research/performance/25_multiscale_supervision/README.md)<br>[김윤지] | 기존 복원 지점에 GT 기반 보조 L1을 추가했습니다. | 중간 head L1 감소; PSNR −0.0425 dB, SAM +0.0264°; 잠정 QNR +0.0101 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [26. 평가 가중치의 지수 이동 평균](research/performance/26_ema_weight_averaging/README.md)<br>[송준현] | 일반 MSE 학습을 유지하고 평가 가중치에 EMA를 적용했습니다. | seed42: PSNR +0.0893 dB, SAM −0.0244°; 잠정 QNR −0.0018 | 평가&nbsp;완료<br>seed42 | 부분&nbsp;개선 |
| [27. Backbone 채널 어텐션 추가](research/performance/27_backbone_channel_attention/README.md)<br>[김경찬] | Backbone 잔차 블록 4개에 채널 어텐션을 추가했습니다. | seed42: RR PSNR +0.4803 dB, SAM −0.1369°; 잠정 FR QNR −0.0226 | 평가&nbsp;완료<br>seed42 | 부분&nbsp;개선 |
| [28. 원본 복원 후 관측 오차 기반 잔차 보정](research/performance/28_residual_refinement/README.md)<br>[이병현] | 원본 복원 후 두 번의 잔차 보정·오차 입력·가중치 공유를 비교했습니다. | 기존 G2 대비 G5: PSNR +0.1317 dB, SAM -0.0391°; 잠정 QNR -0.0020. G4는 QNR +0.0079 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [29. 병렬 특징 용량 확장과 독립 잔차 보정](experiments/performance/29_capacity_refinement/README.md)<br>[이병현] | 원본 backbone에 저해상도·원해상도 가지를 추가하고 독립 보정기·LRPAN 제거 결합을 비교했습니다. | R0 대비 W2: RR PSNR +0.7234 dB, SAM -0.3542°; 잠정 QNR -0.0922. 후보 모두 RR 개선·FR 악화 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [30. Focal Frequency Loss 추가](experiments/performance/30_focal_frequency_loss/README.md)<br>[송준현] | 기존 MSE에 주파수 오차 가중 FFL을 추가했습니다. | seed42: PSNR -0.0553 dB, SAM -0.0073°; 잠정 QNR -0.0022. validation도 기준선 우세 | 평가&nbsp;완료<br>seed42 | 부분&nbsp;개선 |
| [31. 보정 잔차의 저주파 억제](experiments/performance/31_low_frequency_preserving_residual/README.md)<br>[김경찬] | G5 두 보정기의 잔차에서 고정 Gaussian 저주파 성분을 제거했습니다. | P0 대비 PSNR -0.9426 dB·SAM +0.5369°; 잠정 QNR +0.01671. RR 유지 실패·채택 보류 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [32. PAN 조건 특징 변조와 국소 회귀 사전 정보의 빠른 선별](experiments/performance/32_feature_modulation_local_regression_screen/README.md)<br>[이병현] | A0 best에서 PAN 조건 특징 변조(W1)·대조군(W1c)·국소 회귀 사전 정보(W2)를 10에폭 이어 학습해 선별했습니다. | 이어 학습 대조군 대비 best validation: W1 −0.06%, W1c −0.06%, W2 −0.03%; 기준 3% 미달, test 미평가 | 사전&nbsp;검증&nbsp;완료 | 판단&nbsp;보류 |

</details>

**판정 기준:** 개선 = 같은 조건에서 주요 RR·FR 지표가 함께 개선, 부분 개선 = 지표·후보·센서별 결과 혼재, 개선 없음 = 평가에서 향상을 확인하지 못함, 판단 보류 = 최종 평가 부족입니다. 비교 기준·성능 비교 아님은 개선 판정 대상에서 제외합니다. 모든 FR QNR은 구현 대응 검증이 끝나지 않은 잠정 지표이며, 단일 실험의 개선은 최종 채택을 뜻하지 않습니다.

<a id="files"></a>

## 폴더 구조

```text
SSA-MRN/
├── research/
│   ├── catalog.json          연구 분야·담당자
│   ├── experiments.json      전체 실험 목록·경로
│   ├── ssai/
│   │   ├── README.md         SSAI 개선 안내·실험 표
│   │   └── <번호>_<변경명>/   새 SSAI 실험
│   ├── variable_scale/
│   │   ├── README.md         가변 배율 안내·실험 표
│   │   └── <번호>_<변경명>/   새 가변 배율 실험
│   └── performance/
│       ├── README.md         성능 개선 안내·실험 표
│       ├── 1_public_code_reproduction/
│       └── ...               기존 1-28번 실험
├── common/                   여러 실험이 공유하는 도구
├── data/                     외부 학습·평가 데이터
├── docs/                     설명·보고서 양식·구조도
└── references/               공식 코드·원 논문
```

실험 폴더마다 `README.md`, 필요한 실행 코드·설정, `checkpoints/`, `results/`, `provenance/`를 함께 둡니다. 필요하지 않은 빈 파일은 만들지 않습니다.

실험 번호는 과거부터 최근 순서이며 새 실험은 마지막 번호 다음에 추가합니다. 완료된 실험 번호는 바꾸지 않습니다. 폴더에 실제로 필요한 파일만 두며, 공통 모델만 쓰는 실험에는 중복 `model.py`를 만들지 않습니다. 가중치 종류는 `best.pt`, `latest.pt`로 통일하고 후보·센서·시드는 상위 폴더와 설정으로 구분합니다.

`provenance`는 기존 가중치를 만든 원본입니다. 그 내부의 역사적 import·파일명과 체크포인트 설정은 보존합니다. 현행 실행은 실험 폴더의 코드 또는 `common`을 사용합니다. 실행 결과는 `runtime/`에 저장하고 정리된 결과만 실험 폴더에 반영합니다.

## 새 실험 등록

1. 연구 분야를 `ssai`, `variable_scale`, `performance` 중 하나로 정하고 실제 실험자를 기록합니다. 분야 담당자와 과거 실험자는 별도로 관리합니다.
2. 최신 main과 열린 PR에서 번호 사용 여부를 확인한 뒤 `research/<분야>/<번호>_<소문자 snake_case 변경명>/` 폴더를 만듭니다.
3. [8절 보고서 양식](docs/experiment_template.md)으로 목적·변경 구조·대조군·실제 배율·선택 기준을 작성합니다. 미실행 실험은 계획으로 표시합니다.
4. `research/experiments.json`에 `research_track`, `experimenter`, `report`를 등록하고, 해당 연구 페이지와 두 README의 같은 분야 표에 보고서 링크를 추가합니다. 표 첫 열은 `실험명 링크<br>[실험자]`입니다.
5. 실행·평가 후 세 표의 상태·확인된 결과를 함께 갱신하고 원본 가중치·결과·manifest를 검증합니다. 새 연구의 실험을 등록하지 않은 상태에서는 빈 표를 유지합니다.

가변 배율 실험은 학습 배율·평가 배율·입력/출력 크기·관측 연산과 배율별 대조군을 먼저 명시합니다. 각 배율의 RR·FR 결과를 구분하고 고정 4배 결과와 조건 없이 합치지 않습니다.

## 환경 준비

저장소 루트에서 실행합니다.

```bash
git submodule update --init SSA-MRN/references/upstream
python -m pip install -r SSA-MRN/requirements.txt
```

데이터는 `SSA-MRN/data/` 아래에 두거나 실행 옵션으로 별도 경로를 지정합니다. `data/processed` 단계는 사용하지 않습니다. 기존 서버의 원본 데이터와 실행 중인 checkout은 이동하지 않았습니다.

<a id="evaluate"></a>

## 가중치 확인·재평가

실험 번호와 해당 실험 안의 가중치 경로를 지정합니다. 이 명령은 학습을 시작하지 않습니다.

```bash
# 데이터 없이 strict load와 CPU 추론 확인
python SSA-MRN/common/checkpoint.py --experiment 20 --checkpoint checkpoints/D4/best.pt

# 지정한 H5 전체 RR 평가; FR이면 --protocol FR과 해당 H5를 사용합니다.
python SSA-MRN/common/checkpoint.py --experiment 20 --checkpoint checkpoints/D4/best.pt --h5 /path/to/test_qb_multiExm1.h5 --protocol RR --output SSA-MRN/runtime/D4_rr.json

# 보존 파일의 해시 확인
python SSA-MRN/common/verify.py --experiment 20
```

`--device cuda`로 GPU 평가가 가능합니다. 공통 재평가는 FP32이며 과거 실험의 추가 지표·관측 방식과 차이가 있을 수 있습니다. 저장된 결과를 덮어쓰지 않습니다. QNR·Q2n·SCC의 MATLAB parity는 미검증입니다.

## 새 학습

서버형 실험은 해당 `run.py --help`와 `prepare.py --help`에서 옵션을 확인합니다. Kaggle형 실험은 `run.py` 전체를 한 셀에 붙여 넣고 파일 상단의 데이터·학습·재개 설정을 지정합니다. Kaggle 코드는 학습 당시 고정된 원본 소스를 사용하므로 Internet이 필요할 수 있습니다.

실행 중인 서버 작업은 기존 checkout과 명령을 유지합니다. 새 폴더의 코드를 실행 중인 checkout에 덮어쓰거나 기존 runtime에 재개하지 않습니다. 재개 fingerprint가 달라질 수 있습니다.

[실험 보고서 양식](docs/experiment_template.md) · [실험 목록](research/experiments.json) · [관리 규칙](AGENTS.md)

## 신규 계열별 결과 보관

앞으로의 새 실험은 `experiments/ssai`, `experiments/variable_scale`, `experiments/performance`의 각 계열에서 독립 번호를 사용합니다. 기존 `research/performance`의 실험 번호·자료는 그대로 보존합니다. SSAI 결과는 [새catalog](experiments/catalog.json)와 [기존연구목록](research/experiments.json)에 같은 경로로 등록했습니다.

```bash
python SSA-MRN/common/verify.py --research-track ssai --experiment all
```
