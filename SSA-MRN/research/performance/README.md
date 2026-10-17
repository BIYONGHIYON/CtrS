# 기존 성능 개선

[전체 연구](../../../README.md) · [실행·등록 안내](../../README.md#새-실험-등록)

**담당: 김경찬·송준현·이병현**

이병현은 2026-10-16 SSAI 1-12번 정리 후 이 연구에 합류했습니다. 과거 실험의 실험자는 바꾸지 않습니다.

## 연구 범위

기존 복원 구조·학습 방법·손실 등의 수치 개선을 계속 비교합니다.

같은 센서·배율·데이터·학습 조건의 대조군과 RR·FR을 함께 비교합니다. validation으로 후보와 가중치를 선택하며 지표·시드·센서가 엇갈리면 부분 개선으로 기록합니다.

<a id="experiments"></a>

## 실험 표

오래된 실험부터 최근 실험 순서입니다.

| 실험 | 변경한 내용 | 확인된 결과 | 상태 | 개선 여부 |
|---|---|---|:---:|:---:|
| [1. 공개 코드 재현](1_public_code_reproduction/README.md)<br>[이병현] | 공개 코드 설정을 재현해 초기 기준을 확보했습니다. | RR·FR 재현 결과 확보; 구조 개선 실험이 아님 | 평⁠가⁠ ⁠완⁠료 | 비⁠교⁠ ⁠기⁠준 |
| [2. 논문 설정 재현](2_paper_setting_reproduction/README.md)<br>[김경찬] | 논문의 내부 특징 차원 설정으로 재현했습니다. | RR·FR 재현 결과 확보; 공개 코드 재현과 장치 조건이 다름 | 평⁠가⁠ ⁠완⁠료 | 비⁠교⁠ ⁠기⁠준 |
| [3. MS 관측 연산자 구현](3_ms_observation/README.md)<br>[이병현] | 센서 MTF·샘플링 위상·경계 처리를 구성했습니다. | 관측 구현 대응 확인; 학습 성능 비교가 아님 | 사⁠전⁠ ⁠검⁠증⁠ ⁠완⁠료 | 성⁠능⁠ ⁠비⁠교⁠ ⁠아⁠님 |
| [4. 23탭 확대·저해상도 보정·고주파 경로](4_interpolation_correction_detail/README.md)<br>[이병현] | 확대 연산, 출력 보정, PAN 고주파 경로를 개별 변경했습니다. | 23탭: PSNR +0.0557 dB, SAM −0.0429°, 잠정 QNR +0.0031; 다른 후보 이득 혼재 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [5. 23탭 확대 위치와 센서 확장](5_interpolation_scope_sensors/README.md)<br>[이병현] | 23탭 적용 위치를 분리하고 시드·센서를 확장했습니다. | RR 이득이 있으나 FR·센서별 결과가 엇갈림 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [6. 23탭 확대의 반복 시드 비교](6_interpolation_seed_repeat/README.md)<br>[이병현] | 전체 확대 및 입력 확대 후보를 추가 시드로 반복했습니다. | 전체 확대의 GF2 PSNR·QNR 개선 반복; SAM 및 QB QNR 혼재 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [7. 밴드별 게이트 고주파 주입](7_band_gated_detail/README.md)<br>[김윤지] | 밴드별 게이트로 PAN 고주파 주입량을 조절했습니다. | RR 악화로 후보 채택 보류 | 평⁠가⁠ ⁠완⁠료 | 개⁠선⁠ ⁠없⁠음 |
| [8. MTF 변화량 데이터 증강](8_mtf_augmentation/README.md)<br>[이병현] | 센서 열화 특성의 변화량을 학습 데이터에 추가했습니다. | MTF: PSNR -0.0064 dB, SAM -0.0061°; 잠정 QNR -0.0014 | 평가⁠ ⁠완료 | 부분⁠ ⁠개선 |
| [9. SSAI 내부 융합 연산 변경](9_ssai_fusion/README.md)<br>[이병현] | SSAI 핵심 융합 연산을 네 구조로 비교했습니다. | A1: SAM -0.0058°, 잠정 QNR +0.0127; 모든 후보의 PSNR은 하락 | 평가⁠ ⁠완료 | 부분⁠ ⁠개선 |
| [10. 반복 복원에 MS 관측 오차 전달](10_recurrent_observation/README.md)<br>[김윤지] | 반복 단계에 입력 MS와 관측한 출력의 차이를 전달했습니다. | PSNR +0.1158 dB, SAM −0.0183°, 잠정 QNR −0.0182 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [11. Haar 대역별 단계 복원과 게이트](11_haar_reconstruction/README.md)<br>[김윤지] | Haar 대역별 복원에 밴드·방향·위치 게이트를 적용했습니다. | 원본 대비 RR 큰 폭 악화; 학습 게이트의 추가 이득도 없음 | 평⁠가⁠ ⁠완⁠료 | 개⁠선⁠ ⁠없⁠음 |
| [12. SSAI 내부 특징 차원 비교](12_feature_dimension/README.md)<br>[이병현] | 같은 환경에서 내부 특징 차원 4와 6을 비교했습니다. | K4: 세 센서 RR PSNR·SAM 우세; WV3 잠정 QNR은 K6 우세 | 평가⁠ ⁠완료 | 비교⁠ ⁠기준 |
| [13. 반복 복원·관측 오차 전달의 재개 학습](13_recurrent_observation_resume/README.md)<br>[김윤지] | 반복 복원과 관측 오차 후보를 재개해 기준선과 비교했습니다. | 기준선 대비 RR 개선·잠정 QNR 하락; 별도 세션 비교 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [14. 분광·관측 일치·경계 손실 추가](14_auxiliary_losses/README.md)<br>[이병현] | 기본 MSE에 보조 손실을 하나씩 추가했습니다. | edge 0.1: PSNR −0.0661 dB, SAM −0.0119°, 잠정 QNR +0.0098 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [15. 단일 복원과 3단계 공유 복원](15_shared_reconstruction/README.md)<br>[김윤지] | 동일 초기값과 학습 조건에서 복원 횟수를 변경했습니다. | 반복 복원: PSNR +0.3000 dB, SAM −0.1078°, 잠정 QNR −0.0393 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [16. 반복 복원의 축소 전 저역통과 필터](16_bandlimited_downsampling/README.md)<br>[이병현] | 축소 전에 고정 Gaussian 필터를 추가했습니다. | PSNR −0.0196 dB, SAM +0.0022°, 잠정 QNR −0.0686 | 평⁠가⁠ ⁠완⁠료 | 개⁠선⁠ ⁠없⁠음 |
| [17. SSAI 단계·잔차 복원·초기 입력 변경](17_structural_ablation/README.md)<br>[이병현] | SSAI 단계, 복원 잔차, LRPAN 입력을 분리해 비교했습니다. | LRPAN 제거: PSNR +0.1297 dB, SAM −0.0704°, 잠정 QNR −0.0111 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [18. 작은 크기 특징 보존과 PAN 기여량 조절](18_coarse_feature_fusion/README.md)<br>[이병현] | 작은 특징을 병렬 전달하고 PAN 기여량을 학습했습니다. | N3: PSNR +0.1875 dB, SAM −0.0134°, 잠정 QNR −0.0147 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [19. 반복 복원과 PAN·MS 분리 경로](19_dual_stream_reconstruction/README.md)<br>[이병현] | 3단계 공유 복원과 분리 특징 경로를 비교했습니다. | C1: PSNR +0.2442 dB, SAM −0.1883°, 잠정 QNR +0.0250. 새 분리 경로 C5는 RR 악화 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [20. 반복 복원의 단계 간 특징 전달과 관측·PAN 차이](20_recurrent_guidance/README.md)<br>[이병현] | 밴드별 SSAI를 유지하며 단계 간 정보 전달을 비교했습니다. | D3: PSNR +0.0956 dB; D4·D5 관측 오차 감소; D6 잠정 QNR +0.0166, RR 악화 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [21. 특징 선택·복원 후 관측 보정·고주파 PAN 차이](21_guidance_refinement/README.md)<br>[이병현] | 공간·밴드별 특징 gate, 복원 후 관측 보정, 고주파 PAN 차이를 비교합니다. | E1: PSNR +0.3345 dB, SAM -0.2219°; E7: 잠정 QNR +0.0437, 지표별 이득 상이 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [22. 단계별 갱신량·특징 변환·마지막 단계 PAN 주입](22_stage_aware_restoration/README.md)<br>[이병현] | 공유 복원기의 단계별 조절과 마지막 단계 PAN 차이를 비교했습니다. | F0 기준 F4: PSNR -0.0009 dB, SAM +0.0064°, 잠정 QNR +0.0018; 추가 이득 혼재 | 평⁠가⁠ ⁠완료 | 부⁠분⁠ ⁠개선 |
| [23. 반복 복원·단계 간 특징 전달의 센서별 수렴 비교](23_multi_sensor_comparison/README.md)<br>[이병현] | 원본·3단계 공유 복원·특징 전달을 세 센서에서 150에포크 비교했습니다. | F0: QB PSNR +0.1271 dB, GF2 +0.8291 dB; WV3 잠정 QNR 하락. QB·GF2 RR은 repeat 우세 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [24. 3단계 공유 복원의 보정 계수 범위 제한](24_alpha_bounded_shared_reconstruction/README.md)<br>[김윤지] | 공유 scalar α의 표현만 sigmoid(a)로 바꿨습니다. | PSNR −0.0849 dB, SAM +0.0101°; 잠정 QNR +0.0159, RR 유지 목표 미달 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [25. 중간 해상도 복원 목표와 보조 손실](25_multiscale_supervision/README.md)<br>[김윤지] | 기존 복원 지점에 GT 기반 보조 L1을 추가했습니다. | 중간 head L1 감소; PSNR −0.0425 dB, SAM +0.0264°; 잠정 QNR +0.0101 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [26. 평가 가중치의 지수 이동 평균](26_ema_weight_averaging/README.md)<br>[송준현] | 일반 MSE 학습을 유지하고 평가 가중치에 EMA를 적용했습니다. | seed42: PSNR +0.0893 dB, SAM −0.0244°; 잠정 QNR −0.0018 | 평가&nbsp;완료<br>seed42 | 부분&nbsp;개선 |
| [27. Backbone 채널 어텐션 추가](27_backbone_channel_attention/README.md)<br>[김경찬] | Backbone 잔차 블록 4개에 채널 어텐션을 추가했습니다. | seed42: RR PSNR +0.4803 dB, SAM −0.1369°; 잠정 FR QNR −0.0226 | 평가&nbsp;완료<br>seed42 | 부분&nbsp;개선 |
| [28. 원본 복원 후 관측 오차 기반 잔차 보정](28_residual_refinement/README.md)<br>[이병현] | 원본 복원 후 두 번의 잔차 보정·오차 입력·가중치 공유를 비교했습니다. | 기존 G2 대비 G5: PSNR +0.1317 dB, SAM -0.0391°; 잠정 QNR -0.0020. G4는 QNR +0.0079 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [29. 병렬 특징 용량 확장과 독립 잔차 보정](../../experiments/performance/29_capacity_refinement/README.md)<br>[이병현] | 원본 backbone에 저해상도·원해상도 가지를 추가하고 독립 보정기·LRPAN 제거 결합을 비교했습니다. | R0 대비 W2: RR PSNR +0.7234 dB, SAM -0.3542°; 잠정 QNR -0.0922. 후보 모두 RR 개선·FR 악화 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [30. Focal Frequency Loss 추가](../../experiments/performance/30_focal_frequency_loss/README.md)<br>[송준현] | 기존 MSE에 주파수 오차 가중 FFL을 추가했습니다. | seed42: PSNR -0.0553 dB, SAM -0.0073°; 잠정 QNR -0.0022. validation도 기준선 우세 | 평가&nbsp;완료<br>seed42 | 부분&nbsp;개선 |
| [31. 보정 잔차의 저주파 억제](../../experiments/performance/31_low_frequency_preserving_residual/README.md)<br>[김경찬] | G5 두 보정기의 잔차에서 고정 Gaussian 저주파 성분을 제거했습니다. | P0 대비 PSNR -0.9426 dB·SAM +0.5369°; 잠정 QNR +0.01671. RR 유지 실패·채택 보류 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [37. 병렬 추가 가지의 위치별 게이트](../../experiments/performance/37_spatial_gate_on_parallel_branch/README.md)<br>[송준현] | W2 추가 특징에 1채널 sigmoid 게이트를 곱했습니다. | seed42: PSNR +0.02817dB·SAM −0.01138°; 잠정 QNR −0.04456, QNR·Ds 20/20장 악화 | 평가&nbsp;완료<br>seed42 | 부분&nbsp;개선 |
| [32. PAN 조건 특징 변조와 국소 회귀 사전 정보의 빠른 선별](../../experiments/performance/32_feature_modulation_local_regression_screen/README.md)<br>[이병현] | A0 best에서 PAN 조건 특징 변조(W1)·대조군(W1c)·국소 회귀 사전 정보(W2)를 10에폭 이어 학습해 선별했습니다. | 이어 학습 대조군 대비 best validation: W1 −0.06%, W1c −0.06%, W2 −0.03%; 기준 3% 미달, test 미평가 | 사전&nbsp;검증&nbsp;완료 | 판단&nbsp;보류 |
| [33. MS 기준 가역 밝기·대비 정규화](../../experiments/performance/33_ms_anchored_reversible_normalization/README.md)<br>[김경찬] | 원본 SSA-MRN 앞뒤에 LR MS 기준 공유 정규화·역변환을 추가했습니다. | seed42: PSNR +0.2111 dB·SAM −0.1242°·잠정 QNR +0.01746; 각 지표 20/20장 개선, 반복 검증 대기 | 평가&nbsp;완료<br>seed42 | 개선 |
| [34. 논문 채널 상관 어텐션과 다중 해상도 복원 재현](../../experiments/performance/34_paper_equation_reproduction/README.md)<br>[이병현] | K6 채널 상관행렬·bicubic·세 단계 잔차를 본문 수식에 맞춰 재현했습니다. | PSNR36.6224dB·잠정QNR0.90102; 과거 A0 대비 주요 RR·FR 악화, 복수 조건 변경 | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [35. 교차 축척 일관성 학습](../../experiments/performance/35_cross_scale_consistency/README.md)<br>[이병현] | 기존 30개 모델로 교차 축척 불일치와 FR Ds의 관계, 8방향 self-ensemble을 학습 없이 확인하고 FR 지표 구현을 toolbox와 대조했습니다. | I_cs↔Ds Spearman 0.58(기본)·0.39(전체)로 기준 미달, 학습 안 함; self-ensemble A0 PSNR −0.93dB; 지표 구조 toolbox와 동일 | 사전&nbsp;검증&nbsp;완료 | 성능&nbsp;비교&nbsp;아님 |
| [36. 저해상도 이득 지도 기반 세부정보 주입 제한](../../experiments/performance/36_gain_map_detail_injection/README.md)<br>[이병현] | 출력 구조 제한의 전제(고전 MTF-GLP의 FR Ds)와 이론 상한을 학습 없이 확인했습니다. | MTF-GLP Ds +0.1275·PSNR −4.92dB(A0 대비); 상한 0.50 < A0 0.871로 두 관문 실패, 학습 안 함 | 사전&nbsp;검증&nbsp;완료 | 성능&nbsp;비교&nbsp;아님 |
| [38. MS 기준 가역 정규화의 반복 시드 검증](../../experiments/performance/38_ms_anchored_normalization_seed_repeat/README.md)<br>[김경찬] | 33번의 정규화·역변환을 유지하고 seed43·44로 반복했습니다. | 세 시드 평균 PSNR +0.1800dB·잠정 QNR +0.00761; seed44 QNR·Ds 악화, 채택 보류 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [39. 국소 MS 통계 기반 가역 정규화](../../experiments/performance/39_local_ms_anchored_reversible_normalization/README.md)<br>[송준현] | N1 전체 MS 통계를 위치별5×5 공통 통계 지도로 바꿨습니다. | 3seed 평균 PSNR −0.13345dB·SAM +0.06825°·잠정 QNR −0.00370; seed44 일부FR 이득, 전체 악화·채택 보류 | 평가&nbsp;완료<br>3seed | 부분&nbsp;개선 |
| [40. MS 기준 평균 제거와 대비 정규화의 기여 분리](../../experiments/performance/40_ms_center_scale_ablation/README.md)<br>[김경찬] | 전역 공유 정규화의 평균 제거·대비 조절을 네 군으로 분리했습니다. | seed42 validation: A3 MSE 최저; A1·A2는A3 대비 +1.220%·+1.810%로 반복 중단. RR·FR 미평가 | 학습&nbsp;완료<br>평가&nbsp;대기 | 판단&nbsp;보류 |

[새 실험 등록 절차](../../README.md#새-실험-등록) · [실험 보고서 양식](../../docs/experiment_template.md)

## 실험 폴더와 네이밍

각 실험의 코드·보고서·가중치·결과는 이 연구 폴더 아래 `<번호>_<영문 변경명>/`에 함께 보관합니다. 세 연구 모두 정수 번호와 소문자 snake_case 변경명을 사용합니다.

가설이 다른 실험은 별도 폴더로 나누고 같은 가설의 후보·센서·반복 시드는 같은 실험의 `checkpoints/`와 `results/` 아래에서 구분합니다. 연구 목록은 이 페이지의 표에 추가하고 다른 분야에서는 복제하지 않고 보고서로 연결합니다.
