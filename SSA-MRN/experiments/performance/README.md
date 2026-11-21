# 성능 개선 실험

[연구 소개](../../../README.md) · [실행 안내](../../README.md) · [판정 기준](../../docs/judgement_criteria.md) · [보고서 양식](../../docs/experiment_template.md)

SSA-MRN의 구조·학습·손실을 바꿔 수치를 개선하는 연구입니다. **진행 중**이며 새 실험은 모두 여기에 등록합니다.

| 항목 | 내용 |
|---|---|
| 대조군 | 기준 모델 **C2MS-SP** ([81번](81_l1_spectral_loss_projection/README.md), 2026-11-14 채택). 경량판 C2MS-SP-Lite는 [83번](83_factorized_c2ms_finetuning/README.md) |
| 판정 | 같은 조건 대조군과 짝지은 RR test 20장의 PSNR·SAM·ERGAS ([v2.4](../../docs/judgement_criteria.md)) |
| 이전 기준 모델 | C2MS ([53번](53_scale_mixed_training/README.md), 2026-10-21), F1 ([49번](49_observation_refinement_seed_repeat/README.md), 2026-10-20) |
| 47번 이전 실험의 판정 | [47번 재평가](47_ground_truth_criteria_reevaluation/README.md)의 측정값을 v2.4로 다시 집계한 값 |

<a id="experiments"></a>

## 실험 표

번호가 큰 실험부터 적습니다. 수치는 대조군 대비 RR test PSNR 차이이며, 표는 [`catalog.json`](../catalog.json)에서 `python SSA-MRN/common/index.py`로 만듭니다.

<!-- experiments:start -->
| 실험 | 핵심 결과 | 상태 | 판정 |
|---|---|:---:|:---:|
| [90. 모델 내부 축소·확대를 데이터셋 방식(23탭 보간, MTF 블러 + 솎음)으로 교체](90_dataset_matched_internal_resampling/README.md)<br>이병현 | 학습 중. N(89번 확정)을 K2D 150에폭으로, 갈래 A·B·C를 50에폭 선별로 학습해 validation으로 고름 | 학습&nbsp;중 | 판단&nbsp;보류 |
| [89. 분광 손실 계수(λ)와 2배 거친 쌍 계수(c) 탐색: K2D 학습법, W2 2×2 합성곱 구현](89_spectral_coarse_loss_weights/README.md)<br>이병현 | λ* = 1.2 선택, c는 0-1.0 축소 범위에서 선택 중. 확정 학습은 90번 N 갈래로 함께 학습 | 학습&nbsp;중 | 판단&nbsp;보류 |
| [88. 일정 학습률 학습 뒤 감쇠 시작 시점의 저랭크 분해(K2)와 감쇠 구간 학습: 150에폭 경량 C2MS-SP 단일 시드 확인](88_factorization_at_decay_start/README.md)<br>이병현 | K2D(연산 66%) +0.036/+0.154/+0.310 dB (QB/GF2/WV3, 1시드), Lite보다 세 센서 모두 개선 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [87. C2MS-SP 잔차 블록 가운데 채널 축소 구조의 20에폭 미세조정 단일 시드 확인](87_reduced_width_short_finetuning/README.md)<br>이병현 | B32W96 −0.076/−0.253/−0.146, B32W64 −0.236/−0.510/−0.443 dB (QB/GF2/WV3, 1시드) | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [86. K10 사영 전치 연산의 FFT 계산과 C2MS-SP 추론 설정 측정](86_fft_adjoint_fast_inference/README.md)<br>이병현 | FFT 사영이 79번과 동일(1.3e-14), C2MS-SP 공식 설정 QB 256² 58.6→26.1 ms, graph_fp16 13.3 ms | 평가&nbsp;완료 | 성능&nbsp;비교&nbsp;아님 |
| [85. C2MS-SP 잔차 블록의 가운데 채널 축소·40에폭 미세조정과 빠른 사영 구현](85_hidden_width_reduction_fast_projection/README.md)<br>이병현 | 축소 갈래 셋 모두 시간 관문 탈락(학습 안 함). 빠른 사영은 결과 동일·256² 2.2-2.9배, FT0L WV3 +0.105 dB(참고) | 평가&nbsp;완료 | 성능&nbsp;비교&nbsp;아님 |
| [84. C2MS-SP 학습 길이 확장: 일정 학습률 학습과 감쇠 갈래로 150-1000에폭 한계 측정](84_long_training_decay_branches/README.md)<br>이병현 | validation 선택 B1000이 C2MS-SP 대비 GF2 +0.218·WV3 +0.433 dB 개선, QB −0.277 dB (seed 42) | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [83. 학습된 C2MS 가중치의 저랭크·depthwise 분해와 짧은 미세조정](83_factorized_c2ms_finetuning/README.md)<br>이병현 | C2MS-SP-Lite(연산 66%)가 C2MS 대비 3시드 QB +0.051·GF2 +0.157·WV3 +0.119 dB, SAM·ERGAS도 개선 | 평가&nbsp;완료 | **개선** |
| [82. 미학습 센서 적용 시 밴드별 센서 보정](82_cross_sensor_band_calibration/README.md)<br>이병현 | 미학습 센서에서 보정 + 사영이 사영만보다 WV3→WV2 +2.15 dB(C2MS, 3시드). 같은 센서에는 쓰지 않음 | 평가&nbsp;완료 | 성능&nbsp;비교&nbsp;아님 |
| [81. L1·분광 손실 C2MS에 관측 일치 사영 결합의 정식 판정과 사영 가속](81_l1_spectral_loss_projection/README.md)<br>이병현 | C2MS-SP가 C2MS 대비 3시드 QB +0.072·GF2 +0.191·WV3 +0.130 dB로 개선, 기준 모델 채택(2026-11-14) | 평가&nbsp;완료 | **개선** |
| [80. 깊고 얇은 W2 가지와 사영의 다중 센서 확인](80_multi_sensor_deep_narrow_projection/README.md)<br>이병현 | 64×16 + 사영이 C2MS 대비 3시드 GF2 +0.073 dB(개선), WV3 −0.332 dB(개선 없음) | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [79. 깊고 얇은 W2 가지와 L1·분광 손실, 관측 일치 사영 결합](79_deep_narrow_branch_projection/README.md)<br>이병현 | 64×16 + 사영이 C2MS 대비 QB 3시드 +0.051 dB·SAM −0.092°(연산 67%). 이득 대부분은 사영 | 평가&nbsp;완료 | **개선** |
| [78. 시드 앙상블 교사 증류로 작은 C2MS 학습](78_seed_ensemble_distillation/README.md)<br>이병현 | 교사 앙상블은 +0.341 dB였지만 학생은 정답만 학습한 대조군 대비 −0.024·+0.006 dB로 옮겨지지 않음 | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [77. 원래 해상도 backbone의 폭·깊이 확대](77_full_resolution_backbone_capacity/README.md)<br>이병현 | E2(64채널 × 8블록)가 C2MS 대비 QB +0.022 dB·SAM −0.018°로 잡음 수준 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [76. 1/4 해상도 병렬 가지(W4) 추가](76_quarter_resolution_branch/README.md)<br>이병현 | C2MS + W4가 C2MS 대비 QB +0.025 dB·SAM −0.010°로 잡음 수준 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [75. 축소한 W2 병렬 가지의 폭과 깊이 비교](75_parallel_branch_width_depth/README.md)<br>이병현 | C2MS 대비 QB 128×4 −0.201 dB, 64×8 −0.127 dB | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [74. W2 병렬 가지의 폭·깊이를 원본 backbone과 같게 줄인 C2MS](74_backbone_matched_parallel_branch/README.md)<br>이병현 | C2MS 대비 QB −0.521 dB·SAM +0.317°. C2MS 이득의 절반 이상이 W2 가지 용량에서 나옴 | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [73. C2MS L1 학습의 높은 학습률 단계 감소와 학습 연장](73_l1_step_learning_rate_schedule/README.md)<br>이병현 | 4e-4 단계 감소는 QB −0.295 dB, 1e-4 cosine 360에폭은 −0.026 dB. 두 갈래 모두 SAM만 개선 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [72. C2MS 이득의 주파수 대역·구조 크기 분해](72_c2ms_gain_frequency_structure_analysis/README.md)<br>이병현 | QB RR 이득 +0.873 dB의 87%가 주기 8 px 미만 세부에서 나옴(학습 없는 분석) | 사전&nbsp;검증&nbsp;완료 | 성능&nbsp;비교&nbsp;아님 |
| [71. 거친 쌍 PAN 흐림 증강](71_coarse_pair_pan_blur/README.md)<br>이병현 | WV3에서 거친 쌍 전용 흐림 −0.026 dB, 양쪽 쌍 흐림 −0.072 dB | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [70. C2MS 합성곱 참조 범위 확장](70_c2ms_dilated_backbone/README.md)<br>이병현 | QB +0.009 dB·SAM −0.004°로 잡음 수준, WV3는 PSNR·SAM 모두 악화 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [69. 65번 손실의 저주파 제곱 오차 보정: 관측 일치 손실과 저주파·고주파 손실 분리](69_low_frequency_squared_error_loss/README.md)<br>이병현 | QB 관측 일치 손실 −0.105 dB, 저주파·고주파 분리 +0.027 dB. WV3는 −0.94·−0.86 dB | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [68. C2MS SSAI 경로 접기·미사용 층 정리, 65번 SAM 이득의 주파수 분해, 시드 앙상블 진단](68_ssai_folding_sam_split_seed_ensemble/README.md)<br>이병현 | 미사용 층 삭제는 출력 동일, SSA 접기는 WV3에서 관문 미달, 3시드 앙상블은 QB test +0.256 dB | 사전&nbsp;검증&nbsp;완료 | 성능&nbsp;비교&nbsp;아님 |
| [67. PAN 밝기·흐림 증강 학습](67_pan_brightness_blur_augmentation/README.md)<br>이병현 | 흐림 증강이 WV2 전이 +1.025 dB(참고)지만 학습한 WV3에서는 −0.204 dB | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [66. L1 학습 모델의 원래 해상도 관측 오차 분석과 관측 일치 사영](66_observation_error_projection_analysis/README.md)<br>이병현 | 사영이 RR PSNR 약 +0.04 dB. 65번 + 사영이 C2MS 대비 QB 3시드 +0.072 dB·SAM −0.102° | 사전&nbsp;검증&nbsp;완료 | 성능&nbsp;비교&nbsp;아님 |
| [65. C2MS에 L1 손실과 분광 손실을 함께 적용](65_l1_spectral_loss/README.md)<br>이병현 | C2MS 대비 3시드 QB +0.028 dB·SAM −0.100°(잡음 수준), GF2 +0.106 dB(개선), WV3 +0.047 dB | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [64. C2MS W2 가지의 픽셀별 아핀 변조](64_w2_pixel_affine_modulation/README.md)<br>이병현 | C2MS 대비 QB −0.087 dB·SAM +0.004°·ERGAS +0.037 | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [63. C2MS 학습 손실을 MSE에서 L1로 교체](63_l1_reconstruction_loss/README.md)<br>이병현 | C2MS 대비 QB +0.084 dB·SAM −0.060°·ERGAS −0.027 | 평가&nbsp;완료 | **개선** |
| [62. C2MS 내부 특징과 넓은 문맥을 쓰는 고주파 밴드 비율 헤드](62_feature_context_band_ratio_head/README.md)<br>이병현 | 가장 나은 F3도 QB +0.043 dB·SAM −0.044°. 입력의 정보 한계로 보고 헤드 방식 정리 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [61. 고주파 세부의 밴드 비율 보정](61_high_frequency_band_ratio_correction/README.md)<br>이병현 | 가장 나은 H1도 QB +0.034 dB·SAM −0.040°로 SAM 목표(−0.15°) 미달 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [60. norm을 보존하는 분광 방향 보정](60_norm_preserving_spectral_direction_refinement/README.md)<br>이병현 | 8×8 국소 밴드 변환 상한이 SAM +0.088°로 관문 미달, 학습하지 않음 | 사전&nbsp;검증&nbsp;완료 | 성능&nbsp;비교&nbsp;아님 |
| [59. 기준 모델 C2MS의 cosine 학습 일정 연장](59_extended_cosine_training/README.md)<br>이병현 | 150에폭 대비 GF2 +0.205·WV3 +0.366 dB지만 QB −0.284 dB, WV2 전이(참고) −1.760 dB | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [58. 기준 모델 C2MS에 분광 손실을 더한 학습의 반복 시드·다른 센서 확인](58_spectral_loss_multi_sensor/README.md)<br>이병현 | 3시드 QB +0.026 dB(잡음 수준), GF2 +0.082 dB(개선), WV3 +0.034 dB(잡음 수준) | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [57. 기준 모델 C2MS의 학습 조건 선별: 가중치 EMA와 분광 손실](57_c2ms_training_conditions/README.md)<br>이병현 | 분광 손실이 C2MS 대비 QB +0.035 dB·SAM −0.048°(문턱 미만), EMA는 효과 없음 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [56. 원본 SSA-MRN에 축척 혼합 학습만 적용한 W2 용량·축척 혼합 2×2 기여 분해](56_original_scale_mixed_training/README.md)<br>이병현 | 병렬 가지 없는 축척 혼합(C0MS)은 원본 대비 3시드 −0.293 dB | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [55. 축척 혼합 학습을 뺀 W2 병렬 가지 모델의 센서 전이 비교](55_parallel_branch_without_scale_mixing/README.md)<br>이병현 | C2가 C2MS 대비 WV3 3시드 +0.055 dB지만 SAM +0.003°. WV2 전이는 C2MS가 +1.305 dB | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [54. 축척 혼합 학습 기준 모델 C2MS의 다른 센서 적용](54_multi_sensor_scale_mixed_training/README.md)<br>이병현 | C2MS가 원본 대비 3시드 GF2 +2.045·WV3 +1.007 dB, SAM·ERGAS도 개선 | 평가&nbsp;완료 | **개선** |
| [53. 2배 거친 학습 쌍을 섞은 축척 혼합 학습](53_scale_mixed_training/README.md)<br>이병현 | C2MS가 F1 대비 3시드 +0.679 dB·SAM −0.226°(원본 대비 +0.873 dB), 기준 모델 채택(2026-10-21) | 평가&nbsp;완료 | **개선** |
| [52. W2 추가 특징의 고주파 선택 제어](52_w2_frequency_selective_attenuation/README.md)<br>송준현 | W2 C2 대비 3시드 +0.017 dB로 잡음 수준 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [51. 정답 없는 FR 세부 지표 후보의 타당성 검증](51_fr_detail_index_validity/README.md)<br>이병현 | FR 세부 지표 네 개 모두 정답을 거꾸로 채점해 판정에 쓰지 않음 | 사전&nbsp;검증&nbsp;완료 | 성능&nbsp;비교&nbsp;아님 |
| [50. W2 가지 출력의 추론 시 축소 진단](50_w2_branch_inference_scaling/README.md)<br>이병현 | W2 출력 세기 α=0.5에서 RR2가 +1.42 dB로 최고. 약점은 세부 주입 세기의 축척 의존성 | 사전&nbsp;검증&nbsp;완료 | 성능&nbsp;비교&nbsp;아님 |
| [49. 관측 오차 보정기(G5) 계열의 3시드 재현](49_observation_refinement_seed_repeat/README.md)<br>이병현 | F1이 원본 대비 3시드 +0.194 dB·SAM −0.139° | 평가&nbsp;완료 | **개선** |
| [48. 저해상도 병렬 특징 용량(W2)과 관측 오차 보정기(G5)의 결합](48_parallel_capacity_observation_refinement/README.md)<br>이병현 | WF가 원본 대비 seed 42 +0.650 dB·SAM −0.372°·ERGAS −0.324 | 평가&nbsp;완료 | **개선** |
| [47. 정답 기반 판정 기준(v2)과 기존 후보 재평가](47_ground_truth_criteria_reevaluation/README.md)<br>이병현 | 보존된 측정값을 v2.4로 다시 집계: 개선 34·부분 개선 29·개선 없음 33 | 평가&nbsp;완료 | 성능&nbsp;비교&nbsp;아님 |
| [46. FR 공간 왜곡 지표의 축소 해상도 타당성 검증](46_fr_spatial_index_validity/README.md)<br>이병현 | FR Ds가 정답을 모델보다 낮게 채점(1-3/20장만 우세)해 판정에 쓰지 않음 | 사전&nbsp;검증&nbsp;완료 | 성능&nbsp;비교&nbsp;아님 |
| [45. 추론 시 정확한 관측 일치 사영](45_exact_observation_projection/README.md)<br>이병현 | 사영이 RR 240/240장 개선(C0 +0.129 dB)이나 당시 FR 기준 미달로 학습하지 않음 | 사전&nbsp;검증&nbsp;완료 | 성능&nbsp;비교&nbsp;아님 |
| [44. 위상 보존 주파수 대역 안내](44_phase_preserving_frequency_guidance/README.md)<br>이병현 | C0가 QB seed 42 +0.089 dB로 개선, F1·F2는 부분 개선 | 평가&nbsp;완료 | **개선** |
| [43. MS 기준 가역 정규화와 저해상도 병렬 특징 용량 확장](43_normalized_parallel_capacity/README.md)<br>이병현 | C3가 원본 대비 3시드 +0.738 dB·SAM −0.323° | 평가&nbsp;완료 | **개선** |
| [42. 백본 일부 합성곱의 dilation 확장](42_dilated_backbone_context/README.md)<br>송준현 | 원본 대비 3시드 +0.122 dB·SAM −0.025°·ERGAS −0.061 | 평가&nbsp;완료 | **개선** |
| [41. 공통 밝기·대비 증강](41_joint_photometric_augmentation/README.md)<br>송준현 | 원본 대비 3시드 +0.024 dB·SAM −0.001° | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [40. MS 기준 평균 제거와 대비 정규화의 기여 분리](40_ms_center_scale_ablation/README.md)<br>김경찬 | A3가 QB seed 42 +0.211 dB·SAM −0.124°·ERGAS −0.117 | 평가&nbsp;완료 | **개선** |
| [39. 국소 MS 통계 기반 가역 정규화](39_local_ms_anchored_reversible_normalization/README.md)<br>송준현 | 원본 대비 3시드 −0.133 dB·SAM +0.068° | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [38. MS 기준 가역 정규화의 반복 시드 검증](38_ms_anchored_normalization_seed_repeat/README.md)<br>김경찬 | N1이 원본 대비 3시드 +0.180 dB·SAM −0.074°·ERGAS −0.093(33번과 합친 결과) | 평가&nbsp;완료 | **개선** |
| [37. 병렬 추가 가지의 위치별 게이트](37_spatial_gate_on_parallel_branch/README.md)<br>송준현 | seed 42 +0.028 dB·SAM −0.011°로 잡음 수준 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [36. 저해상도 이득 지도 기반 세부정보 주입 제한](36_gain_map_detail_injection/README.md)<br>이병현 | MTF-GLP 상한이 대조군보다 낮아 두 관문 실패, 학습하지 않음 | 사전&nbsp;검증&nbsp;완료 | 성능&nbsp;비교&nbsp;아님 |
| [35. 교차 축척 일관성 학습](35_cross_scale_consistency/README.md)<br>이병현 | 축척 일관성 지표와 Ds의 순위 상관 0.58·0.39로 기준 미달, 학습하지 않음 | 사전&nbsp;검증&nbsp;완료 | 성능&nbsp;비교&nbsp;아님 |
| [34. 논문 채널 상관 어텐션과 다중 해상도 복원 재현](34_paper_equation_reproduction/README.md)<br>이병현 | 구조·학습 조건이 함께 바뀌고 같은 조건 대조군이 없음 | 평가&nbsp;완료 | 판단&nbsp;보류 |
| [33. MS 기준 가역 밝기·대비 정규화](33_ms_anchored_reversible_normalization/README.md)<br>김경찬 | N1이 원본 대비 3시드 +0.180 dB·SAM −0.074°·ERGAS −0.093(38번과 합친 결과) | 평가&nbsp;완료 | **개선** |
| [32. PAN 조건 특징 변조와 국소 회귀 사전 정보의 빠른 선별](32_feature_modulation_local_regression_screen/README.md)<br>이병현 | 10에폭 validation 탐색만 했고 RR test 평가 없음 | 사전&nbsp;검증&nbsp;완료 | 판단&nbsp;보류 |
| [31. 보정 잔차의 저주파 억제](31_low_frequency_preserving_residual/README.md)<br>김경찬 | P0 대비 P1 −0.943 dB·SAM +0.537°·ERGAS +0.496 | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [30. Focal Frequency Loss 추가](30_focal_frequency_loss/README.md)<br>송준현 | QB seed 42 −0.055 dB·SAM −0.007° | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [29. 병렬 특징 용량 확장과 독립 잔차 보정](29_capacity_refinement/README.md)<br>이병현 | W2가 QB seed 42 +0.723 dB·SAM −0.354°·ERGAS −0.350 | 평가&nbsp;완료 | **개선** |
| [28. 원본 복원 후 관측 오차 기반 잔차 보정](28_residual_refinement/README.md)<br>이병현 | QB seed 46에서 G3·G4·G5 개선 | 평가&nbsp;완료 | **개선** |
| [27. Backbone 채널 어텐션 추가](27_backbone_channel_attention/README.md)<br>김경찬 | E1이 seed 42 +0.480 dB·SAM −0.137°·ERGAS −0.220(모델 소스 미보존) | 평가&nbsp;완료 | **개선** |
| [26. 평가 가중치의 지수 이동 평균](26_ema_weight_averaging/README.md)<br>송준현 | QB seed 42 +0.089 dB·SAM −0.024° | 평가&nbsp;완료 | **개선** |
| [25. 중간 해상도 복원 목표와 보조 손실](25_multiscale_supervision/README.md)<br>김윤지 | QB seed 42 −0.043 dB·SAM +0.026° | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [24. 3단계 공유 복원의 보정 계수 범위 제한](24_alpha_bounded_shared_reconstruction/README.md)<br>김윤지 | QB seed 42 −0.085 dB·SAM +0.010° | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [23. 반복 복원·단계 간 특징 전달의 센서별 수렴 비교](23_multi_sensor_comparison/README.md)<br>이병현 | QB seed 46 repeat +0.176 dB·SAM −0.092° | 평가&nbsp;완료 | **개선** |
| [22. 단계별 갱신량·특징 변환·마지막 단계 PAN 주입](22_stage_aware_restoration/README.md)<br>이병현 | QB seed 42에서 F4만 부분 개선, 나머지 네 후보는 개선 없음 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [21. 특징 선택·복원 후 관측 보정·고주파 PAN 차이](21_guidance_refinement/README.md)<br>이병현 | QB seed 42에서 E1-E7 일곱 후보 모두 개선 | 평가&nbsp;완료 | **개선** |
| [20. 반복 복원의 단계 간 특징 전달과 관측·PAN 차이](20_recurrent_guidance/README.md)<br>이병현 | D3가 QB seed 42 +0.096 dB·SAM −0.113°. D3·D5 개선 | 평가&nbsp;완료 | **개선** |
| [19. 반복 복원과 PAN·MS 분리 경로](19_dual_stream_reconstruction/README.md)<br>이병현 | C1이 QB seed 42 +0.244 dB·SAM −0.188°. 나머지 네 후보는 개선 없음 | 평가&nbsp;완료 | **개선** |
| [18. 작은 크기 특징 보존과 PAN 기여량 조절](18_coarse_feature_fusion/README.md)<br>이병현 | N3가 QB seed 42 +0.187 dB·SAM −0.013°·ERGAS −0.101 | 평가&nbsp;완료 | **개선** |
| [17. SSAI 단계·잔차 복원·초기 입력 변경](17_structural_ablation/README.md)<br>이병현 | I1이 QB seed 42 +0.130 dB·SAM −0.070°. M1·I1 개선 | 평가&nbsp;완료 | **개선** |
| [16. 반복 복원의 축소 전 저역통과 필터](16_bandlimited_downsampling/README.md)<br>이병현 | QB seed 42 −0.020 dB·SAM +0.002° | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [15. 단일 복원과 3단계 공유 복원](15_shared_reconstruction/README.md)<br>김윤지 | B1이 QB seed 42 +0.300 dB·SAM −0.108°·ERGAS −0.148 | 평가&nbsp;완료 | **개선** |
| [14. 분광·관측 일치·경계 손실 추가](14_auxiliary_losses/README.md)<br>이병현 | edge 0.1이 QB seed 42 −0.066 dB·SAM −0.012° | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [13. 반복 복원·관측 오차 전달의 재개 학습](13_recurrent_observation_resume/README.md)<br>김윤지 | QB seed 42 −0.120 dB·SAM +0.009° | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [12. SSAI 내부 특징 차원 비교](12_feature_dimension/README.md)<br>이병현 | K6이 K4 대비 QB seed 42 −0.034 dB·SAM +0.014° | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [11. Haar 대역별 단계 복원과 게이트](11_haar_reconstruction/README.md)<br>김윤지 | QB seed 42 −0.254 dB·SAM +0.082° | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [10. 반복 복원에 MS 관측 오차 전달](10_recurrent_observation/README.md)<br>김윤지 | B2가 QB seed 42 +0.116 dB·SAM −0.018°·ERGAS −0.054 | 평가&nbsp;완료 | **개선** |
| [9. SSAI 내부 융합 연산 변경](9_ssai_fusion/README.md)<br>이병현 | QB seed 46에서 A1만 부분 개선, A2·A3는 개선 없음 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [8. MTF 변화량 데이터 증강](8_mtf_augmentation/README.md)<br>이병현 | QB seed 46 −0.006 dB·SAM −0.006° | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [7. 밴드별 게이트 고주파 주입](7_band_gated_detail/README.md)<br>김윤지 | QB seed 42 −0.175 dB·SAM +0.009° | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [6. 23탭 확대의 반복 시드 비교](6_interpolation_seed_repeat/README.md)<br>이병현 | 입력 보간 교체가 3시드 +0.058 dB·SAM −0.021°·ERGAS −0.028 | 평가&nbsp;완료 | **개선** |
| [5. 23탭 확대 위치와 센서 확장](5_interpolation_scope_sensors/README.md)<br>이병현 | 출력 보간 교체가 QB seed 42 −0.014 dB·SAM −0.004° | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [4. 23탭 확대·저해상도 보정·고주파 경로](4_interpolation_correction_detail/README.md)<br>이병현 | QB seed 42에서 세 후보 모두 부분 개선 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [3. MS 관측 연산자 구현](3_ms_observation/README.md)<br>이병현 | 관측 연산자 구현이 기준 구현과 대응함을 확인(학습 없음) | 사전&nbsp;검증&nbsp;완료 | 성능&nbsp;비교&nbsp;아님 |
| [2. 논문 설정 재현](2_paper_setting_reproduction/README.md)<br>김경찬 | 논문 설정으로 RR·FR 재현 결과 확보 | 평가&nbsp;완료 | 비교&nbsp;기준 |
| [1. 공개 코드 재현](1_public_code_reproduction/README.md)<br>이병현 | 공개 코드로 세 센서 150에폭 학습, RR·FR 각 20장 평가 완료 | 평가&nbsp;완료 | 비교&nbsp;기준 |
<!-- experiments:end -->
