# 기존 성능 개선

[전체 연구](../../../README.md) · [실행·등록 안내](../../README.md#새-실험-등록)

**담당: 김경찬·송준현**

기존 연구의 구조·학습·손실 변경을 이어서 비교합니다.

## 현재 상태

**기존 연구를 이어갑니다.** 새 분담의 담당자는 김경찬·송준현입니다. 아래 1-28번은 이병현·김윤지의 과거 연구도 포함한 팀 전체 이력이며, 각 실험의 실제 수행자를 그대로 표시합니다. 표시된 상태는 보관된 보고서 기준입니다.

## 연구 범위

기존 복원 구조·학습 방법·손실 등의 수치 개선을 계속 비교합니다. 새 분담으로 시작하는 SSAI 개선은 [SSAI 연구](../ssai/README.md), 가변 배율 확장은 [가변 배율 연구](../variable_scale/README.md)에 등록합니다.

김경찬의 [Backbone 채널 어텐션 실험](27_backbone_channel_attention/README.md)과 송준현의 [평가 가중치 EMA 실험](26_ema_weight_averaging/README.md)을 비롯해 각자의 연구를 이어갑니다. 후속 연구 계획과 실행 상태는 실제 확인 후 추가합니다.

같은 센서·배율·데이터·학습 조건의 대조군과 RR·FR을 함께 비교합니다. validation으로 후보와 가중치를 선택하며 지표·시드·센서가 엇갈리면 부분 개선으로 기록합니다.

<a id="experiments"></a>

## 실험 표

오래된 실험부터 최근 실험 순서입니다. 첫 열의 이름은 실제 실험자입니다.

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

[새 실험 등록 절차](../../README.md#새-실험-등록) · [실험 보고서 양식](../../docs/experiment_template.md)

## 실험 폴더와 네이밍

각 실험의 코드·보고서·가중치·결과는 이 연구 폴더 아래 `<번호>_<영문 변경명>/`에 함께 보관합니다. 세 연구 모두 정수 번호와 소문자 snake_case 변경명을 사용합니다. 번호는 저장소 전체에서 공유하며 최신 main과 열린 PR을 확인한 뒤 마지막 번호 다음을 사용합니다. 분야별로 1부터 다시 시작하지 않습니다.

가설이 다른 실험은 별도 폴더로 나누고 같은 가설의 후보·센서·반복 시드는 같은 실험의 `checkpoints/`와 `results/` 아래에서 구분합니다. 연구 목록은 이 페이지의 표에 추가하고 다른 분야에서는 복제하지 않고 보고서로 연결합니다.
