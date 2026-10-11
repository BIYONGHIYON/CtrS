<p align="center"><img src="icon/CtrS-icon-circle.png" alt="CtrS" width="100" /></p>

# CtrS · SSA-MRN 성능 개선 연구

## 연구 소개

고해상도 흑백 위성영상 **PAN**의 공간 정보와 저해상도 다중분광영상 **MS**의 색·분광 정보를 결합해 고해상도 MS를 복원하는 팬샤프닝 연구입니다. SSA-MRN을 재현하고 구조·손실·데이터 증강을 바꿔 복원 성능을 개선합니다.

## 현재 연구 기준

기준 모델은 **원본 SSA-MRN, 내부 특징 차원 K=4, 기본 MSE 손실**입니다. 동일 환경의 세 센서 validation 비교에서 다수결로 선택했습니다. 모든 센서에서 우세하다는 뜻은 아닙니다.

최근 **3단계 공유 복원 C1**은 같은 조건의 원본 대비 PSNR **+0.2442 dB**, SAM **−0.1883°**, 잠정 QNR **+0.0250**을 기록했습니다. 다른 조건의 반복 복원에서는 FR이 악화한 결과도 있어, 현재는 후속 검증 후보로 유지합니다.

## SSA-MRN 구조

PAN·축소 후 재확대한 PAN(LRPAN)·확대한 MS(LMS)에서 특징을 추출합니다. 전체·절반·1/4 크기에서 밴드별 SSAI로 융합하고, 작은 크기부터 복원하며 이전 크기의 특징을 결합합니다. 아래는 **원본 SSA-MRN의 단일 복원 구조**입니다.

<img src="SSA-MRN/docs/assets/architecture/ssa-mrn.png" alt="SSA-MRN 입력·다중 크기 SSAI·단계별 복원 구조" width="620" />

## 결과 이미지

같은 QB 장면의 **LR MS · PAN · 예측 · 정답** 순서입니다. 확대·대비 조정된 시각화이며, 수치 평가는 원래 값으로 수행했습니다.

### 원본 SSA-MRN · C0

![원본 SSA-MRN 결과](SSA-MRN/docs/assets/windows_dual_stream_qb_50/evaluation/C0_scene_08.png)

### 3단계 공유 복원 · C1

![3단계 공유 복원 결과](SSA-MRN/docs/assets/windows_dual_stream_qb_50/evaluation/C1_scene_08.png)

## 다른 실험

RR은 정답이 있는 축소 해상도 평가, FR은 고해상도 정답이 없는 원래 해상도 평가입니다. 수치 차이는 각 실험의 대조군 대비이며, 서로 다른 조건의 실험끼리 순위를 매기지 않습니다.

| 실험 | 변경한 내용 | 확인된 결과 | 상태 | 개선 여부 |
|---|---|---|---|---|
| [반복 복원과 PAN·MS 분리 경로](SSA-MRN/docs/experiments/windows_dual_stream_qb_50.md) | 3단계 공유 복원과 분리 특징 경로를 비교했습니다. | C1: PSNR +0.2442 dB, SAM −0.1883°, 잠정 QNR +0.0250. 새 분리 경로 C5는 RR 악화 | 평가 완료 | 부분 개선 |
| [작은 크기 특징 보존과 PAN 기여량 조절](SSA-MRN/docs/experiments/windows_coarse_anchor_qb_50.md) | 작은 특징을 병렬 전달하고 PAN 기여량을 학습했습니다. | N3: PSNR +0.1875 dB, SAM −0.0134°, 잠정 QNR −0.0147 | 평가 완료 | 부분 개선 |
| [SSAI 단계·잔차 복원·초기 입력 변경](SSA-MRN/docs/experiments/windows_structure_qb_50.md) | SSAI 단계, 복원 잔차, LRPAN 입력을 분리해 비교했습니다. | LRPAN 제거: PSNR +0.1297 dB, SAM −0.0704°, 잠정 QNR −0.0111 | 평가 완료 | 부분 개선 |
| [단일 복원과 3단계 공유 복원](SSA-MRN/docs/experiments/kaggle_b0_b1_k4_controlled.md) | 동일 초기값과 학습 조건에서 복원 횟수를 변경했습니다. | 반복 복원: PSNR +0.3000 dB, SAM −0.1078°, 잠정 QNR −0.0393 | 평가 완료 | 부분 개선 |
| [반복 복원의 축소 전 저역통과 필터](SSA-MRN/docs/experiments/kaggle_bandlimit_k4.md) | 축소 전에 고정 Gaussian 필터를 추가했습니다. | PSNR −0.0196 dB, SAM +0.0022°, 잠정 QNR −0.0686 | 평가 완료 | 개선 없음 |
| [SSAI 내부 특징 차원 비교](SSA-MRN/docs/experiments/windows_k_baselines.md) | 같은 환경에서 내부 특징 차원 4와 6을 비교했습니다. | 센서별 validation 다수결 2:1로 차원 4 선택; test는 일부만 평가 | 학습 완료·일부 평가 | 비교 기준 |
| [분광·관측 일치·경계 손실 추가](SSA-MRN/docs/experiments/windows_controlled.md) | 기본 MSE에 보조 손실을 하나씩 추가했습니다. | edge 0.1: PSNR −0.0661 dB, SAM −0.0119°, 잠정 QNR +0.0098 | 평가 완료 | 부분 개선 |
| [공개 코드 재현](SSA-MRN/docs/experiments/pan_k4.md) | 공개 코드 설정을 재현해 초기 기준을 확보했습니다. | RR·FR 재현 결과 확보; 구조 개선 실험이 아님 | 평가 완료 | 비교 기준 |
| [논문 설정 재현](SSA-MRN/docs/experiments/pan_k6.md) | 논문의 내부 특징 차원 설정으로 재현했습니다. | RR·FR 재현 결과 확보; 공개 코드 재현과 장치 조건이 다름 | 평가 완료 | 비교 기준 |
| [23탭 확대·저해상도 보정·고주파 경로](SSA-MRN/docs/experiments/a6000_architecture_qb.md) | 확대 연산, 출력 보정, PAN 고주파 경로를 개별 변경했습니다. | 23탭: PSNR +0.0557 dB, SAM −0.0429°, 잠정 QNR +0.0031; 다른 후보 이득 혼재 | 평가 완료 | 부분 개선 |
| [23탭 확대 위치와 센서 확장](SSA-MRN/docs/experiments/a6000_followup_123.md) | 23탭 적용 위치를 분리하고 시드·센서를 확장했습니다. | RR 이득이 있으나 FR·센서별 결과가 엇갈림 | 평가 완료 | 부분 개선 |
| [23탭 확대의 반복 시드 비교](SSA-MRN/docs/experiments/a6000_gf2_repeat_qb_input.md) | 전체 확대 및 입력 확대 후보를 추가 시드로 반복했습니다. | 전체 확대의 GF2 PSNR·QNR 개선 반복; SAM 및 QB QNR 혼재 | 평가 완료 | 부분 개선 |
| [밴드별 게이트 고주파 주입](SSA-MRN/docs/experiments/kaggle_band_gated_hf.md) | 밴드별 게이트로 PAN 고주파 주입량을 조절했습니다. | RR 악화로 후보 채택 보류 | 평가 완료 | 개선 없음 |
| [MTF 변화량 데이터 증강](SSA-MRN/docs/experiments/kaggle_mtf_pair.md) | 센서 열화 특성의 변화량을 학습 데이터에 추가했습니다. | 100에폭 validation 기록 확보; 이 보고서의 RR·FR 결과는 미확정 | 학습 완료·평가 대기 | 판단 보류 |
| [SSAI 내부 융합 연산 변경](SSA-MRN/docs/experiments/ssa_fusion_local.md) | SSAI 핵심 융합 연산을 네 구조로 비교했습니다. | validation MSE는 기존 A0가 최저; RR·FR 평가 대기 | 학습 완료·평가 대기 | 판단 보류 |
| [반복 복원에 MS 관측 오차 전달](SSA-MRN/docs/experiments/kaggle_b1_b2.md) | 반복 단계에 입력 MS와 관측한 출력의 차이를 전달했습니다. | PSNR +0.1158 dB, SAM −0.0183°, 잠정 QNR −0.0182 | 평가 완료 | 부분 개선 |
| [Haar 대역별 단계 복원과 게이트](SSA-MRN/docs/experiments/kaggle_wavelet_a0_a1.md) | Haar 대역별 복원에 밴드·방향·위치 게이트를 적용했습니다. | 원본 대비 RR 큰 폭 악화; 학습 게이트의 추가 이득도 없음 | 평가 완료 | 개선 없음 |
| [반복 복원·관측 오차 전달의 재개 학습](SSA-MRN/docs/experiments/kaggle_b1_b2_100_resume.md) | 반복 복원과 관측 오차 후보를 재개해 기준선과 비교했습니다. | 기준선 대비 RR 개선·잠정 QNR 하락; 별도 세션 비교 | 평가 완료 | 부분 개선 |
| [MS 관측 연산자 구현](SSA-MRN/docs/experiments/observation_validation.md) | 센서 MTF·샘플링 위상·경계 처리를 구성했습니다. | 관측 구현 대응 확인; 학습 성능 비교가 아님 | 사전 검증 완료 | 성능 비교 아님 |

**판정 기준:** 개선 = 같은 조건에서 주요 RR·FR 지표가 함께 개선, 부분 개선 = 지표·후보·센서별 결과 혼재, 개선 없음 = 평가에서 향상을 확인하지 못함, 판단 보류 = 최종 평가 부족입니다. 비교 기준·성능 비교 아님은 개선 판정 대상에서 제외합니다. 모든 FR QNR은 구현 대응 검증이 끝나지 않은 잠정 지표이며, 단일 실험의 개선은 최종 채택을 뜻하지 않습니다.

## 팀원

| 이름 | GitHub | 전공 | 역할 |
|---|---|---|---|
| 이병현 | [BIYONGHIYON](https://github.com/BIYONGHIYON) | 멀티미디어공학과 | 팀장 |
| 김경찬 | [erickks2y-jpg](https://github.com/erickks2y-jpg) | 멀티미디어공학과 | 팀원 |
| 송준현 | [choco-ssalbbang](https://github.com/choco-ssalbbang) | 멀티미디어공학과 | 팀원 |
| 김윤지 | [kimrose1015-max](https://github.com/kimrose1015-max) | 데이터사이언스전공 | 팀원 |
