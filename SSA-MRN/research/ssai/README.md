# SSAI 개선

[전체 연구](../../../README.md) · [실행·등록 안내](../../README.md)

**담당·실험자: 이병현**

**상태: 연구 종료(2026-10-16)**

## 종료 판단

SSAI 1-11번을 마치고 이 연구 분야를 종료했습니다. 근거는 [SSAI 11](../../experiments/ssai/11_ssai_contribution_headroom/README.md)의 학습 없는 진단과 1-10번의 결과입니다.

- 학습된 원본 A0에서 SSA 블록 출력을 0으로 바꿔도 RR·FR 지표가 소수 5자리까지 같습니다. SSAI 단계는 단계·밴드별 상수 보정으로 작동하며, 이 상수를 제거하면 RR PSNR이 1.55 dB 떨어집니다.
- SSAI 형태의 결합(PAN 기저 × 국소 계수)을 GT로 완전하게 맞춘 oracle도 null 대조군을 뺀 설명량이 8×8 칸에서 11.3%, 약 0.5 dB에 그칩니다.
- SSAI 1-10번의 RR 변화는 대부분 ±0.06 dB 안에 머물렀고, 10번의 국소 창 어텐션도 같은 에폭 A0 대비 validation MSE −1% 안팎이었습니다.

따라서 현재 구조에서 SSAI 결합 연산만 바꾸는 개선으로는 큰 RR·FR 향상을 기대하기 어렵다고 판단했습니다. 이 판단은 QB·seed42 원본 하나와 선형 결합 상한에 근거하며, SSAI를 큰 용량의 비선형 모듈로 바꾸는 경우는 포함하지 않습니다. 이후 이병현은 [기존 성능 개선](../performance/README.md) 연구에 참여합니다. 새 SSAI 실험은 등록하지 않으며, SSAI 관련 가설이 다시 필요하면 성능 개선 연구에서 등록하고 이 페이지를 참고 링크로 연결합니다.

SSAI의 PAN 공간정보와 MS 분광정보를 결합하는 연산을 개선합니다. 기준 모델은 **원본 SSA-MRN·K4·MSE·단일복원·공간4배확대**로 유지합니다.

## SSAI를 읽는 방법

각 밴드는 PAN과 현재 전체 MS 특징·해당 밴드를 K채널로 바꿉니다. 두 특징을 곱해 공간 가중치를 만든 뒤 PAN 특징을 가중하며, 밴드별 출력을 모아 원본 Conv로 혼합합니다. 전체·절반·1/4 해상도에 적용합니다. 전체 픽셀 간 행렬 곱이나 MS Value 집계는 없습니다.

정사각 입력의 원본 logit은 `Pᵀ⊙Q`입니다. 두 전치 사이에 원소 곱이 있어 상쇄되지 않습니다. SSAI 1은 같은 위치의 `P⊙Q`로 바꿉니다. 직사각 입력에는 reshape 재배열까지 포함되어 단순 전치 공식으로 설명하지 않습니다.

공간 softmax의 평균은 각 K채널에서 정확히 `1/(H×W)`입니다. SSAI 2는 학습 면적 대비 현재 면적을 곱하고, SSAI 3은 원본 전치를 유지한 고정 크기 sigmoid로 바꿉니다. K4는 내부 특징 차원이며 영상 공간 4배 확대와 별개입니다.

## 실험 표

| 실험 | 변경한 내용 | 확인된 결과 | 상태 | 개선 여부 |
|---|---|---|:---:|:---:|
| [1. 공간 좌표 정렬](../../experiments/ssai/1_spatial_alignment/README.md)<br>[이병현] | 원본의 두 공간 전치를 제거해 같은 위치의 PAN·MS 특징을 결합했습니다. | PSNR +0.0545dB, SAM -0.0351°, 잠정QNR +0.0077 | 평가&nbsp;완료 | 개선 |
| [2. 기준 면적 정규화](../../experiments/ssai/2_reference_area_normalization/README.md)<br>[이병현] | 원본 SSAI 출력에 현재 면적/학습 기준 면적을 곱했습니다. | PSNR -0.0002dB, SAM +0.0009°, 잠정QNR -0.0000 | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [3. 기준 크기 sigmoid 게이트](../../experiments/ssai/3_calibrated_sigmoid_gating/README.md)<br>[이병현] | 원본 전치를 유지하고 공간 softmax를 기준 크기 sigmoid 게이트로 교체했습니다. | PSNR +0.0226dB, SAM -0.0250°, 잠정QNR +0.0090 | 평가&nbsp;완료 | 판단&nbsp;보류 |
| [4. 공간 좌표 정렬과 sigmoid 게이트 결합](../../experiments/ssai/4_aligned_sigmoid_gating/README.md)<br>[이병현] | 좌표 정렬과 기준 크기 sigmoid를 결합하고 네 후보를 동일 조건으로 비교합니다. | A3: PSNR -0.0158dB, SAM +0.0226°, 잠정 QNR +0.0099 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [5. softmax와 sigmoid 혼합 게이트](../../experiments/ssai/5_mixed_gating/README.md)<br>[이병현] | 원본 좌표에서 softmax와 기준 크기 sigmoid를 고정 비율로 혼합합니다. | M25 개선 없음; M50 PSNR -0.0710dB·잠정 QNR +0.0043 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [6. MS 조건부 공간 세부정보 보정](../../experiments/ssai/6_ms_conditioned_detail/README.md)<br>[이병현] | MS가 선택한 PAN 세부정보를 원본 출력에 잔차 보정합니다. | PSNR -0.0017dB·SAM +0.0038°·잠정QNR -0.0046 | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [7. MS–PAN 채널 관계 기반 세부정보 보정](../../experiments/ssai/7_channel_relation_detail/README.md)<br>[이병현] | 4×4 MS–PAN 채널 관계로 PAN 세부정보를 선택·보정합니다. | PSNR -0.0061dB·SAM +0.0158°·잠정QNR -0.0030 | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [8. 공간 가중치 평균 정규화](../../experiments/ssai/8_spatial_mean_normalization/README.md)<br>[이병현] | 원본 공간 softmax 평균을1로 맞추고 고정·학습 gamma를 비교합니다. | N1 개선 없음; N2 PSNR +0.0104dB·SAM +0.0036°·잠정QNR -0.0019; 두 후보 학습 가능화 성공 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [9. PAN 재구성 오차 기반 보정](../../experiments/ssai/9_pan_innovation_correction/README.md)<br>[이병현] | PAN 직접 고주파와 PAN 재구성 오차를 같은 크기의 SSAI에서 비교했습니다. | A1 PSNR -0.0314dB·A2 -0.0352dB, 잠정 QNR 모두 악화 | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [10. 국소 창 교차 어텐션 SSAI](../../experiments/ssai/10_local_window_cross_attention/README.md)<br>[이병현] | 전역 공간 softmax를 7×7 국소 창 어텐션으로 바꾸고 동일 파라미터 convolution 대조군과 비교합니다. | 같은 에폭 A0 대비 validation: L2 −0.81%(21에폭), C −1.15%(45에폭); test 미평가 | 학습&nbsp;중단<br>10-16&nbsp;01:20 | 판단&nbsp;보류 |
| [11. SSAI 경로 제거 분해와 PAN 기저 결합 상한·FR 대리 지표 검증](../../experiments/ssai/11_ssai_contribution_headroom/README.md)<br>[이병현] | 학습 없이 A0 best에서 SSA 출력·SSAI 단계·coarse 경로·backbone 잔차를 제거하고, PAN 기저 결합 oracle 상한과 validation FR 대리 지표를 검증했습니다. | SSA 출력 제거 시 RR·FR 동일; SSAI 단계 제거 PSNR −1.55dB(상수 보정); oracle null 초과 설명량 11.3%; 대리 Ds 순위 상관 −0.18 | 사전&nbsp;검증&nbsp;완료<br>10-16&nbsp;03:02 | 성능&nbsp;비교&nbsp;아님 |

## 비교 조건과 한계

A0·A1은 같은 초기값·시드·샘플 순서와 같은 micro batch 경로로 비교했습니다. 처음 3에폭은 micro4, 이후 micro32·GPU cache로 재개했고 유효 배치32·FP32는 유지했습니다. A2는 처음부터 micro32여서 SSAI만의 효과로 확정하지 않습니다. 현행 최적화는 동시 2개, 실측 합산346.3patch/s·학습 중 GPU99%·VRAM 약7.2/8GiB였습니다. OOM 자동 감축·여유 VRAM 하한 제어는 없으며 이 수치는 실측 상태입니다.

A1의 단일 시드 RR·잠정 FR 동반 개선은 반복 검증과 기준 모델 채택을 대신하지 않습니다. 기존 test는 탐색에 사용됐으며 QNR·Q2n·SCC MATLAB parity는 미검증입니다. R1은64×64 validation에서 원본과 같아 test로 계수나 후보를 고르지 않는 사전 고정 진단입니다.

## 선행 연구

[성능 개선9 · SSAI 융합](../performance/9_ssai_fusion/README.md), [성능 개선12 · 특징차원](../performance/12_feature_dimension/README.md), [성능 개선17 · SSAI 단계](../performance/17_structural_ablation/README.md)를 참고합니다. 기존 원본 자료와 번호를 보존합니다.

SSAI 1-11번 결과는 `SSA-MRN/experiments/ssai/<번호>_<변경명>/`에 보존합니다. 연구를 종료했으므로 새 번호를 추가하지 않습니다.



4번의 네 후보는 처음부터 동일 micro32·FP32 조건으로 각 100에폭 학습하고 RR·FR 각 20장면을 평가했습니다. A3는 FR 개선·RR 악화로 부분 개선이며, 세 후보 모두 validation 후속 기준을 통과하지 못했습니다. 원본 기준 모델을 유지합니다. 실제 best/latest 8개와 학습 소스·설정·전체 숫자·그래프·고정 예시를 4번 폴더에 보관했습니다.

5번은 신규 혼합 후보 두 개만100에폭 학습하고 실험4의 동일조건 A0·A2를 참조했습니다. M25는 개선 없음, M50은 FR 개선·RR 악화이며 두 후보 모두 validation 기준을 통과하지 못했습니다. 새80개 RR/FR 예측·실제 best/latest4개·학습소스·설정·그래프·고정 예시를5번에 보관했습니다. 원본 기준 모델을 유지합니다.

6·7번은원본SSA기반의MS조건부세부정보보정을각100에폭학습했습니다. 7번validation은0.116%좋아졌지만RR주요지표·잠정FR은악화했고두실험모두개선없음입니다. 실제신규80개예측·best/latest4개·원본소스/설정·그래프·사전고정예시를보존하고원본을유지합니다. 다음우선검토는1번좌표정렬의반복시드재현성이며새학습은시작하지않았습니다.

9번은PAN 직접 고주파와PAN 재구성 오차 보정을각100에폭 학습하고 A0·A1·A2의RR20·FR20 전체를 같은 환경에서 평가했습니다. 두 후보 모두SSAI 가중치가 크게 변했지만 주요RR 지표와잠정QNR이 악화해 개선없음입니다. 원본A0를 유지합니다.
