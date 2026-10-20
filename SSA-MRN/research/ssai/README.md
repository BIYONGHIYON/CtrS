# SSAI 개선

[전체 연구](../../../README.md) · [실행·등록 안내](../../README.md)

**담당·실험자: 이병현** · 1-12번 정리 완료, 담당자는 [기존 성능 개선](../performance/README.md)으로 이동

SSAI의 PAN 공간정보와 MS 분광정보를 결합하는 연산을 개선합니다. 기준 모델은 **원본 SSA-MRN·K4·MSE·단일복원·공간4배확대**로 유지합니다.

## 연구 정리

2026-10-16에 SSAI 1-12번 결과를 정리했습니다. 새 SSAI 실험은 계획하지 않으며, 이병현은 기존 성능 개선 연구로 이동했습니다. 기존 실험의 번호·실험자·결과·가중치는 그대로 보존합니다.

| 확인한 사실 | 근거 실험 |
|---|---|
| test까지 평가한 SSAI 후보의 A0 대비 RR PSNR 변화는 −0.071에서 +0.055 dB 사이였습니다(10번은 test 미평가). 같은 조건 A0 두 번의 차이는 PSNR 0.036 dB, QNR 0.004입니다. | 1-9, 12 |
| 현행 v2.3에서 1번 A1(+0.055dB)과 12번 진단용 B3(+0.050dB)는 모든 판정 항목을 통과했지만 이득이 단일 시드 잡음 문턱 0.06dB 미만이라 부분 개선입니다. 1번의 이득은 4번 재비교에서 재현되지 않았습니다. | 1, 4, 12 |
| A0에서 SSA 출력을 0으로 바꿔도 RR·FR 지표가 같았습니다. SSAI 단계 전체를 제거하면 RR PSNR이 −1.55 dB 떨어졌습니다. | 11 |
| SSA 경로를 실제로 쓰게 만든 후보 중 12번 B3는 현행 기준에서 개선이며, 8번 N1·9번·12번 B2는 개선 없음입니다. 과거 QNR 하락만으로 FR 정확도 악화를 판단하지 않습니다. | 8, 9, 12 |
| A0 오차를 PAN 국소 선형 결합으로 줄일 수 있는 이상적 상한은 약 0.52 dB였습니다. | 11 |
| validation에서 계산한 FR 대리 Ds와 test Ds의 순위 상관은 −0.18이었습니다. | 11 |

이 판단은 공개 코드 기반 SSAI 구현, QB 중심 단일 시드, 탐색에 반복 사용한 test, MATLAB 정합을 확인하지 않은 잠정 FR 지표에 한정합니다. 논문 식 (10)-(12)의 K×K 채널 상관 SSAI는 [성능 개선 34번](../../experiments/performance/34_paper_equation_reproduction/README.md)에서 다른 조건과 함께 바뀐 형태로만 확인했습니다. 큰 용량의 비선형 결합이나 다른 센서·배율에서의 효과는 확인하지 않았습니다.

남은 단서로 12번 B2의 FR Dλ 개선(A0·N1 대비 20/20장)이 있습니다. test 결과를 본 뒤 관찰한 것이므로, 이어 보려면 새 시드를 쓰는 별도 가설로 등록해야 합니다.

## SSAI를 읽는 방법

각 밴드는 PAN과 현재 전체 MS 특징·해당 밴드를 K채널로 바꿉니다. 두 특징을 곱해 공간 가중치를 만든 뒤 PAN 특징을 가중하며, 밴드별 출력을 모아 원본 Conv로 혼합합니다. 전체·절반·1/4 해상도에 적용합니다. 전체 픽셀 간 행렬 곱이나 MS Value 집계는 없습니다.

정사각 입력의 원본 logit은 `Pᵀ⊙Q`입니다. 두 전치 사이에 원소 곱이 있어 상쇄되지 않습니다. SSAI 1은 같은 위치의 `P⊙Q`로 바꿉니다. 직사각 입력에는 reshape 재배열까지 포함되어 단순 전치 공식으로 설명하지 않습니다.

공간 softmax의 평균은 각 K채널에서 정확히 `1/(H×W)`입니다. SSAI 2는 학습 면적 대비 현재 면적을 곱하고, SSAI 3은 원본 전치를 유지한 고정 크기 sigmoid로 바꿉니다. K4는 내부 특징 차원이며 영상 공간 4배 확대와 별개입니다.

## 실험 표

[판정 기준 v2.3](../../docs/judgement_criteria.md)와 [현재 표 판정](../../experiments/performance/47_ground_truth_criteria_reevaluation/results/summary_v2_3.json)을 반영했습니다. 단일 시드도 모든 항목을 통과하고 RR 이득이 잡음 문턱(0.06dB) 이상이면 개선으로 표시하며, 여러 시드는 평균으로 판정합니다. 아래 표는 QB 결과이고, 개별 보고서의 원래 판정은 당시 기록으로 보존합니다. QNR·Dλ·Ds는 참고 지표입니다.

| 실험 | 변경한 내용 | 확인된 결과 | 상태 | 개선 여부 |
|---|---|---|:---:|:---:|
| [1. 공간 좌표 정렬](../../experiments/ssai/1_spatial_alignment/README.md)<br>[이병현] | 원본의 두 공간 전치를 제거해 같은 위치의 PAN·MS 특징을 결합했습니다. | QB seed42: A1 부분 개선(잡음 수준). 판정 항목을 모두 통과했지만 RR ΔPSNR +0.055dB가 단일 시드 잡음 문턱 0.06dB 미만입니다. ΔSAM -0.035°. | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [2. 기준 면적 정규화](../../experiments/ssai/2_reference_area_normalization/README.md)<br>[이병현] | 원본 SSAI 출력에 현재 면적/학습 기준 면적을 곱했습니다. | QB seed42: R1 개선 없음. RR ΔPSNR -0.000dB·ΔSAM +0.001°. RR 주 판정 미달입니다. | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [3. 기준 크기 sigmoid 게이트](../../experiments/ssai/3_calibrated_sigmoid_gating/README.md)<br>[이병현] | 원본 전치를 유지하고 공간 softmax를 기준 크기 sigmoid 게이트로 교체했습니다. | QB seed42: A2 부분 개선. RR ΔPSNR +0.023dB·ΔSAM -0.025°. RR2 미달입니다. | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [4. 공간 좌표 정렬과 sigmoid 게이트 결합](../../experiments/ssai/4_aligned_sigmoid_gating/README.md)<br>[이병현] | 좌표 정렬과 기준 크기 sigmoid를 결합하고 네 후보를 동일 조건으로 비교합니다. | QB seed42: A1·A2·A3 개선 없음. RR 주 판정 미달입니다. | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [5. softmax와 sigmoid 혼합 게이트](../../experiments/ssai/5_mixed_gating/README.md)<br>[이병현] | 원본 좌표에서 softmax와 기준 크기 sigmoid를 고정 비율로 혼합합니다. | QB seed42: M25·M50 개선 없음. RR 주 판정 미달입니다. | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [6. MS 조건부 공간 세부정보 보정](../../experiments/ssai/6_ms_conditioned_detail/README.md)<br>[이병현] | MS가 선택한 PAN 세부정보를 원본 출력에 잔차 보정합니다. | QB seed42: D6 개선 없음. RR ΔPSNR -0.002dB·ΔSAM +0.004°. RR 주 판정 미달입니다. | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [7. MS–PAN 채널 관계 기반 세부정보 보정](../../experiments/ssai/7_channel_relation_detail/README.md)<br>[이병현] | 4×4 MS–PAN 채널 관계로 PAN 세부정보를 선택·보정합니다. | QB seed42: D7 개선 없음. RR ΔPSNR -0.006dB·ΔSAM +0.016°. RR 주 판정 미달입니다. | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [8. 공간 가중치 평균 정규화](../../experiments/ssai/8_spatial_mean_normalization/README.md)<br>[이병현] | 원본 공간 softmax 평균을1로 맞추고 고정·학습 gamma를 비교합니다. | QB seed42: N2 부분 개선; N1 개선 없음. RR 주 판정 미달입니다. | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [9. PAN 재구성 오차 기반 보정](../../experiments/ssai/9_pan_innovation_correction/README.md)<br>[이병현] | PAN 직접 고주파와 PAN 재구성 오차를 같은 크기의 SSAI에서 비교했습니다. | QB seed42: A1·A2 개선 없음. RR 주 판정 미달입니다. | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [10. 국소 창 교차 어텐션 SSAI](../../experiments/ssai/10_local_window_cross_attention/README.md)<br>[이병현] | 전역 공간 softmax를 7×7 국소 창 어텐션으로 바꾸고 동일 파라미터 convolution 대조군과 비교합니다. | 같은 에폭 A0 대비 validation: L2 −0.81%(21에폭), C −1.15%(45에폭); test 미평가 | 학습&nbsp;중단<br>10-16&nbsp;01:20 | 판단&nbsp;보류 |
| [11. SSAI 경로 제거 분해와 PAN 기저 결합 상한·FR 대리 지표 검증](../../experiments/ssai/11_ssai_contribution_headroom/README.md)<br>[이병현] | 학습 없이 A0 best에서 SSA 출력·SSAI 단계·coarse 경로·backbone 잔차를 제거하고, PAN 기저 결합 oracle 상한과 validation FR 대리 지표를 검증했습니다. | SSA 출력 제거 시 RR·FR 동일; SSAI 단계 제거 PSNR −1.55dB(상수 보정); oracle null 초과 설명량 11.3%; 대리 Ds 순위 상관 −0.18 | 사전&nbsp;검증&nbsp;완료<br>10-16&nbsp;03:02 | 성능&nbsp;비교&nbsp;아님 |
| [12. 관측 통계로 정합한 밴드별 PAN 입력](../../experiments/ssai/12_observation_matched_pan_guide/README.md)<br>[이병현] | SSAI 8번 N1 블록에서 PAN 입력만 관측 LR MS 밴드의 국소 평균·표준편차로 정합한 PAN으로 바꾸고, 원 PAN(N1)·장면 전체 통계 정합과 비교합니다. | QB seed42: B3 부분 개선(잡음 수준, RR ΔPSNR +0.050dB); B2 개선 없음. B3는 진단 후보이며 최종 채택을 뜻하지 않습니다. | 평가&nbsp;완료 | 부분&nbsp;개선 |

## 비교 조건과 한계

A0·A1은 같은 초기값·시드·샘플 순서와 같은 micro batch 경로로 비교했습니다. 처음 3에폭은 micro4, 이후 micro32·GPU cache로 재개했고 유효 배치32·FP32는 유지했습니다. A2는 처음부터 micro32여서 SSAI만의 효과로 확정하지 않습니다. 현행 최적화는 동시 2개, 실측 합산346.3patch/s·학습 중 GPU99%·VRAM 약7.2/8GiB였습니다. OOM 자동 감축·여유 VRAM 하한 제어는 없으며 이 수치는 실측 상태입니다.

A1의 단일 시드 RR·잠정 FR 동반 개선은 반복 검증과 기준 모델 채택을 대신하지 않습니다. 기존 test는 탐색에 사용됐으며 QNR·Q2n·SCC MATLAB parity는 미검증입니다. R1은64×64 validation에서 원본과 같아 test로 계수나 후보를 고르지 않는 사전 고정 진단입니다.

## 선행 연구

[성능 개선9 · SSAI 융합](../performance/9_ssai_fusion/README.md), [성능 개선12 · 특징차원](../performance/12_feature_dimension/README.md), [성능 개선17 · SSAI 단계](../performance/17_structural_ablation/README.md)를 참고합니다. 기존 원본 자료와 번호를 보존합니다.

새 SSAI 결과는 `SSA-MRN/experiments/ssai/<번호>_<변경명>/`에 모으며 계열 내 번호를 1부터 독립적으로 사용합니다. 보고서·catalog·manifest는 계열과 번호를 함께 표시합니다. 새 실험 등록 전에 최신 main과 열린 PR을 확인합니다.



4번의 네 후보는 처음부터 동일 micro32·FP32 조건으로 각 100에폭 학습하고 RR·FR 각 20장면을 평가했습니다. 현재 v2.3에서는 세 후보 모두 RR 주 지표가 나빠져 개선 없음이며, 당시 validation 후속 기준도 통과하지 못했습니다. 원본 기준 모델을 유지합니다. 실제 best/latest 8개와 학습 소스·설정·전체 숫자·그래프·고정 예시를 4번 폴더에 보관했습니다.

5번은 신규 혼합 후보 두 개만100에폭 학습하고 실험4의 동일조건 A0·A2를 참조했습니다. 현재 v2.3에서는 M25·M50 모두 개선 없음이며, 당시 validation 기준도 통과하지 못했습니다. 새80개 RR/FR 예측·실제 best/latest4개·학습소스·설정·그래프·고정 예시를5번에 보관했습니다. 원본 기준 모델을 유지합니다.

6·7번은원본SSA기반의MS조건부세부정보보정을각100에폭학습했습니다. 7번validation은0.116%좋아졌지만RR주요지표·잠정FR은악화했고두실험모두개선없음입니다. 실제신규80개예측·best/latest4개·원본소스/설정·그래프·사전고정예시를보존하고원본을유지합니다. 다음우선검토는1번좌표정렬의반복시드재현성이며새학습은시작하지않았습니다.

9번은PAN 직접 고주파와PAN 재구성 오차 보정을각100에폭 학습하고 A0·A1·A2의RR20·FR20 전체를 같은 환경에서 평가했습니다. 두 후보 모두SSAI 가중치가 크게 변했지만 주요RR 지표와잠정QNR이 악화해 개선없음입니다. 원본A0를 유지합니다.

12번은 SSA 경로가 쓰이는 8번 N1 블록에서 PAN 입력만 관측 LR MS 통계로 밴드별 정합한 PAN으로 바꾸는 계획입니다. FR Ds 감소를 주 가설로 두고 기존 A0·N1을 대조군으로 재사용합니다. 당시 B2의 Ds 감소 가설은 기각됐습니다. 현재 v2.3에서 B2는 RR 주 판정 미달로 개선 없음이고, 진단용 B3는 RR·보조 판정을 모두 통과했지만 이득(+0.050dB)이 잡음 문턱 미만이라 부분 개선입니다. 원본 A0를 유지하며 반복 시드는 진행하지 않습니다.
