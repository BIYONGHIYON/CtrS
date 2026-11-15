# SSAI 개선 실험

[연구 소개](../../../README.md) · [실행 안내](../../README.md) · [판정 기준](../../docs/judgement_criteria.md)

PAN 공간 정보와 MS 분광 정보를 결합하는 SSAI 연산을 바꾸는 연구입니다. **2026-10-16 종료**했으며 새 실험은 [성능 개선](../performance/README.md)에 등록합니다. 대조군은 원본 SSA-MRN(K=4, MSE, 공간 4배)입니다.

## 결론

| 확인한 사실 | 근거 실험 |
|---|---|
| test까지 평가한 후보의 대조군 대비 RR PSNR 변화는 −0.071에서 +0.055 dB 사이였습니다. 같은 조건 대조군을 두 번 학습한 차이가 0.036 dB입니다. | 1-9, 12 |
| 1번 A1(+0.055 dB)과 12번 B3(+0.050 dB)는 세 지표가 모두 좋아졌지만 잡음 문턱(0.06 dB) 미만이라 부분 개선입니다. 1번의 이득은 4번 재비교에서 재현되지 않았습니다. | 1, 4, 12 |
| 대조군에서 SSA 출력을 0으로 바꿔도 RR·FR 지표가 같았습니다. SSAI 단계 전체를 제거하면 RR PSNR이 1.55 dB 떨어졌습니다. | 11 |
| SSA 경로를 실제로 쓰게 만든 8번 N1·9번·12번 B2는 개선 없음입니다. | 8, 9, 12 |
| 대조군 오차를 PAN 국소 선형 결합으로 줄일 수 있는 이상적 상한은 약 0.52 dB였습니다. | 11 |

<details>
<summary>한계와 SSAI 연산 설명</summary>

- 공개 코드 기반 SSAI 구현, QB 단일 시드, 탐색에 반복 사용한 test에 한정한 결론입니다. 큰 용량의 비선형 결합이나 다른 센서·배율은 확인하지 않았습니다.
- 논문 식 (10)-(12)의 K×K 채널 상관 SSAI는 [성능 34번](../performance/34_paper_equation_reproduction/README.md)에서 다른 조건과 함께 바뀐 형태로만 확인했습니다.
- 12번 B2의 FR Dλ 개선은 test 결과를 본 뒤 관찰한 것이라, 이어 보려면 새 시드를 쓰는 별도 가설로 등록해야 합니다.
- **연산:** 각 밴드는 PAN과 MS 특징을 K채널로 바꿔 곱해 공간 가중치를 만들고 PAN 특징을 가중합니다. 밴드별 출력을 모아 합성곱으로 혼합하며 전체·절반·1/4 해상도에 적용합니다. 공간 softmax의 평균은 각 채널에서 `1/(H×W)`입니다. K는 내부 특징 차원이며 공간 4배 확대와 별개입니다.
- **선행 실험:** [성능 9번](../performance/9_ssai_fusion/README.md), [12번](../performance/12_feature_dimension/README.md), [17번](../performance/17_structural_ablation/README.md)

</details>

<a id="experiments"></a>

## 실험 표

<!-- experiments:start -->
| 실험 | 핵심 결과 | 상태 | 판정 |
|---|---|:---:|:---:|
| [12. 관측 통계로 정합한 밴드별 PAN 입력](12_observation_matched_pan_guide/README.md)<br>이병현 | B3가 QB seed 42 +0.050 dB로 잡음 수준, B2는 개선 없음 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [11. SSAI 경로 제거 분해와 PAN 기저 결합 상한·FR 대리 지표 검증](11_ssai_contribution_headroom/README.md)<br>이병현 | SSA 출력을 지워도 RR·FR이 같음. SSAI 단계 제거는 −1.55 dB(학습 없는 분석) | 사전&nbsp;검증&nbsp;완료 | 성능&nbsp;비교&nbsp;아님 |
| [10. 국소 창 교차 어텐션 SSAI](10_local_window_cross_attention/README.md)<br>이병현 | 같은 에폭 A0 대비 validation −0.81%·−1.15%에서 학습 중단, test 평가 없음 | 학습&nbsp;중단 | 판단&nbsp;보류 |
| [9. PAN 재구성 오차 기반 보정](9_pan_innovation_correction/README.md)<br>이병현 | QB seed 42에서 A1·A2 개선 없음 | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [8. 공간 가중치 평균 정규화](8_spatial_mean_normalization/README.md)<br>이병현 | QB seed 42에서 N2 부분 개선, N1 개선 없음 | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [7. MS–PAN 채널 관계 기반 세부정보 보정](7_channel_relation_detail/README.md)<br>이병현 | QB seed 42 −0.006 dB·SAM +0.016° | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [6. MS 조건부 공간 세부정보 보정](6_ms_conditioned_detail/README.md)<br>이병현 | QB seed 42 −0.002 dB·SAM +0.004° | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [5. softmax와 sigmoid 혼합 게이트](5_mixed_gating/README.md)<br>이병현 | QB seed 42에서 M25·M50 개선 없음 | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [4. 공간 좌표 정렬과 sigmoid 게이트 결합](4_aligned_sigmoid_gating/README.md)<br>이병현 | QB seed 42에서 A1·A2·A3 개선 없음 | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [3. 기준 크기 sigmoid 게이트](3_calibrated_sigmoid_gating/README.md)<br>이병현 | A2가 QB seed 42 +0.023 dB·SAM −0.025° | 평가&nbsp;완료 | 부분&nbsp;개선 |
| [2. 기준 면적 정규화](2_reference_area_normalization/README.md)<br>이병현 | QB seed 42 −0.000 dB·SAM +0.001° | 평가&nbsp;완료 | 개선&nbsp;없음 |
| [1. 공간 좌표 정렬](1_spatial_alignment/README.md)<br>이병현 | A1이 QB seed 42 +0.055 dB·SAM −0.035°로 잡음 수준 | 평가&nbsp;완료 | 부분&nbsp;개선 |
<!-- experiments:end -->
