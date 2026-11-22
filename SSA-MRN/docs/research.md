# 팬샤프닝과 SSA-MRN 설명

[연구 소개](../../README.md) · [실험 표](../experiments/performance/README.md) · [판정 기준](judgement_criteria.md)

## 문제

고해상도 흑백 영상 **PAN**은 공간 구조를, 저해상도 다중분광 영상 **MS**는 밴드별 색 정보를 줍니다. 팬샤프닝은 둘을 합쳐 고해상도 MS를 복원합니다. QB·GF2는 4밴드, WV3·WV2는 8밴드이고 공간 배율은 4배입니다.

## SSA-MRN의 처리 흐름

1. PAN, 4배 확대한 MS(LMS), 저해상도 MS를 입력합니다.
2. PAN의 축소·재확대 영상과 LMS를 결합해 초기 특징을 만듭니다.
3. 각 MS 밴드의 특징과 PAN 특징을 SSAI에서 결합합니다.
4. 전체·1/2·1/4 크기에서 처리한 특징을 작은 크기부터 합쳐 고해상도 MS를 냅니다.

- **K**는 SSAI 내부 투영 채널 수이며 밴드 수와 무관합니다. 2026-10-09부터 기본값은 K=4입니다([근거](../experiments/performance/12_feature_dimension/README.md)). 과거 K=6 실험은 K=4 결과로 바꿔 표기하지 않습니다.
- 공개 코드의 SSAI는 원소별 곱·전치·softmax로 구현돼 있어 논문의 행렬곱 설명과 같다고 보지 않습니다. 공식 코드는 [하위 모듈](../references/README.md)로 고정하고 재현 래퍼의 변경을 따로 관리합니다.
- 기준 모델 SMOC가 원본에 더한 부분은 [연구 소개](../../README.md#기준-모델)에 있습니다.

## 평가

| 시험 | 내용 | 지표 |
|---|---|---|
| RR | 정답이 있는 축소 해상도 test 20장 | PSNR·SAM·ERGAS (판정), SCC·Q2n (참고) |
| FR | 정답이 없는 원래 해상도 test 20장 | QNR·Dλ·Ds (참고) |
| WV2 전이 | WV3로 학습한 모델을 학습하지 않은 WV2에 적용 | RR 지표 (참고) |

- 학습·validation·test를 분리하고 test 수치로 가중치나 설정을 고르지 않습니다.
- RR 지표 구현(`common/metrics`)은 공식 DLPan MATLAB 재평가와 PSNR·SAM·ERGAS·Q4가 소수점 셋째 자리까지 같았습니다([값](c2ms_matlab_evaluation_tables.md)).
- FR의 PAN 축소는 MATLAB이 아닌 scikit-image cubic을 써서 Ds·QNR은 잠정 값입니다.
- 결과 이미지는 입력·예측·정답에 같은 대비를 적용해 표시하고, 지표는 표시 전 값으로 계산합니다.
