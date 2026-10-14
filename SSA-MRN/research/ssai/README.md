# SSAI 개선

[전체 연구](../../../README.md) · [실행·등록 안내](../../README.md)

**담당·실험자: 이병현**

SSAI의 PAN 공간정보와 MS 분광정보를 결합하는 연산을 개선합니다. 기준 모델은 **원본 SSA-MRN·K4·MSE·단일복원·공간4배확대**로 유지합니다.

## 현재 결과

SSAI 1-3의QB seed42 결과를 보관했습니다. A0 원본·A1 좌표 정렬·A2 sigmoid는 각 100에폭 학습했고, R1 면적 보정은A0의 같은 best를 사용했습니다. validation으로 고정한 98에폭 best에서 모든 연산의 RR·FR 각 20개 전체 장면을 평가했습니다. 기존 세 실험의 원본 결과는 유지합니다. 4번에서는 새로운 seed42 대조군과 세 후보를 동일 micro32 조건으로 처음부터 비교합니다.

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
| [4. 공간 좌표 정렬과 sigmoid 게이트 결합](../../experiments/ssai/4_aligned_sigmoid_gating/README.md)<br>[이병현] | 좌표 정렬과 기준 크기 sigmoid를 결합하고 네 후보를 동일 조건으로 비교합니다. | A0·A1 학습 중, A2·A3 대기 | 학습&nbsp;중<br>2026-10-14 10:34 KST | 판단&nbsp;보류 |

## 비교 조건과 한계

A0·A1은 같은 초기값·시드·샘플 순서와 같은 micro batch 경로로 비교했습니다. 처음 3에폭은 micro4, 이후 micro32·GPU cache로 재개했고 유효 배치32·FP32는 유지했습니다. A2는 처음부터 micro32여서 SSAI만의 효과로 확정하지 않습니다. 현행 최적화는 동시 2개, 실측 합산346.3patch/s·학습 중 GPU99%·VRAM 약7.2/8GiB였습니다. OOM 자동 감축·여유 VRAM 하한 제어는 없으며 이 수치는 실측 상태입니다.

A1의 단일 시드 RR·잠정 FR 동반 개선은 반복 검증과 기준 모델 채택을 대신하지 않습니다. 기존 test는 탐색에 사용됐으며 QNR·Q2n·SCC MATLAB parity는 미검증입니다. R1은64×64 validation에서 원본과 같아 test로 계수나 후보를 고르지 않는 사전 고정 진단입니다.

## 선행 연구

[성능 개선9 · SSAI 융합](../performance/9_ssai_fusion/README.md), [성능 개선12 · 특징차원](../performance/12_feature_dimension/README.md), [성능 개선17 · SSAI 단계](../performance/17_structural_ablation/README.md)를 참고합니다. 기존 원본 자료와 번호를 보존합니다.

새 SSAI 결과는 `SSA-MRN/experiments/ssai/<번호>_<변경명>/`에 모으며 계열 내 번호를 1부터 독립적으로 사용합니다. 보고서·catalog·manifest는 계열과 번호를 함께 표시합니다. 새 실험 등록 전에 최신 main과 열린 PR을 확인합니다.

4번의 새 실행은 독립 데이터 캐시를 사용합니다. 여섯 병렬 후보 조합은 최소 VRAM 여유 776MiB로 사전 검증됐으며, 작업 시작 전 512MiB 여유를 검사합니다. 현재 반복 seed43·44와 센서 확장은 시작하지 않았습니다.
