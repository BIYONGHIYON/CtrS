# MTF 변화량 데이터 증강

**변경 내용:** 센서 열화 특성의 변화량을 MS·LMS 학습 입력에 추가했습니다.

| 확인된 결과 | 상태 | 개선 여부 |
|---|---|---|
| 2개 모델의 RR·FR 각 20장 평가를 완료했습니다. | 평가 완료 | **부분 개선** |

## 1. 목적과 상태

센서 열화 특성의 변화량을 MS·LMS 학습 입력에 추가했습니다. 보관된 validation 최적 가중치로 2026-10-12 전체 test 평가를 완료했습니다. 새 학습이나 가중치 변경 없이 같은 CPU FP32 코드로 비교했습니다.

## 2. 변경한 구조와 비교 조건

baseline은 원본 입력을 사용했고 mtf_aug는 QB GNyq [0.34,0.32,0.30,0.22]에 0.9 또는 1.1 배율을 적용한 41탭 열화 변화량을 추가했습니다. LR 경계 5픽셀을 유지하고 LMS에는 interp23 변화량을 반영했습니다. PAN·GT와 추론 구조는 그대로 유지했습니다.

QB, K6, seed 46, 100에폭, Adam lr=1e-4, MSE, effective batch 32, micro batch 16, AMP 학습입니다. 원본 validation MSE로 선택한 best를 평가했습니다.

RR·FR은 각각 원본 H5의 20개 전체 장면을 사용했습니다. 입력 정규화는 QB·WV3 2047, GF2 1023입니다. 예측 배열을 자르거나 clipping하지 않았습니다. 모든 후보는 센서별 동일 H5를 사용했습니다.

PSNR은 공통 평가 코드의 밴드별 평균·peak 2047 정의입니다. GF2도 이 정의를 사용하므로 peak 1023 보고서와 절댓값을 직접 비교하지 않습니다. validation의 peak 1 PSNR과도 구분합니다. QNR·Q2n·SCC는 MATLAB 구현과의 정합이 확인되지 않았으며, FR 수치는 잠정 평가입니다.

## 3. 정량 결과

각 행은 20장 평균이며 가중치·H5 해시와 장면별 지표는 [평가 원본](results/test_evaluation)에 있습니다.

| 모델 | 센서 | best 저장 epoch | RR PSNR ↑ | SAM ↓ | ERGAS ↓ | SCC ↑ | Q2n ↑ | FR Dλ ↓ | Ds ↓ | 잠정 QNR ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | QB | 99 | 37.428742 | 4.911522 | 4.119721 | 0.976853 | 0.922347 | 0.047945 | 0.043476 | 0.911027 |
| mtf_aug | QB | 99 | 37.422342 | 4.905436 | 4.120740 | 0.976815 | 0.922008 | 0.048824 | 0.044113 | 0.909617 |

## 4. 기준선 대비 차이와 해석

차이는 후보−baseline입니다.

| 비교 | ΔPSNR ↑ | ΔSAM ↓ | ΔERGAS ↓ | Δ잠정 QNR ↑ |
|---|---:|---:|---:|---:|
| mtf_aug | -0.006401 | -0.006086 | +0.001019 | -0.001410 |

mtf_aug은 20장 중 PSNR 9장, SAM 14장, 잠정 QNR 5장에서 개선됐습니다.

지표의 방향과 장면별 차이를 함께 판단합니다. 단일 시드의 작은 차이를 통계적으로 확정된 개선으로 해석하지 않습니다. 기존 test는 탐색에 사용됐으므로 독립 최종 holdout으로 주장하지 않습니다.

## 5. 학습·평가 그래프

![실제 학습·validation 기록](results/validation_curves.png)

![전체 test 평균 비교](results/test_evaluation/comparison.png)

학습 그래프는 보존된 실제 이력입니다. test 그래프는 이번 평가의 20장 평균입니다.

## 6. 결과 이미지

고정 RR 장면 인덱스 1·8·11·13·19를 모든 모델에 동일하게 사용했습니다. 패널은 **LR MS·PAN·예측·GT** 순서이며 전체 장면을 추론했습니다. MS만 표시를 위해 nearest 확대했습니다. RGB 순서는 가정이며 QB·GF2는 [2,1,0], WV3는 [4,2,1]의 0부터 시작하는 밴드 인덱스입니다. 각 장면의 GT 1-99% 대비를 MS·예측·GT에 공유했고 PAN은 별도 대비를 적용했습니다. FR에는 GT를 만들지 않았습니다.

<details>
<summary>baseline: 고정 5장면</summary>

![baseline RR 장면 1](results/test_evaluation/baseline/scene_01.png)

![baseline RR 장면 8](results/test_evaluation/baseline/scene_08.png)

![baseline RR 장면 11](results/test_evaluation/baseline/scene_11.png)

![baseline RR 장면 13](results/test_evaluation/baseline/scene_13.png)

![baseline RR 장면 19](results/test_evaluation/baseline/scene_19.png)

</details>

<details>
<summary>mtf_aug: 고정 5장면</summary>

![mtf_aug RR 장면 1](results/test_evaluation/mtf_aug/scene_01.png)

![mtf_aug RR 장면 8](results/test_evaluation/mtf_aug/scene_08.png)

![mtf_aug RR 장면 11](results/test_evaluation/mtf_aug/scene_11.png)

![mtf_aug RR 장면 13](results/test_evaluation/mtf_aug/scene_13.png)

![mtf_aug RR 장면 19](results/test_evaluation/mtf_aug/scene_19.png)

</details>

## 7. 재현 자료

[가중치](checkpoints) · [학습 실행 코드](run.py) · [평가 원본](results/test_evaluation) · [학습 당시 소스](provenance) · [파일 해시](manifest.json)

공식 submodule을 초기화하고 아래 명령으로 원본 테스트 H5가 있는 폴더를 지정합니다. 완료 JSON은 덮어쓰지 않고 건너뜁니다. 다시 평가하려면 기존 결과를 보관하고 별도 checkout에서 실행합니다.

```bash
python SSA-MRN/common/evaluate_retained.py --experiment 8 --data /path/to/test_h5
```

공통 모델·지표·평가 코드 해시는 `results/test_evaluation/evaluation_source.json`에 보관했습니다. 체크포인트 내부 설정과 원본 best/latest는 변경하지 않았습니다.

## 8. 개선 여부와 다음 단계

**판정: 부분 개선.** SAM만 소폭 낮아졌으며 PSNR·ERGAS·잠정 QNR은 악화됐습니다. validation MSE 개선이 RR·FR의 일관된 개선으로 이어지지 않았습니다. 현재 기준 모델에 증강을 채택할 근거는 부족합니다.
