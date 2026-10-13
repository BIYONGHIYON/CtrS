# SSAI 내부 융합 연산 변경

**변경 내용:** SSAI의 정렬·정규화·국소 융합 연산을 네 구조로 비교했습니다.

| 확인된 결과 | 상태 | 개선 여부 |
|---|---|---|
| 4개 모델의 RR·FR 각 20장 평가를 완료했습니다. | 평가 완료 | **부분 개선** |

## 1. 목적과 상태

SSAI의 정렬·정규화·국소 융합 연산을 네 구조로 비교했습니다. 보관된 validation 최적 가중치로 2026-10-12 전체 test 평가를 완료했습니다. 새 학습이나 가중치 변경 없이 같은 CPU FP32 코드로 비교했습니다.

## 2. 변경한 구조와 비교 조건

A0은 기존 전치와 공간 softmax를 사용했습니다. A1은 두 공간 전치를 제거했습니다. A2는 A1의 softmax를 sigmoid로 바꿨습니다. A3은 MS Query·PAN Key/Value의 5×5 국소 교차 어텐션을 사용했습니다. A3의 행 단위 메모리 분할을 유지했습니다. A3은 Value projection도 추가하므로 국소성 하나만의 효과로 해석하지 않습니다.

QB, K6, seed 46, 100에폭, Adam lr=1e-4, MSE, effective batch 32, micro batch 16, AMP 학습입니다. A0/A1과 A2/A3을 각각 병렬 학습했으며 원본 validation MSE로 best를 선택했습니다.

RR·FR은 각각 원본 H5의 20개 전체 장면을 사용했습니다. 입력 정규화는 QB·WV3 2047, GF2 1023입니다. 예측 배열을 자르거나 clipping하지 않았습니다. 모든 후보는 센서별 동일 H5를 사용했습니다.

PSNR은 공통 평가 코드의 밴드별 평균·peak 2047 정의입니다. GF2도 이 정의를 사용하므로 peak 1023 보고서와 절댓값을 직접 비교하지 않습니다. validation의 peak 1 PSNR과도 구분합니다. QNR·Q2n·SCC는 MATLAB 구현과의 정합이 확인되지 않았으며, FR 수치는 잠정 평가입니다.

## 3. 정량 결과

각 행은 20장 평균이며 가중치·H5 해시와 장면별 지표는 [평가 원본](results/test_evaluation)에 있습니다.

| 모델 | 센서 | best 저장 epoch | RR PSNR ↑ | SAM ↓ | ERGAS ↓ | SCC ↑ | Q2n ↑ | FR Dλ ↓ | Ds ↓ | 잠정 QNR ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | QB | 99 | 37.447601 | 4.914212 | 4.108637 | 0.977233 | 0.923369 | 0.050370 | 0.044747 | 0.907367 |
| A1 | QB | 99 | 37.412875 | 4.908364 | 4.126385 | 0.976726 | 0.922969 | 0.042074 | 0.039862 | 0.920068 |
| A2 | QB | 99 | 37.443710 | 4.922966 | 4.113752 | 0.976955 | 0.923151 | 0.038033 | 0.045543 | 0.918606 |
| A3 | QB | 99 | 37.376176 | 4.936767 | 4.141695 | 0.976533 | 0.922434 | 0.049909 | 0.044468 | 0.908109 |

## 4. 기준선 대비 차이와 해석

차이는 후보−A0입니다.

| 비교 | ΔPSNR ↑ | ΔSAM ↓ | ΔERGAS ↓ | Δ잠정 QNR ↑ |
|---|---:|---:|---:|---:|
| A1 | -0.034726 | -0.005848 | +0.017748 | +0.012701 |
| A2 | -0.003891 | +0.008754 | +0.005116 | +0.011239 |
| A3 | -0.071425 | +0.022556 | +0.033059 | +0.000742 |

A1은 20장 중 PSNR 5장, SAM 15장, 잠정 QNR 20장에서 개선됐습니다.

A2은 20장 중 PSNR 9장, SAM 5장, 잠정 QNR 17장에서 개선됐습니다.

A3은 20장 중 PSNR 2장, SAM 1장, 잠정 QNR 16장에서 개선됐습니다.

지표의 방향과 장면별 차이를 함께 판단합니다. 단일 시드의 작은 차이를 통계적으로 확정된 개선으로 해석하지 않습니다. 기존 test는 탐색에 사용됐으므로 독립 최종 holdout으로 주장하지 않습니다.

## 5. 학습·평가 그래프

![실제 학습·validation 기록](results/pair_A0_A1/validation_curves.png)

![실제 학습·validation 기록](results/pair_A2_A3/validation_curves.png)

![전체 test 평균 비교](results/test_evaluation/comparison.png)

학습 그래프는 보존된 실제 이력입니다. test 그래프는 이번 평가의 20장 평균입니다.

## 6. 결과 이미지

고정 RR 장면 인덱스 1·8·11·13·19를 모든 모델에 동일하게 사용했습니다. 패널은 **LR MS·PAN·예측·GT** 순서이며 전체 장면을 추론했습니다. MS만 표시를 위해 nearest 확대했습니다. RGB 순서는 가정이며 QB·GF2는 [2,1,0], WV3는 [4,2,1]의 0부터 시작하는 밴드 인덱스입니다. 각 장면의 GT 1-99% 대비를 MS·예측·GT에 공유했고 PAN은 별도 대비를 적용했습니다. FR에는 GT를 만들지 않았습니다.

<details>
<summary>A0: 고정 5장면</summary>

![A0 RR 장면 1](results/test_evaluation/A0/scene_01.png)

![A0 RR 장면 8](results/test_evaluation/A0/scene_08.png)

![A0 RR 장면 11](results/test_evaluation/A0/scene_11.png)

![A0 RR 장면 13](results/test_evaluation/A0/scene_13.png)

![A0 RR 장면 19](results/test_evaluation/A0/scene_19.png)

</details>

<details>
<summary>A1: 고정 5장면</summary>

![A1 RR 장면 1](results/test_evaluation/A1/scene_01.png)

![A1 RR 장면 8](results/test_evaluation/A1/scene_08.png)

![A1 RR 장면 11](results/test_evaluation/A1/scene_11.png)

![A1 RR 장면 13](results/test_evaluation/A1/scene_13.png)

![A1 RR 장면 19](results/test_evaluation/A1/scene_19.png)

</details>

<details>
<summary>A2: 고정 5장면</summary>

![A2 RR 장면 1](results/test_evaluation/A2/scene_01.png)

![A2 RR 장면 8](results/test_evaluation/A2/scene_08.png)

![A2 RR 장면 11](results/test_evaluation/A2/scene_11.png)

![A2 RR 장면 13](results/test_evaluation/A2/scene_13.png)

![A2 RR 장면 19](results/test_evaluation/A2/scene_19.png)

</details>

<details>
<summary>A3: 고정 5장면</summary>

![A3 RR 장면 1](results/test_evaluation/A3/scene_01.png)

![A3 RR 장면 8](results/test_evaluation/A3/scene_08.png)

![A3 RR 장면 11](results/test_evaluation/A3/scene_11.png)

![A3 RR 장면 13](results/test_evaluation/A3/scene_13.png)

![A3 RR 장면 19](results/test_evaluation/A3/scene_19.png)

</details>

## 7. 재현 자료

[가중치](checkpoints) · [학습 실행 코드](run.py) · [평가 원본](results/test_evaluation) · [학습 당시 소스](provenance) · [파일 해시](manifest.json)

공식 submodule을 초기화하고 아래 명령으로 원본 테스트 H5가 있는 폴더를 지정합니다. 완료 JSON은 덮어쓰지 않고 건너뜁니다. 다시 평가하려면 기존 결과를 보관하고 별도 checkout에서 실행합니다.

```bash
python SSA-MRN/common/evaluate_retained.py --experiment 9 --data /path/to/test_h5
```

공통 모델·지표·평가 코드 해시는 `results/test_evaluation/evaluation_source.json`에 보관했습니다. 체크포인트 내부 설정과 원본 best/latest는 변경하지 않았습니다.

## 8. 개선 여부와 다음 단계

**판정: 부분 개선.** 후보별·지표별 방향을 위 표로 구분합니다. 기존 A0를 대체하는 최종 채택은 반복 시드와 현재 K4 기준선의 독립 비교 이후 판단합니다.
