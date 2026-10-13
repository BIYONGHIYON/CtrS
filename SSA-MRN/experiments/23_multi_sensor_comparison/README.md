# 반복 복원·단계 간 특징 전달의 센서별 수렴 비교

**변경 내용:** 원본 단일 복원과 3단계 공유 복원, 단계 간 특징 전달(F0)을 세 센서에서 동일 조건으로 비교했습니다.

| 확인된 결과 | 상태 | 개선 여부 |
|---|:---:|:---:|
| 9개 모델 × 150에포크와 50·100·150에포크의 RR·FR 전체 평가를 완료했습니다. F0는 원본 대비 QB·GF2에서 RR·잠정 QNR 개선, WV3에서는 지표가 엇갈렸습니다. | 평가&nbsp;완료 | 부분&nbsp;개선 |

[실험 목록](../../README.md#experiments) · [수치](#results) · [그래프](#graphs) · [이미지](#images) · [재현 자료](#evidence)

## 1. 목적과 상태

반복 복원과 단계 간 특징 전달의 효과가 장기 학습과 다른 센서에서도 유지되는지 확인했습니다. 실험자는 이병현입니다. 서버의 완료 기록은 **2026-10-13 16:47:50 KST**입니다. 9개 모델의 150개 연속 에포크 기록, 세 점검 시점의 평가 완료와 장면별 지표를 확인했습니다. 총 1,350모델에포크와 1,080회 장면 평가입니다.

## 2. 변경한 구조와 비교 조건

| 모델 | 입력·연산·출력 변경 | 대조군 |
|---|---|---|
| original | PAN·LMS·MS를 기존 밴드별 다중 크기 SSAI에 넣어 한 번 복원했습니다. | 원본 |
| repeat | 같은 복원기를 3회 공유했습니다. LMS에서 시작하고 `current += alpha * (proposal - current)`로 출력을 갱신했습니다. | original |
| F0 | repeat에 전체·절반·1/4 크기의 이전 단계 특징을 1×1 projection과 학습형 gate로 더했습니다. 특징은 한 forward 안에서 전달하고 다음 배치에서 초기화했습니다. | original·repeat |

| 항목 | 실제 조건 |
|---|---|
| 센서·밴드 | QB·GF2 4밴드, WV3 8밴드 |
| 데이터 | 센서별 기존 전체 train/validation과 독립 RR·FR H5를 유지했습니다. RR·FR 각 20장입니다. |
| 학습 패치 | HR 64×64, MS 16×16; 내부 특징 차원 K4 |
| 학습 | seed46, 모델별 150에포크, Adam, 최종 출력 MSE |
| 학습률 | 1-100에포크 1e-4, 101-150에포크 1e-5 |
| 배치·장치 | effective/micro 64, RTX3060 Ti 8GB, 최대 두 작업 병렬 |
| 수치·공급 | AMP FP16·GradScaler 학습, FP32 validation/test, TF32 비활성, cuDNN benchmark 활성, mmap·worker 1 |
| 통제 | 센서 안에서 core 초기값이 일치하고 같은 에포크별 샘플 순서 함수를 사용했습니다. AMP skip 수와 실제 실행 시간은 모델별로 달랐습니다. |
| 선택 | 각 50·100·150에포크까지 validation MSE 최소, 정확한 동률은 SAM으로 선택했습니다. Test로 후보·에포크를 소급 선택하지 않았습니다. |
| 정규화·평가 | QB·WV3는 2047, GF2는 1023입니다. RR PSNR은 각 센서의 peak를 반영하도록 공통 peak2047 평가기에 GT·예측을 함께 환산했습니다. 예측 crop·clipping은 하지 않았습니다. |

기존 원본 구조와 밴드별 SSAI를 유지했습니다. 입력 MS 관측 오차 피드백, 추가 PAN 차이, 23탭 확대, 추가 손실은 사용하지 않았습니다. repeat의 zero-error encoder bias와 공유 alpha는 학습됩니다. 과거 GF2의 고정 peak2047 수치와 이번 GF2의 절댓값은 직접 비교하지 않습니다.

<a id="results"></a>

## 3. 정량 결과

### 최종 150에포크까지 validation best의 전체 test

모든 수치는 RR·FR 각 20장 평균입니다. best 저장 에포크와 학습 완료 에포크는 다릅니다. QNR·Q2n·SCC의 공식 MATLAB 정합은 확인하지 않았으며 FR 수치는 잠정 평가입니다.

| 센서 | 모델 | best 에포크 | RR PSNR ↑ | SAM ↓ | ERGAS ↓ | 잠정 FR QNR ↑ |
|---|---|---:|---:|---:|---:|---:|
| QB | original | 149 | 37.212768 | 5.049509 | 4.235726 | 0.915419 |
| QB | repeat | 149 | 37.388607 | 4.957387 | 4.137474 | 0.916892 |
| QB | F0 | 149 | 37.339865 | 4.964351 | 4.165098 | 0.921646 |
| GF2 | original | 150 | 39.591910 | 1.021083 | 0.998536 | 0.904036 |
| GF2 | repeat | 147 | 40.515324 | 0.945779 | 0.896645 | 0.910206 |
| GF2 | F0 | 147 | 40.420981 | 0.956271 | 0.907641 | 0.910905 |
| WV3 | original | 149 | 37.293891 | 3.620795 | 2.665517 | 0.943960 |
| WV3 | repeat | 148 | 37.210863 | 3.607973 | 2.659393 | 0.935719 |
| WV3 | F0 | 149 | 37.303250 | 3.580074 | 2.628477 | 0.936375 |

### 50·100·150에포크의 PSNR 추이

각 시점까지 validation으로 선택한 가중치의 test 진단입니다. Test PSNR 최대 시점으로 최종 가중치를 다시 선택하지 않습니다.

| 센서 | 모델 | 50 | 100 | 150 |
|---|---|---:|---:|---:|
| QB | original | 36.408941 | 37.150187 | 37.212768 |
| QB | repeat | 36.664864 | 37.203905 | 37.388607 |
| QB | F0 | 36.579764 | 37.243620 | 37.339865 |
| GF2 | original | 38.464172 | 39.417097 | 39.591910 |
| GF2 | repeat | 39.312502 | 40.458866 | 40.515324 |
| GF2 | F0 | 39.082647 | 40.422244 | 40.420981 |
| WV3 | original | 36.627079 | 37.208410 | 37.293891 |
| WV3 | repeat | 36.845073 | 37.243240 | 37.210863 |
| WV3 | F0 | 36.898622 | 37.119049 | 37.303250 |

[전체 평균·차이 요약](results/summary.json)에 모든 시점의 RR·FR 수치를 보존했습니다. 센서별 `results/<센서>/evaluation/<시점>/metrics.json`에 전체 장면별 지표를 보존했습니다.

## 4. 기준선 대비 차이와 해석

### 원본 대비 최종 차이

| 센서 | 비교 | ΔPSNR ↑ | ΔSAM ↓ | Δ잠정 QNR ↑ | PSNR / SAM / QNR 개선 장면 |
|---|---|---:|---:|---:|---|
| QB | repeat-original | +0.175839 | -0.092123 | +0.001473 | 20/20 · 20/20 · 17/20 |
| QB | F0-original | +0.127097 | -0.085158 | +0.006227 | 19/20 · 19/20 · 20/20 |
| GF2 | repeat-original | +0.923414 | -0.075304 | +0.006170 | 20/20 · 20/20 · 19/20 |
| GF2 | F0-original | +0.829071 | -0.064811 | +0.006869 | 20/20 · 20/20 · 19/20 |
| WV3 | repeat-original | -0.083029 | -0.012822 | -0.008241 | 4/20 · 10/20 · 0/20 |
| WV3 | F0-original | +0.009358 | -0.040721 | -0.007585 | 13/20 · 12/20 · 0/20 |

### 특징 전달의 추가 효과: F0 - repeat

| 센서 | ΔPSNR ↑ | ΔSAM ↓ | Δ잠정 QNR ↑ |
|---|---:|---:|---:|
| QB | -0.048742 | +0.006964 | +0.004753 |
| GF2 | -0.094342 | +0.010492 | +0.000699 |
| WV3 | +0.092387 | -0.027899 | +0.000656 |

- **GF2:** 원본 대비 가장 큰 RR 이득을 확인했습니다. repeat는 PSNR +0.9234 dB, F0는 +0.8291 dB이며 SAM·잠정 QNR도 개선됐습니다. 특징 전달을 추가한 F0의 PSNR·SAM은 repeat보다 낮았습니다.
- **QB:** repeat와 F0 모두 원본 대비 주요 RR·잠정 QNR이 개선됐습니다. repeat의 PSNR·SAM이 더 좋고 F0의 잠정 QNR이 더 좋았습니다.
- **WV3:** F0의 PSNR은 +0.0094 dB로 작고 SAM은 개선됐지만 잠정 QNR은 -0.0076으로 하락했습니다. QNR은 모든 20장에서 원본보다 낮았습니다. repeat의 PSNR도 원본보다 낮았습니다.
- **학습 연장:** 학습률을 낮춘 100-150 구간은 후보·센서별 효과가 달랐습니다. F0의 GF2 PSNR은 거의 변하지 않았고 WV3의 RR은 회복됐습니다. 장기 학습이 항상 모든 지표를 높이지는 않았습니다.

**F0의 원본 대비 개선은 QB·GF2에서 확인했지만, 단순 반복 복원을 넘어서는 추가 RR 이득과 모든 센서에서의 우위는 확인하지 못했습니다.** 각 센서의 validation best MSE는 repeat가 가장 낮았습니다. Test 결과로 선택 기준을 변경하지 않았습니다.

<a id="graphs"></a>

## 5. 학습·평가 그래프

### QB
![QB 150에포크 실제 학습·validation 곡선](results/QB/evaluation/150/curves.png)
![QB F0 대비 최종 평가 차이](results/QB/evaluation/150/comparison.png)

### GF2
![GF2 150에포크 실제 학습·validation 곡선](results/GF2/evaluation/150/curves.png)
![GF2 F0 대비 최종 평가 차이](results/GF2/evaluation/150/comparison.png)

### WV3
![WV3 150에포크 실제 학습·validation 곡선](results/WV3/evaluation/150/curves.png)
![WV3 F0 대비 최종 평가 차이](results/WV3/evaluation/150/comparison.png)

그래프의 막대 기준은 F0이며, 원본 대비 차이는 위 표에 따로 기록했습니다. 50·100에포크 곡선과 비교 그래프도 같은 센서 결과 폴더에 보존했습니다.

<a id="images"></a>

## 6. 결과 이미지

고정 RR 장면 1·8·11·13·19를 동일하게 사용했습니다. 패널은 **LR MS · PAN · 모델 예측 · GT** 순서입니다. 전체 장면을 추론했고 입력을 자르지 않았습니다. LR MS만 nearest 확대했습니다. RGB는 QB·GF2 [2,1,0], WV3 [4,2,1]의 0부터 시작하는 밴드 인덱스를 가정했습니다. GT의 밴드별 1-99% 대비를 MS·예측·GT에 공유하고 PAN은 별도 대비를 적용했습니다. 표시용 clipping은 지표 계산과 별개입니다. FR에는 정답을 만들지 않았습니다.

### QB · 장면 1 비교

![QB original RR 장면 1](results/QB/evaluation/150/original_scene_01.png)

![QB repeat RR 장면 1](results/QB/evaluation/150/repeat_scene_01.png)

![QB F0 RR 장면 1](results/QB/evaluation/150/F0_scene_01.png)

<details>
<summary>QB: 나머지 고정 4장면과 세 모델</summary>

![QB original RR 장면 8](results/QB/evaluation/150/original_scene_08.png)

![QB repeat RR 장면 8](results/QB/evaluation/150/repeat_scene_08.png)

![QB F0 RR 장면 8](results/QB/evaluation/150/F0_scene_08.png)

![QB original RR 장면 11](results/QB/evaluation/150/original_scene_11.png)

![QB repeat RR 장면 11](results/QB/evaluation/150/repeat_scene_11.png)

![QB F0 RR 장면 11](results/QB/evaluation/150/F0_scene_11.png)

![QB original RR 장면 13](results/QB/evaluation/150/original_scene_13.png)

![QB repeat RR 장면 13](results/QB/evaluation/150/repeat_scene_13.png)

![QB F0 RR 장면 13](results/QB/evaluation/150/F0_scene_13.png)

![QB original RR 장면 19](results/QB/evaluation/150/original_scene_19.png)

![QB repeat RR 장면 19](results/QB/evaluation/150/repeat_scene_19.png)

![QB F0 RR 장면 19](results/QB/evaluation/150/F0_scene_19.png)

</details>

### GF2 · 장면 1 비교

![GF2 original RR 장면 1](results/GF2/evaluation/150/original_scene_01.png)

![GF2 repeat RR 장면 1](results/GF2/evaluation/150/repeat_scene_01.png)

![GF2 F0 RR 장면 1](results/GF2/evaluation/150/F0_scene_01.png)

<details>
<summary>GF2: 나머지 고정 4장면과 세 모델</summary>

![GF2 original RR 장면 8](results/GF2/evaluation/150/original_scene_08.png)

![GF2 repeat RR 장면 8](results/GF2/evaluation/150/repeat_scene_08.png)

![GF2 F0 RR 장면 8](results/GF2/evaluation/150/F0_scene_08.png)

![GF2 original RR 장면 11](results/GF2/evaluation/150/original_scene_11.png)

![GF2 repeat RR 장면 11](results/GF2/evaluation/150/repeat_scene_11.png)

![GF2 F0 RR 장면 11](results/GF2/evaluation/150/F0_scene_11.png)

![GF2 original RR 장면 13](results/GF2/evaluation/150/original_scene_13.png)

![GF2 repeat RR 장면 13](results/GF2/evaluation/150/repeat_scene_13.png)

![GF2 F0 RR 장면 13](results/GF2/evaluation/150/F0_scene_13.png)

![GF2 original RR 장면 19](results/GF2/evaluation/150/original_scene_19.png)

![GF2 repeat RR 장면 19](results/GF2/evaluation/150/repeat_scene_19.png)

![GF2 F0 RR 장면 19](results/GF2/evaluation/150/F0_scene_19.png)

</details>

### WV3 · 장면 1 비교

![WV3 original RR 장면 1](results/WV3/evaluation/150/original_scene_01.png)

![WV3 repeat RR 장면 1](results/WV3/evaluation/150/repeat_scene_01.png)

![WV3 F0 RR 장면 1](results/WV3/evaluation/150/F0_scene_01.png)

<details>
<summary>WV3: 나머지 고정 4장면과 세 모델</summary>

![WV3 original RR 장면 8](results/WV3/evaluation/150/original_scene_08.png)

![WV3 repeat RR 장면 8](results/WV3/evaluation/150/repeat_scene_08.png)

![WV3 F0 RR 장면 8](results/WV3/evaluation/150/F0_scene_08.png)

![WV3 original RR 장면 11](results/WV3/evaluation/150/original_scene_11.png)

![WV3 repeat RR 장면 11](results/WV3/evaluation/150/repeat_scene_11.png)

![WV3 F0 RR 장면 11](results/WV3/evaluation/150/F0_scene_11.png)

![WV3 original RR 장면 13](results/WV3/evaluation/150/original_scene_13.png)

![WV3 repeat RR 장면 13](results/WV3/evaluation/150/repeat_scene_13.png)

![WV3 F0 RR 장면 13](results/WV3/evaluation/150/F0_scene_13.png)

![WV3 original RR 장면 19](results/WV3/evaluation/150/original_scene_19.png)

![WV3 repeat RR 장면 19](results/WV3/evaluation/150/repeat_scene_19.png)

![WV3 F0 RR 장면 19](results/WV3/evaluation/150/F0_scene_19.png)

</details>

<a id="evidence"></a>

## 7. 재현 자료

- [학습·평가 코드](run.py), [세 모델 구조](model.py), [데이터 준비](prepare.py), [구조 점검](verify.py)을 보존했습니다.
- `checkpoints/<모델>_<센서>_s46/`에 최종 best/latest와 best_at_50·best_at_100·best_at_150를 보존했습니다. 9개 모델, 총 45개 가중치입니다. 이전 임시 복구용 previous는 최종 보관 대상에서 제외했습니다.
- `results/`에 150에포크 compact 이력, 초기값 확인, 완료 기록, 데이터 manifest, 27개 모델·시점 평가와 고정 이미지를 보존했습니다. 상세 로그·데이터 캐시·예측 배열은 추가하지 않았습니다.
- [45개 가중치 strict load·FP32 4/8밴드 순전파 확인](results/checkpoint_verification.json)을 완료했습니다. 최신 가중치 9개의 optimizer·GradScaler·RNG·150에포크 재개 상태도 확인했습니다. 새 전체 test 재추론은 수행하지 않았습니다.
- [실제 실행 소스](provenance/code/), [실제 실행 설정](provenance/settings/suite_plan.json), [보관 검증 범위](results/archive_verification.json), [파일 manifest](manifest.json)를 보존했습니다. 원본 소스와 가중치 내부 설정은 변경하지 않았습니다.

보관 해시는 다음 명령으로 확인합니다. 해시 확인은 새 성능 재평가가 아닙니다.

```bash
python SSA-MRN/common/verify.py --experiment 23
```

가중치 strict load·FP32 순전파와 특정 H5 재평가는 공통 도구로 수행합니다. 아래 예시는 실험 폴더 상대 경로를 사용합니다.

```bash
python SSA-MRN/common/checkpoint.py --experiment 23 --checkpoint checkpoints/F0_GF2_s46/best.pt
python SSA-MRN/common/checkpoint.py --experiment 23 --checkpoint checkpoints/F0_GF2_s46/best.pt --h5 /path/to/test_gf2_multiExm1.h5 --protocol RR --device cuda --output /path/to/fresh_metrics.json
```

새 학습은 별도 runtime에서 준비·시작합니다. 보관된 latest 내부의 실행 설정을 임의로 변경해 재개하지 않습니다.

## 8. 개선 여부와 다음 단계

**판정은 부분 개선입니다.** QB·GF2는 반복 복원 계열의 원본 대비 주요 RR·잠정 QNR 개선을 확인했습니다. WV3는 지표가 엇갈려 세 센서 전체 개선으로 표시하지 않았습니다. F0는 기준 비교 후보로 유지하지만 특징 전달의 추가 이득이 반복 복원보다 안정적이라고 결론내리지 않습니다.

다음 검증은 repeat와 F0의 추가 시드 반복, WV3의 FR 분광·공간 왜곡 분석, 공식 지표 정합 확인입니다. 단일 시드, 동일 에포크 예산의 비교이며 반복 모델의 계산량·학습 시간은 원본보다 큽니다. 기존 test를 탐색에 사용했으므로 독립 최종 holdout이나 통계적으로 확정된 우위로 주장하지 않습니다.
