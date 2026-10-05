# SSA-MRN · RGB 유도 HSI 초해상도

고해상도 RGB와 저해상도 HSI에서 ×4 HSI 복원을 연구합니다. 아래 결과는 LIB-HSI의 관측 HSI를 합성 축소한 평가이며 실제 센서의 HR-HSI 정답 성능과 구분합니다.

## 연구 개요

204밴드 HSI를 17특징으로 학습 압축하고 R·G·B 각각의 SSA-MRN 경로에서 공간 특징을 추정합니다. 최신 모델은 세 경로의 51개 특징을 **공동 디코더**에 넣어 204밴드 보정량을 만들고, 23탭 보간 결과에 더합니다. 합성 `area` 축소로 만든 LR HSI와 출력의 4×4 평균을 맞춘 뒤 MSE + 0.01(1−cos) 손실로 학습했습니다.

[연구 설명 · 재현부터 17특징·분광 손실까지](docs/guide/research_overview.md)에서 단계별 연산, 텐서 크기, 결과 해석을 확인할 수 있습니다.

## 문서 안내

- [최신 축소 예비실험 보고서](docs/experiments/pilot_subset155.md): 조건·RGB 진단·장면별 결과·가중치 해시
- [기존 PAN–MS 재현 정리](docs/reproduction.md)
- [이전 RGB–HSI 실험 정리](docs/previous_experiments.md)

## 축소 예비실험 결과 · 2026-10-06

**train155 / validation45 축소 예비실험을 완료했습니다.** 검증 전용 RGB 진단과 7개 학습 작업(각 10에폭)을 실행했으며 test는 후보 선택에 사용하지 않았습니다. 전체 학습은 10월 6일 03:37 KST에 정상 종료됐고 현재 실행 중인 학습은 없습니다.

| 작업 | best epoch | MSE ↓ | PSNR dB ↑ | SAM ° ↓ | gradient RMSE ↓ |
|---|---:|---:|---:|---:|---:|
| 00_baseline10 | 10 | 0.0008318284 | 31.6981 | 2.4058 | 0.025434 |
| 02_warm_control10 | 2 | 0.0004969431 | 34.0741 | 2.2505 | 0.021576 |
| 02_warm_low10 | 5 | 0.0004858836 | 34.1811 | 2.2482 | 0.021444 |
| 03_aligned_attention10 | 10 | 0.0008283217 | 31.7231 | 2.4074 | 0.025393 |
| 04_gated_detail10 | 10 | 0.0008279778 | 31.7271 | 2.4037 | 0.025387 |
| 05_global_encoder10 | 10 | 0.0008293195 | 31.7239 | 2.4077 | 0.025410 |
| 06_local_alignment10 | 10 | 0.0008275112 | 31.7334 | 2.4098 | 0.025372 |

warm 비교는 **LR1e-5가 LR1e-4보다 PSNR +0.1070dB, SAM −0.0023°**로 우세했습니다. 구조 후보의 baseline 대비 PSNR 이득은 +0.025~0.035dB로 작습니다. local alignment는 PSNR이 가장 높지만 SAM이 악화됐고, gated detail은 PSNR·SAM 모두 조금 개선됐습니다. 단일 seed·10에폭 결과이며 구조 후보 모두 마지막 에폭이 best이므로 확정적 개선이나 수렴 완료로 해석하지 않습니다.

![축소 예비실험 학습·검증 곡선](docs/assets/pilot_subset155/curves.png)

RGB 원본·흐림·±1px 이동·평균·0 입력 진단, 상세 조건, 장면별 지표 및 가중치 해시는 [축소 예비실험 보고서](docs/experiments/pilot_subset155.md)에 있습니다. 진단은 공통 보수적 마스크를 사용해 위 학습 평가와 직접 비교할 수 없습니다. 기존 RGB07 test75 결과는 아래 기준 연구에 보존했으며 이번 validation 결과와 분할·학습 규모가 달라 직접 비교하지 않습니다.

## 실행 상태와 보관 위치

모든 예비실험이 종료됐습니다. Linux 서버의 `SSA-MRN/experiments/checkpoints/pilot-subset155/suite_20261005T160615Z/<작업명>/`에 best/latest를 보존했습니다. 가중치와 데이터는 Git에 넣지 않았습니다. 기존 Windows용 [6단계 준비 안내](docs/guide/six_stage_pilot.md)와 이번 Linux subset155 실행 환경을 구분합니다.

## 기준 연구 · RGB07의 독립 test 결과

LIB-HSI 합성 ×4 평가입니다. 204→17 grouped 압축(인접 12밴드/특징), R/G/B 독립 core, 공동 디코더, K=4, **2,616,274개 파라미터**를 사용했습니다. HR256/LR64, 내부 평균 축소·23탭 확대, 정합 유효 마스크, 17특징 및 분광 손실 가중치 0.01을 유지했습니다. 이번에는 LR 평균 보정을 **학습 손실과 검증·테스트 모두에 적용**했습니다. best는 100에폭 중 검증 MSE가 가장 낮은 **99에폭**입니다.

| test 75장면 평균 · 300타일 | MSE ↓ | PSNR dB ↑ | SAM ° ↓ |
|---|---:|---:|---:|
| 23탭 baseline + LR 보정 | 0.0011539606 | 30.0649 | 2.4450 |
| 직전 RGB06 + LR 보정 | 0.0004559785 | 34.2173 | 2.1625 |
| **공동 디코더 17특징 + LR 보정** | **0.0004482036** | **34.2831** | **2.1570** |

직전 RGB06의 **보정 후 출력**과 같은 75개 테스트 장면·정합 마스크를 비교하면 PSNR **+0.0658 dB**, SAM **−0.0055°**, MSE **−0.000007775**입니다. 장면별 PSNR은 **44/75개 개선, 31/75개 하락**으로 차이가 작고 균일하지 않습니다. 구조 변경과 학습 중 보정 적용이 함께 바뀌었으므로 개선 원인을 하나로 분리할 수 없습니다. [새 실험 보고서](docs/experiments/rgb07_joint17_consistency.md)에 조건과 한계를 기록했습니다.

<details>
<summary>RGB07 학습 곡선과 고정 test 예시 5종</summary>

![학습 MSE와 검증 PSNR·SAM](docs/assets/rgb07_joint17_consistency/learning.png)

![test 기준선·직전 연구·현재 모델 비교](docs/assets/rgb07_joint17_consistency/test_metrics.png)

기존 RGB06과 LR 보정 후처리 결과는 [이전 실험 목록](docs/previous_experiments.md)에서 볼 수 있습니다.

### test 예시 5종

아래는 수치 확인 전 고정한 서로 다른 테스트 장면 5개입니다. 장면 인덱스 `[3, 0, 43, 18, 63]`의 tile0을 사용했습니다. 이번 결과의 [204밴드 웹 뷰어](https://biyonghiyon.github.io/CtrS/ssa-mrn/)에서 각 장면의 모든 밴드를 볼 수 있습니다. 서버 결과 폴더에도 동일한 오프라인 HTML을 보존했습니다.

왼쪽부터 **LR HSI · RGB 입력 · 공동 디코더 예측 · 정답**입니다. HSI 패널에는 밴드별 동일한 표시 대비를 사용했고, 정량 지표는 원래 204밴드 값을 계산했습니다.

![test 예시 1](docs/assets/rgb07_joint17_consistency/sample_01.png)

![test 예시 2](docs/assets/rgb07_joint17_consistency/sample_02.png)

![test 예시 3](docs/assets/rgb07_joint17_consistency/sample_03.png)

![test 예시 4](docs/assets/rgb07_joint17_consistency/sample_04.png)

![test 예시 5](docs/assets/rgb07_joint17_consistency/sample_05.png)

</details>

## 결과 보고와 파일 보관

완료 실험에는 공통 보고서, 학습/검증 및 비교 그래프, 사전에 선택한 서로 다른 test 5장면의 4패널 이미지를 반드시 남깁니다. 성능이 좋은 장면을 보고 골라서는 안 됩니다.

```powershell
& $py SSA-MRN\scripts\export_results.py --checkpoint '<run>\best.pt' --data-root 'C:\CtrS\SSA-MRN\data\dataset\RGB-HSI\LIB-HSI' --output-dir '<새 결과 폴더>' --previous-metrics '<직전 결과>\test\metrics.json'
```

실행 환경에 `requirements.txt`를 설치해야 합니다. exporter는 체크포인트에 저장된 평가 조건을 사용합니다. [공통 보고서 양식](docs/experiments/template.md), [작업 규칙](AGENTS.md), [데이터셋 감사 자료](references/rgb_hsi_datasets.md)를 참고합니다.

새 실험 시작 전 `cleanup_experiment.py --run-dir <완료 run>`의 목록을 검토하고 `--completed --apply`로 적용합니다. 가중치·숫자 결과는 보존하고 과거 설정·원시 로그·캐시는 정리합니다. 현재 학습의 run에는 적용하지 않습니다.
