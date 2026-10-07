# SSA-MRN · PAN–MS 팬샤프닝 재현과 개선

고해상도 PAN 1채널과 저해상도 MS(QB/GF2 4밴드, WV3 8밴드)를 결합해 고해상도 MS를 복원합니다. 팀 연구의 목표는 **재현된 SSA-MRN을 동일 조건의 기준선으로 삼아 공간·분광 성능을 개선**하는 것입니다.

[연구 설명](docs/research.md) · [재현 연구](docs/reproduction.md) · [실험 이력](docs/previous_experiments.md) · [개선 실험 계획](docs/improvement_plan.md) · [실행 방법](docs/operations/pan_ms.md)

## 현재 기준 · K4/K6 재현

RR 20장·FR 20장/센서 평가 기록입니다. 센서 간 숫자를 평균내어 우열을 정하지 않습니다.

| 센서 | K4 RR PSNR dB ↑ | K6 RR PSNR dB ↑ | K4 RR SAM ° ↓ | K6 RR SAM ° ↓ | K4/K6 FR QNR ↑ |
|---|---:|---:|---:|---:|---|
| QB | 37.3608 | 37.5040 | 4.9557 | 4.9350 | 0.9147 / 0.9072 |
| GF2 | 46.3520 | 45.8631 | 0.9668 | 1.0035 | 0.9106 / 0.9161 |
| WV3 | 37.3894 | 37.6326 | 3.5277 | 3.4504 | 0.9408 / 0.9522 |

K4는 CUDA/A6000, K6는 DirectML/Radeon입니다. K6가 모든 센서·프로토콜에서 낫지는 않으며, K만의 효과로 확정하지 않습니다. 논문 수식과 공개 코드의 attention 연산 차이도 유지된 재현입니다. [상세 비교](docs/experiments/reproduction/pan_k6.md)

![K6 센서별 지표](docs/assets/pan_k6/metrics.png)

아래는 QB 예시이며 왼쪽부터 LR MS · PAN · 예측 · 정답입니다. 나머지 센서와 K4 예시는 재현 보고서에 있습니다.

![QB K6 예시](docs/assets/pan_k6/sample_01.png)

## 다음 단계와 보존 범위

새 학습은 시작하지 않았습니다. 같은 GPU·데이터·학습량에서 기준선을 확인한 뒤 한 번에 한 변수만 바꿉니다. 상세 계획은 [개선 실험 계획](docs/improvement_plan.md)을 따릅니다.

K4/K6 가중치, 전체 평가 JSON, 그래프, 예시는 기존 경로에 보존합니다. RGB–HSI 확장은 [개인 저장소](https://github.com/BIYONGHIYON/RGB-HSI-SR)에서 계속하며 [이관 기록](docs/repository_split.md)에 분리 범위를 적었습니다.
