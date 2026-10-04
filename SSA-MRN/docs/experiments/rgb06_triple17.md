# RGB 06 · RGB별 17특징 23탭 타일

[이전 실험 목록](../previous_experiments.md) · [현재 연구](../../README.md)

## 1. 목적과 상태

완료.

RGB04의 구조를 기준으로 latent feature 수만 12 → 17로 변경하여,
17특징 구성이 12특징 대비 더 나은 RGB 유도 HSI 복원 성능을 보이는지 검증했습니다.

이번 실험은 feature 수 자체의 영향을 확인하기 위한 통제 실험입니다.
K, resampling, loss, 데이터 분할, 학습 및 평가 조건은 RGB04와 동일하게 유지했습니다.

## 2. 선행연구 근거

본 실험의 17특징 설정은 HYDRA의 latent-size ablation 결과를 참고했습니다.

HYDRA는 RGB 입력을 직접 고차원 HSI로 복원하는 대신,
Teacher가 HSI를 저차원 latent space로 압축하고
Student가 RGB에서 해당 latent representation을 예측하도록 학습합니다.

특히 HYDRA의 ablation study에서는
LIB-HSI의 latent size를 13, 17, 34로 비교했으며,
최종 refinement 단계에서 latent size 17이 가장 좋은 성능을 기록했습니다.

| LIB-HSI latent size | MRAE ↓ | RMSE ↓ | PSNR dB ↑ |
|---:|---:|---:|---:|
| 13 | 0.4211 | 0.0120 | 39.64 |
| **17** | **0.3004** | **0.0091** | **42.24** |
| 34 | 0.3201 | 0.0125 | 41.84 |

HYDRA 저자들은 latent size가 성능에 영향을 주며,
HySpecNet-11k와 LIB-HSI에서는 17이 가장 좋은 설정이었다고 보고했습니다.

다만 HYDRA는 Teacher–Student 기반 spectral reconstruction 구조이고,
본 연구는 LR HSI + HR RGB를 이용하는 SSA-MRN 기반 공간 초해상도 구조이므로
17이라는 latent size가 동일하게 최적일 것이라고 가정할 수는 없습니다.

따라서 본 실험에서는 HYDRA의 결과를 직접 적용하는 것이 아니라,
SSA-MRN 구조에서도 17특징이 유효한지 별도로 검증했습니다.

## 3. 변경 사항과 평가 조건

### RGB04 대비 변경 사항

- latent feature: 12 → 17
- 204밴드 → 17특징 grouped 압축
- 특징 1개당 담당 밴드 수: 17밴드 → 12밴드

### 동일하게 유지한 조건

- K=4
- R/G/B 각각 독립 core
- 3개 decoder
- 밴드별 학습 fusion
- 내부 평균 축소 / 23tap 확대
- HSI 합성 축소: area ×4
- loss: MSE
- HR patch: 256×256
- LR HSI: 64×64
- train layout: quadrants
- eval layout: tiles
- seed: 42
- 동일 alignment manifest 및 valid mask
- LIB-HSI train / validation / test = 393 / 45 / 75 장면

RGB04:

```text
204 bands → 12 features
204 / 12 = 17 bands per feature