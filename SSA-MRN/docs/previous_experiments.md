# 이전 RGB–HSI 실험

[전체 연구 설명](guide/research_overview.md)

[현재 연구](../README.md) · [PAN–MS 재현](reproduction.md)

모든 수치는 75개 test 장면 평균입니다. 조건이 다른 행의 순위를 성능 개선의 근거로 삼지 않습니다. 차이와 평가 조건은 각 보고서의 4절에 기록했습니다.

| 연구 | 보고서 | PSNR dB ↑ | SAM ° ↓ |
|---|---|---:|---:|
| RGB 01 · 8특징 Bicubic 128 | [rgb01_latent8](experiments/rgb01_latent8.md) | 32.3048 | 2.3161 |
| RGB 02 · 정합 보정 Bicubic 256 | [rgb02_aligned256](experiments/rgb02_aligned256.md) | 32.4817 | 2.3121 |
| RGB 03 · grouped12 K=4 23탭 | [rgb03_grouped12](experiments/rgb03_grouped12.md) | 30.7574 | 2.0893 |
| RGB 04 · RGB별 12특징 23탭 타일 | [rgb04_triple12](experiments/rgb04_triple12.md) | 34.0530 | 2.2057 |
| RGB 05 · RGB별 12특징 Bilinear 타일 | [rgb05_triple12_bilinear](experiments/rgb05_triple12_bilinear.md) | 33.7897 | 2.2190 |
| RGB 06 · RGB별 17특징 23탭·분광 손실 | [rgb06_triple17_spectral](experiments/rgb06_triple17_spectral.md) | 34.0611 | 2.2006 |
| RGB 06 후처리 · LR 평균 일관성 보정 | [rgb06_lr_consistency](experiments/rgb06_lr_consistency.md) | 34.2173 | 2.1625 |

34특징 실험은 중단 후 요청에 따라 결과를 삭제했습니다. 검증된 최종 test 결과가 없으므로 완료 연구 표에 포함하지 않습니다. 모델 코드는 보존 가중치 평가와 향후 구조 변경을 위해 남깁니다.

## 검증 전용 예비실험

- [6단계10에폭·train155](experiments/pilot_subset155.md)
- [원본 대 미세조정·30에폭후속 비교](experiments/pilot_followup.md)

위 보고서는 validation 결과이며 상단test 표와 직접 비교하지 않습니다.

## 보존 파일

- `experiments/checkpoints/`: 실제 학습된 가중치와 체크섬. smoke 가중치는 제외.
- `experiments/results/`: 전체/장면별 지표, 선택 기록, compact epoch 수치와 정합 manifest.
- `docs/assets/`: 보고서의 그래프와 4패널 예시.
- `references/`: 공식 코드와 데이터셋 감사 자료, 과거 측정된 속도 수치.

원본 데이터, 현재 학습의 로그·설정·resume 가중치는 정리 대상에서 제외합니다.

## 폴더 구조

```text
SSA-MRN/
├── README.md                  최신 완료·현재 학습
├── AGENTS.md                  결과 완료·정리 규칙
├── configs/                   학습 설정과 이전 가중치 평가용 설정
├── docs/
│   ├── reproduction.md        PAN–MS 입구
│   ├── previous_experiments.md RGB–HSI 입구
│   ├── experiments/           공통 양식의 개별 보고서·template
│   └── assets/                실험 ID별 그래프·4패널 예시
├── experiments/
│   ├── checkpoints/           학습된 .pt·checksum
│   ├── results/               pan_k4, pan_k6, rgb01~07, 정합 근거
│   └── logs/                  현재 실행 중인 로그만 로컬 유지
├── references/                공식 코드·데이터 감사·측정 근거
├── scripts/                   학습·정합·평가·결과 내보내기·정리
├── src/ssamrn/                보존 모델과 데이터·지표 구현
└── tests/                     실제 동작/프로토콜 검증
```

과거 모델 구현은 가중치 로딩·추론에 필요하므로 남깁니다. 제거한 코드: `benchmark_lib*.py`와 K6 재학습 전용 `run_k6_local.ps1`. smoke 확인 자체는 새 모델 실행 확인에 필요하여 유지합니다. 삭제는 Git 기록에서 과거 파일을 지우는 작업이 아닙니다.

서버 정리 내역은 [삭제 대상과 가중치 보존 해시](../references/cleanup_20261003.json)에 기록했습니다. 확인된 제한 시험/smoke 및 완료 run의 임시 파일 26개 경로를 삭제하고, 보존 대상 가중치 48개 파일의 SHA-256이 전후 일치함을 확인했습니다. 현재 학습의 best/latest와 runtime은 제외했습니다.
