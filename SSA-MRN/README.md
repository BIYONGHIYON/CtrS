# SSA-MRN: RGB 유도 HSI 초해상도

## 최신 실험 — RGB별 4그룹 · K=4 · 23탭

LIB-HSI의 RGB 3채널과 저해상도 HSI 204밴드를 사용한 합성 x4 공간 초해상도 연구입니다.
기존 PAN–MS 재현과 latent 8/K=6/Bicubic 실험의 지표·이미지는 [기존 실험 기록](./docs/experiment_history.md)에 모았습니다.


현재 [실험 설정](./configs/lib_rgb_hsi_grouped12_k4_23tap.json)을 사용합니다. 기존 실험과 별도 학습·평가 프로토콜을 사용합니다.

| 항목 | 새 구성 |
|---|---|
| HSI 압축·출력 | 연속 17밴드씩 12그룹 → 그룹별 feature 1개 → 최종 204밴드 |
| SSA guide | feature 1–4는 R, 5–8은 G, 9–12는 B; 분기별 guide 1채널 |
| SSA 내부 차원 | K=4 |
| HSI 확대 | 채널별 23탭 LMS 보간; 모델의 중간 확대도 동일 방식 |
| 정합 | RGB에 제한된 homography를 적용해 이동·회전·원근 기울기를 보정했습니다. |
| 학습 | 원본 512×512 장면을 비중첩 256×256 네 조각으로 사용했습니다. |
| 검증·시험 | 전체 512×512 시야를 area 평균으로 256×256으로 축소하며 조각으로 나누지 않습니다. |
| 합성 LR | 256×256 GT를 area 평균으로 64×64로 축소했습니다. |

학습 RGB는 원본에서 동일 위치를 자른 guide이며 HSI만 23탭으로 확대합니다. 검증·시험 RGB도 전체 시야를 256×256으로 축소해 GT와 맞춥니다. 정합으로 생긴 유효하지 않은 테두리는 loss와 양쪽 평가 지표에서 제외했습니다. 그룹과 RGB의 연결은 실험적 분기 배정이며 센서의 실제 파장 응답을 뜻하지 않습니다.

정합은 학습 전에 [장면별 manifest](./experiments/results/lib_registration/projective_alignment.json)로 고정했으며, 실제 3D 회전각·깊이·시차를 복원하지는 않습니다. 정합 추정에 HR HSI를 사용하므로 GT 기반 전처리입니다. 확대·열화·평가 영역이 바뀌었으므로 기존 Bicubic 결과와 직접적인 ablation 비교로 해석하지 않습니다. 새 지표의 비교 기준은 `interp23_*`이며 기존 `bicubic_*`와 구분했습니다.

## 최신 실행과 평가 상태

원격 제어 상태 기록에서 아래 실행의 정상 종료(`finished`, `exit_code=0`)를 확인했습니다.
이는 프로세스 정상 종료 기록이며, 최종 epoch·best 선택·시험 성능 확인을 대신하지 않습니다.

| 항목 | 기록 |
|---|---|
| 실행 ID | `20261003-022329-72b8a17b50de` |
| 재개 | `resume-latest` |
| 시작 / 종료 (서버 기록) | 2026-10-03 02:23:37 / 06:59:59 |
| 실행 시간 | 약 4시간 36분 23초 |
| 체크포인트 폴더 | `experiments/checkpoints/remote-runs/20261003-022329-72b8a17b50de` |
| 시험 평가·5세트 이미지 | 서버 결과 수집 후 추가 예정 |

시험 지표는 **시험 75장 전체·204밴드·유효 영역**에서 계산하며 비교 기준은 **23탭 LMS**입니다.
5세트는 시험 장면을 지표와 무관하게 고정 seed로 선택하고, 각 세트에
**LR HSI · RGB guide · 23탭 LMS · Prediction · GT**를 표시합니다.
현재 이전 모델의 지표·이미지를 최신 실험 결과로 대체하지 않았습니다.

## 결과 생성

Windows 저장소 루트에서 [결과 내보내기 스크립트](./scripts/export_latest_results.ps1)를 실행하면
해당 실행의 best 가중치로 전체 시험 평가와 5세트 비교 이미지를 생성합니다.
원본 데이터와 체크포인트는 수정하지 않으며, 별도 결과 폴더를 사용합니다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File SSA-MRN\scripts\export_latest_results.ps1
```

생성되는 ZIP에는 평가 JSON, 장면 선택 기록, 비교 이미지와 학습 기록만 포함하며 원본 데이터와 가중치는 포함하지 않습니다.
결과를 확인한 뒤 시험 평균 표와 5세트 이미지를 이 README에 추가합니다.

## 자료

- [모델·전처리·정합과 한계](./docs/rgb_hsi_extension.md#12그룹k423탭-신규-프로토콜)
- [RGB–HSI 데이터셋 비교](./docs/rgb_hsi_datasets.md)
- [기존 PAN–MS 및 RGB–HSI 실험 기록](./docs/experiment_history.md)

정합 추정에 HR HSI를 사용합니다. 현재 결과를 실제 LR 센서 환경의 성능으로 해석하지 않으며,
독립 장면·촬영 위치의 분할과 RGB 기여는 추가 검증이 필요합니다.
