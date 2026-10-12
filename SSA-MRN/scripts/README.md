# 실험 코드 사용 안내

문서와 코드는 변경한 구조를 나타내는 같은 영문 이름을 사용합니다. `_suite.py`는 실행 관리, `_cell.py`는 노트북 한 셀, `evaluate_*.py`는 평가입니다. 실제 장치와 데이터 조건은 보고서에서 확인합니다.

| 실험 | 코드 | 보고서 |
|---|---|---|
| 공개 코드 재현 | [evaluate_paper.py](evaluate_paper.py) | [public_code_reproduction](../docs/experiments/public_code_reproduction.md) |
| 논문 설정 재현 | [evaluate_paper.py](evaluate_paper.py) | [paper_setting_reproduction](../docs/experiments/paper_setting_reproduction.md) |
| 23탭 확대·저해상도 보정·고주파 경로 | [interpolation_correction_detail_suite.py](interpolation_correction_detail_suite.py) | [interpolation_correction_detail](../docs/experiments/interpolation_correction_detail.md) |
| 23탭 확대 위치와 센서 확장 | [interpolation_scope_sensors_suite.py](interpolation_scope_sensors_suite.py) | [interpolation_scope_sensors](../docs/experiments/interpolation_scope_sensors.md) |
| 23탭 확대의 반복 시드 비교 | [interpolation_seed_repeat_suite.py](interpolation_seed_repeat_suite.py) | [interpolation_seed_repeat](../docs/experiments/interpolation_seed_repeat.md) |
| SSAI 내부 특징 차원 비교 | [feature_dimension_losses_suite.py](feature_dimension_losses_suite.py) | [feature_dimension](../docs/experiments/feature_dimension.md) |
| 분광·관측 일치·경계 손실 추가 | [feature_dimension_losses_suite.py](feature_dimension_losses_suite.py) | [auxiliary_losses](../docs/experiments/auxiliary_losses.md) |
| 밴드별 게이트 고주파 주입 | [band_gated_detail_cell.py](band_gated_detail_cell.py) | [band_gated_detail](../docs/experiments/band_gated_detail.md) |
| MTF 변화량 데이터 증강 | [mtf_augmentation_cell.py](mtf_augmentation_cell.py) | [mtf_augmentation](../docs/experiments/mtf_augmentation.md) |
| 반복 복원에 MS 관측 오차 전달 | [recurrent_observation_cell.py](recurrent_observation_cell.py) | [recurrent_observation](../docs/experiments/recurrent_observation.md) |
| Haar 대역별 단계 복원과 게이트 | [haar_reconstruction_cell.py](haar_reconstruction_cell.py) | [haar_reconstruction](../docs/experiments/haar_reconstruction.md) |
| 반복 복원·관측 오차 전달의 재개 학습 | [recurrent_observation_resume_cell.py](recurrent_observation_resume_cell.py) | [recurrent_observation_resume](../docs/experiments/recurrent_observation_resume.md) |
| 단일 복원과 3단계 공유 복원 | [shared_reconstruction_cell.py](shared_reconstruction_cell.py) | [shared_reconstruction](../docs/experiments/shared_reconstruction.md) |
| 반복 복원의 축소 전 저역통과 필터 | [bandlimited_downsampling_cell.py](bandlimited_downsampling_cell.py) | [bandlimited_downsampling](../docs/experiments/bandlimited_downsampling.md) |
| SSAI 단계·잔차 복원·초기 입력 변경 | [structural_ablation_suite.py](structural_ablation_suite.py) | [structural_ablation](../docs/experiments/structural_ablation.md) |
| 작은 크기 특징 보존과 PAN 기여량 조절 | [coarse_feature_fusion_suite.py](coarse_feature_fusion_suite.py) | [coarse_feature_fusion](../docs/experiments/coarse_feature_fusion.md) |
| 반복 복원과 PAN·MS 분리 경로 | [dual_stream_reconstruction_suite.py](dual_stream_reconstruction_suite.py) | [dual_stream_reconstruction](../docs/experiments/dual_stream_reconstruction.md) |
| SSAI 내부 융합 연산 변경 | [ssai_fusion_cell.py](ssai_fusion_cell.py) | [ssai_fusion](../docs/experiments/ssai_fusion.md) |
| MS 관측 연산자 구현 | [verify_observation.py](verify_observation.py) | [ms_observation](../docs/experiments/ms_observation.md) |
| 반복 복원의 단계 간 특징 전달과 관측·PAN 차이 | [recurrent_guidance_suite.py](recurrent_guidance_suite.py) | [recurrent_guidance](../docs/experiments/recurrent_guidance.md) |

## 보관 파일 확인

저장소 루트에서 실행합니다.

```bash
python SSA-MRN/scripts/verify_experiment.py --experiment all
python SSA-MRN/scripts/verify_experiment.py --experiment dual_stream_reconstruction
```

이 검증은 보관 파일의 크기와 SHA-256을 확인합니다. 학습 재현·모델 추론·지표 재계산은 별도 평가입니다. 실험별 가중치와 소스 위치는 [실험 목록 파일](../experiments/catalog.json), 현재 파일 해시는 [보존 목록](../experiments/retained_files.json)에 있습니다.

## 재평가

입력 H5는 별도로 준비하고 해당 보고서의 데이터·정규화·best/latest 선택 조건을 맞춥니다. 과거 결과의 재평가는 보관된 원본 코드와 설정을 사용합니다. 새 코드로 재평가한 결과는 별도 출력 폴더에 저장합니다.

- 원본 재현: `evaluate_paper.py`의 `--checkpoint-root`, `--data-root`, `--output-dir`를 지정합니다. 이 평가기는 재현 실험의 `latest.pt`를 읽습니다.
- 23탭·보정 비교: `evaluate_interpolation_correction_detail.py`, `evaluate_interpolation_scope_sensors.py`, `evaluate_interpolation_seed_repeat.py`를 사용합니다. 각 평가기의 입력 조건은 대응 운영 문서에 있습니다.
- 구조 선별·작은 특징 융합·분리 경로: 해당 `_suite.py`의 `evaluate --runtime ...`를 사용합니다. 실행 폴더에는 원본 plan과 각 모델의 best 가중치가 필요하며 plan의 데이터 경로를 실행 환경에 맞춰 준비합니다.
- 노트북 실험: 해당 `_cell.py`의 재개 입력에 이전 가중치·설정을 연결합니다. 평가만 실행하는 방법은 보관된 worker의 평가 절차를 따릅니다. 셀 전체 실행은 학습까지 수행할 수 있습니다.

`docs/assets/` 내부의 과거 run ID와 snapshot 파일명은 원본 설정의 일부여서 유지합니다. 새 문서·공개 실행 파일 이름은 위 표를 기준으로 사용합니다.
