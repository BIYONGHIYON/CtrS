# SSA-MRN: RGB 유도 HSI 초해상도

## 현재 실험 — RGB별 12특징 · K=4 · 내부 Bilinear

LIB-HSI의 고해상도 RGB와 저해상도 HSI 204밴드로 합성 x4 초해상도를 연구합니다.
이번에는 기존 triple12와 특징 수·배치·분할을 유지하고, 코어 내부 확대·축소를 PAN–MS 재현 코드와 같은 Bilinear로 맞춥니다.

| 항목 | 현재 설정 |
|---|---|
| 압축·복원 | HSI 204밴드 → 연속 17밴드씩 12그룹 → 그룹당 특징 1개 → 204밴드 |
| RGB 처리 | R·G·B 각각 독립 코어에서 12특징 모두 처리 |
| SSAI / 파라미터 | K=4 / 1,950,186개 |
| guide 축소 후 재확대 | Bilinear → Bilinear |
| 코어 내부 분광 입력·특징 확대 | Bilinear |
| 코어 내부 특징·guide 축소 | Bilinear |
| Bilinear 옵션 | 재현 코드와 동일한 scale_factor, align_corners=True |
| 입력 HSI baseline·압축 입력 LMS | 23탭 유지 |
| 최종 출력 | 204밴드 잔차 3개를 밴드별 결합 + 23탭 HSI baseline |
| 합성 LR | HSI GT 256×256 → area 평균 64×64 |
| 학습·검증·시험 | 동일 비중첩 256×256 타일; train quadrants / eval tiles |
| 분할 | train 393 / validation 45 / test 75장면; 1,572 / 180 / 300타일 |
| 배치 | 학습 4 / 누적 1 / 검증 2; 기존 triple12와 동일 |
| 정합 | 고정 homography manifest와 유효 영역 마스크 사용 |

12개는 원본 밴드가 아니라 압축된 특징 수입니다.
[실험 설정](./configs/lib_rgb_hsi_triple12_k4_bilinear_tiles.json)과 [실행·구성 상세](./docs/triple_rgb_12_bilinear.md)를 참고하세요.

## 결과와 이전 실험

**새 Bilinear 실험의 정식 시험 결과는 아직 없습니다.**
실제 GPU smoke를 통과하고 2026-10-03 22:02:10부터 별도 예약 작업으로 100 epoch 학습을 시작했습니다.
상태·로그 확인 명령은 [실행 문서](./docs/triple_rgb_12_bilinear.md#실제-데이터-smoke와-원격-실행)에 있습니다.
완료된 triple12/내부 23탭 실험의 시험 지표와 이미지 5세트를
[이전 실험 기록](./docs/experiment_history.md)에 옮겼습니다. 기존 파일과 가중치는 보존합니다.

이전 triple12는 시험 75장면·300타일에서 PSNR 34.0530 dB, SAM 2.2057°였습니다.
이 값은 새 Bilinear 모델의 성능이 아닙니다. 같은 시험 타일·정합·마스크로 평가한 뒤 비교합니다.
중단한 34특징 실행의 체크포인트·smoke 결과·학습 로그는 사용자 요청으로 삭제했습니다.

## 재현과 평가 범위

- 모델 종류: `rgb_triple_grouped12_bilinear`. 기존 가중치를 재개하지 않고 새 실험으로 학습합니다.
- 별도 기본 출력 폴더: `SSA-MRN/experiments/checkpoints/lib_rgb_hsi_triple12_k4_bilinear_tiles`.
- 23탭은 입력 LMS와 HSI baseline에 유지하며, 내부 공간 연산만 PAN–MS 재현과 맞췄습니다.
- 시험 타일 오차를 원래 장면별로 합산하고 75장면 평균을 보고합니다. 타일을 독립 장면으로 세지 않습니다.
- PSNR은 range=1의 전체 204밴드 MSE로 계산하고 예측은 clip하지 않습니다. SAM은 영벡터 GT 픽셀을 제외합니다.
- HR HSI를 이용한 정합과 합성 area 축소 평가이므로 실제 LR 센서 쌍의 성능을 입증하지 않습니다.
- 제공 분할 내 촬영 위치·건물 중복과 RGB 기여는 추가 검증이 필요합니다.

## 자료

- [12특징 Bilinear 모델과 실행 방법](./docs/triple_rgb_12_bilinear.md)
- [이전 실험 지표와 예시 이미지](./docs/experiment_history.md)
- [RGB–HSI 데이터셋 비교](./docs/rgb_hsi_datasets.md)
- [전처리·정합과 한계](./docs/rgb_hsi_extension.md)
