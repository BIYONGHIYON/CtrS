# RGB 05 · RGB별 12특징 Bilinear 타일

[현재 연구](../../README.md) · [이전 실험 목록](../previous_experiments.md)

## 1. 목적과 상태

학습 중. 코드 `d947f84`, run `20261003-220209-d4faa37d0365`. 직전 triple12의 내부 보간만 재현 네트워크와 맞춰 공간 선명도와 분광 오차를 비교합니다. 완료 test 성능은 아직 없습니다.

## 2. 변경 사항과 평가 조건

| 항목 | 직전 triple12 | 현재 실험 |
|---|---|---|
| RGB core | R/G/B 각각 12특징 | 동일 |
| HSI 압축 | 204→12, 17밴드씩 | 동일 |
| SSAI K / parameters | 4 / 1,950,186 | 동일 |
| 내부 guide·특징 축소 | 평균 풀링 | Bilinear align_corners=True |
| 내부 guide·특징 확대 | 23탭 | Bilinear align_corners=True |
| HSI 합성 축소 / LMS | area ×4 / 23탭 | 동일 |
| train / validation / test | 393 / 45 / 75 장면 | 동일 |
| HR / LR / 평가 | 256 / 64 / tiles | 동일 |
| 정합·유효 마스크 | projective fixed manifest | 동일 |
| RGB 출력 결합 | bandwise learned fusion | 동일 |

RTX 3060 Ti 8GB, torch 2.7.1+cu128, AMP, batch 4. 원격 controller 실행이므로 Mac/SSH 접속 종료와 학습 실행은 분리되어 있습니다. 내부 보간은 PAN–MS 재현 코드와 맞췄지만, area 입력 축소·압축·잔차 구조까지 원 재현 모델과 같아진 것은 아닙니다.

## 3. 정량 결과

최종 test 미평가. smoke는 2 train/1 validation 장면에서 실행 가능성을 확인했으며 본 성능 결과로 사용하지 않습니다.

## 4. 직전 연구와 수치 차이

| 지표 | 직전 연구 | 현재 | 차이 |
|---|---:|---|---|
| MSE ↓ | 0.0004733597 | 대기 | 대기 |
| PSNR dB ↑ | 34.0530 | 대기 | 대기 |
| SAM ° ↓ | 2.2057 | 대기 | 대기 |

완료 후 같은 75장면의 ID·순서·마스크·축소·타일 기준을 검사한 뒤 차이를 기록합니다. 현재 문서에는 개선 여부를 확정하지 않습니다.

## 5. 그래프

완료 시 학습/검증 곡선과 test 비교 그래프를 필수로 추가합니다. 현재 최종 결과 그래프는 없습니다.

## 6. 결과 이미지 예시

완료 시 이전 연구와 같은 5장면 `[3, 0, 43, 18, 63]`의 tile 0을 사용합니다. LR HSI, RGB, 예측, 정답의 4패널을 사용하며 지표에 따른 재선택은 하지 않습니다.

## 7. 가중치와 검증 근거

- run: `C:\CtrS-triple12-bilinear\SSA-MRN\experiments\checkpoints\remote-runs\20261003-220209-d4faa37d0365`
- 현재 학습 설정: `configs/lib_rgb_hsi_triple12_k4_bilinear_tiles.json`
- 정합: `experiments/results/lib_registration/projective_alignment.json`와 체크포인트의 SHA-256.
- 완료 후 `scripts/export_results.py`로 전체 평가·그래프·5예시를 내보냅니다.

## 8. 한계와 다음 판단

합성 축소 평가이며 실제 센서의 HR-HSI 정답 정확도는 별도 검증 대상입니다. 모델 best checkpoint는 validation으로 선택하며 test는 선택에 사용하지 않습니다. 학습 완료 전 run의 설정·로그·가중치를 정리하지 않습니다.
