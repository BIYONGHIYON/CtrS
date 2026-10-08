# PAN–MS 실험 이력

[현재 연구](../README.md) · [재현 연구](reproduction.md)

| 실험 | 변경 | 관측 결과 |
|---|---|---|
| [PAN K4](experiments/reproduction/pan_k4.md) | 공식 코드 기반 학습·평가 복원 | QB/GF2/WV3 기준 결과 |
| [PAN K6](experiments/reproduction/pan_k6.md) | SSAI K4→K6, 실행 장치 변경 포함 | QB/WV3 RR 일부 개선, GF2 RR 악화 |
| [A6000 QB 구조 비교](experiments/improvements/a6000_architecture_qb.md) | 기준선·내부 23탭·LR 보정·고주파 경로 | 23탭 PSNR +0.0557dB, 반복 검증 전 잠정 후보 |

| [A6000 후속 검증](experiments/improvements/a6000_followup_123.md) | QB 3시드·확대 위치·GF2/WV3 | QB RR PSNR +0.0421±0.0127dB, FR 평균 악화; GF2 개선·WV3 악화 |

최신 후속 검증은 새10개 학습과 재사용2개 모델의 전체 RR/FR 평가입니다. 센서 전체 적용은 보류하며 GF2 반복시드·FR 구현 검증을 다음 판단으로 남깁니다.

RGB01~11 및 LIB-HSI 예비실험은 [개인 RGB–HSI 연구 이력](https://github.com/BIYONGHIYON/RGB-HSI-SR/blob/main/SSA-MRN/docs/previous_experiments.md)으로 옮겼습니다.
