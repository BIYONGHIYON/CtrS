# 3. 봄·겨울 혼합 진단 연구

## 1. 연구 질문과 기존 문제

봄 단일 계절 학습은 epoch 16부터 발산했고, 겨울 실험은 저장공간 제약과 조기 종료로 전체 계절을 재현하지 못했습니다. 계절을 섞으면 제한된 학습 표본으로도 두 계절의 구름·지표 특성을 함께 학습할 수 있는지 살펴봤습니다.

## 2. 논문 기준

논문은 SEN12MS-CR 네 계절과 공식 ROI 분할을 사용해 최대 200 epoch 학습합니다. batch 16, AdamW, 초기 LR 4e-4, 정체 5 epoch 후 LR 0.1배 감소, early-stop patience 10을 적용합니다. SAR VV는 [-25, 0] dB, VH는 [-35, 0] dB입니다. 논문 Table 1 결과는 MAE 0.0164, PSNR 33.37 dB, SSIM 0.932, SAM 4.693°, LPIPS 0.188입니다. [논문 PDF](https://zzaiyan.github.io/assets/pubs/ecrformer.pdf)

## 3. 수정한 점

- ECRformer 전체 모델 구조는 유지했습니다. 논문 구현을 실행하기 위해 데이터 경로, 고정 표본 목록, 로그·checkpoint 재개와 학습 설정을 맞췄습니다.
- 봄과 겨울에서 각각 seed 42로 3,000개씩 고정해 총 6,000개를 사용했습니다.
- 학습 상한은 20 epoch로 줄였습니다. ReduceLROnPlateau는 factor 0.5, patience 3으로 설정했습니다.
- 실행 도중 콘솔이 닫힌 뒤 last.ckpt에서 재개했습니다. 한 번에 끝까지 이어진 실행은 아니지만 최종 지표는 재개된 동일 학습 이력의 결과입니다.

## 4. 데이터·서버·학습 설정

Windows 공유 서버의 RTX 3060 Ti 8GB에서 C:/CtrS-ecrformer-winter/ECRformer/Official_ECRformer 경로로 실행했습니다.

| 항목 | 설정 |
| --- | --- |
| train | 봄 3,000 + 겨울 3,000, seed 42 고정 목록 |
| validation | 봄 756 + 겨울 2,771 = 3,527 |
| test | 봄 3,983 + 겨울 1,567 = 5,550 |
| 모델·optimizer | ECRformer 전체 모델(약 11.37M), AdamW, weight decay 1e-3 |
| crop·precision | 128×128, FP16 mixed |
| batch | train 2 × accumulation 8 = 유효 batch 16; validation 1 |
| epoch·조기 종료 | 최대 20, patience 10 |
| LR | 초기 4e-4; plateau 시 factor 0.5, patience 3 |

## 5. 실행 결과

총 epoch 0–15, 16 epoch를 완료했고 조기 종료됐습니다. 최적 validation checkpoint는 epoch 5 (epoch=5-step=2250.ckpt, loss 0.0394439)입니다. 테스트는 FP32, batch 1, crop 없이 원본 256×256 패치로 평가했습니다.

| test 범위 | 패치 | RMSE↓ | MAE↓ | PSNR↑ | SAM↓ | SSIM↑ | LPIPS↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 봄 | 3,983 | 0.04485 | 0.03065 | 27.716 dB | 9.669° | 0.87421 | 0.41115 |
| 겨울 | 1,567 | 0.04570 | 0.03141 | 27.199 dB | 11.844° | 0.84082 | 0.43569 |
| 합산, 패치 수 가중 | 5,550 | 0.04509 | 0.03086 | 27.570 dB | 10.283° | 0.86478 | 0.41808 |

## 6. 논문과 비교

| 평가 | MAE↓ | PSNR↑ | SSIM↑ | SAM↓ | LPIPS↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 논문, 네 계절 전체 | 0.0164 | 33.37 dB | 0.932 | 4.693° | 0.188 |
| 혼합 실험, 봄·겨울 test | 0.03086 | 27.570 dB | 0.86478 | 10.283° | 0.41808 |

이번 점수는 논문보다 낮지만 학습량, 계절 범위, 전처리와 스케줄이 달라 동등 조건의 재현 비교는 아닙니다. 실행 기록은 SAR 두 채널을 모두 [-25, 0] dB로 잘라 논문의 VH [-35, 0] dB와 다릅니다. 이 차이가 성능 하락을 일으켰는지는 검증하지 않았습니다.

## 7. 결론·한계

학습과 test 평가가 끝나고 프로세스도 정상 종료됐지만 best는 epoch 5였고 이후 validation 성능은 정체했습니다. 제한된 두 계절 표본으로 얻은 결과는 논문 재현 성공으로 볼 수 없습니다. 단일 seed, 축소된 학습량, SAR 전처리 차이도 남아 있어 원인을 구분할 수 없습니다. 현재 증거가 뒷받침하는 결론은 **학습이 실행됐으나 안정적이고 논문 수준의 재현은 달성하지 못했다**입니다.

## 8. 이력·산출물

- [세부 지표·학습 목록·분할 점검·평가 코드](../../spring_winter6000/README.md)
- GitHub 결과 기록: [f302fdf](https://github.com/BIYONGHIYON/CtrS/commit/f302fdf)
