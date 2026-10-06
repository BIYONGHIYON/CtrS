# 1. 겨울 데이터 분할 연구

## 1. 연구 질문과 기존 문제

원 논문의 전체 재현에 앞서 겨울 데이터에서 ECRformer가 학습되는지와 복원 성능을 확인하려 했습니다. 서버 저장공간 50GB 제한으로 전체 겨울 데이터를 한 번에 준비하기 어려워 데이터를 두 구간으로 나누어 실행했습니다. 학습이 매끄럽게 진전되지 않고 validation 개선이 일찍 멈춰, 두 구간 결과 모두 전체 겨울 데이터 재현으로 볼 수 없습니다.

## 2. 논문 기준

논문은 네 계절의 SEN12MS-CR과 공식 ROI 분할로 최대 200 epoch 학습합니다. 전체 모델, batch 16, AdamW, 초기 학습률 4e-4와 validation 기반 학습률 감소를 사용합니다. 논문 test 지표는 MAE 0.0164, PSNR 33.37 dB, SSIM 0.932, SAM 4.693°, LPIPS 0.188입니다. [논문 PDF](https://zzaiyan.github.io/assets/pubs/ecrformer.pdf)

## 3. 수정한 점

- 50GB 저장공간 제약에 맞게 겨울 데이터를 두 구간으로 분리했습니다.
- 첫 구간(half1)은 기준 모델 학습으로 실행했습니다.
- 두 번째 구간(half2)은 half1의 모델 가중치에서 시작해 미세조정했습니다. optimizer와 scheduler 상태를 이어받은 완전한 checkpoint 재개는 아니었습니다.
- ECRformer 전체 모델은 유지했고, 데이터 범위와 두 번째 실행의 초기화·학습률을 바꿨습니다.

## 4. 데이터·서버·학습 설정

| 실행 | 데이터·환경 | 설정 |
| --- | --- | --- |
| half1 | 겨울 train 6,833 / validation 1,386 / test 783패치(1 ROI, 제공된 학습 변경 이력 문서 기준). RTX A6000 | seed 42, AdamW, LR 4e-4, batch 4 × accumulation 4, FP32, 최대 200 epoch, early-stop patience 10 |
| half2 | train 7,162 / validation 1,385 / test 784패치. train/validation 수는 제공된 학습 변경 이력 문서 기준이며, 저장소의 hparams에는 수량이 기록되지 않음 | half1 가중치에서 미세조정, LR 1e-4, FP32, batch 4 × accumulation 4. best epoch=2-step=1344 checkpoint |

## 5. 실행 결과

half1은 epoch 0–13, 총 14 epoch를 실행했습니다. best validation checkpoint는 epoch 3 (valid_loss 약 0.038)이며 이후 validation 개선이 없어 조기 종료됐습니다.

| half1 test, 783패치 | RMSE↓ | MAE↓ | PSNR↑ | SAM↓ | SSIM↑ | LPIPS↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 결과 | 0.03438 | 0.02544 | 29.61 dB | 11.25° | 0.84656 | 0.39683 |

half2 학습은 총 13 epoch 실행됐고 best checkpoint는 epoch 2였습니다. half2 test 784패치에서 half1 가중치와 미세조정 checkpoint를 각각 평가했습니다.

| half2 test | RMSE↓ | MAE↓ | PSNR↑ | SAM↓ | SSIM↑ | LPIPS↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 미세조정 전 | 0.05518 | 0.03932 | 25.25 dB | 11.48° | 0.81873 | 0.48422 |
| 미세조정 후 | 0.05146 | 0.03661 | 25.89 dB | 10.78° | 0.82994 | 0.47428 |

## 6. 논문과 비교

half1의 MAE·PSNR·SSIM·SAM·LPIPS는 논문 표보다 낮은 성능을 보입니다. 그러나 이 실험은 겨울 데이터의 일부만 학습했고 test도 해당 겨울 ROI 패치입니다. half2는 또 다른 test 집합이므로 논문 또는 half1과 직접 동등 비교할 수 없습니다.

## 7. 결론·한계

저장공간 문제에 대응해 두 구간을 실행했지만, 첫 학습은 200 epoch에 도달하지 못하고 validation 정체로 조기 종료됐습니다. half2 미세조정은 같은 half2 test에서 일부 개선을 보였으나, 이는 모델 구조 개선이 아닌 기존 가중치의 두 번째 겨울 데이터 적응 결과입니다. 안정적인 전체 겨울 학습이나 원 논문 재현을 입증하지 않습니다. 따라서 이 단계에서 얻은 것은 제한된 겨울 기준선과 실패 기록입니다.

## 8. 이력·산출물

- [겨울 half1 결과·지표](../../winter_half1/README.md)
- [겨울 half2 지표와 비교 파일](../../winter_half2/)
- GitHub 이력: [6dc9581](https://github.com/BIYONGHIYON/CtrS/commit/6dc9581), [c72efdc](https://github.com/BIYONGHIYON/CtrS/commit/c72efdc), [be01000](https://github.com/BIYONGHIYON/CtrS/commit/be01000)
