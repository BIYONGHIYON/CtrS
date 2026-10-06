# 2. 봄 6,000개 진단 연구

## 1. 연구 질문과 기존 문제

겨울 학습이 짧게 끝나고 논문보다 낮은 결과를 보여, 계절 특성·데이터 크기·학습 안정성 중 무엇이 문제인지 살펴보려 했습니다. 먼저 전체 봄 데이터 시작 시도는 epoch 0 도중 끝났습니다. 이어 작은 고정 표본으로 실행을 반복하고 진단 로그를 남겨, 학습이 어디서 무너지는지 확인했습니다.

## 2. 논문 기준

논문은 네 계절의 공식 ROI 분할, 최대 200 epoch, batch 16, AdamW, 초기 LR 4e-4를 사용합니다. validation이 5 epoch 정체되면 LR을 0.1배 낮추고 조기 종료 patience는 10입니다. 논문 성능은 MAE 0.0164, PSNR 33.37 dB, SSIM 0.932, SAM 4.693°, LPIPS 0.188입니다. [논문 PDF](https://zzaiyan.github.io/assets/pubs/ecrformer.pdf)

## 3. 수정한 점

- 봄 train pool에서 seed 42로 6,000개를 고정 선택하고 실행 목록을 저장해 재시작 간 표본을 동일하게 했습니다.
- 모델 구조는 바꾸지 않았습니다. 학습 표본 선택·기록과 Windows GPU 환경에 맞는 batch/precision 설정을 다뤘습니다.
- 원인 확인을 위해 training_diagnostics.jsonl에 진행 상황을 200 batch 간격으로 남겼습니다. checkpoint 재개 실행도 기록에 포함했습니다.
- 이 단계는 발산 원인을 입증하는 통제 실험이 아니라, 원인을 좁히기 위한 진단 실행입니다.

## 4. 데이터·서버·학습 설정

Windows 서버의 RTX 3060 Ti 8GB, RAM 16GB에서 실행했습니다. 저장된 로그 위치는 C:/CtrS/ECRformer/Official_ECRformer입니다.

| 항목 | 설정 |
| --- | --- |
| 데이터 | 봄 train pool 24,378개 중 고정 6,000개, validation 756개 |
| 모델·crop | ECRformer 전체 모델(약 11.37M 파라미터), 128×128 |
| 정밀도·배치 | FP16 mixed, train 2 × accumulation 8 = 유효 batch 16; validation batch 1 |
| optimizer·LR | AdamW, 4e-4 |
| 최대 epoch·조기 종료 | 200 / patience 10 |
| gradient clipping | 0.5 |

## 5. 실행 결과

최신 재개 실행은 기존 last.ckpt에서 이어서 학습했고 epoch 18까지 진행한 뒤 exit code 0으로 조기 종료됐습니다. best validation checkpoint는 epoch 8입니다.

| epoch | validation MAE↓ | PSNR↑ | SSIM↑ | validation loss↓ |
| ---: | ---: | ---: | ---: | ---: |
| 8 (best) | 0.022640 | 30.0305 dB | 0.912831 | 0.029093 |
| 15 | 0.031105 | 28.2592 dB | 0.870403 | 0.040954 |
| 16 | 21.068518 | -26.8539 dB | -0.000260 | 19.061686 |
| 18 | 64.329369 | -36.4211 dB | 0.000013 | 57.996456 |

epoch 16부터 학습 MAE와 validation 오차가 함께 급증했습니다. 이 양상은 일반적인 validation 과적합보다는 학습 발산에 가깝습니다. test 분할 평가는 기록되지 않았으므로 위 수치는 validation 결과뿐입니다.

## 6. 논문과 비교

best validation 수치가 논문 test 수치보다 가까워 보이더라도 validation 대 test, 봄 일부 대 네 계절 전체라 직접 비교할 수 없습니다. epoch 16 발산 직전에도 LR은 4e-4였습니다. 당시 MultiStepLR 이정표가 120 epoch 이후여서 실제 학습 구간에서는 LR이 낮아지지 않았습니다. 이는 설정 차이로 확인됐지만 최초 발산 원인이라고 입증된 것은 아닙니다.

FP32 대조가 없으므로 FP16을 원인으로 단정하지 않습니다. GPU OOM이나 데이터 경로 오류도 기록에서 확인되지 않았습니다. 원인을 확정하려면 동일 표본에서 학습률과 precision을 각각 하나씩 바꾼 대조 실험이 필요합니다.

## 7. 결론·한계

고정 표본과 상세 로그로 발산 구간을 epoch 16으로 좁혔지만, 발산한 배치·연산 또는 단일 원인을 확인하지 못했습니다. epoch 8 checkpoint의 test 성능도 아직 없습니다. 따라서 이 연구는 원인 규명에 성공한 실험이 아니라 **학습이 안정적으로 진행되지 않는다는 점을 확인한 진단**입니다.

## 8. 이력·산출물

- [전체 로그 분석과 미실행 개선 계획](../../../docs/spring_subset6000_log_analysis_20261005.md)
- GitHub 이력: [4c14d09](https://github.com/BIYONGHIYON/CtrS/commit/4c14d09), [a1fe219](https://github.com/BIYONGHIYON/CtrS/commit/a1fe219), [b448ad9](https://github.com/BIYONGHIYON/CtrS/commit/b448ad9), [854cacf](https://github.com/BIYONGHIYON/CtrS/commit/854cacf)
