# ECRformer 재현 실험 이력

## 판정 요약

기록된 재현 연구는 **겨울 → 봄 → 봄·겨울 혼합**의 세 단계입니다. 겨울 연구는 서버 저장공간 50GB 제한 때문에 두 구간으로 나누어 실행했으므로, 학습 실행 횟수로 세면 겨울 2회와 봄·혼합 각 1회입니다. 네 번의 독립 반복 실험이나 네 개의 seed 비교를 뜻하지는 않습니다.

세 단계 모두 원 논문과 같은 전체 데이터·학습 조건으로 재현하지 못했습니다. 겨울은 validation 개선이 멈춰 조기 종료됐고, 봄은 후반에 학습과 validation 오차가 함께 발산했습니다. 혼합 학습은 프로세스가 정상 종료됐지만 20 epoch 이하의 축소 실험이며 논문 점수보다 낮았습니다. 따라서 결론은 **학습 프로세스가 실행된 적은 있지만, 논문 수준의 안정적 재현은 확인되지 않았다**입니다.

## 논문 기준

원 논문은 SEN12MS-CR 네 계절의 122,218패치와 공식 ROI 분할(155 train / 10 validation / 10 test)을 사용합니다. ECRformer 전체 모델을 최대 200 epoch 학습하며, batch 16, AdamW와 초기 학습률 4e-4를 사용합니다. validation loss가 5 epoch 개선되지 않으면 학습률을 0.1배 낮추고, 조기 종료 patience는 10입니다. SAR는 VV [-25, 0] dB, VH [-35, 0] dB로 처리합니다. 논문 Table 1 결과는 MAE 0.0164, PSNR 33.37 dB, SSIM 0.932, SAM 4.693°, LPIPS 0.188입니다. [논문 PDF](https://zzaiyan.github.io/assets/pubs/ecrformer.pdf)

## 단계별 변화와 결과

| 단계 | 무엇을 바꿨나 | 결과와 판단 |
| --- | --- | --- |
| [1. 겨울 데이터 분할 연구](./studies/01_winter/README.md) | 50GB 저장공간 제한으로 겨울 데이터를 두 구간으로 나눠 실행. 첫 구간 기준 학습 후 두 번째 구간에서 가중치 미세조정 | 첫 구간은 14 epoch에서 조기 종료. 두 번째 구간 미세조정은 별도 test에서 일부 지표가 개선됐으나, 전체 겨울 재현은 아님 |
| [2. 봄 6,000개 진단 연구](./studies/02_spring_6000/README.md) | 봄 train에서 seed 42로 6,000개를 고정 추출하고 샘플 목록·학습 진단 로그를 저장 | best validation은 epoch 8. epoch 16부터 오차가 급증했고 epoch 18에서 조기 종료. 원인은 특정되지 않았고 test 결과도 없음 |
| [3. 봄·겨울 혼합 연구](./studies/03_spring_winter_mixed/README.md) | 봄·겨울에서 3,000개씩 고정 추출. 모델 구조는 유지하고 데이터·로그·재개·학습 설정을 실행 환경에 맞춤 | 16 epoch 수행, best epoch 5. 합산 test MAE 0.03086, PSNR 27.57 dB, SSIM 0.86478. 논문과 동등 조건이 아님 |

## 비교할 때의 한계

혼합 실험의 합산 점수는 논문보다 낮지만 직접적인 우열 비교로 볼 수 없습니다. 혼합 실험은 두 계절 6,000개만 학습했고 봄·겨울 test 패치 5,550개만 평가했습니다. 논문은 네 계절의 공식 분할을 사용합니다. 학습 상한과 학습률 스케줄도 다릅니다. 실행 코드의 SAR 전처리 기록도 두 채널 모두 [-25, 0] dB여서 논문의 VH 범위 [-35, 0] dB와 차이가 있습니다.

봄 실험 로그에서는 epoch 16 이후 학습 오차도 validation 오차와 함께 크게 증가했습니다. 이는 단순한 validation 과적합과 다른 발산 양상입니다. 다만 FP32 대조, 학습률 단독 대조, 문제 배치 재현이 없어 원인을 확정하지 않았습니다. 혼합 실험의 exit code 0은 조기 종료 로직까지 정상 실행됐다는 뜻이지, 논문 재현에 성공했다는 뜻은 아닙니다.

## GitHub 문서 이력

- [6dc9581 — 겨울 half1 결과](https://github.com/BIYONGHIYON/CtrS/commit/6dc9581)
- [c72efdc — 겨울 half2 미세조정 결과](https://github.com/BIYONGHIYON/CtrS/commit/c72efdc)
- [4c14d09 — 봄 소규모 학습 설정](https://github.com/BIYONGHIYON/CtrS/commit/4c14d09)
- [a1fe219 — 고정 학습 샘플 선택과 목록 기록](https://github.com/BIYONGHIYON/CtrS/commit/a1fe219)
- [b448ad9 — 봄 로그 분석과 안정화 계획](https://github.com/BIYONGHIYON/CtrS/commit/b448ad9)
- [f302fdf — 봄·겨울 혼합 결과와 평가 산출물](https://github.com/BIYONGHIYON/CtrS/commit/f302fdf)

각 연구 문서는 같은 순서로 정리했습니다: 연구 질문, 논문 기준, 수정 사항, 데이터·서버·학습 설정, 결과, 논문과 비교, 결론·한계, 이력·산출물. 기존 상세 결과와 평가 파일은 원래 위치에 보존했습니다.
