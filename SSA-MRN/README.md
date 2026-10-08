# SSA-MRN · PAN–MS 팬샤프닝 재현과 개선

고해상도 PAN 1채널과 저해상도 MS를 결합해 고해상도 MS를 복원합니다. 목표는 **재현된 SSA-MRN의 공간·분광 복원 성능 개선**입니다.

## 최신 결과 · A6000 후속 검증

2026-10-08, **K=6·100에폭 새 학습10개를 완료**하고 기존 QB 시드42 두 모델을 재사용했습니다. validation MSE로 고른 best 가중치로 모델12개 각각 **RR20장·FR20장 전체**를 평가했습니다. 반복시드·확대 위치·센서를 검증했으며 Windows 손실 결합은 제외했습니다.

| 검증 항목 | 변형 − 같은 센서·시드 기준선 | 판단 |
| --- | --- | --- |
| QB 반복 3시드 | RR PSNR +0.0421 ± 0.0127dB; FR QNR -0.011602 ± 0.015583 | RR PSNR은 3/3 개선, FR은 2/3 악화 |
| QB 입력만 · s42 | RR PSNR +0.0807dB; QNR -0.000981 | 전체 23탭보다 validation MSE가 높음 |
| QB 출력만 · s42 | RR PSNR -0.0138dB; QNR +0.000293 | 기준선보다 validation MSE가 높음 |
| GF2 · s42 | RR PSNR +0.1490dB; SAM -0.0242°; QNR +0.005198 | 개선 후보, 반복 시드 필요 |
| WV3 · s42 | RR PSNR -0.0368dB; SAM +0.0219°; QNR -0.001746 | validation 이득이 test로 이어지지 않음 |

±는 QB 시드42·43·44의 paired 차이 표본 표준편차(n=3)입니다. **QB의 RR PSNR 이득은 반복됐지만 FR 개선은 유지되지 않았고 WV3는 악화돼, 전체23탭의 최종 채택을 보류합니다.** QNR은 MATLAB resize 일치성이 미검증인 잠정 값이며 GF2/WV3는 단일시드입니다. QB test는 기존 구조 탐색에 사용한 데이터라 새로운 독립 검증으로 표시하지 않습니다.

### 센서별 실제 예측

왼쪽부터 **LR MS · PAN · 예측 · 정답**입니다. 각 센서의 사전 고정 첫 장면을 보여 주며, RGB 밴드 [2,1,0]은 가정입니다. 동일 장면의 MS 패널에는 공통 대비를 적용하고 LR만 최근접 확대로 표시합니다.

**QB 전체23탭 · 시드43**

![QB 테스트 예측](docs/assets/a6000_followup_123/QB_interp23_k6_s43_scene_01.png)

**GF2 전체23탭 · 시드42**

![GF2 테스트 예측](docs/assets/a6000_followup_123/GF2_interp23_k6_s42_scene_01.png)

**WV3 전체23탭 · 시드42**

![WV3 테스트 예측](docs/assets/a6000_followup_123/WV3_interp23_k6_s42_scene_01.png)

[상세 보고서: 시드별 수치·위치 분석·곡선·고정35장면·원본24개 가중치](docs/experiments/improvements/a6000_followup_123.md)

## Windows 손실 탐색

이 채팅은 Linux 서버 실험을 담당합니다. Windows의 마지막 기록은 2026-10-08 11:39 KST, QB K6 72/100에폭이며 **이번 작업에서 현재 상태를 재확인하지 않았습니다.** 두 서버의 결과는 자체 기준선과 비교하고 하나의 반복시드 평균으로 합치지 않습니다.

## 이전 재현과 연구 자료

[직전 QB 단일시드 구조 비교](docs/experiments/improvements/a6000_architecture_qb.md)에서 전체23탭의 PSNR +0.0557dB가 관측됐습니다. 이번 반복평균 이득은 +0.0421dB이며 FR 이득은 유지되지 않았습니다. 기존 K4/K6는 장치가 달라 K만의 효과로 해석하지 않습니다.

[연구 설명](docs/research.md) · [K4/K6 재현](docs/reproduction.md) · [실험 이력](docs/previous_experiments.md) · [Windows 실험 계획](docs/improvement_plan.md) · [학습 현황](docs/current_training.md)

[Linux 후속 실행·평가·복구](docs/operations/a6000_followup.md) · [Windows 실행](docs/operations/controlled_suite.md)
