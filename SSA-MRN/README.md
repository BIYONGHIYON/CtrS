# SSA-MRN · PAN–MS 팬샤프닝 재현과 개선

고해상도 PAN 1채널과 저해상도 MS를 결합해 고해상도 MS를 복원합니다. 목표는 **재현된 SSA-MRN의 공간·분광 복원 성능 개선**입니다.

## 최신 결과 · 학교 A6000 구조 비교

2026-10-08, **QB·K=6·시드 42·100에폭**의 네 모델을 비교했습니다. validation으로 고른 **best 98에폭**을 사용해 RR20장·FR20장 전체를 평가했습니다.

| 모델 | RR PSNR dB ↑ | RR SAM ° ↓ | FR QNR ↑* |
|---|---:|---:|---:|
| 기준선 | 37.4242 | 4.9639 | 0.914868 |
| 23탭 | 37.4799 | 4.9210 | 0.918011 |
| LR 보정 | 37.3839 | 4.9604 | 0.911718 |
| 고주파 경로 | 37.3621 | 4.9408 | 0.910940 |

**23탭이 후속 검증 우선 후보입니다.** 기준선 대비 PSNR +0.0557dB, SAM −0.0429°, 잠정 QNR +0.003143입니다. 단일 시드의 작은 차이이며 최종 채택은 보류합니다. LR 보정과 고주파 경로는 일부 분광 지표 개선과 PSNR·QNR 악화가 함께 나타났습니다.

*FR QNR은 MATLAB resize 일치성이 미검증인 잠정 수치입니다. 23탭의 FR Dλ는 악화됐으므로 모든 지표의 개선을 뜻하지 않습니다.*

### 테스트 이미지 · 세 변형

동일한 사전 고정 장면 index 1입니다. 왼쪽부터 **LR MS · PAN 입력 · 예측 · 정답**입니다. 입력은 자르지 않았고 MS만 표시용 최근접 확대했습니다. RGB 밴드 [2,1,0]은 가정이며 세 그림에 같은 대비를 적용했습니다.

**내부 23탭**

![23탭 QB 테스트](docs/assets/a6000_architecture_qb/interp23_scene_01.png)

**LR 출력 보정**

![LR 보정 QB 테스트](docs/assets/a6000_architecture_qb/lr_correction_scene_01.png)

**PAN 고주파 경로**

![고주파 경로 QB 테스트](docs/assets/a6000_architecture_qb/high_frequency_scene_01.png)

[전체 결과 보고서: 각 변형 5장면·곡선·장면별 수치·가중치](docs/experiments/improvements/a6000_architecture_qb.md)

## 진행 중 · Windows 손실 탐색

2026-10-08 11:39 KST 확인 시 QB K=6 기준선의 72/100에폭을 실행 중입니다. 학교의 구조 비교와 별도로 K 기준선 6회 → 보조 손실 후보 9회 → 최종 후보 이어 학습을 직렬 실행합니다. 총 940에폭이며 Windows 결과는 아직 완료되지 않았습니다.

## 이전 재현과 연구 자료

기존 K4/K6 재현은 QB·GF2·WV3에서 완료했습니다. K4는 A6000/CUDA, K6는 Radeon/DirectML 환경이어서 같은 장치의 K 대조군으로 해석하지 않습니다. 과거 결과는 이번 학교 기준선과 별도입니다.

[연구 설명](docs/research.md) · [K4/K6 재현](docs/reproduction.md) · [실험 이력](docs/previous_experiments.md) · [Windows 실험 계획](docs/improvement_plan.md) · [학습 현황](docs/current_training.md)

[Windows 실행](docs/operations/controlled_suite.md) · [학교 실행·평가](docs/operations/a6000_suite.md)
