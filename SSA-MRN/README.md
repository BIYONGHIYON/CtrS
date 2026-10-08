# SSA-MRN · PAN–MS 재현과 성능 개선

[저장소 홈](../README.md) · [알고리즘과 평가](docs/research.md)

PAN의 공간 정보와 MS의 분광 정보를 결합해 고해상도 MS를 복원합니다. QB·GF2는 4밴드, WV3는 8밴드이며 **K는 관측 밴드 수가 아니라 모델 내부 특징 차원**입니다.

<a id="experiments"></a>

## 실험 한눈에 보기

실험명을 누르면 중간 안내 페이지 없이 보고서가 열립니다. 모든 보고서는 **목적 → 조건 → 수치 → 기준선 대비 → 그래프 → 이미지 → 가중치·근거 → 판단** 순서입니다.

| 실험 | 바꾼 점 / 확인할 내용 | 상태와 판단 |
|---|---|---|
| [K4 재현](docs/experiments/pan_k4.md) | 공개 코드의 학습·평가 복원 | RR/FR 평가 완료, 첫 기준 결과 |
| [K6 재현](docs/experiments/pan_k6.md) | SSAI 내부 차원 4→6 | 평가 완료, 장치도 달라 K만의 효과는 미확정 |
| [QB 구조 비교](docs/experiments/a6000_architecture_qb.md) | 23탭 확대·LR 보정·고주파 경로 | 평가 완료, 단일 시드에서 23탭을 후속 후보로 선정 |
| [23탭 후속 검증](docs/experiments/a6000_followup_123.md) | QB 3시드·확대 위치·GF2/WV3 | 평가 완료, 모든 센서에 적용하는 최종 채택은 보류 |
| [Windows K·손실 비교](docs/experiments/windows_controlled.md) | 같은 장치의 K4/K6 → 손실 9개 → 후보 연장 | 마지막 확인 시 학습 중, 손실 효과 미확정 |
| [QB 관측 연산자 검증](docs/experiments/observation_validation.md) | consistency 손실의 MTF·패치 위상 검증 | 구현 검증 완료, 모델 성능 평가와 구분 |

<a id="latest"></a>

## 최신 완료 결과 · 23탭 후속 검증

2026-10-08 보고서 기준, K6·100에폭 새 학습 10개와 기존 QB 시드 42 모델 2개를 합쳐 **12개 모델 각각 RR 20장·FR 20장 전체**를 평가했습니다. best 가중치는 validation MSE로 선택했습니다.

| 검증 | 같은 센서·시드 기준선 대비 변화 | 판단 |
|---|---|---|
| QB · 3시드 | RR PSNR **+0.0421 ± 0.0127 dB**, FR QNR **−0.011602 ± 0.015583** | PSNR은 3/3 개선, QNR은 2/3 악화 |
| GF2 · 시드 42 | PSNR **+0.1490 dB**, SAM **−0.0242°**, QNR **+0.005198** | 개선 후보, 반복 시드 필요 |
| WV3 · 시드 42 | PSNR **−0.0368 dB**, SAM **+0.0219°**, QNR **−0.001746** | validation 이득이 test로 이어지지 않음 |

QB 입력만 23탭은 시드 42에서 PSNR +0.0807 dB, 출력만은 −0.0138 dB였습니다. 입력만 모델도 전체 23탭보다 validation MSE가 높아 test PSNR을 근거로 선택을 바꾸지 않았습니다.

**판단:** QB의 RR PSNR 이득은 반복됐지만 FR 개선은 유지되지 않았고 WV3도 악화했습니다. 다음 검토 대상은 GF2 반복 시드와 FR 평가 구현입니다. ±는 QB 3시드의 기준선 대비 차이 표본 표준편차이며 통계적 유의성을 뜻하지 않습니다. GF2·WV3는 단일 시드이고, QB test는 앞선 구조 탐색에 사용했던 데이터입니다. QNR은 MATLAB resize 일치성이 미검증인 잠정 값입니다.

![같은 센서·시드 기준선 대비 test 차이](docs/assets/a6000_followup_123/test_metrics.png)

### 센서별 예측 이미지

왼쪽부터 **LR MS · PAN · 예측 · 정답**입니다. 각 센서의 사전 고정 첫 장면입니다. RGB 표시 밴드 [2,1,0]은 가정이며, 같은 장면의 MS 패널에는 공통 대비를 적용하고 LR만 최근접 확대로 표시했습니다.

**QB · 전체 23탭 · 시드 43**

![QB 예측](docs/assets/a6000_followup_123/QB_interp23_k6_s43_scene_01.png)

**GF2 · 전체 23탭 · 시드 42**

![GF2 예측](docs/assets/a6000_followup_123/GF2_interp23_k6_s42_scene_01.png)

**WV3 · 전체 23탭 · 시드 42**

![WV3 예측](docs/assets/a6000_followup_123/WV3_interp23_k6_s42_scene_01.png)

[23탭 후속 보고서](docs/experiments/a6000_followup_123.md)에는 시드별 수치, 실측 곡선, 고정 35장면, 원본 best/latest 24개와 해시가 있습니다.

<a id="status"></a>

## 서버별 마지막 확인 기록

아래는 시각이 명시된 기록이며 실시간 상태가 아닙니다.

| 서버 | 마지막 확인 | 확인된 상태 | 남은 판단 |
|---|---|---|---|
| Linux · RTX A6000 | 2026-10-08 18:07 KST · 저장소 기록 | 후속 학습 10개 완료. 이후 재사용 2개 포함 전체 평가·가중치 보관 완료 | GF2 반복 시드, FR 구현 검증 |
| Windows · RTX 3060 Ti | 2026-10-08 16:24 KST · SSH 확인 | QB K4/K6 각각 100에폭, GF2 K4 49/100에폭 완료 후 학습 중 | 나머지 기준선 → 손실 9개 → 최종 후보 연장 → 평가 |

Windows 계획은 총 **940에폭 직렬 실행**입니다. 두 서버는 장치·환경·micro batch가 달라 각각 자체 기준선과 비교합니다. 구조+손실 결합의 효과는 아직 미확인입니다. 진행 확인 명령과 복구 절차는 아래 실행 안내에 있습니다.

<a id="files"></a>

## 실행과 파일 위치

| 필요한 작업 | 안내 |
|---|---|
| 기본 환경·기존 모델 평가 | [PAN–MS 실행](docs/operations/pan_ms.md) |
| Windows 진행 확인·재개 | [K·손실 실행](docs/operations/controlled_suite.md) |
| Linux 후속 검증·평가·복구 | [A6000 후속 실행](docs/operations/a6000_followup.md) |
| 최초 QB 구조 비교 재현 | [A6000 구조 비교 실행](docs/operations/a6000_suite.md) |
| 새 실험 기록 | [공통 보고서 양식](docs/experiments/template.md) |

보고서는 `docs/experiments/실험명.md`에, 이미지·수치·가중치는 기존 `docs/assets/실험ID/` 또는 `experiments/` 경로에 보관합니다. **새 실험의 실제 best/latest 가중치도 커밋 대상에 포함**하고 에폭·SHA-256을 함께 기록합니다. 원본 데이터·공식 하위 모듈·실행 중인 학습 파일은 별도로 보존합니다.
