<p align="center"><img src="icon/CtrS-icon-circle.png" alt="CtrS" width="120" /></p>

# CtrS · SSA-MRN 성능 개선

고해상도 흑백 위성영상 **PAN**과 저해상도 다중분광영상 **MS**를 결합해 고해상도 MS를 복원합니다. SSA-MRN을 재현하고 구조·손실·학습 데이터 변화가 복원 성능에 미치는 영향을 비교하는 연구입니다.

## 현재 결정 · K=4

**2026-10-09 현재 연구의 기본 설정은 K=4로 확정했습니다.** Windows 동일 환경에서 QB·GF2·WV3의 K4/K6를 각각 100에폭 학습하고, 센서별 best validation MSE 다수결 **2:1**로 선택했습니다. 앞으로 별도 승인된 K 비교가 아닌 신규 기준선·개선 비교는 K4를 기준으로 설계합니다.

K는 모델 내부 특징 차원이며 센서 밴드 수와 다릅니다. WV3에서는 K6가 낮았으므로 이 결정은 모든 센서·test 지표에서 K4가 우세하다는 뜻은 아닙니다. K 선택에는 test를 사용하지 않았습니다.
| 센서 | K4 best validation MSE ↓ | K6 best validation MSE ↓ | 선택 |
|---|---:|---:|---|
| QB | 0.000171861819 | 0.000172164394 | K4 |
| GF2 | 0.000080382687 | 0.000082206216 | K4 |
| WV3 | 0.000360564225 | 0.000358851370 | K6 |

![Windows K4/K6 validation 비교](SSA-MRN/docs/assets/windows_k_baselines/validation_comparison.png)

[선택 근거·100에폭 곡선·원본 가중치 12개](SSA-MRN/docs/experiments/windows_k_baselines.md)

## 연구 흐름과 남은 일

**K4/K6 통제 비교 완료 → K4 확정 → K4 손실 탐색 → 최종 후보 연장 → 같은 조건의 기준선과 RR/FR 평가** 순서입니다.

2026-10-09 21:11 KST Windows 확인 기록에서는 spectral·consistency 6개가 완료되고 edge 0.001이 학습 중이었습니다. 이는 마지막 확인 기록이며 실시간 진행률이 아닙니다. 최종 손실 선택과 구조+손실 결합 효과는 아직 미확정입니다.

이전 K6 구조·증강 연구는 후보와 한계를 파악한 근거입니다. 그 결과를 K4 성능으로 바꾸어 해석하지 않으며, K4에 적용할 후보는 K4 기준선과 다시 비교해야 합니다.

## 실험 바로 보기

각 보고서에서 **목적·조건·수치·기준선 대비·그래프·이미지·가중치·판단**을 함께 볼 수 있습니다.

| 단계 | 실험 | 실제 K | 확인된 결과·상태 |
|---|---|---|---|
| 현재 기준 | [Windows K4/K6 통제 비교](SSA-MRN/docs/experiments/windows_k_baselines.md) | 4·6 | 6개 × 100에폭 완료, 원본 12개 보관. 다수결로 **K4 확정**, test 평가 대기 |
| 현재 진행 | [Windows 손실 탐색](SSA-MRN/docs/experiments/windows_controlled.md) | **4** | spectral·consistency 완료, edge 진행 기록. 최종 손실 효과 미확정 |
| 이전 재현 | [공개 코드 K4 재현](SSA-MRN/docs/experiments/pan_k4.md) | 4 | RR/FR 평가 완료, 초기 기준 |
| 이전 재현 | [논문 설정 K6 재현](SSA-MRN/docs/experiments/pan_k6.md) | 6 | RR/FR 평가 완료, K4와 장치도 달랐음 |
| 구조 탐색 | [QB 23탭·LR 보정·고주파](SSA-MRN/docs/experiments/a6000_architecture_qb.md) | 6 | 단일 시드에서 23탭을 후속 후보로 선정 |
| 구조 검증 | [23탭 반복·위치·센서 확장](SSA-MRN/docs/experiments/a6000_followup_123.md) | 6 | QB RR 이득·FR 혼재, GF2 개선·WV3 악화 |
| 구조 검증 | [GF2 반복·QB 입력 23탭](SSA-MRN/docs/experiments/a6000_gf2_repeat_qb_input.md) | 6 | 3시드 평가 완료, GF2 PSNR/QNR 개선·SAM 혼재 |
| 구조 탐색 | [QB 밴드별 게이트 고주파](SSA-MRN/docs/experiments/kaggle_band_gated_hf.md) | 6 | RR 악화로 채택 보류 |
| 증강 탐색 | [QB MTF 변화량 증강](SSA-MRN/docs/experiments/kaggle_mtf_pair.md) | 6 | 100에폭·원본 4개 보관, validation만 확인·RR/FR 대기 |
| 사전 검증 | [QB 관측 연산자](SSA-MRN/docs/experiments/observation_validation.md) | 해당 없음 | consistency 관측 구현 검증, 학습 성능 결과와 구분 |

## 문서와 실행 안내

| 필요한 내용 | 바로가기 |
|---|---|
| 현재 기준·이전 연구의 결론·다음 판단 | [SSA-MRN 연구 안내](SSA-MRN/README.md) |
| PAN/MS·K·23탭·평가 지표 설명 | [알고리즘과 평가](SSA-MRN/docs/research.md) |
| Windows 상태 확인·로그·재개 | [Windows 관리 스크립트](SSA-MRN/docs/operations/controlled_suite.md) |
| 환경 설정·기존 모델 재평가 | [기본 실행 안내](SSA-MRN/docs/operations/pan_ms.md) |

## 코드와 결과 받기

```bash
git clone --recurse-submodules https://github.com/BIYONGHIYON/CtrS.git
cd CtrS
```

실제 학습된 best/latest 원본과 에폭·SHA-256·복구 검증은 각 보고서 **7절**에 있습니다. 학습 완료·test 평가 완료·가중치 보관을 구분합니다. 원본 데이터는 Git에 포함하지 않습니다. RGB–HSI 연구는 [별도 저장소](https://github.com/BIYONGHIYON/RGB-HSI-SR)에서 관리합니다.

## 팀원

| 이름 | GitHub | 전공 | 역할 |
|---|---|---|---|
| 이병현 | [BIYONGHIYON](https://github.com/BIYONGHIYON) | 멀티미디어공학과 | 팀장 |
| 김경찬 | [erickks2y-jpg](https://github.com/erickks2y-jpg) | 멀티미디어공학과 | 팀원 |
| 송준현 | [choco-ssalbbang](https://github.com/choco-ssalbbang) | 멀티미디어공학과 | 팀원 |
| 김윤지 | [kimrose1015-max](https://github.com/kimrose1015-max) | 데이터사이언스전공 | 팀원 |
