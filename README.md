<p align="center"><img src="./icon/CtrS-icon-circle.png" alt="CtrS" width="160" /></p>

# CtrS · PAN–MS 팬샤프닝 연구

**SSA-MRN의 PAN–MS 팬샤프닝 재현을 기준으로 성능을 개선**하는 연구 저장소입니다. 고해상도 PAN 영상과 저해상도 다중분광 MS 영상에서 고해상도 MS를 복원합니다.

## 연구 안내

| 자료 | 내용 |
|---|---|
| [현재 PAN–MS 연구](SSA-MRN/README.md) | 기준 모델, 재현 결과, 성능 개선 방향 |
| [알고리즘과 평가](SSA-MRN/docs/research.md) | PAN/MS/LMS, SSAI, RR·FR 평가 |
| [K4·K6 재현 기록](SSA-MRN/docs/reproduction.md) | 조건·수치·이미지·가중치 |
| [현재 학습 현황](SSA-MRN/docs/current_training.md) | Windows 기록과 Linux 후속 실험 완료 상태 |
| [다음 실험 계획](SSA-MRN/docs/improvement_plan.md) | 동일 환경 기준선 및 검증 기준 |
| [실행 방법](SSA-MRN/docs/operations/pan_ms.md) | 환경·학습·평가·VS Code 실행 |

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

![QB 테스트 예측](SSA-MRN/docs/assets/a6000_followup_123/QB_interp23_k6_s43_scene_01.png)

**GF2 전체23탭 · 시드42**

![GF2 테스트 예측](SSA-MRN/docs/assets/a6000_followup_123/GF2_interp23_k6_s42_scene_01.png)

**WV3 전체23탭 · 시드42**

![WV3 테스트 예측](SSA-MRN/docs/assets/a6000_followup_123/WV3_interp23_k6_s42_scene_01.png)

[상세 보고서: 시드별 수치·위치 분석·곡선·고정35장면·원본24개 가중치](SSA-MRN/docs/experiments/improvements/a6000_followup_123.md)

Linux 새 학습은 모두 완료됐습니다. Windows의 현재 상태는 별도 채팅에서 관리하며 이번에 재확인하지 않았습니다. 원본 데이터는 Git에 포함하지 않고 실제 **best/latest 원본24개와 해시**는 이번 결과에 포함했습니다.

## 코드 받기

```bash
git clone --recurse-submodules https://github.com/BIYONGHIYON/CtrS.git
cd CtrS
```

## 팀원

| 이름 | GitHub | 전공 | 역할 |
|---|---|---|---|
| 이병현 | [BIYONGHIYON](https://github.com/BIYONGHIYON) | 멀티미디어공학과 | 팀장 |
| 김경찬 | [erickks2y-jpg](https://github.com/erickks2y-jpg) | 멀티미디어공학과 | 팀원 |
| 송준현 | [choco-ssalbbang](https://github.com/choco-ssalbbang) | 멀티미디어공학과 | 팀원 |
| 김윤지 | [kimrose1015-max](https://github.com/kimrose1015-max) | 데이터사이언스전공 | 팀원 |
