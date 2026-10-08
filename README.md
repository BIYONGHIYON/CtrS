<p align="center"><img src="icon/CtrS-icon-circle.png" alt="CtrS" width="120" /></p>

# CtrS · SSA-MRN 성능 개선

고해상도 흑백 위성영상 **PAN**과 저해상도 다중분광영상 **MS**를 결합해 고해상도 MS를 복원하는 연구입니다. SSA-MRN을 재현한 뒤 **구조와 손실 함수를 같은 조건의 기준 모델과 비교**합니다.

## 지금까지의 결론

2026-10-08 23:17 KST, GF2 반복 시드와 QB 입력 23탭 추가 학습 6개 및 전체 평가가 완료됐습니다. GF2는 PSNR·QNR 이득이 3시드 모두 반복됐지만 SAM은 혼재했고, QB 입력은 PSNR·SAM이 3시드 모두 개선됐지만 QNR은 2/3 악화해 최종 채택을 보류했습니다. Windows 손실 비교는 별도 실험입니다.

| 검증 | ΔPSNR (dB) | ΔSAM (°) | ΔMSE (peak=1) | ΔQNR | 판단 |
|---|---:|---:|---:|---:|---|
| GF2 전체 23탭 · 3시드 | +0.1095 ± 0.0698 | +0.00029 ± 0.02157 | -2.711e-06 | +0.005261 ± 0.002222 | PSNR/QNR 3/3 개선, SAM 2/3 악화 |
| QB 입력 23탭 · 3시드 | +0.0575 ± 0.0369 | -0.02123 ± 0.02598 | -2.543e-06 | +0.003569 ± 0.011528 | PSNR/SAM 3/3 개선, QNR 2/3 악화 |
| QB 전체 23탭 · 3시드 (재사용) | +0.0421 ± 0.0127 | -0.01894 ± 0.02250 | -2.116e-06 | -0.011602 ± 0.015583 | PSNR 3/3 개선, QNR 2/3 악화 |

RR은 정답이 있는 축소 해상도, FR은 정답이 없는 실제 해상도 평가입니다. ±는 시드 42·43·44의 기준선 대비 차이 표본 표준편차이며 통계적 유의성이 아닙니다. FR QNR은 MATLAB 일치성 미검증인 잠정 값입니다.

왼쪽부터 **LR MS · PAN · 예측 · 정답**입니다.

![GF2 전체 23탭 시드 43](SSA-MRN/docs/assets/a6000_gf2_repeat_qb_input/GF2_interp23_k6_s43_scene_01.png)

[최신 상세 보고서](SSA-MRN/docs/experiments/a6000_gf2_repeat_qb_input.md)에 15개 모델의 전체 RR/FR 평가, 45개 고정 이미지, 원본 best/latest 30개와 해시를 보존했습니다.

## 실험 바로 보기

각 보고서 한 페이지에서 **조건·결과·비교·그래프·이미지·가중치**를 볼 수 있습니다.

| 실험 | 바꾼 점 / 확인할 내용 | 상태와 판단 |
|---|---|---|
| [K4 재현](SSA-MRN/docs/experiments/pan_k4.md) | 공개 코드의 학습·평가 복원 | RR/FR 평가 완료, 첫 기준 결과 |
| [K6 재현](SSA-MRN/docs/experiments/pan_k6.md) | SSAI 내부 차원 4→6 | 평가 완료, 장치도 달라 K만의 효과는 미확정 |
| [QB 구조 비교](SSA-MRN/docs/experiments/a6000_architecture_qb.md) | 23탭 확대·LR 보정·고주파 경로 | 평가 완료, 단일 시드에서 23탭을 후속 후보로 선정 |
| [QB 게이트 고주파 계획](SSA-MRN/docs/experiments/kaggle_band_gated_hf.md) | 밴드·위치별 PAN 고주파 주입량 학습 | 계획·미실행, baseline·기존 경로와 paired 비교 예정 |
| [23탭 후속 검증](SSA-MRN/docs/experiments/a6000_followup_123.md) | QB 3시드·확대 위치·GF2/WV3 | 평가 완료, 모든 센서에 적용하는 최종 채택은 보류 |
| [GF2 반복·QB 입력 검증](SSA-MRN/docs/experiments/a6000_gf2_repeat_qb_input.md) | GF2 전체/QB 입력 23탭 각각 3시드 | 전체 평가 완료, GF2 PSNR/QNR 개선 반복·SAM 혼재 |
| [Windows K·손실 비교](SSA-MRN/docs/experiments/windows_controlled.md) | 같은 장치의 K4/K6 → 손실 9개 → 후보 연장 | 마지막 확인 시 학습 중, 손실 효과 미확정 |
| [QB 관측 연산자 검증](SSA-MRN/docs/experiments/observation_validation.md) | consistency 손실의 MTF·패치 위상 검증 | 구현 검증 완료, 모델 성능 평가와 구분 |

## 연구와 실행 안내

- [연구 전체 보기](SSA-MRN/README.md): 최신 결과, 센서별 이미지, 서버별 마지막 확인 기록
- [알고리즘과 평가](SSA-MRN/docs/research.md): PAN/MS, K, 23탭, RR·FR 설명
- 실행: [Windows K·손실 비교](SSA-MRN/docs/operations/controlled_suite.md) · [Linux 후속 검증](SSA-MRN/docs/operations/a6000_followup.md) · [기본 환경·평가](SSA-MRN/docs/operations/pan_ms.md)

## 코드 받기

```bash
git clone --recurse-submodules https://github.com/BIYONGHIYON/CtrS.git
cd CtrS
```

원본 데이터는 포함하지 않습니다. 실제 학습 가중치·에폭·해시·평가 근거는 각 보고서의 **7절**에서 확인합니다. RGB–HSI 연구는 [별도 저장소](https://github.com/BIYONGHIYON/RGB-HSI-SR)에서 관리합니다.

## 팀원

| 이름 | GitHub | 전공 | 역할 |
|---|---|---|---|
| 이병현 | [BIYONGHIYON](https://github.com/BIYONGHIYON) | 멀티미디어공학과 | 팀장 |
| 김경찬 | [erickks2y-jpg](https://github.com/erickks2y-jpg) | 멀티미디어공학과 | 팀원 |
| 송준현 | [choco-ssalbbang](https://github.com/choco-ssalbbang) | 멀티미디어공학과 | 팀원 |
| 김윤지 | [kimrose1015-max](https://github.com/kimrose1015-max) | 데이터사이언스전공 | 팀원 |
