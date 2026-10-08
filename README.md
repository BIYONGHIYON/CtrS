<p align="center"><img src="icon/CtrS-icon-circle.png" alt="CtrS" width="120" /></p>

# CtrS · SSA-MRN 성능 개선

고해상도 흑백 위성영상 **PAN**과 저해상도 다중분광영상 **MS**를 결합해 고해상도 MS를 복원하는 연구입니다. SSA-MRN을 재현한 뒤 **구조와 손실 함수를 같은 조건의 기준 모델과 비교**합니다.

## 지금까지의 결론

23탭 확대는 QB의 PSNR을 반복해서 높였지만 실제 해상도 지표와 다른 센서에서는 일관된 이득이 없어 **전체 적용을 보류**했습니다. Windows의 K·손실 비교는 별도 실험이며, 구조와 손실을 결합한 효과는 아직 확인하지 않았습니다.

| 검증 | 같은 센서·시드 기준선 대비 변화 | 판단 |
|---|---|---|
| QB · 3시드 | RR PSNR **+0.0421 ± 0.0127 dB**, FR QNR **−0.011602 ± 0.015583** | PSNR은 3/3 개선, QNR은 2/3 악화 |
| GF2 · 시드 42 | PSNR **+0.1490 dB**, SAM **−0.0242°**, QNR **+0.005198** | 개선 후보, 반복 시드 필요 |
| WV3 · 시드 42 | PSNR **−0.0368 dB**, SAM **+0.0219°**, QNR **−0.001746** | validation 이득이 test로 이어지지 않음 |

RR은 정답이 있는 축소 해상도 평가, FR은 고해상도 정답이 없는 실제 해상도 평가입니다. PSNR·QNR은 높을수록, SAM은 낮을수록 좋습니다. ±는 QB 시드 42·43·44의 기준선 대비 차이의 표본 표준편차이며, FR QNR은 MATLAB 구현과의 일치성 검증이 남은 잠정 값입니다.

왼쪽부터 **LR MS · PAN · 예측 · 정답**입니다. GF2 전체 23탭 모델의 고정 장면 예시이며 같은 장면의 MS 패널에 공통 대비를 적용했습니다.

![GF2 전체 23탭 예측](SSA-MRN/docs/assets/a6000_followup_123/GF2_interp23_k6_s42_scene_01.png)

## 실험 바로 보기

각 보고서 한 페이지에서 **조건·결과·비교·그래프·이미지·가중치**를 볼 수 있습니다.

| 실험 | 바꾼 점 / 확인할 내용 | 상태와 판단 |
|---|---|---|
| [K4 재현](SSA-MRN/docs/experiments/pan_k4.md) | 공개 코드의 학습·평가 복원 | RR/FR 평가 완료, 첫 기준 결과 |
| [K6 재현](SSA-MRN/docs/experiments/pan_k6.md) | SSAI 내부 차원 4→6 | 평가 완료, 장치도 달라 K만의 효과는 미확정 |
| [QB 구조 비교](SSA-MRN/docs/experiments/a6000_architecture_qb.md) | 23탭 확대·LR 보정·고주파 경로 | 평가 완료, 단일 시드에서 23탭을 후속 후보로 선정 |
| [23탭 후속 검증](SSA-MRN/docs/experiments/a6000_followup_123.md) | QB 3시드·확대 위치·GF2/WV3 | 평가 완료, 모든 센서에 적용하는 최종 채택은 보류 |
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
