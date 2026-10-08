<p align="center"><img src="./icon/CtrS-icon-circle.png" alt="CtrS" width="160" /></p>

# CtrS · PAN–MS 팬샤프닝 연구

**SSA-MRN의 PAN–MS 팬샤프닝 재현을 기준으로 성능을 개선**하는 연구 저장소입니다. 고해상도 PAN 영상과 저해상도 다중분광 MS 영상에서 고해상도 MS를 복원합니다.

## 연구 안내

| 자료 | 내용 |
|---|---|
| [현재 PAN–MS 연구](SSA-MRN/README.md) | 기준 모델, 재현 결과, 성능 개선 방향 |
| [알고리즘과 평가](SSA-MRN/docs/research.md) | PAN/MS/LMS, SSAI, RR·FR 평가 |
| [K4·K6 재현 기록](SSA-MRN/docs/reproduction.md) | 조건·수치·이미지·가중치 |
| [현재 학습 현황](SSA-MRN/docs/current_training.md) | Windows 손실 탐색과 학교 서버의 구조 비교 4개 병렬 |
| [다음 실험 계획](SSA-MRN/docs/improvement_plan.md) | 동일 환경 기준선 및 검증 기준 |
| [실행 방법](SSA-MRN/docs/operations/pan_ms.md) | 환경·학습·평가·VS Code 실행 |

## 최신 결과 · SSA-MRN 구조 개선

학교 A6000에서 **QB·K=6·100에폭·시드 42**로 기준선과 세 변형을 비교했습니다. validation으로 고른 best 가중치의 전체 RR20/FR20 평가입니다.

| 모델 | RR PSNR dB ↑ | RR SAM ° ↓ | FR QNR ↑* |
|---|---:|---:|---:|
| 기준선 | 37.4242 | 4.9639 | 0.914868 |
| 23탭 | 37.4799 | 4.9210 | 0.918011 |
| LR 보정 | 37.3839 | 4.9604 | 0.911718 |
| 고주파 경로 | 37.3621 | 4.9408 | 0.910940 |

23탭의 PSNR은 기준선 대비 **+0.0557dB**입니다. 단일 시드 탐색 결과라 반복 시드·센서 확장 검증을 거친 뒤 채택합니다. *FR QNR은 MATLAB 일치성 미검증인 잠정 수치입니다.*

### 테스트 이미지 3종

같은 장면이며 **LR MS · PAN · 예측 · 정답** 순서입니다. 입력을 자르지 않았고 MS만 표시용으로 확대했습니다. RGB 밴드 순서는 가정이며 같은 대비를 적용했습니다.

**23탭**

![23탭 테스트](SSA-MRN/docs/assets/a6000_architecture_qb/interp23_scene_01.png)

**LR 보정**

![LR 보정 테스트](SSA-MRN/docs/assets/a6000_architecture_qb/lr_correction_scene_01.png)

**고주파 경로**

![고주파 테스트](SSA-MRN/docs/assets/a6000_architecture_qb/high_frequency_scene_01.png)

[전체 보고서·5장면 세트·가중치](SSA-MRN/docs/experiments/improvements/a6000_architecture_qb.md)

Windows 서버의 K 비교·손실 후보 탐색은 별도로 진행 중입니다. 과거 K4/K6 재현과 현재 구조 비교의 조건·결과를 구분해 보관합니다. 원본 데이터는 Git에 포함하지 않습니다.

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
