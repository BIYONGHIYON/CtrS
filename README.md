<p align="center"><img src="./icon/CtrS-icon-circle.png" alt="CtrS" width="160" /></p>

# CtrS · PAN–MS 팬샤프닝 연구

동국대학교 종합설계 팀 프로젝트입니다. **SSA-MRN의 PAN–MS 팬샤프닝 재현을 기준으로 성능을 개선**합니다. 고해상도 PAN 영상과 저해상도 다중분광 MS 영상에서 고해상도 MS를 복원합니다.

## 연구 안내

| 자료 | 내용 |
|---|---|
| [현재 PAN–MS 연구](SSA-MRN/README.md) | 기준 모델, 재현 결과, 성능 개선 방향 |
| [알고리즘과 평가](SSA-MRN/docs/research.md) | PAN/MS/LMS, SSAI, RR·FR 평가 |
| [K4·K6 재현 기록](SSA-MRN/docs/reproduction.md) | 조건·수치·이미지·가중치 |
| [다음 실험 계획](SSA-MRN/docs/improvement_plan.md) | 동일 환경 기준선 및 검증 기준 |
| [실행 방법](SSA-MRN/docs/operations/pan_ms.md) | 환경·학습·평가·VS Code 실행 |

## 현재 근거

QB/GF2/WV3에서 공개 구현 K=4와 논문 설정 K=6을 평가했습니다. K6의 RR PSNR은 QB 37.5040 dB, GF2 45.8631 dB, WV3 37.6326 dB입니다. K4는 CUDA/A6000, K6는 DirectML/Radeon으로 실행 환경도 달라 K만의 효과로 해석하지 않습니다. 후속 개선은 같은 장치·분할·학습량의 기준 모델과 비교합니다.

원본 데이터는 포함하지 않습니다. 기존 학습 가중치와 평가 수치·이미지는 보존합니다. **분리 작업에서 새 학습은 시작하지 않았습니다.**

2026-10-07에 PAN–MS 경로만 남기도록 Git 이력을 재작성했습니다. 이전 clone에서는 pull/merge하지 말고 [이관 안내](SSA-MRN/docs/repository_split.md)에 따라 새로 clone하세요.

```bash
git clone --recurse-submodules https://github.com/BIYONGHIYON/CtrS.git
cd CtrS
```

[저장소 분리 기록](SSA-MRN/docs/repository_split.md) · [개인 연구 웹뷰어](https://biyonghiyon.github.io/RGB-HSI-SR/ssa-mrn/)

## 팀원

| 이름 | GitHub | 전공 | 역할 |
|---|---|---|---|
| 이병현 | [BIYONGHIYON](https://github.com/BIYONGHIYON) | 멀티미디어공학과 | 팀장 |
| 김경찬 | [erickks2y-jpg](https://github.com/erickks2y-jpg) | 멀티미디어공학과 | 팀원 |
| 송준현 | [choco-ssalbbang](https://github.com/choco-ssalbbang) | 멀티미디어공학과 | 팀원 |
| 김윤지 | [kimrose1015-max](https://github.com/kimrose1015-max) | 데이터사이언스전공 | 팀원 |
