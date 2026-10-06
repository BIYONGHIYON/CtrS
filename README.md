<p align="center">
  <img src="./icon/CtrS-icon-circle.png" alt="CtrS" width="160" />
</p>

# CtrS

동국대학교 종합설계 연구 프로젝트입니다. 팀원 4명이 **SSA-MRN**과 **ECRformer**를 각각 재현·확장하고, 결과와 실험 가능성을 비교해 최종 연구 주제를 정합니다. 최종 주제는 아직 선정하지 않았습니다.

## 두 연구 한눈에 보기

| 연구 | 문제 | 현재까지 확인된 결과 | 다음 검증 |
| --- | --- | --- | --- |
| SSA-MRN | PAN–MS 팬샤프닝 재현과 RGB 기반 204밴드 HSI 공간 초해상도 | PAN–MS 공개 구현 K=4·논문 설정 K=6 평가 완료. RGB–HSI 확장 RGB11은 test 75장면에서 PSNR 34.3948 dB, SAM 2.14986° | 실제 센서 HR-HSI 정답이 없는 한계를 구분하고, 재현 설정 비교와 정합·센서 일반화 검증 |
| ECRformer | 광학·SAR 입력에서 구름 없는 Sentinel-2 13밴드 영상 복원 | 봄·겨울 각 3,000개 패치 혼합 학습을 마치고 검증 best epoch 5 평가. 테스트 5,550패치의 가중 평균 PSNR 27.570 dB, SAM 10.283° | 더 넓은 계절·ROI와 논문 조건에서 재현 범위를 확장하고 분광 보존을 검증 |

두 연구는 입력·데이터·평가 범위가 달라 **PSNR이나 SAM 숫자만으로 서로의 우열을 비교하지 않습니다.** 결과를 해석할 때는 테스트 범위, 독립 장면 수, 학습 조건, 미확인 한계를 함께 봅니다.

## 현재 결과

### SSA-MRN · PAN–MS 재현과 RGB–HSI 확장

PAN–MS 재현에서는 공개 코드의 SSAI 차원 K=4와 논문에 기재된 K=6을 QB·GF2·WV3에서 평가했습니다. K=6은 WV3와 QB 일부 지표에서 나아졌지만 GF2 RR 결과는 낮아졌습니다. 학습 장치도 달라 차이를 K 설정 하나의 효과로 결론짓지 않았습니다. 후속 실험은 K=6을 기준으로 삼고 K=4 결과를 함께 보존합니다. [재현 결과와 구현 차이](SSA-MRN/docs/reproduction.md)

RGB–HSI 확장은 고해상도 RGB와 64×64 저해상도 HSI로 256×256, 204밴드 HSI를 복원합니다. 최신 RGB11은 RGB07 계열의 17특징·공동 디코더·23탭 보간·LR 평균 보정에 HSI 경계 손실을 더해 대조군과 네 강도를 비교했습니다. 검증으로 선택한 후보의 test 결과는 **MSE 0.0004376169, PSNR 34.3948 dB, SAM 2.14986°**입니다. 동일 학습량 대조군 대비 개선은 PSNR +0.0032 dB, SAM −0.00085°로 작고, RGB09보다 MSE는 약 0.105% 높습니다. 모든 지표에서 앞서거나 흐릿함을 해결했다고 주장하지 않습니다.

평가는 LIB-HSI 관측 HSI를 합성 area ×4 축소한 결과입니다. 실제 센서의 고해상도 HSI 정답을 이용한 평가는 아닙니다. [최신 연구·수치·이미지](SSA-MRN/README.md) · [204밴드 웹뷰어](https://biyonghiyon.github.io/CtrS/ssa-mrn/) · [연구 설명](SSA-MRN/docs/research.md) · [이전 실험](SSA-MRN/docs/previous_experiments.md)

### ECRformer · 봄·겨울 혼합 학습

SEN12MS-CR 봄과 겨울에서 각각 3,000개 패치를 골라 총 6,000개로 학습했습니다. 검증 loss 최저인 epoch 5 체크포인트를 사용해 두 계절 테스트 분할 3,983개와 1,567개, 총 **5,550개 패치**를 평가했습니다. 패치 수는 독립 장면 수와 같지 않습니다.

테스트 패치별 지표 평균의 계절별 결과는 봄 PSNR 27.716 dB / SAM 9.669°, 겨울 27.199 dB / 11.844°입니다. 논문 SEN12MS-CR 전체 평가와 데이터 범위·학습량·전처리가 달라 직접적인 재현 성공이나 모델 간 동등 비교로 해석할 수 없습니다. 이전 겨울 half1→half2 미세조정은 별도 실험이며 이번 혼합 학습의 전후 향상으로 합치지 않습니다. 테스트 예시에서도 경계 흐림과 구름 잔여가 관찰되어 완전 복원으로 표현하지 않습니다.

[학습·평가 기록과 예시](ECRformer/README.md) · [봄·겨울 실험 상세](ECRformer/reproduction/spring_winter6000/README.md) · [이전 겨울 실험](ECRformer/reproduction/winter_half1/README.md)

## 연구별 코드와 자료

| 폴더 | 내용 |
| --- | --- |
| [SSA-MRN](./SSA-MRN/README.md) | PAN–MS 재현, RGB–HSI 현재 결과 및 평가 재실행 안내 |
| [ECRformer](./ECRformer/README.md) | 원 논문 재현 배경, 학습 코드, 봄·겨울 실험 및 결과 |

학습·테스트 원본 데이터와 학습 가중치는 저장소에 넣지 않았습니다. SSA-MRN 공식 구현은 Git 하위 모듈이므로 저장소를 받을 때 다음 명령을 사용합니다.

```bash
git clone --recurse-submodules https://github.com/BIYONGHIYON/CtrS.git
```

## 최종 주제 선정 기준

1. 논문과 데이터·모델·평가 조건의 차이를 설명할 수 있는가?
2. 독립 장면을 분리하고 같은 조건에서 개선을 확인했는가?
3. 정량 결과와 시각적 결과, 실패 사례를 함께 설명할 수 있는가?
4. 남은 기간에 데이터 준비·학습·추가 검증·발표를 마칠 수 있는가?

## 팀원

| 이름 | GitHub | 전공 | 역할 |
| --- | --- | --- | --- |
| 이병현 | [BIYONGHIYON](https://github.com/BIYONGHIYON) | 멀티미디어공학과 | 팀장 |
| 김경찬 | [erickks2y-jpg](https://github.com/erickks2y-jpg) | 멀티미디어공학과 | 팀원 |
| 송준현 | [choco-ssalbbang](https://github.com/choco-ssalbbang) | 멀티미디어공학과 | 팀원 |
| 김윤지 | [kimrose1015-max](https://github.com/kimrose1015-max) | 데이터사이언스전공 | 팀원 |
