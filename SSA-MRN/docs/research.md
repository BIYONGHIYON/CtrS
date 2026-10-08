# PAN–MS 연구 설명

[연구 전체](../README.md) · [실험 목록](../README.md#experiments)

고해상도 PAN은 공간 구조를, 저해상도 MS는 밴드별 분광 정보를 제공합니다. 제공 LMS는 MS를 목표 크기로 보간한 입력입니다. QB/GF2는 4밴드, WV3는 8밴드입니다.

## 처리 흐름

1. PAN, LMS, 저해상도 MS를 입력합니다. PAN/LMS는 같은 공간 크기이고 MS는 가로·세로가 각각 1/4입니다.
2. PAN의 축소·재확대 영상과 LMS를 결합해 초기 특징을 만듭니다.
3. 각 MS 밴드의 특징과 PAN을 SSAI에서 결합합니다. K는 내부 투영 채널 수이며 관측 밴드 수와 다릅니다.
4. 전체·1/2·1/4 공간 크기에서 처리한 특징을 결합해 HR-MS를 출력합니다.

제공 LMS는 23탭 보간 계열이고 재현 모델 내부 resampling은 Bilinear(`align_corners=True`)입니다. 이 경로를 개인 RGB–HSI의 평균 축소/23탭 내부 연산과 혼동하지 않습니다.

공개 코드의 원소별 곱·전치·softmax가 논문 설명의 행렬곱과 같다고 주장하지 않습니다. 공식 submodule을 고정하고 재현 래퍼의 변경을 따로 관리합니다. 후속 attention 변경은 별도 실험으로 검증합니다.

## 무엇을 비교하는가

| 변경 | 쉬운 설명 | 실험 |
|---|---|---|
| K4 / K6 | 내부 특징 채널 수를 4 또는 6으로 설정 | [Windows 통제 비교](experiments/windows_controlled.md) |
| 23탭 확대 | 내부에서 영상을 키울 때 bilinear 대신 23개 계수의 보간 필터 사용 | [구조 비교](experiments/a6000_architecture_qb.md), [후속 검증](experiments/a6000_followup_123.md) |
| 보조 손실 | 기본 MSE에 분광 방향·관측 일관성·경계 오차 중 하나를 추가 | [Windows 손실 탐색](experiments/windows_controlled.md) |

입력으로 제공되는 LMS의 23탭 보간과 **네트워크 내부 확대를 23탭으로 바꾸는 실험**은 서로 다른 위치의 연산입니다. validation으로 설정과 가중치를 선택하고 test는 성능 보고에 사용합니다.

## 평가

- RR: GT가 있는 축소 조건에서 PSNR·SAM·ERGAS·SCC·Q를 계산합니다.
- FR: 실제 해상도에서 QNR·Dλ·Ds를 사용합니다. FR에 고해상도 GT가 있다고 가정하지 않습니다.
- 학습·검증·테스트를 분리하고 테스트 수치로 가중치나 설정을 고르지 않습니다.
- 시각화는 입력과 예측·정답에 공통 대비를 적용합니다. 지표는 표시 전 원래 값으로 계산합니다.

[공식 코드 출처](../references/README.md) · [실행 방법](operations/pan_ms.md)

현재 평가 구현의 FR PAN 축소는 MATLAB 대신 scikit-image cubic을 사용하므로 Ds/QNR은 잠정 값입니다. Q2n·SCC의 MATLAB 대조 검증과 논문 PSNR 계산법 확인도 남아 있습니다. 성능 개선의 선행 단계로 이 평가 차이를 기록하고 검증합니다.
