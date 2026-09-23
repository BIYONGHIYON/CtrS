# SSA-MRN 재현 계획과 진행 기록

## 범위

원 논문의 pansharpening 문제를 우선 재현한다. RGB–HSI 초해상도는 원 재현을 마친 뒤 별도 단계로 다룬다.

## 참고 출처

- 논문: Xu et al., *Spectral–Spatial Attention-Guided Multi-Resolution Network for Pansharpening*, IEEE JSTARS (2025), https://doi.org/10.1109/JSTARS.2025.3543827
- 공식 코드: https://github.com/zhouchuanxu/SSA-MRN
- 공식 데이터 안내: https://github.com/liangjiandeng/PanCollection

## 공개 코드 점검

2026-09-23 기준 공식 GitHub 저장소의 파일 목록에는 `network.py`와 `README.md`가 있다. 네트워크 정의 이외에 학습 루프, 데이터셋 로더, 실험 설정, 평가 지표 구현, 공개 체크포인트는 저장소에서 확인되지 않는다. 재현 구현은 원본 네트워크를 기준으로 하되, 별도 모듈마다 원 논문 수식·그림·표의 근거를 기록한다.

## 논문에서 확인한 설정

| 항목 | 기재 내용 | 구현 기록 |
|---|---|---|
| 입력 | PAN, LRPAN, MS | 구현 전 tensor shape 확인 |
| 출력 | HRMS | 센서별 MS 밴드 수와 일치 확인 |
| scale ratio | 4 | 고정 |
| 손실 | MSE / L2 | 확인 필요: 평균/합산 reduction |
| optimizer | Adam | 나머지 optimizer 인자는 논문 원문과 대조 |
| batch size | 32 | 메모리 제한 시 변경 사실을 기록 |
| learning rate | 1e-4 | 스케줄 여부는 원문 전체에서 확인 |
| epochs | 100 | 재현 기준 |
| SSAI K | 6 | 고정 |
| 학습 표본 수 | QB 19,044; GF2 22,010; WV3 10,794 | PanCollection 버전/분할과 대조 |
| 실행 장비 | GeForce RTX 2080 Ti | 기준 환경; 현재 장비 정보 별도 기록 |

RR 지표: SAM, ERGAS, PSNR, SCC, Q2ⁿ.  
FR 지표: QNR, Dλ, Ds.

## 구현 단계와 완료 기준

| 단계 | 작업 | 완료 근거 | 상태 |
|---|---|---|---|
| 0 | 공식 코드·논문 설정 목록화 | 출처, 누락 코드, 기준값을 이 문서에 기록 | 진행 |
| 1 | 환경과 PanCollection 경로 설정 | 실행 환경 및 데이터 manifest | 대기 |
| 2 | 로더와 Wald protocol 확인 | 센서별 shape/range 및 시각화 샘플 | 대기 |
| 3 | SSA-MRN forward 구현/연결 | 입력 shape부터 HRMS 출력까지 확인 | 대기 |
| 4 | 학습·체크포인트 복구 | 짧은 학습, 손실 추적, 재시작 확인 | 대기 |
| 5 | 기준 재현 학습 | 논문 고정 조건의 학습 로그·설정 파일 | 대기 |
| 6 | RR/FR 평가 | 지표별 결과와 실행 설정 | 대기 |
| 7 | 비교 및 교차 위성 실험 | 논문 표와 동일 조건 비교 및 차이 설명 | 대기 |
| 8 | ablation | LRPAN, multi-resolution, bandwise SSAI 등 | 대기 |

## 먼저 채울 구현 누락

1. PanCollection 파일 포맷과 공식 train/test 분할을 기록하는 manifest
2. 센서별 MS/PAN/LRPAN 전처리 및 Wald protocol 열화
3. 원본 네트워크와 대응되는 multi-resolution feature path
4. 논문 정의에 따른 SSAI와 bandwise 처리
5. MSE 학습 루프, Adam optimizer, checkpoint 저장·복구
6. RR 및 FR 지표와 논문 표 형식의 결과 export
7. seed, 버전, 설정 해시와 실행 명령을 남기는 실험 기록

## 실험 기록 템플릿

```text
run_id:
dataset / sensor:
protocol: RR | FR
split/version:
paper settings deviations:
code revision:
python / pytorch / cuda:
GPU:
seed:
command:
checkpoint:
metrics:
notes:
```
