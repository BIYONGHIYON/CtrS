# 성능 59. 기준 모델 C2MS의 cosine 학습 일정 연장

연구 분야: 기존 성능 개선 · 실제 실험자: 이병현

C2MS 구조와 축척 혼합 손실을 유지하고, 처음부터 300에폭 cosine 일정으로 학습합니다. 성능 58번의 분광 손실은 사용하지 않습니다.

| 확인된 결과 | 상태 | 개선 여부 |
| --- | :---: | :---: |
| 서버 사전 검증과 학습 실행을 준비합니다. | 계획 | 판단&nbsp;보류 |

## 1. 실험 목적과 현재 상태

시드 42로 QB → GF2 → WV3 순서의 연속 학습을 준비합니다. 각 센서는 독립적인 새 모델로 시작합니다. WV2는 학습 분할이 없으므로 WV3 학습 모델로 전이 평가합니다. 기준 모델은 C2MS입니다.

## 2. 변경한 구조와 비교 조건

| 조건 | 설정 |
| --- | --- |
| 구조·입력·출력 | 기존 C2MS(SSA-MRN K4 + W2 병렬 가지), PAN·LMS·MS → 복원 MS |
| 손실 | 원래 축척 MSE + 2배 거친 쌍 MSE, 가중치 1.0 |
| 변경 | 150에폭에서 300에폭으로 전체 일정 연장, 새 초기화 |
| optimizer·학습률 | Adam, cosine 0.0001 → 0.000001, 300에폭 |
| 데이터·시드 | 기존 train/validation 분할·정규화·센서 연산 유지, seed 42 |
| 배치·정밀도 | batch/micro 32, FP16 AMP, loss/validation FP32, TF32 끔 |
| 선택 | 원래 축척 validation MSE 최소(0에폭 포함, 동률은 먼저 도달한 가중치) |
| 대조군 | 성능 53번 QB·54번 GF2/WV3의 같은 시드 C2MS 150에폭 best |

총 900에폭을 직렬 실행합니다. 조기 종료 없이 매 에폭 validation을 수행합니다. latest에는 모델·Adam·AMP scaler·난수·샘플 순서·학습 기록을 함께 저장하여 중단 시 같은 일정으로 재개합니다. 300에폭이 언제나 더 나은 test 수치를 보장하지는 않습니다.

## 3. 정량 결과

아직 300에폭 학습·test 결과는 없습니다. 완료 후 QB·GF2·WV3 및 WV2 전이를 각 20장 전체에서 RR·RR2·FR_WALD·FR 관측 일치로 평가합니다. QNR·Dλ·Ds는 참고로 보고합니다. test는 가중치나 설정 선택에 사용하지 않습니다.

## 4. 기준선 대비 차이와 해석

이 비교는 300에폭에 맞춘 전체 학습 일정의 효과를 확인합니다. 150에폭 가중치에 150에폭을 덧붙인 실험이 아니며, 학습률은 첫 에폭 이후부터 기존 150에폭 일정과 달라집니다. 단일 시드와 재사용 공개 test의 한계가 있습니다.

## 5. 학습·평가 그래프

실측 학습 MSE·validation PSNR·SAM 곡선과 대조군 대비 test 그래프는 학습 후 자동 생성합니다.

## 6. 결과 이미지

고정 RR 장면 0·1·3·7·14·17을 두 형식으로 생성합니다. 원본 비교는 LR MS·PAN·C2MS 150·C2MS 300·정답과 공통 척도 오차 지도입니다. 확대 비교는 정답 전체·기준·후보·정답·기준 오차·후보 오차 순서이며 후보 개선이 가장 큰 64×64 창을 사용합니다. 창은 후보에 유리하게 선택됨을 수치와 함께 기록합니다. 타 센서의 같은 번호는 QB와 다른 장소입니다. FR에 정답을 만들지 않습니다.

## 7. 재현 자료

- [설정](config.json), [모델](model.py), [센서 연산](prepare.py), [학습](train.py), [실행 제어](run.py), [평가](evaluate.py)
- 실행 경로·환경·사전 검증·학습 시작 근거는 `provenance/`에 보존합니다. 서버 실행본은 별도 `C:/CtrS-p59/`를 사용합니다.
- 기존 캐시는 데이터·캐시 SHA-256과 크기를 확인한 후 새 runtime의 독립 경로로 연결합니다. 학습 중 기존 실행 폴더를 참조하거나 변경하지 않습니다.

```bash
python SSA-MRN/experiments/performance/59_extended_cosine_training/run.py --mode prepare
python SSA-MRN/experiments/performance/59_extended_cosine_training/run.py --mode optimize
python SSA-MRN/experiments/performance/59_extended_cosine_training/run.py --mode run
# 같은 sealed runtime에서 중단된 작업을 재개할 때만 --resume을 추가합니다.
```

완료 후 best·지표·compact 기록·그래프·이미지·manifest를 Git에, latest·initial을 GitHub Release에 보관하고 해시를 검증합니다.

## 8. 개선 여부와 다음 단계

**판단 보류**입니다. 학습 완료 후 판정 기준 v2.3으로 같은 시드 대조군과 비교합니다. 단일 시드 결과만으로 기준 모델을 자동 교체하지 않습니다.
