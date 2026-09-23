# SSA-MRN 복구: QuickBird 실행 가이드

## 확인한 사실과 수정 범위

- 공식 [SSA-MRN 저장소](https://github.com/zhouchuanxu/SSA-MRN)는 `network.py`만 제공한다. 학습/평가 스크립트와 가중치는 없다.
- 공식 코드의 `PansharpeningNet.forward(pan, lms)`는 내부에서 정의되지 않은 `ms`를 사용한다. 공식 파일은 `references/upstream/`에 그대로 두고, `src/ssamrn/models/ssa_mrn.py`에서 세 번째 입력 `ms`를 명시한 보정 실행 경로를 구현했다. 따라서 **원본과 동일한 공식 가중치를 복원한 것은 아니다**.
- [PanCollection](https://github.com/liangjiandeng/PanCollection)의 QuickBird ReducedData H5 `test_qb_multiExm1.h5`를 확인했다. `gt`와 `lms`: `(20, 4, 256, 256)`, `ms`: `(20, 4, 64, 64)`, `pan`: `(20, 1, 256, 256)`이다. 20개는 **시험용**이며 학습에 섞지 않는다.
- 데이터 로더는 QB/WV3/WV2를 `2047`, GF2를 `1023`으로 나눈다. 이는 [PanCollection 데이터 설명](https://github.com/XiaoXiao-Woo/PanCollection)의 센서별 값에 맞춘 것이며, SSA-MRN 논문의 전체 전처리와 일치하는지는 추가 검증이 필요하다.

## 진행 단계

| 단계 | 상태 | 완료 근거 또는 다음 작업 |
| --- | --- | --- |
| 0. 공개 범위 확인 | 완료 | 공식 모델만 있고 학습 코드·가중치는 없음 |
| 1. QuickBird H5 확인 | 완료 | 키, 20개 샘플, 밴드·공간 크기 확인 |
| 2. 로더 작성 | 완료 | NCHW·4배 축척 검증, 센서별 정규화 |
| 3. 모델 순전파 복구 | 완료 | 누락된 `ms` 인자 연결, 공식 레이어 보존, 실제 H5 입력 실행 |
| 4. 학습 루프 복구 | 부분 완료 | MSE/Adam/체크포인트 재시작을 **합성 데이터**로만 점검 |
| 5. 논문 조건 학습 | 대기 | 별도 QuickBird 학습·검증 H5 확보 후 실행 |
| 6. RR/FR 논문 지표 평가 | 대기 | 공식 평가 절차 및 지표 구현·대조 필요 |
| 7. GF2/WV3/WV2·ablation | 대기 | 해당 데이터와 재현 기준값 필요 |

## 실행

`SSA-MRN/` 디렉터리에서 실행한다. PyTorch는 장비에 맞는 버전을 먼저 설치하고 `requirements.txt`를 설치한다.

```bash
python scripts/inspect_data.py data/raw/QuickBird/test_qb_multiExm1.h5
python scripts/smoke_test.py --crop 32
python scripts/smoke_test.py --crop 256
python -m unittest discover -s tests -v
```

`smoke_test.py`는 **무작위 초기화 모델의 실행/shape와 파일 저장만** 확인한다. `experiments/results/quickbird_smoke/`에 4밴드 출력 `.npy`와 첫 3밴드를 대비 조정한 PNG 미리보기를 저장한다. PNG는 실제 RGB 색 재현이 아니며, 둘 다 학습된 복원 결과나 PSNR 등 성능으로 해석하면 안 된다.

학습·검증 파일을 각각 확보한 뒤에만 다음 명령을 쓴다. 검증 파일로 QuickBird 테스트 H5를 쓸 경우 그 결과는 개발 중 검증 결과로만 기록하고, 최종 시험 수치로 다시 사용하지 않는다.

```bash
python scripts/train.py \
  --train data/raw/QuickBird/train_qb.h5 \
  --val data/raw/QuickBird/valid_qb.h5 \
  --sensor QB --epochs 100 --batch-size 32 --lr 0.0001
```

장비 메모리가 부족하면 `--batch-size`를 줄이고 논문 설정에서 벗어난 사실을 실험 기록에 남긴다. 중단된 학습은 `--resume experiments/checkpoints/latest.pt`로 재개한다. 외부에서 받은 체크포인트는 신뢰한 출처인지 확인한 후 사용한다.

## 아직 확인할 것

1. 실제 학습 H5의 키·shape와 표본 수, 독립된 검증 split.
2. 논문 원문과 공식 코드의 다중 해상도 처리, `SSAI` 내부 차원 및 전처리 간 차이.
3. Wald protocol과 RR 지표(SAM, ERGAS, PSNR, SCC, Q2ⁿ), FR 지표(QNR, Dλ, Ds)의 기준 구현.
4. 논문 표의 수치와 같은 조건에서 재현되는지 여부.

QuickBird 테스트 H5 한 파일만 Git에 포함한다. 다른 원본 H5, 체크포인트와 실험 산출물은 제외한다. `data_overview_qb_reduce.png`는 데이터 개요 이미지이고, `DLR_HySU.zip`은 다른 연구용 데이터로 현재 QuickBird 실행에는 쓰지 않는다.
