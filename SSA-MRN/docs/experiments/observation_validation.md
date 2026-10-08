# QB 관측 연산자 검증

[실험 목록](../../README.md#experiments) · [수치](#results) · [그래프](#graphs) · [이미지](#images) · [가중치](#evidence)

| 비교 기준 | 변경 | 관측된 차이 | 판단 |
|---|---|---|---|
| 출처 미확인 bicubic 후보 | 공식 MATLAB MTF + 고정 패치 위상 | 전체 TRAIN 내부가 수치 오차 수준으로 일치 | 3단계 QB 보조 손실에 사용 가능 |

## 1. 목적과 상태

2026-10-08 사전 관측 검증 완료. 본 학습·RR/FR 모델 평가는 수행하지 않았습니다. 검증 코드는 이 문서와 같은 커밋의 `verify_observation.py`이며, 공식 참조 버전은 `a34f884af88967580e3ef4f61a89e12fdfe4cda5`입니다.

## 2. 변경 사항과 평가 조건

`MSE(관측연산자(예측), 입력 MS)`를 추가합니다. 임의 bicubic 대신 공식 MATLAB 프로토콜의 QB 밴드별 Nyquist MTF 계수 `[0.34, 0.32, 0.30, 0.22]`, 41×41 FIR, radial Kaiser window, x4 decimation을 사용합니다. FIR의 부호와 DC gain을 보존하며 clipping이나 재정규화를 하지 않습니다.

공식 패치 생성 코드에는 원본·상하 뒤집기·좌우 뒤집기가 있습니다. x4의 짝수 축소율 때문에 대응 위상은 각각 `(2,2)`, `(1,2)`, `(2,1)`로 달라집니다. 각 패치의 위상은 **학습 시작 전에 원본 TRAIN GT/MS에서만 확인하여 고정**하며, 예측 결과나 test/validation에서 선택하지 않습니다. `observation_profiles/qb.json`의 목록을 DataLoader sample index로 조회합니다. 원본 H5는 변경하지 않습니다.

전체 영상에서 필터링한 뒤 잘라낸 패치에는 주변 정보가 빠져 있으므로, LR 경계 5픽셀을 제외한 내부 6×6에서 손실을 계산합니다. 경계를 억지로 맞추거나 원본 데이터를 자르지 않습니다. 제외는 보조 손실에만 적용하며 기본 GT MSE는 전체 64×64를 사용합니다.


전체 QB TRAIN 17,139개, GT 64×64×4 / MS 16×16×4의 원시 DN을 CPU float64에서 대조했습니다. 필터·위상 후보는 공식 구현과 증강 방식에서 정했으며, 모델 예측·validation·test는 사용하지 않았습니다. 허용 오차는 사전 고정한 평균 RMSE≤1 DN, 최대 패치 RMSE≤2 DN입니다.

<a id="results"></a>

## 3. 정량 결과

| 항목 | 결과 |
|---|---:|
| 검증한 TRAIN 패치 | 17,139 / 17,139 |
| 관측 RMSE (DN) | 2.23619600876091e-13 |
| 최대 패치 RMSE (DN) | 9.879355127045927e-13 |
| 위상 (2,2): 원본 | 5,731 |
| 위상 (1,2): 상하 뒤집기 대응 | 5,671 |
| 위상 (2,1): 좌우 뒤집기 대응 | 5,737 |

모델 PSNR/SAM, 독립 RR/FR test 지표는 N/A — 본 학습과 모델 평가 전입니다.

## 4. 직전 연구와 수치 차이

이 문서는 관측 연산자 구현 검증입니다. 기존 K4/K6 모델 재현 성능과 직접 비교할 수 없습니다. 모델 성능 차이는 이후 같은 조건의 학습·독립 평가로 확인합니다.

<a id="graphs"></a>

## 5. 그래프

학습하지 않았으므로 epoch 곡선이나 모델 test 비교 그래프가 없습니다.

<a id="images"></a>

## 6. 결과 이미지 예시

고정 TRAIN probe 32개의 ID·해시를 프로파일에 기록했습니다. 모델 test 예시 5장은 아직 생성하지 않았습니다.

<a id="evidence"></a>

## 7. 가중치와 검증 근거

- [전체 위상·검증 프로파일](../../scripts/observation_profiles/qb.json)
- [관측 연산자](../../src/ssamrn/observation.py)
- [전체 TRAIN 검증 스크립트](../../scripts/verify_observation.py)
- [K4·K6 연결 검증](../../scripts/verify_consistency_loss.py)
- [SciPy 수치·뒤집기·해시·CUDA 테스트](../../tests/test_observation.py)

프로파일 SHA-256을 계획에 고정합니다. 연산자 코드 해시, 원본 파일 크기·mtime, 전체 위상 목록을 검사합니다. 네 테스트와 K4/K6의 손실 역전파가 통과했으며 연결 검증의 optimizer step은 0회입니다. 사전 실행기 검증에서 2 epoch 연속 학습과 1 epoch 후 latest 재개의 가중치·DataLoader RNG도 일치했습니다. 모두 성능 평가가 아닙니다.

서버 `scripts` 폴더에서 학습 없이 재검증합니다:

```powershell
C:\CtrS\.venv\Scripts\python.exe .\verify_observation.py --sensor QB --train "C:\CtrS\SSA-MRN\data\dataset\QuickBird\Training Dataset\train_qb.h5" --all-samples --output .\observation_profiles\qb.json
C:\CtrS\.venv\Scripts\python.exe .\verify_consistency_loss.py
```

공식 출처:

- [DLPan genMTF](https://github.com/liangjiandeng/DLPan-Toolbox/blob/a34f884af88967580e3ef4f61a89e12fdfe4cda5/02-Test-toolbox-for-traditional-and-DL(Matlab)/Tools/genMTF.m)
- [DLPan resize_images](https://github.com/liangjiandeng/DLPan-Toolbox/blob/a34f884af88967580e3ef4f61a89e12fdfe4cda5/02-Test-toolbox-for-traditional-and-DL(Matlab)/Tools/resize_images.m)
- [공식 패치·뒤집기 생성](https://github.com/liangjiandeng/DLPan-Toolbox/blob/a34f884af88967580e3ef4f61a89e12fdfe4cda5/03-Data-Simulation(Matlab)/Demo_DataSimu_qb.m)
- [MathWorks fwind1 수식](https://www.mathworks.com/help/images/ref/fwind1.html)


## 8. 한계와 다음 판단

패치 내부의 RR 관측 대응을 확인한 결과이며 FR 물리 모델이나 성능 개선의 증거가 아닙니다. GF2 측정 MTF 계수는 임의로 가정하지 않습니다. 검증 당시에는 본 학습 전이었으며, 이후 계획과 확인된 진행은 [Windows K·손실 비교](windows_controlled.md)에 기록합니다. 코드·데이터·프로파일이 바뀌면 재검증하고, 실행 기록과 계획의 SHA-256을 함께 검토합니다.
