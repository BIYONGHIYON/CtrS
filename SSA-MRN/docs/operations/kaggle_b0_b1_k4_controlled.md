# Kaggle B0/B1 K4 통제 재학습

[설계·검증 보고서](../experiments/kaggle_b0_b1_k4_controlled.md) · [단일 셀 Python](../../scripts/kaggle_b0_b1_k4_controlled_one_cell.py) · [붙여넣기 텍스트](../../scripts/kaggle_b0_b1_k4_controlled_one_cell.txt)

1. T4×2, Internet을 켠다. QB 경로에 필요한 H5가 없으면 `kagglehub.dataset_download("biyonghiyon/ctrs-pan-sharpening-dataset")`로 데이터셋을 자동 연결/다운로드하고 반환된 폴더에서 정확한 파일명을 찾는다. 이미 있는 파일은 재사용한다. kagglehub가 없으면 자동 설치한다. 다운로드 실패·필수 파일 누락·중복이면 학습 전에 중지한다. Kaggle Notebook에서는 [공식 kagglehub 설명](https://github.com/Kaggle/kagglehub)에 따라 Input으로 자동 연결되며 공유 캐시를 사용한다. Session options에서 Persistence를 **Files only**로 설정한다. Python 또는 텍스트 파일 **전체**를 한 코드 셀에 붙인다. 패키지 torch/h5py/numpy/scipy/scikit-image/matplotlib/Pillow/psutil은 Kaggle 기본 환경을 사용하며 없으면 import 오류로 멈춘다.
2. 첫 실행은 `REQUIRE_RESUME=False`, `RESUME_INPUT=""`. `ssa_B0_B1_K4_s42_<UTC timestamp>` 독립 폴더를 만든다. GPU0=B0, GPU1=B1이다. 다른 worker가 발견되면 시작하지 않는다. OOM이면 자동으로 batch를 줄이지 않고 실패 기록을 남긴다. batch32 조건을 바꾸려면 별도 실험으로 설계해야 한다.

초기 core 해시 확인은 독립 CPU 프로세스에서 수행한다. 이전 셀의 `ssamrn`·`controlled_models` import 캐시가 남아 있어도 새 코드로 확인하며, 이 캐시 때문에 커널을 재시작할 필요는 없다. 학습 worker/model 소스와 체크포인트 protocol은 유지하므로 기존 저장본 재개에 같은 호환성 검사를 적용한다.
3. 출력되는 정확한 suite 경로를 기록한다. 화면은 이전 출력을 지우고 B0/B1 각각 최신 상태 한 줄만 갱신한다. 약15초 간격의 모델별 epoch/batch, 메모리, 실측 ETA 및 매 에폭 validation 지표를 확인한다(부모 셀 확인 주기5초). 과거 에포크 화면은 누적되지 않으며 전체 기록은 각 모델의 `worker.log`와 `history.json`에 유지한다. 초기 ETA는 학습만 추정하고 첫 에폭부터 실측 validation 시간을 포함한다. bootstrap·test·포장 시간은 ETA 밖이다.
4. 중단 시 `REQUIRE_RESUME=True`와 `RESUME_INPUT="/kaggle/working/정확한_suite_폴더"`를 지정한다. 세션 파일이 사라졌으면 보관한 Notebook Saved Output을 Input으로 연결해 해당 suite 폴더를 지정한다. **ZIP만 연결했다면 먼저 별도 새 디렉터리에 해제**해 suite 폴더를 지정한다. ZIP CRC와 `.sha256.json`을 확인한다.
5. 재개는 원본을 새 독립 폴더에 복사한다. 데이터·소스·설정, 양쪽 checkpoint를 확인한 후 실행한다. latest가 손상되면 latest.prev를 검사한다. 둘 다 없거나 불일치면 처음부터 시작하지 않는다. 원본 best와 latest의 에폭/지표 관계가 어긋난 저장 트랜잭션도 중지하며 일관된 Saved Output 복구본이 필요하다.
6. 완료 시 suite 전체·results ZIP·ZIP SHA-256 JSON을 다운로드하거나 Save Version의 Output으로 보관한다. 중간 중단 시에도 suite의 code/protocol/selection/B0/B1 전체를 보관한다. Kaggle이 이미 삭제한 파일은 복구할 수 없다. 코드가 Persistence의 실제 설정이나 원격 보관을 보장하지 않는다.

저장은 완료 배치 경계에서 약60초 간격, 첫 학습 전 epoch0, validation 직전, 매 에폭 종료에 수행한다. 저장 자체와 긴 GPU 연산 시간 때문에 정확히60초 간격은 아니다. validation 동안 학습 상태는 변하지 않으며 중단되면 validation을 다시 한다. 강제 종료 뒤에는 저장되지 않은 배치를 다시 계산한다. 모델·optimizer·scaler·Python/NumPy/CPU/CUDA RNG·GPU permutation generator·해당 에폭 순서·완료 배치·누적 loss·시간·history가 저장된다. 협조적 SIGINT/SIGTERM은 다음 배치 완료 후 저장한다. 강제 종료는 마지막 저장점만 복원한다.

결과 보관은 `best.pt`와 `latest.pt` **원본 바이트**, 장면별 RR/FR JSON, prediction NPZ80개, RR 패널10개, 실측 learning/test 비교 PNG, 초기화·배치순서 증거와 artifact manifest를 포함한다. 외부 보관소에서 재다운로드해 SHA-256과 복구를 확인해야 보관 완료다. 이번 작업에서는 이전 실험 폴더를 삭제하지 않는다. seed43은 자동 실행하지 않는다.
