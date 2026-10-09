# PAN–MS 실행 방법

저장소 최상위에서 실행합니다. PyTorch는 장치에 맞는 버전을 별도로 설치하세요.

```bash
git submodule update --init --recursive
python -m pip install -r SSA-MRN/requirements.txt
python -m unittest discover -s SSA-MRN/tests -p 'test_*.py'
```

## 로컬 실행 예시 · K4 명시

`train.py`는 `--train`, `--val`로 다른 H5 파일을 받습니다. H5에는 `pan`, `lms`, `ms`, `gt`가 있어야 합니다. 센서별 정규화·차원 검사를 수행합니다. 현재 연구 기본 조건은 **K4 / 100에폭 / 배치32 / Adam / LR1e-4 / seed42**입니다. 기존 `train.py`의 CLI 기본값은 아직 6이므로 **`--ssai-dimension 4`를 반드시 명시**합니다. 이 문서 정리는 실행 코드의 기본값을 변경하지 않습니다.

```bash
python SSA-MRN/scripts/train.py --train /data/train.h5 --val /data/val.h5 --sensor QB --ssai-dimension 4 --device cuda --checkpoint-dir SSA-MRN/experiments/checkpoints/local/new_qb_k4
```

이 명령은 로컬 학습 예시입니다. Windows 서버는 [SSH 독립 관리 스크립트](controlled_suite.md)를 사용하며 원격 터미널에서 직접 학습을 시작하지 않습니다.

위 경로는 예시이며 실제 파일로 바꿉니다. 기존 재현 가중치 디렉터리를 출력 폴더로 사용하지 마세요. 이어 학습할 때 `--resume <latest.pt>`와 동일한 센서·K·학습률·배치를 지정합니다. 현재 trainer는 매 에폭 latest를 저장하며 best 자동 선택 기능은 없습니다. 과거 재현 프로토콜과 새 validation 선택 정책을 구분해 기록해야 합니다.

## 기존 재현 가중치의 RR/FR 평가

```bash
python SSA-MRN/scripts/evaluate_paper.py --sensor QB --protocol both --checkpoint-root SSA-MRN/experiments/checkpoints --data-root /data/pancollection --output-dir SSA-MRN/experiments/results/local/new_qb_k4_eval --device cuda
```

데이터 하위 구조와 요구 파일명은 `evaluate_paper.py`의 센서별 경로를 확인하세요. 기본 값만으로 원격 데이터가 존재한다고 가정하지 않습니다. 위 예시는 기존 K4 재현 가중치 형식용입니다. K6 재현은 `SSA-MRN/experiments/checkpoints/k6`를 사용합니다. 새 Windows `train_controlled.py` 체크포인트는 메타데이터 형식이 달라 이 명령에 경로만 바꾸어 넣지 않습니다. 해당 기준선의 RR/FR 평가는 아직 대기 상태입니다.

## VS Code

원하는 Python 인터프리터를 선택하고 실행 및 디버그에서 `PAN-MS: train K4/K6` 또는 `PAN-MS: evaluate RR + FR`을 선택합니다. 입력 경로·센서·장치·출력 경로는 실행 시 묻습니다. Python 학습을 VS Code 터미널에서 시작하면 연결 종료 이후 생존을 보장하지 않습니다. Windows의 장시간 학습은 `run_controlled_suite.ps1` 관리 스크립트로 실행합니다.
