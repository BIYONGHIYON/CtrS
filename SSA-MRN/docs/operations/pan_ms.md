# PAN–MS 실행 방법

저장소 최상위에서 실행합니다. PyTorch는 장치에 맞는 버전을 별도로 설치하세요.

```bash
git submodule update --init --recursive
python -m pip install -r SSA-MRN/requirements.txt
python -m unittest discover -s SSA-MRN/tests -p 'test_*.py'
```

## 학습

`train.py`는 `--train`, `--val`로 다른 H5 파일을 받습니다. H5에는 `pan`, `lms`, `ms`, `gt`가 있어야 합니다. 센서별 정규화·차원 검사를 수행합니다. 기본 학습 조건은 K6 / 100에폭 / 배치32 / Adam / LR1e-4 / seed42입니다. K4는 `--ssai-dimension 4`로 지정합니다.

```bash
python SSA-MRN/scripts/train.py --train /data/train.h5 --val /data/val.h5 --sensor QB --ssai-dimension 6 --device cuda --checkpoint-dir SSA-MRN/experiments/checkpoints/local/new_qb_k6
```

위 경로는 예시이며 실제 파일로 바꿉니다. 기존 재현 가중치 디렉터리를 출력 폴더로 사용하지 마세요. 이어 학습할 때 `--resume <latest.pt>`와 동일한 센서·K·학습률·배치를 지정합니다. 현재 trainer는 매 에폭 latest를 저장하며 best 자동 선택 기능은 없습니다. 과거 재현 프로토콜과 새 validation 선택 정책을 구분해 기록해야 합니다.

## RR/FR 평가

```bash
python SSA-MRN/scripts/evaluate_paper.py --sensor QB --protocol both --checkpoint-root SSA-MRN/experiments/checkpoints/k6 --data-root /data/pancollection --output-dir SSA-MRN/experiments/results/local/new_qb_k6_eval --device cuda
```

데이터 하위 구조와 요구 파일명은 `evaluate_paper.py`의 센서별 경로를 확인하세요. 기본 값만으로 원격 데이터가 존재한다고 가정하지 않습니다. K4 재평가에서는 checkpoint-root를 `SSA-MRN/experiments/checkpoints`로 지정합니다.

## VS Code

원하는 Python 인터프리터를 선택하고 실행 및 디버그에서 `PAN-MS: train K4/K6` 또는 `PAN-MS: evaluate RR + FR`을 선택합니다. 입력 경로·센서·장치·출력 경로는 실행 시 묻습니다. Python 학습을 VS Code 터미널에서 시작하면 연결 종료 이후 생존을 보장하지 않습니다. 서버 독립 실행 준비는 별도 작업이며 이 이관에서는 서버를 변경하지 않았습니다.
