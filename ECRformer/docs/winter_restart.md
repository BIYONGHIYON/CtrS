# 겨울 기준 코드 복구 (2026-10-05)

- 브랜치: `ecrformer-winter`. 최신 main `e5c6c91`에서 분기했습니다.
- `Official_ECRformer/`의 추적 파일을 겨울 첫 번째 기준 실험 결과가 등록된 `6dc9581cc9a096189e96205299430195da9fcca0` 버전으로 복구했습니다.
- 당시 실행 코드의 정확한 commit ID는 기록되지 않았습니다. 따라서 이 버전은 확인 가능한 겨울 시점의 기준 코드이며, 당시 서버의 미커밋 변경까지 재현했다는 의미는 아닙니다.
- 봄의 고정 샘플 선택, 가중치 전용 초기화 옵션 등 이후 학습 기능은 이 기준 코드에 포함되지 않습니다. 기존 모델·겨울 시점 전처리·손실·MultiStepLR을 유지합니다.
- SSA-MRN과 저장소 공통 파일은 변경하지 않았습니다. 겨울 두 번째 결과와 기존 분석 문서는 삭제하지 않고 역사 자료로 보존했습니다. 봄 안정화 코드와 오늘의 테스트 결과는 `research/ecrformer-stable-training` 브랜치에 남아 있습니다.
- 원본 데이터·서버 로그·체크포인트는 Git 복구 대상이 아닙니다. 학습은 실행하지 않았습니다.

## 서버 반영 위치

현재 겨울 코드 작업 위치는 **`C:\CtrS\ECRformer\Official_ECRformer`**입니다. 사용자의 추가 요청에 따라 원래 경로의 코드만 `origin/ecrformer-winter` 버전으로 교체했습니다. 해당 폴더의 추적 파일이 `6dc9581`과 동일함을 Git diff로 확인했고 Python 23개 파일 구문 검사를 통과했습니다. 학습은 실행하지 않았습니다.

`C:\CtrS` 전체의 브랜치 전환은 다른 팀의 미커밋 변경과 추적되지 않은 실행 스크립트 때문에 차단돼 진행하지 않았습니다. 따라서 **공용 저장소의 현재 브랜치 이름은 `research/ecrformer-stable-training`이지만, ECRformer 실행 코드는 겨울 버전**입니다. 이 차이는 미커밋 변경으로 표시됩니다. 이 공용 checkout에서 그대로 commit/push하거나 pull하면 안 됩니다. 다음 변경을 GitHub에 반영할 때는 겨울 브랜치와 변경 파일의 대응을 확인해야 합니다.

커밋·푸시된 봄 코드는 `research/ecrformer-stable-training`에 보존돼 있습니다. 서버 전용 미커밋 코드는 원 저장소 stash `6442ca0f32f244bbeb77816064ab7a6b99796b8c`에 추가로 보관했고, 이전 전체 ECRformer 안전 사본 `699157d94f0693a7e0aa8ffe02ae3c1e0f4b1e69`도 유지했습니다. 새 봄 전용 실행 파일은 백업 후 활성 코드 폴더에서 제외했습니다. 원본 데이터·로그·체크포인트와 기존 문서는 유지했으며 `.gitignore`와 다른 팀의 수정 파일은 전후 해시가 동일합니다.

별도 worktree `C:\CtrS-ecrformer-winter`도 삭제하지 않고 겨울 브랜치의 깨끗한 Git 작업 공간으로 유지합니다. 원래 경로에서 코드를 실행·수정할 수 있지만, Git commit/push의 대상 브랜치와는 구분해야 합니다.

## 다음 실행 전에 확인할 내용

겨울 첫 실험 설정은 `reproduction/winter_half1/hparams.yaml`에 있습니다. 당시 데이터 경로는 Linux 경로이므로 현재 서버의 실제 겨울 데이터 경로를 별도로 확인해야 합니다. 봄 데이터 경로를 겨울 데이터로 간주하면 안 됩니다.

기준 설정은 FP32, 학습률 0.0004, 배치 4 / accumulation 4, 최대 200 epoch, early stopping patience 10입니다. 당시 GPU는 RTX A6000이었으므로 8GB RTX 3060 Ti에서 동일 배치가 가능한지는 미검증입니다. 메모리 문제 시 배치·accumulation 조정은 별도 변경으로 기록합니다.

이 버전의 `train.py`는 `--config`, `--name`, `--gpu`, `--no-resume`을 지원합니다. 이후 추가된 `--data-root`, `--max-train-samples`, `--init-weights` 옵션은 지원하지 않습니다. 데이터 경로와 배치 설정을 먼저 준비한 뒤 새 실험 이름과 `--no-resume`으로 시작해야 합니다. 기존 봄 `last.ckpt`를 재개하지 않습니다.
