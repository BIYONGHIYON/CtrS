# 겨울 기준 코드 복구 (2026-10-05)

- 브랜치: `ecrformer-winter`. 최신 main `e5c6c91`에서 분기했습니다.
- 처음에는 겨울 첫 결과 등록 커밋 `6dc9581`로 복구했고, 후속 요청으로 `966a2f2` 직전까지의 실행 기능을 다시 포함했습니다. 현재 `train.py`는 해당 직전 버전에 검증 정체 기반 스케줄러만 추가한 상태입니다.
- 당시 실행 코드의 정확한 commit ID는 기록되지 않았습니다. 따라서 이 버전은 확인 가능한 겨울 시점의 기준 코드이며, 당시 서버의 미커밋 변경까지 재현했다는 의미는 아닙니다.
- Windows 경로 처리, 가중치 전용 초기화, 고정 샘플 선택과 목록 기록, workers 0, 데이터 경로·학습량·학습률 실행 옵션을 포함합니다. 모델·전처리·기존 손실은 유지합니다. `966a2f2`의 FP32 손실 강제, 정밀도 비교, 발산 진단, 선택형 확장 hooks는 포함하지 않습니다.
- 서버의 기존 미커밋 로더 변경도 복원했습니다. 같은 계절 폴더가 두 번 중첩된 압축 해제 구조와 일반 구조를 모두 지원하고 TIFF만 인덱싱합니다. 채널 순서·정규화·영상 값 처리는 바꾸지 않습니다.
- MultiStepLR 대신 `valid_loss` 기준 ReduceLROnPlateau(factor 0.5, patience 3, min_lr 1e-7)를 사용합니다. 미개선 3회를 허용하고 다음 미개선에서 감소합니다. 초기 학습률은 0.0004 그대로이며, 논문의 patience 5 / factor 0.1 설정과는 다릅니다.
- SSA-MRN과 저장소 공통 파일은 변경하지 않았습니다. 겨울 두 번째 결과와 기존 분석 문서는 삭제하지 않고 역사 자료로 보존했습니다. 봄 안정화 코드와 오늘의 테스트 결과는 `research/ecrformer-stable-training` 브랜치에 남아 있습니다.
- 원본 데이터·서버 로그·체크포인트는 Git 복구 대상이 아닙니다. 학습은 실행하지 않았습니다.

## 서버 반영 위치

현재 작업 위치는 **`C:\CtrS\ECRformer\Official_ECRformer`**입니다. 원래 경로의 실행 코드를 겨울 브랜치 버전으로 반영합니다. 최초 복구 때 `6dc9581` 일치를 확인했지만, 실행 기능 복원 이후 더는 그 커밋과 완전히 동일하지 않습니다. 학습은 실행하지 않습니다.

`C:\CtrS` 전체의 브랜치 전환은 다른 팀의 미커밋 변경과 추적되지 않은 실행 스크립트 때문에 차단돼 진행하지 않았습니다. 따라서 **공용 저장소의 현재 브랜치 이름은 `research/ecrformer-stable-training`이지만, ECRformer 실행 코드는 겨울 버전**입니다. 이 차이는 미커밋 변경으로 표시됩니다. 이 공용 checkout에서 그대로 commit/push하거나 pull하면 안 됩니다. 다음 변경을 GitHub에 반영할 때는 겨울 브랜치와 변경 파일의 대응을 확인해야 합니다.

커밋·푸시된 봄 코드는 `research/ecrformer-stable-training`에 보존돼 있습니다. 서버 전용 미커밋 코드는 원 저장소 stash `6442ca0f32f244bbeb77816064ab7a6b99796b8c`에 추가로 보관했고, 이전 전체 ECRformer 안전 사본 `699157d94f0693a7e0aa8ffe02ae3c1e0f4b1e69`도 유지했습니다. 새 봄 전용 실행 파일은 백업 후 활성 코드 폴더에서 제외했습니다. 원본 데이터·로그·체크포인트와 기존 문서는 유지했으며 `.gitignore`와 다른 팀의 수정 파일은 전후 해시가 동일합니다.

별도 worktree `C:\CtrS-ecrformer-winter`도 삭제하지 않고 겨울 브랜치의 깨끗한 Git 작업 공간으로 유지합니다. 원래 경로에서 코드를 실행·수정할 수 있지만, Git commit/push의 대상 브랜치와는 구분해야 합니다.

## 다음 실행 전에 확인할 내용

겨울 첫 실험 설정은 `reproduction/winter_half1/hparams.yaml`에 있습니다. 당시 데이터 경로는 Linux 경로이므로 현재 서버의 실제 겨울 데이터 경로를 별도로 확인해야 합니다. 봄 데이터 경로를 겨울 데이터로 간주하면 안 됩니다.

일반 `--config ecrformer`의 기본값은 FP32, 학습률 0.0004, 배치 4 / accumulation 4, 최대 200 epoch입니다. 이전 서버 설정을 복원한 `--config ecrformer_spring`은 아래와 같습니다.

| 항목 | 서버 실행 설정 |
| --- | --- |
| 데이터 root | `E:\윤지\DataSet\ECRformer Data\SEN12MSCR_spring` |
| 고정 학습 샘플 | train 분할에서 6,000개, seed 42 |
| 최대 epoch | 100 |
| 학습 / 검증 배치 | 2 / 1 |
| accumulation / 유효 배치 | 8 / 16 |
| 정밀도 / 초기 학습률 | 16-mixed / 0.0004 |
| workers / 조기 종료 patience | 2 / 10 |

서버 설정의 경로는 봄 데이터입니다. 브랜치 이름이 winter라고 해서 데이터가 겨울로 바뀌지는 않습니다. 현재는 전체 train에서 무작위 고정 선택하며 계절 균등 선택은 구현하지 않았습니다. 사계절 경로와 세부 조건을 받은 뒤 총 6,000개를 계절별 1,500개로 선택하도록 변경할 예정입니다. validation/test는 그대로 유지합니다. 실학습의 메모리·발산·성능은 아직 검증하지 않았습니다.

`--data-root`, `--max-train-samples`, `--init-weights`, `--max-epochs`, `--num-workers`, `--lr`를 다시 지원합니다. 경로는 `--data-root`로 덮어쓸 수 있습니다. 고정 목록은 로그 폴더의 `train_subset.json`에 저장하며 목록이 달라지는 재개는 거부합니다. 기존 scheduler 상태가 들어 있는 checkpoint를 자동 재개하면 다른 scheduler와 충돌할 수 있으므로 새 실험 이름과 `--no-resume`을 사용합니다. 필요하면 정상 checkpoint에 `--init-weights`를 사용해 가중치만 가져옵니다. 발산한 봄 `last.ckpt`는 사용하지 않습니다.

실행 예시(명령을 적은 것이며 자동 실행하지 않음):

```powershell
cd C:\CtrS\ECRformer\Official_ECRformer
C:\CtrS\.venv\Scripts\python.exe -u train.py --config ecrformer_spring --name winter_base_subset6000_plateau_v1 --no-resume
```

CPU 기능 검증: `python tests/test_winter_training.py`. 이 검증은 학습·모델 생성·TIFF 로딩을 호출하지 않습니다.

검증 결과: CPU 테스트 7개 통과(일반·중첩 경로의 p100~p104 각 5개 인식 포함). 실제 서버 데이터에서 train 24,378개 중 6,000개 선택, validation 756개 유지와 선택 샘플 1개의 SAR `(2,128,128)` / cloudy·target `(13,128,128)` 로딩을 확인했습니다. Trainer.fit과 추론은 호출하지 않았고 메모리·학습 안정성 검증은 수행하지 않았습니다.
