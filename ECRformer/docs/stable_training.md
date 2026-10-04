# 발산 진단용 안정화 실행 (아직 실학습 미검증)

`Official_ECRformer/train_stable.py`는 기존 모델·SEN12MS-CR 전처리·증강을 유지하는 선택형 실행 진입점입니다. 기존 `train.py` 실행은 기본 설정과 스케줄러를 그대로 사용합니다. 서버의 기존 spring config와 상세 로그 변경은 덮어쓰지 않습니다.

## 적용 내용

- 기본 학습률 0.0004 → 0.0001, 배치 2 / accumulation 8 유지.
- 고정 epoch 120 이후 감소 대신, 검증 `valid_loss`를 감시하는 ReduceLROnPlateau 사용. factor 0.5, patience 3(미개선 3회 허용, 그 다음 미개선에서 감소), 최솟값 1e-7. 조기 종료 patience는 10 유지.
- FP16 모델 연산 유지, 주/다중 스케일 L1·SSIM 학습 손실과 검증 지표 계산은 autocast를 해제해 FP32로 수행. `--precision 32-true`로 전체 FP32 학습 비교 가능.
- 모든 학습 배치의 경로, 증강 후 입력·정답·예측 범위 및 유한값 여부, 주·보조 손실, 학습률을 JSONL에 기록. optimizer 업데이트 직전 unscale 후 clipping 전후 gradient norm 기록.
- 200배치마다 동일 입력을 eval 상태의 FP32·FP16으로 추가 추론해 출력 차이 기록. 모델 모드·RNG 상태를 복원하며 추가 추론은 no_grad. BatchNorm 통계를 변경하지 않음. `--probe-every 0`으로 끌 수 있음. CPU에서는 비교 생략 기록.
- 비유한 입력·예측·손실 또는 업데이트 직전 gradient는 기록 후 해당 실험을 오류로 중단. FP16에서 GradScaler가 복구할 수 있는 overflow도 중단될 수 있는 보수적 진단 정책임.
- 기존 checkpoint의 optimizer 상태를 자동 재개하지 않음. 필요하면 `--init-weights`로 정상 checkpoint의 가중치만 로딩하여 새 optimizer·scheduler로 시작.

로그: 새 실험의 `version_*\stability_diagnostics.jsonl`. JSON에는 tensor가 아닌 scalar만 기록합니다. 추가 추론·GPU 동기화·매 배치 디스크 쓰기에 따른 실행 시간 증가가 있으며, 기존 로그도 함께 생성됩니다. grad norm은 accumulation 전체의 값이며 optimizer 업데이트 단위이지 매 배치 값이 아닙니다.

동일 입력 eval 비교는 FP16 영향의 단서를 주지만 학습 gradient 및 업데이트의 정밀도 영향을 분리하는 실험은 아닙니다. 학습률·스케줄러·손실 정밀도를 동시에 바꾸는 첫 실행에서 개선되더라도 특정 변경의 효과를 확정할 수 없습니다. 별도 동일 초기 상태·데이터·seed FP32/FP16 학습 비교가 필요합니다.

## 다음 실행 (자동 시작·예약하지 않음)

다른 사용자의 학습이 끝나고 GPU가 비어 있는 것을 확인한 뒤, Windows 서버에서 다음 명령을 수동 실행합니다. 기존 실험과 다른 이름을 사용하고 실제 데이터 root를 명시합니다.

```powershell
cd C:\CtrS\ECRformer\Official_ECRformer
$springRoot = (C:\CtrS\.venv\Scripts\python.exe -c "from config.ecrformer_spring_config import EcrformerSpringConfig; print(EcrformerSpringConfig().dataset.root)")
C:\CtrS\.venv\Scripts\python.exe -u train_stable.py --data-root "$springRoot" --name spring_stable6000_v1 --max-train-samples 6000 --max-epochs 100
```

`$springRoot` 행은 서버 로컬 spring 설정에서 경로를 읽습니다. 다른 서버에서는 실제 데이터 경로를 `--data-root`에 직접 지정합니다.

초기화를 맞춘 후속 비교에서는 다른 이름의 실험에 `--precision 32-true`를 추가합니다. 필요하면 첫 실행과 비교 실행 모두에 동일한 `--init-weights <정상 checkpoint>`를 지정합니다. 발산한 `last.ckpt` 상태는 사용하지 않습니다.

CPU 전용 단위 검증은 `python tests/test_stable_training.py`입니다. 실데이터 학습이나 `Trainer.fit`은 호출하지 않습니다. 실제 GPU의 FP16/FP32 비교와 성능 개선은 다음 실행까지 미검증입니다.
