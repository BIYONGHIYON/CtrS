# LIB-HSI 정합 보정 64→256 가중치

100 epoch 학습 결과를 공개했습니다. 검증 장면 평균 MSE가 최소인 98 epoch `best.pt`를
README의 시험 평가와 이미지 생성에 사용했습니다. `latest.pt`는 100 epoch 최종 상태입니다.
두 파일에는 모델·optimizer·AMP scaler·RNG·학습 설정이 포함돼 있습니다.
학습 시점의 로컬 경로가 설정에 남아 있으므로 시험 실행 시 아래 공유 설정과 `--data-root`를 사용합니다.
파일 SHA256은 `checksums.json`에 기록했습니다.

저장소 루트에서 필요한 PyTorch와 `SSA-MRN/requirements.txt` 의존성을 설치하고,
LIB-HSI의 `train/validation/test` 디렉터리 구조를 유지합니다. 원본 데이터는 Git에 포함하지 않았습니다.

```powershell
python -m pip install -r SSA-MRN/requirements.txt
python SSA-MRN/scripts/train_lib.py --config SSA-MRN/configs/lib_rgb_hsi_aligned_256_test.json --data-root "C:/your-data/LIB-HSI" --checkpoint SSA-MRN/experiments/checkpoints/lib_rgb_hsi_aligned_256_b4/best.pt --evaluate
```

CUDA가 없는 PC에서는 위 명령에 `--device cpu`를 추가합니다. 공유 설정은 worker 0으로
구성했습니다. 결과는 `SSA-MRN/experiments/checkpoints/lib_rgb_hsi_aligned_256_b4/test/`에 저장됩니다.

단일 시험 타일의 5개 패널 비교는 다음 명령으로 생성합니다.

```powershell
python SSA-MRN/scripts/preview_lib.py --checkpoint SSA-MRN/experiments/checkpoints/lib_rgb_hsi_aligned_256_b4/best.pt --data-root "C:/your-data/LIB-HSI" --split test --sample-index 0 --output-dir SSA-MRN/experiments/results/my_lib_preview
```

정합 manifest는 저장소에 포함했습니다. HSI를 90도 회전한 원본 좌표를 유지하고 RGB만
고정된 정수 평행이동으로 보정합니다. 시험에서 보정 테두리를 제외합니다. HSI GT를 사용한
정합과 합성 bicubic x4 열화 조건의 결과이며 실제 저해상도 센서 성능으로 일반화하지 않습니다.
공유한 manifest나 패치·모델 구조를 변경하면 동일 조건 시험이 아니며 학습 스크립트가
정합 해시 또는 구조 불일치를 거절합니다.

검증 환경은 PyTorch 2.7.1이었습니다. CPU 평가는 float32, GPU 평가는 AMP로 실행하므로 지표에 작은 수치 차이가 있을 수 있습니다. GPU·CPU 각각 첫 시험 장면의 실행을 확인했습니다.
