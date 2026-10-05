# ECRformer 겨울 절반 데이터 재현 결과

SEN12MS-CR 겨울 데이터의 업로드된 절반만 사용한 실험입니다. 전체 계절 또는 겨울 전체를 학습한 논문 결과와 직접 비교할 수 없습니다. 공식 분할의 겨울 ROI 가운데 이 실험에 포함된 패치는 학습 6,833개, 검증 1,386개, 테스트 783개입니다. 테스트 783개는 독립적인 장면 783개가 아니라 동일 테스트 ROI의 패치입니다.

## 학습과 체크포인트

- RTX A6000, 시드 42, AdamW, 학습률 4e-4, 배치 4, gradient accumulation 4, 최대 200 epochs, early stopping patience 10.
- 검증 손실 최저점은 epoch 3 (`epoch=3-step=1708.ckpt`, `valid_loss` 약 0.038). 개선이 없어 epoch 13 종료 시점에 조기 중단되었습니다. 총 14 epochs를 실행했고, 200 epochs를 완료한 것은 아닙니다.
- [`model_weights.pt`](./model_weights.pt)는 이 최고 성능 체크포인트에서 추출한 **모델 가중치 전용** 파일입니다. 원본 Lightning 체크포인트의 optimizer·scheduler 상태는 포함하지 않아 학습 재개용은 아닙니다. 원본 전체 체크포인트는 131MB로 GitHub 일반 파일 한도를 넘어 저장소에 넣지 않았습니다.
- [`hparams.yaml`](./hparams.yaml)은 서버에서 저장된 학습 설정입니다.

## 테스트 결과

최고 성능 체크포인트를 공식 `test` 분할에 적용한 783개 패치의 평균입니다. 계산은 이 저장소의 `Official_ECRformer/test.py` 및 `util/util.py` 구현을 따릅니다.

| RMSE ↓ | MAE ↓ | PSNR ↑ | SAM ↓ | SSIM ↑ | LPIPS ↓ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.03438 | 0.02544 | 29.61 dB | 11.25° | 0.84656 | 0.39683 |

- [평균 지표와 평가 설정](./summary.json)
- [샘플별 지표 783개](./metrics.csv)
- [비교 이미지 8장](./comparisons/): SAR 첫 밴드(회색조) · 구름 낀 광학 입력 · 복원 결과 · 구름 없는 정답. 광학영상은 밴드 인덱스 `(3, 2, 1)`, 밝기 배율 3.0으로 시각화한 RGB 미리보기이며, 원본 13밴드 데이터 자체는 아닙니다.

![겨울 테스트 샘플 비교](./comparisons/test_0000.png)

## 다른 환경에서 평가

원본 SEN12MS-CR TIFF 데이터는 Git에 포함하지 않았습니다. 같은 겨울 절반 데이터를 `datasets/sen12mscr_winter/`에 준비한 뒤, 아래처럼 **가중치 파일 경로를 명시**해 평가합니다. `model_weights.pt`는 `last.ckpt`처럼 학습 재개에 사용하지 마세요.

```bash
cd ECRformer/Official_ECRformer
python -m pip install -r requirements.txt rasterio
python test.py \
  --config ecrformer \
  --name winter_half1 \
  --data-root datasets/sen12mscr_winter \
  --split test \
  --ckpt-path ../reproduction/winter_half1/model_weights.pt \
  --export-format none \
  --output-dir results/winter_half1_test
```

논문과의 정량 비교는 전체 데이터 범위, 평가 분할, 전처리 및 지표 정의를 일치시킨 후 별도로 진행해야 합니다.
