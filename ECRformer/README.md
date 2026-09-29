# ECRformer 원 논문 재현 및 분광 확장 연구

> 원본 문서: [Notion — Spectral-Semantic Decoupled Learning을 적용한 ECRformer](https://app.notion.com/p/Spectral-Semantic-Decoupled-Learning-ECRformer-3e313c366e45805fa96be3d38c8fc107?source=copy_link)

## 현재 연구 단계

ECRformer 담당 팀원 2명이 SSA-MRN 팀과 동시에 **ECRformer 원 논문 재현**을 진행합니다. 공개 구현과 논문의 데이터 처리, 모델, 학습·평가 조건을 확인하고 기준 성능을 확보합니다. Spectral-Semantic Decoupled Learning과 SAM 손실을 이용한 확장은 원본 모델의 재현 결과를 확인한 뒤 실험합니다.

## 연구 요약

광학 영상과 SAR 영상을 함께 이용하는 ECRformer 기반 위성영상 구름 제거 연구입니다. 먼저 원 논문의 Structure–Texture 복원 성능을 재현합니다. 이후 Structure–Spectral–Texture 확장과 Sentinel-2의 13개 밴드 관계를 보존하는 Spectral Angle Mapper(SAM) 손실의 효과를 검토합니다.

후속 확장의 핵심 질문은 다음과 같습니다.

> 기존 ECRformer에 분광 정보 학습과 spectral fidelity 제약을 명시적으로 추가하면, 시각적 복원 품질을 유지하면서 구름 제거 결과의 분광 충실도를 더 높일 수 있는가?

## 1. 연구 배경

구름 제거 결과는 RGB로 보았을 때 자연스러워도 13개 spectral band의 관계가 실제 구름 없는 영상과 다를 수 있습니다. 분광 관계가 왜곡되면 식생이나 토지 피복의 특성이 부정확해지고 복원 영상을 후속 원격탐사 분석에 사용하기 어려워집니다.

따라서 시각적 품질뿐 아니라 다음을 함께 보존해야 합니다.

- 지형, 건물, 도로와 경계의 공간 구조
- Sentinel-2 밴드 사이의 spectral signature
- 표면의 세부 질감

## 2. 기존 ECRformer

기존 ECRformer는 광학 영상과 SAR 영상을 입력으로 사용해 구름 없는 13밴드 광학 영상을 복원합니다.

```text
Structure → Texture
Encoder   → 구조 복원
Decoder   → 질감 복원
Output    → Cloud-free 13-band image
```

논문의 SDFL(Semantic-Decoupled Feature Learning)은 구조 복원과 질감 렌더링의 역할을 분리해 학습합니다. 이 연구에서는 해당 분리 학습에 분광 정보 보존 단계를 명시적으로 추가합니다.

## 3. 재현 이후 제안 방식: SSDFL

SSDFL(Spectral-Semantic Decoupled Feature Learning)의 목표 흐름은 다음과 같습니다.

```text
Structure → Spectral → Texture
Encoder   → Decoder  → Late Decoder / Refinement
```

- **Encoder — Structure:** 지형, 건물, 도로와 경계 등 공간 구조를 학습합니다.
- **Decoder — Spectral:** 13개 밴드 사이의 관계와 spectral signature를 보존합니다.
- **후반 Decoder / Refinement — Texture:** 최종 세부 질감을 복원합니다.

기존 ECRformer의 장점을 유지하면서 분광 정보를 별도의 학습 목표로 명확히 다루는 것이 핵심입니다.

## 4. 재현 이후 Spectral fidelity 제약

### SAM

SAM(Spectral Angle Mapper)은 대응하는 픽셀의 분광 벡터 사이 각도를 계산해 spectral fidelity를 평가합니다. 각도가 작을수록 복원 결과의 분광 형태가 정답에 가깝습니다.

본 연구에서는 SAM을 평가 지표로만 사용하지 않고 학습 손실에 직접 포함합니다.

```text
L_total = L_reconstruction + λ_sam · L_SAM
```

기본 재구성 손실에는 L1을 사용하고 SAM 손실의 가중치 `λ_sam`은 검증 세트에서 정합니다. 수치 안정성을 위해 분모에 작은 `ε`를 두고 cosine 값을 유효 범위로 제한합니다.

### 기대 효과

- 식생과 토지 피복의 분광 특성을 더 정확히 보존
- 복원 영상의 각 밴드 관계를 GT에 가깝게 유지
- 시각적으로 자연스러우면서 spectral analysis에도 적합한 결과 생성

## 5. 데이터셋

주 데이터셋은 **SEN12MS-CR**입니다.

- Sentinel-1 SAR와 Sentinel-2 광학 영상을 포함한 다중모달 구름 제거 데이터
- 입력: 구름이 포함된 광학 영상 + SAR 영상
- 정답: 대응하는 구름 없는 광학 영상
- 출력: 구름이 제거된 13밴드 광학 영상

데이터 링크:

- [SEN12MS-CR 프로젝트 페이지](https://patricktum.github.io/cloud_removal/sen12mscr/)
- [TUM 데이터 다운로드](https://dataserv.ub.tum.de/index.php/s/m1554803)

데이터 분할은 장면 단위로 수행해 같은 지역의 패치가 학습과 시험 세트에 동시에 들어가는 누수를 방지합니다.

## 6. 실험 계획

### 겨울 절반 데이터 재현 결과 (2026-09-29)

RTX A6000에서 SEN12MS-CR **겨울 데이터의 약 절반**으로 ECRformer 기준 모델을 학습했습니다. 공식 ROI 분할 중 확보한 패치는 학습 6,833개, 검증 1,386개, 테스트 783개입니다. 테스트 783개는 서로 독립적인 위성 장면이 아니라 **겨울 테스트 ROI 한 곳의 패치**입니다.

시드 42, AdamW, 학습률 4e-4, 배치 4와 gradient accumulation 4(유효 배치 16), 최대 200 epochs 및 early stopping patience 10으로 실행했습니다. 검증 손실의 최저점은 epoch 3(`valid_loss` 약 0.038)이었고, 10회 연속 개선이 없어 epoch 13 종료 시점에 중단됐습니다. **총 14 epochs를 실행했으며 200 epochs를 완료한 것은 아닙니다.** [서버에서 저장한 학습 설정](./reproduction/winter_half1/hparams.yaml)을 함께 보관합니다.

최고 성능 가중치로 겨울 `test` 분할의 783개 패치를 평가했습니다. 아래 논문 수치는 [ECRformer 논문 Table 1의 SEN12MS-CR 전체 테스트 결과](https://zzaiyan.github.io/assets/pubs/ecrformer.pdf)입니다. `차이`는 현재 값에서 논문 값을 뺀 것으로, **평가 범위가 달라 동일 조건의 재현 오차를 뜻하지 않습니다.**

| 지표 | 논문 ECRformer | 현재 겨울 절반 실험 | 차이 (현재 − 논문) |
| --- | ---: | ---: | ---: |
| MAE ↓ | 0.0164 | 0.02544 | +0.00904 |
| SAM ↓ | 4.693° | 11.250° | +6.557° |
| PSNR ↑ | 33.37 dB | 29.61 dB | −3.76 dB |
| SSIM ↑ | 0.932 | 0.84656 | −0.08544 |
| LPIPS ↓ | 0.188 | 0.39683 | +0.20883 |

현재 실험의 RMSE는 0.03438입니다(논문 Table 1에는 RMSE가 없어 차이값을 계산하지 않았습니다). [평균 지표와 평가 설정](./reproduction/winter_half1/summary.json), [783개 샘플별 지표](./reproduction/winter_half1/metrics.csv), [비교 이미지 8장](./reproduction/winter_half1/comparisons/)과 [평가용 모델 가중치](./reproduction/winter_half1/model_weights.pt)를 저장소에 포함했습니다. 이 가중치는 최고 성능 Lightning 체크포인트에서 모델 텐서만 추출한 약 45MB 파일로, optimizer·scheduler 상태가 없어 학습 재개용은 아닙니다. 131MB 원본 체크포인트와 원본 TIFF 데이터는 Git에 포함하지 않았습니다.

![테스트 패치 예시: SAR · 구름 낀 광학 입력 · 복원 결과 · 구름 없는 정답](./reproduction/winter_half1/comparisons/test_0000.png)

그림의 SAR는 첫 밴드를 회색조로, 세 광학영상은 Sentinel-2 밴드 인덱스 `(3, 2, 1)`에 밝기 배율 3.0을 적용해 표시한 RGB 미리보기입니다. 13밴드 원본 영상이나 전체 테스트의 대표 사례로 해석하면 안 됩니다.

### 겨울 두 번째 부분 미세조정 결과 (2026-09-29)

첫 번째 부분의 [모델 가중치](./reproduction/winter_half1/model_weights.pt)에서 새 optimizer로 시작해 겨울 데이터의 두 번째 부분으로 미세조정했습니다. 학습률은 `1e-4`, 시드는 42, 배치 4와 gradient accumulation 4(유효 배치 16)입니다. 두 번째 부분에서 학습 7,162개, 검증 1,385개, 테스트 784개가 인식됐습니다. 검증 손실이 epoch 2에서 최저점(`0.043013`)을 기록했고, 이후 10회 개선이 없어 epoch 12 종료 시 조기 중단됐습니다. 총 13 epochs를 실행했습니다. [학습 설정](./reproduction/winter_half2/hparams.yaml)을 함께 보관합니다.

아래 수치는 **두 번째 부분의 동일한 `test` 패치 784개**를 같은 평가 코드와 설정으로 측정한 결과입니다. 미세조정 전에는 첫 번째 부분의 가중치를, 이후에는 epoch 2 최적 체크포인트의 가중치를 사용했습니다. 이 테스트 패치는 한 겨울 테스트 ROI의 패치이며 독립 장면 784개가 아닙니다.

| 지표 | 미세조정 전 | 미세조정 후 | 변화 (후 − 전) |
| --- | ---: | ---: | ---: |
| RMSE ↓ | 0.05518 | 0.05146 | −0.00372 |
| MAE ↓ | 0.03932 | 0.03661 | −0.00271 |
| PSNR ↑ | 25.245 dB | 25.888 dB | +0.643 dB |
| SAM ↓ | 11.478° | 10.779° | −0.698° |
| SSIM ↑ | 0.81873 | 0.82994 | +0.01121 |
| LPIPS ↓ | 0.48422 | 0.47428 | −0.00994 |

[미세조정 전 요약](./reproduction/winter_half2/before_summary.json), [후 요약](./reproduction/winter_half2/after_summary.json), 샘플별 지표 ([전](./reproduction/winter_half2/before_metrics.csv)·[후](./reproduction/winter_half2/after_metrics.csv)), [평가용 가중치](./reproduction/winter_half2/model_weights.pt)를 저장했습니다. 가중치 파일은 optimizer 상태가 없는 평가용 파일이며 원본 Lightning 체크포인트와 TIFF 데이터는 저장소에 포함하지 않았습니다.

![두 번째 부분 테스트 패치 4개의 SAR, 구름 입력, 미세조정 전후 복원 결과와 정답](./reproduction/winter_half2/comparisons/before_after_0000_0003.png)

이미지는 테스트 패치 0–3의 시각적 예시입니다. SAR는 첫 밴드를 회색조로, 광학영상은 밴드 인덱스 `(3, 2, 1)`에 밝기 배율 3.0을 적용해 표시했습니다. 전체 테스트의 대표 사례나 13밴드 원본으로 해석하면 안 됩니다. 코드의 `train`·`val`·`test` ROI와 두 부분의 저장된 테스트 샘플 ID에는 중복이 없었습니다. 다만 첫 번째 부분의 전체 학습 샘플 명세가 없어 **두 부분 전체의 중복 여부는 확정하지 못했습니다.**

두 번째 부분 원본 TIFF가 `Official_ECRformer/datasets/sen12mscr_winter/`에 있을 때, 후 지표는 다음 명령으로 재계산할 수 있습니다.

```bash
cd ECRformer/Official_ECRformer
python test.py --config ecrformer \
  --data-root datasets/sen12mscr_winter --split test \
  --ckpt-path ../reproduction/winter_half2/model_weights.pt \
  --export-format none --output-dir results/winter_half2_after_finetune
```

논문과 현재 실험은 데이터 범위 외에도 조건 차이가 있습니다. 논문은 검증 손실이 5 epochs 개선되지 않으면 학습률을 0.1배로 낮추지만, 현재 공개 코드의 학습률 스케줄러는 정해진 epoch에서만 낮추므로 이번 조기 종료 전에는 학습률 감소가 없었습니다. 논문의 SAR 전처리는 VV를 `[-25, 0]` dB, VH를 `[-35, 0]` dB로 자르지만 현재 데이터 로더는 두 채널 모두 `[-25, 0]` dB로 처리합니다. 따라서 현 수치만으로 논문 모델의 성능을 재현했다고 결론 내릴 수 없으며, 전체 데이터와 전처리·학습 조건을 맞춘 뒤 다시 비교해야 합니다.

같은 겨울 절반 데이터가 `Official_ECRformer/datasets/sen12mscr_winter/`에 준비된 경우 다음과 같이 지표를 재계산할 수 있습니다.

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

### 원 논문 재현

1. [공식 구현](https://github.com/zzaiyan/ECRformer)과 논문의 모델·학습 설정을 대조합니다.
2. SEN12MS-CR 데이터 구성과 분할을 확인해 원본 모델을 학습·평가합니다.
3. 논문과 동일한 조건의 성능을 기록하고 차이가 나는 설정 및 실패 사례를 정리합니다.

### 비교 모델

원본 ECRformer를 기준 모델로 재현한 뒤, 후속 실험에서 다음 변형을 비교합니다.

1. ECRformer + spectral branch/SSDFL
2. ECRformer + SAM loss
3. ECRformer + SSDFL + SAM loss

### 평가 지표

- L1 또는 MAE: 픽셀 단위 복원 오차
- PSNR: 전체적인 수치 복원 품질
- SSIM: 구조적 유사도
- SAM: 픽셀별 분광 벡터의 각도, 낮을수록 좋음
- 밴드별 MAE/PSNR: 특정 밴드의 성능 저하 확인
- 계산 비용: 파라미터 수, 추론 시간과 GPU 메모리

PSNR·SSIM 개선과 SAM 개선을 따로 확인합니다. SAM만 좋아지고 시각적 선명도가 떨어지거나, 반대로 RGB 결과만 좋아지고 분광 관계가 악화되는 경우를 모두 분석합니다.

### 제거 실험

- spectral 단계 제거
- SAM loss 제거
- SAM loss 가중치 변화
- Structure–Spectral–Texture 단계별 기여도 비교

### 정성 분석

- 얇은 구름과 두꺼운 구름 영역 비교
- 건물·도로·해안선·농경지 경계 복원 비교
- RGB 합성 결과와 개별 spectral band를 함께 시각화
- 대표 픽셀의 GT·기존 모델·제안 모델 분광 곡선 비교

## 7. 연구 주장 범위와 주의점

- ECRformer 자체를 새로 제안하는 연구가 아니라 분광 충실도를 명시적으로 강화하는 확장 연구입니다.
- SAM을 손실로 추가하는 것만으로 충분한 차별성이 있는지는 선행연구와 비교해야 합니다.
- 분광 손실이 과도하면 구조와 질감이 흐려질 수 있으므로 가중치 조절이 중요합니다.
- SAR와 광학 영상의 시점 차이와 정합 오차가 결과에 영향을 줄 수 있습니다.
- 개선 효과는 같은 분할과 학습 조건에서 기존 ECRformer와 비교합니다.

## 8. 중간 산출물

- 재현 가능한 SEN12MS-CR 데이터 준비 절차
- 원 논문 설정에 따른 ECRformer 학습·평가 파이프라인과 성능 비교표
- 후속 SSDFL·SAM 확장 모델의 비교표
- SAM, 밴드별 오차와 시각 품질의 관계 분석
- 구름 유형과 지표별 실패 사례
- 제거 실험 및 계산 비용 분석

현재는 두 팀이 2명씩 나뉘어 SSA-MRN과 ECRformer 원 논문을 동시에 재현합니다. 중간 시점에 두 팀의 재현 결과와 후속 연구 가능성을 비교해 최종 주제 하나를 선택하고, 이후 팀원 4명이 함께 개발합니다.

## 9. 참고 자료

- [ECRformer 공식 구현](https://github.com/zzaiyan/ECRformer)
- [ECRformer 논문](https://doi.org/10.1016/j.isprsjprs.2026.04.009)
- [SEN12MS-CR](https://patricktum.github.io/cloud_removal/sen12mscr/)
