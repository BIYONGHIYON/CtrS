# ECRformer 원 논문 재현 및 분광 확장 연구

## 최신 학습·평가 결과 (2026-10-06)

**봄·겨울 각 3,000개 패치로 학습한 혼합 데이터 실험을 완료했습니다.** 최적 검증 체크포인트인 Epoch 5를 선택해 두 계절의 test 분할을 각각 평가했습니다. 이전 겨울 half1/half2 미세조정 실험과는 별개의 실행입니다.

### 학습 조건과 종료

| 항목 | 실행 조건 |
| --- | --- |
| 서버 GPU | RTX 3060 Ti 8GB |
| 학습 데이터 | 봄 3,000 + 겨울 3,000 = 6,000개, seed 42 고정 목록 |
| 선택 전 train 패치 | 봄 24,378 / 겨울 13,995 |
| 검증 데이터 | 봄 756 + 겨울 2,771 = 3,527개 |
| 모델 / optimizer | ECRformer / AdamW, weight decay 1e-3 |
| 정밀도 / 배치 | FP16 mixed / 배치 2 × accumulation 8 = 유효 배치 16 |
| 검증 배치 / workers | 1 / 2 |
| 학습 crop | 128×128 |
| 초기 학습률 | 4e-4 |
| 학습률 감소 | 검증 loss 기준 ReduceLROnPlateau, factor 0.5, patience 3 |
| 조기 종료 | 검증 loss가 10회 연속 개선되지 않으면 종료 |
| 실제 실행 | Epoch 0–15, 총 16 epochs; 중단 후 Epoch 4부터 체크포인트 재개 |
| 최적 체크포인트 | `epoch=5-step=2250.ckpt`, 검증 loss 0.0394439 |
| 종료 | 2026-10-06 03:41 KST, 조기 종료, 프로세스 종료 코드 0 |

검증 loss는 `0.9 × MAE + 0.1 × (1 − SSIM)`입니다. Epoch 5 이후 최적값을 갱신하지 못했고, 학습률은 Epoch 10에서 2e-4, Epoch 14에서 1e-4로 감소했습니다. 지표는 매 epoch 단조롭게 개선되지 않았습니다.

| Epoch (0부터) | 검증 MAE↓ | 검증 loss↓ | 학습률 |
| ---: | ---: | ---: | ---: |
| 0 | 0.03254 | 0.04356 | 4e-04 |
| 3 | 0.03114 | 0.04027 | 4e-04 |
| 5 | 0.03013 | 0.03944 | 4e-04 |
| 10 | 0.03130 | 0.04133 | 2e-04 |
| 15 | 0.03097 | 0.04028 | 1e-04 |

![검증 MAE와 loss의 epoch별 변화](./reproduction/spring_winter6000/validation_history.png)

### 최적 체크포인트의 테스트 결과

**FP32, 배치 1, workers 2, crop 없이 256×256 원본 패치**를 평가했습니다. RMSE·MAE·PSNR·SAM·SSIM은 13밴드, LPIPS는 RGB 밴드 `(3, 2, 1)`와 VGG를 사용합니다. 지표는 패치별 계산값의 평균이며, 합계는 계절별 패치 수로 가중한 평균입니다. 검증에서는 학습과 같은 mixed precision을 사용했으므로 검증값과 테스트값의 절댓값을 직접 비교하지 않습니다.

| 테스트 범위 | 패치 수 | RMSE↓ | MAE↓ | PSNR↑ (dB) | SAM↓ (°) | SSIM↑ | LPIPS↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 봄 | 3,983 | 0.04485 | 0.03065 | 27.716 | 9.669 | 0.87421 | 0.41115 |
| 겨울 | 1,567 | 0.04570 | 0.03141 | 27.199 | 11.844 | 0.84082 | 0.43569 |
| 합계 (패치 수 가중 평균) | 5,550 | 0.04509 | 0.03086 | 27.570 | 10.283 | 0.86478 | 0.41808 |

### 논문 수치와의 참고 비교

| 평가 범위 | MAE↓ | SAM↓ (°) | PSNR↑ (dB) | SSIM↑ | LPIPS↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 논문 Table 1: SEN12MS-CR 전체 | 0.0164 | 4.693 | 33.37 | 0.932 | 0.188 |
| 이번 실행: 봄·겨울 테스트 합계 | 0.03086 | 10.283 | 27.570 | 0.86478 | 0.41808 |

논문 수치는 [저자 공개 PDF의 Table 1](https://zzaiyan.github.io/assets/pubs/ecrformer.pdf#page=7)을 확인했습니다. **평가 범위와 학습 조건이 다르므로 동일 조건의 모델 비교나 재현 성공 판정에 사용할 수 없습니다.** 논문 표에 RMSE는 없어 이 참고 표에서 제외했습니다.

![봄·겨울 테스트 비교: SAR, 구름 입력, 복원, 정답](./reproduction/spring_winter6000/comparison.png)

각 계절의 자연 정렬된 test 목록에서 **앞 두 패치**를 표시했습니다. SAR는 VV 채널 회색조, 광학 영상은 밴드 인덱스 `(3, 2, 1)`의 RGB에 밝기 배율 3.0을 적용했습니다. 일부 RGB 예시이며, 13밴드 전체의 복원 품질이나 전체 테스트의 대표 사례로 해석하지 않습니다.

표시한 겨울 패치에서는 정답에 비해 경계와 질감이 흐리고 구름 형태의 잔여 패턴이 보입니다. 구름 제거가 완전하거나 세부 구조가 충분히 복원됐다고 해석하지 않습니다.

### 데이터 분할 확인과 해석 범위

- 저장된 학습 목록은 6,000개이며 중복 sample ID가 없습니다. 재개 시 원래 TIFF 경로와 목록의 일치를 확인했습니다.
- 현재 학습·검증·테스트 사이의 **동일 sample ID 및 동일 계절/ROI 디렉터리 중복은 각 0개**입니다. [분할 점검 결과](./reproduction/spring_winter6000/split_audit.json)에 계절별 수치를 기록했습니다.
- 서로 다른 ROI 사이의 지리적 근접·공간 중첩과 이전 실험 데이터 전체의 중복은 이번 점검 범위에 포함하지 않았습니다.
- 논문은 SEN12MS-CR 전체 계절을 평가합니다. 이번 결과는 봄·겨울 데이터와 축소된 학습량을 사용하므로 **논문과 동등 조건의 재현 결과가 아닙니다.** SAR 두 채널 `[-25, 0]` dB 전처리와 학습률 조정 설정에도 차이가 있습니다.
- 이전 겨울 half1/half2와 테스트 범위가 다릅니다. 이번 결과를 이전 실험 대비 성능 향상으로 해석하지 않습니다.

### 결과 파일과 재평가

[봄 평균 지표](./reproduction/spring_winter6000/spring/summary.json) · [겨울 평균 지표](./reproduction/spring_winter6000/winter/summary.json) · [합계 지표](./reproduction/spring_winter6000/combined_summary.json) · [검증 기록](./reproduction/spring_winter6000/validation_history.csv) · [학습 샘플 목록](./reproduction/spring_winter6000/train_samples.csv) · [학습 설정](./reproduction/spring_winter6000/training_config.json) · [체크포인트 해시·평가 환경](./reproduction/spring_winter6000/provenance.json)

샘플별 지표는 `reproduction/spring_winter6000/spring/metrics.csv`와 `winter/metrics.csv`에 있습니다. 원본 TIFF와 학습 재개용 체크포인트는 서버에 보존하며 Git에 포함하지 않습니다.

서버 PowerShell에서 평가를 다시 수행할 때는 새 출력 폴더를 지정합니다.

```powershell
cd C:\CtrS-ecrformer-winter\ECRformer\Official_ECRformer
& 'C:\CtrS\.venv\Scripts\python.exe' ..\reproduction\spring_winter6000\evaluate.py `
  --checkpoint '.\experiments\ecrformer_spring_winter_3000_each_seed42_20ep\version_1\checkpoints\epoch=5-step=2250.ckpt' `
  --output '.\results\spring_winter6000_retest'
```

`evaluate.py`는 신뢰하는 서버 체크포인트의 설정과 모델을 읽어 두 계절의 test 분할을 평가합니다. 동일 데이터·저장된 학습 목록과 체크포인트가 필요합니다.


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

### 이전 겨울 재현 기록 (2026-09-29)

RTX A6000에서 SEN12MS-CR **겨울 데이터 절반**으로 ECRformer 기준 모델을 학습했습니다. 200 epochs를 설정했으나 검증 손실이 개선되지 않아 총 14 epochs 후 조기 종료되었고, 최고 성능 가중치(epoch 3)로 학습에 쓰지 않은 겨울 테스트 패치 783개를 평가했습니다. PSNR 29.61 dB, SSIM 0.84656, SAM 11.25°입니다. 평균·샘플별 지표, 모델 가중치 및 입력/복원/정답 비교 이미지는 [겨울 절반 재현 결과](./reproduction/winter_half1/README.md)에 있습니다. 아직 **논문 수치와 동등 조건의 비교를 완료한 것은 아닙니다**.


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
