# ECRformer 원 논문 재현 및 분광 확장 연구

## 재현 연구 현황

2026-10-06 기준으로 겨울, 봄, 봄·겨울 혼합의 세 연구 흐름을 확인했습니다. 겨울 흐름은 서버 저장공간 50GB 제한으로 데이터를 두 구간에 나눠 두 차례 실행했습니다. 봄 흐름은 고정 표본과 상세 로그로 학습 발산을 조사했고, 혼합 흐름은 논문 모델 구조를 유지한 채 두 계절 각 3,000개로 축소 실행했습니다.

**원 논문과 같은 전체 데이터·학습 조건의 재현에는 아직 성공하지 못했습니다.** 겨울 학습은 validation 정체로 일찍 종료됐고, 봄 학습은 epoch 16부터 학습·검증 오차가 함께 발산했습니다. 혼합 학습은 실행을 마쳤지만 조기 종료됐으며 논문 결과보다 낮은 test 지표를 보였습니다. exit code 0은 프로그램이 정상 종료됐음을 뜻하며, 모델 학습이 성공적으로 수렴했다는 뜻은 아닙니다.

전체 실험 순서와 논문 비교는 [재현 실험 이력](./reproduction/README.md)에서 확인할 수 있습니다.

- [1. 겨울 데이터 분할 연구](./reproduction/studies/01_winter/README.md)
- [2. 봄 6,000개 진단 연구](./reproduction/studies/02_spring_6000/README.md)
- [3. 봄·겨울 혼합 진단 연구](./reproduction/studies/03_spring_winter_mixed/README.md)


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

2026-10-06 기준으로 겨울·봄·혼합 재현을 진행했지만 원 논문과 동등한 조건의 안정적인 재현은 확인되지 않았습니다. 다음 단계는 재현 실험의 원인과 한계를 정리하고 학습 안정성을 검증하는 것입니다. SSDFL·SAM 확장은 안정적인 ECRformer 기준선을 확보한 뒤 진행합니다.

## 9. 참고 자료

- [ECRformer 공식 구현](https://github.com/zzaiyan/ECRformer)
- [ECRformer 논문](https://doi.org/10.1016/j.isprsjprs.2026.04.009)
- [SEN12MS-CR](https://patricktum.github.io/cloud_removal/sen12mscr/)
