# Spectral-Semantic Decoupled Learning을 적용한 ECRformer

> 원본 문서: [Notion — Spectral-Semantic Decoupled Learning을 적용한 ECRformer](https://app.notion.com/p/Spectral-Semantic-Decoupled-Learning-ECRformer-3e313c366e45805fa96be3d38c8fc107?source=copy_link)

## 연구 요약

광학 영상과 SAR 영상을 함께 이용하는 ECRformer 기반 위성영상 구름 제거 연구입니다. 기존 ECRformer의 Structure–Texture 복원 흐름을 **Structure–Spectral–Texture**로 확장하고, Sentinel-2의 13개 밴드 사이의 관계를 보존하도록 Spectral Angle Mapper(SAM)를 학습 손실에 직접 적용했을 때의 효과를 검증합니다.

핵심 질문은 다음과 같습니다.

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

## 3. 제안 방식: SSDFL

SSDFL(Spectral-Semantic Decoupled Feature Learning)의 목표 흐름은 다음과 같습니다.

```text
Structure → Spectral → Texture
Encoder   → Decoder  → Late Decoder / Refinement
```

- **Encoder — Structure:** 지형, 건물, 도로와 경계 등 공간 구조를 학습합니다.
- **Decoder — Spectral:** 13개 밴드 사이의 관계와 spectral signature를 보존합니다.
- **후반 Decoder / Refinement — Texture:** 최종 세부 질감을 복원합니다.

기존 ECRformer의 장점을 유지하면서 분광 정보를 별도의 학습 목표로 명확히 다루는 것이 핵심입니다.

## 4. Spectral fidelity 제약

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

### 비교 모델

1. 기존 ECRformer 재현 모델
2. ECRformer + spectral branch/SSDFL
3. ECRformer + SAM loss
4. ECRformer + SSDFL + SAM loss

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
- 기존 ECRformer의 최소 학습·평가 파이프라인
- 기존 모델과 SSDFL·SAM 확장 모델의 비교표
- SAM, 밴드별 오차와 시각 품질의 관계 분석
- 구름 유형과 지표별 실패 사례
- 제거 실험 및 계산 비용 분석

현재 팀 운영은 SSA-MRN 트랙과 본 트랙을 중간까지 병렬 개발한 뒤, 데이터 현실성·재현성·정량 성능·구현 난도를 비교해 더 유망한 한 주제에 팀원 4인이 집중하는 방식입니다.

## 9. 참고 자료

- [ECRformer 공식 구현](https://github.com/zzaiyan/ECRformer)
- [ECRformer 논문](https://doi.org/10.1016/j.isprsjprs.2026.04.009)
- [SEN12MS-CR](https://patricktum.github.io/cloud_removal/sen12mscr/)
