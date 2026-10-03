# RGB 3경로 × 12그룹 및 타일 평가 실험

기존 grouped12/K=4/23탭 모델과 별도 모델·설정을 사용합니다. 기존 가중치로 재개할 수 없습니다.

## 구조

204밴드를 연속 17밴드씩 나누는 공유 Encoder(204→12)는 유지합니다.
R·G·B 각각 독립 SSA-MRN 코어가 **12개 특징 전체**를 처리합니다.
각 코어는 256/128/64 해상도에서 scalar guide와 12분기를 융합합니다.
각 경로의 독립 grouped Decoder(12→204)가 204밴드 보정 영상을 만듭니다.
세 보정 영상을 밴드별 1×1 convolution으로 융합하고 23탭 확대 HSI에 더합니다.
융합 가중치는 R/G/B 각각 1/3으로 시작하고 함께 학습합니다.

- K=4, area x4 열화, 23탭 확대, homography manifest 유지
- 파라미터: 기존 652,312 → 새 모델 1,950,186
- 그룹당 하나인 Encoder 특징은 유지하나, 최종 보정에는 R/G/B의 서로 다른 공간 패턴 3개 사용
- RGB 각각의 경로를 학습하는 것이며 세 이미지 단순 합산으로 밝기를 3배 만드는 방식이 아님

## 공간 크기와 평가

학습은 원본 512×512의 비중첩 256×256 4개 영역입니다.
검증·시험도 동일한 4개 영역을 잘라 사용하며 전체 시야를 먼저 축소하지 않습니다.
LR HSI는 각 타일의 area x4 평균으로 만든 64×64입니다.
학습에서만 공동 flip을 사용하며 검증·시험에는 증강하지 않습니다.

- train: 393장 × 4 = 1,572타일
- validation: 45장 × 4 = 180타일
- test: 75장 × 4 = 300타일
- 타일 SSE·SAM을 장면별로 모아 장면 평균 평가; 독립 시험 단위는 75장
- 유효 테두리 마스크와 HR HSI 기반 정합 한계는 기존과 동일

기존 full256 시험은 물체 크기·평가 시야가 달라 수치를 직접 비교하지 않습니다.
구조의 효과를 검증하려면 이전 모델도 같은 타일 프로토콜로 평가하고,
완전한 학습 비교에서는 이전 모델도 같은 타일 검증으로 best를 선택해야 합니다.

## 실행

설정: `configs/lib_rgb_hsi_triple12_k4_23tap_tiles.json`
결과: `experiments/checkpoints/lib_rgb_hsi_triple12_k4_23tap_tiles`

VS Code 실행 목록:

- `LIB: triple12 tiles smoke`
- `LIB: triple12 tiles train (new run)`
- `LIB: triple12 tiles evaluate best`

CPU 소형 입력에서 forward/backward, 3경로 전체 gradient, scalar guide의 벡터화와
12분기 반복 연산 일치, 밴드별 융합 순서를 검증했습니다.
실제 CUDA 256×256 smoke와 전체 학습은 아직 실행하지 않았습니다.
연산·활성화 메모리가 늘어나므로 기존 batch 4가 GPU 8GB에 맞는지는 실제 smoke로 확인해야 합니다.
이번 변경만으로 선명도·PSNR 개선을 주장하지 않습니다.
