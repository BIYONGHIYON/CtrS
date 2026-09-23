# SSA-MRN 연구 트랙

## 목표

고해상도 RGB의 공간 정보와 저해상도 HSI의 분광 정보를 융합하여 고해상도 HSI를 복원하고, SSA-MRN의 밴드별 spectral-spatial attention과 다중 해상도 융합이 RGB 유도 HSI 초해상도에 유효한지 검증합니다.

## 현재 핵심 과제

- 기존 SSA-MRN 구조 및 공개 코드 재현
- PAN 입력을 RGB 입력으로, MS 출력을 HSI 출력으로 확장
- 합성 RGB가 아닌 실제 RGB-HSI 데이터 사용 가능성 검토
- RGB-HSI 정합 오차와 센서 차이에 대한 대응
- Bicubic, 기본 융합 모델 및 제거 실험과 비교
- PSNR, SSIM, SAM, ERGAS 기반 공간·분광 품질 평가

## 현재 보존한 코드

- `scripts/visualize_arad_hsi.py`: ARAD 형식 HSI 큐브를 CIE 1931 근사 기반 sRGB로 변환하고 밴드·분광 곡선을 시각화합니다.

이 RGB는 실제 RGB 카메라 측정값이 아니라 HSI로부터 만든 시각화용 근사 영상입니다. 실제 RGB-HSI 융합 실험의 입력으로 간주하지 않습니다.

## 중간 산출물

- 재현 가능한 데이터 준비 절차
- 최소 학습·평가 파이프라인
- 기준 모델과 SSA-MRN 확장 모델 비교표
- 실제 데이터와 합성 데이터의 차이 분석
- 실패 사례와 분광 왜곡 분석

