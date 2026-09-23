# ECRformer 연구 트랙

## 목표

광학 영상과 SAR 영상을 이용한 위성영상 구름 제거에서 ECRformer를 재현하고, Spectral-Semantic Decoupled Learning 및 spectral fidelity 제약의 효과를 검증합니다.

## 현재 핵심 과제

- ECRformer 공식 구조와 학습 과정 재현
- 구조 복원과 질감·분광 복원을 분리한 학습 전략 검토
- SEN12MS-CR 등 광학-SAR 데이터셋 준비
- 기준 모델과 동일 조건 비교
- PSNR, SSIM, SAM 및 밴드별 오류 평가
- 제안 요소 제거 실험과 계산 비용 비교

## 중간 산출물

- 재현 가능한 데이터 준비 절차
- 최소 학습·평가 파이프라인
- 기준 ECRformer와 확장 모델 비교표
- 구름 영역별 정성 결과와 실패 사례
- 분광 보존 효과 및 계산 비용 분석
