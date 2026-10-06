# 이전 실험 · RGB–HSI

[최신 결과](../README.md) · [연구 설명](research.md) · [재현 연구](reproduction.md)

보고서는 모두 **목적 → 변경·조건 → 수치 → 이전 대비 → 그래프 → 이미지 → 가중치·근거 → 판단** 순서입니다. 이름을 누르면 해당 실험의 수치와 결과 이미지를 함께 볼 수 있습니다.

## 전체 test 평가 이력

LIB-HSI test 75장면 평균입니다. 초기 실험의 정합·타일 조건이 달라 전체 표를 하나의 성능 순위로 읽지 않습니다. 특히 RGB03의 full256과 RGB04 이후 tiles는 서로 다른 평가입니다.

| 연구 | 바꾼 점 | PSNR dB ↑ | SAM ° ↓ | 비교에서 확인할 점 |
|---|---|---:|---:|---|
| [RGB01](experiments/rgb_hsi/rgb01_latent8.md) | 8특징·Bicubic·128 타일 | 32.3048 | 2.3161 | 최초 RGB–HSI 기준 |
| [RGB02](experiments/rgb_hsi/rgb02_aligned256.md) | 정합 보정·256 타일 | 32.4817 | 2.3121 | 정합·크기 동시 변경 |
| [RGB03](experiments/rgb_hsi/rgb03_grouped12.md) | grouped12·23탭 | 30.7574 | 2.0893 | full256 평가 |
| [RGB04](experiments/rgb_hsi/rgb04_triple12.md) | RGB별 12특징·타일 평가 | 34.0530 | 2.2057 | 구조·평가 동시 변경 |
| [RGB05](experiments/rgb_hsi/rgb05_triple12_bilinear.md) | 내부 Bilinear | 33.7897 | 2.2190 | RGB04 대비 악화 |
| [RGB06](experiments/rgb_hsi/rgb06_triple17_spectral.md) | 17특징·분광 손실·23탭 복구 | 34.0611 | 2.2006 | 여러 변경 포함 |
| [RGB06 + LR 보정](experiments/rgb_hsi/rgb06_lr_consistency.md) | 관측 LR 평균과 일치 | 34.2173 | 2.1625 | 같은 가중치 후처리 |
| [RGB07](experiments/rgb_hsi/rgb07_joint17_consistency.md) | 공동 디코더 | 34.2831 | 2.1570 | 후속 기본 구조 |
| [RGB08](experiments/rgb_hsi/rgb08_gated_detail.md) | RGB 고주파 경로 | 34.2533 | 2.1593 | 100에폭에서 이득 없음 |
| [RGB09](experiments/rgb_hsi/rgb09_detail_finetune.md) | 저학습률 추가 학습 | 34.3929 | 2.1532 | 평균 MSE는 RGB11보다 낮음 |
| [RGB11](experiments/rgb_hsi/rgb11_gradient_suite.md) | RGB07 계열 + 경계 손실 | 34.3948 | 2.1499 | 대조군 대비 이득 작음 |

## 검증 전용 실험

다음은 validation으로 후보를 고른 기록입니다. test 이미지가 없는 이유를 각 문서에 명시하며, 위 test 수치와 섞어 비교하지 않습니다.

| 연구 | 목적 | 후속 연결 |
|---|---|---|
| [6단계 예비실험 · train155](experiments/pilot/pilot_subset155.md) | 작은 학습 데이터로 구조 후보 탐색 | 전체 데이터 후보 선별 |
| [30에폭 후속 비교](experiments/pilot/pilot_followup.md) | 원본 학습과 미세조정 구분 | RGB08·09 대조 설계 |
| [RGB10](experiments/rgb_hsi/rgb10_warm07_control.md) | RGB07에도 저학습률 추가 학습 | RGB11의 시작 가중치 선택 |

34특징 실험은 중단 후 요청에 따라 결과를 삭제했으며 검증된 최종 test 수치가 없습니다. 완료 연구로 표시하지 않습니다.

## 보존 자료

각 보고서 7절에서 전체 수치·가중치 경로·해시·장면 선택 기록을 확인할 수 있습니다. 이전 204밴드 뷰어는 실험별 오프라인 HTML로 보존하고, [공개 Pages](https://biyonghiyon.github.io/CtrS/ssa-mrn/)에는 RGB11 한 연구만 표시합니다.

[폴더 구조와 보존 기준](structure.md) · [새 결과 보고서 양식](experiments/template.md)
