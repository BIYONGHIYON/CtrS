# SSA-MRN · RGB 유도 HSI 초해상도

고해상도 RGB와 저해상도 HSI를 결합해 **204밴드 HSI를 공간 ×4 복원**합니다. LIB-HSI 관측 영상을 합성 축소한 연구이며, 실제 센서의 HR-HSI 정답 성능과 구분합니다.

| 문서 | 내용 |
|---|---|
| [연구 설명](docs/research.md) | 현재 알고리즘, 분광 압축·공동 디코더·LR 보정, 손실과 평가 |
| [재현 연구](docs/reproduction.md) | PAN–MS K4·K6의 변경점, 수치와 결과 이미지 |
| [이전 실험](docs/previous_experiments.md) | RGB01~11 변경 이력과 비교, 검증 전용 예비실험 |

## 최신 결과 · RGB11 경계 손실

RGB07 계열의 **17특징·공동 디코더·23탭·LR 평균 보정**에 HSI 경계 손실을 추가했습니다. 동일 시작 가중치에서 대조군과 네 강도를 각각 저학습률로 10에폭 학습한 뒤, validation 45장면의 MSE로 후보를 선택했습니다.

| test 75장면 · 300타일 | MSE ↓ | PSNR dB ↑ | SAM ° ↓ |
|---|---:|---:|---:|
| 23탭 + LR 보정 | 0.0011539606 | 30.0649 | 2.44504 |
| 동일 학습량 대조군 | 0.0004378418 | 34.3916 | 2.15071 |
| **RGB11 · 선택 후보** | **0.0004376169** | **34.3948** | **2.14986** |
| 이전 공개 RGB09 | 0.0004371593 | 34.3929 | 2.15321 |

대조군보다 **PSNR +0.0032 dB, SAM −0.00085°**로 이득이 작습니다. 같은 타일 평가를 사용한 최근 완료 모델 중 PSNR·SAM은 가장 좋지만 RGB09보다 MSE는 약 0.105% 높습니다. 모든 지표에서 우세하거나 흐릿함을 해결했다고 결론짓지 않습니다.

**[204밴드 웹뷰어 열기](https://biyonghiyon.github.io/CtrS/ssa-mrn/)** · [상세 보고서: 변경점·차이·5장면](docs/experiments/rgb_hsi/rgb11_gradient_suite.md) · [오프라인 HTML](docs/assets/rgb11_gradient_suite/band_viewer/index.html)

왼쪽부터 **LR HSI · RGB 입력 · 예측 · 정답**입니다. 동일한 HSI 대비를 적용했으며 지표는 전체 204밴드로 계산합니다.

![RGB11 대표 결과](docs/assets/rgb11_gradient_suite/sample_01.png)

![RGB11 test 비교](docs/assets/rgb11_gradient_suite/test_metrics.png)

## 진행 및 보관

RGB11 다섯 학습은 2026-10-06 19:21:59(KST)에 모두 정상 종료됐습니다. 이 결과 정리에서 새 학습은 시작하지 않았습니다. 가중치와 전체 수치의 위치·해시는 각 실험 보고서에 기록합니다.

[문서·파일 위치 안내](docs/structure.md) · [공통 보고서 양식](docs/experiments/template.md) · [작업 규칙](AGENTS.md)
