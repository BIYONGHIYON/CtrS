# SSA-MRN · RGB 유도 HSI 초해상도

고해상도 RGB와 저해상도 HSI에서 ×4 HSI 복원을 연구합니다. 아래 결과는 LIB-HSI의 관측 HSI를 합성 축소한 평가이며 실제 센서의 HR-HSI 정답 성능과 구분합니다.

## 연구 개요

204밴드 HSI의 보간 결과를 유지하면서, 인접한 12밴드씩 학습 압축한 17특징에서 공간 보정량을 추정합니다. R·G·B 각각의 독립 SSA-MRN 경로와 밴드별 학습 fusion으로 보정량을 결합합니다. 내부 평균 축소·23탭 확대를 사용하고, MSE에 분광 방향 손실(1−cos)을 λ=0.01로 더해 학습했습니다.

[연구 설명 · 재현부터 17특징·분광 손실까지](docs/guide/research_overview.md)에서 단계별 연산, 텐서 크기, 결과 해석을 확인할 수 있습니다.

## 연구 문서

- [기존 PAN–MS 재현 정리](docs/reproduction.md)
- [이전 RGB–HSI 실험 정리](docs/previous_experiments.md)

## 가장 최근 완료 연구 · 17특징 23탭 + 분광 손실

204→17 grouped 학습 압축(12밴드/특징), R/G/B 독립 core, K=4, 2,601,086 parameters. 내부 평균 축소·23탭 확대, 입력 area ×4·23탭 baseline, HR256/LR64. MSE에는 정합 유효 마스크를 적용하고, 분광 항은 GT·예측 norm >1e−6 픽셀만 사용해 MSE + 0.01(1−cos)을 학습했습니다.

| test 75장면 평균 | MSE ↓ | PSNR dB ↑ | SAM ° ↓ |
|---|---:|---:|---:|
| 23탭 baseline | 0.0013944695 | 29.2259 | 2.4790 |
| 17특징 + 분광 손실 | 0.0004716392 | 34.0611 | 2.2006 |

best epoch 99, 100 epoch 완료. [상세 보고서](docs/experiments/rgb06_triple17_spectral.md). 직전 Bilinear 대비 PSNR **+0.2714 dB**, SAM **-0.0184°**. 복구 기준인 12특징·23탭 대비는 PSNR **+0.0081 dB**, SAM **-0.0051°**로 차이가 작습니다. 단일 seed이며 17특징과 손실을 함께 바꿨으므로 분광 손실만의 개선으로 해석하지 않습니다.

![학습·검증 그래프](docs/assets/rgb06_triple17_spectral/learning.png)

![test 비교 그래프](docs/assets/rgb06_triple17_spectral/test_metrics.png)

### test 예시 5종

[204밴드 웹 뷰어](https://biyonghiyon.github.io/CtrS/ssa-mrn/) · [오프라인 HTML](docs/assets/rgb06_triple17_spectral/band_viewer/index.html). Pages에는 최신 결과만 배포하고 이전 실험의 HTML은 저장소에 보존합니다.

왼쪽부터 **LR HSI · RGB 입력 · 예측 · 정답**입니다. 이전과 동일한 사전 고정 5장면 tile0입니다. HSI 표시는 공통 대비이며 정량 지표는 204밴드 원래 값으로 계산합니다. HTML은 같은 epoch99 가중치의 CPU float32 추론으로 CUDA AMP PNG와 미세한 차이가 있을 수 있습니다.

[예시 1 · 204밴드](https://biyonghiyon.github.io/CtrS/ssa-mrn/sample_01.html)

![test 예시 1](docs/assets/rgb06_triple17_spectral/sample_01.png)

[예시 2 · 204밴드](https://biyonghiyon.github.io/CtrS/ssa-mrn/sample_02.html)

![test 예시 2](docs/assets/rgb06_triple17_spectral/sample_02.png)

[예시 3 · 204밴드](https://biyonghiyon.github.io/CtrS/ssa-mrn/sample_03.html)

![test 예시 3](docs/assets/rgb06_triple17_spectral/sample_03.png)

[예시 4 · 204밴드](https://biyonghiyon.github.io/CtrS/ssa-mrn/sample_04.html)

![test 예시 4](docs/assets/rgb06_triple17_spectral/sample_04.png)

[예시 5 · 204밴드](https://biyonghiyon.github.io/CtrS/ssa-mrn/sample_05.html)

![test 예시 5](docs/assets/rgb06_triple17_spectral/sample_05.png)

## 현재 학습 상태

보고 대상 run `20261004-235831-565c8ba31ed7`은 2026-10-05 05:10(KST), 종료 코드 0으로 완료됐습니다. best/latest를 서버에 보존했으며 새 학습은 실행하지 않았습니다.

## 결과 보고와 파일 보관

완료 실험에는 공통 보고서, 학습/검증 및 비교 그래프, 사전에 선택한 서로 다른 test 5장면의 4패널 이미지를 반드시 남깁니다. 성능이 좋은 장면을 보고 골라서는 안 됩니다.

```powershell
& $py SSA-MRN\scripts\export_results.py --checkpoint '<run>\best.pt' --data-root 'C:\CtrS\SSA-MRN\data\dataset\RGB-HSI\LIB-HSI' --output-dir '<새 결과 폴더>' --previous-metrics '<직전 결과>\test\metrics.json'
```

실행 환경에 `requirements.txt`를 설치해야 합니다. exporter는 체크포인트에 저장된 평가 조건을 사용합니다. [공통 보고서 양식](docs/experiments/template.md), [작업 규칙](AGENTS.md), [데이터셋 감사 자료](references/rgb_hsi_datasets.md)를 참고합니다.

새 실험 시작 전 `cleanup_experiment.py --run-dir <완료 run>`의 목록을 검토하고 `--completed --apply`로 적용합니다. 가중치·숫자 결과는 보존하고 과거 설정·원시 로그·캐시는 정리합니다. 현재 학습의 run에는 적용하지 않습니다.
