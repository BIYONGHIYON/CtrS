# SSA-MRN · RGB 유도 HSI 초해상도

고해상도 RGB와 저해상도 HSI에서 ×4 HSI 복원을 연구합니다. 아래 결과는 LIB-HSI의 관측 HSI를 합성 축소한 평가이며 실제 센서의 HR-HSI 정답 성능과 구분합니다.

## 연구 개요

204밴드 HSI의 보간 결과를 유지하면서, 인접한 17밴드씩 학습 압축한 12특징에서 공간 보정량을 추정합니다. R·G·B 각각의 독립 SSA-MRN 경로와 밴드별 학습 fusion으로 보정량을 결합합니다. 현재는 같은 구조에서 내부 23탭/평균 resampling과 Bilinear의 영향을 비교합니다.

[연구 설명 · 재현부터 RGB별 12특징까지](docs/guide/research_overview.md)에서 단계별 연산, 텐서 크기, 결과 해석과 교수님 보고용 요약을 확인할 수 있습니다.

## 연구 문서

- [기존 PAN–MS 재현 정리](docs/reproduction.md)
- [이전 RGB–HSI 실험 정리](docs/previous_experiments.md)

## 가장 최근 완료 연구 · RGB별 12특징 23탭 타일

204밴드를 17개씩 grouped 압축해 12특징으로 만들고, R/G/B 각각 독립 core·decoder의 출력을 밴드별로 결합합니다. K=4, 내부 평균 축소/23탭 확대, 입력 baseline 23탭. 학습·test 모두 256 타일, LR HSI 64×64입니다.

| test 75장면 평균 | MSE ↓ | PSNR dB ↑ | SAM ° ↓ |
|---|---:|---:|---:|
| 23탭 baseline | 0.0013944695 | 29.2259 | 2.4790 |
| RGB별 SSA-MRN | 0.0004733597 | 34.0530 | 2.2057 |

best epoch 99. [상세 보고서·직전 연구와 차이](docs/experiments/rgb04_triple12.md). 직전 연구와는 test full256→tiles 변경이 있어 수치 차이를 통제된 개선으로 해석하지 않습니다.

![학습 및 검증 그래프](docs/assets/rgb04_triple12/learning.png)

![test baseline 및 직전 연구 수치 비교](docs/assets/rgb04_triple12/test_metrics.png)

### test 예시 5종

[204밴드 슬라이더 뷰어](docs/assets/rgb04_triple12/band_viewer/index.html) · HTML 폴더를 내려받아 `index.html`을 브라우저에서 여세요. 각 장면 HTML도 단독 실행됩니다. 같은 epoch 99 가중치의 CPU 추론이며, 기존 CUDA AMP 예시와 미세한 수치 차이가 있을 수 있습니다.

왼쪽부터 **LR HSI · RGB 입력 · 예측 · 정답**입니다. 동일한 HSI 대비 범위를 사용하며, 지표는 204밴드 원래 값으로 계산합니다. 이전 보고서와 같은 5장면의 tile 0입니다.

![test 예시 1](docs/assets/rgb04_triple12/sample_01.png)

![test 예시 2](docs/assets/rgb04_triple12/sample_02.png)

![test 예시 3](docs/assets/rgb04_triple12/sample_03.png)

![test 예시 4](docs/assets/rgb04_triple12/sample_04.png)

![test 예시 5](docs/assets/rgb04_triple12/sample_05.png)

## 현재 학습 · RGB별 12특징 Bilinear 타일

직전 모델의 내부 guide·분광 특징 확대/축소를 재현 모델과 같은 **Bilinear, align_corners=True**로 바꿉니다. HSI 입력의 area 축소와 23탭 LMS baseline은 유지합니다. 구조·분할·타일 평가 조건을 고정해 내부 보간의 영향을 비교합니다.

- 모델 `rgb_triple_grouped12_bilinear`, K=4, 1,950,186 parameters.
- LIB-HSI train/validation/test 393/45/75 장면. 패치 수는 독립 장면 수가 아닙니다.
- 설정: `configs/lib_rgb_hsi_triple12_k4_bilinear_tiles.json` (현재 학습 설정만 보관).
- 서버 실행 `20261003-220209-d4faa37d0365`, 코드 `d947f84`에서 새 학습 시작. 원격 controller로 SSH 종료 후에도 실행.
- 최종 test 결과는 아직 없습니다. [진행 보고서와 완료 기준](docs/experiments/rgb05_triple12_bilinear.md).

### 원격 클라이언트에서 상태·로그 확인

VS Code Remote SSH의 PowerShell에서 실행합니다.

```powershell
$py = 'C:\CtrS\.venv\Scripts\python.exe'
$ctl = 'C:\CtrS-triple12-bilinear\scripts\remote-training.py'
& $py $ctl status
& $py $ctl logs --follow
```

로그 보기의 Ctrl+C는 학습을 중단하지 않습니다. 현재 학습 중에는 실행 코드·run config·로그 폴더를 정리하거나 갱신하지 않습니다.

## 결과 보고와 파일 보관

완료 실험에는 공통 보고서, 학습/검증 및 비교 그래프, 사전에 선택한 서로 다른 test 5장면의 4패널 이미지를 반드시 남깁니다. 성능이 좋은 장면을 보고 골라서는 안 됩니다.

```powershell
& $py SSA-MRN\scripts\export_results.py --checkpoint '<run>\best.pt' --data-root 'C:\CtrS\SSA-MRN\data\dataset\RGB-HSI\LIB-HSI' --output-dir '<새 결과 폴더>' --previous-metrics '<직전 결과>\test\metrics.json'
```

실행 환경에 `requirements.txt`를 설치해야 합니다. exporter는 체크포인트에 저장된 평가 조건을 사용합니다. [공통 보고서 양식](docs/experiments/template.md), [작업 규칙](AGENTS.md), [데이터셋 감사 자료](references/rgb_hsi_datasets.md)를 참고합니다.

새 실험 시작 전 `cleanup_experiment.py --run-dir <완료 run>`의 목록을 검토하고 `--completed --apply`로 적용합니다. 가중치·숫자 결과는 보존하고 과거 설정·원시 로그·캐시는 정리합니다. 현재 학습의 run에는 적용하지 않습니다.
