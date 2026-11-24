# 다른 팬샤프닝 모델 평가

다른 연구의 팬샤프닝 모델을 우리 환경(같은 데이터 분할·test 장면·정규화·지표 코드)에서 직접 학습·평가해 기준 모델 SMOC와 비교합니다. 우리 모델을 바꾸는 실험은 [성능 개선](../experiments/performance/README.md)에 둡니다.

[관리 규칙](../../AGENTS.md#8-다른-팬샤프닝-모델-평가) · [보고서 양식](../docs/benchmark_template.md) · [모델 목록](catalog.json) · [논문 보고값 조사](../docs/pansharpening_survey/QB_GF_WV2_WV3_팬샤프닝_성능조사_2026-10-27.md)

<a id="models"></a>

## 평가 표

최근에 등록한 모델이 위입니다. 판정은 붙이지 않고 같은 시드 SMOC와의 RR 차이를 숫자로 적습니다. 저자 시드가 없으면 SMOC 3시드 평균과 비교합니다. 기존 평가의 절대 수치는 그대로 두고, 아래 차이만 SMOC로 다시 집계했습니다. 개별 보고서·수치·이미지의 C2MS-SP 비교는 당시 기록으로 보존합니다.

| 모델 | 출처·학습 조건 | 확인된 결과 | 상태 |
|---|---|---|:---:|
| [ADWM](adwm/README.md)<br>[김경찬·이병현] | CVPR 2025 · 저자 WV3, WV2 전이 · QB 200·GF2 500에폭 직접 학습 seed 42 · 실제 MATLAB 채점 | QB 38.523 (SMOC seed 42 +0.156), GF2 49.580 (+0.161), WV3 37.858 dB; WV2 전이 29.601 dB. 네 센서 고정 6장면 이미지 | 평가 완료 |
| [GPT Image 2.6 Sunburst](gpt_image_2_6/README.md)<br>[센서별 담당](#generative) | 생성형 이미지 모델 · `zeroshot` 학습 없음·단일 실행 · RGB 3밴드 실제 MATLAB 채점 | RGB RR PSNR QB 31.170, GF2 36.893, WV3 31.555, WV2 31.030 dB. 네 센서 각 20장·정답 없는 지표·고정 6장면 이미지 | 평가 완료 |
| [Nano Banana 2.1](nano_banana_2_1/README.md)<br>[센서별 담당](#generative) | 생성형 이미지 모델 · `zeroshot` 학습 없음·단일 실행 · RGB 3밴드 실제 MATLAB 채점 | RGB RR PSNR QB 29.626, GF2 33.114, WV3 29.223, WV2 30.194 dB. 네 센서 각 20장·정답 없는 지표·고정 6장면 이미지 | 평가 완료 |
| [FusionNet](fusionnet/README.md)<br>[김윤지] | 저자 QB·WV3, WV2 전이 · GF2 직접 학습 seed 42 · 실제 MATLAB 채점 | QB 31.485, GF2 45.547, WV3 32.519 dB; WV2 전이 27.058 dB. 네 센서 고정 6장면 이미지. GF2 peak 1023 기준 39.522 dB | 평가 완료 |
| [ARConv](arconv/README.md)<br>[김윤지] | CVPR 2025 · `author` 저자 가중치 · 실제 MATLAB 채점 | QB 38.425, GF2 49.385, WV3 39.167 dB. SMOC 3시드 평균 대비 각각 +0.081, -0.067, +0.031 dB; WV2 전이(참고) 29.853 dB(-2.158). 네 데이터셋 고정 6장면 이미지 | 평가 완료 |
| [U2Net](u2net/README.md)<br>[송준현] | ACM MM 2023 · 저자 WV3·WV2 전이, QB·GF2 360에폭 seed42 · micro4×누적4(논문 학습법 차이) · 실제 MATLAB | QB 36.500 (SMOC seed42 −1.868), GF2 48.906 (−0.513), WV3 38.844 (SMOC 3시드 평균 −0.292), WV2 30.497 dB (−1.515). 네 센서 고정6장면·가중치 | 평가 완료 |
| [CANNet](cannet/README.md)<br>[이병현] | CVPR 2024 · `author` 저자 가중치 재현(재학습 없음) · Kaggle T4 · 실제 MATLAB 채점 | QB RR PSNR 38.261(SMOC 평균 대비 -0.083 dB), GF2 49.274(-0.178), WV3 38.977(-0.159). WV3는 논문값 재현, QB·GF2는 논문보다 조금 낮음 | 평가 완료 |

<a id="generative"></a>

## 생성형 이미지 모델 시험

기업 멘토링의 "LLM·멀티모달 모델과 비교하고, 할루시네이션도 실제로 측정하라"는 피드백에 따라, 이미지를 출력하는 생성형 모델 두 개(GPT Image 2.6 Sunburst 품질 high, Nano Banana 2.1 1K)로 학습 없이 팬샤프닝을 시험했습니다. FLUX 2 [pro]는 크레딧 한도 때문에 하지 않습니다. 센서 하나를 한 사람이 맡아 두 모델로 RR test 20장씩 만들고 MATLAB 채점과 결과 확인을 마쳤습니다([#209](https://github.com/BIYONGHIYON/CtrS/issues/209) QB · [#210](https://github.com/BIYONGHIYON/CtrS/issues/210) GF2 · [#211](https://github.com/BIYONGHIYON/CtrS/issues/211) WV3 · [#212](https://github.com/BIYONGHIYON/CtrS/issues/212) WV2).

**완료 범위:** 두 모델 × 네 센서 × 20장, 총 160장입니다. 예측·원본 이미지는 모델별 Release에 있고 RGB 채점·검사·고정 6장면 이미지를 보관했습니다. 수치는 위 평가 표와 모델별 보고서에서 확인합니다. PSNR은 밴드 평균·peak 2047입니다.

**실제 평가자:** QB 이병현 · GF2 김윤지 · WV3 김경찬 · WV2 송준현입니다. QB·GF2는 API, WV3는 웹 화면으로 생성했고, WV2는 고정 설정·프롬프트 사용을 확인받은 제공 원본을 평가했습니다. 생성 경로와 확인 범위는 센서별 `generation.json`에 있습니다.

- **입력:** 모두 Release [`genimage-inputs`](https://github.com/BIYONGHIYON/CtrS/releases/tag/genimage-inputs)의 같은 입력을 씁니다. 1번 이미지는 데이터셋 LMS의 RGB 밴드, 2번 이미지는 PAN이며 둘 다 1024×1024 PNG입니다. 정답은 입력을 만들 때도 밝기 범위를 정할 때도 쓰지 않습니다.
- **생성:** QB·GF2의 API 생성에는 `common/genimage_io.py generate`를 썼고, 웹 화면으로 만든 센서의 실제 경로도 생성 기록에 적었습니다. 프롬프트와 설정은 고정했습니다. API를 실행할 때는 키를 `BAZE_API_KEY` 환경변수로 넘기고 `--estimate-only`로 비용을 확인한 뒤 `--max-credits` 한도 안에서 생성하며 시도 기록을 남깁니다.
- **되돌리기와 채점:** 받은 이미지는 `common/genimage_io.py restore`로 256×256 DN `.mat`과 zip으로 바꿉니다. 이 모델들은 RGB 3장만 출력하므로 `matlab-benchmark-author.yml`의 `bands: rgb`로 정답의 RGB 밴드에서만 채점합니다. 4·8밴드 전체 수치와 섞지 않으며 루트 README 비교 표에는 넣지 않습니다. [결과 비교 사이트](https://biyonghiyon.github.io/CtrS/)에서는 SMOC와 같은 RGB 3밴드로 계산해 비교합니다.
- **[변환 한계 대조군](genimage_identity/README.md):** 1번 이미지를 모델 없이 그대로 되돌린 결과입니다. RGB 3밴드 실제 MATLAB PSNR은 QB 29.216, GF2 37.953, WV3 29.225, WV2 27.695 dB이고, 생성형 모델은 이 값을 넘어야 PAN의 세부를 넣었다고 볼 수 있습니다.
- **정답 없는 지표:** `common/genimage_io.py checks`가 관측 일치 오차, PAN 구조 일치도, 위치 어긋남을 장면별로 계산하고 정답의 같은 값을 함께 적습니다.

## 폴더 구조

```text
benchmarks/
├── README.md                  이 안내와 평가 표
├── catalog.json               모델별 출처·라이선스·평가자·상태
└── <모델>/                    소문자 snake_case 모델 이름 (예: adwm, cannet, lformer)
    ├── README.md              보고서 (docs/benchmark_template.md)
    ├── config.json            출처·고정 커밋·라이선스·설정별 학습 조건
    ├── model.py               어댑터: build(bands, sensor, setting, seed)
    ├── train.py               우리 데이터로 학습할 때만
    ├── upstream/              공식 코드 Git 하위 모듈 (고정 커밋, 수정 금지)
    ├── checkpoints/
    │   └── <설정>_<센서>_s<시드>/best.pt
    ├── results/
    │   ├── <설정>_<센서>_s<시드>/metrics.json   장면별 RR·FR, 가중치·데이터 해시
    │   ├── <설정>_<센서>_s<시드>/history.json   우리 학습의 에폭별 기록
    │   ├── summary_smoc.json  시드 평균과 같은 시드 SMOC와의 차이
    │   ├── learning.png       학습·validation 곡선
    │   └── images_smoc/<설정>_<센서>_s<시드>/          고정 6장면 두 형식
    ├── provenance/            환경, 공식 코드 수정분(upstream.diff), 원본 학습 설정
    └── manifest.json          보관 파일 경로·크기·SHA-256
```

재개용 `latest.pt`와 초기 `initial.pt`는 Git에 넣지 않습니다. 모델별 GitHub Release(`weights-benchmarks-<모델>`)에 올립니다. 예측 배열·데이터·로그도 추적하지 않습니다.

## 평가 설정

| 설정 | 내용 | 시드 |
|---|---|---|
| `author` | 저자가 배포한 가중치를 학습 없이 평가합니다. 저자 가중치가 있는 센서는 재학습하지 않습니다. 저자 학습 데이터가 우리 분할과 다를 수 있으므로 보고서에 적습니다. | 없음 |
| `retrain` | 학습 센서 QB·GF2·WV3 중 저자 가중치·평가가 없는 데이터셋을 공식 코드와 논문 설정(손실·배치·학습률 등)으로 우리 학습 데이터에서 직접 학습하고, 가중치·예측·결과를 함께 올립니다. 논문에 보고된 설정을 그대로 쓰고, 논문에 없는 값은 공식 코드 기본값을 씁니다. 에폭·배치·학습률 등을 줄이거나 바꾸지 않습니다. 둘 다에 없는 값만 SMOC와 같게 정합니다(에폭 150). 축척 혼합·사영·밴드 보정은 붙이지 않습니다. | 42 |

모든 설정은 PanCollection의 같은 학습·validation·test 분할을 씁니다. 정규화는 QB·WV3·WV2 2047, GF2 1023입니다. test는 RR·FR 각 20장 전체를 평가합니다. 가중치는 validation으로만 고르고 test를 보고 에폭·설정을 바꾸지 않습니다. GPU 메모리나 시간이 부족해 논문 설정대로 학습할 수 없으면 에폭·배치를 줄이거나 모델을 가볍게 고치지 않고, 더 가벼운 다른 모델을 평가합니다(사유는 이슈에 남깁니다). Kaggle 세션 한도는 중간 저장과 이어서 학습하기로 넘깁니다. 배치·손실·기울기를 바꾸지 않고 시간만 더 드는 메모리 절약(층 재계산, `torch.utils.checkpoint`)은 쓸 수 있으며, 썼는지 `history.json`과 보고서에 적습니다. WV2는 학습하지 않고 WV3 가중치(저자 또는 직접 학습)를 적용한 전이 결과로 참고만 합니다. 저자 WV3 가중치가 없으면 WV3를 직접 학습한 뒤 WV2에 적용합니다.

## 평가 실행

어댑터 `model.py`는 `build(bands, sensor, setting, seed)`로 네트워크를 돌려줍니다. 네트워크는 `forward(pan, lms, ms)`를 받아 같은 크기의 출력을 냅니다. 입력과 출력은 DN을 정규화 값으로 나눈 값입니다. 원래 코드의 입력 순서·패딩·정규화 차이는 어댑터 안에서 맞춥니다. 체크포인트 형식이 다르면 `load(net, path)`도 정의합니다.

```bash
# 시드별 RR·FR 평가 → results/<설정>_<센서>_s<시드>/metrics.json
python SSA-MRN/common/benchmark_evaluate.py --model adwm --setting retrain --sensor QB --seed 42 --device cuda

# 처음 두 장면만으로 어댑터 확인 (metrics_smoke.json, 요약에서 제외)
python SSA-MRN/common/benchmark_evaluate.py --model adwm --setting retrain --sensor QB --seed 42 --limit 2 --skip-fr

# 시드 평균과 같은 시드 SMOC와의 차이 → results/summary_smoc.json
python SSA-MRN/common/benchmark_evaluate.py --model adwm --summary

# 고정 6장면 두 형식 (SMOC · 해당 모델) → results/images_smoc/
python SSA-MRN/common/benchmark_evaluate.py --model adwm --setting retrain --sensor QB --seed 42 --figures --device cuda

# 보관 해시 확인, Release 자산까지 확인할 때는 --releases
python SSA-MRN/common/verify.py --benchmark adwm
```

보고서·표의 RR 수치는 실제 MATLAB으로 채점합니다. 위 평가기로 만든 장면별 예측(DN, `H × W × C` `.mat` 20개)을 `<모델>_<설정>_<센서>_rr.zip`으로 묶어 Release `predictions-benchmarks-<모델>-<설정>`에 올립니다. 이어서 GitHub Actions의 `MATLAB score benchmark author-weight predictions` 워크플로(`.github/workflows/matlab-benchmark-author.yml`)를 model·setting·release·sensors를 입력해 실행합니다. 워크플로는 test 원본의 SHA-256을 확인합니다. 그다음 DLPan-Toolbox(`a34f884`) `score_suite.m`을 MATLAB R2025b로 실행해 RR 지표를 내고, 같은 배열의 Python 값과 비교합니다. 결과는 `results/matlab.json`으로 보관합니다.

평가 완료로 표시하려면 네 센서(QB·GF2·WV3·WV2 전이)의 예측 zip이 모두 Release에 있어야 합니다. zip 안의 파일 이름은 `output_mulExm_<0-19>.mat`, 변수는 `sr`입니다. 이 Release는 MATLAB 채점과 결과 비교 사이트 비교군이 함께 쓰는 원본이므로 지우거나 같은 이름으로 다른 내용을 덮어쓰지 않습니다. 저자 가중치를 재배포할 수 없는 모델도 우리가 추론한 예측은 올립니다.

평가자는 예측 Release와 평가 PR까지 합니다. 평가 PR이 병합되면 BIYONGHIYON이 같은 Release로 결과 비교 사이트의 비교군에 올립니다(`viewer/registry.json` 등록, `export.py --models <id>`, `release_data.py`, `data.sha256` PR, [안내](../viewer/README.md#모델-추가)). 평가자는 뷰어 데이터를 직접 만들거나 올리지 않습니다. `export.py`는 로컬에 있는 기존 뷰어 데이터에 새 모델을 이어 붙이므로, 기존 데이터가 없는 컴퓨터에서 만든 데이터를 올리면 다른 모델이 사이트에서 빠집니다. 고정 6장면 이미지도 가중치 없이 같은 Release에서 만들 수 있습니다(`benchmark_evaluate.py --figures --release`).

```bash
gh workflow run matlab-benchmark-author.yml -f model=adwm -f setting=retrain -f release=predictions-benchmarks-adwm-retrain -f sensors='["QB","GF2","WV3","WV2"]'
```

Python 평가기는 기준 수치를 만든 `common/metrics`를 그대로 씁니다. 원본 SSA-MRN 가중치로 확인했을 때 QB seed 42 RR PSNR·SAM·ERGAS·SCC·Q2n이 [기준 비교 수치](../docs/images/smoc/baseline_metrics.json)와 같았습니다.

기본 비교 모델은 SMOC입니다. `--summary`는 `results/summary_smoc.json`, `--figures`는 `results/images_smoc/`에 저장하므로 기존 C2MS-SP 기록을 덮어쓰지 않습니다. 과거 결과를 재현할 때만 `--baseline c2ms_sp`를 명시합니다.

## 결과 보고

다른 모델 평가에는 판정(개선 여부, SMOC 대비 우열)을 붙이지 않습니다. 실제 MATLAB으로 채점한 RR test PSNR·SAM·ERGAS·SCC·Q2n과 같은 시드 SMOC와의 차이를 숫자로 보고합니다. 같은 배열에서 Python 공통 평가기와의 차이도 적습니다. 학습 에폭과 그 출처(논문, 공식 코드, 미공개라 150에폭)를 함께 적습니다. FR QNR·Dλ·Ds와 WV2 전이는 참고로 따로 적습니다. MATLAB 채점까지 끝난 결과만 루트 README의 비교 표에 `모델 (Ours 환경 재현)` 행으로 넣고, 논문 보고값 행과 구분합니다.
