# 다른 팬샤프닝 모델 평가

다른 연구의 팬샤프닝 모델을 우리 환경에서 직접 학습·평가해 기준 모델 C2MS-SP와 같은 조건으로 비교합니다. 같은 데이터 분할, 같은 test 장면, 같은 정규화, 같은 지표 코드를 씁니다. 그래서 논문 보고값과 달리 학습·평가 조건 차이 없이 직접 비교할 수 있습니다. 우리 모델을 바꾸는 개선 실험은 [성능 개선 연구](../research/performance/README.md)(`experiments/performance/`)에 둡니다. 이 폴더에는 다른 모델의 재현·평가만 둡니다.

[관리 규칙](../../AGENTS.md#다른-팬샤프닝-모델-평가) · [보고서 양식](../docs/benchmark_template.md) · [모델 목록](catalog.json) · [논문 보고값 조사](../docs/pansharpening_survey/QB_GF_WV2_WV3_팬샤프닝_성능조사_2026-10-27.md)

<a id="models"></a>

## 평가 표

최근에 등록한 모델부터 적습니다. 첫 열은 모델 보고서 링크와 실제 평가자입니다. 판정은 하지 않고, 확인된 결과에 같은 시드 C2MS-SP와의 RR 차이를 숫자로 적습니다.

| 모델 | 출처·학습 조건 | 확인된 결과 | 상태 |
|---|---|---|:---:|
| 아직 등록된 모델이 없습니다. | | | |

## 폴더 구조

```text
benchmarks/
├── README.md                  이 안내와 평가 표
├── catalog.json               모델별 출처·라이선스·평가자·상태
└── <모델>/                    소문자 snake_case 모델 이름 (예: adwm, cannet, lformer)
    ├── README.md              8절 보고서 (docs/benchmark_template.md)
    ├── config.json            출처·고정 커밋·라이선스·설정별 학습 조건
    ├── model.py               어댑터: build(bands, sensor, setting, seed)
    ├── train.py               우리 데이터로 학습할 때만
    ├── upstream/              공식 코드 Git 하위 모듈 (고정 커밋, 수정 금지)
    ├── checkpoints/
    │   └── <설정>_<센서>_s<시드>/best.pt
    ├── results/
    │   ├── <설정>_<센서>_s<시드>/metrics.json   장면별 RR·FR, 가중치·데이터 해시
    │   ├── <설정>_<센서>_s<시드>/history.json   우리 학습의 에폭별 기록
    │   ├── summary.json       시드 평균과 같은 시드 C2MS-SP와의 차이
    │   ├── learning.png       학습·validation 곡선
    │   └── images/<설정>_<센서>_s<시드>/          고정 6장면 두 형식
    ├── provenance/            환경, 공식 코드 수정분(upstream.diff), 원본 학습 설정
    └── manifest.json          보관 파일 경로·크기·SHA-256
```

재개용 `latest.pt`와 초기 `initial.pt`는 Git에 넣지 않습니다. 모델별 GitHub Release(`weights-benchmarks-<모델>`)에 올립니다. 예측 배열·데이터·로그도 추적하지 않습니다.

## 평가 설정

| 설정 | 내용 | 시드 |
|---|---|---|
| `author` | 저자가 배포한 가중치를 학습 없이 평가합니다. 저자 학습 데이터가 우리 분할과 다를 수 있으므로 보고서에 적습니다. | 없음 |
| `retrain` | 공식 코드와 논문 설정(손실·배치·학습률 등)으로 우리 학습 데이터에서 학습합니다. 에폭은 논문이나 공식 코드에 공개된 값을 씁니다. 공개되지 않았으면 C2MS-SP와 같은 150에폭으로 학습합니다. 축척 혼합·사영·밴드 보정은 붙이지 않습니다. | 42·43·44 |

모든 설정은 PanCollection의 같은 학습·validation·test 분할을 씁니다. 정규화는 QB·WV3·WV2 2047, GF2 1023입니다. test는 RR·FR 각 20장 전체를 평가합니다. 가중치는 validation으로만 고르고 test를 보고 에폭·설정을 바꾸지 않습니다. WV2는 WV3 가중치를 적용한 전이 결과로 참고만 합니다.

## 평가 실행

어댑터 `model.py`는 `build(bands, sensor, setting, seed)`로 네트워크를 돌려줍니다. 네트워크는 `forward(pan, lms, ms)`를 받아 같은 크기의 출력을 냅니다. 입력과 출력은 DN을 정규화 값으로 나눈 값입니다. 원래 코드의 입력 순서·패딩·정규화 차이는 어댑터 안에서 맞춥니다. 체크포인트 형식이 다르면 `load(net, path)`도 정의합니다.

```bash
# 시드별 RR·FR 평가 → results/<설정>_<센서>_s<시드>/metrics.json
python SSA-MRN/common/benchmark_evaluate.py --model adwm --setting retrain --sensor QB --seed 42 --device cuda

# 처음 두 장면만으로 어댑터 확인 (metrics_smoke.json, 요약에서 제외)
python SSA-MRN/common/benchmark_evaluate.py --model adwm --setting retrain --sensor QB --seed 42 --limit 2 --skip-fr

# 시드 평균과 같은 시드 C2MS-SP와의 차이 → results/summary.json
python SSA-MRN/common/benchmark_evaluate.py --model adwm --summary

# 고정 6장면 두 형식 (C2MS-SP · 해당 모델) → results/images/
python SSA-MRN/common/benchmark_evaluate.py --model adwm --setting retrain --sensor QB --seed 42 --figures --device cuda

# 보관 해시 확인, Release 자산까지 확인할 때는 --releases
python SSA-MRN/common/verify.py --benchmark adwm
```

평가 도구는 기준 수치를 만든 `common/metrics`를 그대로 씁니다. 원본 SSA-MRN 가중치로 확인했을 때 QB seed 42 RR PSNR·SAM·ERGAS·SCC·Q2n이 [기준 비교 수치](../docs/images/c2ms_sp/baseline_metrics.json)와 같았습니다.

## 결과 보고

다른 모델 평가에는 판정(개선 여부, C2MS-SP 대비 우열)을 붙이지 않습니다. RR test PSNR·SAM·ERGAS·SCC·Q2n의 시드 평균과 시드별 값, 같은 시드 C2MS-SP와의 차이를 숫자로 보고합니다. 학습 에폭과 그 출처(논문, 공식 코드, 미공개라 150에폭)를 함께 적습니다. FR QNR·Dλ·Ds와 WV2 전이는 참고로 따로 적습니다. 3시드 평가가 끝난 결과만 루트 README의 비교 표에 `모델 (Ours 환경 재현)` 행으로 넣고, 논문 보고값 행과 구분합니다.
