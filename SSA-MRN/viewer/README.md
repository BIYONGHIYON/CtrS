# 팬샤프닝 결과 비교 사이트

같은 RR test 장면에서 **기준 모델 C2MS-SP(성능 81번)와 고른 비교 모델 하나**의 예측 영상·오차 지도·수치를 비교하는 정적 웹페이지입니다. 비교 모델은 화면에서 고르며 기본값은 원본 SSA-MRN입니다. 그 밖에 우리 모델(경량판 C2MS-SP-Lite, 성능 83번)과 우리 환경에서 평가한 다른 연구 모델(현재 CANNet·ARConv·FusionNet), 학습 없이 시험한 생성형 이미지 모델(GPT Image 2.6·Nano Banana 2.1과 그 대조군, RGB 3밴드만)을 고를 수 있습니다. CANNet·ARConv와 FusionNet의 QB·WV3·WV2는 저자 공개 가중치이고, 저자 가중치가 없는 FusionNet GF2는 우리 학습 데이터에서 직접 학습한 seed 42 가중치입니다. 데이터셋은 QB·GF2·WV3·WV2 전이입니다. 우리 모델은 seed 42 가중치이고, C2MS-SP·SP-Lite는 관측 일치 사영(공액 기울기법 10회)과 WV2의 성능 82번 밴드별 센서 보정을 포함합니다. 원본과 다른 연구 모델은 발표된 방식 그대로이며 우리 방법을 붙이지 않았습니다. 모델 정의는 `common/baseline_models.py`, 다른 연구 모델의 평가는 `benchmarks/<모델>/`에 있습니다.

배포 주소: **[CtrS Viewer 열기](https://biyonghiyon.github.io/CtrS/)** (이 저장소의 GitHub Pages 루트, 배포 워크플로 [`.github/workflows/pages.yml`](../../.github/workflows/pages.yml)). 같은 사이트의 `/portfolio/<GitHub ID>/`에는 팀원별 소개 페이지([`portfolio/`](../../portfolio/README.md))가 있고, 이 사이트에서 그쪽으로 가는 링크는 두지 않습니다.

## 실행

```bash
python SSA-MRN/viewer/export.py
python -m http.server 8765 -d SSA-MRN/viewer
```

브라우저에서 `http://localhost:8765`를 엽니다. `export.py`는 `SSA-MRN/data/`의 RR test H5와 저장소의 `best.pt`로 예측과 장면 썸네일을 만들어 `viewer/data/`(Git에서 제외, 약 370MB)에 씁니다. torch·h5py·numpy·scipy·scikit-image·pillow가 필요합니다. C2MS-SP·SP-Lite의 사영과 보정은 float64를 쓰므로 `--device cpu`로 내보냅니다(세 모델·네 데이터셋에 몇 분).

## 화면

- **데이터셋:** 맨 위 탭에서 고릅니다.
- **장면:** 탭 아래 썸네일 줄에서 고릅니다. 썸네일 아래 숫자는 **C2MS-SP − 비교 모델**의 장면 PSNR 차이(dB, 파랑이면 C2MS-SP가 높음)이고, 밑줄이 있는 번호는 표준 결과 이미지 장면(0·1·3·7·14·17)입니다. `[` `]` 키로 이전·다음 장면으로 넘어갑니다.
- **모델:** C2MS-SP는 항상 표시하고, 다른 한 열은 "비교 모델" 목록에서 고릅니다(원본 / 우리 모델 / 다른 연구 모델 / 생성형 이미지 모델 · RGB만, 기본값 원본 SSA-MRN). 고른 모델의 예측만 내려받습니다.
- **RGB 3밴드 비교:** 생성형 이미지 모델은 RGB 3밴드만 내므로, 이 모델을 고르면 C2MS-SP를 포함한 영상·오차 지도·확대 영역 수치·장면과 데이터셋 수치를 모두 데이터셋의 RGB 밴드(QB·GF2 2 1 0, WV3·WV2 4 2 1)로만 계산합니다. 화면 위에 "RGB 3밴드 비교"가 표시되고 NIR 위색과 다른 밴드의 오차 지도는 고를 수 없습니다. 이 수치는 전체 밴드 수치와 직접 비교하지 않습니다.
- **영상 열:** 두 모델의 예측 · 정답 · 모델별 오차 |예측 − 정답| · C2MS-SP 개선(비교 모델 오차 − C2MS-SP 오차, 파랑 개선·주황 악화)을 한 열씩 놓았습니다. 각 열의 위가 장면 전체, 바로 아래 노란 테두리가 확대 영역입니다.
- **확대 영역:** 아무 영상이나 누르거나 끌면 노란 상자가 옮겨집니다. 크기는 16·32·64·128 픽셀, 방향키로 1픽셀(Shift는 8픽셀)씩 움직입니다. "개선 최대"·"악화 최대"는 개선 지도의 창 평균이 가장 크거나 작은 곳으로 옮깁니다.
- **수치:** 영상 아래에 장면·영역 PSNR과 평균 오차를 적고, 아래 표에 확대 영역(PSNR·SAM·ERGAS·평균 오차), 장면 전체(PSNR·SAM·ERGAS·SCC·Q2n), 데이터셋 평균을 두 모델별로 놓았습니다. C2MS-SP 옆의 차이는 비교 모델 대비이며 파랑이 C2MS-SP 쪽이 좋은 것입니다.
- **표시 설정:** 오차 지도 밴드, 색 범위, 확대 영역 밝기, LR MS·PAN 입력 표시를 바꿉니다. 확대 영상에 마우스를 올리면 모든 열에 같은 픽셀이 표시되고 그 픽셀 값이 아래에 나옵니다.
- 주소창의 `#ds=...&scene=...&x=...&y=...&w=...&cmp=...`가 현재 데이터셋·장면·영역·비교 모델을 담고 있어 링크로 같은 화면을 공유할 수 있습니다. 이전의 `models=` 링크도 그 안의 첫 비교 모델(대개 원본)을 골라 엽니다.
- 글꼴(IBM Plex Sans KR·Mono)은 Google Fonts에서 받으며, 인터넷이 없으면 시스템 글꼴로 표시됩니다.

## 수치 확인

- 원본의 예측 PSNR은 성능 53·54번이 보관한 결과 이미지 기록(`result_images.json`, CUDA)과 네 데이터셋의 표준 6장면 모두에서 0.0007 dB 이내로 일치했습니다(로컬 MPS 추론). C2MS-SP·SP-Lite의 데이터셋 평균은 `common/baseline_images.py`의 seed 42 값과 대조했습니다.
- 예측은 1/8 DN 단위 int16으로 저장합니다. 브라우저가 저장값으로 계산한 장면 전체 수치는 원래 예측으로 계산한 값과 PSNR 0.0004 dB, SAM 0.0001° 이내였습니다.
- PSNR은 `rr_metrics`와 같이 peak 2047의 밴드별 PSNR 평균입니다(GF2는 10bit라 약 6 dB 높게 나옵니다).
- 우리 모델의 수치는 seed 42 가중치 한 개의 결과이고 실험 판정은 `docs/judgement_criteria.md`의 시드 평균을 따릅니다. 다른 연구 모델은 저자 가중치(시드 미공개) 또는 직접 학습 가중치(seed 42)의 예측이며 판정을 붙이지 않습니다. 그 데이터셋 평균은 Python `rr_metrics` 값이라 보고서의 MATLAB 채점값과 소수 셋째 자리에서 다를 수 있습니다.

## 모델 추가

뷰어 데이터 갱신(모델 추가·설명 수정 포함)은 BIYONGHIYON이 합니다. 다른 연구 모델의 평가자는 예측 Release와 평가 PR까지 하고 이 절차는 하지 않습니다. `export.py`는 로컬 `data/index.json`에 있는 다른 모델의 결과에 이어 붙이므로, 현재 배포 데이터 전체가 없는 컴퓨터에서 만든 데이터를 올리면 다른 모델이 사이트에서 빠집니다.

`registry.json`의 `models`에 항목을 추가한 뒤 그 모델만 내보냅니다. 기존 모델의 결과는 그대로 유지됩니다. `group`은 `original`(원본)·`baseline`(기준 모델)·`ours`(우리 다른 모델)·`benchmark`(다른 연구 모델)·`generative`(생성형 이미지 모델)이며, `baseline`을 뺀 나머지가 비교 모델 목록에 나옵니다.

```bash
python SSA-MRN/viewer/export.py --models <모델 id>
```

- **저장소 코드로 만드는 모델:** `module`(SSA-MRN 기준 경로), `factory`, `args`(`"{bands}"`는 밴드 수, `"{dataset}"`은 데이터셋 id로 바뀜), 데이터셋별 인자 `kwargs_by_dataset`과 데이터셋별 `checkpoints`를 적습니다. 입력은 (pan, lms, ms)를 데이터셋 `scale`로 나눈 값이고 출력에 다시 곱합니다. 체크포인트의 state dict 키가 `model`이 아니면 `state_key`를 적습니다.
- **다른 연구 모델(평가 완료):** `"benchmark": {"model": "<폴더 이름>", "setting": "author|retrain", "datasets": ["QB", "GF2", "WV3", "WV2"]}`을 적습니다. `export.py`가 MATLAB 채점에 쓴 예측 Release `predictions-benchmarks-<모델>-<설정>`의 zip을 받아 GitHub SHA-256을 확인하고 읽습니다(`common/benchmark_predictions.py`). 가중치가 필요 없으므로 재배포할 수 없는 저자 가중치 모델도 올릴 수 있습니다. 데이터셋마다 설정이 다르면 `"settings_by_dataset": {"GF2": "retrain"}`처럼 바꿀 데이터셋만 적고, 화면의 모델 설명 표시도 `role_by_dataset`으로 데이터셋별로 바꿉니다(FusionNet). `group`은 `benchmark`로 둡니다. 예측이 RGB 3밴드뿐이면(생성형 이미지 모델) `"bands": "rgb"`를 더하고 `group`을 `generative`로 둡니다. 이 예측은 3밴드로 저장·채점하고, 모든 모델의 결과에 RGB 밴드만의 수치(`rgb`)를 함께 적으므로 화면이 같은 밴드로 비교합니다. RGB 수치가 없는 이전 결과의 기준 모델은 `--models c2ms_sp`로 다시 내보냅니다.
- **H5로 준비한 예측:** `"predictions": {"QB": "<h5 경로>"}`를 적습니다. H5에는 `pred` 데이터셋을 RR test와 같은 순서·크기 `(20, C, 256, 256)`의 DN 값으로 넣습니다.
- 체크포인트나 예측이 없는 데이터셋은 그 모델에서 건너뜁니다. `label`·`short`·`role`·`description`은 화면에 그대로 표시됩니다.

## 배포 갱신

예측 데이터(약 370MB, 묶음 약 300MB)는 Git에 넣지 않고 Release [`viewer-data`](https://github.com/BIYONGHIYON/CtrS/releases/tag/viewer-data)에 묶음 하나로 올립니다. `release_data.py`가 `data/`를 같은 내용이면 같은 해시가 나오도록 묶어 `viewer-data-<해시 12자>.tar.gz`로 올리고, GitHub가 보고한 SHA-256으로 업로드를 확인한 뒤 그 이름과 해시를 `data.sha256`에 씁니다. 이전 버전 자산은 지우지 않습니다.

```bash
python SSA-MRN/viewer/export.py --models <모델 id>
python SSA-MRN/viewer/release_data.py
```

그다음 새 브랜치에서 `data.sha256`(코드를 바꿨다면 `index.html`·`app.js`도)을 커밋하고 PR을 올립니다. main에 병합되면 Pages 워크플로가 `data.sha256`에 적힌 자산을 받아 해시를 확인하고 `index.html`·`app.js`와 함께 배포합니다(1-2분). 해시가 맞지 않으면 배포하지 않습니다. 사이트가 돌아가는 데 로컬 데이터는 필요하지 않습니다.
