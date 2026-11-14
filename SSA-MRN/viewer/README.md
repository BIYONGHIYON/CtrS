# 팬샤프닝 결과 비교 사이트

같은 RR test 장면에서 여러 모델의 예측 영상·오차 지도·수치를 비교하는 정적 웹페이지입니다. 현재 원본 SSA-MRN, 기준 모델 C2MS-SP(성능 81번), 경량판 C2MS-SP-Lite(성능 83번)를 seed 42로 QB·GF2·WV3·WV2 전이 데이터셋에 올렸습니다(2026-11-14). C2MS-SP·SP-Lite는 관측 일치 사영(공액 기울기법 10회)을 포함하고, WV2에서는 성능 82번 밴드별 센서 보정도 포함합니다. 원본은 발표된 방식 그대로입니다. 모델 정의는 `common/baseline_models.py`에 있습니다.

배포 주소: **https://biyonghiyon.github.io/CtrS-viewer/** (배포 저장소 [BIYONGHIYON/CtrS-viewer](https://github.com/BIYONGHIYON/CtrS-viewer), GitHub Pages)

## 실행

```bash
python SSA-MRN/viewer/export.py
python -m http.server 8765 -d SSA-MRN/viewer
```

브라우저에서 `http://localhost:8765`를 엽니다. `export.py`는 `SSA-MRN/data/`의 RR test H5와 저장소의 `best.pt`로 예측과 장면 썸네일을 만들어 `viewer/data/`(Git에서 제외, 약 190MB)에 씁니다. torch·h5py·numpy·scipy·scikit-image·pillow가 필요합니다. C2MS-SP·SP-Lite의 사영과 보정은 float64를 쓰므로 `--device cpu`로 내보냅니다(세 모델·네 데이터셋에 몇 분).

## 화면

- **데이터셋:** 맨 위 탭에서 고릅니다.
- **장면:** 탭 아래 썸네일 줄에서 고릅니다. 썸네일 아래 숫자는 비교 대상과 비교 기준의 장면 PSNR 차이이고, 밑줄이 있는 번호는 표준 결과 이미지 장면(0·1·3·7·14·17)입니다. `[` `]` 키로 이전·다음 장면으로 넘어갑니다.
- **영상 열:** 모델별 예측 · 정답 · 모델별 오차 |예측 − 정답| · 개선(기준 오차 − 해당 모델 오차, 파랑 개선·주황 악화)을 한 열씩 놓았습니다. 각 열의 위가 장면 전체, 바로 아래 노란 테두리가 확대 영역입니다.
- **확대 영역:** 아무 영상이나 누르거나 끌면 노란 상자가 옮겨집니다. 크기는 16·32·64·128 픽셀, 방향키로 1픽셀(Shift는 8픽셀)씩 움직입니다. "개선 최대"·"악화 최대"는 개선 지도의 창 평균이 가장 크거나 작은 곳으로 옮깁니다.
- **수치:** 영상 아래에 장면·영역 PSNR과 평균 오차를 적고, 아래 표에 확대 영역(PSNR·SAM·ERGAS·평균 오차), 장면 전체(PSNR·SAM·ERGAS·SCC·Q2n), 데이터셋 평균을 모델별로 놓았습니다. 차이는 비교 기준 대비이며 파랑이 개선입니다.
- **표시 설정:** 비교 기준 모델, 오차 지도 밴드, 색 범위, 확대 영역 밝기, LR MS·PAN 입력 표시를 바꿉니다. 확대 영상에 마우스를 올리면 모든 열에 같은 픽셀이 표시되고 그 픽셀 값이 아래에 나옵니다.
- 주소창의 `#ds=...&scene=...&x=...&y=...&w=...`가 현재 데이터셋·장면·영역을 담고 있어 링크로 같은 화면을 공유할 수 있습니다.
- 글꼴(IBM Plex Sans KR·Mono)은 Google Fonts에서 받으며, 인터넷이 없으면 시스템 글꼴로 표시됩니다.

## 수치 확인

- 원본의 예측 PSNR은 성능 53·54번이 보관한 결과 이미지 기록(`result_images.json`, CUDA)과 네 데이터셋의 표준 6장면 모두에서 0.0007 dB 이내로 일치했습니다(로컬 MPS 추론). C2MS-SP·SP-Lite의 데이터셋 평균은 `common/baseline_images.py`의 seed 42 값과 대조했습니다.
- 예측은 1/8 DN 단위 int16으로 저장합니다. 브라우저가 저장값으로 계산한 장면 전체 수치는 원래 예측으로 계산한 값과 PSNR 0.0004 dB, SAM 0.0001° 이내였습니다.
- PSNR은 `rr_metrics`와 같이 peak 2047의 밴드별 PSNR 평균입니다(GF2는 10bit라 약 6 dB 높게 나옵니다).
- 화면의 수치는 seed 42 가중치 한 개의 결과입니다. 실험 판정은 `docs/judgement_criteria.md`의 시드 평균을 따릅니다.

## 모델 추가

`registry.json`의 `models`에 항목을 추가한 뒤 그 모델만 내보냅니다. 기존 모델의 결과는 그대로 유지됩니다.

```bash
python SSA-MRN/viewer/export.py --models <모델 id>
```

- **저장소 코드로 만드는 모델:** `module`(SSA-MRN 기준 경로), `factory`, `args`(`"{bands}"`는 밴드 수, `"{dataset}"`은 데이터셋 id로 바뀜), 데이터셋별 인자 `kwargs_by_dataset`과 데이터셋별 `checkpoints`를 적습니다. 입력은 (pan, lms, ms)를 데이터셋 `scale`로 나눈 값이고 출력에 다시 곱합니다. 체크포인트의 state dict 키가 `model`이 아니면 `state_key`를 적습니다.
- **다른 코드로 만든 예측:** 다른 팬샤프닝 연구 코드로 학습·추론했다면 `"predictions": {"QB": "<h5 경로>"}`를 적습니다. H5에는 `pred` 데이터셋을 RR test와 같은 순서·크기 `(20, C, 256, 256)`의 DN 값으로 넣습니다.
- 체크포인트나 예측이 없는 데이터셋은 그 모델에서 건너뜁니다. `label`·`short`·`role`·`description`은 화면에 그대로 표시됩니다.

## 배포 갱신

모델을 추가하거나 데이터를 다시 만든 뒤 배포 저장소에 `index.html`·`app.js`·`data/`를 덮어써서 푸시하면 1-2분 뒤 사이트에 반영됩니다. 예측 데이터는 CtrS 저장소에 넣지 않고 배포 저장소에만 둡니다.

```bash
git clone https://github.com/BIYONGHIYON/CtrS-viewer.git ../CtrS-viewer
rsync -a --delete SSA-MRN/viewer/data/ ../CtrS-viewer/data/
cp SSA-MRN/viewer/index.html SSA-MRN/viewer/app.js ../CtrS-viewer/
git -C ../CtrS-viewer add -A
git -C ../CtrS-viewer commit -m "결과 비교 사이트 데이터 갱신"
git -C ../CtrS-viewer push
```
