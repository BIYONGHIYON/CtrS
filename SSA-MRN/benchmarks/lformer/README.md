# LFormer

| 평가자 | 설정 | 상태 |
|:---:|:---:|:---:|
| 김경찬 | `author` WV3 (저자 가중치) · `retrain` QB · GF2 (논문 보충자료 설정), seed 42 | 계획 |

**출처:** Hou 외, Linearly-evolved Transformer for Pan-sharpening, ACM MM, 2024, [DOI 10.1145/3664647.3680979](https://doi.org/10.1145/3664647.3680979) ([arXiv](https://arxiv.org/abs/2404.12804), [OpenReview·보충자료](https://openreview.net/forum?id=pCxZTmGr4O)) · 공식 코드 [coder-JMHou/LFormer](https://github.com/coder-JMHou/LFormer) `cb10d43` · 라이선스 없음(코드·WV3 가중치 재배포 불가, 하위 모듈 참조만) · 이슈 [#184](https://github.com/BIYONGHIYON/CtrS/issues/184)

**결과:** 아직 평가하지 않았습니다.

[평가 표](../README.md#models) · [결과](#results) · [해석](#verdict) · [방법](#method) · [그림](#figures) · [재현 자료](#evidence)

<a id="results"></a>

## 결과

아직 평가하지 않았습니다. 비교할 보고값은 다음과 같습니다.

| 센서 | PSNR ↑ | SAM ↓ | ERGAS ↓ | Q ↑ | 출처 | 우리 평가 설정 |
|---|---:|---:|---:|---:|---|---|
| WV3 | 39.0748 | 2.8985 | 2.1645 | 0.9193 (Q8) | 논문 표 1 | `author` |
| QB | 38.674 | — | — | — | SRINO 보충자료의 재학습값 (논문에 QB 없음) | `retrain` seed 42 |
| GF2 | 44.1958 | 0.6481 | 0.5778 | 0.9851 (Q4) | 논문 표 1 | `retrain` seed 42 |
| WV2 | — | — | — | — | — | `author` WV3 가중치 전이(참고) |

GF2 PSNR은 논문마다 약 43-45 dB와 49-50 dB 계열이 섞여 있어 우리 평가기 값과 직접 비교하지 않습니다.

<a id="verdict"></a>

## 해석과 정리

평가 후 작성합니다.

<a id="method"></a>

<details>
<summary><b>방법</b> · 목적과 상태 · 모델 · 학습 설정 · 실행 순서</summary>

### 목적과 상태

QB RR PSNR 38.674 dB(SRINO 보충자료의 재학습값)가 C2MS-SP보다 높게 보고된 가벼운 트랜스포머입니다. 첫 단계에서만 교차 주의를 계산하고, 이후 단계는 그 주의 지도를 1차원 합성곱으로 갱신해 재사용합니다. 같은 데이터·지표로 이 수치가 재현되는지 확인합니다.

| 설정 | 센서 | 상태 |
|---|---|---|
| `author` | WV3, WV2 전이 | 계획 |
| `retrain` seed 42 | QB, GF2 | 계획 (논문 보충자료 설정) |

### 모델

공식 코드 `upstream/models/LFormer.py`의 `AttnFuseMain`이며, 크기는 공식 `configs/config_<센서>.yaml` 값입니다. 논문에는 채널 수가 없습니다.

- **입력 → 출력:** PAN을 Q, LMS를 K·V로 한 교차 주의(8×8 창, 8헤드)로 시작합니다. 주의 지도를 1×3 합성곱으로 갱신하는 단계 5개와 고주파 가지(PAN·LMS 라플라시안 초기화 → Resblock)를 거쳐 잔차를 만들고 LMS에 더합니다.
- **크기:** WV3 lms 8 / attn 64 / hp 64 (2,274,635개), QB 4 / 32 / 48 (1,068,539개), GF2 4 / 64 / 64 (2,273,307개).
- **어댑터:** 공식 입력 `(lms, pan)`으로 호출하고 ms는 쓰지 않습니다. 추론 출력은 공식 `val_step`처럼 [0, 1]로 자르고, 학습 중에는 자르지 않습니다. 공식 코드는 고치지 않습니다.

**논문과 공개 코드의 차이:** 공개 코드와 저자 가중치는 서로 맞습니다. 하지만 논문의 모델과는 아래가 다릅니다. 평가는 공개 코드로 하고, 차이는 이 보고서에만 적습니다.

| 항목 | 논문 (ACM 최종본) | 공개 코드·저자 가중치 |
|---|---|---|
| 첫 단계 주의 | 창으로 나누지 않음 (창 주의 확장은 표 6에 따로: GF2 43.0769 dB, 0.573M) | 8×8 창 |
| 주의 갱신 합성곱 | 1×5 (표 5) | 1×3 |
| 파라미터 수 | 0.589M (GF2, 표 5) | GF2 2.27M |
| README | — | 다른 프로젝트(BiMPan) 설명이 섞임, "Complete code will be released soon" |

### 추론 방식

공식 test는 컴파일된 `PatchMergeModule`(Linux `.so`, Python 3.7-3.10·3.12)로 영상을 타일링합니다. 이 모듈은 추론에만 쓰이고, 학습(`train_step`)에는 쓰이지 않습니다. 우리는 네 센서·두 설정 모두 `patch_merge=False`로 영상 전체를 한 번에 추론합니다. 그래야 Windows 공용 서버에서도 돌고 설정 간 조건이 같습니다.

공식 타일링과의 출력 차이는 Linux에서 `compare_patch_merge.py`로 재서 `results/patch_merge_check.json`에 적습니다. 이 방식은 실행 전에 정했습니다. 차이가 얼마로 나오든 추론 방식은 바꾸지 않고 차이만 보고합니다(test를 보고 고르지 않음).

### 학습 설정 (`retrain`)

논문 설정(`paper`)으로 학습합니다. ACM 최종본 본문은 학습 설정을 보충자료로 미루고, 보충자료 PDF 1절(Experimental Settings)에 설정이 있습니다. ACM DL에는 발표 영상만 있고, PDF는 [OpenReview 투고 페이지](https://openreview.net/forum?id=pCxZTmGr4O)에서 받았습니다. 값은 arXiv v1 4.1절과 같습니다.

| 항목 | 사용: `paper` (보충자료 1절) | 참고만: `code` (공식 config) |
|---|---|---|
| 에폭 | 800 | 500 |
| 배치 | 32 | 32 |
| 옵티마이저 | Adam (0.9, 0.999), weight decay 0.1 | Adam, weight decay 1e-6 |
| 학습률 | 3e-4, 300·500에폭에 0.1배 | 1e-3, 250에폭마다 0.1배 |
| 출처 | 논문 보충자료 (RTX 4090 1장) | `configs/config_qb.yaml`·`config_gf2.yaml`, `models/pipeline.py` |

논문 설정을 쓰는 이유는 세 가지입니다. AGENTS.md 8절이 논문 설정을 우선하고, 비교 대상인 논문 표 1 수치도 이 설정으로 보고됐습니다("In all experiments"). 또 저자 WV3 가중치(`epoch500`)가 어느 설정으로 학습됐는지는 알 수 없습니다. 공식 코드가 25에폭마다 저장하므로 800에폭 학습에서도 500에폭 파일이 생기기 때문입니다.

공개 코드 모델이 논문 모델과 다르다는 점(위 표)은 어느 설정을 쓰든 같으므로 차이로만 적습니다. weight decay 0.1이 학습을 막는지는 QB 첫 세션의 validation 손실 추이로 확인합니다. 이상하면 이슈에서 다시 논의합니다(설정을 바꾸면 처음부터 다시 학습).

**공통:**
- 손실은 L1 + 0.1·(1 − SSIM)입니다. 공식 `get_loss('l1ssim')`이고 논문 식 15-17과도 같습니다.
- 섞어서 배치를 만들고 마지막 미완성 배치는 버립니다. `cudnn.deterministic`을 켭니다.

**메모리:** 배치 32(64×64)의 학습 활성값은 CPU 측정으로 QB 약 7GB, GF2 약 10GB입니다. GPU 실측이 필요합니다. 들어가지 않으면 배치를 줄이지 않고 `--checkpointing`을 씁니다. 각 단계를 역전파 때 다시 계산하는 방식이고, 기울기가 같음을 확인했습니다(GF2 기준 표본당 약 300MB → 110MB). 썼는지는 `history.json`의 `meta.checkpointing`과 보고서에 적습니다.

**우리 규칙 때문에 바꾸는 점** (학습 조건 변경은 아님):
- seed는 42입니다. 공식 값은 3407입니다.
- validation은 `valid_<센서>.h5`로 매 에폭 공식 손실을 계산합니다. 출력은 자르지 않습니다. 공식 코드는 학습 파일로 검증하고 25에폭마다 저장만 합니다.
- best.pt는 validation 손실이 가장 낮은 에폭입니다.
- `latest.pt`로 이어서 학습할 수 있습니다.

| 항목 | C2MS-SP | LFormer (`retrain`) |
|---|---|---|
| 학습 데이터·분할 | PanCollection 학습·validation | 같음 |
| 정규화 | QB·WV3 2047, GF2 1023 | 같음 |
| 손실 | L1 + 거친 쌍 L1 + 0.46 분광 | L1 + 0.1 (1 − SSIM) |
| 에폭(출처)·배치·학습률 | 150 · 32 · 1e-4 → 1e-6 cosine | 800 (논문 보충자료) · 32 · 3e-4, 300·500에폭에 0.1배 |
| 가중치 선택 | validation MSE 최소 | validation 손실(공식 손실) 최소 |
| 후처리 | 관측 일치 사영(K10) | 없음 |
| 시드 | 42·43·44 | 42 |
| 파라미터 수 | | QB 1,068,539 · GF2 2,273,307 |

### 실행 순서

경로는 저장소 루트 기준입니다. `data/`는 각자의 PanCollection 위치로 바꾸세요.

```bash
# 0. 준비: 브랜치, 하위 모듈(고정 커밋), 계획 상태 등록 (catalog.json, benchmarks/README.md 평가 표)
git switch -c lformer-benchmark
git submodule add https://github.com/coder-JMHou/LFormer SSA-MRN/benchmarks/lformer/upstream
git -C SSA-MRN/benchmarks/lformer/upstream checkout cb10d43
sha256sum SSA-MRN/benchmarks/lformer/upstream/weights/WV3/LFormer_epoch500.pth   # 8b521a5f...170a 확인
pip install einops            # 공식 코드 의존성 (torch·h5py·numpy 외)

# 1. author WV3: 어댑터 확인 → 전체 평가 → WV2 전이
python SSA-MRN/common/benchmark_evaluate.py --model lformer --setting author --sensor WV3 --limit 2 --skip-fr \
  --checkpoint SSA-MRN/benchmarks/lformer/upstream/weights/WV3/LFormer_epoch500.pth
python SSA-MRN/common/benchmark_evaluate.py --model lformer --setting author --sensor WV3 --device cuda \
  --checkpoint SSA-MRN/benchmarks/lformer/upstream/weights/WV3/LFormer_epoch500.pth
python SSA-MRN/common/benchmark_evaluate.py --model lformer --setting author --sensor WV2 --device cuda \
  --checkpoint SSA-MRN/benchmarks/lformer/upstream/weights/WV3/LFormer_epoch500.pth

# 2. 타일링 차이 기록 (Linux, Python 3.10 또는 3.12)
python SSA-MRN/benchmarks/lformer/compare_patch_merge.py --rr data/test_wv3_multiExm1.h5 \
  --fr data/test_wv3_OrigScale_multiExm1.h5 --device cuda

# 3. retrain: 동작 확인 (임시 폴더, 2에폭) → 에폭당 시간으로 전체 시간 계산
python SSA-MRN/benchmarks/lformer/train.py --sensor QB --recipe paper --train data/train_qb.h5 --val data/valid_qb.h5 \
  --limit 256 --epochs 2 --out /tmp/lformer_smoke
#    전체 데이터 1에폭 시간도 재 봅니다 (--epochs 1 --out /tmp/lformer_epoch)

# 4. 본 학습 (공용 서버면 시작 전에 팀에 알림; Kaggle이면 --deadline과 --resume)
python SSA-MRN/benchmarks/lformer/train.py --sensor QB  --recipe paper --train data/train_qb.h5  --val data/valid_qb.h5  --device cuda
python SSA-MRN/benchmarks/lformer/train.py --sensor GF2 --recipe paper --train data/train_gf2.h5 --val data/valid_gf2.h5 --device cuda
cp SSA-MRN/benchmarks/lformer/checkpoints/retrain_QB_s42/history.json SSA-MRN/benchmarks/lformer/results/retrain_QB_s42/

# 5. retrain 평가 (checkpoints/retrain_<센서>_s42/best.pt를 자동으로 읽음)
python SSA-MRN/common/benchmark_evaluate.py --model lformer --setting retrain --sensor QB  --seed 42 --device cuda
python SSA-MRN/common/benchmark_evaluate.py --model lformer --setting retrain --sensor GF2 --seed 42 --device cuda

# 6. 예측 Release + 실제 MATLAB 채점 → 그림 → 보관 확인
#    (author는 WV3·WV2, retrain은 QB·GF2의 RR 예측 zip; 방법은 benchmarks/README.md)
gh workflow run matlab-benchmark-author.yml -f model=lformer -f setting=author -f release=predictions-benchmarks-lformer-author
gh workflow run matlab-benchmark-author.yml -f model=lformer -f setting=retrain -f release=predictions-benchmarks-lformer-retrain
python SSA-MRN/common/benchmark_evaluate.py --model lformer --setting retrain --sensor QB --seed 42 --figures --device cuda
python SSA-MRN/common/release_weights.py SSA-MRN/benchmarks/lformer
python SSA-MRN/common/verify.py --benchmark lformer --releases
python SSA-MRN/common/index.py --check
```

</details>

<a id="figures"></a>

<details>
<summary><b>그림</b> · 학습·평가 그래프 · 결과 이미지</summary>

학습 후 `results/learning.png`와 고정 6장면 이미지를 넣습니다. `author` WV3는 학습 곡선이 없습니다.

</details>

<a id="evidence"></a>

<details>
<summary><b>재현 자료</b> · 코드, 가중치, 설정, 환경</summary>

- 공식 코드: 하위 모듈 `upstream/` (`cb10d43`, 수정 없음, 라이선스 없음)
- 저자 WV3 가중치: `upstream/weights/WV3/LFormer_epoch500.pth`, SHA-256 `8b521a5f8e8bbe8104ebb11f1c97e91d9a0a19d3f994d342cd3952551827170a`. 재배포하지 않습니다.
- retrain: `checkpoints/retrain_<센서>_s42/best.pt`(Git), `latest.pt`·`initial.pt`(Release `weights-benchmarks-lformer`). 공개 범위는 결정 후 적습니다.
- 원본 학습 설정: 라이선스가 없어 `upstream/configs/*.yaml`을 복사하지 않습니다. 값은 `config.json`에 적고, `provenance/`에는 환경(torch·CUDA·GPU)과 하위 모듈 커밋만 둡니다.

</details>
