# LIB-HSI 로컬 RGB–HSI 실험

## VS Code 실행

`C:\Users\pilot\Desktop\CtrS`를 작업 폴더로 열고 Run and Debug에서 다음을 선택한다.
Python / Python Debugger 확장이 필요하다. 기존에 다른 Python을 선택했다면
`Python: Select Interpreter`에서 `.venv\Scripts\python.exe`를 선택한다.

1. **LIB: smoke (external E drive)**: 실제 데이터로 optimizer 2회 실행, 검증 2배치.
2. 성공하면 **LIB: train optimized (3060 Ti)**: 공식 train 393장 / validation 45장 사용.
3. 중단 후 **LIB: resume best**: validation MSE가 가장 낮은 epoch의 가중치·optimizer로 재개.
4. 학습 후 **LIB: evaluate test (best)**: validation MSE가 가장 낮은 모델로 test 75장 평가.

속도 측정/본학습에는 Ctrl+F5(디버깅 없이 실행)를 사용한다.
`Terminal → Run Task → LIB: train`도 사용할 수 있다.
PowerShell 터미널 명령은 다음과 같다. 가상환경 활성화는 필수가 아니다.

```powershell
.\.venv\Scripts\python.exe SSA-MRN\scripts\train_lib.py --smoke
.\.venv\Scripts\python.exe SSA-MRN\scripts\train_lib.py
.\.venv\Scripts\python.exe SSA-MRN\scripts\train_lib.py --resume SSA-MRN\experiments\checkpoints\lib_rgb_hsi_fast\best.pt
.\.venv\Scripts\python.exe SSA-MRN\scripts\train_lib.py --evaluate
```

경로와 학습 설정은 `SSA-MRN/configs/lib_rgb_hsi.json`에서 변경한다.
현재 기본 경로는 `E:/윤지/병현/SSA-MRN Data/RGB-HSI/LIB-HSI`이며 그 아래에
`train`, `validation`, `test`와 각 split의 `rgb`, `reflectance_cubes`가 있어야 한다.
복사 중에는 전체 학습을 시작하지 않는다. 파일 수와 DAT 크기를 검사해 불완전한 복사를 거부한다.
전체 513쌍 검사: `.\.venv\Scripts\python.exe SSA-MRN\scripts\check_lib.py`.
외장하드에서 사전 테스트하려면 다음 명령을 사용한다.

```powershell
.\.venv\Scripts\python.exe SSA-MRN\scripts\train_lib.py --smoke --data-root 'E:\윤지\병현\SSA-MRN Data\RGB-HSI\LIB-HSI'
```

## 실험 정의와 메모리

기존 PAN–MS 논문 재현과 분리한 **실험용 RGB–HSI 확장**이다.
RGB 3채널을 backbone과 각 SSA guide convolution에 직접 넣는다.
204밴드 LR HSI를 학습 가능한 1×1 convolution으로 8개 latent feature로 압축하고,
기존 복원 모델의 3단계 SSA-MRN 연산(K=6)을 적용한 뒤 204밴드 residual을 복원한다.
최종 출력은 bicubic LR HSI 보간 + 학습 residual이다. 8개 feature는 파장 밴드가 아니다.
PAN–MS 체크포인트와 호환되지 않으며 이 구조의 연구적 유효성은 별도 검증해야 한다.

3060 Ti 8GB / RAM 16GB 최적화 설정: 128×128 HR 패치, 32×32 LR HSI,
train batch 4, accumulation 1(유효 batch 4), validation/test batch 16,
CUDA AMP, workers 2, prefetch 2, persistent workers.
한 학습 배치가 한 장면의 crop 4개이며 한 검증 배치가 한 장면의 tile 16개다.
각 장면은 하나의 worker에서 읽어 중복 큐브 읽기를 피한다. 이 설정에서 batch_size는 4로 유지한다.
큰 배치는 유효 batch/optimizer 갱신 수를 바꾸므로 GPU 사용률만 높이려고 늘리지 않는다.
RAM에 전체 데이터셋을 올리지 않는다. ENVI 메타데이터·바이트 수는 memmap으로 검사하고,
현재 read_mode=stripes는 필요한 원본 BIL 행을 64행씩 연속 읽고 장면별로 캐시한다.
worker마다 한 장면만 유지하며 큐브 캐시는 최대 약 204 MiB다.
train/validation loader는 각각 2 workers이며 캐시 합계 상한은 약 816 MiB다.
crop 회전 시 임시 복사와 prefetch/pinned batch 버퍼·Python 프로세스 메모리도 추가로 사용한다.
RAM에 80GB 이상의 전체 LIB 데이터셋을 올리지는 않는다.
원본 BIL의 회전 패치를 직접 memmap으로 읽으면 HDD 랜덤 접근이 매우 느려질 수 있다.
장면 순서를 epoch마다 shuffle하고, 한 장면의 random crop 4개를 연이어 처리해 캐시를 재사용한다.
기본 유효 batch 4는 같은 장면의 서로 다른 random crop 4개이다.
파일은 원본 그대로 두며 RGB와 HSI에 동일한 crop/flip을 적용한다.
HSI는 제공 `create_patches.py`와 같이 시계 방향 90도 회전한다.
실제 RGB–HSI 정합 정확도가 입증된 것은 아니므로 preview와 랜드마크를 확인해야 한다.
OOM 발생 시 우선 val_batch_size를 8로 줄여 smoke를 확인한다. validation workers도 1로 줄이면
동일 장면의 검증 tile batch를 서로 다른 worker에서 중복 읽는 것을 피할 수 있다.
256 패치나 batch 증가 전에는 같은 설정으로 smoke 메모리를 확인한다.

LIB의 원본 RGB와 HSI는 둘 다 512×512이다. 본 실험은 반사율을 고정 [0,1]로 clip하고
HSI 패치에 antialiased bicubic x4 축소를 적용한다(외부 Gaussian blur 없음).
이 합성 축소 규칙은 코드와 지표 파일에 기록된다. NaN/Inf는 오류로 처리한다.
사전 검사한 첫 train cube는 약 0.0083–1.2222 범위이며 약 0.40%의 값이 [0,1] 밖에 있다.
이 설정은 그 값을 clip하므로 원본 반사율 전체를 보존하는 실험과 구분한다.
이 결과는 합성 공간 초해상도 실험이며 실제 저해상도 센서 HSI 복원 성능의 증거가 아니다.
본학습은 장면당 epoch마다 random crop 4개, 검증·시험은 비중첩 타일로 512 전체를 덮는다.
split은 배포본을 유지하고, train/validation 동일 파일명 누출은 거부한다.
건물·촬영일 단위의 추가 누출 감사는 별도로 필요하다.

## 결과와 해석

`SSA-MRN/experiments/checkpoints/lib_rgb_hsi_fast/` 아래:
기존 `lib_rgb_hsi/` 학습 결과는 보관하고 새 가중치로 시작한다.

- `best.pt`: validation MSE가 가장 낮은 모델·optimizer·AMP scaler·epoch·torch RNG·config.
  학습 재개와 시험 평가에서 사용하는 체크포인트다.
- `latest.pt`: 마지막 완료 epoch 백업. 기본 실행 구성에서 가중치 소스로 사용하지 않는다.
- `history.jsonl`: epoch별 train MSE, validation model/bicubic PSNR·SAM, 시간·GPU 메모리.
- `validation_metrics.json`: 장면별 지표와 장면 평균. PSNR은 모든 밴드 MSE에서 계산하며 range=1.
- `preview_rgb_gt_bicubic_prediction.png`: 좌→우 실제 RGB, GT, bicubic, prediction.
  HSI preview는 ENVI default bands(0-based 69/52/18)를 [0,1] 표시하며 실제 RGB 색과 다를 수 있다.
- `test/metrics.json`: 별도 test 결과. 검증·시험에는 output clipping 없이 지표를 계산한다.

SAM은 degree이며 GT spectrum norm이 1e-6 이하인 픽셀만 제외한다.
smoke 결과는 `smoke/`에 따로 저장하고 부분 검증임을 표시한다.
기존 `latest.pt`가 있는 본학습 폴더에는 새 학습을 덮어쓰지 않는다.
재개하려면 `--resume SSA-MRN/experiments/checkpoints/lib_rgb_hsi_fast/best.pt`, 새 실험이면
`--output-dir SSA-MRN/experiments/checkpoints/lib_rgb_hsi_run2`를 사용한다.
smoke 통과는 실행 가능성 확인이다. 학습 성능 향상은 본학습 후 bicubic 및 HSI-only
ablation과 비교해야 하며, SSIM/ERGAS·정합 감사·다중 seed는 후속 연구에 추가한다.
best 재개는 best epoch 다음부터 다시 학습한다. best 이후의 epoch는 다시 실행될 수 있다.

## 환경 재설치

Python 3.10, 공식 CUDA PyTorch wheel을 프로젝트 venv에 설치한다.
별도 CUDA Toolkit 설치는 이 wheel 실행에 필요하지 않다.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r SSA-MRN\requirements.txt
git submodule update --init --recursive
```

CUDA wheel 선택 참고: https://pytorch.org/get-started/locally/

## 초기 실행에서 확인한 사항 (2026-10-01, 최적화 전 설정)

- 프로젝트 `.venv`: Python 3.10 / PyTorch 2.7.1+cu128, RTX 3060 Ti CUDA 실행 확인.
- D 드라이브: train 393 / validation 45 / test 75 RGB–HSI 쌍의 헤더·크기 검사 통과.
  세 split 사이 동일 파일명 없음. 전 픽셀·체크섬 무결성이나 정합 검증을 의미하지 않는다.
- D 드라이브 실제 파일로 학습 2 step + validation 2배치 + checkpoint/preview 저장 통과.
  최종 smoke 연산·I/O 약 3.81초(초기 import·데이터 검사 제외), peak allocated 132.2 MiB,
  peak reserved 166 MiB. CUDA context·화면 표시·타 앱의 VRAM은 이 수치에 포함되지 않는다.
- 새로운 RGB–HSI 학습/gradient 누적/재개/시험 평가와 기존 PAN–MS를 포함한 테스트 20개 통과.
- 본학습 100 epochs는 아직 실행하지 않았다. smoke PSNR을 학습 성능 결과로 해석하지 않는다.

## 최적화 구현

- 8개 latent SSA 분기를 grouped convolution으로 묶어 GPU 호출을 줄인다.
  기존 branch별 weight/bias와 width/height transpose, softmax를 그대로 사용한다.
  state_dict 구조를 유지하며 loop/grouped 출력과 gradient 비교를 통과했다.
- float32 antialiased bicubic x4 열화를 GPU에서 계산한다. CPU/CUDA 열화 일치 테스트를 통과했다.
  모델 AMP와 별도로 float32로 실행하므로 GT/열화 규칙을 바꾸지 않는다.
- 배치 1×누적 4를 실제 배치 4×누적 1로 바꿨다. 배치 간 연산이 독립인 이 모델의 유효 배치는 동일하다.
- CPU 로더에서 scene/cache를 연속 CHW로 정리하고 numpy crop/clip/flip을 수행한다.
- 원본 BIL에서 patch에 필요한 행 구간만 읽어 캐시한다(read_mode=stripes).
  4개 random crop이 사용하지 않는 행을 읽거나 전체 큐브를 회전·복사하는 비용을 줄인다.
  추가 데이터 파일을 만들지 않고 float32·원본 데이터·crop/회전 정의를 그대로 유지한다.
  whole/stripes 제한 실험의 train MSE와 bicubic MSE가 일치했고 toy 회전/경계 테스트를 통과했다.
- Windows worker 2개가 읽기·crop을 선행하며 pinned memory와 별도 CUDA stream으로 전송/열화를 겹친다.
- fused Adam, cuDNN autotuning을 사용한다. 첫 배치는 프로세스 시작과 autotuning으로 느릴 수 있다.
  channels_last는 이 PC의 측정에서 느려 기본값 false로 두었다.
- 매 step loss.item()/지표 CPU 복사로 GPU를 기다리지 않는다. loss 표시는 20배치마다 갱신한다.
  검증 지표는 GPU에서 배치별로 계산·장면별 누적하고 끝에서 CPU로 옮긴다.
- crop/flip과 장면 순서는 seed/epoch/index로 결정해 persistent workers에서도 재개 가능한 순서를 사용한다.
  비결정적인 cuDNN 알고리즘/AMP 때문에 비트 단위 동일 결과를 보장하지는 않는다.

GPU만 측정한 optimizer 처리량은 이전 batch 1/branch loop 약 13.3 patches/s,
batch 4/grouped 약 150.0 patches/s다(데이터 I/O 제외). GPU 사용률 100%를 보장하는 수치는 아니다.
whole 모드의 읽기량은 epoch마다 약 87 GiB이며 stripes 모드는 train의 필요 구간만 읽어 이를 줄인다.
train_cube_requested_gib / validation_cube_requested_gib는 fromfile에 요청한 큐브 바이트 수다.
Windows 파일 캐시·read-ahead 때문에 물리 디스크에서 실제 읽힌 양과 다를 수 있다.
최적화 후에도 저장장치/USB 속도가 한계가 될 수 있다.
성능 측정 파일은 `experiments/results/lib_optimization/` 아래에 저장한다.
`--profile` 실행은 train GPU stream 시간, loader 대기, train/validation 시간을 기록한다.
loader 대기와 GPU 시간은 겹칠 수 있으므로 합산해서 전체 시간을 계산하지 않는다.
`--max-train-scenes`, `--max-val-scenes`는 제한된 benchmark이며 본학습에 사용하지 않는다.

## 최종 전체 데이터 검증 결과

2026-10-01, 외장 Samsung T7 Shield의 train 393장/validation 45장으로 새 가중치에서
최적화 설정 2 epochs를 실제 실행했다. 각 epoch는 동일하게 train 1,572 random patches와
validation 720 tiles를 사용한다. train batch 4라 진행 막대는 393 batches, validation batch 16이라
45 batches로 표시된다. 데이터 수를 줄인 측정이 아니다.

| 항목 | 이전 사용자 학습 로그 | 최적화 전체 검증 |
|---|---:|---:|
| epoch 시간 | 약 1,050초 | 114.75초 / 108.53초 |
| train / validation (2번째 epoch) | 분리 기록 없음 | 89.56초 / 18.97초 |
| 최대 모델 할당 GPU 메모리 | 132 MiB | 1,453 MiB |
| 전체 큐브 요청 읽기량 | 약 87.26 GiB/epoch | 약 71.57 GiB/epoch |

이전 로그와 최종 실행은 완전히 통제된 동일 실행 비교가 아니므로 순수한 코드 변경 효과만
분리한 수치는 아니다. 실제 현재 PC에서의 완료 시간 개선을 보여준다. CUDA-only benchmark는
같은 입력을 사용한 별도 측정이다. 현재 시간 유지 시 100 epochs는 대략 3시간이며 변동될 수 있다.
새 모델의 최종 정확도 개선을 보장하는 결과는 아니다.

전체 검증 산출물은 `experiments/results/lib_optimization/full_stripes/`에 보관한다.
본학습은 `experiments/checkpoints/lib_rgb_hsi_fast/`에서 1 epoch부터 새로 시작한다.
기존 학습은 `experiments/checkpoints/lib_rgb_hsi/`에 그대로 남아 있다.
