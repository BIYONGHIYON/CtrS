# LIB-HSI RGB–HSI 확장과 로컬 학습 최적화

이 브랜치는 기존 PAN–MS 재현을 출발점으로 LIB-HSI에서 RGB 유도 HSI 공간 초해상도의
실행 가능성을 확인하는 연구용 구현이다. 전체 학습은 진행 중이며 최종 시험 성능을 보고하는 자료가 아니다.

## 모델 변경

| 항목 | 기존 4밴드 PAN–MS | LIB RGB–HSI |
|---|---|---|
| HR guide | PAN 1채널 | 실제 RGB 3채널 |
| LR spectral input | MS 4밴드 | HSI 204밴드 |
| SSA 처리 채널 | 실제 MS 밴드 | 학습 가능한 latent feature 8개 |
| SSAI 내부 차원 | K=4 또는 6 | K=6 |
| 출력/감독 | HR MS 4밴드 | HR HSI 204밴드 |
| 공간 비율 | x4 | 합성 x4 |

LR HSI 204채널을 1×1 convolution으로 8채널에 투영한다. 이를 보간한 HR feature와 RGB를
기존 3단계 SSA-MRN에서 융합하고, 1×1 decoder로 204밴드 보정값을 복원한다.
최종 출력은 LR HSI의 bicubic 보간 + 보정값이다. backbone 초기 입력은 latent 8 + RGB 3 +
처리된 RGB 3 = 14채널이며, 각 SSA guide convolution은 RGB 3채널을 직접 받는다.
8개 latent feature는 특정 파장 밴드가 아니다. 기존 PAN–MS 가중치를 전이하지 않고 새로 초기화한다.
기존 PAN–MS 모델의 기본 1채널 guide 동작과 파라미터 이름은 유지했다.

## 데이터와 실험 프로토콜

- 배포 split 유지: train 393 / validation 45 / test 75쌍, RGB·HSI 모두 512×512, HSI 204밴드.
- float32 ENVI BIL 파일을 읽고 제공 `create_patches.py`와 같이 HSI를 시계 방향 90도 회전한다.
- 반사율 GT를 고정 [0,1]로 clip한다. 첫 train cube에서는 약 0.40%의 값이 이 범위를 벗어났다.
- 128×128 HR GT에서 antialiased bicubic x4 축소로 32×32 LR HSI를 만든다. 별도 Gaussian blur 없음.
- train은 epoch마다 장면별 random crop 4개와 RGB/HSI 공동 flip, 총 1,572패치.
- validation/test는 128×128 비중첩 타일 16개로 장면 전체를 덮는다. validation은 720타일.
- 손실은 최종 204밴드 출력과 GT 사이 MSE. 검증 scene-mean MSE가 낮은 `best.pt`를 선택한다.
- PSNR은 장면 전체 204밴드 MSE, data range=1에서 계산하고 장면 평균을 보고한다.
  SAM은 degree이며 GT spectrum norm ≤1e-6 픽셀을 제외한다. 예측은 지표 계산 시 clip하지 않는다.

LIB 원본 RGB와 HSI 해상도가 같으므로 이 실험은 **합성 공간 초해상도**다.
실제 저해상도 센서 HSI에 대한 성능을 입증하지 않는다. 회전 보정이 정합 정확도 검증을 대신하지 않는다.
동일 파일명 split 누출은 없지만 건물/촬영일/인접 위치 단위 누출은 추가 감사가 필요하다.
Agro/BUSI 학습 로더나 통합 학습은 이번 구현에 포함하지 않았다.

## 성능 최적화

1. 실제 batch 1×누적 4 → batch 4×누적 1로 바꿔 유효 배치 4를 유지한다.
2. 독립 SSA 8개 분기를 grouped convolution과 batched spatial softmax로 묶는다.
   기존 weight/bias·transpose·softmax를 유지하며 loop/grouped 출력·gradient 테스트를 통과했다.
3. 원본 BIL의 필요한 행을 64행 구간으로 읽고 장면별 캐시해 전체 큐브 읽기·회전을 줄인다.
   추가 데이터 파일이나 양자화 없이 float32를 유지한다. 원래 로더와 실제 픽셀이 일치한다.
4. Windows persistent workers 2개, prefetch 2, pinned memory, 별도 CUDA stream으로
   읽기/전처리/전송을 겹친다. train batch는 한 장면, validation batch 16은 한 장면 전체다.
5. x4 열화를 float32 GPU 연산으로 이동한다. CPU/CUDA 열화 일치 테스트를 통과했다.
6. AMP, fused Adam, cuDNN autotuning을 사용한다. channels_last는 측정에서 느려 비활성화했다.
7. 매 step의 loss.item()/지표 CPU 전송을 줄이고 GPU에서 장면별 검증 지표를 누적한다.
8. crop/flip과 장면 순서를 seed/epoch/index로 생성해 persistent worker에서도 재개 순서를 유지한다.

## 로컬 검증 결과 (2026-10-01)

Ryzen 5600X / RTX 3060 Ti 8GB / RAM 16GB / 외장 Samsung T7 Shield,
Python 3.10 / PyTorch 2.7.1+cu128에서 실행했다.

| 측정 | 결과 |
|---|---:|
| 이전 사용자 실행 로그의 epoch 시간 | 약 1,050초 |
| 최종 최적화 검증 epoch 1 / 2 | 114.75초 / 108.53초 |
| 새 본학습 epoch 1 / 2 | 117.92초 / 111.53초 |
| 최적화 검증의 peak allocated GPU memory | 약 1,453MiB |
| 전체 큐브 요청 읽기량(whole → stripes) | 약 87.26 → 71.57GiB/epoch |
| CUDA-only optimizer 처리량(loop batch 1 → grouped batch 4) | 13.3 → 150.0패치/초 |

전체 데이터와 패치 수를 줄인 측정이 아니다. 이전 사용자 로그와 새 실행은 완전히 통제된
동일 조건 비교가 아니므로 모든 차이를 코드 최적화만의 효과로 해석하지 않는다.
CUDA-only 측정은 같은 입력을 반복한 별도 측정이며 I/O를 제외한다.
바이트 수는 fromfile 요청량으로, OS cache/read-ahead를 포함한 실제 디스크 전송량과 다를 수 있다.
GPU 사용률이 계속 100%일 필요는 없으며 현재도 데이터 읽기 대기가 남아 있다.

새 본학습 첫 두 epoch의 validation PSNR은 30.429 / 30.548dB,
bicubic은 30.061dB다. SAM은 모델 2.446 / 2.450°, bicubic 2.441°로 아직 개선되지 않았다.
최종 정확도, SSIM/ERGAS, HSI-only ablation, multi-seed 결과는 후속 검증이 필요하다.
재현용 소규모 측정 스냅샷은 [benchmark JSON](benchmarks/lib_rgb_hsi_optimization.json)에 기록했다.

## 초기 smoke 이미지

추가 학습 없이 초기 optimizer 2회 실행 후 저장한 best 가중치를 사용한 단일 validation 128패치다.
왼쪽부터 LR HSI, 실제 RGB guide, bicubic, prediction, HSI GT.
HSI 표시에는 0-based bands 69/52/18과 공유 GT 기반 1–99% stretch를 사용한다.
이 이미지의 가중치는 최적화 본학습 완료 모델이 아니다.

![LIB smoke 5-panel comparison](../experiments/results/lib_rgb_hsi_smoke/comparison.png)

## 실행과 저장

[Windows/VS Code 실행 안내](lib_local_training.md)를 따른다.
`configs/lib_rgb_hsi.json`의 `data_root`는 실행 PC에 맞게 수정한다.
100 epochs, lr=1e-4, train batch 4, validation batch 16, seed=42가 기본값이다.
VS Code에서 `LIB: train optimized (3060 Ti)`를 Ctrl+F5로 실행한다.

체크포인트·가상환경·원본 데이터는 Git에 포함하지 않는다. 최적화 본학습 결과는
`experiments/checkpoints/lib_rgb_hsi_fast/`, 이전 실행은 `lib_rgb_hsi/`에 보관한다.
재개와 시험 평가는 `best.pt`를 사용하고 `latest.pt`는 백업이다.
CUDA autotuning/AMP 때문에 비트 단위 동일 학습 결과를 보장하지 않는다.
