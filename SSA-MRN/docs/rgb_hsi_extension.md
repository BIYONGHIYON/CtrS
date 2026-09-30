# LIB-HSI RGB–HSI 확장 연구

기존 PAN–MS SSA-MRN을 기반으로 RGB 유도 HSI 공간 초해상도 모델을 구현했습니다.
LIB-HSI에서 실행 가능성과 학습 속도를 확인했으며, 최종 시험 성능은 아직 검증하지 않았습니다.

## 모델 변경

| 항목 | 기존 PAN–MS | LIB RGB–HSI |
|---|---|---|
| 고해상도 guide | PAN 1채널 | RGB 3채널 |
| 저해상도 입력 | MS 4밴드 | HSI 204밴드 |
| SSA 처리 채널 | 실제 MS 밴드 | 학습 가능한 latent feature 8개 |
| 출력 | HR MS 4밴드 | HR HSI 204밴드 |
| 공간 배율 | x4 | 합성 x4 |

204밴드 LR HSI를 1×1 convolution으로 8채널에 투영하고, 보간한 feature와 RGB를
3단계 SSA-MRN에서 융합했습니다. RGB guide 입력은 3채널, SSAI 내부 차원은 K=6으로
설정했습니다. 1×1 decoder로 204밴드 보정값을 복원하고 bicubic HSI에 더하도록 구성했습니다.
latent feature는 특정 파장 밴드를 의미하지 않습니다. 기존 PAN–MS 모델의 기본 동작은
유지했으며, 확장 모델은 새 가중치로 초기화했습니다.

## 실험 구성

- LIB 배포 split인 train 393 / validation 45 / test 75쌍을 유지했습니다.
- 512×512 RGB와 204밴드 HSI를 사용했습니다. ENVI BIL HSI는 제공 코드와 동일하게
  시계 방향 90도 회전했으며, GT 반사율은 [0,1]로 clip했습니다.
- 128×128 GT를 antialiased bicubic으로 x4 축소해 32×32 LR HSI를 생성했습니다.
  별도 Gaussian blur는 적용하지 않았습니다.
- 학습은 장면별 random crop 4개와 공동 flip으로 epoch당 1,572패치를 사용했습니다.
  검증과 시험은 장면별 비중첩 타일 16개로 전체 영역을 평가하도록 구성했습니다.
- 204밴드 MSE로 학습하고, 검증 장면 평균 MSE가 가장 낮은 `best.pt`를 재개와 평가에
  사용하도록 설정했습니다. 기본 설정은 100 epochs, lr=1e-4, 유효 배치 4, seed=42입니다.
- PSNR은 data range=1의 장면 전체 MSE를 기준으로 장면 평균을 계산했습니다.
  SAM은 GT spectrum norm ≤1e-6인 픽셀을 제외했으며, 평가 시 예측값은 clip하지 않았습니다.

원본 RGB와 HSI의 해상도가 같아 **합성 공간 초해상도**로 구성했습니다.
실제 저해상도 센서에 대한 일반화, RGB–HSI 정합 정확도, 촬영 위치·날짜 단위 split 누출은
추가 검증이 필요합니다. Agro/BUSI 로더와 통합 학습은 이번 구현에 포함하지 않았습니다.

## 학습 속도 최적화

| 변경 | 목적 및 검증 |
|---|---|
| 실제 batch 1×누적 4 → batch 4×누적 1 | 유효 배치 4를 유지하면서 GPU 병렬 처리를 늘렸습니다. |
| SSA 8개 분기를 grouped convolution과 batched softmax로 통합 | 반복 연산을 줄였으며, 기존 분기와 출력·gradient 일치를 검증했습니다. |
| BIL 부분 읽기와 장면별 캐시 | 필요한 행을 64행 구간으로 읽어 전체 큐브 읽기·회전을 줄였습니다. float32를 유지하고 원래 로더와 픽셀 일치를 확인했습니다. |
| persistent workers 2개, prefetch 2, pinned memory, 별도 CUDA stream | 데이터 읽기·전처리·GPU 전송을 겹쳤습니다. 같은 장면의 패치를 배치로 묶어 재읽기를 줄였습니다. |
| float32 GPU x4 열화 | CPU 전처리 부담을 줄였으며, CPU/CUDA 결과 일치를 검증했습니다. |
| AMP, fused Adam, cuDNN autotuning | GPU 연산을 최적화했습니다. channels_last는 측정에서 느려 비활성화했습니다. |
| GPU 검증 지표 누적 및 step별 CPU 동기화 축소 | 매 step의 loss.item()과 지표 전송으로 생기는 대기를 줄였습니다. |

crop·flip·장면 순서는 seed/epoch/index로 생성해 재개 순서를 유지하도록 구성했습니다.
AMP와 autotuning으로 비트 단위 동일 결과를 보장하지는 않습니다.

## 측정 결과와 한계

Ryzen 5600X, RTX 3060 Ti 8GB, RAM 16GB, Samsung T7 Shield에서
PyTorch 2.7.1+cu128로 측정했습니다.

| 측정 | 결과 |
|---|---:|
| 이전 학습 로그의 epoch 시간 | 약 1,050초 |
| 최적화 후 전체 데이터 검증 epoch 1 / 2 | 114.75초 / 108.53초 |
| 새 본학습 epoch 1 / 2 | 117.92초 / 111.53초 |
| 최적화 검증의 peak allocated GPU memory | 약 1,453MiB |
| 큐브 요청 읽기량(전체 읽기 → 부분 읽기) | 약 87.26 → 71.57GiB/epoch |
| I/O 제외 CUDA 처리량(기존 batch 1 → grouped batch 4) | 13.3 → 150.0패치/초 |

전체 데이터와 패치 수를 유지한 상태에서 속도 향상을 확인했습니다. 다만 이전 로그와
새 실행은 완전히 통제된 동일 조건 비교가 아니므로 모든 차이를 코드 변경의 효과로
해석할 수는 없습니다. 요청 읽기량은 OS cache를 반영한 실제 디스크 전송량과 다르며,
최적화 후에도 데이터 읽기 대기가 남았습니다.

본학습 첫 두 epoch의 validation PSNR은 30.429 / 30.548dB로 bicubic 30.061dB보다
높았습니다. SAM은 모델 2.446 / 2.450°, bicubic 2.441°로 아직 개선되지 않았습니다.
최종 시험 평가, SSIM/ERGAS, HSI-only ablation과 여러 seed 비교가 필요합니다.
측정 상세는 [benchmark JSON](benchmarks/lib_rgb_hsi_optimization.json)에 기록했습니다.

## 초기 실행 확인 이미지

optimizer 2회 실행 후 저장한 best 가중치로 validation 128×128 패치를 시각화했습니다.
왼쪽부터 LR HSI, RGB guide, bicubic, prediction, HSI GT입니다.
HSI는 0-based bands 69/52/18과 공통 GT 기반 1–99% stretch로 표시했습니다.
최적화 본학습 완료 모델의 결과가 아닌 초기 실행 확인 자료입니다.

![LIB smoke 5-panel comparison](../experiments/results/lib_rgb_hsi_smoke/comparison.png)
