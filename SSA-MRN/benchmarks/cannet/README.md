# CANNet

**출처:** Yule Duan, Xiao Wu, Haoyu Deng, Liang-Jian Deng, Content-Adaptive Non-Local Convolution for Remote Sensing Pansharpening, CVPR, 2024, [arXiv 2404.07543](https://arxiv.org/abs/2404.07543)

**공식 코드:** [Duanyll/CANConv](https://github.com/Duanyll/CANConv) · `0b7cc6f` (kmcuda 하위 모듈 `6906cc3`) · GPL-3.0(코드·저자 가중치 재배포 가능)

**평가자:** 이병현

**평가 설정:** `author`(저자 가중치 QB·GF2·WV3, WV2는 WV3 가중치 전이) / `retrain`(QB, 500에폭, 공식 코드 `default.json`, 단일 시드 42)

| 확인된 결과 | 상태 |
|---|---|
| `author` QB·GF2·WV3·WV2 평가를 Kaggle에서 마쳤습니다(결과 파일은 세션 종료 후 회수). `retrain` QB 시드 42를 학습 중입니다(31에폭). | 학습 중<br>(11-16 17:55) |

[평가 표](../README.md#models) · [수치](#results) · [그래프](#graphs) · [이미지](#images) · [재현 자료](#evidence)

## 1. 목적과 상태

CANNet은 특징 지도의 화소를 k-means로 묶고, 묶음마다 다른 합성곱 커널을 만들어 적용하는 내용 적응 비국소 합성곱(CANConv)을 U자 구조에 넣은 모델입니다. 논문 비교 표의 QB RR PSNR은 38.488dB로, 우리 기준 모델 C2MS-SP와 같은 조건에서 비교하려고 골랐습니다. 저자 가중치가 QB·GF2·WV3 모두 공개되어 있어 `author` 평가를 학습 없이 할 수 있습니다.

| 설정 | 센서 | 시드 | 상태 (2026-11-16 17:55 KST) |
|---|---|---|---|
| `author` | QB·GF2·WV3, WV2(전이) | 없음 | Kaggle 평가 완료, 결과 파일 회수 대기 |
| `retrain` | QB | 42 | 학습 중(Kaggle 버전 1, 14:39 시작, 17:55 기준 31에폭) |

## 2. 모델과 비교 조건

입력 PAN과 4배 확대 MS(LMS)를 이어 붙여 3×3 합성곱으로 32채널 특징을 만듭니다. CAN 잔차 블록 → 내림 → CAN 잔차 블록 → 내림 → CAN 잔차 블록 → 올림 → CAN 잔차 블록 → 올림 → CAN 잔차 블록을 거친 뒤 3×3 합성곱으로 잔차를 만들어 LMS에 더합니다. CAN 잔차 블록의 합성곱은 화소 특징을 32개 묶음으로 k-means 군집화하고 묶음별 커널로 계산합니다. 같은 해상도의 내림·올림 쪽 블록은 군집 결과를 공유합니다. 학습 중에는 학습 패치별 군집 결과를 캐시해 몇 에폭마다만 다시 계산합니다.

어댑터(`model.py`)는 공식 `CANNet`을 수정 없이 감싸며 `forward(pan, lms, ms)`에서 `ms`를 쓰지 않습니다. 입력과 출력은 공식 코드와 같이 DN을 정규화 값으로 나눈 값입니다. 공식 코드는 고치지 않았습니다(`provenance/upstream.diff` 없음).

| 항목 | C2MS-SP | CANNet (`retrain`) |
|---|---|---|
| 학습 데이터·분할 | PanCollection 학습·validation | 같음 |
| 정규화 | QB·WV3 2047, GF2 1023 | 같음 |
| 손실 | L1 + 거친 쌍 L1 + 0.46 분광 | L1 |
| 에폭(출처)·배치·학습률 | 150 · 32 · 1e-4 → 1e-6 cosine | 500(공식 코드 `default.json`) · 32 · QB 5e-4, StepLR 500(500에폭 안에서는 감소 없음, `presets.json`) |
| 가중치 선택 | validation MSE 최소 | validation L1 최소(10에폭마다) |
| 후처리 | 관측 일치 사영(K10) | 없음 |
| 시드 | 42·43·44 | 42 (단일 시드) |
| 학습 환경 | 공용 서버 | Kaggle GPU T4 x2(공식 코드는 GPU 하나로만 학습) |
| 파라미터 수 | | 평가 후 기록 |

공식 trainer는 고정 에폭(25에폭마다·처음 20에폭)만 저장하고 가중치를 고르지 않습니다. 우리 규칙(validation 선택)을 위해 `train.py`가 공식 설정의 validation 간격(10에폭)마다 validation L1을 재고 가장 낮은 가중치를 `best.pt`로 저장합니다. Kaggle 세션은 12시간 제한이 있어 `latest.pt`로 이어서 학습합니다. 학습 패치별 k-means 캐시는 크기 때문에 저장하지 않아, 재개 직후 1에폭은 캐시를 새로 만들고 다음 에폭부터 공식 정책을 따릅니다. 이 차이는 재개 횟수와 함께 결과에 적습니다.

**시드 수:** 다른 모델 평가 규칙은 3시드(42·43·44)이지만, 2026-11-16 사용자 결정으로 단일 시드 42만 학습합니다. Kaggle에서 시드 하나에 GPU 약 37시간(캐시 재사용 에폭 약 3.9분, 캐시 재계산·validation 에폭 약 11분)이 걸려 3시드는 약 3주가 필요했기 때문입니다. Kaggle 버전 1에서 함께 시작한 시드 43은 이어서 학습하지 않고 결과로 쓰지 않습니다. C2MS-SP와는 같은 시드 42끼리 비교합니다. 3시드가 아니므로 루트 README 비교 표에는 넣지 않습니다.

<a id="results"></a>

## 3. 정량 결과

아직 평가하지 않았습니다.

논문 보고값(QB RR): PSNR 38.488, SAM 4.496, ERGAS 3.698, Q4 0.937([S2WMamba](https://arxiv.org/html/2512.06330v1) Table 3). CANNet 논문 Table 2는 SAM 4.507, ERGAS 3.652, Q4 0.937이며 PSNR을 보고하지 않았습니다.

## 4. 논문 보고값·C2MS-SP와의 차이와 해석

평가 후 작성합니다.

<a id="graphs"></a>

## 5. 학습·평가 그래프

학습이 끝나면 `results/learning.png`를 표시합니다.

<a id="images"></a>

## 6. 결과 이미지

평가 후 고정 6장면을 표시합니다.

<a id="evidence"></a>

## 7. 재현 자료

- 공식 코드: `upstream/`(하위 모듈, 고정 커밋). 네이티브 연산(`pwacnative.so`, `libKMCUDA.so`)은 `upstream/build.sh`로 GPU 구조에 맞춰 빌드합니다. Kaggle T4에서는 `CUDA_ARCH=75`로 빌드했습니다.
- 저자 가중치: `upstream/weights/cannet_{qb,gf2,wv3}.pth`, SHA-256은 [config.json](config.json)에 있습니다.
- Kaggle 실행 노트북: [provenance/kaggle_cannet.py](provenance/kaggle_cannet.py)(percent 형식, `# %%`가 한 셀, Kaggle에서 File → Import Notebook으로 불러옵니다). 데이터는 비공개 Kaggle 데이터셋(PanCollection 원본 h5)을 씁니다. 평가는 k-means 라이브러리(kmcuda)가 여러 GPU 사이 peer access를 요구하므로 `CUDA_VISIBLE_DEVICES=0`으로 GPU 하나만 보이게 실행합니다.
- 학습 환경: Kaggle GPU T4 x2, torch 2.11.0+cu130, CUDA 13.0, Python 3.13.
- 보관 파일 해시: [manifest.json](manifest.json), `python SSA-MRN/common/verify.py --benchmark cannet`으로 확인합니다. 결과·best.pt는 평가 후 추가합니다.

## 8. 정리와 다음 단계

평가 후 작성합니다. 판정은 적지 않고 같은 시드 C2MS-SP와의 RR 차이만 숫자로 보고합니다.
