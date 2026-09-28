<p align="center">
  <img src="./icon/CtrS-icon-circle.png" alt="CtrS" width="180" />
</p>

# CtrS

동국대학교 종합설계 연구 저장소입니다. 팀원 4명이 2명씩 두 팀으로 나뉘어 **SSA-MRN과 ECRformer 원 논문 재현을 동시에 진행**합니다.

## 팀원

| 이름 | GitHub | 전공 | 역할 |
| --- | --- | --- | --- |
| 이병현 | [BIYONGHIYON](https://github.com/BIYONGHIYON) | 멀티미디어공학과 | 팀장 |
| 김경찬 | [erickks2y-jpg](https://github.com/erickks2y-jpg) | 멀티미디어공학과 | 팀원 |
| 송준현 | [choco-ssalbbang](https://github.com/choco-ssalbbang) | 멀티미디어공학과 | 팀원 |
| 김윤지 | [kimrose1015-max](https://github.com/kimrose1015-max) | 데이터사이언스전공 | 팀원 |

## 팀 운영 방식

1. SSA-MRN 담당 2명과 ECRformer 담당 2명이 각 원 논문의 데이터, 모델, 학습 및 평가 절차를 재현합니다.
2. 각 팀은 실행 환경, 공개 코드에서 빠진 부분, 논문 설정과 재현 결과의 차이를 기록합니다.
3. 중간 시점에 재현 결과와 후속 연구 가능성을 같은 기준으로 비교해 최종 주제 하나를 선택합니다.
4. 주제가 결정되면 팀원 4명이 해당 연구의 모델 개선, 추가 실험, 시연 및 보고서를 함께 진행합니다.

## 병렬 재현 연구

### [SSA-MRN 원 논문 재현](./SSA-MRN/README.md)

SSA-MRN 팀 2명이 고해상도 PAN과 저해상도 MS로 고해상도 MS를 복원하는 원 연구를 재현합니다. 공식 코드를 버전 고정된 하위 모듈로 연결하고 데이터 로더·학습·체크포인트·단일 샘플 테스트를 복구했습니다. **QB·GF2·WV3 100-epoch 학습과 각 센서의 첫 테스트 샘플 추론을 완료**했으며, 논문 지표를 이용한 전체 데이터 평가·논문 수치 비교는 다음 단계입니다.

저장소를 처음 내려받았거나 하위 모듈 파일이 비어 있다면 다음 명령으로 공식 코드를 가져옵니다.

```bash
git submodule update --init --recursive
```

### [ECRformer 원 논문 재현](./ECRformer/README.md)

ECRformer 팀 2명이 광학 영상과 SAR 영상을 이용한 구름 제거 모델의 원 논문 결과를 재현합니다. 데이터 준비, 공개 코드 실행, 학습·평가 조건 확인과 기준 성능 확보를 우선 진행합니다. Spectral-Semantic Decoupled Learning 확장은 재현 결과를 확인한 뒤 검토합니다.

## 폴더와 현재 결과

| 위치 | 내용 |
| --- | --- |
| [SSA-MRN](./SSA-MRN/README.md) | 팬샤프닝 재현 코드·설정·문서 |
| [ECRformer](./ECRformer/README.md) | 구름 제거 재현 계획 |
| [SSA-MRN 가중치](./SSA-MRN/experiments/checkpoints/) | QB·GF2·WV3의 epoch 100 체크포인트 |
| [SSA-MRN 학습 로그](./SSA-MRN/experiments/logs/) | 센서별 epoch 1–100 학습·검증 MSE |
| [SSA-MRN 단일 샘플 결과](./SSA-MRN/README.md#3개-센서-학습단일-샘플-테스트-현황-2026-09-28) | 각 센서의 입력·출력·정답 비교, 수치, 재실행 방법 |
| [QuickBird 스모크 결과](./SSA-MRN/experiments/results/quickbird_smoke/README.md) | 무작위 초기화 모델의 배열·미리보기 (학습 결과 아님) |

### SSA-MRN 학습 모델의 첫 테스트 샘플 결과

| 센서 | LMS 기준 PSNR | 학습 모델 PSNR | 비교 이미지 |
|---|---:|---:|---|
| QuickBird | 35.45 dB | 41.35 dB | ![QuickBird 비교](./SSA-MRN/experiments/results/quickbird_trained/comparison.png) |
| Gaofen 2 | 31.26 dB | 38.65 dB | ![Gaofen 2 비교](./SSA-MRN/experiments/results/gaofen2_trained/comparison.png) |
| WorldView 3 | 29.06 dB | 37.66 dB | ![WorldView 3 비교](./SSA-MRN/experiments/results/worldview3_trained/comparison.png) |

각 그림은 입력 MS·PAN·LMS·학습 모델 출력·정답 순서입니다. 테스트 H5의 **20개 중 첫 1개**만 사용했으며 입력 영상을 자르지 않았습니다. 이 수치는 실행 확인용 MSE 기반 PSNR(정규화 범위의 peak=1)이며 **논문 전체 성능과 비교한 결과가 아닙니다**. 시각화의 RGB 밴드 순서는 가정입니다. 자세한 조건과 한계는 [SSA-MRN README](./SSA-MRN/README.md)에 기록했습니다.

### 로컬에서 확인

```bash
git clone --recurse-submodules https://github.com/BIYONGHIYON/CtrS.git
cd CtrS/SSA-MRN
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch -r requirements.txt
python scripts/test_checkpoint.py --sensor QB --checkpoint experiments/checkpoints/qb_full/latest.pt --data data/raw/QuickBird/test_qb_multiExm1.h5 --output-dir experiments/results/local_qb
```

테스트 H5는 Git에 넣지 않았습니다. 실행 전에 [PanCollection](https://github.com/liangjiandeng/PanCollection)의 센서별 ReducedData H5를 내려받아 [정해진 위치](./SSA-MRN/README.md#다른-사람이-테스트-재실행하기)에 놓아야 합니다. 체크포인트·로그·미리보기 결과는 저장소에 포함되어 있습니다.

## 중간 비교 기준

- 원 논문 설정과 결과의 재현 정도
- 데이터셋 확보 및 전처리 가능성
- 정량 성능과 실패 사례
- 후속 개선 아이디어의 효과를 검증할 수 있는지
- 남은 기간의 학습 비용과 구현 난도

## 공통 원칙

- 논문에 기재된 설정과 재현 구현에서 바꾼 설정을 구분해 기록합니다.
- SSA-MRN의 원본 학습·검증·테스트 H5는 Git에서 제외합니다. 재현 코드, 세 센서의 가중치·로그·테스트 결과는 공유합니다.
- 재현 완료는 실행 성공만으로 판단하지 않고 논문 결과와 비교한 뒤 기록합니다.

## 원격 저장소

- GitHub: https://github.com/BIYONGHIYON/CtrS
