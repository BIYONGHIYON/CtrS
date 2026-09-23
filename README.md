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

SSA-MRN 팀 2명이 고해상도 PAN과 저해상도 MS로 고해상도 MS를 복원하는 원 연구를 재현합니다. 공식 코드를 버전 고정된 하위 모듈로 연결했고, 현재 데이터 로더·모델 실행·학습 루프 일부를 복구했습니다. 논문 지표 평가는 아직 진행 전입니다.

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
| [QuickBird 테스트 데이터](./SSA-MRN/data/raw/QuickBird/) | 공개 테스트 H5 20개 사례, 학습용 아님 |
| [QuickBird 실행 결과](./SSA-MRN/experiments/results/quickbird_smoke/README.md) | 무작위 초기화 모델의 배열·미리보기 |

32×32 실행은 첫 번째 QuickBird 샘플 전체(PAN·LMS 256×256, MS 64×64)를 각각 32×32와 8×8로 축소합니다. 256×256 실행은 같은 샘플을 원래 크기로 사용합니다. 두 결과는 서로 다른 부분을 잘라낸 타일이 아닙니다. 학습된 체크포인트가 없으므로 복원 품질이나 성능 지표로 해석하면 안 됩니다.

### QuickBird 첫 샘플 입력·출력 비교 (256×256 실행)

| 입력 MS: 64×64 (보기용 256×256 확대) | 입력 PAN: 256×256 | 출력 MS: 256×256 (학습 전) |
| --- | --- | --- |
| ![입력 MS RGB 합성](./SSA-MRN/experiments/results/quickbird_smoke/ms_input_rgb_256.png) | ![입력 PAN](./SSA-MRN/experiments/results/quickbird_smoke/pan_input_256.png) | ![무작위 초기화 출력 MS RGB 합성](./SSA-MRN/experiments/results/quickbird_smoke/random_output_preview_256.png) |

MS와 출력의 RGB 합성은 [QuickBird 제품 가이드](https://engineering.purdue.edu/~bethel/qbguide.pdf)의 Blue·Green·Red·NIR 순서를 H5에도 적용한다고 **가정**해 3·2·1번 밴드를 R·G·B로 표시했습니다. H5에는 밴드 이름이 없어 이 순서를 확정할 수 없습니다. 각 PNG는 보기용으로 대비를 별도 조정했고, 입력 MS 확대는 모델 입력 크기를 바꾸지 않습니다. **출력은 미학습 모델의 실행 확인용**입니다.

### 로컬에서 확인

```bash
git clone --recurse-submodules https://github.com/BIYONGHIYON/CtrS.git
cd CtrS/SSA-MRN
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch -r requirements.txt
python scripts/smoke_test.py --size 32
```

## 중간 비교 기준

- 원 논문 설정과 결과의 재현 정도
- 데이터셋 확보 및 전처리 가능성
- 정량 성능과 실패 사례
- 후속 개선 아이디어의 효과를 검증할 수 있는지
- 남은 기간의 학습 비용과 구현 난도

## 공통 원칙

- 논문에 기재된 설정과 재현 구현에서 바꾼 설정을 구분해 기록합니다.
- SSA-MRN QuickBird 테스트 H5 한 파일과 스모크 테스트 결과만 공유합니다. 대용량 학습 데이터와 모델 가중치는 저장소에 올리지 않습니다.
- 재현 완료는 실행 성공만으로 판단하지 않고 논문 결과와 비교한 뒤 기록합니다.

## 원격 저장소

- GitHub: https://github.com/BIYONGHIYON/CtrS
