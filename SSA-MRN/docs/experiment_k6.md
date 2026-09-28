# SSA-MRN K=6 로컬 실험

## 목적과 변경 변수

기존 재현 가중치는 공개 `network.py`의 SSAI 차원 K=4로 학습했다. [논문](https://doi.org/10.1109/JSTARS.2025.3543827)은 K=6을 명시한다. 이번 실험은 **K만 4에서 6으로 바꾸고**, 센서별 데이터 분할, MSE, Adam, 배치 32, 초기 학습률 1e-4, 100 epochs, 시드 42를 유지한다. 시드 42는 재현 실험 값이며 논문에 공개된 값은 아니다. K=6은 합성곱의 채널 수를 바꾸므로 기존 K=4 가중치에서 이어 학습할 수 없고 처음부터 학습해야 한다.

이 저장소는 공식 모델 코드를 하위 모듈로 고정해 둔다. K=6 변형은 재현용 래퍼에서 사용되는 SSAI 계층과 융합 계층에만 적용한다. K=4 경로와 기존 체크포인트는 계속 읽을 수 있다. 논문 수식의 어텐션 행렬곱과 공개 코드의 원소별 곱 차이는 이번 실험에서 변경하지 않는다. 따라서 K=6 결과도 논문 구현과 완전히 같다고 주장할 수 없다.

## Windows 노트북 환경

Radeon 780M은 NVIDIA CUDA 장치가 아니다. 이 컴퓨터에서 `torch-directml` 0.2.5.dev240914 / PyTorch 2.4.1과 DirectML 어댑터 `AMD Radeon 780M Graphics`를 확인했다. K=6, 64×64 패치, 배치 32로 실제 QuickBird H5의 학습·검증과 체크포인트 재개를 짧게 시험했다. DirectML의 기본 PReLU 역전파가 배치 4 이상에서 오류를 내어 같은 PReLU 수식으로 계산하는 호환 경로를 `--device dml`에만 적용했다. DirectML Adam의 일부 연산은 CPU로 대체된다는 경고가 출력될 수 있다.

새 가상환경은 저장소의 `SSA-MRN` 폴더에서 만든다. Python 3.12와 `uv`가 있다면:

```powershell
uv venv .venv-dml --python 3.12
uv pip install --python .venv-dml\Scripts\python.exe torch-directml==0.2.5.dev240914
uv pip install --python .venv-dml\Scripts\python.exe -r requirements.txt
.\.venv-dml\Scripts\python.exe -c "import torch_directml; print(torch_directml.device_name(0))"
```

이 PC의 원본 H5는 `C:\Users\erick\대학교\Ctrs\SSA-MRN`에 있다. 학습용 H5 확인 결과는 QB 17,139/1,905, GF2 19,809/2,201, WV3 9,714/1,080개(학습/검증)이며 각 센서 합계가 논문의 19,044/22,010/10,794와 일치한다. 모든 학습 입력 PAN/GT는 64×64, MS는 16×16이다. 원본 H5는 읽기만 하며 Git에 올리지 않는다.

## 짧은 실행과 전체 학습

먼저 QB 32쌍과 검증 32쌍으로 장치와 저장을 확인한다. 이 결과는 성능 비교에 사용하지 않는다.

```powershell
.\scripts\run_k6_local.ps1 -DataRoot 'C:\Users\erick\대학교\Ctrs\SSA-MRN' -Sensors QB -Smoke
```

전체 학습은 센서별로 순차 실행한다. 새 체크포인트는 `experiments/checkpoints/k6/{qb_full,gf2_full,wv3_full}/latest.pt`, 로그는 `experiments/logs/k6/`에 남는다. 기존 K=4 결과를 덮어쓰지 않는다.

```powershell
.\scripts\run_k6_local.ps1 -DataRoot 'C:\Users\erick\대학교\Ctrs\SSA-MRN'
```

중단 뒤에는 같은 명령에 `-Resume`을 추가한다. 마지막 완료 epoch부터 이어서 실행한다. 체크포인트에는 K, 학습률, 배치 크기, 시드, 난수 상태가 기록된다. 중간에 학습률이나 배치 크기를 바꾸려면 별도 실험 폴더와 새 학습을 사용한다.

320개 QB 학습 쌍과 32개 검증 쌍으로 측정한 10배치 실행은 약 12초였다. 전체 3개 센서 100 epochs는 이 노트북에서 **수십 시간 이상** 걸릴 수 있다. 이는 짧은 시험에서 외삽한 추정치이며 실제 전체 데이터 읽기, 발열 제한, 검증 시간에 따라 달라진다.

## 비교 평가

학습이 완료되면 같은 테스트 H5와 지표 계산으로 기존 K=4 및 K=6을 비교한다. 시험 데이터에서 하이퍼파라미터를 고르지 않는다. 논문 수치와 비교할 때 PSNR peak·집계, SCC 경계, Q2n MATLAB 일치성 및 FR PAN 축소 차이 때문에 지표 자체의 불확실성을 함께 기록한다.

```powershell
.\.venv-dml\Scripts\python.exe scripts\evaluate_paper.py `
  --sensor all --protocol both --device cpu `
  --data-root 'C:\Users\erick\대학교\Ctrs\SSA-MRN' `
  --checkpoint-root experiments\checkpoints\k6 `
  --output-dir experiments\results\paper_comparison_k6
```

평가 JSON에는 체크포인트 경로와 K가 기록된다. 모든 모델의 RR·FR 결과를 확인한 뒤, SAM·ERGAS·PSNR·SCC·Q2ⁿ 및 QNR·Dλ·Ds를 각각 비교한다. 하나의 지표만으로 전체 성능 향상을 판단하지 않는다.
