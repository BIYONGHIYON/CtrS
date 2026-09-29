# SSA-MRN K=6 로컬 실험

## 목적과 변경 변수

기존 재현 가중치는 공개 `network.py`의 SSAI 차원 K=4로 학습했다. [논문](https://doi.org/10.1109/JSTARS.2025.3543827)은 K=6을 명시한다. 이번 실험은 SSAI 차원을 6으로 바꾸고, 센서별 데이터 분할, MSE, Adam, 배치 32, 초기 학습률 1e-4, 100 epochs, 시드 42를 유지했다. 시드 42는 재현 실험 값이며 논문에 공개된 값은 아니다. K=6은 합성곱의 채널 수를 바꾸므로 기존 K=4 가중치에서 이어 학습할 수 없고 처음부터 학습했다.

K=4는 RTX A6000에서, K=6은 아래 Radeon DirectML 환경에서 학습했다. 모델 차원 외에 실행 장치와 백엔드도 달라, 관측된 성능 차이를 **K 값만의 인과 효과**로 해석할 수 없다. 같은 환경에서 두 설정을 다시 학습·평가해야 통제된 비교가 된다. 여기서는 확보된 결과를 비교 기준으로 기록한다.

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

전체 학습은 센서별로 순차 실행한다. 새 실행의 체크포인트는 `experiments/checkpoints/local/k6/{qb_full,gf2_full,wv3_full}/latest.pt`, 로그는 `experiments/logs/local/k6/`에 남아 저장소에 포함된 가중치를 덮어쓰지 않는다. 공유된 epoch 100 가중치는 `experiments/checkpoints/k6/`에 있다.

```powershell
.\scripts\run_k6_local.ps1 -DataRoot 'C:\Users\erick\대학교\Ctrs\SSA-MRN'
```

중단 뒤에는 같은 명령에 `-Resume`을 추가한다. 마지막 완료 epoch부터 이어서 실행한다. 체크포인트에는 K, 학습률, 배치 크기, 시드, 난수 상태가 기록된다. 중간에 학습률이나 배치 크기를 바꾸려면 별도 실험 폴더와 새 학습을 사용한다.

320개 QB 학습 쌍과 32개 검증 쌍으로 측정한 10배치 실행은 약 12초였다. 전체 3개 센서 100 epochs는 이 노트북에서 **수십 시간 이상** 걸릴 수 있다. 이는 짧은 시험에서 외삽한 추정치이며 실제 전체 데이터 읽기, 발열 제한, 검증 시간에 따라 달라진다.

## 비교 평가

학습 완료 후 같은 테스트 H5와 지표 계산으로 기존 K=4 및 K=6을 비교했다. 시험 데이터에서 하이퍼파라미터를 고르지 않았다. 논문 수치와 비교할 때 PSNR peak·집계, SCC 경계, Q2n MATLAB 일치성 및 FR PAN 축소 차이 때문에 지표 자체의 불확실성을 함께 기록한다.

```powershell
.\.venv-dml\Scripts\python.exe scripts\evaluate_paper.py `
  --sensor all --protocol both --device cpu `
  --data-root 'C:\Users\erick\대학교\Ctrs\SSA-MRN' `
  --checkpoint-root experiments\checkpoints\k6 `
  --output-dir experiments\results\local_k6_eval
```

평가 JSON에는 체크포인트 경로와 K가 기록된다. 모든 모델의 RR·FR 결과를 확인한 뒤, SAM·ERGAS·PSNR·SCC·Q2ⁿ 및 QNR·Dλ·Ds를 각각 비교한다. 하나의 지표만으로 전체 성능 향상을 판단하지 않는다.

## K=4 vs K=6 최종 비교 결과

논문에서는 SSAI의 feature dimension인 K를 6으로 설정한다. 공개 코드의 기존 재현 설정은 K=4였으므로, 다른 주요 학습 조건은 유지한 채 K=6으로 변경하여 QB, GF2, WV3를 각각 100 epochs 재학습하였다.

평가는 각 센서의 Reduced Resolution(RR) 및 Full Resolution(FR) 데이터에서 각각 20 samples를 사용하였다.

### QuickBird (QB)

| Metric | 논문 | K=4 | K=6 |
|---|---:|---:|---:|
| SAM ↓ | 4.8478 | 4.9557 | 4.9350 |
| ERGAS ↓ | 4.0726 | 4.1555 | 4.0875 |
| PSNR ↑ | 37.6645 | 37.3608 | 37.5040 |
| SCC ↑ | 0.9702 | 0.9765 | 0.9774 |
| Q4 ↑ | 0.9243 | 0.9214 | 0.9256 |
| Dλ ↓ | 0.0341 | 0.0479 | 0.0484 |
| Ds ↓ | 0.0360 | 0.0396 | 0.0470 |
| QNR ↑ | 0.9311 | 0.9147 | 0.9072 |

QB에서는 K=6 적용 후 RR 지표가 전반적으로 개선되었으나, FR 지표는 K=4보다 일부 악화되었다.

### Gaofen2 (GF2)

| Metric | 논문 | K=4 | K=6 |
|---|---:|---:|---:|
| SAM ↓ | 0.9434 | 0.9668 | 1.0035 |
| ERGAS ↓ | 0.8755 | 0.9125 | 0.9693 |
| PSNR ↑ | 46.9734 | 46.3520 | 45.8631 |
| SCC ↑ | 0.9898 | 0.9816 | 0.9793 |
| Q4 ↑ | 0.9688 | 0.9661 | 0.9632 |
| Dλ ↓ | 0.0395 | 0.0350 | 0.0355 |
| Ds ↓ | 0.0487 | 0.0565 | 0.0502 |
| QNR ↑ | 0.9137 | 0.9106 | 0.9161 |

GF2에서는 K=6 적용 후 RR 지표는 K=4보다 낮아졌지만, FR의 Ds와 QNR은 개선되는 혼합된 결과가 나타났다.

### WorldView3 (WV3)

| Metric | 논문 | K=4 | K=6 |
|---|---:|---:|---:|
| SAM ↓ | 3.4873 | 3.5277 | 3.4504 |
| ERGAS ↓ | 2.5866 | 2.5822 | 2.5329 |
| PSNR ↑ | 37.5691 | 37.3894 | 37.6326 |
| SCC ↑ | 0.9735 | 0.9779 | 0.9789 |
| Q8 ↑ | 0.8930 | 0.8753 | 0.8829 |
| Dλ ↓ | 0.0329 | 0.0221 | 0.0155 |
| Ds ↓ | 0.0617 | 0.0382 | 0.0330 |
| QNR ↑ | 0.9077 | 0.9408 | 0.9522 |

WV3에서는 K=6 적용 후 K=4 대비 RR 및 FR 지표가 전반적으로 개선되었다.

### 결론

논문에서 사용한 K=6 설정으로 변경했을 때 센서별로 서로 다른 변화가 나타났다.

- QB: RR 개선, FR 일부 악화
- GF2: RR 악화, FR 일부 개선
- WV3: RR 및 FR 전반적 개선

공개 코드의 K=4 설정은 논문 재현 결과 차이의 후보 원인이다. 다만 이번 비교는 실행 환경도 달라 K의 영향을 분리하지 못하며, K를 6으로 변경하는 것만으로 논문과 공개 코드 사이의 결과 차이를 모두 설명할 수도 없다.

논문에서 기술된 SSAI attention 연산과 공개 코드 구현 사이의 차이는 수정하지 않았다. K=4·K=6의 실행 환경도 달라 같은 환경에서 재비교하고 해당 연산 차이를 추가로 검증해야 한다.

K=6의 상세 평가 결과는 `experiments/results/paper_comparison_k6/`의 JSON 파일에 저장하였다.
