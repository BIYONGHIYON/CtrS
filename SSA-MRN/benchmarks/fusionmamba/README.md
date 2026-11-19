# FusionMamba

| 평가자 | 설정 | 상태 |
|:---:|:---:|:---:|
| 김윤지 | 저자 WV3 가중치 | WV3 일부 평가 · MATLAB 대기 |

**출처:** Peng 외, FusionMamba, TGRS 2024 · [공식 코드](https://github.com/PSRben/FusionMamba). 저자 코드와 가중치의 재배포 조건은 확인하지 않아 포함하지 않았습니다.

**결과:** WV3 test RR·FR 각 20장의 추론과 Python 지표 계산을 마쳤습니다. WV3 RR 예측은 Release에 보관했고 실제 MATLAB 채점은 진행 중입니다. 다른 센서 학습 및 평가는 미완료입니다. 논문값 재현이나 전체 모델 비교 완료로 해석하지 않습니다.

[평가 표](../README.md#models) · [결과](#results) · [해석](#verdict) · [방법](#method) · [그림](#figures) · [재현 자료](#evidence)

<a id="results"></a>

## 결과

WV3 RR test 20장 평균입니다. FusionMamba의 PSNR·SAM·ERGAS는 Python으로 계산했으며 MATLAB 채점값이 아닙니다. 기존 모델의 숫자는 해당 보고서의 기존 결과입니다. 각 결과의 데이터·전처리·시드·평가 방법이 다를 수 있으므로 표의 차이를 공식 순위나 같은 시드 비교로 해석하지 않습니다.

| 모델 | 설정 | PSNR (dB) | SAM (도) | ERGAS | 채점 |
|---|---|---:|---:|---:|---|
| FusionMamba | WV3 저자 가중치 1개 | 39.344 | 2.816 | 2.079 | Python, MATLAB 대기 |
| C2MS-SP | 3시드 평균 | 38.673 | 2.984 | 2.241 | 기존 Python |
| U2Net | WV3 저자 가중치 1개 | 38.844 | 2.897 | 2.200 | 기존 Python |
| ARConv | WV3 저자 가중치 1개 | 39.167 | 2.858 | 2.118 | 기존 MATLAB |
| CANNet | WV3 저자 가중치 1개 | 38.978 | 2.925 | 2.160 | 기존 MATLAB |
| FusionNet | WV3 저자 가중치 1개 | 32.519 | 4.930 | 4.689 | 기존 MATLAB |

기존 수치는 [기준 모델](../../docs/images/c2ms_sp/baseline_metrics.json), [U2Net](../u2net/README.md), [ARConv](../arconv/README.md), [CANNet](../cannet/README.md), [FusionNet](../fusionnet/README.md)의 보고값입니다. FusionMamba 입력 RR 파일의 해시는 과거 성능 54번 보존 manifest의 RR 파일과 일치합니다. 현재 C2MS-SP 실행 당시의 전체 입력 파일 해시와는 아직 대조하지 않았습니다. FusionMamba는 원본 test.py의 출력 clipping을 사용하고 C2MS-SP는 관측 일치 사영을 사용합니다. 저자 가중치의 학습 시드·학습 데이터 분할은 확인되지 않았습니다.

FR Python 참고값은 QNR 0.9656, Dλ 0.0166, Ds 0.0182입니다. PAN 축소는 MATLAB 근사 구현이며 FR에는 정답 영상이 없습니다. [RR 장면별 수치](results/rr_metrics.json) · [FR 장면별 수치](results/fr_metrics_approx.json).

<a id="verdict"></a>

## 해석과 정리

WV3에서만 사전 측정한 수치이며 다른 센서의 성능이나 공식 우열을 확정하지 않습니다. 동일 예측의 실제 MATLAB 채점, 데이터·후처리 확인 및 비교군별 조건 대조가 필요합니다.

<a id="method"></a>

## 방법

공식 WV3 가중치를 strict load하고, 저자 test.py로 PAN과 MS를 DN/2047로 입력해 256 타일 단위로 추론하고 출력을 [0,1]로 자른 뒤 2047을 곱했습니다. RR은 밴드별 PSNR을 평균하고 SAM은 화소별 분광각의 도 단위 평균, ERGAS는 배율 4로 계산했습니다. Kaggle Tesla T4, Python 3.13, PyTorch 2.11, CUDA 13에서 평가했습니다. Vim 버전의 CUDA 아키텍처 및 인터페이스 호환 수정을 적용했으며 핵심 scan 및 causal convolution을 각각 기준 연산과 대조했습니다. 원본 README의 400에폭 설명과 파일명 420의 차이는 확인되지 않았습니다.

<a id="figures"></a>

## 그림

고정 6장면 비교 이미지는 아직 만들지 않았습니다.

<a id="evidence"></a>

## 재현 자료

[설정과 해시](config.json). WV3 RR 예측 20장은 [GitHub Release](https://github.com/BIYONGHIYON/CtrS/releases/tag/predictions-benchmarks-fusionmamba-author)에 보관했으며 파일 SHA-256은 `13a7d284b52fd5c577b57288c7d95745d8d9795c826cac8bf133fc724874055b`입니다. [실제 MATLAB 채점](https://github.com/BIYONGHIYON/CtrS/actions/runs/44560761200)은 진행 중입니다. FR 예측은 로컬 압축 파일에 보관했습니다. QB·GF2 직접 학습, WV2 전이, 전체 MATLAB 평가 및 Release 검증은 미완료입니다.
