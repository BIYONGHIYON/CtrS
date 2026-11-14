# 모델 이름 (논문 표기)

`benchmarks/<모델>/README.md`에 작성합니다. 등록할 때 [평가 표](../benchmarks/README.md#models)와 [catalog](../benchmarks/catalog.json)를 함께 갱신합니다. 결과를 보관할 때 해당 폴더의 manifest.json을 갱신합니다.

**출처:** 저자, 제목, 학회·저널, 연도, DOI 또는 URL

**공식 코드:** 저장소 URL · 평가에 쓴 커밋 · 라이선스(재배포 가능 여부)

**평가자:** 실제 평가자를 적습니다.

**평가 설정:** `author` / `retrain` 중 평가한 설정, `retrain`의 학습 에폭과 출처(논문 / 공식 코드 / 미공개라 150에폭)

| 확인된 결과 | 상태 |
|---|---|
| 아직 평가하지 않았습니다. | 계획 |

[평가 표](../README.md#models) · [수치](#results) · [그래프](#graphs) · [이미지](#images) · [재현 자료](#evidence)

## 1. 목적과 상태

이 모델을 고른 이유(논문 보고 성능, 구조상 참고할 점)와 확인하려는 질문을 적습니다. 설정·센서·시드별 학습·평가 상태와 마지막 확인 시각을 구분합니다.

## 2. 모델과 비교 조건

모델 구조를 입력 → 주요 연산 → 출력 순서로 설명하고 구조도가 있으면 출처와 함께 표시합니다. 어댑터에서 바꾼 점(입력 순서, 패딩, 정규화, 출력 처리)과 공식 코드를 고친 부분(`provenance/upstream.diff`)을 적습니다.

설정별 조건을 C2MS-SP와 나란히 표로 적습니다.

| 항목 | C2MS-SP | 이 모델 (`retrain`) |
|---|---|---|
| 학습 데이터·분할 | PanCollection 학습·validation | |
| 정규화 | QB·WV3 2047, GF2 1023 | |
| 손실 | L1 + 거친 쌍 L1 + 0.46 분광 | |
| 에폭(출처)·배치·학습률 | 150 · 32 · 1e-4 → 1e-6 cosine | |
| 가중치 선택 | validation MSE 최소 | |
| 후처리 | 관측 일치 사영(K10) | |
| 시드 | 42·43·44 | |
| 파라미터 수 | | |

<a id="results"></a>

## 3. 정량 결과

RR test 20장 PSNR·SAM·ERGAS·SCC·Q2n의 시드 평균과 시드별 값을 적습니다. 논문 보고값, 우리 환경 재현값, 같은 시드 C2MS-SP를 한 표에 놓고 차이를 적습니다. FR QNR·Dλ·Ds와 WV2 전이는 참고로 따로 표시합니다. 숫자는 `results/summary.json`과 일치해야 합니다.

## 4. 논문 보고값·C2MS-SP와의 차이와 해석

재현값이 논문 보고값과 다르면 원인 후보(데이터 분할, 학습 예산, 평가 코드, 정규화)를 적습니다. 숫자를 보정하지 않습니다. C2MS-SP와의 차이가 어떤 장면·센서에서 나는지 설명하고, 우리 모델에 가져올 만한 점을 적습니다.

<a id="graphs"></a>

## 5. 학습·평가 그래프

우리 데이터로 학습했다면 실측 학습 손실·validation 곡선(`results/learning.png`)과 C2MS-SP와의 비교 그래프를 표시합니다. `author` 설정만 평가했다면 학습 곡선이 없다고 적습니다.

<a id="images"></a>

## 6. 결과 이미지

고정 6장면(QB RR test 0-based 0·1·3·7·14·17)의 두 형식을 `common/benchmark_evaluate.py --figures`로 만들어 표시합니다. 원본 크기 비교는 LR MS · PAN · C2MS-SP · 이 모델 · 정답과 두 오차 지도입니다. 확대 비교는 이 모델이 C2MS-SP보다 오차를 가장 많이 줄인 64×64 창이며, 이 모델에 유리하게 고른 창이라고 적습니다. 전체 차이는 장면 PSNR로 함께 보고합니다.

<a id="evidence"></a>

## 7. 재현 자료

best 가중치, Release의 latest·initial, 장면별 지표, 학습 기록, config, 공식 코드 커밋·수정분, 환경(torch·CUDA·GPU), manifest를 연결합니다. 저자 가중치를 재배포할 수 없으면 받은 주소와 SHA-256을 적습니다.

## 8. 정리와 다음 단계

판정(개선 여부, C2MS-SP 대비 우열)은 적지 않습니다. 같은 시드 C2MS-SP와의 RR 차이, 시드 수·시드별 결과, 학습 에폭과 출처를 요약합니다. 이 결과로 기준 모델을 바꾸지 않습니다. 가져올 아이디어가 있으면 성능 개선 연구의 새 실험으로 등록할 계획을 적습니다.

## 부록: config.json 예

```json
{
  "model": "adwm",
  "display_name": "ADWM",
  "evaluator": "평가자 이름",
  "paper": {"title": "논문 제목", "venue": "학회·저널 연도", "url": "https://..."},
  "code": {"url": "https://github.com/...", "commit": "고정 커밋 전체 해시", "license": "MIT", "redistribution": true},
  "author_weights": {"url": "https://...", "sha256": "...", "redistribution": false},
  "normalization": {"QB": 2047, "GF2": 1023, "WV3": 2047, "WV2": 2047},
  "settings": {
    "retrain": {"loss": "L1", "epochs": 500, "epochs_source": "논문 4.1절 (공개되지 않았으면 150, \"미공개, C2MS-SP와 같음\")", "batch": 32, "optimizer": "Adam", "lr": "5e-4, step 0.5 / 200", "selection": "validation 최소 손실"}
  },
  "sensors": ["QB", "GF2", "WV3"],
  "transfer": {"WV2": "WV3 가중치 적용, 참고"},
  "seeds": [42, 43, 44],
  "post_processing": "없음"
}
```

## 부록: model.py 어댑터 예

```python
"""ADWM adapter: upstream/ code, inputs and output in DN / scale."""
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent / 'upstream'))
from model import ADWM  # noqa: E402  (공식 코드, 수정하지 않음)


class Adapter(nn.Module):
    def __init__(self, bands):
        super().__init__(); self.net = ADWM(spectral_num=bands)

    def forward(self, pan, lms, ms):
        return self.net(pan, lms)  # 공식 코드의 입력 순서에 맞춤


def build(bands, sensor, setting, seed):
    return Adapter(bands)


def load(net, path):  # 저자 가중치처럼 저장 형식·키 이름이 다를 때만 정의
    state = torch.load(path, map_location='cpu', weights_only=False)
    net.net.load_state_dict(state.get('model', state), strict=True)
```
