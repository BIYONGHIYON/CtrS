# RGB별 12특징 · 재현 방식 Bilinear 실험

기존 triple12처럼 HSI 204밴드를 연속 17밴드씩 12그룹으로 압축하며,
R·G·B 각각 12특징을 처리합니다. K=4, 23탭 HSI baseline, 최종 204밴드 잔차 결합은 유지합니다.

| 처리 | 이전 triple12 | 이번 구성 |
|---|---|---|
| HSI baseline·압축 입력 LMS | 23탭 | 23탭 유지 |
| guide 축소 후 재확대 | 평균 풀링 → 23탭 | Bilinear → Bilinear |
| 코어 내부 분광 입력·특징 확대 | 23탭 | Bilinear |
| 코어 내부 특징·guide 축소 | 평균 풀링 | Bilinear |

Bilinear는 PAN–MS 재현의 실제 코드와 동일한 scale_factor와 `align_corners=True`를 사용합니다.
논문 전체 구현과 동일하다는 주장은 아닙니다. [34특징 Bilinear 비교 설정](./triple_rgb_34_bilinear.md)도 보존합니다.

- 모델 종류: `rgb_triple_grouped12_bilinear`.
- [설정](../configs/lib_rgb_hsi_triple12_k4_bilinear_tiles.json).
- 파라미터: 1,950,186개로 이전 triple12와 동일.
- 물리 배치 4 / 누적 1 / 검증 배치 2: 이전 triple12와 동일.
- 학습 quadrants / 검증·시험 tiles, 256패치, area x4, 정합·마스크·seed 42 유지.
- 기본 출력: `SSA-MRN/experiments/checkpoints/lib_rgb_hsi_triple12_k4_bilinear_tiles`.

같은 12특징 구성의 공간 연산 변경 효과를 비교하는 실험입니다.
이전 가중치와 텐서 모양은 같지만 모델 종류가 달라 재개·시험 프로토콜 검사에서 혼용을 거부합니다.
기존 가중치에서 이어서 학습하지 않고 새 가중치로 시작해야 합니다.

## 실행

새 코드가 설치된 서버 저장소 루트의 PowerShell에서:

```powershell
& C:\CtrS\.venv\Scripts\python.exe SSA-MRN\scripts\train_lib.py `
  --config SSA-MRN\configs\lib_rgb_hsi_triple12_k4_bilinear_tiles.json `
  --data-root 'C:\CtrS\SSA-MRN\data\dataset\RGB-HSI\LIB-HSI' `
  --smoke
```

VS Code 실행 구성: `LIB: triple12 bilinear tiles smoke / train (new run) / evaluate best`.
원본 데이터와 이전 결과·가중치는 변경하지 않습니다.

사용 확대·축소 8개 모듈과 전체 scalar 코어의 출력·기울기를 PAN–MS 재현 연산과 비교한 테스트가 있습니다.
기존 12·34특징 프로토콜의 테스트도 보존합니다.


## 실제 데이터 smoke와 원격 실행

RTX 3060 Ti / PyTorch 2.7.1+cu128 / AMP에서 실제 데이터 smoke를 통과했습니다.
학습 두 단계 + 부분 검증 약 13.45초, 최대 GPU 할당 3,883.38MiB / 최대 예약 5,344MiB였습니다.
제한된 smoke의 지표는 정식 시험 성능이 아닙니다.

서버 작업 폴더는 `C:\CtrS-triple12-bilinear`, 예약 작업은 `CtrS-Triple12-Bilinear-Control`입니다.
이전 실행기와 구분해 공용 Python·데이터를 사용하도록 별도 controller/config를 설치했습니다.
서버의 `scripts/remote-training.py`와 `triple12-bilinear-config.json`은 서버 로컬 파일이며 Git에 포함하지 않습니다.

SSH/VS Code Remote SSH로 접속한 PowerShell에서 상태와 로그를 확인합니다.

```powershell
$py = 'C:\CtrS\.venv\Scripts\python.exe'
$ctl = 'C:\CtrS-triple12-bilinear\scripts\remote-training.py'
& $py $ctl status
& $py $ctl logs --follow
```

클라이언트 로그 보기의 Ctrl+C는 학습을 중단하지 않습니다.
학습은 예약 작업 아래에서 실행하므로 SSH 종료·Mac 절전과 독립적입니다.
서버는 켜져 있어야 하고 `trainer` 계정은 로그인 상태를 유지해야 합니다. 화면 잠금은 가능합니다.


정식 실행은 `20261003-220209-d4faa37d0365`이며 2026-10-03 22:02:10에 새 가중치로 시작했습니다.
결과 폴더:

```text
C:\CtrS-triple12-bilinear\SSA-MRN\experiments\checkpoints\remote-runs\20261003-220209-d4faa37d0365
```

명시적으로 중단한 실행을 저장 지점에서 재개할 때는 해당 실행 폴더를 지정합니다.
최소 한 epoch가 완료된 `latest.pt`가 있어야 합니다. 현재 학습 중이면 아래 명령을 추가로 실행하지 마세요.

```powershell
& $py $ctl resume-latest --from-dir 'C:\CtrS-triple12-bilinear\SSA-MRN\experiments\checkpoints\remote-runs\20261003-220209-d4faa37d0365'
```

SSH 연결 종료 후 재접속하여 동일 실행 ID·PID 6004를 확인했고, epoch 1이 1/393 → 129/393 단계로 증가했습니다.
