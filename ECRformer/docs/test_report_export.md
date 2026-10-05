# 체크포인트 평가와 비교 이미지 내보내기

[현재 연구](../README.md) · [결과 보고 기준](reporting.md)

`Official_ECRformer/export_test_report.py`는 학습을 실행하지 않습니다. 저장된 best checkpoint의 모델 설정을 사용해 전체 test를 FP32로 평가하고, 동일 데이터의 구름 입력 baseline과 비교합니다. 원래 SEN12MS-CR 로더·모델·metric 함수를 사용하며 모델 구조나 학습 설정을 수정하지 않습니다. 저장된 설정에 서버 로컬 config 클래스가 있으면 해당 모듈이 필요합니다.

## 처리

1. 출력 폴더는 새 이름이어야 합니다. 기존 결과를 덮어쓰지 않습니다.
2. 지표를 계산하기 전에 시드로 ROI별 5사례를 고정하고 목록을 기록합니다. ROI가 부족하면 부족한 조건을 그대로 기록합니다.
3. 저장된 고정 train subset과 test의 S1 파일 경로가 같은지 확인합니다. 공간·시간의 완전한 독립성을 검증하는 기능은 아닙니다.
4. best 모델과 구름 입력 baseline을 전체 test에 평가합니다. 지표는 clip하지 않은 원래 13밴드 값이며 LPIPS는 RGB 기반입니다.
5. 마지막 checkpoint는 같은 선정 5사례에만 평가합니다. 마지막 모델의 전체 test 평균으로 표시하지 않습니다.
6. SAR · cloudy · best · last · target 비교 PNG, 실제 기록 구간의 곡선, baseline 비교 그래프, 숫자/선정/해시 manifest를 저장합니다.

같은 학습의 best/last 비교이지 안정화 설정으로 새로 학습한 전후 비교가 아닙니다. 학습 FP16과 별개로 이 평가는 FP32, TTA 없음, 전체 256×256 패치를 사용합니다. RGB만 밴드 `(3,2,1)`·밝기 3.0으로 clip해 표시하고 SAR는 정규화된 VV의 고정 0~1 회색조입니다. 발산한 last가 포화된 그림으로 보일 수 있어 수치·출력 범위를 함께 확인합니다.

## 실행 예시

다른 사용자의 학습과 GPU 상태를 확인한 후 실행합니다. 아래는 기존 2026-10-04 학습의 예시이며 재실행할 때 출력 폴더를 바꿉니다.

```powershell
cd C:\CtrS\ECRformer\Official_ECRformer
C:\CtrS\.venv\Scripts\python.exe export_test_report.py --best experiments\ecrformer_spring_spring_subset6000\version_0\checkpoints\epoch=8-step=3375.ckpt --last experiments\ecrformer_spring_spring_subset6000\version_0\checkpoints\last.ckpt --diagnostics experiments\ecrformer_spring_spring_subset6000\version_0\training_diagnostics.jsonl --train-manifest experiments\ecrformer_spring_spring_subset6000\version_0\train_subset.json --output-dir results\spring_subset6000_test_new
```

경로가 달라지면 `--data-root`를 추가합니다. `--gpu -1`은 CPU 평가입니다. matplotlib과 기존 공식 구현 의존성이 필요합니다. 과거 로그에 없는 epoch를 생성하지 않으며, 커밋이 기록되지 않은 과거 학습은 training commit을 미확보로 둡니다. 평가 당시 Git HEAD와 실제 서버 source SHA-256은 별도 기록합니다.

## Git에 올릴 파일과 검증

`summary.json`, `metrics.csv`, `selection.json`, `selected_metrics.json`, `provenance.json`, `curves.json`, `report_manifest.json`, `learning.png`, `test_metrics.png`, `sample_01.png`~`sample_05.png`만 소규모 보고 자료로 올립니다. `model_stdout.log`, checkpoint, 원본 TIFF 및 전체 NPZ는 올리지 않습니다.

```powershell
python validate_test_report.py '<보고 자료 폴더>'
python tests/test_export_report.py
```

검증기는 manifest 해시, 전체 행 수·평균·선정 사례 대응을 확인합니다. 순수 helper 단위 테스트는 모델 추론·학습·GPU·실데이터를 사용하지 않습니다. 결과를 본 뒤 선정 사례를 바꾸지 않습니다.

Git에 보고 자료를 보관할 때 JSON/CSV를 `-text`로 지정하는 폴더별 `.gitattributes`를 사용해 줄바꿈 변환으로 artifact SHA-256이 바뀌지 않게 합니다.
