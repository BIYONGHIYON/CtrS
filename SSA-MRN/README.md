# SSA-MRN 실행 안내

[연구 소개](../README.md) · [성능 개선 실험 표](experiments/performance/README.md) · [판정 기준](docs/judgement_criteria.md) · [작업 규칙](../AGENTS.md)

## 문서 찾기

| 찾는 것 | 위치 |
|---|---|
| 기준 모델과 대표 결과 | [연구 소개](../README.md) |
| 실험 목록과 결과 | [성능 개선](experiments/performance/README.md) · 종료: [SSAI 개선](experiments/ssai/README.md), [가변 배율 확장](experiments/variable_scale/README.md) |
| 다른 모델을 우리 환경에서 평가한 결과 | [다른 팬샤프닝 모델 평가](benchmarks/README.md) |
| 팬샤프닝·SSA-MRN·지표 설명 | [연구 설명](docs/research.md) |
| 기준 모델 이미지·수치 전체 | [C2MS-SP 결과 이미지](docs/c2ms_sp_result_images.md) · 이전 기준 C2MS: [이미지](docs/c2ms_result_images.md), [수치](docs/c2ms_evaluation_tables.md), [MATLAB 재평가](docs/c2ms_matlab_evaluation_tables.md) |
| 논문 보고값 조사 | [QB·GF2·WV3·WV2](docs/pansharpening_survey/QB_GF_WV2_WV3_팬샤프닝_성능조사_2026-10-27.md) · [QB](docs/pansharpening_survey/QB_팬샤프닝_성능조사_2026-10-27.md) |
| 보고서 양식 | [실험](docs/experiment_template.md) · [다른 모델 평가](docs/benchmark_template.md) |
| 결과 비교 사이트 | [CtrS Viewer 열기](https://biyonghiyon.github.io/CtrS/) · [실행·배포 안내](viewer/README.md) |
| 공식 코드와 원 논문 | [references](references/README.md) |

## 폴더 구조

```text
SSA-MRN/
├── experiments/
│   ├── catalog.json           전체 실험 목록 (실험 표의 원본)
│   └── <분야>/                performance · ssai · variable_scale
│       ├── README.md          분야 페이지와 실험 표
│       └── <번호>_<변경명>/
│           ├── README.md      보고서
│           ├── model.py · train.py · run.py · evaluate.py · prepare.py · config.json   (필요한 것만)
│           ├── checkpoints/   <후보·센서·시드>/best.pt
│           ├── results/       수치·학습 기록·그래프·이미지
│           ├── provenance/    학습 당시 소스·설정 (바꾸지 않음)
│           └── manifest.json  보관 파일의 경로·크기·SHA-256
├── benchmarks/<모델>/          다른 팬샤프닝 모델 평가
├── common/                    공유 도구 (데이터·모델·지표·평가·검증)
├── data/                      학습·평가 데이터 (Git 제외)
├── docs/                      판정 기준, 양식, 기준 모델 자료
├── references/                공식 코드(하위 모듈)와 원 논문
├── tests/
└── viewer/                    결과 비교 사이트
```

재개용 `latest.pt`와 초기 `initial.pt`는 Git이 아니라 실험별 GitHub Release에 둡니다. 실행 결과는 `runtime/`(Git 제외)에 만들고 정리한 결과만 실험 폴더에 넣습니다.

## 환경

저장소 루트에서 실행합니다.

```bash
git submodule update --init SSA-MRN/references/upstream
python -m pip install -r SSA-MRN/requirements.txt
```

데이터는 `SSA-MRN/data/`에 두거나 실행 옵션으로 경로를 지정합니다.

## 새 실험

1. 최신 main과 열린 PR에서 번호를 확인하고 `experiments/performance/<번호>_<변경명>/`을 만듭니다.
2. [양식](docs/experiment_template.md)으로 `README.md`를 쓰고 `experiments/catalog.json`에 **계획** 상태로 등록합니다.
3. 실험 표를 다시 만듭니다.

```bash
python SSA-MRN/common/index.py
```

4. 학습·평가는 실험 폴더의 `run.py --help`, `prepare.py --help`를 따릅니다. 실행 중인 checkout에 새 코드를 덮어쓰지 않습니다.
5. 결과가 나오면 catalog의 `status`·`improvement`·`result`를 고치고 3번을 다시 실행합니다.

catalog 항목의 필수 값은 `display_name`, `experimenter`, `status`, `improvement`, `result`(한 줄, 100자 이내)입니다.

<a id="evaluate"></a>

## 가중치 확인과 재평가

학습을 시작하지 않는 명령입니다.

```bash
# 데이터 없이 strict load와 CPU 추론 확인
python SSA-MRN/common/checkpoint.py --experiment 20 --checkpoint checkpoints/D4/best.pt

# H5 전체 RR 평가 (FR은 --protocol FR)
python SSA-MRN/common/checkpoint.py --experiment 20 --checkpoint checkpoints/D4/best.pt --h5 /path/to/test_qb_multiExm1.h5 --protocol RR --output SSA-MRN/runtime/D4_rr.json
```

`--device cuda`로 GPU에서 평가합니다. 공통 재평가는 FP32이며 저장된 결과를 덮어쓰지 않습니다.

## 결과 보관과 검증

```bash
# latest·initial 가중치를 Release에 올리고 manifest에 기록
python SSA-MRN/common/release_weights.py SSA-MRN/experiments/performance/<번호>_<변경명>

# 보관 파일 해시 확인 (Release 자산까지는 --releases)
python SSA-MRN/common/verify.py --experiment 81
python SSA-MRN/common/verify.py --research-track ssai --experiment all

# catalog와 실험 표가 맞는지 확인
python SSA-MRN/common/index.py --check
```

분야마다 번호가 따로 있으므로 번호가 겹치면 `--research-track`을 붙입니다. 해시 확인은 성능 재평가가 아닙니다.
