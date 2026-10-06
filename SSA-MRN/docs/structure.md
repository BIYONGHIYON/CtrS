# 문서와 파일 위치

[최신 결과](../README.md) · [연구 설명](research.md) · [재현 연구](reproduction.md) · [이전 실험](previous_experiments.md)

```text
SSA-MRN/
├── README.md                 최신 결과 요약과 대표 이미지
├── docs/
│   ├── research.md           현재 알고리즘과 평가 설명
│   ├── reproduction.md       PAN–MS 재현 목록
│   ├── previous_experiments.md RGB–HSI 이력과 검증 전용 실험
│   ├── experiments/
│   │   ├── template.md       공통 8절 양식
│   │   ├── reproduction/     PAN K4·K6 보고서
│   │   ├── rgb_hsi/          RGB01~11 보고서
│   │   └── pilot/            검증 전용 예비실험 보고서
│   ├── assets/<실험 ID>/     그래프·4패널 이미지·오프라인 HTML
│   └── operations/           예비실험 실행 방법
├── experiments/
│   ├── results/<실험 ID>/    전체/장면별 수치·곡선·선택·해시
│   ├── checkpoints/          보존된 학습 가중치
│   └── logs/                 실행 로그 (주로 서버 로컬)
├── references/               논문·데이터 감사·공식 submodule
├── configs/                  실행 및 평가에 쓰는 설정
├── scripts/                  학습·평가·정합·산출물 내보내기
├── src/ssamrn/               모델·데이터·손실·지표
└── tests/                    동작과 평가 프로토콜 검증
```

보고서는 문서 폴더에, 큰 시각화 산출물은 실험 ID별 assets에, 숫자 근거는 results에 둡니다. 서버 가중치는 보고서에 서버 경로임을 명시합니다. 과거 모델 구현은 가중치 추론에 필요하므로 유지합니다.

이 정리는 문서 경로와 중복 설명을 정돈한 작업입니다. 학습 데이터·가중치·결과 JSON·이미지·컨트롤러 경로는 유지합니다. 현재 서버 작업의 상태를 확인하지 않은 상태에서 로그·설정·캐시를 삭제하지 않습니다.

## 실행 참고 자료

- [짧은 예비실험 운영](operations/pilot_experiments.md)
- [6단계 예비실험 구성](operations/six_stage_pilot.md)
- [데이터셋 감사](../references/rgb_hsi_datasets.md)
- [과거 서버 정리와 가중치 보존 해시](../references/cleanup_20261003.json)

## 웹뷰어 갱신

공개 Pages는 `gh-pages` 브랜치의 `ssa-mrn/`에서 제공합니다. 현재 선택은 RGB11이며, 근거는 동일 타일 평가를 사용한 최근 연구 중 PSNR·SAM입니다. RGB09의 MSE가 더 낮다는 제한도 함께 표시합니다.

배포 원본은 `docs/assets/rgb11_gradient_suite/band_viewer/`입니다. `scripts/prepare_viewer_site.py`로 별도 배포 폴더를 준비합니다. 이전 HTML은 각 실험의 assets에 보존합니다. 웹페이지 이미지 데이터는 8비트 시각화이며 원본 HSI 큐브가 아닙니다.
