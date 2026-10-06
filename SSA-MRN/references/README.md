# SSA-MRN 공식 코드 참조

`upstream/`은 [SSA-MRN 공식 저장소](https://github.com/zhouchuanxu/SSA-MRN)를 커밋 `a4ca40e407b12bf4c30f804384405ce321d11c51`에 고정한 Git 하위 모듈입니다. 이 코드는 [원 논문](https://doi.org/10.1109/JSTARS.2025.3543827)의 모델 정의를 확인하는 기준으로 사용합니다.

공식 저장소에는 `network.py`와 README가 있지만 전체 학습·평가 파이프라인, 데이터 설정, 사전 학습 가중치는 없습니다. 부족한 재현 코드는 이 프로젝트의 `src/ssamrn/`과 `scripts/`에 두고, 확인한 차이는 [재현 보고서](../docs/experiments/reproduction/pan_k4.md)에 기록합니다. `upstream/`의 원본 파일은 수정하지 않습니다.

데이터 감사는 [RGB–HSI 데이터셋](rgb_hsi_datasets.md), 과거 속도 측정 수치는 `benchmarks/`에 보존합니다.
