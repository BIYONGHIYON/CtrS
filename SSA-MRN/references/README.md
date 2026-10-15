# SSA-MRN 공식 코드 참조

`upstream/`은 [SSA-MRN 공식 저장소](https://github.com/zhouchuanxu/SSA-MRN)를 커밋 `a4ca40e407b12bf4c30f804384405ce321d11c51`에 고정한 Git 하위 모듈입니다. 이 코드는 [원 논문](https://doi.org/10.1109/JSTARS.2025.3543827)의 모델 정의를 확인하는 기준으로 사용합니다.

공식 저장소에는 `network.py`와 README가 있지만 전체 학습·평가 파이프라인, 데이터 설정, 사전 학습 가중치는 없습니다. 부족한 재현 코드는 이 프로젝트의 `common/`과 `research/<분야>/<번호>_<변경명>/`에 두고, 확인한 차이는 [재현 보고서](../research/performance/1_public_code_reproduction/README.md)에 기록합니다. `upstream/`의 원본 파일은 수정하지 않습니다.

## 원 논문

[`paper/xu2025_ssa_mrn_jstars.pdf`](paper/xu2025_ssa_mrn_jstars.pdf)는 SSA-MRN 원 논문의 출판본 PDF입니다. 내용을 수정하지 않은 원본이며 SHA-256은 `485b0eb92869267c65aa052897abcac35689a258ccd0c10b11bf11abac187392`입니다.

S. Xu, S. Zhong, H. Li, and C. Gong, "Spectral–Spatial Attention-Guided Multi-Resolution Network for Pansharpening," *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, vol. 18, pp. 7559-7571, 2025. DOI: [10.1109/JSTARS.2025.3543827](https://doi.org/10.1109/JSTARS.2025.3543827)

© 2025 The Authors. 이 논문은 [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) 라이선스로 공개되었으며, 이 저장소는 해당 라이선스에 따라 출처를 표시해 재배포합니다. 이 저장소의 실험과 결과는 저자나 출판사의 검토·보증을 받지 않았습니다.
