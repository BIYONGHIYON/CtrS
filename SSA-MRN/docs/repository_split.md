# PAN–MS 전용 저장소 전환

2026-10-07부터 CtrS 팀 연구는 PAN–MS 팬샤프닝 재현과 성능 개선에 집중합니다. RGB–HSI 확장은 [BIYONGHIYON/RGB-HSI-SR](https://github.com/BIYONGHIYON/RGB-HSI-SR)로 이전했습니다. ECRformer는 현재 파일과 브랜치 이력에서 제거했습니다.

## 보존과 이력 정리

원본 기준 커밋은 `23c4d9b86fed044d4578d397e533a4f9e9e4ada5`입니다. 삭제 전 전체 Git 이력을 소유자의 로컬 `CtrS-archive-20261007/repository.bundle`에 백업했고 bundle 검증을 완료했습니다. 이 파일은 일반 clone에 포함하지 않습니다.

PAN–MS 재현의 K6 완료 시점 `b3baaf4`까지의 이력에서 PAN 전용 코드·설정·보고서·가중치·평가 경로를 추출했습니다. 무관한 파일만 변경한 커밋은 제거하고 저자·날짜는 유지했습니다. 그 위에 현재 PAN–MS 문서 구조와 동일한 가중치·결과를 반영했습니다. [커밋 매핑](history_rewrite_map.tsv)에서 남은 과거 커밋의 새 ID를 확인할 수 있습니다.

PAN–MS K4/K6 산출물 66개의 SHA-256이 이전 main과 일치합니다. 개인 저장소의 가중치·수치·이미지·HTML 402개도 이관 전과 일치합니다. 원본 데이터와 서버 실행 파일은 변경하지 않았습니다. 서버 전용 RGB 가중치는 개인 저장소 보고서에 원래 경로·해시를 보존했습니다.

GitHub의 과거 PR 참조·캐시는 브랜치 이력과 별개로 남을 수 있습니다. 이 정리는 현재 공개 브랜치와 일반 clone의 이력을 정리한 것이며 GitHub 내부 저장소의 즉시 물리 삭제를 의미하지 않습니다.

## 팀원 작업 폴더 전환

기존 clone을 새 main과 merge하거나 강제 push하면 제거한 이력이 다시 연결될 수 있습니다. 로컬 작업은 먼저 백업하고 새 폴더에 clone하세요.

```bash
git clone --recurse-submodules https://github.com/BIYONGHIYON/CtrS.git CtrS-PAN-MS
```

데이터와 미반영 코드는 기존 폴더에서 필요한 파일만 옮기고 `.git`은 복사하지 않습니다. 새로운 연구 변경은 새 브랜치→PR 절차를 따릅니다. 이번 이력 재작성은 저장소 소유자가 명시적으로 승인한 일회성 전환입니다.

[팀 연구](../README.md) · [개인 연구 이관 기록](https://github.com/BIYONGHIYON/RGB-HSI-SR/blob/main/MIGRATION.md)
