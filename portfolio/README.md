# 팀원별 포트폴리오 페이지

CtrS 연구를 소개하는 팀원별 정적 페이지입니다. 사이트 루트(https://biyonghiyon.github.io/CtrS/)는 결과 비교 사이트이고, 이 폴더의 페이지는 그 아래 `/portfolio/<GitHub ID>/`로 배포됩니다.

| 팀원 | 폴더 | 주소 |
|---|---|---|
| BIYONGHIYON | [biyonghiyon](biyonghiyon/) | https://biyonghiyon.github.io/CtrS/portfolio/biyonghiyon/ |

## 페이지 추가

1. 본인 GitHub ID를 소문자로 쓴 폴더 `portfolio/<github-id>/`를 만들고 `index.html`을 넣습니다. 이미지·CSS는 같은 폴더 안에 두고 상대 경로(`assets/...`)로 연결합니다.
2. 위 표에 한 줄을 추가합니다.
3. 새 브랜치에서 본인 폴더와 이 README만 커밋하고 PR을 올립니다. main에 병합되면 `.github/workflows/pages.yml`이 1-2분 안에 배포합니다.

- 다른 팀원의 폴더는 수정하지 않습니다.
- 결과 비교 사이트에서 포트폴리오로 가는 링크는 두지 않습니다. 포트폴리오에서 결과 비교 사이트로 가는 링크는 `https://biyonghiyon.github.io/CtrS/`를 씁니다.
- 브라우저에서 바로 열리는 정적 파일만 둡니다. 빌드 단계는 없고 폴더 내용이 그대로 배포됩니다.
