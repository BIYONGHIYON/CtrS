<p align="center">
  <img src="./CtrS-icon-circle.png" alt="CtrS" width="180" />
</p>

# CtrS

동국대학교 종합설계 연구 저장소입니다. 현재 우선순위는 SSA-MRN 원 논문의 pansharpening 결과를 재현하는 것입니다. 원 논문 재현을 먼저 진행한 뒤 RGB–HSI 확장 여부를 검토합니다.

## 현재 진행 연구

### [SSA-MRN 원 논문 재현](./SSA-MRN/README.md)

고해상도 PAN, 저해상도 PAN(LRPAN), 저해상도 MS를 입력으로 사용해 고해상도 MS를 복원하는 원 연구를 재현합니다. 공식 저장소를 버전 고정된 하위 모듈로 연결했고, 누락된 데이터 로더·학습·평가 파이프라인은 논문 설정에 맞춰 구성합니다.

저장소를 처음 내려받았거나 하위 모듈 파일이 비어 있다면 다음 명령으로 공식 코드를 가져옵니다.

```bash
git submodule update --init --recursive
```

### [ECRformer 연구안](./ECRformer/README.md)

별도 연구 후보로 문서를 보존합니다. 현재 개발 우선순위는 SSA-MRN 원 논문 재현입니다.

## 공통 원칙

- 논문에 기재된 설정과 재현 구현에서 바꾼 설정을 구분해 기록합니다.
- 데이터셋과 모델 가중치는 저장소에 올리지 않고 각 연구 폴더의 안내에 따라 로컬에 둡니다.
- 재현 완료는 실행 성공만으로 판단하지 않고 논문 결과와 비교한 뒤 기록합니다.

## 원격 저장소

- GitHub: https://github.com/BIYONGHIYON/CtrS
