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
