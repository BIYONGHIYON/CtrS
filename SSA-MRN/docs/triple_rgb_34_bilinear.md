# 34특징 코어의 확대·축소를 PAN–MS 재현 방식에 맞춘 실험

## 변경 범위

비교 기준은 저장소의 PAN–MS 재현 `RestoredPansharpeningNet`과 공개 upstream 모듈의 실제 연산입니다.
논문 전체 구현과 동일하다는 의미는 아닙니다. 기존 34특징 모델과 별도 설정·출력 경로로 보존합니다.

| 처리 | 기존 34특징 | 새 Bilinear 구성 |
|---|---|---|
| 입력 HSI 64→256 baseline | 23탭 | 23탭 유지 |
| 압축된 입력 특징의 LMS 생성 | 23탭 | 23탭 유지 |
| guide 256→64→256 보조 경로 | 평균 풀링 → 23탭 | Bilinear → Bilinear |
| 코어 내부 분광 입력·특징의 ×2/×4 확대 | 23탭 | Bilinear |
| 코어 내부 특징·guide의 ×1/2·×1/4 축소 | 평균 풀링 | Bilinear |
| 합성 LR 생성 | area x4 | area x4 유지 |
| RGB별 34특징·6밴드 그룹·K=4 | 사용 | 유지 |

Bilinear는 재현 코드와 같은 `align_corners=True`와 `scale_factor`를 사용합니다.
실제 사용되는 `upsample1/100/101/102`, `downsample1/2/100/200`을 모두 맞췄습니다.
외부 HSI baseline과 입력 LMS는 재현 연구의 사전 LMS에 대응하므로 23탭을 유지합니다.
최종 출력은 기존 확장 연구처럼 HSI baseline + RGB별 결합 잔차입니다.

## 새 설정

- 모델 종류: `rgb_triple_grouped34_bilinear`.
- [설정 파일](../configs/lib_rgb_hsi_triple34_k4_bilinear_tiles.json).
- 출력: `SSA-MRN/experiments/checkpoints/lib_rgb_hsi_triple34_k4_bilinear_tiles`.
- 물리 배치 1 / 누적 4, 검증 배치 1, train quadrants / eval tiles, 패치 256 유지.
- 파라미터 수는 6,157,180개로 기존 34특징과 동일합니다.

모델 종류가 다르므로 기존 34특징 체크포인트를 이 설정으로 재개·평가하면 프로토콜 검사에서 거부합니다.
학습 가능한 가중치 모양이 같더라도 공간 연산이 달라지므로 새 실험으로 학습해야 합니다.
기존 설정으로 기존 가중치를 평가하는 동작은 유지합니다.

## 실행과 검증

새 코드가 설치된 저장소 루트에서, 서버 데이터와 Python을 지정하여 smoke를 실행합니다.

```powershell
& C:\CtrS\.venv\Scripts\python.exe SSA-MRN\scripts\train_lib.py `
  --config SSA-MRN\configs\lib_rgb_hsi_triple34_k4_bilinear_tiles.json `
  --data-root 'C:\CtrS\SSA-MRN\data\dataset\RGB-HSI\LIB-HSI' `
  --smoke
```

VS Code에는 `LIB: triple34 bilinear tiles smoke / train (new run) / evaluate best`를 추가했습니다.
기존 예약 작업은 34특징 area/23탭 설정에 고정되어 있습니다. 현재 실행 중인 학습을 수정하거나 중단하지 않았습니다.
이 새 설정의 실제 GPU smoke와 백그라운드 학습 전환은 아직 수행하지 않았습니다.

코드 검증:

- 사용되는 확대·축소 8개 모듈의 출력이 PAN–MS 재현과 정확히 같음.
- 동일 가중치·입력의 전체 scalar 코어 출력과 입력 기울기를 재현 연산과 비교.
- HSI 23탭 baseline 유지 및 세 RGB 코어의 순전파·역전파 확인.
- 기존 12특징·34특징 테스트도 통과.

새 설정의 성능 결과는 아직 없습니다. 같은 시험 타일·정합·마스크에서 기존 34특징과 비교해야 하며,
RGB guide가 PAN과 같은 센서 응답이라는 의미는 아닙니다.
