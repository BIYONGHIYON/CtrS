# RGB–HSI 데이터셋 비교

2026-09-29 기준 공개 논문·배포 자료의 수치입니다. 아래 `픽셀 크기`는 영상의 가로·세로 배열 크기이며, 지상 표본 거리나 센서의 물리적 픽셀 피치가 아닙니다. `쌍/세트 수`는 공개 자료의 규모로, 이 저장소에서 다운로드·무결성 검사를 마친 수량을 뜻하지 않습니다.

| 데이터셋 | 장면과 촬영 특성 | RGB 크기 | HSI 크기·밴드 | 공개 쌍/세트 수 | RGB 유도 HSI 공간 초해상도에 사용할 때 |
|---|---|---:|---:|---:|---|
| [BUSIFusion 실제 데이터](https://github.com/CPREgroup/Real-Spec-RGB-Fusion#real-world-dataset) | 서로 다른 Huawei RGB 카메라와 Specim IQ로 촬영한 실제 센서 쌍 | 5472×7296 px | 512×512 px, 204밴드(400–1000 nm) | 142쌍 | 고해상도 RGB와 저해상도 HSI 구성이 직접 맞지만, 카메라 간 시야·정합을 검사해야 함 |
| [Agro-HSR](https://doi.org/10.1016/j.compag.2025.111103) | 세 품종 고구마 790개체를 Specim IQ로 촬영. 원래 204밴드 중 31밴드를 선택해 배포 | 별도 원본 크기 미확인 | 512×512 px, 31밴드(400–1000 nm) | 1,322쌍(790개체) | 원래 과제는 RGB→HSI 복원. HSI를 합성 축소한 공간 초해상도 평가에 사용 가능하며 개체 단위로 분할해야 함 |
| [HSI Road](https://github.com/NUST-Machine-Intelligence-Laboratory/hsi_road) | 주행 중 동기화 촬영한 RGB·가시광 HSI·근적외 HSI. 아래 크기는 [공개 전처리 코드](https://github.com/NUST-Machine-Intelligence-Laboratory/hsi_road/blob/master/hsi_builder.py)의 중앙 크롭 결과 | 704×1280 px | 가시광 256×480 px·16밴드, 근적외 192×384 px·25밴드 | 3,799프레임 세트 | 서로 다른 카메라의 실제 해상도 차이가 있으나 기하 정합·시야 교집합을 확인해야 함. 연속 프레임을 독립 장면으로 세지 않음 |
| [LIB-HSI](https://researchdata.edu.au/lib-hsi-rgb-building-facades/2006456) | 건물 외벽을 Specim IQ로 촬영. HSI에서 만든 의사 RGB와 별도 RGB 센서 영상을 구별해야 함 | 별도 RGB 원본 크기 미확인 | 512×512 px, 204밴드(400–1000 nm) | 513쌍 | 별도 센서 RGB의 실제 크기·정합을 압축 해제 후 확인해야 함. 배포 분할은 학습 393/검증 45/시험 75장 |
| [HSIFoodIngr-64](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/E7WDNQ) | 음식 21종·재료 64종의 RGB–HSI 쌍과 재료 분할 마스크 | 512×512 px | 512×512 px, 204밴드(400–1000 nm) | 3,389쌍 | 현재는 링크만 보관하고 다운로드하지 않음. 추후 실험에 필요하면 다운로드를 검토 |

**로컬 확보 상태:** `LIB-HSI.zip`은 외장 드라이브의 `SSA-MRN Data/RGB-HSI` 아래에서 다운로드 중인 부분 파일로 확인했으며, 완료와 압축 무결성은 아직 확인하지 않았습니다. BUSIFusion·Agro-HSR·HSI Road의 원본 파일 및 실제 쌍 수는 해당 경로에서 확인하지 못했습니다. HSIFoodIngr-64는 현재 다운로드하지 않고 링크만 보관합니다. 따라서 표의 크기·쌍 수를 로컬 검사 결과로 해석하지 않습니다.

**평가 단위:** 패치 수가 많아도 독립 촬영 대상·장면 수가 늘지는 않습니다. Agro-HSR은 고구마 개체, HSI Road는 주행 구간, LIB-HSI는 건물/촬영 위치 등으로 누출을 막는 분할을 검토합니다. 원본 고해상도 HSI 정답이 없는 실제 센서 쌍에서는 정량 참값 지표를 주장하지 않고, 정합·일관성·시각 결과를 별도로 살핍니다. 참값이 필요한 PSNR·SSIM·SAM 등은 고해상도 HSI를 인위적으로 축소한 **합성 평가**로 명시합니다.
