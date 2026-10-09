# Kaggle QB B1/B2 실행·결과 복구

[완료 보고서](../experiments/kaggle_b1_b2.md) · [단일 셀 Python](../../scripts/kaggle_b1_b2_one_cell.py) · [단일 셀 notebook](../../scripts/kaggle_b1_b2.ipynb)

Kaggle GPU **T4×2**, Internet ON, QB H5 데이터셋 연결 후 Python 파일 전체를 셀 하나에 붙여 넣거나 한 셀 notebook을 가져온다. B0은 실행하지 않는다. 기본값은 K6·seed42·100에폭·Adam1e−4·effective/micro32이다. GPU0=B1/GPU1=B2이며 `SWAP_GPUS`로 교환할 수 있다.

입력 루트는 `/kaggle/input/datasets/biyonghiyon/ctrs-pan-sharpening-dataset`이며 `QuickBird__Training_Dataset__train_qb.h5`, `QuickBird__Training_Dataset__valid_qb.h5`, `QuickBird__Testing_Dataset_ReducedData_H5_Format__test_qb_multiExm1.h5`, `QuickBird__Testing_Dataset_FullData_H5_Format__test_qb_OrigScale_multiExm1.h5`를 직접 지정한다. `._` 보조 파일을 사용하지 않는다.

시작 즉시 START를 출력한다. 소스 다운로드는 30초 연결 제한, 데이터 SHA는 진행률, 전체 TRAIN 관측 검사는 GPU FFT 진행률을 표시한다. 이후 worker별 GPU 데이터 적재→초기 shape/gradient 검사→micro32 메모리 검사→학습→FP32 RR/FR 전체 평가→ZIP·해시 보관 순서다. 실제 코드 순서는 smoke 검사 뒤 GPU 적재와 benchmark이며 두 모델의 본체 초기화와 epoch 순서 해시를 비교한다.

데이터는 원본 H5에서 각 GPU에 FP32로 상주하고 배치는 CUDA에서 생성한다. 큰 정규화 디스크 캐시를 만들지 않는다. micro32가 OOM이면 **양쪽 함께 16으로 변경하고 새 OUTPUT_DIR**로 실행한다. 유효 배치32는 유지하지만 부동소수점 누적이 달라 동일 설정의 bitwise 결과라고 표시하지 않는다. 서로 다른 모델의 micro를 독립적으로 선택하지 않는다.

기본 출력은 `/kaggle/working/ssa_kaggle_B1_B2_shared3_batch32_v3`, ZIP은 같은 working 아래 `_results.zip`이다. 완료 후 ZIP·별도 SHA JSON을 다운로드한다. 코드 셀 삭제와 Kaggle 세션 초기화는 구분한다. 세션 종료 전에 best/latest·결과·실행 코드의 원격 보관과 해시 검증이 끝났는지 확인한다. 현재 제공된 Git 보관은 [repository manifest](../assets/kaggle_b1_b2/repository_manifest.json)에 있다.

재개는 같은 output의 정확히 같은 config·source·worker에서 epoch 경계 `latest.pt`를 이용한다. **구 v1/v2 결과를 자동 이어받지 않는다.** config·worker를 바꾸면 새 output이 필요하다. best/latest는 optimizer·scaler·CPU/CUDA RNG·loader RNG·100에폭 history를 포함한다. 실험별 코드 snapshot·checkpoint config를 임의 수정하지 않는다.

원격 복구는 `git fetch origin ssa-b1-b2-results` 후 별도 checkout에서 `SSA-MRN/docs/assets/kaggle_b1_b2/`를 가져온다. `verification.json`의 4개 weight SHA와 `repository_manifest.json`의 파일별 SHA를 검증한다. 저장소에 원본 H5는 없으므로 dataset manifest와 일치하는 Kaggle 데이터를 연결한다. source snapshot이 `code/CtrS/SSA-MRN/src`·`references/upstream/network.py`를 함께 포함한다. 보관된 worker/model snapshot은 실제 실행 원본이고 단일 셀의 내장 문자열 해시와 일치한다.

Git 정리 전에 실행 중/재개 예정 실험과 다른 채팅 사용 여부, 원격 보관·복구·hash를 확인한다. 이번 문서 정리에서는 실제 서버 실행 폴더나 다운로드 결과를 삭제하지 않았다.
