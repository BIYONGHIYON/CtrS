# Kaggle QB K4 B1/B1_AA · 실행 및 보관

[완료 보고서](../experiments/kaggle_bandlimit_k4.md) · [한 셀 Python](../../scripts/kaggle_bandlimit_k4_cell.py) · [동일 ipynb](../../scripts/kaggle_bandlimit_k4_cell.ipynb)

현재 결과는 각200에폭까지 완료했다. 이 셀은 T4×2에서 GPU당1개 모델을 배정하며 새 학습은 사용자가 요청할 때만 시작한다. 모델·worker 문자열은 실제 산출물과 SHA-256가 일치한다. 제공 셀은 입력 Output의 자동 탐색이 기본값이며 실제 재개 세션에서는 RESUME_INPUT에 아래 경로를 명시했다.

```python
RESUME_INPUT = "/kaggle/input/notebooks/biyonghiyon/ssa-mrn-test/ssa_kaggle_bandlimit_k4_v1_fix_20261010"
```

재개 시 Saved Notebook Output의 원하는 버전을 Add Input → Notebook → Your Work에서 연결한다. 버전5는 완료 결과이고 버전4는 중간 결과다. **연결된 버전을 먼저 확인**하며 경로는 Kaggle Copy file path로 확인한다. 기존 working suite가 있으면 셀은 이를 우선 사용한다. 빈 새 세션에서는 RESUME_INPUT에서 matching suite_config를 검증한 뒤 working으로 복사한다. 같은 설정 후보가 여러개면 명시 경로가 필요하다.

원본 OUTPUT_DIR은 `/kaggle/working/ssa_kaggle_bandlimit_k4_v1_fix_20261010`이다. 데이터 경로는 flat `QuickBird__...h5`이며 자동 탐색 시 AppleDouble `._`와 비HDF5를 제외한다. 원본 셀의 명시 경로가 없는 경우에만 자동 탐색한다.

완료 결과 재개는 evaluation_complete와 results가 있으면 해당 run을 건너뛴다. incomplete run은 latest의 모델·optimizer·scaler·RNG·history·mid-epoch 샘플 순서/배치 위치를 복원한다. latest의 best_recovery로 best를 복원한다. **epochs200·hold100·seed·batch·AMP·소스·worker·모델 해시를 바꾸면 재개가 차단**된다.100에폭으로 설정을 바꿔 같은 checkpoint를 이어서는 안 된다. 소스 변경은 별도 OUTPUT_DIR의 새 실험으로 분리한다.

주기 checkpoint는 약60초와 매 완료 epoch에 atomic rename으로 저장한다. 셀 중단 시 worker를 terminate하므로 마지막 checkpoint 이후 작업은 되돌아갈 수 있다. 세션 종료 전 best/latest·설정·코드·로그·manifest를 Output 포함 Quick Save로 저장하고 saved Output에서 파일을 확인한다. **기본 Quick Save나 Draft 저장만으로 가중치가 보관되지 않는다.** Save & Run All은 새 실행을 시작하므로 현재 산출물 저장에는 Quick Save + Advanced Settings → Save output for this version을 선택한다.

Git 보관에서 CPU 복구/검증:

```bash
python SSA-MRN/scripts/verify_kaggle_bandlimit.py
python SSA-MRN/scripts/verify_kaggle_bandlimit.py --torch
```

`--torch`는 torch·numpy·scipy가 필요하고 GPU 학습을 수행하지 않는다. 원본코드는 assets 아래에서 읽는다. 원본 source/가중치 경로와 해시를 임의로 수정하지 않는다. 원격 full prediction NPZ는 Kaggle Output에서 복구한다. 원본데이터·이전실험을 삭제하는 명령은 제공하지 않는다.
