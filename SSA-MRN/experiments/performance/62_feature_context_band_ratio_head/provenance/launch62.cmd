@echo off
set CUBLAS_WORKSPACE_CONFIG=:4096:8
set PYTHONIOENCODING=utf-8
cd /d C:\CtrS-p62\SSA-MRN\experiments\performance\62_feature_context_band_ratio_head
if not exist C:\CtrS-p62\run mkdir C:\CtrS-p62\run
C:\CtrS\.venv\Scripts\python.exe -u run.py --mode prepare > C:\CtrS-p62\run\prepare.log 2>&1 || exit /b 1
C:\CtrS\.venv\Scripts\python.exe -u run.py --mode preflight > C:\CtrS-p62\run\preflight.log 2>&1 || exit /b 2
C:\CtrS\.venv\Scripts\python.exe -u run.py --mode run > C:\CtrS-p62\run\controller.log 2>&1
