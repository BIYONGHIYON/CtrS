@echo off
set CUBLAS_WORKSPACE_CONFIG=:4096:8
set PYTHONIOENCODING=utf-8
cd /d C:\CtrS-p65\SSA-MRN\experiments\performance\65_l1_spectral_loss
if not exist C:\CtrS-p65\run mkdir C:\CtrS-p65\run
C:\CtrS\.venv\Scripts\python.exe -u run.py --mode prepare > C:\CtrS-p65\run\prepare.log 2>&1 || exit /b 1
C:\CtrS\.venv\Scripts\python.exe -u run.py --mode optimize > C:\CtrS-p65\run\optimize.log 2>&1 || exit /b 2
C:\CtrS\.venv\Scripts\python.exe -u run.py --mode run > C:\CtrS-p65\run\controller.log 2>&1
