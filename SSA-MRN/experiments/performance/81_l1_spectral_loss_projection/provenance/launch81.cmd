@echo off
set CUBLAS_WORKSPACE_CONFIG=:4096:8
set PYTHONIOENCODING=utf-8
cd /d C:\CtrS-p81\SSA-MRN\experiments\performance\81_l1_spectral_loss_projection
if not exist C:\CtrS-p81\run mkdir C:\CtrS-p81\run
C:\CtrS-p73\.venv\Scripts\python.exe -u run.py --mode prepare > C:\CtrS-p81\run\prepare.log 2>&1 || exit /b 1
C:\CtrS-p73\.venv\Scripts\python.exe -u run.py --mode run > C:\CtrS-p81\run\controller.log 2>&1
