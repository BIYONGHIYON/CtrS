@echo off
set CUBLAS_WORKSPACE_CONFIG=:4096:8
set PYTHONIOENCODING=utf-8
cd /d C:\CtrS-p59\SSA-MRN\experiments\performance\59_extended_cosine_training
if not exist C:\CtrS-p59\run mkdir C:\CtrS-p59\run
C:\CtrS\.venv\Scripts\python.exe -u run.py --mode prepare > C:\CtrS-p59\run\prepare.log 2>&1 || exit /b 1
C:\CtrS\.venv\Scripts\python.exe -u run.py --mode optimize > C:\CtrS-p59\run\optimize.log 2>&1 || exit /b 2
C:\CtrS\.venv\Scripts\python.exe -u run.py --mode run > C:\CtrS-p59\run\controller.log 2>&1
