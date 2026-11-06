@echo off
set CUBLAS_WORKSPACE_CONFIG=:4096:8
set PYTHONIOENCODING=utf-8
cd /d C:\CtrS-p77\SSA-MRN\experiments\performance\77_full_resolution_backbone_capacity
if not exist C:\CtrS-p77\run mkdir C:\CtrS-p77\run
C:\CtrS-p73\.venv\Scripts\python.exe -u run.py --mode prepare > C:\CtrS-p77\run\prepare.log 2>&1 || exit /b 1
C:\CtrS-p73\.venv\Scripts\python.exe -u run.py --mode optimize > C:\CtrS-p77\run\optimize.log 2>&1 || exit /b 2
C:\CtrS-p73\.venv\Scripts\python.exe -u run.py --mode run > C:\CtrS-p77\run\controller.log 2>&1
