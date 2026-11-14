@echo off
set CUBLAS_WORKSPACE_CONFIG=:4096:8
set PYTHONIOENCODING=utf-8
cd /d C:\CtrS-p83\SSA-MRN\experiments\performance\83_factorized_c2ms_finetuning
C:\CtrS-p73\.venv\Scripts\python.exe -u run.py --mode run --resume >> C:\CtrS-p83\run\controller.log 2>&1
