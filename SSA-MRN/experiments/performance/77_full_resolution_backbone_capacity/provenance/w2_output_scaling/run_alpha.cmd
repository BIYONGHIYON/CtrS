@echo off
set CUBLAS_WORKSPACE_CONFIG=:4096:8
set PYTHONIOENCODING=utf-8
cd /d C:\CtrS-alpha
C:\CtrS-p73\.venv\Scripts\python.exe -u sweep.py --data C:\CtrS\PAN-MS --alphas 0.75 1.0 1.1 1.2 1.3 1.5 1.75 2.0 2.5 3.0 --out C:\CtrS-alpha\alpha_sweep.json > C:\CtrS-alpha\sweep.log 2>&1
