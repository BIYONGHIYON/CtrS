@echo off
set CUBLAS_WORKSPACE_CONFIG=:4096:8
set PYTHONIOENCODING=utf-8
cd /d C:\CtrS-p76\SSA-MRN\experiments\performance\76_quarter_resolution_branch
C:\CtrS-p73\.venv\Scripts\python.exe -u train.py --config C:\CtrS-p76\run\runtime_config.json --mode evaluate --sensor QB --arm C2MS_W64D8_W4 --seed 42 > C:\CtrS-p76\run\runs\QB_C2MS_W64D8_W4_s42\evaluate_rerun.log 2>&1 || exit /b 1
C:\CtrS-p73\.venv\Scripts\python.exe -u train.py --config C:\CtrS-p76\run\runtime_config.json --mode evaluate --sensor QB --arm C2MS_W4 --seed 42 > C:\CtrS-p76\run\runs\QB_C2MS_W4_s42\evaluate_rerun.log 2>&1 || exit /b 2
echo done > C:\CtrS-p76\run\evaluation_rerun_done.txt
