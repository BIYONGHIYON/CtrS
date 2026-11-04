@echo off
rem Experiment 74: disposable two-job concurrency benchmark after training and evaluation finished (GPU otherwise idle).
set CUBLAS_WORKSPACE_CONFIG=:4096:8
set PYTHONIOENCODING=utf-8
cd /d C:\CtrS-p74\SSA-MRN\experiments\performance\74_backbone_matched_parallel_branch
set PY=C:\CtrS-p73\.venv\Scripts\python.exe
set CFG=C:\CtrS-p74\run\runtime_config.json
%PY% -u train.py --config %CFG% --mode bench --sensor QB --arm C2MS_M --tag solo > C:\CtrS-p74\bench_solo.log 2>&1
start "" /b cmd /c "%PY% -u train.py --config %CFG% --mode bench --sensor QB --arm C2MS_M --tag pairA > C:\CtrS-p74\bench_pairA.log 2>&1 & echo done > C:\CtrS-p74\pairA.done"
%PY% -u train.py --config %CFG% --mode bench --sensor QB --arm C2MS_M --tag pairB > C:\CtrS-p74\bench_pairB.log 2>&1
:wait
if not exist C:\CtrS-p74\pairA.done ( timeout /t 2 /nobreak > nul & goto wait )
echo ALL DONE > C:\CtrS-p74\bench2.done
