@echo off
rem Experiment 73: new Python environment on the reorganized server (the C:\CtrS\.venv and Python 3.10 runtime were gone on 2026-11-03).
set LOG=C:\CtrS-p73\install.log
echo START %DATE% %TIME% > %LOG%
C:\Users\trainer\ctrs-python\cpython-3.11.17-windows-x86_64-none\python.exe -m venv C:\CtrS-p73\.venv >> %LOG% 2>&1 || (echo FAIL venv >> %LOG% & exit /b 1)
C:\CtrS-p73\.venv\Scripts\python.exe -m pip install --upgrade pip >> %LOG% 2>&1 || (echo FAIL pip >> %LOG% & exit /b 2)
C:\CtrS-p73\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128 >> %LOG% 2>&1 || (echo FAIL torch >> %LOG% & exit /b 3)
C:\CtrS-p73\.venv\Scripts\python.exe -m pip install numpy Pillow scipy h5py PyYAML tqdm scikit-image "opencv-python-headless>=4.10,<5" "matplotlib>=3.8,<4" >> %LOG% 2>&1 || (echo FAIL packages >> %LOG% & exit /b 4)
C:\CtrS-p73\.venv\Scripts\python.exe -c "import torch,numpy,h5py,scipy,skimage,matplotlib;print('VERSIONS',torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.get_device_name(0),numpy.__version__,h5py.__version__,scipy.__version__,skimage.__version__)" >> %LOG% 2>&1 || (echo FAIL check >> %LOG% & exit /b 5)
C:\CtrS-p73\.venv\Scripts\python.exe -m pip freeze > C:\CtrS-p73\pip_freeze.txt
echo INSTALL DONE %DATE% %TIME% >> %LOG%
