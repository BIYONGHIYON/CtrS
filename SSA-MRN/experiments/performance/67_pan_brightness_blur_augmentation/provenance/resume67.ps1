# Resume experiment 67 after the reboot (same sealed runtime; the trainer resumes from latest.pt). Refuses if any Python runs.
$ErrorActionPreference = 'Stop'
$active = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' }
if ($active) { throw 'another Python process is running; refusing to resume' }
$cmd = 'cmd.exe /c "cd /d C:\CtrS-p67\SSA-MRN\experiments\performance\67_pan_brightness_blur_augmentation && set CUBLAS_WORKSPACE_CONFIG=:4096:8&& set PYTHONIOENCODING=utf-8&& C:\CtrS\.venv\Scripts\python.exe -u run.py --mode run --resume >> C:\CtrS-p67\run\controller.log 2>&1"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd}
if ($r.ReturnValue -ne 0) { throw 'launcher creation failed' }
$r | ConvertTo-Json | Set-Content -Encoding UTF8 C:/CtrS-p67/resume_launcher.json
'RESUMED pid ' + $r.ProcessId
