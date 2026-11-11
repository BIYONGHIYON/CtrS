$ErrorActionPreference = 'Stop'
if (-not (Test-Path C:/CtrS-p78/run/prepare_optimize_done.txt)) { throw 'prepare/optimize not complete' }
$active = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' }
if ($active) { throw 'another Python process is running; refusing to launch' }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c C:\CtrS-p78\launch78_run.cmd'}
if ($r.ReturnValue -ne 0) { throw 'independent launcher creation failed' }
$r | ConvertTo-Json | Set-Content -Encoding UTF8 C:/CtrS-p78/launcher_run.json
'LAUNCHED pid ' + $r.ProcessId
