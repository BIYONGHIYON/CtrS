$ErrorActionPreference = 'Stop'
$active = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' }
if ($active) { throw 'another Python process is running; refusing to resume' }
if (Test-Path C:/CtrS-p81/run/all_complete.json) { throw 'already complete' }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c C:\CtrS-p81\resume81.cmd'}
if ($r.ReturnValue -ne 0) { throw 'launcher creation failed' }
$r | ConvertTo-Json | Set-Content -Encoding UTF8 C:/CtrS-p81/launcher_resume1.json
'RESUMED pid ' + $r.ProcessId
