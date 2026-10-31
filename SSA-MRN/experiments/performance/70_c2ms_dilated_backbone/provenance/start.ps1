$ErrorActionPreference = 'Stop'
$active = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' }
if ($active) { throw 'another Python process is running; refusing to launch' }
if (!(Test-Path C:/CtrS-p70-71/run70/preflight/optimization.json)) { throw '70 preflight missing' }
if (!(Test-Path C:/CtrS-p70-71/precheck71/preflight/benchmark_n1_WV3_BLUR_COARSE.json)) { throw '71 benchmark missing' }
if (!(Test-Path C:/CtrS-p70-71/precheck71/preflight/checks.json)) { throw '71 preflight missing' }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c C:\CtrS-p70-71\SSA-MRN\experiments\performance\70_c2ms_dilated_backbone\provenance\launch.cmd'}
if ($r.ReturnValue -ne 0) { throw 'independent launcher creation failed' }
$r | ConvertTo-Json | Set-Content -Encoding UTF8 C:/CtrS-p70-71/launcher.json
'LAUNCHED pid ' + $r.ProcessId
