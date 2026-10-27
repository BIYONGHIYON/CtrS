$ErrorActionPreference = 'Stop'
$dst = 'C:\CtrS-p66s3'
# stop the sequential watcher (PowerShell only) so that it does not start seed 43 again; its running python child keeps going
$w = Get-CimInstance Win32_Process -Filter "ProcessId=6356"
if ($w -and $w.CommandLine -match 'run66s3.ps1') { Stop-Process -Id 6356 -Force; Add-Content -Encoding UTF8 "$dst\watcher.log" ((Get-Date -Format s) + ' watcher stopped by user request; seed 43 runs in run66s3b.ps1') }
Copy-Item "$env:TEMP\p66s3\run66s3b.ps1" $dst -Force
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine="powershell.exe -NoProfile -ExecutionPolicy Bypass -File $dst\run66s3b.ps1"}
if ($r.ReturnValue -ne 0) { throw 'launcher creation failed' }
'LAUNCHED pid ' + $r.ProcessId
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Select-Object ProcessId, ParentProcessId, @{n='cmd';e={$_.CommandLine.Substring(0, [Math]::Min(120, $_.CommandLine.Length))}} | Format-Table -AutoSize | Out-String -Width 200
