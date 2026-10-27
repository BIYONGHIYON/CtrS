$ErrorActionPreference = 'Stop'
$dst = 'C:\CtrS-p66s3'
Copy-Item "$env:TEMP\p66s3\run66s44.ps1" $dst -Force
Copy-Item "$env:TEMP\p66s3\prepare.py" "$dst\prepare44.py" -Force
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine="powershell.exe -NoProfile -ExecutionPolicy Bypass -File $dst\run66s44.ps1"}
if ($r.ReturnValue -ne 0) { throw 'launcher creation failed' }
'LAUNCHED pid ' + $r.ProcessId
