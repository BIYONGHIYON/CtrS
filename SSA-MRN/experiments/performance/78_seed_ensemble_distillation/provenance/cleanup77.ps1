$ErrorActionPreference = 'Stop'
$active = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' }
if ($active) { throw 'python running; refusing cleanup' }
$before = (Get-PSDrive C).Free
foreach ($p in 'C:\CtrS-p77\run','C:\CtrS-p77\SSA-MRN','C:\CtrS-p77\ctrs77-source.tgz') {
  if (Test-Path $p) {
    $s = if ((Get-Item $p).PSIsContainer) { (Get-ChildItem $p -Recurse -File -Force | Measure-Object Length -Sum).Sum } else { (Get-Item $p).Length }
    Remove-Item $p -Recurse -Force
    '{0} removed {1:N2} GB' -f $p, ($s/1GB)
  }
}
'freed {0:N2} GB, free now {1:N2} GB' -f ((Get-PSDrive C).Free-$before)/1GB, ((Get-PSDrive C).Free/1GB)
cmd /c dir C:\CtrS-p77 /B
