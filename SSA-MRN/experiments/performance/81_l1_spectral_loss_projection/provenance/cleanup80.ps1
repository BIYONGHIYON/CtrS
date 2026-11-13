$ErrorActionPreference = 'Stop'
$active = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' }
if ($active) { throw 'python running; refusing cleanup' }
$before = (Get-PSDrive C).Free
foreach ($p in 'C:\CtrS-p80\run','C:\CtrS-p80\SSA-MRN','C:\CtrS-p80\ctrs80-source.tgz','C:\CtrS-p80\fetch80.tgz') {
  if (Test-Path $p) {
    $s = if ((Get-Item $p).PSIsContainer) { (Get-ChildItem $p -Recurse -File -Force | Measure-Object Length -Sum).Sum } else { (Get-Item $p).Length }
    Remove-Item $p -Recurse -Force
    '{0} removed {1:N2} GB' -f $p, ($s/1GB)
  }
}
$freed = ((Get-PSDrive C).Free - $before) / 1GB
$free = (Get-PSDrive C).Free / 1GB
"freed $([math]::Round($freed,2)) GB, free now $([math]::Round($free,2)) GB"
