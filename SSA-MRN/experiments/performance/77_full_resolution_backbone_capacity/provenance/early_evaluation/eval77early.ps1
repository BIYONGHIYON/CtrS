$ErrorActionPreference = 'Stop'
$run = 'C:\CtrS-p77\run'
$src = 'C:\CtrS-p77\SSA-MRN\experiments\performance\77_full_resolution_backbone_capacity'
$py = 'C:\CtrS-p73\.venv\Scripts\python.exe'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'; $env:PYTHONIOENCODING = 'utf-8'
Set-Location $src
"started $(Get-Date -Format s)" | Out-File -Encoding ascii "$run\evaluation_early_status.txt"
foreach ($a in 'C2MS_B64D8_W64D8', 'C2MS_B128D8') {
  $d = "$run\runs\QB_${a}_s42"
  while (-not (Test-Path "$d\training_complete.json")) { Start-Sleep -Seconds 30 }
  while ((Test-Path "$d\worker.lock") -and (Get-Process -Id (Get-Content "$d\status.json" -Raw | ConvertFrom-Json).pid -ErrorAction SilentlyContinue)) { Start-Sleep -Seconds 15 }
  if (Test-Path "$d\evaluation_complete.json") { continue }
  "evaluating $a $(Get-Date -Format s)" | Out-File -Append -Encoding ascii "$run\evaluation_early_status.txt"
  cmd /c "`"$py`" -u train.py --config $run\runtime_config.json --mode evaluate --sensor QB --arm $a --seed 42 > $d\evaluate_early.log 2>&1"
  if ($LASTEXITCODE -ne 0) { "failed $a $LASTEXITCODE $(Get-Date -Format s)" | Out-File -Append -Encoding ascii "$run\evaluation_early_status.txt"; exit 1 }
  "done $a $(Get-Date -Format s)" | Out-File -Append -Encoding ascii "$run\evaluation_early_status.txt"
}
"all_done $(Get-Date -Format s)" | Out-File -Append -Encoding ascii "$run\evaluation_early_status.txt"
