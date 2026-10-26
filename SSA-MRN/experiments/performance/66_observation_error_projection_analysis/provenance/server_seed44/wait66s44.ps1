# 66 seed 44 projection on the server CPU, after the 65 stage-2 controller (training + evaluation) has finished.
$ErrorActionPreference = 'Stop'
$run = 'C:\CtrS-p65s2\run'; $dst = 'C:\CtrS-p66s44'; $py = 'C:\CtrS\.venv\Scripts\python.exe'
function Log($m) { Add-Content -Encoding UTF8 "$dst\watcher.log" ((Get-Date -Format s) + ' ' + $m) }
Log 'watcher started'
while (-not (Test-Path "$run\all_complete.json")) {
  if (Test-Path "$run\controller_failure.json") { Log '65 controller failure; not running'; exit 1 }
  Start-Sleep 60
}
Log '65 all_complete found'
while (Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'CtrS-p65s2' }) { Start-Sleep 20 }
Start-Sleep 30; Log '65 python processes exited'
robocopy C:\CtrS-p65s2\SSA-MRN "$dst\SSA-MRN" /E /XD __pycache__ /NFL /NDL /NJH /NJS | Out-Null
if ($LASTEXITCODE -ge 8) { Log "robocopy failed $LASTEXITCODE"; exit 2 }
$exp = "$dst\SSA-MRN\experiments\performance\66_observation_error_projection_analysis"
New-Item -ItemType Directory -Force $exp | Out-Null
Copy-Item "$dst\prepare.py" "$exp\prepare.py" -Force
$data = 'C:\CtrS\SSA-MRN\data\dataset\QuickBird'
$rr = "$data\Testing Dataset (ReducedData, H5 Format)\test_qb_multiExm1.h5"
$fr = "$data\Testing Dataset (FullData, H5 Format)\test_qb_OrigScale_multiExm1.h5"
$l1s = "$run\runs\QB_C2MS_L1S_s44\checkpoints\best.pt"
$h = @{ rr = (Get-FileHash $rr -Algorithm SHA256).Hash; fr = (Get-FileHash $fr -Algorithm SHA256).Hash; l1s = (Get-FileHash $l1s -Algorithm SHA256).Hash; c2ms = (Get-FileHash "$dst\SSA-MRN\experiments\performance\53_scale_mixed_training\checkpoints\C2MS_QB_s44\best.pt" -Algorithm SHA256).Hash; prepare = (Get-FileHash "$exp\prepare.py" -Algorithm SHA256).Hash }
$h | ConvertTo-Json | Set-Content -Encoding UTF8 "$dst\inputs_sha256.json"
if ($h.rr -ne '9842A1232AD2D5B9A61C9FD7FB353E8FC6214EEAC6E947D888F221AD6ECDDA35' -or $h.fr -ne '4583C2890009E047DDC1E0DAF88DB3F5DB43041FC301FA9FF8EB52A37F84B88F') { Log 'test data sha mismatch'; exit 3 }
Log 'running 66 prepare.py --seed 44'
$ErrorActionPreference = 'Continue'
$a = "-u prepare.py --seed 44 --rr `"$rr`" --fr `"$fr`" --l1s `"$l1s`" --out `"$dst\results`""
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $exp -RedirectStandardOutput "$dst\p66s44.log" -RedirectStandardError "$dst\p66s44.err" -Wait -PassThru -NoNewWindow
$code = $p.ExitCode
Log "66 seed 44 finished with exit code $code"
if ($code -eq 0) { Set-Content -Encoding UTF8 "$dst\done.json" ('{"time": "' + (Get-Date -Format s) + '"}') }
exit $code
