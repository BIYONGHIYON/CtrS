# 66 seed 43 started in parallel (user request 2026-10-27 18:40); the first job (run66s3.ps1) is stopped after seed 42.
# Separate folder; the running checkout C:\CtrS-p65s3 is only read. 4 CPU threads, below-normal priority, alongside the seed 42 job.
$ErrorActionPreference = 'Stop'
$src = 'C:\CtrS-p65s3'; $dst = 'C:\CtrS-p66s3'; $py = 'C:\CtrS\.venv\Scripts\python.exe'
function Log($m) { Add-Content -Encoding UTF8 "$dst\watcher43.log" ((Get-Date -Format s) + ' ' + $m) }
Log 'started'
# copy made by the first job

$exp = "$dst\SSA-MRN\experiments\performance\66_observation_error_projection_analysis"
New-Item -ItemType Directory -Force $exp, "$dst\results" | Out-Null
Copy-Item "$dst\prepare.py" "$exp\prepare.py" -Force
$data = 'C:\CtrS\SSA-MRN\data\dataset'
$ErrorActionPreference = 'Continue'
foreach ($seed in 43) {
  foreach ($sensor in 'GF2', 'WV3') {
    $l1s = "$src\run\runs\${sensor}_C2MS_L1S_s$seed\checkpoints\best.pt"
    $c2ms = "$exp\..\54_multi_sensor_scale_mixed_training\checkpoints\C2MS_${sensor}_s$seed\best.pt"
    if (-not (Test-Path "$src\run\runs\${sensor}_C2MS_L1S_s$seed\evaluation_complete.json")) { Log "$sensor s$seed not evaluated by 65; skipped"; continue }
    $h = @{ l1s = (Get-FileHash $l1s -Algorithm SHA256).Hash; c2ms = (Get-FileHash $c2ms -Algorithm SHA256).Hash; prepare = (Get-FileHash "$exp\prepare.py" -Algorithm SHA256).Hash }
    $h | ConvertTo-Json | Set-Content -Encoding UTF8 "$dst\results\inputs_${sensor}_s$seed.json"
    Log "running $sensor seed $seed"
    $a = "-u prepare.py --sensor $sensor --seed $seed --threads 4 --data `"$data`" --l1s `"$l1s`" --out `"$dst\results`""
    $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $exp -RedirectStandardOutput "$dst\results\p66_${sensor}_s$seed.log" -RedirectStandardError "$dst\results\p66_${sensor}_s$seed.err" -NoNewWindow -PassThru; $null = $p.Handle
    Start-Sleep 2; try { $p.PriorityClass = 'BelowNormal' } catch { Log 'priority not set' }
    $p.WaitForExit(); $code = $p.ExitCode
    Log "$sensor seed $seed finished with exit code $code"
  }
}
Set-Content -Encoding UTF8 "$dst\done43.json" ('{"time": "' + (Get-Date -Format s) + '"}')
Log 'done'
