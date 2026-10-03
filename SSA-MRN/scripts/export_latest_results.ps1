param(
    [string]$RunId = '20261003-022329-72b8a17b50de',
    [string]$DataRoot = 'C:\CtrS\SSA-MRN\data\dataset\RGB-HSI\LIB-HSI'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $root
$py = Join-Path $root '.venv/Scripts/python.exe'
$run = Join-Path $root "SSA-MRN/experiments/checkpoints/remote-runs/$RunId"
$checkpoint = Join-Path $run 'best.pt'
$config = Join-Path $run 'run_config.json'
$out = Join-Path $root "SSA-MRN/experiments/results/lib_grouped12_$RunId"
foreach ($path in @($py, $checkpoint, $config, $DataRoot)) {
    if (-not (Test-Path $path)) { throw "Missing path: $path" }
}
if (Test-Path $out) { throw "Result folder exists; preserve it: $out" }
& $py SSA-MRN/scripts/train_lib.py --config $config --checkpoint $checkpoint --data-root $DataRoot --output-dir $out --evaluate
if ($LASTEXITCODE -ne 0) { throw 'Test evaluation failed' }
$selectionCode = "import random,json; print(json.dumps({'seed':20261003,'split':'test','sample_indices':random.Random(20261003).sample(range(75),5),'selection':'Fixed before inspecting metrics; full256 one index per scene'}))"
$selection = & $py -c $selectionCode
if ($LASTEXITCODE -ne 0) { throw 'Selection failed' }
$selection | Set-Content (Join-Path $out 'selection.json') -Encoding UTF8
$indices = ($selection | ConvertFrom-Json).sample_indices
for ($i = 0; $i -lt $indices.Count; $i++) {
    $sampleOut = Join-Path $out ('sample_{0:D2}' -f ($i + 1))
    & $py SSA-MRN/scripts/preview_lib.py --checkpoint $checkpoint --data-root $DataRoot --split test --sample-index $indices[$i] --output-dir $sampleOut
    if ($LASTEXITCODE -ne 0) { throw "Preview failed: $($indices[$i])" }
}
foreach ($name in @('history.jsonl', 'run_config.json', 'validation_metrics.json')) {
    $source = Join-Path $run $name
    if (Test-Path $source) { Copy-Item $source $out }
}
$zip = "$out.zip"
if (Test-Path $zip) { throw "ZIP exists: $zip" }
Compress-Archive -Path "$out/*" -DestinationPath $zip
Write-Host "Results: $out"
Write-Host "Attach this ZIP: $zip"
