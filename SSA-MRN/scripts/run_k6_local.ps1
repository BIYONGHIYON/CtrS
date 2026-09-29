param(
    [Parameter(Mandatory = $true)][string]$DataRoot,
    [string]$Python = (Join-Path $PSScriptRoot '..\.venv-dml\Scripts\python.exe'),
    [ValidateSet('QB', 'GF2', 'WV3')][string[]]$Sensors = @('QB', 'GF2', 'WV3'),
    [switch]$Resume,
    [switch]$Smoke
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$dataset = @{
    QB = @{ Folder = 'QuickBird'; Stem = 'qb'; Run = 'qb_full' }
    GF2 = @{ Folder = 'Gaofen 2'; Stem = 'gf2'; Run = 'gf2_full' }
    WV3 = @{ Folder = 'WorldView 3'; Stem = 'wv3'; Run = 'wv3_full' }
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}
$logs = Join-Path $projectRoot 'experiments\logs\k6'
New-Item -ItemType Directory -Path $logs -Force | Out-Null

foreach ($sensor in $Sensors) {
    $entry = $dataset[$sensor]
    $source = Join-Path (Join-Path $DataRoot $entry.Folder) 'Training Dataset'
    $train = Join-Path $source "train_$($entry.Stem).h5"
    $valid = Join-Path $source "valid_$($entry.Stem).h5"
    if (-not (Test-Path -LiteralPath $train -PathType Leaf) -or
        -not (Test-Path -LiteralPath $valid -PathType Leaf)) {
        throw "Missing training or validation H5 for $sensor in $source"
    }

    $runGroup = if ($Smoke) { 'k6_smoke' } else { 'k6' }
    $checkpointDir = Join-Path $projectRoot "experiments\checkpoints\$runGroup\$($entry.Run)"
    $latest = Join-Path $checkpointDir 'latest.pt'
    $arguments = @(
        (Join-Path $PSScriptRoot 'train.py'),
        '--train', $train, '--val', $valid, '--sensor', $sensor,
        '--epochs', $(if ($Smoke) { '1' } else { '100' }), '--batch-size', '32', '--lr', '0.0001',
        '--seed', '42', '--ssai-dimension', '6', '--device', 'dml',
        '--checkpoint-dir', $checkpointDir
    )
    if ($Smoke) {
        $arguments += @('--max-train-samples', '32', '--max-val-samples', '32')
    }
    if ($Resume -and (Test-Path -LiteralPath $latest -PathType Leaf)) {
        $arguments += @('--resume', $latest)
    }
    $log = Join-Path $logs "$($runGroup)_$($entry.Run)_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
    & $Python @arguments 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) {
        throw "$sensor training failed; see $log"
    }
}
