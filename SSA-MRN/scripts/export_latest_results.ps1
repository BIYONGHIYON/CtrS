param(
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [string]$DataRoot = 'C:\CtrS\SSA-MRN\data\dataset\RGB-HSI\LIB-HSI',
    [string]$PreviousMetrics,
    [string]$Python = 'C:\CtrS\.venv\Scripts\python.exe'
)
$ErrorActionPreference = 'Stop'
$arguments = @((Join-Path $PSScriptRoot 'export_results.py'), '--checkpoint', $Checkpoint, '--output-dir', $OutputDir, '--data-root', $DataRoot)
if ($PreviousMetrics) { $arguments += @('--previous-metrics', $PreviousMetrics) }
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw 'Result export failed; report is incomplete' }
