param([ValidateSet("start","resume-latest","resume-best","status","inspect","logs","check")][string]$Command="status",[switch]$Follow)
$ErrorActionPreference="Stop"
$Python="C:\CtrS\.venv\Scripts\python.exe"
$Arguments=@("$PSScriptRoot\controlled_suite.py",$Command)
if($Follow){$Arguments+="--follow"}
& $Python @Arguments
if($LASTEXITCODE -ne 0){throw "Suite command failed: $LASTEXITCODE"}
