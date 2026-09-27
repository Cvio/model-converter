# Runs the seven conversion steps for one model, in order, stopping at the
# first one that fails.
#
#     .\convert.ps1 whisper-to-onnx\configs\es-small-hitz.yaml
#
# If a step stops, fix what it says, then carry on from that step:
#
#     .\convert.ps1 whisper-to-onnx\configs\es-small-hitz.yaml -From 4
#
# The step you start from is rerun with --force, because it may have left a
# partial output folder behind. The steps after it run normally.
# -Force reruns every step from the start, replacing earlier output.

param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Config,
    [ValidateRange(1, 7)][int]$From = 1,
    [switch]$Force
)
$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot

$steps = @(
    "1_check_setup.py",
    "2_download.py",
    "3_to_openai_format.py",
    "4_export_onnx.py",
    "5_check_int8.py",
    "6_assemble.py",
    "7_verify.py"
)

if (-not (Test-Path $Config)) {
    Write-Host "STOP: there is no config at $Config" -ForegroundColor Red
    exit 1
}
$configPath = (Resolve-Path $Config).Path
if (-not (Test-Path (Join-Path $repo ".venv"))) {
    Write-Host "STOP: this folder is not set up yet. Run .\setup.ps1 first." -ForegroundColor Red
    exit 1
}

$started = Get-Date
Push-Location $repo
try {
    for ($n = $From; $n -le 7; $n++) {
        $script = "whisper-to-onnx\steps\" + $steps[$n - 1]
        $stepArgs = @("run", "python", $script, "--config", $configPath)
        if ($Force -or ($n -eq $From -and $PSBoundParameters.ContainsKey("From"))) { $stepArgs += "--force" }

        Write-Host ""
        Write-Host ("=" * 70) -ForegroundColor Cyan
        Write-Host " Step $n of 7: $($steps[$n - 1])" -ForegroundColor Cyan
        Write-Host ("=" * 70) -ForegroundColor Cyan
        & uv @stepArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Host "Step $n stopped. Fix what it says above, then continue with:" -ForegroundColor Red
            Write-Host "    .\convert.ps1 $Config -From $n" -ForegroundColor Yellow
            exit 1
        }
    }
}
finally {
    Pop-Location
}

$minutes = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)
Write-Host ""
$which = if ($From -eq 1) { "All seven steps" } else { "Steps $From to 7" }
Write-Host "$which passed in $minutes minutes." -ForegroundColor Green
Write-Host "Open cnverc, tick 'Compare recognizers', and speak to hear the new model." -ForegroundColor Green
