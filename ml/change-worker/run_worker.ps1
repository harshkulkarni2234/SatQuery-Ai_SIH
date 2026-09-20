param(
    [int]$Port = 8002
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if ($env:CHANGE_WORKER_PORT) { $Port = [int]$env:CHANGE_WORKER_PORT }
if (-not $env:CHANGE_WORKER_URL) { $env:CHANGE_WORKER_URL = "http://127.0.0.1:$Port" }

$py = $env:CHANGE_WORKER_PYTHON
if (-not $py) {
    $candidate = Join-Path $scriptDir "venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $candidate) {
        $py = $candidate
    } else {
        Write-Error "No worker python found. Set CHANGE_WORKER_PYTHON or create ml/change-worker/venv (see README.md)."
        exit 1
    }
}

if (-not $env:MODEL_PATH) {
    $env:MODEL_PATH = Join-Path $scriptDir "..\change_model\trained\best_change_model.pt"
}

Write-Host "Starting change detection worker on http://127.0.0.1:$Port"
Write-Host "  python:       $py"
Write-Host "  MODEL_PATH:   $env:MODEL_PATH"

Push-Location $scriptDir
try {
    & $py -m uvicorn worker_service:app --host 127.0.0.1 --port $Port
} finally {
    Pop-Location
}
