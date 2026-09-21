<#
.SYNOPSIS
Starts the local change detection worker (Siamese CNN) on 127.0.0.1.

.DESCRIPTION
Uses an isolated worker python environment. Resolves it in this order:
  1. $env:CHANGE_WORKER_PYTHON (absolute path to python.exe)
  2. ml/change-worker/venv (created by installing requirements.change-worker.txt)
ASCII only on purpose - Windows PowerShell 5.1 reads .ps1 files as ANSI unless
they carry a BOM, so non-ASCII characters here can break parsing.

.PARAMETER Port
HTTP port (default 8002, or $env:CHANGE_WORKER_PORT).
#>
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
