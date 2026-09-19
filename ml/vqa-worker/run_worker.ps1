<#
.SYNOPSIS
Starts the local VQA worker (Base SmolVLM + experimental Stage 3 LoRA) on 127.0.0.1.
.DESCRIPTION
Uses an isolated worker python environment. Resolves it in this order:
  1. $env:VQA_WORKER_PYTHON (absolute path to python.exe, e.g. the tested Phase 7C env)
  2. ml/vqa-worker/venv (created by installing requirements.vqa-worker.txt)
.PARAMETER Port
HTTP port (default 8001, or $env:VQA_WORKER_PORT).
.PARAMETER NoSpecialist
Do not load the experimental LoRA adapter.
#>
param(
    [int]$Port = 8001,
    [switch]$NoSpecialist
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if ($env:VQA_WORKER_PORT) { $Port = [int]$env:VQA_WORKER_PORT }
if (-not $env:VQA_WORKER_URL) { $env:VQA_WORKER_URL = "http://127.0.0.1:$Port" }

$py = $env:VQA_WORKER_PYTHON
if (-not $py) {
    $candidate = Join-Path $scriptDir "venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $candidate) {
        $py = $candidate
    } else {
        Write-Error "No worker python found. Set VQA_WORKER_PYTHON or create ml/vqa-worker/venv (see README.md)."
        exit 1
    }
}

if (-not $env:MODEL_DIR) {
    $env:MODEL_DIR = "HuggingFaceTB/SmolVLM-256M-Instruct"
}
if (-not $env:ADAPTER_DIR) {
    $env:ADAPTER_DIR = Join-Path $scriptDir "..\smolvlm\lora_stage3_v2"
}

Write-Host "Starting VQA worker on http://127.0.0.1:$Port"
Write-Host "  python:       $py"
Write-Host "  MODEL_DIR:    $env:MODEL_DIR"
Write-Host "  ADAPTER_DIR:  $env:ADAPTER_DIR"
if ($NoSpecialist) {
    Write-Host "  SPECIALIST:   disabled (base model only)"
    $env:ADAPTER_DIR = ""
}

Push-Location $scriptDir
try {
    & $py -m uvicorn worker_service:app --host 127.0.0.1 --port $Port
} finally {
    Pop-Location
}