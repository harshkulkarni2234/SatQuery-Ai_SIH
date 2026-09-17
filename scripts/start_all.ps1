<#
.SYNOPSIS
Starts the full SatQuery AI stack (backend, VQA worker, frontend) for a demo.

.DESCRIPTION
Windows mirror of scripts/start_all.sh, authored to match it step-for-step.
NOTE: this script has NOT been executed on a real Windows machine in this
session (development happened on macOS) — scripts/start_all.sh is the one
actually run and verified end-to-end (see docs/DEMO_RUNBOOK.md section 6).
Smoke-test this .ps1 on an actual Windows machine before relying on it for
judging day.

.PARAMETER NoVqa
Skip starting the VQA worker (description queries will use the honest
fallback instead).
#>
param(
    [switch]$NoVqa
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $PSScriptRoot ".logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Host "== SatQuery AI: starting all services =="

# 1. PostgreSQL check (does not start it)
$pgReady = Get-Command pg_isready -ErrorAction SilentlyContinue
if ($pgReady) {
    & pg_isready -q
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK]   PostgreSQL is accepting connections"
    } else {
        Write-Host "[WARN] PostgreSQL does not appear to be running — start the service first"
    }
} else {
    Write-Host "[WARN] pg_isready not found — cannot verify PostgreSQL is running"
}

# 2. Backend
Write-Host "[..]   Starting backend (uvicorn) on :8000"
$backendVenv = Join-Path $RepoRoot "backend\venv\Scripts\python.exe"
Push-Location (Join-Path $RepoRoot "backend")
$backendProc = Start-Process -FilePath $backendVenv `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000" `
    -RedirectStandardOutput (Join-Path $LogDir "backend.log") `
    -RedirectStandardError (Join-Path $LogDir "backend.err.log") `
    -PassThru -WindowStyle Hidden
Pop-Location
$backendProc.Id | Out-File (Join-Path $LogDir "backend.pid")
Start-Sleep -Seconds 2
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 5
    Write-Host "[OK]   Backend is up (pid $($backendProc.Id)) — status: $($health.status)"
} catch {
    Write-Host "[FAIL] Backend did not respond on :8000 — check $LogDir\backend.log"
}

# 3. VQA worker (optional)
if (-not $NoVqa) {
    Write-Host "[..]   Starting VQA worker on :8001"
    Push-Location (Join-Path $RepoRoot "ml\vqa-worker")
    & .\run_worker.ps1
    Pop-Location
    Write-Host "[..]   (VQA worker takes longer to become healthy — model load in progress)"
} else {
    Write-Host "[SKIP] VQA worker (-NoVqa passed) — description queries will use the honest fallback"
}

# 4. Frontend
Write-Host "[..]   Starting frontend (vite dev server) on :5173"
Push-Location (Join-Path $RepoRoot "frontend")
$frontendProc = Start-Process -FilePath "npm" -ArgumentList "run", "dev", "--", "--port", "5173" `
    -RedirectStandardOutput (Join-Path $LogDir "frontend.log") `
    -RedirectStandardError (Join-Path $LogDir "frontend.err.log") `
    -PassThru -WindowStyle Hidden
Pop-Location
$frontendProc.Id | Out-File (Join-Path $LogDir "frontend.pid")
Write-Host "[OK]   Frontend starting (pid $($frontendProc.Id))"

Write-Host ""
Write-Host "== Done. Open http://localhost:5173 =="
Write-Host "PIDs recorded in $LogDir\*.pid — stop each with: Stop-Process -Id <pid>"
Write-Host "Run scripts\smoke_test.ps1 once all services report healthy."
