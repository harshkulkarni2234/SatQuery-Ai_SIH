<#
.SYNOPSIS
Starts the full SatQuery AI stack (backend, VQA worker, frontend) for a demo.

.DESCRIPTION
Windows mirror of scripts/start_all.sh, authored to match it step-for-step.
NOTE: this script has NOT been executed on a real Windows machine in this
session (development happened on macOS) -- scripts/start_all.sh is the one
actually run and verified end-to-end (see docs/DEMO_RUNBOOK.md section 6).
Smoke-test this .ps1 on an actual Windows machine before relying on it for
judging day.

.PARAMETER NoVqa
Skip starting the VQA worker (description queries will use the honest
fallback instead).

.PARAMETER NoChange
Skip starting the change-detection worker (the deterministic pixel-difference
method is used instead).
#>
param(
    [switch]$NoVqa,
    [switch]$NoChange
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $PSScriptRoot ".logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# Put PostgreSQL's bin on PATH only if pg_isready is not already reachable.
# A hardcoded version path works on one machine and silently does nothing on
# every other one, so discover the newest install instead.
if (-not (Get-Command pg_isready -ErrorAction SilentlyContinue)) {
    $pgBin = Get-ChildItem -Path "C:\Program Files\PostgreSQL" -Directory -ErrorAction SilentlyContinue |
        Sort-Object { [int]($_.Name -replace '\D', '0') } -Descending |
        ForEach-Object { Join-Path $_.FullName "bin" } |
        Where-Object { Test-Path -LiteralPath (Join-Path $_ "pg_isready.exe") } |
        Select-Object -First 1
    if ($pgBin) {
        $Env:PATH = "$pgBin;" + $Env:PATH
        Write-Host "[OK]   Using PostgreSQL tools from $pgBin"
    } else {
        Write-Host "[WARN] pg_isready not on PATH and no PostgreSQL install found under C:\Program Files\PostgreSQL"
        Write-Host "       Continuing anyway - this only affects the readiness check, not the backend."
    }
}

Write-Host "== SatQuery AI: starting all services =="

# 1. PostgreSQL check (does not start it)
$pgReady = Get-Command pg_isready -ErrorAction SilentlyContinue
if ($pgReady) {
    & pg_isready -q
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK]   PostgreSQL is accepting connections"
    } else {
        Write-Host "[WARN] PostgreSQL does not appear to be running -- start the service first"
    }
} else {
    Write-Host "[WARN] pg_isready not found -- cannot verify PostgreSQL is running"
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
    Write-Host "[OK]   Backend is up (pid $($backendProc.Id)) -- status: $($health.status)"
} catch {
    Write-Host "[FAIL] Backend did not respond on :8000 -- check $LogDir\backend.log"
}

# 3. VQA worker (optional)
if (-not $NoVqa) {
    Write-Host "[..]   Starting VQA worker on :8001"
    $vqaScript = Join-Path (Join-Path $RepoRoot "ml\vqa-worker") "run_worker.ps1"
    $vqaProc = Start-Process -FilePath "powershell" `
        -ArgumentList "-NoProfile", "-File", "`"$vqaScript`"" `
        -RedirectStandardOutput (Join-Path $LogDir "vqa_worker.log") `
        -RedirectStandardError (Join-Path $LogDir "vqa_worker.err.log") `
        -PassThru -WindowStyle Hidden
    $vqaProc.Id | Out-File (Join-Path $LogDir "vqa_worker.pid")
    Write-Host "[..]   (VQA worker takes longer to become healthy -- model load in progress)"
} else {
    Write-Host "[SKIP] VQA worker (-NoVqa passed) -- description queries will use the honest fallback"
}

# 4. Change-detection worker (optional -- needs its own venv AND the trained
# weights, which are not committed; without them the deterministic
# pixel-difference method runs and the result says so).
$changeDir = Join-Path $RepoRoot "ml\change-worker"
$changeWeights = Join-Path $RepoRoot "ml\change_model\trained\best_change_model.pt"
$changePython = if ($env:CHANGE_WORKER_PYTHON) { $env:CHANGE_WORKER_PYTHON } else { Join-Path $changeDir "venv\Scripts\python.exe" }
if ($NoChange) {
    Write-Host "[SKIP] Change worker (-NoChange passed) -- deterministic change detection will be used"
} elseif (-not (Test-Path -LiteralPath $changePython) -or -not (Test-Path -LiteralPath $changeWeights)) {
    Write-Host "[SKIP] Change worker (no worker python or no trained weights) -- deterministic change detection will be used"
} else {
    Write-Host "[..]   Starting change worker on :8002"
    $changeScript = Join-Path $changeDir "run_worker.ps1"
    $changeProc = Start-Process -FilePath "powershell" `
        -ArgumentList "-NoProfile", "-File", "`"$changeScript`"" `
        -RedirectStandardOutput (Join-Path $LogDir "change_worker.log") `
        -RedirectStandardError (Join-Path $LogDir "change_worker.err.log") `
        -PassThru -WindowStyle Hidden
    $changeProc.Id | Out-File (Join-Path $LogDir "change_worker.pid")
}

# 5. Frontend
Write-Host "[..]   Starting frontend (vite dev server) on :5173"
Push-Location (Join-Path $RepoRoot "frontend")
$frontendProc = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "npm", "run", "dev", "--", "--port", "5173" `
    -RedirectStandardOutput (Join-Path $LogDir "frontend.log") `
    -RedirectStandardError (Join-Path $LogDir "frontend.err.log") `
    -PassThru -WindowStyle Hidden
Pop-Location
$frontendProc.Id | Out-File (Join-Path $LogDir "frontend.pid")
Write-Host "[OK]   Frontend starting (pid $($frontendProc.Id))"

Write-Host ""
Write-Host "== Done. Open http://localhost:5173 =="
Write-Host "PIDs recorded in $LogDir\*.pid -- stop each with: Stop-Process -Id <pid>"
Write-Host "Run scripts\smoke_test.ps1 once all services report healthy."
