<#
.SYNOPSIS
End-to-end smoke test against a running SatQuery AI backend.

.DESCRIPTION
Windows mirror of scripts/smoke_test.sh, authored to match it step-for-step:
hits /health, uploads the real data/demo/ scenario files, runs the 4 demo
queries, downloads one PDF report, prints PASS/FAIL.
NOTE: this script has NOT been executed on a real Windows machine in this
session -- scripts/smoke_test.sh is the one actually run and verified
end-to-end (see docs/DEMO_RUNBOOK.md section 6).

.PARAMETER BaseUrl
Backend base URL. Default http://127.0.0.1:8000.
#>
param(
    [string]$BaseUrl = "http://127.0.0.1:8000"
)

$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$DemoDir = Join-Path $RepoRoot "data\demo"
$Fails = 0

function Pass($msg) { Write-Host "[PASS] $msg" }
function Fail($msg) { Write-Host "[FAIL] $msg"; $script:Fails++ }

Write-Host "== SatQuery AI smoke test against $BaseUrl =="

# 1. /health
try {
    $health = Invoke-RestMethod -Uri "$BaseUrl/health" -TimeoutSec 5
    if ($health.database) {
        Pass "/health responded: $($health | ConvertTo-Json -Compress)"
    } else {
        Fail "/health missing expected fields"
    }
} catch {
    Fail "/health did not respond: $($_.Exception.Message)"
}

function Upload-Image($Path, $Modality, $CaptureDate) {
    try {
        $filePath = Resolve-Path $Path
        $url = "$BaseUrl/images/upload"
        $args = @("-s", "-X", "POST", $url, "-F", "file=@`"$filePath`"", "-F", "modality=$Modality")
        if ($CaptureDate) { $args += "-F"; $args += "capture_date=$CaptureDate" }
        $json = curl.exe @args
        $resp = $json | ConvertFrom-Json
        return $resp.image_id
    } catch {
        return $null
    }
}

# 2. Upload demo images
$singleId = Upload-Image (Join-Path $DemoDir "scenario_A_single\single_image.jpg") "OPTICAL" $null
if ($singleId) { Pass "Uploaded scenario A single image ($singleId)" } else { Fail "Scenario A upload failed" }

$beforeId = Upload-Image (Join-Path $DemoDir "scenario_B_temporal\before.tif") "OPTICAL" "2018-08-24"
$afterId  = Upload-Image (Join-Path $DemoDir "scenario_B_temporal\after.tif") "OPTICAL" "2022-08-23"
if ($beforeId -and $afterId) { Pass "Uploaded scenario B temporal pair ($beforeId, $afterId)" } else { Fail "Scenario B upload failed" }

$opticalId = Upload-Image (Join-Path $DemoDir "scenario_C_optical_sar\optical.png") "OPTICAL" "2017-08-08"
$sarId     = Upload-Image (Join-Path $DemoDir "scenario_C_optical_sar\sar.png") "SAR" "2017-08-08"
if ($opticalId -and $sarId) { Pass "Uploaded scenario C optical+SAR pair ($opticalId, $sarId)" } else { Fail "Scenario C upload failed" }

function Run-Query($Text, $ImageIds) {
    try {
        $body = @{ query_text = $Text; image_ids = $ImageIds } | ConvertTo-Json
        $resp = Invoke-RestMethod -Uri "$BaseUrl/query" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 60
        return $resp.query_id
    } catch {
        return $null
    }
}

# 3. Run the 4 demo queries
$lastQueryId = $null
if ($singleId) {
    $qid = Run-Query "What can you tell me about this image?" @($singleId)
    if ($qid) { Pass "VQA query ran (query_id $qid)"; $lastQueryId = $qid } else { Fail "VQA query failed" }

    $qid = Run-Query "Show me the water body." @($singleId)
    if ($qid) { Pass "Grounding query ran (query_id $qid)"; $lastQueryId = $qid } else { Fail "Grounding query failed" }
}

if ($beforeId -and $afterId) {
    $qid = Run-Query "What changed between these two dates?" @($beforeId, $afterId)
    if ($qid) { Pass "Change detection query ran (query_id $qid)"; $lastQueryId = $qid } else { Fail "Change detection query failed" }
}

if ($opticalId -and $sarId) {
    $qid = Run-Query "Use the optical and SAR images together to identify built-up and water-covered regions." @($opticalId, $sarId)
    if ($qid) { Pass "Cross-modal query ran (query_id $qid)"; $lastQueryId = $qid } else { Fail "Cross-modal query failed" }
}

# 4. Download one report
if ($lastQueryId) {
    $logDir = Join-Path $PSScriptRoot ".logs"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $reportPath = Join-Path $logDir "smoke_test_report.pdf"
    try {
        Invoke-WebRequest -Uri "$BaseUrl/query/$lastQueryId/report.pdf" -OutFile $reportPath -TimeoutSec 30
        $bytes = Get-Content -Path $reportPath -Encoding Byte -TotalCount 4
        $header = [System.Text.Encoding]::ASCII.GetString($bytes)
        if ($header -eq "%PDF") {
            Pass "Downloaded a real PDF report to $reportPath"
        } else {
            Fail "Report download did not produce a valid PDF"
        }
    } catch {
        Fail "Report download failed: $($_.Exception.Message)"
    }
} else {
    Fail "No successful query to download a report for"
}

Write-Host ""
if ($Fails -eq 0) {
    Write-Host "== SMOKE TEST: PASS =="
    exit 0
} else {
    Write-Host "== SMOKE TEST: FAIL ($Fails failure(s)) =="
    exit 1
}
