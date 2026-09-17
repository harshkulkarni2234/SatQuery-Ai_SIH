#!/usr/bin/env bash
# Phase C9: end-to-end smoke test against a running backend. Hits /health,
# uploads the real data/demo/ scenario files through the real API, runs the
# 4 demo queries, downloads one PDF report, and prints a final PASS/FAIL.
# Actually run and verified on this project's macOS dev machine — see
# scripts/smoke_test.ps1 for the Windows mirror (authored to match this
# script step-for-step, but not executed on a real Windows machine in this
# session; see docs/DEMO_RUNBOOK.md section 6).
#
# Usage: bash scripts/smoke_test.sh [base_url]   (default http://127.0.0.1:8000)
set -uo pipefail

BASE_URL="${1:-http://127.0.0.1:8000}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_DIR="$REPO_ROOT/data/demo"
FAILS=0

pass() { echo "[PASS] $1"; }
fail() { echo "[FAIL] $1"; FAILS=$((FAILS + 1)); }

echo "== SatQuery AI smoke test against $BASE_URL =="

# 1. /health
health_json="$(curl -fsS "$BASE_URL/health" 2>/dev/null)"
if [[ -n "$health_json" ]] && echo "$health_json" | grep -q '"database"'; then
  pass "/health responded: $health_json"
else
  fail "/health did not respond or missing expected fields"
fi

# 2. Upload demo images
upload() {
  local path="$1" modality="$2" date="$3" name
  name="$(basename "$path")"
  local data=(-F "file=@${path}" -F "modality=${modality}")
  if [[ -n "$date" ]]; then
    data+=(-F "capture_date=${date}")
  fi
  curl -fsS -X POST "$BASE_URL/images/upload" "${data[@]}" 2>/dev/null \
    | python3 -c 'import sys, json; print(json.load(sys.stdin).get("image_id", ""))' 2>/dev/null
}

single_id="$(upload "$DEMO_DIR/scenario_A_single/single_image.jpg" OPTICAL "")"
[[ -n "$single_id" ]] && pass "Uploaded scenario A single image ($single_id)" || fail "Scenario A upload failed"

before_id="$(upload "$DEMO_DIR/scenario_B_temporal/before.tif" OPTICAL "2018-08-24")"
after_id="$(upload "$DEMO_DIR/scenario_B_temporal/after.tif" OPTICAL "2022-08-23")"
[[ -n "$before_id" && -n "$after_id" ]] && pass "Uploaded scenario B temporal pair ($before_id, $after_id)" || fail "Scenario B upload failed"

optical_id="$(upload "$DEMO_DIR/scenario_C_optical_sar/optical.png" OPTICAL "2017-08-08")"
sar_id="$(upload "$DEMO_DIR/scenario_C_optical_sar/sar.png" SAR "2017-08-08")"
[[ -n "$optical_id" && -n "$sar_id" ]] && pass "Uploaded scenario C optical+SAR pair ($optical_id, $sar_id)" || fail "Scenario C upload failed"

# 3. Run the 4 demo queries
run_query() {
  local text="$1" ids="$2"
  curl -fsS -X POST "$BASE_URL/query" \
    -H "Content-Type: application/json" \
    -d "{\"query_text\": \"${text}\", \"image_ids\": ${ids}}" 2>/dev/null
}

last_query_id=""
if [[ -n "$single_id" ]]; then
  resp="$(run_query "What can you tell me about this image?" "[\"$single_id\"]")"
  qid="$(echo "$resp" | python3 -c 'import sys, json; print(json.load(sys.stdin).get("query_id", ""))' 2>/dev/null)"
  [[ -n "$qid" ]] && { pass "VQA query ran (query_id $qid)"; last_query_id="$qid"; } || fail "VQA query failed: $resp"

  resp="$(run_query "Show me the water body." "[\"$single_id\"]")"
  qid="$(echo "$resp" | python3 -c 'import sys, json; print(json.load(sys.stdin).get("query_id", ""))' 2>/dev/null)"
  [[ -n "$qid" ]] && { pass "Grounding query ran (query_id $qid)"; last_query_id="$qid"; } || fail "Grounding query failed: $resp"
fi

if [[ -n "$before_id" && -n "$after_id" ]]; then
  resp="$(run_query "What changed between these two dates?" "[\"$before_id\", \"$after_id\"]")"
  qid="$(echo "$resp" | python3 -c 'import sys, json; print(json.load(sys.stdin).get("query_id", ""))' 2>/dev/null)"
  [[ -n "$qid" ]] && { pass "Change detection query ran (query_id $qid)"; last_query_id="$qid"; } || fail "Change detection query failed: $resp"
fi

if [[ -n "$optical_id" && -n "$sar_id" ]]; then
  resp="$(run_query "Use the optical and SAR images together to identify built-up and water-covered regions." "[\"$optical_id\", \"$sar_id\"]")"
  qid="$(echo "$resp" | python3 -c 'import sys, json; print(json.load(sys.stdin).get("query_id", ""))' 2>/dev/null)"
  [[ -n "$qid" ]] && { pass "Cross-modal query ran (query_id $qid)"; last_query_id="$qid"; } || fail "Cross-modal query failed: $resp"
fi

# 4. Download one report
if [[ -n "$last_query_id" ]]; then
  report_path="$REPO_ROOT/scripts/.logs/smoke_test_report.pdf"
  mkdir -p "$(dirname "$report_path")"
  if curl -fsS "$BASE_URL/query/$last_query_id/report.pdf" -o "$report_path" 2>/dev/null \
     && [[ "$(head -c4 "$report_path")" == "%PDF" ]]; then
    pass "Downloaded a real PDF report to $report_path"
  else
    fail "Report download did not produce a valid PDF"
  fi
else
  fail "No successful query to download a report for"
fi

echo ""
if [[ "$FAILS" -eq 0 ]]; then
  echo "== SMOKE TEST: PASS =="
  exit 0
else
  echo "== SMOKE TEST: FAIL ($FAILS failure(s)) =="
  exit 1
fi
