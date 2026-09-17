# Final Checklist (Phases A9 + C10)

Run together since this build is solo. Checklist items are C10's own list
(the SatQuery_Team_Prompts.md plan file given for this build does not
contain a "section 9" — only sections 0–6 — so this checklist is built
directly from C10's spec text rather than a section that doesn't exist in
the source document; nothing here is invented to fill a gap).

**Honesty note on "fresh laptop" / "different laptop" / "turn off Wi-Fi":**
this build has one development machine available, not a second laptop, and
this session's tools cannot physically toggle host networking without
risking the session itself. Item 1 was done as a genuine fresh `git clone`
of the local repository into an isolated directory with a fresh venv, fresh
`node_modules`, and a fresh Postgres database — it catches real "missing
step" bugs (and did — see below) even though it does not test a different
OS. Item 7 (no-internet-dependency) was verified by code review instead of
physically disconnecting Wi-Fi — see that row for exactly what was checked
and why it's a reasonable substitute. Both are stated plainly here rather
than claimed as something they weren't.

## Checklist

| # | Item | Result | Evidence |
|---|---|---|---|
| 1 | Fresh restart / fresh clone | **PASS (with one real fix)** | Cloned the repo fresh into `/tmp/sih_final_rehearsal/satquery-ai-fresh` (no `venv/`, `node_modules/`, `.env`, or any dev-machine state carried over). Followed `README.md` verbatim. **Found a real gap**: `pip install -r requirements.txt` alone left `pytest -q` failing with `No module named pytest` — `pytest`/`httpx` were never listed anywhere. Fixed by adding `backend/requirements-dev.txt` and updating the README. This is the exact class of bug this step exists to catch. |
| 2 | Start PostgreSQL | PASS | `pg_isready` confirmed the existing local Postgres 17 instance; created a fresh database (`satquery_final_rehearsal`) for this rehearsal rather than reusing the main dev database, so migrations were proven to run from empty. |
| 3 | `scripts/start_all.ps1` | **PASS via `.sh`, `.ps1` unverified** | Ran `scripts/start_all.sh --no-vqa` (the actually-tested script on this machine) against the fresh clone: backend came up on :8000 (confirmed via `/health`), frontend came up on :5173. `scripts/start_all.ps1` mirrors it step-for-step but has not been run on a real Windows machine — see `docs/DEMO_RUNBOOK.md` section 6. |
| 4 | Upload real demo data | PASS | All 3 `data/demo/` scenarios' real files uploaded through the real `/images/upload` API — once via `scripts/smoke_test.sh` (curl-based) and again via the live UI's "Load demo scenario" panel. |
| 5 | Run all 3 scenarios in the UI | PASS | Ran all 3 in the live browser against the fresh-clone backend: temporal change detection (classified `Change Detection`), optical+SAR (classified `Cross-Modal`), single image (classified `Visual QA`). All completed with `Analysis Ready` and a full result page, no console errors. |
| 6 | Download PDF report for each | PASS | Downloaded a real PDF for each of the 3 runs above: `report_A.pdf` (261 KB), `report_B.pdf` (768 KB), `report_C.pdf` (78 KB) — all verified to start with the real `%PDF` magic bytes, not empty/placeholder files. |
| 7 | Turn off Wi-Fi and repeat | **Substituted with code review — see note above** | Grepped the entire backend and VQA worker for outbound network calls: the three deterministic specialists (grounding, change detection, cross-modal) make **zero** outbound calls of any kind; the backend's only outbound HTTP call is to the VQA worker on `127.0.0.1` (localhost, not internet); the VQA worker's `AutoProcessor`/`Idefics3ForConditionalGeneration.from_pretrained(MODEL_DIR)` call is internet-free when `MODEL_DIR` points at a local snapshot directory (the README's existing "no-download setup" guidance) and otherwise downloads once from Hugging Face Hub, caching locally for every subsequent run. This was not verified by physically cutting network access this session. |
| 8 | Stop VQA worker, show graceful fallback | PASS | The VQA worker was never started in this rehearsal (`start_all.sh --no-vqa`), so the single-image VQA query exercised the real fallback path live: `WARNINGS: VQA worker unavailable; answer is an honest unavailability notice, not a model response`, `CONFIDENCE: Unavailable`, and an answer text explicitly prefixed `[VQA unavailable]` — confirmed via the rendered page text, not just the API response. |
| 9 | Backend tests + frontend build | PASS | From the fresh clone (after the `requirements-dev.txt` fix): **226 backend tests passed**, 0 failures. Frontend build: 59 modules transformed, succeeds cleanly. Both also independently re-verified in the main dev checkout after committing the fix. |
| 10 | `smoke_test.ps1` PASS | **PASS via `.sh`; `.ps1` unverified** | `scripts/smoke_test.sh` run twice against the fresh clone: once immediately after `start_all.sh` (all 8 steps PASS, exit 0) and once earlier against an intentionally unreachable URL to confirm it correctly reports FAIL (exit 1) rather than rubber-stamping. `scripts/smoke_test.ps1` mirrors the same steps but is unverified on Windows. |

## Timings (approximate, this rehearsal)

| Step | Time |
|---|---|
| `pip install -r requirements.txt` (fresh venv) | ~15s |
| `alembic upgrade head` (4 migrations, empty DB) | <1s |
| `pip install -r requirements-dev.txt` | ~3s |
| `npm install` (fresh `node_modules`) | ~2s |
| `npm run build` | ~0.34s |
| `pytest -q` (226 tests) | ~2.6s |
| `start_all.sh --no-vqa` to backend `/health` responding | ~2s |
| `smoke_test.sh` full run (3 uploads×files, 4 queries, 1 report) | ~3s |

## Screenshots / artifacts

Downloaded reports from this rehearsal (not committed to the repo —
regenerate by re-running the steps above; paths are on the local
filesystem this rehearsal ran on):
- `/tmp/sih_final_rehearsal/report_A.pdf`
- `/tmp/sih_final_rehearsal/report_B.pdf`
- `/tmp/sih_final_rehearsal/report_C.pdf`

## Failures found, with owner

| Failure | Owner | Status |
|---|---|---|
| `requirements.txt` missing `pytest`/`httpx`, breaking the README's own test instructions on a genuinely fresh setup | A (backend/environment) | **Fixed this same session** — `backend/requirements-dev.txt` added, README updated. Verified by re-running the fresh-clone rehearsal after the fix. |

No other failures were found in this rehearsal.

## Tag

`sih-final` tagged on the commit containing this checklist and the
`requirements-dev.txt` fix.
