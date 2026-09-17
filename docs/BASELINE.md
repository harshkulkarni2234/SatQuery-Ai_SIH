# Baseline (Phase A0)

Recorded before any Wave 1+ work started. Tag: `baseline-round2` (commit `f1f7513`).

## Environment

| Tool | Version |
|---|---|
| Python | 3.11.7 |
| Node.js | v24.7.0 |
| npm | 11.5.1 |
| PostgreSQL | 17.11 (Homebrew) |
| OS | macOS (Darwin 27.0.0, arm64) |

Note: the committed `backend/venv/` is a Windows venv (`Scripts/python.exe`) from a
prior contributor's machine; it does not run on macOS/Linux. This build uses a
fresh `backend/venv_mac/` (gitignored, not committed) created with
`python3 -m venv venv_mac && venv_mac/bin/pip install -r requirements.txt pytest`.
`frontend/node_modules` also had to be reinstalled (`@rollup/rollup-darwin-arm64`
optional dependency missing — known npm issue
[npm/cli#4828](https://github.com/npm/cli/issues/4828)); `package-lock.json` was
regenerated on this platform.

## Backend tests

```
cd backend
venv_mac/bin/python -m alembic upgrade head
venv_mac/bin/python -m pytest -q
```

Result: **104 passed**, 2 warnings (both pre-existing `StarletteDeprecationWarning`
/ `anyio` deprecation notices from the `httpx`+`starlette.testclient` combo, not
from project code).

## Frontend build

```
cd frontend
npm run build
```

Result: **build succeeds**, 28 modules transformed, 0 warnings.
Output: `dist/index.html` (0.42 kB), `dist/assets/index-*.css` (22.38 kB),
`dist/assets/index-*.js` (173.90 kB).

## Database

Fresh local Postgres 17 database `satquery`, migrated with
`alembic upgrade head` from the single existing revision
`c11e454b7c43_create_initial_tables`. No manual schema edits.

## Pre-existing uncommitted working-tree changes (not part of this baseline)

At the start of this session, `git status` showed local, uncommitted modifications to:
- `ml/smolvlm/lora_stage3/adapter_config.json`
- `notebooks/geochat_feasibility_test.ipynb`

These predate Phase A0 and were left untouched (not authored by this build effort).

## Final (Phase A9/C10, recorded after a genuine fresh-clone rehearsal)

See `docs/FINAL_CHECKLIST.md` for the full rehearsal log. Summary:

- **Fresh-clone setup**: cloned the local repo (`git clone` from the working
  tree, not a physically different laptop — see the checklist for why) into
  `/tmp/sih_final_rehearsal/satquery-ai-fresh`, followed `README.md`
  verbatim: fresh `venv_mac`, `pip install -r requirements.txt`, fresh
  Postgres database (`satquery_final_rehearsal`), `alembic upgrade head`
  from empty, `npm install`, `npm run build`.
- **Found and fixed one real missing-step gap**: `pip install -r
  requirements.txt` alone left `pytest -q` failing with `No module named
  pytest` — `pytest`/`httpx` were never in `requirements.txt` and there was
  no dev-requirements file. Added `backend/requirements-dev.txt` and
  updated the README's setup and "Tests and build" sections to install it.
  This is exactly the kind of gap this rehearsal exists to catch.
- **Backend tests from the fresh clone (after the fix above)**: **226
  passed**, 0 failures, 55 warnings (all pre-existing rasterio/starlette
  deprecation notices, not from project code — same warning set as the
  main dev checkout).
- **Frontend build from the fresh clone**: succeeds, 59 modules
  transformed, `dist/assets/index-*.js` 187.65 kB (matches the main dev
  checkout's build exactly — same source, same result).
- **`scripts/start_all.sh` + `scripts/smoke_test.sh`** run against the
  fresh clone: both real commands, both real `PASS`.
- **All 3 demo scenarios run in the live UI** against the fresh-clone
  backend (not the main dev checkout) via the "Load demo scenario" panel;
  a real PDF report downloaded for each (`report_A.pdf` 261 KB, `report_B.pdf`
  768 KB, `report_C.pdf` 78 KB — sizes differ because each scenario embeds
  different real evidence images).
- **VQA-worker-down fallback**: the worker was never started in this
  rehearsal, so the single-image VQA query exercised the real fallback
  path live: the UI showed a `WARNINGS` banner ("VQA worker unavailable;
  answer is an honest unavailability notice, not a model response"),
  `CONFIDENCE: Unavailable`, and an answer text explicitly prefixed
  `[VQA unavailable]` — never a fabricated caption.
- **No-internet-dependency**: verified by code review rather than
  physically disconnecting Wi-Fi (see `docs/FINAL_CHECKLIST.md` for why).
  Confirmed: the three deterministic specialists (grounding, change
  detection, cross-modal) make zero outbound network calls; the backend
  only ever calls the VQA worker on `127.0.0.1`; the VQA worker's own
  `from_pretrained(MODEL_DIR)` call is internet-free when `MODEL_DIR`
  points at a local snapshot (as the README's "no-download setup" note
  already recommends) and otherwise downloads once from Hugging Face Hub
  and caches locally for every run after.
- Tagged `sih-final` on the commit that includes this rehearsal's fix and
  documentation.
