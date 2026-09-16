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

## Final (filled in at Phase A9/C10)

_Not yet recorded._
