# AGENTS.md — SatQuery AI

## Project summary

SatQuery AI is a multimodal remote-sensing vision-language system built for
SIH 2026, problem statement SIH26167. Repo layout:

- `backend/` — FastAPI + SQLAlchemy + PostgreSQL + Alembic. Router in
  `app/services/router.py`, specialists in
  `app/services/{vqa,grounding,change_detection,cross_modal}.py`, routes in
  `app/routes/{images,query}.py`. **Backend must stay torch-free.**
- `ml/vqa-worker/` — separate SmolVLM-256M-Instruct worker process
  (+ experimental Stage-3 BigEarthNet LoRA adapter).
- `frontend/` — React + Vite (`src/App.jsx`, `src/api.js`).
- `docs/` — build specs, contracts, evaluation, demo runbook.

Read `docs/CONTRACTS.md` before writing code that crosses the
frontend/backend/model boundary.

Dev commands (macOS in this build; original dev machine was Windows/PowerShell —
adjust venv activation only, everything else is the same):

```
cd backend && venv_mac/bin/python -m pytest -q
cd frontend && npm run build
```

## Non-negotiable rules

1. Do NOT rewrite the project from scratch. Extend what exists.
2. Never fabricate confidence, accuracy, benchmark scores, bounding boxes or
   percentages. If a value can't be justified: `confidence=null`,
   `confidence_source="unavailable"`.
3. Never claim deterministic OpenCV pipelines are trained models; never claim
   pixel difference = semantic change; never claim optical-SAR co-registration
   unless verified; never claim GeoChat runs locally; never claim the LoRA
   universally beats base SmolVLM.
4. UI must never imply a model executed unless the backend says it did.
5. Keep existing tests passing. Add tests for everything new.
6. Follow the shared contracts in `backend/app/contracts.py` and
   `docs/CONTRACTS.md` exactly. Don't change a contract silently.
7. At the end of each phase: run tests/build, summarize files changed, and
   commit with `feat(<phase-id>): ...` / `fix(<phase-id>): ...`.

## Roles (this build is executed solo, in role order — see docs/SOLO_PROGRESS.md)

This project was originally scoped for three parallel contributors. It is
being executed solo, one phase at a time, in the wave order below, but the
same file-ownership boundaries are kept so the history stays legible and
future contributors could still pick up any one role.

- **A — Backend & Geospatial Core** (also integrator): owns shared files.
- **B — ML, Specialists & Evaluation**.
- **C — Frontend, Report & Demo/QA**.

## File ownership

| Owner | Files / folders |
|---|---|
| A | `backend/app/models.py`, `schemas.py`, `contracts.py`, `main.py`, `database.py`, `routes/images.py`, `routes/query.py`, `services/router.py`, `services/raster_ingest.py`, `services/compatibility.py`, `services/registry.py`, `services/planner.py`, `services/trace.py`, `alembic/`, `backend/requirements.txt`, `AGENTS.md` |
| B | `backend/app/services/vqa.py`, `grounding.py`, `change_detection.py`, `cross_modal.py`, `services/specialists/`, `ml/` (all), `evaluation/`, `notebooks/`, `data/demo/` sourcing |
| C | `frontend/` (all), `backend/app/services/report.py`, `backend/app/routes/report.py`, `backend/tests/test_hardening_*.py`, `docs/DEMO_RUNBOOK.md`, `scripts/`, `README.md` (final pass) |

Tests: each phase's own area gets its tests in `backend/tests/test_<area>.py`.
Never delete another phase's tests.

## Git rules

1. Before starting any phase: `git checkout master && git pull --rebase`
   (solo build: skip if no remote changes since last phase).
2. Work on a short branch per phase, e.g. `a/p1-raster-ingestion`,
   `b/p4-rsvqa`, `c/p3-trace-ui`, merge same day.
3. Before merging: rebase on latest `master`, run backend tests + frontend
   build, then merge and (if requested) push.
4. Only Alembic migrations touch `alembic/versions/` — one linear history.
5. Commit style: `feat(A3): geospatial compatibility engine`,
   `fix(C4): ...`.

## Progress tracking

See [docs/SOLO_PROGRESS.md](docs/SOLO_PROGRESS.md) for phase-by-phase status,
resumed automatically across sessions.
