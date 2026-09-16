# Solo build progress

This repo's build was originally scoped as a 3-person parallel plan
(`SatQuery_Team_Prompts.md`, roles A/B/C, waves 1-4 + final). It is being
executed solo, one phase at a time, in wave order, resuming here whenever a
session restarts. Each row is updated the moment a phase's acceptance
criteria are met and its commit lands.

Legend: `done` / `in_progress` / `blocked` / `todo`

## Day 0

| Phase | Status | Commit | Notes |
|---|---|---|---|
| A0 — Baseline freeze + shared contracts | done | (pending commit) | Tag `baseline-round2` on `f1f7513`. `AGENTS.md`, `docs/CONTRACTS.md`, `docs/BASELINE.md`, `backend/app/contracts.py`, `backend/tests/test_contracts.py` added. 111 backend tests pass, frontend builds clean. Had to rebuild `backend/venv_mac/` (mac venv; committed `venv/` is Windows) and reinstall `frontend/node_modules` (missing `@rollup/rollup-darwin-arm64` optional dep) — see docs/BASELINE.md. |

## Wave 1

| Phase | Status | Commit | Notes |
|---|---|---|---|
| A1 — Real GeoTIFF/TIFF raster ingestion | todo | | |
| A2 — Persist metadata + expose in API (GATE G1) | todo | | |
| B1 — Genuine demo & evaluation data | todo | | |
| B2 — Reproducible BigEarthNet LoRA adaptation | todo | | |
| C1 — Split App.jsx into components | todo | | |
| C2 — Real metadata display on upload | todo | | mock until G1 |

## Wave 2

| Phase | Status | Commit | Notes |
|---|---|---|---|
| A3 — Geospatial compatibility engine | todo | | |
| A4 — Specialist registry (GATE G2) | todo | | |
| B3 — Evaluation framework scaffold | todo | | |
| B4 — RSVQA runner | todo | | |
| B5 — CDVQA runner | todo | | |
| C3 — Honest execution trace timeline | todo | | mock until G3 |
| C4 — Validation & compatibility UX | todo | | |

## Wave 3

| Phase | Status | Commit | Notes |
|---|---|---|---|
| A5 — Agent planner + deterministic validator | todo | | |
| A6 — Event-driven execution trace (GATE G3) | todo | | |
| B6 — VRSBench runner | todo | | |
| B7 — Raster-aware change detection + native SpecialistResult | todo | | WAIT for G1 + G2 |
| B8 — Semantic change specialist | todo | | |
| C5 — Downloadable PDF report (backend) | todo | | WAIT for G2 |
| C6 — Report download + evidence/confidence standard in UI | todo | | |
| C7 — Cross-modal + change result UI upgrades | todo | | |

## Wave 4

| Phase | Status | Commit | Notes |
|---|---|---|---|
| A7 — Integrate B's upgraded specialists | todo | | WAIT for B7, B8, B9 |
| A8 — Backend hardening | todo | | |
| B9 — Optical + SAR co-registered fusion | todo | | |
| B10 — Full benchmark runs + results | todo | | WAIT for A7 |
| C8 — Hardening: API + UI edge cases | todo | | |
| C9 — Demo scenarios, runbook, README | todo | | WAIT for B1 |

## Final

| Phase | Status | Commit | Notes |
|---|---|---|---|
| A9 — Cold-start & final integration | todo | | |
| C10 — Final cold-start rehearsal | todo | | |

## Environment notes (persist across sessions)

- macOS (Darwin, arm64), Python 3.11.7, Node v24.7.0, PostgreSQL 17.11 (Homebrew).
- Backend venv: use `backend/venv_mac/` (created fresh — the committed
  `backend/venv/` is a Windows venv and won't run here). Not committed
  (gitignored via existing venv ignore rule — verify before assuming).
- DB: local Postgres role `shlok`, no password, DB name `satquery`, created
  with `createdb satquery`. `backend/.env` `DATABASE_URL` was changed from
  the original `postgres:iharsh@...` to `postgresql://shlok@localhost:5432/satquery`
  locally (`.env` is gitignored, this edit is not part of any commit).
- Frontend: `frontend/node_modules` was reinstalled from scratch
  (`rm -rf node_modules package-lock.json && npm install`) to fix a missing
  `@rollup/rollup-darwin-arm64` optional dependency. `package-lock.json` was
  regenerated on this platform — check `git diff` on it before assuming it's
  identical to the Windows-generated lockfile.
- Test command: `cd backend && venv_mac/bin/python -m pytest -q`
- Build command: `cd frontend && npm run build`
- No dedicated GPU on this machine for Phase B training work (B2, B8) — see
  notes recorded in each phase's row once reached; will follow the plan's
  honesty rule (no fabricated benchmark numbers) and document real
  hardware/limitations rather than pretending otherwise.

## Deviations from the original plan

- Executed solo instead of 3 parallel people — same file-ownership
  boundaries and wave order are kept for clarity, but there's no daily sync;
  this file is that sync record instead.
- Git workflow simplified: phases are done as small commits directly (still
  following the `feat(<phase-id>): ...` message convention) rather than
  short-lived per-phase branches, since there's no concurrent second author
  to rebase against. Will branch if/when asked to open PRs.
