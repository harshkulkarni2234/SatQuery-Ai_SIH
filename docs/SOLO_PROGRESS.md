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
| A0 — Baseline freeze + shared contracts | done | `5b5bf3a` | Tag `baseline-round2` on `f1f7513`. `AGENTS.md`, `docs/CONTRACTS.md`, `docs/BASELINE.md`, `backend/app/contracts.py`, `backend/tests/test_contracts.py` added. 111 backend tests pass, frontend builds clean. Had to rebuild `backend/venv_mac/` (mac venv; committed `venv/` is Windows) and reinstall `frontend/node_modules` (missing `@rollup/rollup-darwin-arm64` optional dep) — see docs/BASELINE.md. |

## Wave 1

| Phase | Status | Commit | Notes |
|---|---|---|---|
| A1 — Real GeoTIFF/TIFF raster ingestion | done | (pending) | `backend/app/services/raster_ingest.py`: `extract_metadata()` (GeoTIFF via rasterio incl. bounds_wgs84 reprojection, TIFF-tag acquisition date with user-date override, PNG/JPG/BMP via Pillow, JP2 attempted via rasterio with graceful degradation, MAX_RASTER_PIXELS guard, magic-byte validation via actual rasterio/Pillow open not just extension) and `load_rgb_preview()` (percentile stretch for 16-bit, band 4/3/2 selection for band_count>=4, downsampling). 14 new tests, all synthetic (rasterio in-memory-style tmp files) + verified manually against a realistic synthetic 4-band uint16 GeoTIFF. rasterio added to requirements.txt (backend stays torch-free). DB untouched (that's A2). |
| A2 — Persist metadata + expose in API (GATE G1) | done | (pending) | Alembic migration `93227407ec1b` adds width/height/band_count/dtype/bounds/bounds_wgs84/transform/nodata/file_format/is_georeferenced/acquisition_date_source/file_size_bytes/metadata_warnings to `images`, widens `crs` to String(255). `models.py`, `schemas.py` (`ImageUploadResponse.metadata`, new `ImageDetailResponse`), `routes/images.py` (calls `extract_metadata` on upload, deletes file + returns 400 on `RasterIngestError`, new `GET /images/{id}`) updated. 4 new tests (`test_images_metadata.py`): real GeoTIFF upload returns full metadata in response+DB, corrupt upload -> 400 + no row, GET works, unknown id -> 404. 129 backend tests pass. **GATE G1 reached** — real image metadata now in the API.|
| B1 — Genuine demo & evaluation data | todo | | |
| B2 — Reproducible BigEarthNet LoRA adaptation | todo | | |
| C1 — Split App.jsx into components | done | (pending) | `App.jsx` 1455 → 210 lines. New structure: `components/icons/index.jsx` (all Icon* + `ic()` helper), `components/input/{ImageCard,InputWorkspace}.jsx`, `components/agent/{AgentSelection,AnalysisReady}.jsx`, `components/results/{GroundingImage,VQAResult,GroundingResult,ChangeResult,CrossModalResult,ResultContent,AnalysisResult,TechnicalDetails}.jsx`, `components/common/{Stat,StatGrid,ToggleChip,ViewerToggles,OverlayImage,ConfidenceMeta}.jsx`, `constants/{specialists,scenarios}.js`, `lib/format.js`. Zero markup/logic changes — verified: `npm run build` 28→50 modules, output bundle byte-identical size (173.90kB→173.91kB), zero warnings; clicked through all 4 screens live in the browser (input → drag-drop upload → preset query → analyze → orbit animation → ready → full result report with visual evidence, answer, confidence, technical details → back to input), no console errors. Added `.claude/launch.json` (frontend dev server) to make future browser verification easy. |
| C2 — Real metadata display on upload | done | (pending) | Since G1 was already merged when this phase started, went straight to real data (mocks still added for offline dev: `src/mocks/metadata.js` + `VITE_USE_MOCKS`). `api.js`: `uploadImage` sends optional `capture_date`, added `getImage()`. Changed upload timing: images now upload as soon as they're added (not deferred to Analyze) so metadata shows immediately; modality/date changes re-upload (no update/delete endpoint exists yet, so the old row is orphaned — documented limitation). `ImageCard` shows format/dimensions/bands/capture-date+source/georeferenced/CRS/resolution/extent(WGS84)/warnings, "Not available" for missing values, and per-card upload errors. Added 2-image "pair summary" (informational only). Analyze button disabled while uploading. Found and fixed a real bug while testing live: `raster_ingest.py`'s corrupt-file error messages leaked the full server-side file path (GDAL/PIL exception text) to the client — sanitized both messages, added regression tests. 130 backend tests pass; verified live in the browser (drag-dropped a real synthetic GeoTIFF through the dev server, saw full real metadata; drag-dropped a corrupt file, saw the clean sanitized error). |

## Wave 2

| Phase | Status | Commit | Notes |
|---|---|---|---|
| A3 — Geospatial compatibility engine | done | (pending) | `backend/app/services/compatibility.py`: `check_temporal_pair` (modality match, date-distinct with WARN not FAIL on missing dates, CRS match, bbox-IoU overlap vs `MIN_OVERLAP` env default 0.5, resolution-ratio WARN >1.5x, SKIPPED geo checks when not georeferenced), `check_optical_sar_pair` (same checks + `coregistration`: "verified" only when same CRS and matching transform grid, "assumed" behind an opt-in `coregistered_hint` param not yet wired to an upload field — documented gap, "unverified" otherwise), `check_single_image`. Wired into `routes/query.py` before dispatching CHANGE_DETECTION/CROSS_MODAL: FAIL -> 422 with `{message, compatibility}` body; the pre-existing `_temporal_metadata_compatible` same-date short-circuit in `services/router.py` is untouched (still runs first, during classification). 14 unit tests on synthetic dicts (same-date, non-overlapping, different CRS, wrong modality, non-georeferenced, verified co-registered) + 3 new API-level integration tests through real GeoTIFF uploads (non-overlapping pair -> 422 with report; overlapping pair -> 200; optical+SAR overlapping -> 200). 147 backend tests pass. |
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
