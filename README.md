# SatQuery AI

**SatQuery AI** is a query-driven satellite-image analysis assistant built for **SIH 2026 Problem Statement 26167** — Multimodal Remote Sensing (Satellite) Vision-Language Model for Natural-Language Interaction and Intelligent Analysis.

Users upload one or two satellite images, ask a question in natural language, and receive an answer from a task-appropriate **specialist**, backed by visual evidence, honest confidence handling, and an auditable execution trace. Every query is routed through an EO-aware agent instead of being answered by a single generic model.

## Overview

SatQuery AI treats a satellite-analysis question as a *capability-routing* problem rather than a text-generation problem. A lightweight router inspects the natural-language intent and the uploaded image combination, validates the input, and dispatches to the specialist best able to answer with **localised visual evidence** — a local vision-language model for visual QA, and deterministic computer-vision pipelines for spatial grounding, bi-temporal change detection, and optical + SAR cross-modal analysis. Because specialists are deterministic where the evidence is geometric ("where did the pixels differ?"), the system can be scientifically honest: it reports what it actually detected, shows the detected regions on the image, and refuses to invent percentages or boxes when spatial correspondence cannot be verified.

## What makes our approach different

- **EO-specific agentic orchestration** — a query/router → specialist architecture tailored to remote-sensing tasks, not a single general-purpose chat loop.
- **Query and input validation** — modality enforcement (OPTICAL/SAR), supported-target checking, and graceful rejection of ambiguous inputs before any analysis runs.
- **Capability-aware specialist selection** — the router picks the specialist whose outputs can actually be evidenced for the given query and image pair (priority: CROSS_MODAL → CHANGE_DETECTION → GROUNDING → VQA).
- **Spatial / temporal compatibility checks** — change detection refuses to report a change percentage for images whose geographic footprint or capture dates provably do not match.
- **Evidence-first answers** — bounding boxes, change masks, overlays, and per-region statistics are real detections shown on the imagery, never fabricated for the UI.
- **Confidence handling** — confidence is only reported when a meaningful quantity exists (e.g. bounding-box fill ratio); otherwise it is explicitly shown as unavailable.
- **Auditable execution trace** — every response records the selected tool, model version, detected modalities, routing reason, execution time, and completion status.

## Supported workflows

1. **Single-image VQA** — natural-language questions (including object-specific ones) answered by a local vision-language model, optionally augmented by deterministic grounding evidence.
2. **Single-image spatial grounding** — locate water, vegetation, built-up areas, roads, or farmland with deterministic region boxes.
3. **Bi-temporal change analysis** — pixel-level change regions, change masks, and overlays between two same-area images.
4. **Optical + SAR cross-modal analysis** — complementary spectral and radar evidence reported independently per sensor.

## Architecture

This reflects what actually ships today, not the original plan document —
see `docs/SOLO_PROGRESS.md` for the phase-by-phase build log and any gaps.

```
React frontend (frontend/, Vite)
        │  HTTP via Vite dev proxy → http://127.0.0.1:8000
        ▼
FastAPI backend (backend/)                          PostgreSQL (metadata + trace)
        │  POST /query
        ▼
Agent Planner (app/services/planner.py — build_plan)
  ├─ classify_query()           intent + task classification (app/services/router.py)
  ├─ compatibility check        modality/date/overlap/coregistration (app/services/compatibility.py)
  ├─ registry.select()          specialist selection with honest fallback (app/services/registry.py)
  └─ ExecutionPlan              validated task + parameters + rejection reasons (app/contracts.py)
        ▼
Specialist registry (6 declared specialists; dispatch via app/services/registry_adapters.py)
  ├── vqa.smolvlm_base                    → local VQA worker (SmolVLM)         [app/services/vqa.py]
  ├── vqa.smolvlm_bigearthnet_lora_stage3 → experimental LoRA (worker-internal routing)
  ├── grounding.deterministic_cv          → deterministic OpenCV/NumPy         [app/services/grounding.py]
  ├── change.semantic_model               → NOT IMPLEMENTED (always unavailable — no GPU; see Limitations)
  ├── change.deterministic_cv             → deterministic OpenCV/NumPy + real geo-reprojection [app/services/change_detection.py]
  └── cross_modal.feature_fusion          → deterministic fusion + coregistration-aware confidence [app/services/cross_modal.py]
        ▼
TraceRecorder (app/services/trace.py) — every step timed with real perf_counter(), persisted even on failure
        ▼
QueryResponse (answer + evidence + confidence + trace_events + compatibility + warnings)
        ▼
Report builder (app/services/report.py) → GET /query/{id}/report.pdf | report.json
```

The backend never imports `torch`; the heavy ML stack lives in a separate VQA worker service to keep the backend dependency-light.

## Specialist implementation details

- **VQA** uses the local **SmolVLM-256M-Instruct** model with the experimental **Stage-3 BigEarthNet LoRA** specialist/fallback architecture. The worker exposes a small HTTP API (`POST /vqa`, `GET /health`) and the backend calls it with a graceful fallback message if it is unreachable.
- **Grounding** uses a deterministic computer-vision pipeline (OpenCV + NumPy): HSV thresholding, morphological cleaning, connected-component analysis.
- **Change detection** uses real geographic reprojection (`rasterio.warp.reproject`) when both images are georeferenced, falling back to phase-correlation registration otherwise; reports real changed area in m² from actual pixel resolution (never estimated), and always states its real alignment method.
- **Optical + SAR fusion** uses deterministic multimodal feature analysis (speckle-filtered SAR backscatter/VV-VH stats, optical colour/NDVI-NDWI stats where bands allow), with per-class (water/vegetation/built-up) attribution reported as `both`/`optical_only`/`sar_only`/`none` — never claims combined evidence unless both sensors actually produced a real signal. Pixel-level agreement confidence is only computed when `coregistration` is independently verified or assumed from real file metadata — never guessed.
- **Semantic (learned-model) change detection** is a declared-but-unimplemented specialist (`change.semantic_model`): the registry always reports it honestly as unavailable rather than silently omitting it, since no GPU was available to train/host it (see Limitations).

## Scientific honesty

- Deterministic computer-vision pipelines are **not** described as — and are not — a trained satellite foundation model. They are rule-based detectors.
- The Stage-3 LoRA adapter is **experimental** and is **not** claimed to be universally better than the base model; it is applied only to narrow supported question types and falls back to the base model on failure.
- A local **GeoChat feasibility study** (see `notebooks/geochat_feasibility_test.ipynb`) found the full GeoChat model unsuitable for the available 4 GB T600 GPU, which is why the system uses the much smaller SmolVLM base.

## Setup

Prerequisites: Python 3.11+, Node.js 18+, PostgreSQL 14+. A CUDA GPU is
recommended for the VQA worker (~4 GB VRAM) but **not required** — this
build was developed and tested entirely on a GPU-less machine (macOS,
CPU-only); the worker runs on CPU, just slower, and every deterministic
specialist (grounding/change/cross-modal) has no GPU dependency at all.

Commands below are shown for both platforms; `scripts/start_all.sh` /
`scripts/start_all.ps1` automate steps 1–3 together (see
`docs/DEMO_RUNBOOK.md`). **Only the `.sh` scripts and macOS commands were
actually run and verified in this build's development** — the `.ps1`
scripts and Windows commands mirror them step-for-step but have not been
executed on a real Windows machine; smoke-test them there before relying
on them.

### 1. Backend

macOS/Linux:
```bash
cd backend
python3 -m venv venv_mac      # any venv name; "venv_mac" avoids clashing with a committed Windows venv/
source venv_mac/bin/activate
pip install -r requirements.txt

# Create the database and configure backend/.env (copy from .env.example —
# see "Required environment variables" below), then:
alembic upgrade head
```
Windows:
```powershell
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
alembic upgrade head
```

Start the backend:
```bash
venv_mac/bin/uvicorn app.main:app --reload --port 8000      # macOS/Linux
venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000   # Windows
```

### 2. VQA worker (separate environment, optional)

The worker uses **its own** virtual environment, never the backend one. The
system runs and demos correctly without it — VQA queries return an honest
"worker unavailable" fallback answer instead of erroring.

```bash
cd ml/vqa-worker
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.vqa-worker.txt   # CPU works; see file for the CUDA torch wheel line if you have a GPU
python -m uvicorn worker_service:app --host 127.0.0.1 --port 8001
```

Windows can instead run `.\run_worker.ps1`, which honours
`VQA_WORKER_PYTHON` (or falls back to `ml\vqa-worker\venv`) and the
`MODEL_DIR` / `ADAPTER_DIR` variables — see `ml/vqa-worker/.env.example`.

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The **"Load demo scenario"** panel (real B1
demo files, uploaded through the normal API — see `docs/DEMO_RUNBOOK.md`)
appears automatically in dev mode.

## Required environment variables

`backend/.env` (copy `backend/.env.example`):

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | PostgreSQL connection string, e.g. `postgresql://USER:PASS@localhost:5432/satquery` |
| `VQA_WORKER_URL` | VQA worker endpoint, e.g. `http://127.0.0.1:8001` |
| `VQA_WORKER_TIMEOUT_S` | Backend→worker request timeout in seconds (default shown in example) |
| `MAX_UPLOAD_SIZE_BYTES` | Upload rejection threshold in bytes (default 200 MB); enforced via chunked reads, never buffers the whole file first |
| `MAX_RASTER_PIXELS` | Pixel-count warning threshold (default ~100 MP); metadata is still extracted, pixel data is not read past this |
| `MIN_OVERLAP` | Minimum bounding-box IoU for a temporal/cross-modal pair to be considered spatially compatible (default `0.5`) |

`ml/vqa-worker/.env` (copy `ml/vqa-worker/.env.example` — non-secret):

| Variable | Purpose |
| --- | --- |
| `VQA_WORKER_PORT` | Worker HTTP port (default `8001`) |
| `MODEL_DIR` | SmolVLM base model: HF repo id or tested local snapshot path |
| `ADAPTER_DIR` | Experimental Stage-3 LoRA adapter directory |
| `VQA_WORKER_PYTHON` | Optional absolute path to a python.exe with worker deps installed |

## API endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Reports `database`, `vqa_worker`, and `semantic_change_model` availability separately (not just a flat "ok") |
| POST | `/images/upload` | Multipart `file` + `modality` (OPTICAL/SAR); optional `capture_date`. Enforces `MAX_UPLOAD_SIZE_BYTES` (413 on overflow) and `MAX_RASTER_PIXELS` (warns, doesn't reject) |
| GET | `/images/{image_id}` | Real extracted metadata for a stored image (CRS, bounds, resolution, bands, warnings) |
| POST | `/query` | JSON `{query_text, image_ids}` (1–2 ids) → validated plan + dispatched specialist → `QueryResponse` with `trace_events`, `compatibility`, `confidence_source`, `warnings`, `used_fallback` |
| GET | `/query/{query_id}` | Retrieve a stored `QueryResponse` (overlay URLs rebuilt) |
| GET | `/specialists` | Lists all declared specialists (id, kind, priority, availability, RS-adapted flag) — drives the frontend's specialist badge |
| GET | `/query/{query_id}/report.pdf` | Downloadable PDF report reconstructed from stored data only (never re-runs the specialist) |
| GET | `/query/{query_id}/report.json` | Same report data as JSON — guaranteed to match the PDF exactly (same builder function) |
| GET | `/masks/{filename}` | Served change masks / change overlays (static) |
| GET | `/demo-assets/...` | Static mount of the repo-root `data/` directory (demo scenario files + manifest) — used only by the frontend's "Load demo scenario" feature to fetch real bytes for a normal upload; never serves user-uploaded data |
| GET | `/docs` | Swagger UI |

`capture_date` is optional and never fabricated by the frontend; it is stored as `NULL` when not supplied and is used only as internal metadata for temporal-compatibility checks. Every error response returns a client-safe message (`{detail}` or `{detail: {message, suggestion, compatibility?}}`) — an unhandled server error returns a generic `{"detail": "Internal server error"}` with full detail logged server-side, never a raw traceback (see `docs/HARDENING_REPORT.md`).

## Tests and build

Backend tests (226 passing as of this writing — run it yourself for the
current number, don't trust a stale count):

```bash
cd backend
venv_mac/bin/python -m pytest -q      # Windows: venv\Scripts\python.exe -m pytest -q
```

Frontend production build:

```bash
cd frontend
npm run build      # outputs to frontend/dist
```

Or run `scripts/smoke_test.sh` (`.ps1` on Windows) against a running
backend to exercise the real API end-to-end: health check, upload the 3
real demo scenarios, run the 4 demo queries, download one report, PASS/FAIL.

## Demo data and limitations

Real demo scenario files ship in `data/demo/` (Phase B1 — see
`data/manifest.json` for exact source/license/citation per file, and each
scenario's own `README.md` for honesty caveats specific to that file). The
frontend's "Load demo scenario" panel uploads these through the normal API;
you can also upload your own `.png` / `.jpg` / `.tif` files.

Known limitations, stated honestly rather than hidden:

- Change detection measures **pixel-level visual differences**, not semantic land-cover change (no autonomous "building constructed" claims). It never estimates changed area when pixel resolution is unknown.
- No GPU was available during development (see `notebooks/geochat_feasibility_test.ipynb` for why the much smaller SmolVLM was chosen over GeoChat). As a direct consequence:
  - **The semantic (learned-model) change-detection specialist (`change.semantic_model`) was never implemented** — the deterministic CV specialist always runs instead, and the registry reports this honestly via `used_fallback` rather than silently substituting one for the other.
  - **The BigEarthNet LoRA fine-tuning run (Phase B2) and the full RSVQA/CDVQA/VRSBench benchmark evaluation (Phases B3–B6, B10) were not completed.** There is no `docs/EVALUATION.md` in this build — do not reference one or invent numbers; see `docs/SOLO_PROGRESS.md` for the exact scope decision and status of every phase.
- Optical ↔ SAR **spatial correspondence is never claimed** unless verified from real file metadata (matching CRS + transform); the cross-modal result reports per-sensor/per-class attribution and states honestly when correspondence is unverified, even for a pair that is genuinely co-registered by construction but ships without embedded georeferencing (see `data/demo/scenario_C_optical_sar/README.md`).
- Display-region caps are enforced for readability (up to 10 grounding boxes; up to 8 per-sensor cross-modal regions).
- Confidence is reported only when meaningful (e.g. bounding-box fill ratio, or pixel-mask agreement under verified coregistration); otherwise it is shown as **unavailable**.
- Without the VQA worker running, VQA returns an honest "worker unavailable" fallback message while every deterministic workflow still works normally.
- A systematic API + UI edge-case hardening pass (Phase C8, see `docs/HARDENING_REPORT.md`) found and fixed a missing upload size limit and an inconsistent huge-raster warning; no further issues were found in that pass.

## Demo workflow

See `docs/DEMO_RUNBOOK.md` for the full startup order, click-paths, what
to say to judges about each honest limitation, and a live backup-plan
demonstration (stopping the VQA worker mid-demo to show the graceful
fallback). Summary:

1. **Visual QA** — upload one OPTICAL image, then ask *"What can you tell me about this image?"* or *"Is there water in this image?"*. Object-specific questions also show grounding evidence when it exists.
2. **Grounding** — upload one image and ask *"Where is the water located?"*, *"Show me the roads"*, etc. Labels are e.g. *Water Body 1*, *Building 1*, *Road 1*.
3. **Change detection** — upload two same-area images (ideally with different capture dates) and ask *"What changed between these two dates?"*. The result shows a change overlay, changed-region boxes, a change mask, real changed area in m², and the real alignment method used. Non-comparable pairs are refused with an honest message and a real compatibility report rather than a fabricated percentage.
4. **Cross-modal** — upload one OPTICAL and one SAR image and ask *"Use the optical and SAR images together to identify built-up and water-covered regions."*. Evidence is shown per sensor with a combined/optical-only/SAR-only attribution legend.

All four are one click away via "Load demo scenario" using real B1 data —
no manual file hunting needed for a live demo.

## Project structure

```
satquery-ai/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app, global exception handler, /health, static mounts
│   │   ├── contracts.py            # shared pydantic contracts (frontend/backend boundary)
│   │   ├── routes/                 # /images, /query, /specialists, /query/{id}/report.* REST endpoints
│   │   ├── services/                # planner, router, registry(+adapters), compatibility, trace, report, specialists
│   │   └── database.py, models.py, schemas.py
│   ├── alembic/                    # DB migrations
│   ├── tests/                      # backend test suite (226 tests)
│   ├── requirements.txt
│   └── .env.example
├── ml/
│   ├── vqa-worker/                 # separate SmolVLM worker service
│   │   ├── worker_service.py, model_provider.py
│   │   ├── run_worker.ps1
│   │   └── requirements.vqa-worker.txt
│   ├── smolvlm/lora_stage3/        # experimental BigEarthNet LoRA adapter
│   └── data/                       # demo-data fetch scripts (e.g. download_lakemead_temporal_pair.py)
├── frontend/                       # React (Vite) single-page app
│   ├── src/                        # App.jsx, api.js, components/, constants/, lib/
│   ├── package.json
│   └── vite.config.js
├── notebooks/                      # GeoChat feasibility study
├── scripts/                        # start_all / smoke_test (.sh verified, .ps1 mirrors — see docs/DEMO_RUNBOOK.md)
├── docs/
│   ├── satquery-master-build-spec.md  # original 3-person plan this build follows solo
│   ├── SOLO_PROGRESS.md               # phase-by-phase build log — what's done, deferred, or blocked, and why
│   ├── CONTRACTS.md                   # narrative contract docs with JSON examples
│   ├── HARDENING_REPORT.md            # edge-case checklist: case → expected → actual → pass/fail
│   ├── DEMO_RUNBOOK.md                # startup order, click-paths, judge talking points, backup plan
│   └── BASELINE.md                    # environment/versions recorded before any changes
└── data/                            # data/demo/ (real, source-controlled demo scenarios) +
                                      # data/manifest.json; backend/data/ (uploaded images/masks,
                                      # git-ignored) is a SEPARATE directory, never served statically
```

## Honest project status

This was built solo against a plan originally scoped for 3 parallel
contributors (Person A: backend/geospatial, Person B: ML/eval, Person C:
frontend/report/demo) — see `docs/satquery-master-build-spec.md` for the
original plan and `docs/SOLO_PROGRESS.md` for exactly which phases are
done, deferred, or blocked, and the real reason for each (mainly: no GPU on
the development machine, and large third-party dataset downloads not yet
authorized). Nothing above claims work that file doesn't corroborate.

## License

For SIH 2026 prototype/demo purposes. No license has been selected yet.