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

```
React frontend (frontend/, Vite)
        │  HTTP via Vite dev proxy → http://127.0.0.1:8000
        ▼
FastAPI backend (backend/)                 PostgreSQL (metadata)
        │  POST /query
        ▼
Agent / Router  (app/services/router.py — classify_query)
        ▼
Specialist services
  ├── VQA            → local VQA worker (SmolVLM)      [app/services/vqa.py]
  ├── GROUNDING      → deterministic OpenCV/NumPy    [app/services/grounding.py]
  ├── CHANGE_DETECTION→ deterministic OpenCV/NumPy    [app/services/change_detection.py]
  └── CROSS_MODAL    → deterministic OpenCV/NumPy    [app/services/cross_modal.py]
        ▼
Result + visual evidence → QueryResponse → execution trace persisted
```

The backend never imports `torch`; the heavy ML stack lives in a separate VQA worker service to keep the backend dependency-light.

## Specialist implementation details

- **VQA** uses the local **SmolVLM-256M-Instruct** model with the experimental **Stage-3 BigEarthNet LoRA** specialist/fallback architecture. The worker exposes a small HTTP API (`POST /vqa`, `GET /health`) and the backend calls it with a graceful fallback message if it is unreachable.
- **Grounding and change detection** use deterministic computer-vision pipelines (OpenCV + NumPy): HSV thresholding, morphological cleaning, connected-component analysis, and phase-correlation registration for change detection.
- **Optical + SAR** uses deterministic multimodal feature analysis: spectral/colour statistics for optical imagery and backscatter statistics for SAR, reported independently because the two sensors are assumed **not** co-registered unless verified.

## Scientific honesty

- Deterministic computer-vision pipelines are **not** described as — and are not — a trained satellite foundation model. They are rule-based detectors.
- The Stage-3 LoRA adapter is **experimental** and is **not** claimed to be universally better than the base model; it is applied only to narrow supported question types and falls back to the base model on failure.
- A local **GeoChat feasibility study** (see `notebooks/geochat_feasibility_test.ipynb`) found the full GeoChat model unsuitable for the available 4 GB T600 GPU, which is why the system uses the much smaller SmolVLM base.

## Setup (Windows)

Prerequisites: Python 3.11+, Node.js 18+, PostgreSQL 14+, a CUDA GPU (rated ~4 GB VRAM for the VQA worker; CPU works but is slow).

### 1. Backend

```powershell
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Create the database and configure backend\.env
# (copy from .env.example — see "Required environment variables" below)
alembic upgrade head
```

Start the backend:

```powershell
venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

### 2. VQA worker (separate environment)

The worker uses **its own** virtual environment, never the backend one.

```powershell
cd ml\vqa-worker
python -m venv venv
venv\Scripts\activate

# GPU (tested build) — install the CUDA torch wheel first:
pip install torch==2.14.0+cu130 --index-url https://download.pytorch.org/whl/cu130
# then:
pip install -r requirements.vqa-worker.txt
```

Start the worker (defaults to `http://127.0.0.1:8001`):

```powershell
.\run_worker.ps1
```

`run_worker.ps1` honours `VQA_WORKER_PYTHON` (or falls back to `ml\vqa-worker\venv`) and the `MODEL_DIR` / `ADAPTER_DIR` variables. For a no-download setup, point `MODEL_DIR` at the tested local SmolVLM snapshot (see `ml\vqa-worker\.env.example`).

### 3. Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:5173.

## Required environment variables

`backend\.env` (copy `backend\.env.example`):

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | PostgreSQL connection string, e.g. `postgresql://USER:PASS@localhost:5432/satquery` |
| `VQA_WORKER_URL` | VQA worker endpoint, e.g. `http://127.0.0.1:8001` |
| `VQA_WORKER_TIMEOUT_S` | Backend→worker request timeout in seconds (default shown in example) |

`ml\vqa-worker\.env` (copy `ml\vqa-worker\.env.example` — non-secret):

| Variable | Purpose |
| --- | --- |
| `VQA_WORKER_PORT` | Worker HTTP port (default `8001`) |
| `MODEL_DIR` | SmolVLM base model: HF repo id or tested local snapshot path |
| `ADAPTER_DIR` | Experimental Stage-3 LoRA adapter directory |
| `VQA_WORKER_PYTHON` | Optional absolute path to a python.exe with worker deps installed |

## API endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Backend liveness |
| POST | `/images/upload` | Multipart `file` + `modality` (OPTICAL/SAR); optional `capture_date` |
| POST | `/query` | JSON `{query_text, image_ids}` → classified task + `QueryResponse` |
| GET | `/query/{query_id}` | Retrieve a stored `QueryResponse` (overlay URLs rebuilt) |
| GET | `/masks/{filename}` | Served change masks / change overlays (static) |
| GET | `/docs` | Swagger UI |

`capture_date` is optional and never fabricated by the frontend; it is stored as `NULL` when not supplied and is used only as internal metadata for temporal-compatibility checks.

## Tests and build

Backend tests:

```powershell
cd backend
venv\Scripts\python.exe -m pytest -q
```

Frontend production build:

```powershell
cd frontend
npm run build      # outputs to frontend/dist
```

## Demo data and limitations

The repository ships no large demo imagery; you can upload your own `.png` / `.jpg` / `.tif` files. Known limitations are stated honestly:

- Change detection measures **pixel-level visual differences**, not semantic land-cover change (no autonomous "building constructed" claims).
- The demo data does **not** contain a genuine same-area before/after pair; synthetic/controlled change examples are used to demonstrate the change pipeline where applicable.
- Optical ↔ SAR **spatial correspondence is never claimed** unless verified; the cross-modal result reports per-sensor regions and states this explicitly.
- Display-region caps are enforced for readability (up to 10 grounding boxes; up to 8 per-sensor cross-modal regions).
- Confidence is reported only when meaningful (e.g. bounding-box fill ratio); otherwise it is shown as **unavailable**.
- Without the VQA worker running, VQA returns an honest "worker unavailable" message while the deterministic workflows still work.

## Demo workflow

1. **Visual QA** — upload one OPTICAL image, then ask *"What can you tell me about this image?"* or *"Is there water in this image?"*. Object-specific questions also show grounding evidence when it exists.
2. **Grounding** — upload one image and ask *"Where is the water located?"*, *"Show me the roads"*, etc. Labels are e.g. *Water Body 1*, *Building 1*, *Road 1*.
3. **Change detection** — upload two same-area images (ideally with different capture dates) and ask *"Show me the changes between these two images."*. The result shows a change overlay, changed-region boxes, and a change mask. Non-comparable pairs are refused with an honest message rather than a fabricated percentage.
4. **Cross-modal** — upload one OPTICAL and one SAR image and ask *"Compare the optical and SAR images."*. Evidence is shown per sensor.

## Project structure

```
satquery-ai/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app, static /masks mount
│   │   ├── routes/                 # /images, /query REST endpoints
│   │   ├── services/               # router + specialist services
│   │   └── database.py, models.py, schemas.py
│   ├── alembic/                    # DB migrations
│   ├── tests/                      # backend test suite
│   ├── requirements.txt
│   └── .env.example
├── ml/
│   ├── vqa-worker/                 # separate SmolVLM worker service
│   │   ├── worker_service.py, model_provider.py
│   │   ├── run_worker.ps1
│   │   └── requirements.vqa-worker.txt
│   └── smolvlm/lora_stage3/        # experimental BigEarthNet LoRA adapter
├── frontend/                       # React (Vite) single-page app
│   ├── src/                        # App.jsx, api.js, App.css
│   ├── package.json
│   └── vite.config.js
├── notebooks/                      # GeoChat feasibility study
├── docs/                           # master build specification
└── data/                           # runtime artifacts (git-ignored)
```

## License

For SIH 2026 prototype/demo purposes. No license has been selected yet.