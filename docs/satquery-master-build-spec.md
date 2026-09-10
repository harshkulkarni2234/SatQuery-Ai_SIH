# SatQuery AI — Master Build Specification
### For autonomous execution by OpenCode (or any agentic coding tool)

Paste this entire document as your initial instruction. Work through the
phases in order — each phase has explicit acceptance criteria; do not move
to the next phase until the current one's criteria are met. Commit to git
after each phase completes successfully.

---

## 0. Project Overview

Build **SatQuery AI**: a query-driven assistant where a user uploads
satellite image(s), asks a question in plain language, and receives an
answer backed by visual evidence, a confidence score, and a visible
execution trace showing which analysis tool was used and why.

The core value is **not** any individual AI model's quality — it's the
**agent that decides which specialist tool applies to a given question and
input combination, validates that the combination is even valid, and makes
that decision auditable**. Every component below should be built with that
priority: correctness and explainability of the routing decision matter
more than sophistication of any single model.

The system must support three end-to-end demo scenarios:
1. **Single-image VQA + grounding**: "What's visible here?" / "Show me the water body"
2. **Bi-temporal change detection**: two images, different dates → "What changed?"
3. **Optical + SAR cross-modal fusion**: two images, different modalities → combined analysis, with graceful handling when optical is cloud-obscured

---

## 1. Tech Stack

- **Backend**: FastAPI (Python), SQLAlchemy, PostgreSQL
- **Frontend**: React, plain CSS or Tailwind (no heavy component library needed)
- **Image processing**: OpenCV, NumPy, Pillow
- **VQA**: either MBZUAI/geochat-7B via HuggingFace transformers (if GPU-viable) or a hosted VLM API with remote-sensing-adapted prompting (fallback) — build behind a swappable interface so either works
- **Deployment target**: local (single startup script) with optional Vercel/Netlify (frontend) + Render/Railway (backend) free-tier deployment

---

## 2. Repository Structure

Create this structure first, commit it empty/stubbed before writing any logic:

```
satquery-ai/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── models.py
│   │   ├── routes/
│   │   │   ├── images.py
│   │   │   └── query.py
│   │   └── services/
│   │       ├── router.py
│   │       ├── change_detection.py
│   │       ├── grounding.py
│   │       ├── vqa.py
│   │       └── cross_modal.py
│   └── requirements.txt
├── frontend/
│   └── (React app)
├── notebooks/
│   └── geochat_feasibility_test.ipynb
├── data/
│   └── sample_images/
├── docs/
│   └── (this file, plus README.md)
└── README.md
```

---

## 3. Database Schema

Implement via SQLAlchemy models in `backend/app/models.py`:

```sql
CREATE TABLE images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename VARCHAR(255) NOT NULL,
    modality VARCHAR(20) NOT NULL,        -- 'OPTICAL' | 'SAR'
    capture_date DATE,
    file_path TEXT NOT NULL,
    crs VARCHAR(50),
    bbox_coords JSONB,
    resolution_m FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE queries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text TEXT NOT NULL,
    image_ids UUID[] NOT NULL,
    task_classified VARCHAR(50),          -- 'VQA' | 'GROUNDING' | 'CHANGE_DETECTION' | 'CROSS_MODAL'
    selected_tool VARCHAR(100),
    model_version VARCHAR(50),
    confidence_score FLOAT,
    execution_time_ms INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE query_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id UUID REFERENCES queries(id),
    answer_text TEXT NOT NULL,
    bounding_boxes JSONB,
    change_mask_path TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Acceptance criteria**: tables created, migrations run cleanly, can insert/query a test row via a quick script.

---

## 4. API Contract

Implement in `backend/app/routes/`:

```
POST /images/upload
  Request: multipart/form-data { file, modality: "OPTICAL"|"SAR", capture_date? }
  Response: { image_id, filename, modality, crs, resolution_m }

POST /query
  Request: { query_text: string, image_ids: [uuid] | [uuid, uuid] }
  Response: {
    query_id, task_classified, answer_text, confidence_score,
    bounding_boxes: [[x_min,y_min,x_max,y_max]] | null,
    change_mask_url: string | null,
    execution_trace: {
      selected_tool, model_version, modalities_detected: [string],
      confidence_score, execution_time_ms, reason: string
    }
  }

GET /query/:query_id
  Response: same shape as POST /query
```

**Acceptance criteria**: both endpoints return correctly-shaped JSON (stub data is fine at this stage) and are testable via curl/Postman.

---

## 5. Service Modules

Build each as an independently testable Python module under
`backend/app/services/`, each matching a clean function interface so the
router can call any of them interchangeably.

### 5.1 `router.py` — Agent classifier
```
Function: classify_query(query_text: str, images: list[dict]) -> dict
Returns: { task_classified, modalities_detected, validation_passed, reason }

Logic:
- Keywords like "changed", "compare", "between", "increased", "decreased"
  + 2 images, same modality, different dates -> CHANGE_DETECTION
- Keywords like "show me", "where is", "highlight", "locate" + 1 image -> GROUNDING
- Single image, descriptive question -> VQA
- 2 images, different modalities (one OPTICAL, one SAR) -> CROSS_MODAL
- If required image count/modality doesn't match the detected intent,
  set validation_passed: false and explain why in `reason` instead of guessing
- `reason` must always be a plain-English explanation of the classification
  decision — this is what powers the "why this model?" UI feature
```

### 5.2 `change_detection.py` — no ML model, real CV
```
Function: detect_change(image_path_before: str, image_path_after: str) -> dict
Returns: { change_mask_path, change_percentage, answer_text }

Logic: grayscale both images -> absolute pixel-wise difference -> Gaussian
blur to reduce noise -> threshold to binary change mask -> connected-
component analysis to filter small noise regions -> compute % area changed
-> generate a human-readable summary (e.g. "Built-up area appears to have
increased in the northern region, approximately 18% of the frame changed.")
```

### 5.3 `grounding.py` — no ML model, deterministic
```
Function: ground_object(image_path: str, object_type: str) -> dict
Returns: { bounding_boxes, confidence_score }

Support categories via thresholding:
- water: HSV blue/cyan range
- vegetation: HSV green range
- built-up: Canny edge density + gray/tan color range

For each: threshold -> find contours -> filter by minimum area -> return
largest matching region as [x_min, y_min, x_max, y_max] -> confidence from
contour area ratio or color variance. Keep thresholds as named constants
at the top of the file for easy tuning against real sample images.
```

### 5.4 `vqa.py` — VQA backend, swappable
```
Function: answer_question(image_path: str, query_text: str) -> dict
Returns: { answer_text, confidence_score }

Implement TWO code paths behind this same interface, selected via a config
flag (VQA_BACKEND=geochat|api):

Path A (geochat): Load MBZUAI/geochat-7B via transformers, trust_remote_code=True,
4-bit quantization via bitsandbytes if VRAM < 16GB, load once at startup not
per-request. Confidence from token probabilities if available, else a fixed
heuristic based on answer specificity.

Path B (api fallback): Call a hosted VLM API with a system prompt engineered
for remote-sensing description ("You are analyzing a satellite/aerial image.
Describe only what's visible in terms of land cover, structures, water
bodies, and vegetation. Be specific about location within the frame.") plus
2-3 few-shot satellite Q&A examples. Ask the model to self-report a
confidence 0-1 in a structured response.

Before deciding which path to ship, test Path A in a standalone Colab
notebook (notebooks/geochat_feasibility_test.ipynb): check GPU/VRAM via
nvidia-smi, install transformers/torch/accelerate, attempt to load the
model with quantization if needed, run one test inference, wrap each stage
in try/except with clear failure messages. If it loads and answers within
a couple minutes, use Path A; otherwise use Path B. Build vqa.py so
switching paths requires no changes to any calling code.
```

### 5.5 `cross_modal.py` — SAR + optical fusion
```
Function: fuse_optical_sar(optical_path: str, sar_path: str, query_text: str) -> dict
Returns: { answer_text, bounding_boxes, confidence_score, modality_contribution_note }

Logic: run grounding.py's thresholds on the optical image; run SAR-specific
backscatter thresholding on the SAR image (low backscatter = water,
high/double-bounce pattern = built-up — research and hardcode defensible
threshold values, citing the rule of thumb used); combine results —
prefer optical where it's usable, fall back to SAR-derived classification
for any region that's cloud-obscured in optical (near-uniform white/gray
patch), and explicitly note in modality_contribution_note which modality
drove the answer and why.
```

**Acceptance criteria for this whole section**: each module is callable
standalone with a test image and returns correctly-shaped output — verify
with a quick script per module before wiring into the API layer.

---

## 6. Wire Everything Together

In `POST /query`: call `router.classify_query()` first. If
`validation_passed` is false, return a clear error response explaining
what's wrong (e.g. "Change detection needs 2 images, only 1 provided") —
do not call any service. Otherwise dispatch to the matching real service
module, measure actual execution time with a timer, and populate the full
`execution_trace` object with real values (not placeholders) — the
`reason` field from the router's classification flows through to the
trace response.

**Acceptance criteria**: an integration test script that runs all 3 demo
scenarios against the live backend and prints the full response including
`execution_trace` for each, flagging with a clear failure marker if any
expected field is missing or empty. All 3 must pass before moving on.

---

## 7. Frontend

Build a React app with three screens:

1. **Upload**: drag-and-drop for 1 or 2 images, toggle for "single image" /
   "compare two dates" / "optical + SAR pair"
2. **Query**: chat-style input with example prompt chips ("What's visible
   here?", "What changed?", "Show me the water body")
3. **Results**: main panel showing the image with overlay rendering for
   bounding boxes and change masks, plus a persistent right-side panel
   showing the **Agent Trace** in a monospace, terminal-log style

**Processing/trace UI** (this is the single most important UI element —
build it carefully): while a query processes, show a live-updating
checklist: "✓ Query understood" → "✓ Images analysed" → "✓ Modality
detected: [X]" → "✓ Task identified: [X]" → "✓ Compatible image pair"
(2-image tasks only) → "✓ Selecting [X] Model" → "✓ Generating spatial
evidence" → "✓ Verifying result", ending on a large confidence percentage.
Make the "Selected: [Tool Name]" trace entry **clickable** — expanding to
show the `reason` field from the backend response in a popover.

**Visual design**: deep navy/slate background, cyan accents for overlays,
mission-control aesthetic. Consistent loading and error states — errors
should look intentional (styled message), not like a crash.

**Acceptance criteria**: launch the app in-browser and manually verify all
3 screens render correctly with real backend data (not mocks), the trace
animation plays and is clickable, and overlays (bounding boxes / change
masks) render correctly positioned on their images.

---

## 8. Hardening

Test and fix graceful handling for:
- Uploading a non-image file
- Uploading only 1 image for a change-detection query
- Empty or nonsensical question text
- 2 images that clearly aren't the same location
- Very large image files

None of these should crash the app or hang — each should return/display a
clear, styled message.

---

## 9. Demo Data

Curate real sample images for the 3 demo scenarios (do not use synthetic/
fake placeholder images for the final demo):
1. A single clear optical image with visible water body, vegetation, and
   built-up area
2. A bi-temporal pair of the same location showing visible built-up growth
   between two dates (Sentinel-2 open data, Google Earth Engine exports, or
   curated VRSBench/BigEarthNet samples)
3. A co-registered optical + SAR pair, ideally with visible cloud cover in
   the optical image, to demonstrate the SAR fallback

Store these under `data/sample_images/` and reference them in the
integration test script from Section 6.

---

## 10. Deployment

Set up either:
- Frontend on Vercel/Netlify free tier + backend on Render/Railway free tier, OR
- A single local startup script (`start_demo.sh`) launching backend and frontend together

Prioritize whichever is more reliable without live internet dependency
during a judged demo — test on the actual machine that will run the demo.

**Acceptance criteria**: a cold start of the demo environment (simulating
demo-day conditions) successfully serves all 3 scenarios end-to-end.

---

## 11. Documentation

Write `README.md` covering: problem statement, architecture overview
(agentic router + 4 specialist tools), setup instructions, API contract
summary, and a note on which VQA backend was ultimately used and why. This
should describe what was actually built, not the original plan, if the two
diverged.

---

## Build Order Summary (dependency-ordered, not time-boxed)

1. Repo structure
2. Database schema + models
3. API route stubs (contract-correct, logic stubbed)
4. Frontend skeleton (against mock data matching the contract)
5. `change_detection.py` (no dependencies, build anytime)
6. `grounding.py` (no dependencies, build anytime)
7. `router.py`
8. GeoChat feasibility test (notebook) → decide `vqa.py` backend path
9. `vqa.py`
10. `cross_modal.py` (depends on grounding.py's thresholds + SAR research)
11. Wire router + all services into `/query` endpoint
12. Agent Trace UI + "why this model?" interaction
13. Hardening / edge cases
14. Demo data curation
15. Deployment setup
16. README

Work through these in order; several (5, 6, 8 in parallel with 1-4) have
no interdependencies and can be built in any sequence relative to each
other, but everything after step 11 depends on steps 1–10 being complete
and correct.
