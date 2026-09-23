# SatQuery AI — Full Project Brief

**Smart India Hackathon 2026 · Problem Statement SIH26167 · Team 140221 "Stack Overflower"**

> **Who this is for.** A teammate who has not written any of the code and needs
> enough context to build a presentation, answer a judge's question, or run the
> demo. Everything here is checked against the running system. Read §1 and §16
> at minimum; §16 tells you what you may and may not claim.

---

## Table of contents

1. [TL;DR](#1-tldr)
2. [The problem statement](#2-the-problem-statement)
3. [The problem in human terms](#3-the-problem-in-human-terms)
4. [What we built](#4-what-we-built)
5. [The core idea (our USP)](#5-the-core-idea-our-usp)
6. [User journey](#6-user-journey)
7. [System architecture](#7-system-architecture)
8. [The six specialists](#8-the-six-specialists)
9. [The two models we trained](#9-the-two-models-we-trained)
10. [Honesty engineering — the real differentiator](#10-honesty-engineering--the-real-differentiator)
11. [Evaluation results](#11-evaluation-results)
12. [Performance and hardware](#12-performance-and-hardware)
13. [Technology stack](#13-technology-stack)
14. [Repository map](#14-repository-map)
15. [How to run it](#15-how-to-run-it)
16. [RULES FOR THE PPT — what you may and may not claim](#16-rules-for-the-ppt--what-you-may-and-may-not-claim)
17. [Demo script](#17-demo-script)
18. [Known limitations](#18-known-limitations)
19. [Likely judge questions and honest answers](#19-likely-judge-questions-and-honest-answers)
20. [Glossary](#20-glossary)
21. [Sources and licences](#21-sources-and-licences)

---

## 1. TL;DR

SatQuery AI lets someone upload one or two satellite images, ask a question in
plain English, and get an answer backed by visual evidence and a **replayable
record of how the answer was produced**.

The product is **not** any single AI model. It is the **agent** that decides
which analysis tool applies to a given question and image pair, checks whether
that combination is even valid, and makes that decision auditable.

| Fact | Value |
|---|---|
| Routed specialists | 6 |
| Models trained by us | 2 (a change-detection CNN and a VLM adapter) |
| Public benchmarks run | 3 (RSVQA-LR, CDVQA, VRSBench) |
| Automated tests | 292 (258 backend + 34 evaluation), green on macOS **and** Windows |
| Audit trail | 8 timed steps recorded per query |
| Runs offline | Yes — no cloud, no API key, no internet needed at query time |
| Trains + serves on | One laptop with a 4 GB NVIDIA RTX 2050 |

---

## 2. The problem statement

- **ID:** SIH26167
- **Title:** *SatQuery AI — An Interactive Vision-Language Assistant for
  Multimodal Remote Sensing Image Analysis through Text Queries*
- **Organisation:** Indian Space Research Organisation (ISRO)
- **Theme:** Space Technology
- **Category:** Software

The statement asks for a system that analyses remote-sensing imagery from
natural-language queries, across multiple modalities (optical and SAR), and it
explicitly names **RSVQA, CDVQA and VRSBench** as the benchmarks to evaluate
against. We ran all three.

> ⚠️ Verify the exact title wording against the official portal (sih.gov.in)
> before printing it on a slide. Ours came from third-party problem-statement
> catalogues and was never confirmed against the official page.

---

## 3. The problem in human terms

Today an analyst who wants to answer *"what changed in this area between 2018
and 2022?"* has to:

1. pick a tool for that specific question,
2. re-project both scenes onto a common grid,
3. choose a threshold, run a difference, and eyeball the result,
4. repeat with a **different tool chain** for the next question type.

Two things are missing:
- **Accessibility** — every step needs GIS expertise.
- **Accountability** — there is no record of *why* a number came out the way it
  did, which matters if the answer feeds an operational decision.

---

## 4. What we built

A web application with three parts:

- **Frontend** (React + Vite): upload, ask, view evidence, replay the trace,
  download a PDF/JSON report.
- **Backend** (FastAPI): the agent. Classifies the question, validates the
  images, selects a specialist, records every step. **Imports no ML library**,
  so it starts instantly and never crashes on a model problem.
- **Model workers** (two separate processes): the heavy PyTorch models. Both
  are **optional** — if one is down the system still answers, and says which
  method it actually used.

Four analysis types behind a single text box:

| Type | Question example | What it returns |
|---|---|---|
| Visual Q&A | "What can you tell me about this image?" | Text answer from a vision-language model |
| Grounding | "Show me the built-up area." | Bounding boxes over the matching land cover |
| Bi-temporal change | "What changed between these two dates?" | Change mask + % + area in real m² |
| Optical + SAR fusion | "Compare these optical and SAR images." | Per-sensor evidence, works when cloud blocks optical |

---

## 5. The core idea (our USP)

Most hackathon entries wire one model to one UI. Ours is built around a
**specialist registry**: one shared contract (`SpecialistResult`), six
implementations, and a selection step that picks by **priority** and a **live
health probe**.

That buys three things a single-model system cannot have:

1. **Graceful degradation.** A learned model going down does not break the
   product; a deterministic method takes over and the UI says so.
2. **Swappability.** A better VLM drops in behind the same contract without
   touching the agent.
3. **Auditability.** Because selection is an explicit step, it can be recorded
   and replayed.

---

## 6. User journey

```
Upload 1–2 scenes (GeoTIFF / PNG, optical or SAR)
        ↓
Type a question in plain English
        ↓
Agent: classify → validate → route → execute
        ↓
Answer + evidence (mask / boxes / stats) + 8-step replayable trace
        ↓
Download PDF or JSON report
```

The **trace replay** is the moment that lands with judges: the UI plays back
the backend's own recorded steps with real millisecond timings, including which
specialist was *planned* versus which one actually *ran*, and why.

---

## 7. System architecture

Six tiers, top to bottom:

| Tier | Contents |
|---|---|
| 1 · Client | React 18 + Vite SPA — upload, query box, evidence viewer, trace replay, report download |
| 2 · API | FastAPI — `POST /images/upload`, `POST /query`, `GET /query/{id}`, `GET /query/{id}/report.pdf`, `GET /specialists`, `GET /health` |
| 3 · Agent core | `raster_ingest` → `router` → `compatibility` → `planner` → `registry.select()` → `TraceRecorder` |
| 4 · Specialist registry | 6 implementations of one contract |
| 5 · Execution | VQA worker `:8001`, change worker `:8002`, in-process OpenCV |
| 6 · Data | PostgreSQL + Alembic, `/masks` static mount, `/demo-assets` static mount |

**Protocols across boundaries:** client → API is REST/JSON (uploads are
`multipart/form-data`); API → change worker is `multipart` in, **base64 PNG mask**
out. The backend — not the worker — saves the mask, so the URL the UI links to
always exists.

Ready-to-paste Mermaid diagrams for all of this live in
`~/Desktop/SIH2026_Deck/diagrams/` (5 files: architecture, request lifecycle,
change-detection decision logic, data model, read path).

### The agent core, step by step

1. **`raster_ingest`** — reads real GeoTIFF metadata with rasterio: CRS, bounds,
   transform, resolution, acquisition date.
2. **`router`** — classifies the question into one of four tasks.
3. **`compatibility`** — for a pair: same modality? distinct dates? same CRS?
   Do the bounding boxes overlap (IoU)? Is co-registration verified?
4. **`planner`** — builds an `ExecutionPlan`, or **rejects the pair with a
   stated reason**.
5. **`registry.select()`** — highest-priority specialist that passes a live
   `/health` probe.
6. **`TraceRecorder`** — 8 steps with real `perf_counter` timings, persisted
   even when execution fails.

---

## 8. The six specialists

| ID | Task | Kind | Priority | Notes |
|---|---|---|---|---|
| `vqa.smolvlm_base` | VQA | learned | 10 | Base SmolVLM-256M-Instruct |
| `vqa.smolvlm_bigearthnet_lora_stage3` | VQA | learned | 5 | Our LoRA adapter, RS-adapted |
| `grounding.deterministic_cv` | GROUNDING | deterministic | 10 | HSV + morphology, 5 land-cover targets |
| `change.siamese_binary_cnn` | CHANGE_DETECTION | learned | 5 | Our Siamese CNN |
| `change.deterministic_cv` | CHANGE_DETECTION | deterministic | 10 | **Marked `is_fallback`** — pixel diff + real geographic re-projection |
| `cross_modal.feature_fusion` | CROSS_MODAL | rules | 10 | Optical + SAR statistics |

Lower priority number = preferred. So for change detection the learned model is
tried first, and the deterministic one is the declared fallback.

---

## 9. The two models we trained

### 9a. Change model — `change.siamese_binary_cnn`

An **original architecture written for this project**. It is *not* TinyCD, BIT
or ChangeFormer; those were reviewed in `ml/change_model/SELECTION.md` and not
used.

| Property | Value |
|---|---|
| Architecture | Shared Siamese encoder (4 conv blocks, 3→64→128→256→512) → feature-diffusion fusion (`abs(f1−f2)`, `f1`, `f2`, `f1·f2`) → transposed-conv decoder |
| Parameters | 4,874,241 (~19 MB checkpoint) |
| Input | Two co-sized 256×256 RGB images |
| Output | Binary change mask |
| Training data | SECOND-derived, 1,700 train / 300 val |
| Loss / epochs | BCEWithLogits, 5 epochs, fp16 |
| Training time | 5.2 minutes on an RTX 2050 |
| **Best validation** | **F1 0.4677, IoU 0.3268** |

**Scope — say this plainly:** it is a **binary** change/no-change segmenter. It
shows *where* pixels changed, never *what* they changed to, and it does not
classify land cover.

**Reproducibility:** `ml/change_model/prepare_second_data.py` rebuilds the
training arrays **byte-for-byte identically** — verified, zero differing bytes.

### 9b. VLM adapter — `smolvlm256m-ben-lora-s3-v2.0`

| Property | Value |
|---|---|
| Base model | HuggingFaceTB/SmolVLM-256M-Instruct (Idefics3) |
| Method | LoRA, r=8, alpha=16, dropout 0.1, 7 target modules |
| Adapter size | 2.44 M trainable params, 9.4 MB |
| Labels | Official BigEarthNet v2.0 (Zenodo parquet, CDLA-Permissive-1.0) |
| Dataset built | 25,645 QA pairs from 4,008 matched patches; patch-level 80/10/10 split, seed 42 |
| Actually trained on | 2,400 rows, 450 steps, lr 2e-4, fp16 |
| Training time | 3.24 hours on an RTX 2050 |
| **Held-out BigEarthNet** | **presence 99.3% (149/150)**, **count 96.0% (48/50)** |
| Baselines for those | presence 63.1% (always-"yes"), count 25.3% (majority class) |

The worker decides **internally** whether a given question goes to the adapter
or the base model — narrow question types go to the adapter.

### 9c. A third model exists but is deliberately NOT used

`best_semantic_change_model.pt` is a per-class semantic change model we trained
to try to answer *what* changed. **We do not ship it**, because:
- its derived binary change is **worse** than the shipped model (F1 0.363 vs 0.468),
- per-class IoU is poor (water 0.002, playgrounds 0.015),
- it was still improving at epoch 5 — undertrained.

It is not wired into the registry. **Do not put it on a slide.**

---

## 10. Honesty engineering — the real differentiator

This is the part judges remember. The system is built so it **cannot** quietly
mislead:

| Guard | Behaviour |
|---|---|
| **Refuses rather than guesses** | If metadata proves two scenes cover different areas, it states a refusal instead of returning a percentage |
| **Training-domain gate** | The learned change model only runs on imagery ≤ 3 m/pixel (its training domain). Coarser imagery auto-routes to the deterministic method |
| **Near-full-frame guard** | If the model marks > 85% of the frame as changed, that is treated as "these are not the same place", not a result |
| **Mask validation** | A corrupt or non-binary mask is rejected — it never becomes a fabricated "0%" |
| **Planned vs executed** | The trace records the specialist *selected* and the one that *actually ran*, with the reason they differ |
| **Confidence may be "unavailable"** | When no confidence can be justified, the UI says unavailable rather than inventing a number |
| **Fallbacks are labelled** | Every fallback names why it happened, in the UI and in the PDF |

**Live example you can show:** the Lake Mead demo pair is 10 m/pixel Sentinel-2.
The agent *plans* the learned model, then skips it because 10 m is coarser than
its 0.5–3 m training domain, runs the deterministic method instead, and prints:

> *"Planned change.siamese_binary_cnn but ran change.deterministic_cv: the
> imagery is about 10 m per pixel, coarser than the 0.5-3 m aerial RGB image
> pairs (SECOND dataset) this model was trained on."*

That is the system refusing to use a model outside its competence — **on stage,
visibly.** Frame it as a feature, because it is one.

---

## 11. Evaluation results

All numbers below are real, scored runs. Sample sizes are given because they
matter.

### RSVQA-LR (60 questions per type, 240 total)

| Question type | Base model | With our adapter |
|---|---|---|
| rural / urban | 40.7% | **61.7%** |
| comparison | 65.0% | **70.0%** |
| presence | **73.3%** | 70.0% |
| count (exact) | 0% (34 scored) | 5.1% (2/39 scored) |

**Honest read:** better on two types, slightly worse on presence, counting still
unreliable. At 60 questions per type the margin is roughly ±12 points, so only
the +21-point rural/urban gain is clearly outside noise.

### VRSBench (180 VQA + 51 referring + 30 captioning)

- VQA overall **33.9%**; best type object existence **86.7%**; worst object
  direction **6.7%**.
- Referring/grounding: **0 of 51 samples ever reached the grounding specialist.**
  Our grounding supports 5 land-cover classes; VRSBench asks about 26 object
  classes (vehicle, ship, airplane…). Zero vocabulary overlap. We report this
  rather than a fake IoU.
- Captioning: BLEU-1 9.8%, BLEU-4 0.2%, ROUGE-L 12.2%.

### CDVQA (120 questions, 19 routing errors)

| Type | Deterministic | Learned model |
|---|---|---|
| change_ratio | 14.3% (1/7 scored) | 30.8% (4/13 scored) |
| change_ratio_types | 37.5% (3/8) | 33.3% (5/15) |
| yes/no + categorical types | 0 scored | 0 scored |

**Honest read:** the "overall 0.321 vs 0.267" figure is computed over *scored*
answers only (28 vs 15), so it is **not** like-for-like. Do not present it as a
clean improvement. The categorical types score zero because our model states
aggregate change, not a land-cover class — a real capability gap.

### Train/test leakage: checked, none

CDVQA's 968 test pairs are a subset of SECOND's 2,968, and our model trained on
the other 2,000. `ml/change_model/check_cdvqa_leakage.py` found **0 of 968**
test pairs in our training or validation split. The evaluation is still
*in-distribution* (same imagery source).

---

## 12. Performance and hardware

Measured on the **RTX 2050 (4 GB)** laptop, warm median, first call discarded:

| Operation | Latency |
|---|---|
| Change model, per 256×256 pair | **24 ms** |
| VQA, short answer | 1,373 ms |
| VQA, descriptive answer | 2,198 ms |
| **Full Lake Mead change query, end to end** | **591 ms** |

Deterministic paths (macOS reference): pixel-diff change 6 ms on a small pair,
217 ms on a 1008×1379 GeoTIFF *including real re-projection*; grounding 4 ms;
cross-modal fusion 2 ms.

**Headline for a slide:** both models were **trained AND are served** on the
same 4 GB laptop GPU. No cluster, no cloud bill.

---

## 13. Technology stack

| Layer | Technology |
|---|---|
| Frontend | React 18.3, Vite 6.4, plain CSS (no component library) |
| Backend | FastAPI, SQLAlchemy, Alembic, PostgreSQL, Pydantic, python-multipart |
| Geospatial | rasterio, GDAL, OpenCV (headless), NumPy, Pillow |
| ML | PyTorch, Transformers, PEFT (LoRA), safetensors |
| Reporting | ReportLab (server-side PDF) |
| Testing / eval | pytest (292 tests), BLEU/ROUGE, IoU/F1, seeded benchmark runners |

---

## 14. Repository map

```
backend/          FastAPI app — stays torch-free
  app/
    routes/       images.py, query.py, specialists.py, report.py
    services/     router, compatibility, planner, registry (+adapters),
                  change_detection, change_specialists, grounding,
                  cross_modal, vqa, raster_ingest, trace, report
    contracts.py  shared pydantic contracts (frontend/backend/model boundary)
    models.py     images, queries, query_results
  tests/          258 tests
ml/
  vqa-worker/     SmolVLM worker (port 8001)
  change-worker/  Siamese CNN worker (port 8002)
  change_model/   training script, data prep, model cards, weights
  adaptation/     BigEarthNet dataset prep + LoRA training
  smolvlm/        adapter weights and version metadata
frontend/         React + Vite SPA
evaluation/       runners, metrics, scored results for all 3 benchmarks
data/demo/        3 real demo scenarios (served read-only)
docs/             this brief, contracts, demo runbook, model cards, progress log
scripts/          start_all / smoke_test (.sh and .ps1)
```

**Key documents to read next:** `AGENTS.md` (the rules the code follows),
`docs/DEMO_RUNBOOK.md` (demo day), `ml/change_model/MODEL_CARD.md`,
`ml/adaptation/MODEL_CARD.md`, `evaluation/README.md`.

---

## 15. How to run it

**Prerequisites:** Python 3.11+, Node 18+, PostgreSQL 14+, a database named
`satquery`, and `backend/.env` with a `DATABASE_URL`.

**Windows (the demo machine):**
```powershell
.\scripts\start_all.ps1
.\scripts\smoke_test.ps1
```

**macOS / Linux:**
```bash
bash scripts/start_all.sh
bash scripts/smoke_test.sh
```

Then open **http://localhost:5173**.

Check `http://127.0.0.1:8000/health` — you want:
```json
{"status":"ok","database":"up","vqa_worker":"up","learned_change_model":"available"}
```

Each worker needs its **own** virtual environment and the CUDA torch wheel.
The backend venv must never contain torch.

**Tests:**
```bash
cd backend && python -m pytest -q          # expect 258 passed
python -m pytest evaluation/tests -q       # expect 34 passed
```

---

## 16. RULES FOR THE PPT — what you may and may not claim

Our code follows a strict honesty rule (`AGENTS.md`): **never fabricate
confidence, accuracy or percentages.** The slides must follow the same rule. A
judge who catches one inflated claim will distrust everything else.

### ✅ You MAY say

- "6 routed specialists, 2 models trained in-house, 3 public benchmarks run."
- "292 automated tests, green on macOS and Windows."
- "99.3% presence accuracy on a held-out BigEarthNet split, against a 63.1% baseline."
- "+21 points on rural/urban in RSVQA-LR versus the base model (60 questions)."
- "Validation F1 0.468 / IoU 0.327 for our change model (300 validation pairs)."
- "24 ms per pair on a 4 GB laptop GPU; a full query end to end in 591 ms."
- "Zero of CDVQA's 968 test pairs appear in our training data — verified by a committed script."
- "7,776,900 m² of real measured change at Lake Mead, 2018→2022."
- "It refuses to answer when two scenes cannot be compared."

### ❌ You MUST NOT say

- ❌ "Semantic change detection" — our shipped model is **binary only**.
- ❌ "Our model beats the baseline on CDVQA" — not like-for-like (28 vs 15 scored).
- ❌ "State-of-the-art" / "99% accurate" as a blanket claim — the 99.3% is one
  question type on one in-domain split.
- ❌ Any mention of the semantic model's numbers — it is not shipped.
- ❌ "Real-time" without qualification — quote the measured millisecond figure.
- ❌ Inventing a confidence score for the change detection — there isn't one.
- ❌ Claiming GeoChat or TinyCD is used — both were evaluated and **not** used.

### Framing advice

Lead with the **agent and the audit trail**, not with model accuracy. Our model
scores are modest and honest; our routing, validation and traceability are
genuinely unusual. That is the story.

---

## 17. Demo script

Four demo cards load **real files through the normal upload API** — nothing is
pre-baked.

| # | Card | Query | What to point at |
|---|---|---|---|
| 1 | Single image | "What can you tell me about this image?" | The VLM answers; the report header shows the RS-adapted model was used |
| 2 | Single image | "Show me the built-up area." | **8 bounding boxes** over the city, confidence 0.472 from a real measured bbox fill ratio |
| 3 | Temporal pair (Lake Mead) | "What changed between these two dates?" | **The fallback badge** — the agent skips the learned model because 10 m/pixel is outside its training domain. Real re-projection, 5.6%, 7,776,900 m² |
| 4 | Optical + SAR | "Compare these optical and SAR images." | Per-sensor evidence and an honest "co-registration unverified" banner |

**Do not ask scenario 1's image for water.** That scene's river is grey-brown,
so the HSV detector honestly reports "no water regions detected". It is correct
behaviour but a weak opening. Farmland and vegetation also collapse to one
full-frame box on that image. Built-up is the target that demos well.

**The money moment:** after any query, the trace replay shows 8 steps with real
timings and names the specialist that actually ran.

---

## 18. Known limitations

State these before a judge finds them.

1. **Binary change only** — no land-cover class for what changed.
2. **Grounding covers 5 land-cover classes**, not arbitrary objects. VRSBench's
   26 object classes are entirely out of scope.
3. **Counting is unreliable** — the base VLM cannot count; our adapter barely
   improves it on RSVQA-LR.
4. **SECOND dataset licence is unstated** — the change model is research-only
   until that is resolved.
5. **Small evaluation samples** (60/type RSVQA-LR, ~15/type CDVQA and VRSBench).
6. **In-distribution evaluation** — CDVQA uses SECOND imagery, the same source
   our model trained on (no leakage, but same domain).
7. **No georeferencing awareness in the models** — they see raw pixels; CRS
   handling lives in the agent, not the network.
8. **256 M-parameter VLM** — small by design for a 4 GB GPU; open-vocabulary
   range is limited.

---

## 19. Likely judge questions and honest answers

**"Why not just send the image to GPT-4V or Gemini?"**
Four reasons: the imagery never leaves the facility; zero per-query cost; a
cloud VLM does no geospatial maths (no CRS, no re-projection, no area in m²);
and you get no audit trail. Ours also runs with no internet at all.

**"Your accuracy numbers aren't very high."**
Correct, and we report them as measured. The contribution is the agent: routing,
validation, refusal and traceability. Models are swappable behind one contract —
a larger VLM drops in without touching the agent.

**"Is this actually working or a mock-up?"**
Working. 292 automated tests, three public benchmarks scored end to end, and the
demo uploads real files through the real API. Happy to run it live.

**"Did you train anything yourselves?"**
Two models. A 4.87 M-parameter Siamese CNN for change detection, and a LoRA
adapter for the VLM — both trained on one 4 GB laptop GPU.

**"How do we know you didn't test on your training data?"**
We wrote a checker and committed it. Zero of CDVQA's 968 test pairs appear in
our 2,000 training images. The data-prep script also reproduces our training
arrays byte-for-byte.

**"What happens if a model fails during the demo?"**
The query still answers. A deterministic method takes over and the UI shows a
fallback badge naming the reason. We can demonstrate it by killing a worker.

**"Can it scale to a real ISRO workload?"**
The architecture is designed for it: workers are separate processes that scale
independently, and the backend imports no ML stack. We have not load-tested it —
that would be the next step.

---

## 20. Glossary

| Term | Meaning |
|---|---|
| **VQA** | Visual Question Answering — answering a text question about an image |
| **VLM** | Vision-Language Model |
| **SAR** | Synthetic Aperture Radar — radar imaging that sees through cloud |
| **Optical** | Ordinary camera-like satellite imagery (e.g. Sentinel-2) |
| **Bi-temporal** | Two images of the same place at different times |
| **CRS / EPSG** | Coordinate Reference System, identified by an EPSG code |
| **GeoTIFF** | A TIFF image carrying geographic metadata |
| **Re-projection** | Warping one image onto another's coordinate grid |
| **IoU** | Intersection over Union — overlap metric |
| **F1** | Harmonic mean of precision and recall |
| **LoRA** | Low-Rank Adaptation — fine-tuning a few million params, not all |
| **Co-registration** | Whether two images are pixel-aligned to the same ground |
| **Specialist** | One analysis implementation behind our shared contract |
| **Fallback** | A different specialist running because the preferred one could not |

---

## 21. Sources and licences

**Benchmarks**
- RSVQA-LR — Lobry et al., IEEE TGRS 2020 — CC-BY-4.0 — zenodo.org/records/6344334
- CDVQA — Yuan et al., IEEE TGRS 2022 — Apache-2.0 — github.com/YZHJessica/CDVQA
- VRSBench — Li et al., NeurIPS 2024 — CC-BY-4.0 — huggingface.co/datasets/xiang709/VRSBench

**Training data**
- BigEarthNet v2.0 — CDLA-Permissive-1.0 — zenodo.org/records/10891137
- SECOND — **licence unstated** — captain-whu.github.io/SCD — research use only

**Models and methods**
- SmolVLM-256M-Instruct — HuggingFaceTB
- LoRA — Hu et al., ICLR 2022 — arxiv.org/abs/2106.09685
- TinyCD / BIT / ChangeFormer — reviewed, **not used**
- GeoChat — feasibility-tested, too large for the target GPU

**Tooling**
- rasterio / GDAL, Sentinel-1 & Sentinel-2 (Copernicus), STAC, EPSG registry, OpenCV / NumPy

---

*Every figure in this brief was produced by the running system or read from a
committed result file. If you need a number that is not here, ask rather than
estimating it.*
