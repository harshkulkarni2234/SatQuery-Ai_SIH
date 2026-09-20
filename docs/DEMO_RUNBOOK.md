# Demo Runbook

For live demo day (SIH 2026, problem statement SIH26167). Read this before
presenting, and skim the "What to say about limitations" column for each
scenario — SatQuery's honesty rules mean it will sometimes refuse to give a
confident answer, and that refusal is a *feature* to explain, not a bug to
hide.

## 1. Startup order

1. **PostgreSQL must be running first.**
   - macOS (this dev machine): `brew services start postgresql@17` (or
     whatever version is installed — `pg_isready` to check).
   - Windows (judge/teammate machine): start the PostgreSQL service from
     Services, or `pg_ctl start` if run manually.
2. **Backend** (from `backend/`):
   ```bash
   cd backend
   venv_mac/bin/uvicorn app.main:app --port 8000
   ```
   (Windows: activate `venv\Scripts\Activate.ps1` first, then
   `uvicorn app.main:app --port 8000`.)
   Confirm it's up: `curl http://127.0.0.1:8000/health` should return
   `{"status": "ok", "database": "up", ...}`.
3. **VQA worker** (optional but recommended — needed for the "What can you
   tell me about this image?" / description-style queries to use the real
   SmolVLM model instead of the honest "unavailable" fallback):
   ```bash
   cd ml/vqa-worker
   python -m uvicorn worker_service:app --host 127.0.0.1 --port 8001
   ```
   (Windows: `.\run_worker.ps1` does the equivalent, using its own isolated
   `ml/vqa-worker/venv` — see `ml/vqa-worker/README.md`.)
   Re-check the backend's `/health` — `vqa_worker` should flip to `"up"`
   within a couple of seconds once the model finishes loading. The
   worker's own `http://127.0.0.1:8001/health` reports which device it
   loaded onto — verified for real on this dev machine as `"device":
   "mps"` (Apple M2 GPU), not `"cpu"` — see `ml/vqa-worker/README.md`.
4. **Change worker** (optional — only if the trained weights
   `ml/change_model/trained/best_change_model.pt` are present on this machine
   and `ml/change-worker/venv` exists; see `ml/change-worker/README.md`):
   ```bash
   cd ml/change-worker
   venv/bin/python -m uvicorn worker_service:app --host 127.0.0.1 --port 8002
   ```
   (Windows: `.\run_worker.ps1`.) `/health` should then report
   `"learned_change_model": "available"`. Without it, change queries run the
   deterministic pixel-difference method and are flagged as a fallback.
5. **Frontend** (from `frontend/`):
   ```bash
   npm run dev
   ```
   Open the printed `http://localhost:5173` URL. In dev mode the
   **"Load demo scenario"** section is visible automatically at the bottom
   of the input screen (gated behind `import.meta.env.DEV`; for a built
   production demo, set `VITE_ENABLE_DEMO_SCENARIOS=true` before
   `npm run build` to keep it visible).

Run `scripts/smoke_test.sh` (or `.ps1` on Windows) after step 5 to confirm
the whole stack end-to-end before judges arrive — see section 5.

## 2. Per-scenario click-path and expected output

All four cards live under **"Load demo scenario (real files, real
upload)"** at the bottom of the input screen. Clicking one uploads the
*real* demo files through the normal `/images/upload` API (visible in the
Network tab if a judge asks) and pre-fills the query — nothing is
pre-baked. Click **Analyze** yourself after it loads.

| Card | Click-path | Expected output |
|---|---|---|
| **Single image** — "What can you tell me about this image?" | Load card → Analyze → View Analysis | VQA specialist (SmolVLM if the worker is up, else an honest "unavailable" fallback answer) describing the scene. |
| **Single image** — "Show me the water body." | Load card → Analyze → View Analysis | Grounding specialist (deterministic CV) draws a bounding box over the river/water region with a real confidence score (bbox fill ratio), not a learned-model guess. |
| **Temporal pair (Lake Mead)** — "What changed between these two dates?" | Load card → Analyze → View Analysis | Change Detection (deterministic CV + real geographic reprojection — the learned Siamese model is deliberately skipped for this pair because 10 m/pixel is outside its 0.5–3 m training domain, and the result is flagged as a fallback if the change worker is up). Real output as verified in this session: **~5.6% of the frame changed (~7,776,900 m²)**, largest change in the upper-left, alignment method "geographic reprojection" (both files are real georeferenced Sentinel-2 GeoTIFFs, CRS EPSG:32611, 10 m resolution). |
| **Optical + SAR pair** — "Use the optical and SAR images together to identify built-up and water-covered regions." | Load card → Analyze → View Analysis | Cross-Modal fusion. Per-class attribution legend (optical-only / SAR-only / combined) and a **"Co-registration unverified — evidence is reported per sensor, not pixel-aligned"** banner (see limitations below — this is correct, expected behavior for this specific pair, not an error). |

## 3. What to say to judges about each honest limitation

Lead with these proactively if a judge asks "why didn't it give a percentage/
confidence there?" — it demonstrates the system's core design principle
(never fabricate what it can't verify), which is the actual pitch, not a
gap to apologize for.

- **Single image (scenario A):** "This particular demo image's exact source
  and license couldn't be independently verified from the file alone — we
  say so plainly rather than presenting it as verified. It has no visible
  watermark, but if a judge wants a fully-cited alternative, `farm2.jpg`
  (Copernicus Sentinel-2 L2A, 2023-05-11, Bełżyce, Poland) is a backup in
  the same drop-in shape, just without a visible water body."
- **Temporal pair (scenario B):** "These are two real Sentinel-2 scenes of
  the same location, ~4 years apart, both low cloud cover. The change
  percentage and area shown are computed live by the pipeline, not looked
  up — we deliberately never caption this with a claimed number ourselves
  ahead of time. If asked how confident we are in the exact magnitude: a
  raw pixel diff of the full scene shows much noisier numbers than what the
  UI reports, because the pipeline's own region-based measurement filters
  texture/shadow noise the raw diff doesn't — that's the whole point of not
  just reporting `cv2.absdiff` directly."
- **Optical+SAR pair (scenario C):** "This pair *is* genuinely co-registered
  by construction (same BigEarthNet-MM tile and patch-grid index, same real
  date), but it ships as plain PNG with no embedded CRS/transform. SatQuery
  correctly reports `coregistration: unverified` for it — it refuses to
  claim pixel-level spatial agreement it cannot check from the files
  themselves, even though we know from the metadata that it would be
  correct in this specific case. That's the honesty rule working exactly
  as designed, not a limitation to explain away — the system doesn't take
  our word for it either."
- **General:** if the VQA worker isn't running (or has been intentionally
  stopped for the backup-plan demo below), the description-style query
  still returns an answer — labeled with a real "used_fallback" badge and
  an honest reason — instead of erroring out.

## 4. Backup plan — if a live specialist fails

The system is designed so a specialist going down never blanks the screen
or 500s (see `docs/HARDENING_REPORT.md`). To demonstrate this live:

1. Stop the VQA worker process (Ctrl+C in its terminal) while the frontend
   is still up.
2. Re-run the "What can you tell me about this image?" query.
3. Expected: the query still returns 200, `used_fallback: true`, and a
   clear reason in the UI (e.g. "VQA worker unavailable") instead of an
   error page — the honest fallback surfaces live in the same UI a judge
   is already watching, no scripted alternate screen needed.
4. Restart the worker afterward if you need VQA again for later scenarios.

For **change detection**, there are two specialists. `change.siamese_binary_cnn`
is a learned binary change/no-change model served by the optional change
worker (`ml/change-worker/`, port 8002, needs its own venv and the trained
weights, which are committed under `ml/change_model/trained/`). `change.deterministic_cv`
needs no worker and is always available. To demonstrate the honest fallback
live: with the change worker running, run a change query, then stop the
worker (Ctrl+C) and re-run it. Expected: the query still returns 200, the
answer now comes from the deterministic method, and the UI shows a
"Used fallback" badge with the reason (and `/health` reports
`"learned_change_model": "unavailable"`). If the worker is not running at
all, the deterministic method runs from the start (also flagged as a
fallback). The learned model is skipped, with the reason shown, for
size-mismatched pairs, pairs known to be coarser than 3 m/pixel, and
non-corresponding pairs; the deterministic "spatial correspondence could not
be verified" refusal (non-georeferenced pair vs scenario B's real GeoTIFFs)
still applies as before.

## 5. Smoke test

Run before judges arrive, and again if anything about the environment
changes (new machine, restarted services):

```bash
bash scripts/smoke_test.sh
```

(Windows: `scripts/smoke_test.ps1` — mirrors the same steps; see the
troubleshooting note in section 6 about it being unverified on this dev
machine.)

It hits `/health`, uploads the 3 real demo scenarios' files, runs the 4
demo queries, downloads one PDF report, and prints a final `PASS`/`FAIL`
summary with the specific step that failed if something's wrong.

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `/health` shows `"database": "down"` | Postgres not started, or `backend/.env`'s `DATABASE_URL` doesn't match the local role/db | Start Postgres; check `DATABASE_URL` matches an existing role (`createdb satquery` if the DB itself doesn't exist yet) |
| `/health` shows `"vqa_worker": "down"` | Worker not started, still loading the model, or crashed | Check the worker's terminal for errors; description-style queries still work via the honest fallback in the meantime |
| "Load demo scenario" section doesn't appear | Frontend was built for production without `VITE_ENABLE_DEMO_SCENARIOS=true`, or you're not running `npm run dev` | Run `npm run dev` (dev mode always shows it), or rebuild with the env var set |
| A demo scenario card shows "Could not load..." | Backend not running, or the `/demo-assets` static mount isn't up (very old backend checkout before this phase) | Confirm `curl http://127.0.0.1:8000/demo-assets/manifest.json` returns 200; restart the backend if it was started before this feature was added |
| Upload succeeds but analysis 400/422s unexpectedly | Normal for a genuinely incompatible pair (see `docs/HARDENING_REPORT.md`) — check the on-screen compatibility checklist for the real reason before assuming it's a bug | Read the checklist; it's usually correct honest behavior |
| `.ps1` scripts fail on Windows | These were authored and reviewed on this project's macOS dev machine and their `.sh` counterparts were the ones actually executed here — the `.ps1` scripts mirror the same steps but have **not** been run on a real Windows machine in this session | Smoke-test the `.ps1` scripts on an actual Windows machine before relying on them for judging day; fall back to running the equivalent commands manually if one fails |

## 7. Known, deliberate scope limits (say these plainly if asked)

- The learned change model is **binary** change/no-change only: it says
  where pixels changed, never what they changed to, and it does not
  classify land cover. It is an original small Siamese CNN (~4.9M params)
  trained on a SECOND-derived split (1,700 train / 300 val; val F1 0.468,
  IoU 0.327), 0.5–3 m aerial RGB only. The SECOND dataset's license is
  unstated, so treat the model as research-only. A leakage check confirmed none of the
  CDVQA test images were in its training/validation split (it is still an
  in-distribution evaluation) — see `ml/change_model/MODEL_CARD.md`
  (Known Limitations 7–9).
- The CDVQA numbers for it are a small sample (13–15 questions per type)
  and not like-for-like with the deterministic baseline, but on images the model
  never trained on (leakage checked: none) — don't quote them as an
  improvement (`evaluation/README.md`).
- The BigEarthNet LoRA adapter v2.0 is real (trained on official
  BigEarthNet v2.0 labels), but its RSVQA-LR results (60 questions/type) are **mixed** versus the base
  model, not a clear win: rural/urban 61.7% vs 40.7% and comparison 70.0%
  vs 65.0%, but presence 70.0% vs 73.3%, and counting is still unreliable
  (2/39 exact, 21 unparseable). Its
  adapter weights are not committed to the repo. See
  `ml/adaptation/MODEL_CARD.md` and `evaluation/results/rsvqa_lr_score_v2.json`.
- Real benchmark runs exist for RSVQA-LR, CDVQA and VRSBench (three tasks),
  each on a seeded sample, not the full test sets — see
  `evaluation/README.md` for the sample sizes and caveats. There is no
  `docs/EVALUATION.md`; do not reference it or invent numbers.
- Scenario A's single image has unconfirmed provenance (documented above).
