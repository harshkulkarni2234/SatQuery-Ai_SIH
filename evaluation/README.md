# SatQuery AI — Evaluation Framework

Built for Phases B3–B6 of the original 3-person plan (`docs/satquery-master-build-spec.md`).

## What this is, honestly

The original plan (B4/B5/B6) called for downloading and running against
**RSVQA, CDVQA, and VRSBench** — official, labeled remote-sensing benchmark
datasets. That did not happen in this build: when reached, downloading
those datasets required explicit user permission per this build's own
process (large third-party downloads), and the user's actual instruction
was **"you can use the dataset available in the testing folder"** instead.

The local `testing/` folder (inventoried in Phase B1 — see
`data/manifest.json`'s `unused_raw_material` entry) contains ~4,000 real
Sentinel-2 optical patches, 1 real Sentinel-1 SAR image, and a handful of
misc images — but **zero label files of any kind** (verified: no
`.json`/`.csv`/`.txt`/label file anywhere under it). There is no ground
truth in this data to compute RSVQA/CDVQA/VRSBench-style accuracy against.

So what actually exists here is:

1. **A real, working evaluation framework** (`runners/base.py`,
   `metrics/`, `results/generate_summary.py`) — dataset-independent,
   fully unit-tested, ready to point at a real labeled benchmark the
   moment one is fetched.
2. **One real runner** (`runners/run_testing_folder_eval.py` +
   `runners/testing_folder_adapter.py`) that drives the live SatQuery
   backend over its real HTTP API against a seeded random sample of
   `testing/`'s real images, using the same two fixed prompts as the C9
   demo scenario A card. It records real, measured system-behavior
   statistics (task routing, specialist selection, latency, confidence
   availability, error rate, fallback rate) — **it does not and cannot
   report accuracy**, because there is no ground truth to score against.

**If you are looking for RSVQA/CDVQA/VRSBench numbers for this project:
they do not exist. Do not present the numbers below as benchmark accuracy
— they are routing/latency/error statistics, not correctness scores.**

## Layout

```
evaluation/
├── runners/
│   ├── base.py                      # generic sample -> HTTP API -> JSONL runner (resumable, --limit)
│   ├── testing_folder_adapter.py    # the ONE dataset adapter that exists (unlabeled, see its docstring)
│   └── run_testing_folder_eval.py   # CLI entry point
├── metrics/
│   ├── vqa_metrics.py               # exact-match, yes/no, count accuracy/RMSE (unit-tested, unused by
│   ├── spatial_metrics.py           # box IoU/acc@0.5, pixel-level change F1/IoU  the current run — no
│   └── text_metrics.py              # lightweight BLEU-n / ROUGE-L                labeled data to score)
├── results/
│   ├── generate_summary.py          # JSONL -> SUMMARY.md (real stats only, never fabricates accuracy)
│   └── *.jsonl, SUMMARY.md          # actual run output (git-ignored except this README)
├── configs/                         # reserved for future labeled-benchmark configs (empty today)
├── tests/                           # unit tests for metrics/, hand-computed examples (23 tests)
└── requirements.txt                 # numpy, requests, pytest — separate from backend/requirements.txt
```

Why HTTP API and not in-process (the plan's stated preference): see
`runners/base.py`'s module docstring — it's a deliberate isolation
tradeoff, not an oversight.

## Running it

Start a backend first (`scripts/start_all.sh` or manually — see
`docs/DEMO_RUNBOOK.md`), then:

```bash
cd /path/to/satquery-ai
pip install -r evaluation/requirements.txt
python -m evaluation.runners.run_testing_folder_eval \
    --api-base-url http://127.0.0.1:8000 \
    --n-images 100 \
    --output evaluation/results/testing_folder_smoke.jsonl
python -m evaluation.results.generate_summary evaluation/results/testing_folder_smoke.jsonl
```

`--n-images` controls how many real images are sampled (deterministic,
seeded — pass `--seed` to change it); each image gets 2 prompts (VQA +
grounding), so `n_images=100` produces 200 samples. `--limit` caps the
total number of (image, prompt) samples actually run in one invocation —
useful for a fast smoke check before committing to a full pass. Re-running
the same command resumes from `--output`'s existing rows rather than
re-running everything (or pass `--no-resume` to start over).

## Metrics unit tests

```bash
cd /path/to/satquery-ai
python -m pytest evaluation/tests -q
```

23 tests, all with hand-computed expected values (not just "does it not
crash") — e.g. `pixel_mask_f1_iou` is tested against a manually-solved
2×2 mask where precision/recall/F1/IoU are all worked out by hand in the
test's own comment. These test the metric functions in isolation; they
are not run against the testing/-folder data (which has no labels for
them to score).

## Real run recorded in this repo

One smoke run was actually executed against a live backend and its real
output is summarized in `results/SUMMARY.md` (regenerate with the command
above against `results/testing_folder_smoke.jsonl` — the JSONL itself is
git-ignored as generated output, per the project's convention for runtime
artifacts). See that file for the real numbers: sample count, task
routing distribution, specialist distribution, error rate, latency
percentiles, confidence-availability rate. All of it is measured system
behavior, none of it is benchmark accuracy.

## If a real labeled benchmark is fetched later

1. Get explicit permission for the download (per this project's process).
2. Write a new adapter under `runners/` following
   `testing_folder_adapter.py`'s `generate_samples()` shape, but with
   `meta["ground_truth"]` populated from the real labels.
3. Extend `results/generate_summary.py` (or write a benchmark-specific
   summary script) to call the appropriate function in `metrics/` — they
   are already implemented and unit-tested, just never wired to real
   ground truth in this build.
4. Update this README's "What this is, honestly" section — don't leave
   it claiming no ground truth exists once one does.
