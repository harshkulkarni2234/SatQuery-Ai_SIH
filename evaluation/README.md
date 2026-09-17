# SatQuery AI — Evaluation Framework

Built for Phases B3–B6 of the original 3-person plan (`docs/satquery-master-build-spec.md`).

## What this is, honestly

The original plan (B4/B5/B6) called for downloading and running against
**RSVQA, CDVQA, and VRSBench** — official, labeled remote-sensing benchmark
datasets. Early in this build, when first reached, that didn't happen:
downloading those datasets needed explicit permission per this project's
process, and the instruction at the time was to use the local `testing/`
folder instead (see the "Unlabeled smoke test" section below — that
folder genuinely has zero label files, so it can only produce
routing/latency statistics, never accuracy).

**That has since changed for RSVQA.** The real, official SIH26167 problem
statement explicitly names RSVQA, VRSBench, and CDVQA as the prescribed
public benchmarks — so fetching them is no longer a workaround, it's
literally what's asked for. RSVQA-LR was fetched (with explicit
permission) and run for real — see "Real labeled benchmark: RSVQA-LR"
below for actual, computed accuracy numbers, not routing statistics.
CDVQA/VRSBench are pending the same explicit-permission step (CDVQA's
underlying images come from the SECOND dataset, a ~3.79GB Google-Drive-only
download with an unstated license — flagged separately before fetching).

## Real labeled benchmark: RSVQA-LR

**Source**: official Zenodo record [6344334](https://zenodo.org/records/6344334)
(Lobry et al.), **CC-BY-4.0** — see `evaluation/rsvqa_lr/README.md` for
the exact download commands. Only the official **test** split was used
(100 images, 10,004 labeled questions) — not train/val, since this is
evaluation-only.

**What was run**: the *existing, untrained* base SmolVLM-256M-Instruct
VQA specialist (no LoRA fine-tuning applied — that's what the RTX 2050
training session is for) against a fixed-seed stratified sample (60
questions per type, capped at the type's real total — 240 total) via the
real `/query` API, scored with `metrics/vqa_metrics.py` against the real
ground truth. Reproduce:

```bash
python -m evaluation.runners.run_rsvqa_lr_eval --per-type 60 \
    --output evaluation/results/rsvqa_lr_eval.jsonl
python -m evaluation.results.score_rsvqa_lr evaluation/results/rsvqa_lr_eval.jsonl
```

**Real results** (`evaluation/results/rsvqa_lr_score.json`, 0 errors across
240 real samples):

| Question type | n scored | Accuracy | Metric |
|---|---|---|---|
| presence (e.g. "Is there a road?") | 60 | **73.3%** | yes/no |
| comp (e.g. "Are there more X than Y?") | 60 | **65.0%** | yes/no |
| rural_urban | 59 (1 unparseable) | **40.7%** | categorical keyword match |
| count (e.g. "How many roads?") | 34 (26 unparseable) | **0% exact**, RMSE 206.6 | count regression |

This is a real **pre-training baseline** for the base model, not its
ceiling — the whole point of running it now is to have a genuine
before/after once the RTX 2050 session fine-tunes a LoRA adapter on this
same task family. Two honest things this run surfaced, not smoothed
over:
- The base model **cannot reliably count** — 26/60 count answers didn't
  even contain an extractable number, and the 34 that did were wildly
  off (RMSE 206.6 against real counts that go into the hundreds).
- A real scoring bug was caught and fixed while reviewing raw output:
  the first pass used literal string equality for rural_urban, which
  scored a genuinely correct free-text answer ("It is an urban area.")
  as wrong against ground truth "urban". Fixed with a new
  `categorical_accuracy` metric (extracts the matching keyword from free
  text, same approach as the existing `parse_yes_no`) — with its own
  unit test reproducing the exact bug. The corrected number (40.7%) is
  what's reported above; the broken first pass (1.7%) is not.

## Unlabeled smoke test: local `testing/` folder

The local `testing/` folder (inventoried in Phase B1 — see
`data/manifest.json`'s `unused_raw_material` entry) contains ~4,000 real
Sentinel-2 optical patches, 1 real Sentinel-1 SAR image, and a handful of
misc images — but **zero label files of any kind** (verified: no
`.json`/`.csv`/`.txt`/label file anywhere under it). There is no ground
truth in this data, so the run described here reports real
routing/latency/error statistics, never accuracy — do not read the
numbers in `results/SUMMARY.md` as correctness scores.

## Layout

```
evaluation/
├── runners/
│   ├── base.py                      # generic sample -> HTTP API -> JSONL runner (resumable, --limit)
│   ├── testing_folder_adapter.py    # unlabeled smoke-test adapter (see its docstring)
│   ├── run_testing_folder_eval.py   # CLI entry point for the smoke test
│   ├── rsvqa_lr_adapter.py          # REAL labeled adapter — official RSVQA-LR test split
│   └── run_rsvqa_lr_eval.py         # CLI entry point for the real RSVQA-LR eval
├── metrics/
│   ├── vqa_metrics.py               # exact-match, yes/no, categorical, count accuracy/RMSE —
│   ├── spatial_metrics.py           # box IoU/acc@0.5, pixel-level change F1/IoU   used for real
│   └── text_metrics.py              # lightweight BLEU-n / ROUGE-L                 on RSVQA-LR now
├── results/
│   ├── generate_summary.py          # unlabeled run -> SUMMARY.md (never fabricates accuracy)
│   ├── score_rsvqa_lr.py            # REAL labeled run -> real per-type accuracy JSON
│   └── *.jsonl, *_score.json, SUMMARY.md  # actual run output (JSONL git-ignored, scored JSON/MD committed)
├── rsvqa_lr/README.md               # official source, license, exact download commands
├── configs/                         # reserved for future labeled-benchmark configs (empty today)
├── tests/                           # unit tests for metrics/, hand-computed examples (25 tests)
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

25 tests, all with hand-computed expected values (not just "does it not
crash") — e.g. `pixel_mask_f1_iou` is tested against a manually-solved
2×2 mask where precision/recall/F1/IoU are all worked out by hand in the
test's own comment, and `categorical_accuracy` has a regression test
reproducing the exact free-text-matching bug found while scoring the
real RSVQA-LR run (see above).

## CDVQA and VRSBench (pending)

Both are named in the official problem statement. CDVQA's question/answer
labels are clean (official GitHub repo, Apache-2.0, no download friction),
but its images come from the SECOND dataset — Google-Drive-only, ~3.79GB,
license entirely unstated on the official page. VRSBench hasn't been
researched yet. Neither is fetched in this build as of this writing;
follow the same pattern as `rsvqa_lr_adapter.py`/`run_rsvqa_lr_eval.py`
once permission is given and the raw data is in hand:

1. Get explicit permission for the download (per this project's process
   — flag exact size and license before fetching, same as every other
   external dataset in this build).
2. Write a new adapter under `runners/` following `rsvqa_lr_adapter.py`'s
   shape (real ground truth in `meta`, documented stratified sampling if
   the full set is too slow to run serially).
3. Write a benchmark-specific scorer following `score_rsvqa_lr.py`'s
   shape, calling the appropriate function in `metrics/` — grounding
   uses `spatial_metrics.grounding_acc_at_iou`, captioning would need
   `text_metrics.bleu_n`/`rouge_l` (implemented, unit-tested, never yet
   run against real labeled captions).
