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

**That has since changed for all three.** The real, official SIH26167
problem statement explicitly names RSVQA, VRSBench, and CDVQA as the
prescribed public benchmarks — so fetching them was no longer a
workaround, it's literally what's asked for. All three were fetched
(with explicit permission at each step, real sizes/licenses disclosed
first) and run for real against the live system — see the sections
below for actual, computed numbers, not routing statistics.

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

## Real labeled benchmark: CDVQA (bi-temporal change-VQA)

**Source**: official GitHub repo [YZHJessica/CDVQA](https://github.com/YZHJessica/CDVQA)
(Apache-2.0 labels) over images from the **SECOND** dataset
([captain-whu.github.io/SCD](https://captain-whu.github.io/SCD/), license
unstated on the official page — fetched with explicit user go-ahead after
disclosing the real ~3.79GB size and that caveat). See
`evaluation/cdvqa/README.md` and `evaluation/second_dataset/README.md` for
exact download commands (including a real gotcha hit during this fetch:
Google Drive's large-file bypass silently truncated the download once
mid-transfer with no error — verify with `unzip -t`, don't trust the
reported size alone).

Real result (`evaluation/results/cdvqa_score.json`, 120 real bi-temporal
queries, 19 errors — see below), seeded stratified sample (15/type):

| Question type | Accuracy | What actually happened |
|---|---|---|
| change_or_not / increase_or_not / decrease_or_not | **0% scored (all unparseable)** | SatQuery's `change.deterministic_cv` answer is always a descriptive sentence about coverage % and location — it never states a literal "yes"/"no", so these three types couldn't be scored at all. Not a metric bug (verified by reading the raw answers) — a real capability gap: the specialist has no per-class change/no-change verdict. |
| smallest_change / largest_change / change_to_what | **0% scored (all unparseable)** | Same root cause as B8's whole rationale: no per-land-cover-class semantic understanding, only aggregate change measurement. Expected, stated up front in `cdvqa/README.md` before this run. |
| change_ratio | **14.3%** (7/13 scored) | The one place our system's real stated percentage could be extracted and bucketed — see the `percentage_bucket_accuracy` fix above. |
| change_ratio_types | **37.5%** (8/15 scored) | Same mechanism, best-performing category. |

19/120 samples errored with `"Two images supplied without a change-related
question"` — some of CDVQA's real question phrasings don't trigger
SatQuery's keyword-based change-detection routing at all, a second real
routing gap alongside the semantic one.

## Real labeled benchmark: VRSBench (VQA, referring/grounding, captioning)

**Source**: official HuggingFace dataset
[xiang709/VRSBench](https://huggingface.co/datasets/xiang709/VRSBench),
**CC-BY-4.0** (clean, verified via the HF API). See
`evaluation/vrsbench/README.md`.

### VQA (`evaluation/results/vrsbench_vqa_score.json`, 180 real samples, seeded 15/type)

| Type | Accuracy | Metric |
|---|---|---|
| object existence | **86.7%** | yes/no |
| object color | 46.7% | contains-ground-truth |
| reasoning | 46.7% | contains-ground-truth |
| object quantity | 45.5% exact (11/15 scored), RMSE 1.41 | count |
| rural or urban | 60.0% | contains-ground-truth |
| object position | 45.5% | contains-ground-truth |
| object size | 36.4% | contains-ground-truth |
| scene type | 33.3% | contains-ground-truth |
| object category | 13.3% | contains-ground-truth |
| object shape | 13.3% | contains-ground-truth |
| image | 13.3% | contains-ground-truth |
| object direction | **6.7%** | contains-ground-truth |

A real bug was caught and fixed running this: VRSBench's ground truth
mixes digit ("3") and spelled-out ("Three", "Single") number forms for
the same "object quantity" type — a bare `float()` call crashed the
whole scoring run. `parse_count` (`metrics/vqa_metrics.py`) now handles
both forms, with a regression test reproducing the exact crash.

### Referring/grounding (`evaluation/results/vrsbench_referring_score.json`, 51 real samples)

**0 out of 51 samples ever reached the GROUNDING specialist — 100%
misrouted to VQA.** This was verified live (a real referring expression
sent through the actual running backend) *before* the scorer was even
written, not discovered after the fact: SatQuery's grounding specialist
supports exactly 5 land-cover targets (water/vegetation/built-up/roads/
farmland); VRSBench's 26 real object classes (vehicle, ship, airplane,
harbor, tennis-court, ...) have zero vocabulary overlap. No IoU number is
reported as a headline result — reporting one would misrepresent a task
that was never actually attempted.

### Captioning (`evaluation/results/vrsbench_captioning_score.json`, 30 real samples)

| Metric | Score |
|---|---|
| BLEU-1 | 9.8% |
| BLEU-4 | 0.2% |
| ROUGE-L F1 | 12.2% |

The one VRSBench task with no ontology mismatch (reuses the ordinary
description-style VQA path) — low but real scores, consistent with
MODEL_CARD.md's earlier finding that the 256M-param base model visibly
struggles on remote-sensing imagery without fine-tuning.

## What all five real evaluations add up to

Across RSVQA-LR, CDVQA, and VRSBench's three tasks, a consistent, honest
picture: **single-image VQA works, with real but modest accuracy on an
untrained base model** (presence/existence in the 70-87% range, most
open-vocabulary attributes in the 10-50% range); **grounding only covers
5 land-cover targets and cannot attempt object-level referring at all**;
**change detection can state an aggregate percentage but has no semantic
or per-class understanding**. None of this is smoothed over — it's the
real, current baseline this build measured, ready to compare against
once real fine-tuning (BigEarthNet LoRA) happens.
