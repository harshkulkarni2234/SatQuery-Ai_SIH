# RSVQA-LR (real, labeled benchmark)

Unlike `testing_folder_adapter.py` (Phase B3-B10's unlabeled smoke test),
this is a genuine, officially labeled remote-sensing VQA benchmark —
real ground-truth answers exist, so a real accuracy number can be
computed for the first time in this build.

- **Source**: Lobry et al., "RSVQA: Visual Question Answering for Remote
  Sensing Data" — official data release: [Zenodo record 6344334](https://zenodo.org/records/6344334)
- **License**: CC-BY-4.0 (stated on the Zenodo record itself — a clean,
  standard open-data license, no ambiguity)
- **What's used**: only the official **test** split (100 images,
  10,004 labeled questions across 4 types: presence, count, comparison,
  rural/urban) — not the training or validation splits, since this is
  for evaluation only.

## Getting the raw data (not committed — see `.gitignore`)

```bash
mkdir -p evaluation/rsvqa_lr/raw
cd evaluation/rsvqa_lr/raw
for f in Images_LR.zip LR_split_test_answers.json LR_split_test_images.json LR_split_test_questions.json; do
  curl -sL --retry 3 --retry-delay 5 -m 300 -o "$f" "https://zenodo.org/records/6344334/files/$f?download=1"
done
unzip -q Images_LR.zip
```

Verify: `Images_LR.zip` should be exactly 95,008,155 bytes (773 real
`.tif` files inside); `LR_split_test_questions.json` should be ~2.6MB.
If a download comes back suspiciously small, it's a truncated/504
response from Zenodo — retry it (this happened once during this
project's own fetch).

## Running the evaluation

```bash
python -m evaluation.runners.run_rsvqa_lr_eval \
    --api-base-url http://127.0.0.1:8000 \
    --per-type 60 \
    --output evaluation/results/rsvqa_lr_eval.jsonl
python -m evaluation.results.score_rsvqa_lr evaluation/results/rsvqa_lr_eval.jsonl
```

`--per-type 60` draws a fixed-seed (`--seed 42` default), stratified
sample of up to 60 questions per type (240 total across the 4 types,
since `rural_urban` only has 100 total anyway — capped there). The full
10,004-question test set is not run by default: at real per-query VQA
latency (single serial inference slot, ~3-12s/query measured on this
machine's GPU), the full set would take on the order of 14 hours. This
is documented, not hidden — see `runners/rsvqa_lr_adapter.py`'s
docstring. Pass a larger `--per-type` (or write a version with no cap)
if you want closer to the full set and have the time.
