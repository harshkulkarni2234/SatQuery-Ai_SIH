# CDVQA (real, labeled bi-temporal change-VQA benchmark)

- **Source**: Yuan, Mou, Xiong, Zhu — "Change Detection Meets Visual
  Question Answering", IEEE TGRS 2022. Official repo:
  [YZHJessica/CDVQA](https://github.com/YZHJessica/CDVQA)
- **License**: **Apache-2.0** (the repo itself is licensed this way — a
  clean, permissive, standard license, verified via the GitHub API).
- **What's used**: only the official **Test** split's question/answer
  labels (968 real bi-temporal image pairs, ~39,700 questions across 8
  types: change_or_not, increase_or_not, decrease_or_not, change_to_what,
  smallest_change, largest_change, change_ratio, change_ratio_types). A
  `Test2` split also exists (same 968 pairs, paraphrased questions) —
  not used by default.
- **Images**: CDVQA's labels reference filenames from the **SECOND**
  dataset (`evaluation/second_dataset/README.md`) — that download is
  separate, larger, and has its own (unstated) license, approved
  separately by the user before fetching.

## Getting the raw data (not committed — see `.gitignore`)

```bash
mkdir -p evaluation/cdvqa/raw
cd evaluation/cdvqa/raw
for f in Test_questions.json Test_answers.json Test_images.json \
         Test2_questions.json Test2_answers.json Test2_images.json; do
  curl -sL --retry 3 --retry-delay 5 -m 60 -o "$f" \
    "https://raw.githubusercontent.com/YZHJessica/CDVQA/main/$f"
done
```

Verify sizes: `Test_questions.json` ~7.9MB, `Test_answers.json` ~4.2MB,
`Test_images.json` ~2.0MB.

## Running the evaluation

Requires `evaluation/second_dataset/raw/im1/` and `im2/` to exist first
(see that directory's README).

```bash
python -m evaluation.runners.run_cdvqa_eval \
    --api-base-url http://127.0.0.1:8000 \
    --per-type 30 \
    --output evaluation/results/cdvqa_eval.jsonl
python -m evaluation.results.score_cdvqa evaluation/results/cdvqa_eval.jsonl
```

`--per-type 30` draws a documented, seeded, stratified sample (up to 30
per question type across the 8 types = up to 240 total) rather than the
full ~39,700-question set, for the same real-inference-time reason as
`rsvqa_lr/README.md` — each query here also runs the deterministic
change-detection pipeline over a full 512×512 bi-temporal pair, which is
slower per-sample than single-image VQA.

## Expected result, stated honestly before running it

SatQuery's `change.deterministic_cv` specialist measures **aggregate**
pixel/area change across a geographic-reprojection-aligned pair — it has
no per-land-cover-class semantic understanding at all. Questions like
"What have the regions of buildings mainly changed to?" ask for exactly
the kind of semantic classification this specialist was never designed
to produce. Expect near-floor scores on the categorical question types
(`change_to_what`, `smallest_change`, `largest_change`) — that is the
real, honest capability gap this benchmark exists to surface, not
something to be smoothed over in how the results are reported.
