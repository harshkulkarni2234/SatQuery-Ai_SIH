# VRSBench (real, labeled single-image benchmark: captioning, grounding, VQA)

- **Source**: official HuggingFace dataset — [xiang709/VRSBench](https://huggingface.co/datasets/xiang709/VRSBench)
- **License**: **CC-BY-4.0** (verified via the HF API — clean, permissive, no ambiguity)
- **What's used**: only the **EVAL** (validation) split files — not the
  8.36GB `Images_train.zip` (training images, not needed for evaluation).
  - `VRSBench_EVAL_vqa.json` (37,409 real VQA questions, 12 real types:
    object existence/quantity/position/category/color/shape/size/
    direction, scene type, rural_or_urban, reasoning, image)
  - `VRSBench_EVAL_referring.json` (16,159 real grounding/referring
    expressions with real bounding-box ground truth, format
    `{<x1><y1><x2><y2>}` on a 0-100 normalized scale)
  - `VRSBench_EVAL_Cap.json` (9,350 real detailed image captions)
  - `Images_val.zip` (**~3.7GB** — the real images referenced by all
    three eval files above; the val split isn't separable further
    without downloading the whole zip)

## Getting the raw data (not committed — see `.gitignore`)

```bash
mkdir -p evaluation/vrsbench/raw
cd evaluation/vrsbench/raw
for f in VRSBench_EVAL_Cap.json VRSBench_EVAL_referring.json VRSBench_EVAL_vqa.json; do
  curl -sL --retry 3 --retry-delay 5 -m 60 -o "$f" \
    "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/$f"
done
curl -sL --retry 3 --retry-delay 5 -o Images_val.zip \
  "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Images_val.zip"
unzip -q Images_val.zip
```

## Status

Labels + images fetched (with explicit user permission — the whole
CC-BY-4.0 dataset, real size disclosed as ~3.7GB before pulling it).
Runner/adapter/scorer for VQA, referring, and captioning not yet written
as of this commit — see `docs/SOLO_PROGRESS.md`'s B6 row for current
status. Planned to follow the same pattern as
`runners/rsvqa_lr_adapter.py`/`runners/cdvqa_adapter.py`:
- VQA: per-type scoring, reusing `metrics/vqa_metrics.py` where the
  answer space fits (yes/no for "object existence", numeric for
  "object quantity", categorical/free-text for the rest).
- Referring (grounding): parse the `{<x1><y1><x2><y2>}` ground-truth
  format into a real box, compare against SatQuery's own grounding
  specialist's output box using `metrics/spatial_metrics.grounding_acc_at_iou`
  (already implemented, unit-tested, never yet run against real boxes).
- Captioning: `metrics/text_metrics.bleu_n`/`rouge_l` against the real
  reference caption (already implemented, unit-tested, never yet run
  against real captions).
