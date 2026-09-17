# ml/adaptation — Stage-3 LoRA reproducibility (Phase B2)

## What exists here, and why the rest doesn't

The original plan asked for three scripts:

1. **`prepare_bigearthnet_vqa.py`** — builds a question/answer dataset
   from BigEarthNet labels. **Not written.** The BigEarthNet-style
   imagery available in this build (`testing/`, ~4,000 patches) has
   **zero label files of any kind** — verified directly, no
   `.json`/`.csv`/`.txt`/label file anywhere under it (see
   `data/manifest.json`). There is no real label data in this repo to
   build this script against. Writing it against invented/placeholder
   labels would violate this project's core honesty rule (never fabricate
   data presented as real), so it was left unwritten rather than faked.
2. **`train_lora.py`** — trains SmolVLM-256M-Instruct + LoRA. **Not
   written**, for the same reason as above: with no real Q&A training
   set, there is nothing genuine to train against. Writing an
   unrunnable/untested training script would not be a real, reproducible
   component — it would just be more surface area claiming capability
   that was never verified.
3. **`eval_lora.py`** — evaluates BASE vs ADAPTER. **Written and
   actually run** (see below and `MODEL_CARD.md`) — this one doesn't
   need labeled ground truth, it compares the two modes' real outputs to
   each other.

This is the same scope pattern as `evaluation/`'s B3-B6 rows: real
infrastructure was built for what's genuinely possible without labeled
data, and what isn't possible without labeled data is stated plainly
rather than faked.

**If real BigEarthNet labels become available** (e.g. the official
per-patch metadata for the exact tiles already in `testing/`:
`T34UDG/2017-07-20` and `T35ULA/2017-08-08`), `prepare_bigearthnet_vqa.py`
and `train_lora.py` should be written then, following the original plan's
spec (question templates from label→answer mapping, YAML-configured
training via `ml/adaptation/configs/stage3.yaml`, fp16/bf16 + gradient
checkpointing). Nothing about this build makes that harder later.

## What's real here today

- **`configs/stage3.yaml`** — the existing adapter's real LoRA
  hyperparameters (from `adapter_config.json`), with every training
  hyperparameter that isn't recoverable from anything in this repo
  marked `unknown` explicitly.
- **`eval_lora.py`** — runs the same real image through base and
  specialist modes for the same prompts and reports real, measured
  agreement/activation statistics (not accuracy — see its own docstring
  and `MODEL_CARD.md` for exactly why). Requires the `ml/vqa-worker`
  virtual environment:
  ```bash
  ml/vqa-worker/venv/bin/python ml/adaptation/eval_lora.py \
      --images <real image paths> \
      --output ml/adaptation/results/eval_run.json
  ```
- **`results/eval_run.json`** — a real run of the above, committed as
  evidence (git commit hash and dataset hash recorded inside it).
- **`MODEL_CARD.md`** — the full model card: real LoRA config, honestly
  `unknown` training provenance (with the one inherited, unverified
  narrative claim clearly attributed), the real evaluation result above,
  and the real confirmation (new in this build) that the adapter loads
  and runs successfully on this machine's Apple M2 GPU via PyTorch's MPS
  backend — previously untested, since `model_provider.py` only ever
  checked for CUDA and silently fell back to CPU.
- **`ml/vqa-worker/model_provider.py`** and **`ml/smolvlm/lora_stage3/version.json`**
  were updated alongside this: the worker now detects MPS (not just
  CUDA), and reads its specialist version string from `version.json`
  (pointing at this card) rather than a hardcoded string — the `/vqa`
  response now includes `adapter_used` and `reason` fields, per this
  phase's spec item 5.
