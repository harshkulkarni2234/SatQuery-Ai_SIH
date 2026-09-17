# Model Card: smolvlm256m-ben-lora-s3-v1.0

Phase B2 ("turn the Stage-3 LoRA experiment into a reproducible
component"). This card documents what is actually known and actually
verified about `ml/smolvlm/lora_stage3/` — nothing here is invented to
fill a gap; unknowns are marked `unknown` explicitly.

## Source dataset

**Unknown, with one unverified narrative claim.** No training log, config
file, git history, or dataset file for the original Stage 3 run exists
anywhere in this repository. The adapter's own `README.md`
(`ml/smolvlm/lora_stage3/README.md`) is an entirely unfilled Hugging Face
PEFT auto-generated template — every section reads `[More Information
Needed]`.

The only surviving narrative source is `ml/vqa-worker/README.md` section
9 (written before this solo build, presumably by whoever ran the original
training): it states the adapter *"was trained on a small (~64-example)
BigEarthNet slice"*. This claim is reproduced here **as reported, not
independently re-verified** — there is nothing to check it against.

`prepare_bigearthnet_vqa.py` (the original plan's item 1, a script to
*build* a question/answer dataset from real BigEarthNet land-cover
labels) was **not written**: the BigEarthNet imagery available in this
build (`testing/`, ~4,000 patches, see `data/manifest.json`) ships with
**zero label files of any kind** — verified directly (no
`.json`/`.csv`/`.txt`/label file anywhere under `testing/`). There is no
real label data in this repository to build such a script against.

## Model

- Base: `HuggingFaceTB/SmolVLM-256M-Instruct` (Idefics3 architecture, 256M params).
- Method: LoRA (PEFT) applied to a `CausalLM`-facade wrapper
  (`DecoderShim` in `ml/vqa-worker/model_provider.py`) over the base
  model's text decoder only — the vision tower is untouched. This
  wrapper shape exists because PEFT's `get_peft_model` needs a plain
  `forward()`/`prepare_inputs_for_generation()` interface that Idefics3's
  bare decoder submodule doesn't expose directly.
- Attaches to the same loaded base model in place (no duplicated model
  copy); toggled per-request via `peft`'s `disable_adapter()` context
  manager.

## LoRA configuration (real — from `adapter_config.json`)

| Field | Value |
|---|---|
| `r` | 8 |
| `lora_alpha` | 16 |
| `lora_dropout` | 0.1 |
| `bias` | none |
| `task_type` | CAUSAL_LM |
| `target_modules` | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

See `ml/adaptation/configs/stage3.yaml` for the same values plus the
training-hyperparameter fields marked `unknown` (learning rate, steps,
batch size, seed, precision, image size, training hardware — none of
these are recoverable from anything in this repo).

## Evaluation (real, but not accuracy — read this section before quoting a number)

**No labeled held-out split exists** (see "Source dataset" above), so
this build cannot report per-question-type accuracy/exact-match the way
the original plan describes. What was actually run and is real:

`ml/adaptation/eval_lora.py` runs the same real image through both base
and specialist modes, for the same prompts, and reports whether their
answers agree — a real, reproducible statistic, not an accuracy score.
Reproduce with:

```bash
ml/vqa-worker/venv/bin/python ml/adaptation/eval_lora.py \
    --images testing/S2A_MSIL2A_20170720T100031_N9999_R122_T34UDG_65_03.png \
              testing/S2A_MSIL2A_20170720T100031_N9999_R122_T34UDG_66_00.png \
    --output ml/adaptation/results/eval_run.json
```

Real result from this exact run (`ml/adaptation/results/eval_run.json`,
git commit recorded in that file), n=8 (2 images × 4 prompts — a tiny
sample, stated as such, not a claim of statistical significance):

| Question type | n | Specialist ran | Exact agreement with base |
|---|---|---|---|
| presence | 4 | 4/4 | 0/4 |
| count | 2 | 2/2 | 0/2 |
| description | 2 | 2/2 | 0/2 |

Specialist activation rate: 1.00 (loads and runs without error on every
call in this run — this was previously **unverified** in this build;
Phase B2's GPU re-check confirmed it for the first time on real hardware,
Apple M2 via MPS). Exact agreement rate with base: 0.00 (expected — the
specialist gives short, direct answers by design; see below).

One real, observed example from that run (not cherry-picked for effect —
it's row 4 of the JSON): asked *"What can you tell me about this
image?"* on a real 120×120 BigEarthNet optical patch, the **base** model
hallucinated *"a close-up view of a light-colored... surface... possibly
glass or a polished metal"* (256M-param general VLMs are known to
struggle badly on tiny, low-context satellite crops), while the
**specialist** answered *"A high-resolution image of a green and blue
gradient background"* — shorter and closer to the actual patch content,
though still not a real land-cover description. Neither answer was
checked against ground truth (none exists); this is a single qualitative
observation, not a claim that the specialist is generally better.

**This does NOT confirm or refute** `ml/vqa-worker/README.md`'s inherited
claim that "captions regressed (became empty/terse)" under the
specialist — n=8 with no ground truth cannot support or contradict a
claim about caption quality either way. That claim is carried forward
in this build only as an attributed, unverified statement from before
this session, not as something this evaluation re-derived.

## Supported question types (narrow, unchanged from the original design)

Specialist mode is gated to only presence/existence questions ("is there
X", "does the image contain X") and simple counting questions ("how many
X") — see `should_use_specialist()` in both `model_provider.py` and
`backend/app/services/vqa.py` (kept in sync). Everything else (captions,
descriptions, spatial/grounding language, change/comparison language,
optical/SAR language) always uses the base model. Any specialist
inference failure falls back to base automatically (see
`model_provider.py`'s `answer_question`).

## Verified working, for the first time in this build (Phase B2/B8 GPU re-check)

Before this phase, this build's own documentation incorrectly stated "no
GPU" for the whole project — that conflated "no CUDA GPU" with "no GPU at
all". This development machine has an Apple M2 GPU, and PyTorch's MPS
(Metal) backend was verified to work: `torch.backends.mps.is_available()`
is `True`, a real matrix multiply ran on it, and — most relevantly here —
**this exact adapter loads and runs inference successfully on it**,
`specialist_available: true`, `specialist_errors: []`. This was never
tested in this build until now; `model_provider.py`'s device selection
previously only ever checked for CUDA and silently ran on CPU on this
machine. Fixed alongside this evaluation (see `model_provider.py` and its
new `ml/vqa-worker/test_model_provider.py`).

Real measured latency on this machine (MPS, fp32 — fp16 is deliberately
not used on MPS, see `model_provider.py`'s comment): base mode ~3-12s
per request depending on answer length, specialist mode ~3-4s (shorter
answers). Not benchmarked against the original CUDA target machine
(a ~4GB T600, per `ml/vqa-worker/README.md`) in this build.

## Limitations

- Training provenance is unverifiable (see "Source dataset"). Do not
  present this adapter's origin with more confidence than "reportedly
  trained on ~64 examples, unconfirmed."
- No labeled evaluation set exists to validate output quality
  numerically. The evaluation in this card is a base-vs-specialist
  agreement/activation check, not an accuracy benchmark.
- Narrow scope by design: only presence/count question types.
- 256M-parameter base model visibly struggles with tiny (120×120),
  low-context satellite crops in both modes — see the real example
  above. Do not present this system's VQA output as reliable land-cover
  classification.

## Version string

`smolvlm256m-ben-lora-s3-v1.0` — read at runtime from
`ml/smolvlm/lora_stage3/version.json` by `model_provider.py`
(`_read_specialist_version`), not hardcoded in worker code, and returned
in every `/vqa` response's `model_version` field alongside `adapter_used`
(bool) and `reason` (why base vs. specialist was used for that request).
