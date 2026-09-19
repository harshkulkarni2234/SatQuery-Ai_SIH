# Model Card: smolvlm256m-ben-lora-s3-v2.0 (current) / v1.0 (legacy)

Phase B2 end-state. This card documents what is actually known and
actually verified about the Stage-3 BigEarthNet LoRA adapters in this
repository — nothing here is invented to fill a gap; unknowns are marked
`unknown` explicitly.

## The two adapters

| | `smolvlm256m-ben-lora-s3-v2.0` (current) | v1.0 (legacy, `ml/smolvlm/lora_stage3/`) |
|---|---|---|
| Provenance | **Real and verifiable** — trained this build | **Unknown** (mystery provenance, see below) |
| Training log | `ml/smolvlm/lora_stage3_v2/training_log.json` (4,363 lines) | None exists |
| Dataset | Real BigEarthNet v2.0 (see next section) | Reported "~64-example slice" — unverified |
| Status | Served by default (`DEFAULT_ADAPTER_DIR` / `ADAPTER_DIR`) | Kept intact as fallback; not served by default |

From here on, "the adapter" / "v2.0" refers to the currently-served
`ml/smolvlm/lora_stage3_v2/`.

## Source dataset (v2.0 — real)

Built by `ml/adaptation/prepare_bigearthnet_vqa.py` from **official
BigEarthNet v2.0 labels** — Zenodo `metadata.parquet` (CDLA-Permissive-1.0
license), not reconstructed or recollected:

- 4,008 / 4,011 local `testing/` patches matched against the official
  label table via patch co-ordinates + timestamp; 3 dropped for label
  table ambiguity (documented in that script).
- All matched patches are in the **official train split** of BigEarthNet.
- 17 of 19 official classes present in the matched set (the 2 absent
  classes never appear in matched patches).
- 25,645 Q/A pairs (`ml/adaptation/dataset/{train,val,test}.jsonl`):
  21,637 presence ("is there X" yes/no) + 4,008 count ("how many X").
- Deterministic 80/10/10 split, seed 42 → 20,532 / 2,567 / 2,546 rows.
- Dataset stats: `ml/adaptation/dataset/stats.json`.

## Model (v2.0)

- Base: `HuggingFaceTB/SmolVLM-256M-Instruct` (Idefics3 architecture, 256M params).
- Method: LoRA (PEFT) applied to a `CausalLM`-facade wrapper
  (`DecoderShim` in `ml/vqa-worker/model_provider.py`) over the base
  model's text decoder only — the vision tower is untouched.
- Attaches to the same loaded base model in place; toggled per-request
  via `peft`'s `disable_adapter()` context manager.
- Trainable parameters: **2,442,240** (LoRA, 7 target modules).

## LoRA configuration (real — from `adapter_config.json`)

| Field | Value |
|---|---|
| `r` | 8 |
| `lora_alpha` | 16 |
| `lora_dropout` | 0.1 |
| `bias` | none |
| `task_type` | CAUSAL_LM |
| `target_modules` | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

## Training (v2.0 — real)

Trained with `ml/adaptation/train_lora.py` using
`ml/adaptation/configs/stage3.yaml` `training_v2:`:

| Field | Value |
|---|---|
| epochs | 3 |
| max train rows | 2400 (from `dataset/train.jsonl`) |
| batch size | 2 |
| gradient accumulation | 8 (effective batch 16) |
| learning rate | 2e-4 |
| warmup fraction | 0.05 |
| seed | 42 |
| precision | fp16 |
| image size | 120×120 (native BigEarthNet patch; processor resizes internally) |
| hardware | NVIDIA GeForce RTX 2050, 4096 MiB VRAM, CUDA |
| steps | 450 |
| final loss | ~0.449 |

Answer-token-only loss masking (prompt tokens masked, `-100`), resumable
checkpointing, deterministic seed. Full per-step log in
`ml/smolvlm/lora_stage3_v2/training_log.json`.

### Held-out validation (real, from training run)

On the deterministic val split (200-row subset: 150 presence + 50 count),
answer exact-match:

| Question type | n | Exact match |
|---|---|---|
| presence | 150 | 149/150 = 99.33% |
| count | 50 | 48/50 = 96.0% |

Note: these are exact-string matches on synthetically generated labels
(short canned answers) — the model is being tested on the same question
format it was trained with. They do **not** transfer unchanged to
RSVQA-LR (see the live benchmark below).

## Live benchmark: RSVQA-LR (real official test split, same runner as baseline)

`evaluation/results/rsvqa_lr_score_v2.json` vs the base-model baseline
`evaluation/results/rsvqa_lr_score.json` — same 240-question stratified
test sample, same HTTP runner, seed 42. Both runs: 0 API errors.

| Question type | Base model | v2.0 adapter | Metric |
|---|---|---|---|
| presence | 73.3% | 70.0% | yes/no accuracy |
| comparative | 65.0% | 70.0% | yes/no accuracy |
| rural/urban | 40.7% | 61.7% | urban/rural keyword match |
| count (exact) | 0.0% | 5.1% | exact integer match |
| count (RMSE) | 206.6 | 2,264,550 | RMSE over parsed counts |
| approx. overall | 50.2% | 56.2% | weighted mix per `score_rsvqa_lr.py` |

Interpretation (be honest — do not over-claim):

- The adapter **improves** comparative (+5.0 pts), rural/urban (+21.0 pts),
  count exact match (0 → 5.1%), and approx. overall (+6.0 pts).
- It **slightly regresses** presence (−3.3 pts).
- Count RMSE is **far worse** (2.26M vs 207): 21/60 unparseable, and
  several wildly overblown numeric guesses (e.g. "1000"/"10000000"
  against ground truth 6–33) dominate the RMSE. Exact-match improved but
  the adapter's count behaviour on RSVQA-LR is not reliable.
- Therefore the v2.0 adapter is **not universally better** than base on
  RSVQA-LR. Present it as a mixed, honest result, not a win.

## Specialist question-type gating (narrow, unchanged)

Specialist mode is gated to presence/existence and simple counting
questions — see `should_use_specialist()` in `model_provider.py` and
`backend/app/services/vqa.py`. Other question types always use the base
model. Any specialist inference failure falls back to base automatically.

## Verification on GPU (v2.0)

`SmolVLMProvider` with `ADAPTER_DIR=ml/smolvlm/lora_stage3_v2` loads on
CUDA (RTX 2050): `specialist_available=true`, `specialist_errors=[]`,
reported `model_version = smolvlm256m-ben-lora-s3-v2.0`, served through
both the worker `/health` and per-request `/vqa` responses. Base adapter
is disabled per request (not a duplicated model copy).

## Legacy v1.0 (mystery provenance — kept, not served)

The original `ml/smolvlm/lora_stage3/` adapter: **no training log, config,
dataset, or git history exists for it**; its own `README.md` is an
unfilled HF PEFT template. The only surviving narrative claim (from
`ml/vqa-worker/README.md` section 9, written before this build) is that it
was *"trained on a small (~64-example) BigEarthNet slice"*. Reproduced as
**reported, not verified** — nothing in this repo supports it. It remains
in the repo for A/B comparison but is not the default.

## Limitations

- 256M-parameter base model struggles with tiny (120×120) low-context
  satellite crops; do not present VQA output as reliable land-cover
  classification.
- Val exact-match numbers reflect the trained question format; live
  RSVQA-LR shows mixed transfer (see table above).
- Count RMSE on RSVQA-LR is dominated by overstated numeric guesses —
  count output is unreliable on that benchmark.
- v2.0 training used 2,400 rows of the 20,532-row train split (practical
  GPU-time limit). A fully-trained adapter (20k rows) may behave
  differently; nothing here claims otherwise.

## Version strings

- v2.0: `smolvlm256m-ben-lora-s3-v2.0` — read at runtime from
  `ml/smolvlm/lora_stage3_v2/version.json` by `model_provider.py`
  (`_read_specialist_version`), returned in every `/vqa` response's
  `model_version` alongside `adapter_used` (bool) and `reason`.
- v1.0: `smolvlm256m-ben-lora-s3-v1.0` (`ml/smolvlm/lora_stage3/version.json`).