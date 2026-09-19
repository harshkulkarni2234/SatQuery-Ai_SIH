---
base_model: HuggingFaceTB/SmolVLM-256M-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- remote-sensing
- bigearthnet
- satquery
---

# SmolVLM-256M-Instruct + BigEarthNet LoRA (Stage 3, v2.0)

LoRA adapter (2,442,240 trainable params) trained for this build on real
BigEarthNet v2.0 land-cover labels. See
[`ml/adaptation/MODEL_CARD.md`](ml/adaptation/MODEL_CARD.md) for the full,
honest record (training provenance, dataset provenance, held-out val and
RSVQA-LR benchmark results). This README mirrors the key facts; the MODEL
CARD is authoritative.

## Model Details

- **Developed by:** SatQuery AI build (Phase B2), solo execution.
- **Base model:** `HuggingFaceTB/SmolVLM-256M-Instruct` (Idefics3, 256M).
- **Method:** LoRA (PEFT) on the text decoder only via `DecoderShim`
  (see `ml/vqa-worker/model_provider.py`); vision tower untouched.
- **LoRA config:** r=8, alpha=16, dropout=0.1, bias none,
  targets q/k/v/o/gate/up/down proj (7 modules).
- **Version:** `smolvlm256m-ben-lora-s3-v2.0` (see `version.json`).

## Training Data

Real BigEarthNet v2.0 labels (Zenodo `metadata.parquet`,
CDLA-Permissive-1.0), matched to this repo's `testing/` imagery
(4,008/4,011 patches), 17 classes present. 25,645 Q/A pairs; the model
was trained on 2,400 rows of the deterministic train split (seed 42).
Builder: `ml/adaptation/prepare_bigearthnet_vqa.py`.

## Training Procedure

Script: `ml/adaptation/train_lora.py`, config `training_v2` in
`ml/adaptation/configs/stage3.yaml`. Full per-step log in
`training_log.json` (4,363 lines).

- epochs 3, max_train_rows 2400, batch size 2, grad accum 8
  (effective 16), lr 2e-4, warmup 0.05, seed 42, fp16, 450 steps,
  final loss ~0.449.
- Answer-token-only loss masking; resumable checkpoints.
- Hardware: NVIDIA GeForce RTX 2050 (4096 MiB VRAM), CUDA.

## Evaluation

Held-out val (200 rows: 150 presence + 50 count): presence 99.33%,
count 96.0% exact match. Live RSVQA-LR (240-question official test split):
overall 56.2% vs 50.2% base, but **mixed per-type** — improves
comparative/rural-urban/count-exact, regresses presence, count RMSE far
worse (overstated guesses). See the MODEL CARD before quoting anything.

## Limitations

Do not present this adapter's output as reliable land-cover
classification. The 256M base struggles on tiny 120×120 crops; RSVQA-LR
transfer is mixed (see MODEL CARD). Specialist mode only handles
presence/count question types (`should_use_specialist`).