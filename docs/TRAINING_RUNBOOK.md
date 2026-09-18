# Training Runbook — GPU Handoff to opencode

A phase-by-phase plan for handing SatQuery AI's two real training jobs to
**opencode** on the RTX 2050 laptop. Every phase ends at a checkpoint: you
paste opencode's real output back to Claude, Claude reviews it and hands
you the next prompt. Nothing later in the plan runs unless the phase
before it actually produced what it says it should.

An interactive, checkbox-tracked version of this same plan is published at:
https://claude.ai/artifact/UmECNwKSMotrp5hzaj65U5

## The loop

1. **You** — copy the phase's prompt block below into opencode.
2. **opencode** — does the work, produces real output (logs, numbers, files).
3. **You** — paste that output to Claude: "here's opencode's output for Phase N."
4. **Claude** — reviews it, tells you go / fix / retry, hands you the next prompt.

Where a phase's prompt references an earlier phase's output (a file path,
a real number), opencode already has it on disk from the previous step —
you don't need to copy it in by hand. If opencode's output ever
contradicts something stated here as fact, trust what opencode actually
found on the real machine, and bring the discrepancy to Claude rather
than resolving it yourself.

---

## Phase 0 — Setup: verify the machine and the repo

**Goal:** confirm the GPU actually works, and check whether the SECOND
dataset (Task A's training data) is already on disk from the folder copy.
**Gate:** none — just verification, no risk.

**Prompt to paste into opencode:**

```
Verify this machine is ready for GPU training work on the SatQuery AI project. Do not start any training or downloads yet — this is a verification pass only.

1. Run `nvidia-smi` and report the GPU model and real VRAM (don't assume a number).
2. Check whether torch is installed with CUDA support:
   `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"`
   If it's missing or CPU-only, install the correct CUDA build for this machine (check pytorch.org's install matrix for the right command — don't guess a version).
3. Check whether `evaluation/second_dataset/raw/SECOND_train_set/{im1,im2,label1,label2}/` already exists in this project folder (it may have been included in a folder copy even though it's gitignored). If present, count the files in each with `ls im1 | wc -l` (expect 2968 in each of the four folders).
4. Read in full: AGENTS.md, docs/SOLO_PROGRESS.md, evaluation/README.md, ml/adaptation/README.md, ml/adaptation/MODEL_CARD.md, ml/change_model/SELECTION.md, evaluation/second_dataset/README.md, ml/vqa-worker/model_provider.py, backend/app/services/registry.py, backend/app/services/grounding.py.
5. Report back: (a) real GPU + VRAM + torch/CUDA versions, (b) whether SECOND data is already present and complete, (c) a short summary in your own words of what SatQuery AI is and what the two training tasks ahead are — so I can confirm you actually understood the context, not just skimmed it.
```

**Expect back:** GPU/torch report, SECOND-data presence, opencode's own
summary of the project.

**Bring to Claude:** paste the whole report. Claude confirms the GPU is
real and usable, tells you whether Task A can start immediately (data
present) or needs a re-download first, and corrects anything opencode
misunderstood before real work begins.

---

## Task A — Real change-detection model

Started first — its data is already local, no permission gate blocks it.

### Phase 1 — Split the data: exclude CDVQA's test pairs

**Goal:** carve out a train/val split from SECOND's 2,968 pairs that
excludes the exact 968 pairs already used as CDVQA's test set — otherwise
the before/after comparison later is meaningless.
**Gate:** correctness checkpoint.

**Prompt to paste into opencode:**

```
Build the real train/validation split for Task A (change-detection model), with NO leakage from CDVQA's existing test set.

1. Import evaluation/runners/cdvqa_adapter.py and call load_real_test_set() to get the real list of 968 file names CDVQA's test split already uses (these live under evaluation/cdvqa/raw/ — read that adapter to see exactly how it identifies them).
2. List all 2,968 real pairs in evaluation/second_dataset/raw/SECOND_train_set/im1/ (each file name also exists in im2/, label1/, label2/).
3. Exclude the 968 CDVQA test file names from the 2,968. You should have ~2,000 remaining.
4. Split the remaining pairs into train/val (pick a real ratio, e.g. 90/10, and a fixed seed — document both).
5. Save the three lists (train, val, excluded_test) as a JSON file, e.g. ml/change_model/data_split.json, with counts for each.
6. Report: the three real counts, confirm they sum to 2,968, and confirm zero overlap between any two of the three lists (write a real check for this, don't eyeball it).
```

**Expect back:** `data_split.json` plus the three counts and the overlap
check result.

**Bring to Claude:** the counts and overlap-check output. Claude verifies
968 + train + val = 2,968 and zero overlap before any GPU time is spent —
this is the single most important correctness gate in the whole plan.

### Phase 2 — Design the model (no training yet)

**Goal:** an original architecture — not TinyCD's code, which is
non-commercial-licensed for both weights and source.
**Gate:** design review before spending GPU-hours.

**Prompt to paste into opencode:**

```
Design and implement an ORIGINAL change-detection model architecture for Task A. Do not read, copy, or adapt code from TinyCD, BIT_CD, or ChangeFormer (see ml/change_model/SELECTION.md for why — their licenses are non-commercial/research-only for the source code too, not just the weights). A standard shape is fine as inspiration in the abstract (a shared-weight Siamese encoder over both images, a difference/fusion step, a decoder producing a per-pixel map) — just write your own implementation.

1. Start with binary change/no-change as the minimum viable target (collapse label1/label2 to changed-vs-not). If VRAM and time allow after that works, extend to real per-class semantic change over the 6 real classes: NVG_surface, buildings, low_vegetation, playgrounds, trees, water (see evaluation/runners/cdvqa_adapter.py's LAND_COVER_CLASSES for the exact real vocabulary).
2. Write the model class in ml/change_model/model.py.
3. Do NOT start training yet. Report back: the real parameter count, your chosen input size/resolution, an estimated VRAM usage for training at a reasonable batch size on this machine's real VRAM (from Phase 0), and a short rationale for the architecture choice.
```

**Expect back:** the model code, parameter count, VRAM estimate,
rationale.

**Bring to Claude:** the rationale and the code (or a summary of it).
Claude checks for any TinyCD lineage, sanity-checks the VRAM estimate
against the real GPU from Phase 0, and confirms binary-vs-semantic scope
before training starts.

### Phase 3 — Train it for real

**Goal:** real, documented hyperparameters — every field that was
`unknown` in the existing model cards gets a real value this time.
**Gate:** convergence check before evaluating.

**Prompt to paste into opencode:**

```
Implement ml/change_model/train_change_model.py and run a real training job using ml/change_model/data_split.json's train/val lists.

1. Choose and DOCUMENT real hyperparameters: learning rate, batch size, epochs/steps, optimizer, seed, precision (fp16/bf16 if needed for VRAM), gradient checkpointing if needed. Write these into the script or a companion config file, not just left in your head.
2. Log per-epoch (or per-N-steps) train and val loss to a real log file.
3. Save the best checkpoint by val loss.
4. Report: final train/val loss, total wall-clock training time, the checkpoint path, and a plain-text summary of the loss curve (e.g. loss per epoch as a short table) so I can see whether it actually converged.
```

**Expect back:** loss curve summary, final numbers, training time,
checkpoint path.

**Bring to Claude:** the loss table. Claude checks it's actually
decreasing and not NaN/diverging/wildly overfit before you spend time on
evaluation — if it looks broken, Claude gives you a corrective prompt
(different LR, smaller model, etc.) to try instead of moving on.

### Phase 4 — Real pixel-level evaluation

**Goal:** first real use of `spatial_metrics.pixel_mask_f1_iou` — it's
been unit-tested since Phase B3 but never run against real masks.
**Gate:** sanity check before wiring into the backend.

**Prompt to paste into opencode:**

```
Evaluate the trained change model from Phase 3 on the held-out validation split from Phase 1.

1. Run inference on every validation pair, produce a predicted change mask for each.
2. Score every prediction against the real label1/label2-derived ground truth using evaluation/metrics/spatial_metrics.pixel_mask_f1_iou (import and call it directly — do not reimplement it).
3. Report the real mean precision/recall/F1/IoU across the validation set.
4. Pick 3 real examples (not cherry-picked for best results specifically — a spread of good/bad) and describe or save the predicted-vs-real mask side by side.
```

**Expect back:** real F1/IoU numbers, 3 qualitative examples.

**Bring to Claude:** the numbers and examples. Claude sanity-checks they
look plausible (not suspiciously perfect, not near-zero) before you spend
time integrating this into the live backend.

### Phase 5 — Build the change-worker + wire the registry

**Goal:** mirror `ml/vqa-worker/`'s exact pattern — its own venv, a
health check, honest fallback.
**Gate:** API-contract check before the full CDVQA re-run.

**Prompt to paste into opencode:**

```
Serve the trained change model as a new worker, following ml/vqa-worker/'s exact pattern (read it first).

1. New folder ml/change-worker/: its own venv, worker_service.py exposing GET /health and POST /change (takes before/after image paths or bytes, returns a mask + optional per-class breakdown + model_version + latency_ms). A run script mirroring run_worker.ps1's shape for this OS.
2. Backend client: add a new function in backend/app/services/registry_adapters.py following run_vqa's exact pattern, calling this worker and returning a SpecialistResult. On worker-down, a domain mismatch (e.g. SAR input this model wasn't trained on), or a size mismatch, return/raise a clear "unavailable" — never fabricate a result. Wire backend/app/services/registry.py's _semantic_change_available() to a real health check against this worker (same shape as _vqa_worker_available()).
3. Test locally: start the worker, call GET /health, then POST /change on one real held-out pair from Phase 1's validation split. Report both raw JSON responses.
```

**Expect back:** health-check JSON, one real `/change` response.

**Bring to Claude:** both JSON responses. Claude checks the shape matches
the rest of the system's honesty conventions (real confidence handling,
real fallback reason) before the full evaluation run.

### Phase 6 — Re-run the real CDVQA evaluation

**Goal:** the actual proof point — compare directly against this
session's baseline.
**Gate:** the headline result for Task A.

**Prompt to paste into opencode:**

```
Re-run the existing, already-built CDVQA evaluation with the new change-worker from Phase 5 active, and compare against the pre-training baseline.

1. Start the backend and the new change-worker.
2. Run: python -m evaluation.runners.run_cdvqa_eval --per-type 15 --output evaluation/results/cdvqa_eval_v2.jsonl
3. Then: python -m evaluation.results.score_cdvqa evaluation/results/cdvqa_eval_v2.jsonl --output evaluation/results/cdvqa_score_v2.json
4. Do NOT overwrite evaluation/results/cdvqa_score.json (the original baseline) — the _v2 files must sit alongside it.
5. Report the full per-question-type table, compared directly against the baseline: change_or_not/increase_or_not/decrease_or_not (was 0%, all unparseable), smallest_change/largest_change/change_to_what (was 0%, all unparseable), change_ratio (was 14.3%), change_ratio_types (was 37.5%).
```

**Expect back:** the real before/after comparison table.

**Bring to Claude:** the full table. Claude reviews whether real
improvement happened and, if a category is still stuck at 0% or looks
wrong, helps diagnose the specific cause before you move to write-up.

### Phase 7 — Document and commit Task A

**Goal:** model card with real numbers, progress log updated, committed.
**Gate:** wrap-up.

**Prompt to paste into opencode:**

```
Close out Task A.

1. Write ml/change_model/MODEL_CARD.md: real architecture summary, real training data/split/seed/hyperparameters from Phase 3, real pixel-level numbers from Phase 4, real CDVQA before/after table from Phase 6, and an honest scope statement (binary vs. semantic — whichever you actually shipped). Note SECOND's license is unstated on its official page — carry forward the same honest caveat already in evaluation/second_dataset/README.md, don't re-litigate it.
2. Update docs/SOLO_PROGRESS.md's B8 row to done (real training, not a skip) with a summary in the same style as the rest of that file.
3. Run the full test suites (backend + evaluation) and confirm nothing regressed.
4. Commit with message style feat(B8): ... , ending with the same Co-Authored-By trailer convention already used in this repo's git log (check `git log` for the exact line).
```

**Expect back:** confirmation of the commit, test results.

**Bring to Claude:** the commit hash and test output, just to confirm the
wrap-up is clean before starting Task B.

---

## Task B — Real BigEarthNet LoRA fine-tune

Has a permission gate — its data isn't sourced yet.

### Phase 8 — Research BigEarthNet: do not download yet

**Goal:** this has genuinely not been verified in any session so far.
Find the real source, license, and size before anyone commits to a
download.
**Gate:** STOP — needs your explicit yes before Phase 9.

**Prompt to paste into opencode:**

```
Research BigEarthNet's real, current official download mechanism — do not download anything yet, this is research only.

1. Find the current official source (bigearth.net as of the source paper — verify it's still live and see exactly what's offered: full imagery archive vs. a lighter labels-only release, v1 vs v2, etc).
2. Report: the real official URL, the real license terms, the real download size for whatever the smallest sufficient real-labeled subset would be (don't assume the full archive is required), and the label format (what a single patch's label metadata actually looks like).
3. Do not fetch anything. Just report your findings so a decision can be made about how much to download.
```

**Expect back:** real URL, license, size, label format — no download yet.

**Bring to Claude:** the research findings. This is the one hard
permission gate in the whole plan — Claude and you decide together
whether to proceed, exactly like every other external dataset fetch in
this project's history (see `docs/SOLO_PROGRESS.md`'s B1/B3/B4/B5/B6 rows
for the established pattern: real size and license disclosed first,
explicit yes required before anything is pulled).

### Phase 9 — Fetch data, build the Q&A dataset

**Goal:** only runs after Phase 8's explicit go-ahead.
**Gate:** quality check before training on it.

**Prompt to paste into opencode:**

```
Using the real source confirmed in Phase 8 (only after explicit approval), fetch the real labeled BigEarthNet data and build a training Q&A dataset.

1. Download the approved subset/release. Verify the download with a real integrity check (checksum or archive test), not just file size — this project's own SECOND-dataset fetch was silently truncated once with no error, so don't trust size alone.
2. Write ml/adaptation/prepare_bigearthnet_vqa.py: build real question/answer pairs from the real land-cover labels. Document your question templates and label-to-answer mapping explicitly (in the script's docstring or a companion doc) — someone else must be able to see exactly how a label became a question/answer pair. Fixed seed. Write train/val/test JSONL plus a stats file (counts per question type, per class).
3. Report: the real dataset stats, and 5-10 real example question/answer pairs so the quality can be checked before any training happens.
```

**Expect back:** dataset stats + real example Q&A pairs.

**Bring to Claude:** the stats and examples. Claude checks the
question/answer generation is sound (sensible templates, correct label
mapping, no obvious garbage-in risk) before any GPU time goes into
training on it.

### Phase 10 — Train the LoRA adapter

**Goal:** stays structurally compatible with `model_provider.py`'s
existing loading code.
**Gate:** convergence check before evaluating.

**Prompt to paste into opencode:**

```
Implement ml/adaptation/train_lora.py and run a real LoRA fine-tune of SmolVLM-256M-Instruct on the dataset from Phase 9.

1. Read ml/adaptation/configs/stage3.yaml and ml/vqa-worker/model_provider.py's DecoderShim wrapper first — your adapter must stay structurally compatible (LoRA on the decoder only). The existing real config is r=8, lora_alpha=16, lora_dropout=0.1, target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]. You may change these with a real, documented reason (e.g. more VRAM headroom allows higher rank) — if you do, test that model_provider.py still loads the result correctly, don't assume.
2. Replace every "unknown" field in ml/adaptation/configs/stage3.yaml with a real value: learning rate, steps, batch size, gradient accumulation, seed, precision, image size. fp16/bf16 + gradient checkpointing for this machine's real VRAM budget (from Phase 0).
3. Save the adapter weights, a real training log (loss curve, wall-clock time), and a new version string smolvlm256m-ben-lora-s3-v2.0 (keep the existing v1.0 adapter as a reference — don't delete it).
4. Report: final train/val loss, training time, loss-per-epoch table, and where the new adapter was saved.
```

**Expect back:** loss curve, training time, adapter location.

**Bring to Claude:** the loss table. Same convergence check as Phase 3 —
decreasing and stable before moving to evaluation.

### Phase 11 — Real evaluation, both ways

**Goal:** held-out accuracy on your own test split, plus a re-run of the
existing RSVQA-LR baseline.
**Gate:** the headline result for Task B.

**Prompt to paste into opencode:**

```
Evaluate the new LoRA adapter from Phase 10, two ways.

1. Extend ml/adaptation/eval_lora.py (or a new eval_lora_v2.py) to evaluate base vs. your new adapter against Phase 9's real held-out test split — this time you have real ground truth, so report real per-question-type accuracy, not just an agreement rate.
2. Update ml/smolvlm/lora_stage3/version.json (or point ADAPTER_DIR at a new versioned directory) so model_provider.py picks up the new version string.
3. Start the VQA worker with the new adapter active, start the backend, then re-run the exact existing RSVQA-LR evaluation:
   python -m evaluation.runners.run_rsvqa_lr_eval --per-type 60 --output evaluation/results/rsvqa_lr_eval_v2.jsonl
   python -m evaluation.results.score_rsvqa_lr evaluation/results/rsvqa_lr_eval_v2.jsonl --output evaluation/results/rsvqa_lr_score_v2.json
   Do not overwrite the original _score.json baseline file.
4. Report both: your own held-out accuracy table, and the RSVQA-LR before/after comparison against the baseline (presence 73.3%, comp 65.0%, rural_urban 40.7%, count 0% exact/RMSE 206.6).
```

**Expect back:** held-out accuracy table + RSVQA-LR before/after
comparison.

**Bring to Claude:** both tables. Claude reviews whether real improvement
happened over the untrained baseline and helps decide whether to accept
this adapter or iterate on hyperparameters.

### Phase 12 — Document and commit Task B

**Goal:** model card updated with every real number, progress log
updated, committed.
**Gate:** wrap-up.

**Prompt to paste into opencode:**

```
Close out Task B.

1. Update ml/adaptation/MODEL_CARD.md with all real training details from Phase 10 (replacing every "unknown"), the real evaluation tables from Phase 11, and a direct comparison against both the base-model baseline and the old v1.0 mystery-provenance adapter.
2. Update docs/SOLO_PROGRESS.md's B2 row to done (real training, not scope-limited) in the same style as the rest of that file.
3. Run the full test suites (backend + evaluation) and confirm nothing regressed.
4. Commit with message style feat(B2): ... , ending with the same Co-Authored-By trailer convention already used in this repo's git log.
```

**Expect back:** confirmation of the commit, test results.

**Bring to Claude:** the commit hash and test output.

---

## Final

### Phase 13 — Full wrap-up

**Goal:** both tasks together, one final confirmation pass.
**Gate:** done.

**Prompt to paste into opencode:**

```
Both training tasks are now complete. Do a final pass:

1. Run the full backend test suite and the full evaluation test suite one more time — confirm everything still passes with both new components in place.
2. Print a short final summary: what was trained, real final numbers for both tasks against their baselines, and the exact file paths of everything new (model card, worker code, split files, score JSONs).
3. Confirm the git log shows both commits from Phase 7 and Phase 12 and that the working tree is clean (nothing uncommitted).
```

**Expect back:** final test results + summary + clean git status.

**Bring to Claude:** the final summary, for one last honest read-through
before you call this done.
