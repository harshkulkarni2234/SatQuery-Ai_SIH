# SatQuery Siamese CNN Change Detection Model — MODEL_CARD

## Model Identity

| Field | Value |
|---|---|
| **Registry ID** | `change.siamese_binary_cnn` (was `change.semantic_model` until the id was renamed — the model is binary, not semantic) |
| **Model Version** | `siamese-cnn-v1` |
| **Artifact** | `ml/change_model/trained/best_change_model.pt` |
| **Training Script** | `ml/change_model/train_change.py` |
| **Architecture** | Small Siamese CNN encoder + feature difference/fusion module + decoder |
| **Task** | Binary change detection (changed / unchanged per pixel) |
| **Input** | Two co-sized 256×256 RGB images (before / after) |
| **Output** | Binary change mask (256×256; served to the backend as a 0/255 PNG) |

---

## Architecture

```
Before Image (256×256×3) ──┐
                            ├──→ SiameseEncoder (shared weights)
After Image  (256×256×3) ──┘
                                   │
                                   ▼
                        FeatureDiffusionFusion
                        (|f1−f2|, f1, f2, f1⊕f2 → 512ch)
                                   │
                                   ▼
                        ChangeDecoder
                        (512 → 256 → 128 → 64 → 32 → 1)
                                   │
                                   ▼
                        Binary Mask (256×256, sigmoid)
```

### SiameseEncoder
- 4 convolutional blocks (3→64→128→256→512 channels), each with BatchNorm + ReLU
- MaxPooling after each block (256→128→64→32→16)
- Shared weights across both temporal branches

### FeatureDiffusionFusion
- Concatenates 4 tensors: absolute difference |f1−f2|, f1, f2, elementwise product f1×f2
- 2×1×1 convolutions with BatchNorm + ReLU to fuse into a single 512-channel tensor

### ChangeDecoder
- Transposed convolution upsampling (512→256→128→64→32)
- Final 1×1 convolution to 1 channel, Sigmoid activation
- Outputs a single-channel probability map

### Parameter Count: ~4.9M

---

## Hyperparameters

| Hyperparameter | Value |
|---|---|
| **Optimizer** | Adam (lr=1e-4, weight_decay=1e-5) |
| **Loss** | BCEWithLogitsLoss |
| **Batch Size** | 8 |
| **Epochs** | 5 |
| **Image Size** | 256×256 |
| **Mixed Precision** | fp16 via `torch.amp.autocast(device_type='cuda', dtype=torch.float16)` |
| **Scaler** | `torch.amp.GradScaler('cuda')` |
| **Dataset** | SECOND-derived (1,700 train / 300 val, 512×512→256×256) |
| **Normalization** | ImageNet mean/std ([0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]) |
| **Best Metric** | Val F1 = 0.4677, Val IoU = 0.3268 (epoch 5) |
| **Hardware** | NVIDIA GeForce RTX 2050 (CUDA) |
| **Training Time** | 40–81 s per epoch, ≈5.2 min for the 5 epochs (from `trained/change_training_log.json`) |

---

## Honest Binary Scope

This model performs **binary change detection** — it outputs a per-pixel mask indicating whether a change occurred between two temporal images. It does **NOT** perform semantic change classification. Specifically:

- **It does NOT say** what class changed (e.g., "water was replaced by buildings").
- **It does NOT identify** land-cover classes for changed regions.
- **It does NOT provide** per-class change breakdowns.
- **It does NOT answer** categorical questions like "What changed?" or "What is the smallest change?"

The model measures aggregate pixel-level change coverage. It reports a percentage of the frame that changed and the number of changed pixels. This is fundamentally different from semantic change detection which labels *what* changed per pixel.

### CDVQA Evaluation (120 samples, real official test split)

| Metric | Baseline (deterministic CV) | V2 (Siamese CNN) |
|---|---|---|
| **Overall Accuracy** | 0.267 | **0.321** |
| **change_ratio** | 0.143 | **0.308** (+0.165) |
| **change_ratio_types** | 0.375 | 0.333 (−0.042) |
| **n_errors** | 19 | 19 |
| **n_total** | 120 | 120 |

**How to read this.** These numbers are a small sample (13–15 questions per type) and should not be over-interpreted:

- `change_ratio` is 4/13 correct vs 1/7 for the baseline — but the baseline could only parse 7 of 13 answers, so the two are not measured on the same set. The difference is within sampling noise.
- The headline "overall accuracy" is computed over *scored* answers only: 9/28 for the Siamese CNN vs 4/15 for the baseline. It is not a like-for-like comparison, and most of the gap is that the learned answer states its percentage in a form the scorer can parse.
- All yes/no types and all categorical types (smallest / largest / change_to_what) remain **unscored (0 parseable)** because the model states aggregate change, not a verdict or a land-cover class. This is an honest capability gap, not something this model addresses.
- **No train/test leakage** (checked, Known Limitation 7): none of the CDVQA test images were in the training or validation split. The evaluation is in-distribution, though — same SECOND imagery source.
- The run was made with the change-specialist implementation as of commits `c4a5fda`/`ecd5116`; answer wording and domain gating have since changed, so re-run before quoting.

**No claim of superiority is made** beyond: on this 120-question sample the learned model's percentage was parseable more often.

---

---

## SECOND License Caveat

This model was trained on images from the **SECOND dataset** (Semantic Change Detection dataset, https://captain-whu.github.io/SCD/). The SECOND dataset's license is **unstated on the official page** — it was flagged to and approved by the user before fetching. The CDVQA evaluation labels used for validation are Apache-2.0 (from https://github.com/YZHJessica/CDVQA).

**IMPORTANT**: The SECOND dataset license is unstated. This model and its evaluation should be treated as research-only. For any commercial or public deployment, the SECOND dataset license status must be resolved first.

**Architecture provenance.** This is an original small Siamese CNN written for this project (`train_change.py`). It is **not** TinyCD, BIT, or ChangeFormer and contains no code from them; those were evaluated as candidates in `SELECTION.md` and not used.

---

## Files

| File | Description |
|---|---|
| `ml/change_model/train_change.py` | Training script (Siamese CNN + fp16 mixed precision) — the script that produced the shipped weights |
| `ml/change_model/trained/best_change_model.pt` | Best model weights (4.9M params, ~19 MB) |
| `ml/change_model/trained/change_training_log.json` | Training log (per-epoch loss, F1, IoU) |
| `ml/change_model/trained/data/` | Pre-computed .npy arrays (im1, im2, mask, sem1, sem2) |
| `ml/change-worker/` | HTTP worker service (FastAPI + model_provider) |
| `ml/change_model/MODEL_CARD.md` | This file |
| `ml/change_model/prepare_second_data.py` | Data-preparation script that builds the `.npy` arrays from SECOND |

---

## Data Preparation

The training/validation arrays under `trained/data/` (`{train,val}_{im1,im2,mask,sem1,sem2}.npy`, `{train,val}_names.json`, `meta.json`) were generated by **`ml/change_model/prepare_second_data.py`**.

The script reads the 2,968 pairs in `evaluation/second_dataset/raw/SECOND_train_set/` (`im1/`, `im2/`, `label1/`, `label2/` subfolders), excludes the 968 pairs in CDVQA's `Test` split, splits the remaining 2,000 names with `seed=42` / `val_frac=0.15` into 1,700 train / 300 val, resizes images to 256×256 (LANCZOS) and labels to 256×256 (NEAREST with 7-class colormap mapping), derives the binary mask as `block_mean_bool(label1 != label2, block=2)`, and writes raw uint8 arrays via `np.memmap` plus `{split}_names.json` and `meta.json`.

**Verification (2026-09-20):** Running `prepare_second_data.py` with `--output-dir` pointing to a temporary directory produced output that was **byte-for-byte identical** to all committed artifacts: `train_names.json`, `val_names.json`, and all five arrays (`{train,val}_{im1,im2,mask,sem1,sem2}.npy`) matched exactly (0 differing bytes). This proves the script reproduces the existing training data.

`meta.json` was truncated in an earlier commit (incomplete `colormap`); it has since been rewritten to the exact value the script emits (seed 42, val_frac 0.15, size 256, the 7-entry SECOND colormap, train_n 1700, val_n 300).

---

## Known Limitations

1. **Binary only**: Cannot answer "what changed" — only "did it change."
2. **Fixed input size**: Trained and evaluated on 256×256 images.
3. **Latency**: CUDA (fp16) is used when available, but is not required — the model is small (4.9M params) and a 256×256 pair ran in ~70–120 ms on an Apple-silicon CPU (one warm sample, not a benchmark). An earlier draft's "~929 ms on RTX 2050" figure was not reproduced and is dropped.
4. **No georeferencing awareness**: The model treats images as raw pixels; it does not use CRS, bounds, or resolution metadata.
5. **SECOND dataset license unstated**: Training data license status is unclear.
6. **Training domain**: SECOND is 0.5–3 m aerial RGB. The backend only runs this model on pairs that are the same size, not provably different areas, and not known to be coarser than 3 m per pixel; otherwise it runs the deterministic method and says so. When the resolution is unknown the model still runs, with a warning.
7. **CDVQA train/test overlap: checked, none.** CDVQA's 968 test pairs are a subset of SECOND's 2,968 public pairs, and the model trained on the other 2,000 (1,700 train / 300 val). `ml/change_model/check_cdvqa_leakage.py` run on the committed `train_names.json`/`val_names.json` against CDVQA's `Test_images.json` found **0 of the 968 test pairs in either split**, so every CDVQA evaluation image is unseen. The evaluation is still *in-distribution* (same SECOND imagery source, same 0.5–3 m aerial domain), so it says little about other imagery.
8. **Weights and split files** are committed under `ml/change_model/trained/` (`best_change_model.pt`, `change_training_log.json`, `data/{train,val}_names.json`, `data/meta.json`); the large `.npy` training arrays are not. Verified: the checkpoint loads strictly into the worker's architecture (4,874,241 parameters) and runs on CPU (~70–120 ms per 256×256 pair on an Apple-silicon CPU, one warm sample).
9. **Unrelated artifacts removed**: `best_model.pth`, `final_model.pth` and `training_log.json` came from an earlier, superseded training script (`train_change_detection.py`, a timm EfficientNet-B0 Siamese U-Net; still in git history at `ae6daf8`). They have been removed from the repository (`chore(B8): remove unused EfficientNet artifacts`). They were never served by the worker and are not what this card describes.
