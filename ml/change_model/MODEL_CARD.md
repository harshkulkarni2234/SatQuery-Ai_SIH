# SatQuery Siamese CNN Change Detection Model — MODEL_CARD

## Model Identity

| Field | Value |
|---|---|
| **Model ID** | `change.semantic_model` |
| **Model Version** | `siamese-cnn-v1` |
| **Artifact** | `ml/change_model/trained/best_change_model.pt` |
| **Training Script** | `ml/change_model/train_change.py` |
| **Architecture** | Small Siamese CNN encoder + feature difference/fusion module + decoder |
| **Task** | Binary change detection (changed / unchanged per pixel) |
| **Input** | Two co-sized 256×256 RGB images (before / after) |
| **Output** | Binary change mask (256×256 uint8, 0/1) |

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
| **Training Time** | ~5 min/epoch (varies) |

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

The change_ratio improvement (+0.165) reflects the model's ability to better identify the *amount* of change. The change_ratio_types slight decline (−0.042) reflects that the model's quantitative output doesn't always map to the discrete percentage bucket labels. The yes_no and categorical types (smallest/largest/change_to_what) all score None because the model returns quantitative change descriptions, not semantic class labels — this is an expected, honest capability gap.

---

## SECOND License Caveat

This model was trained on images from the **SECOND dataset** (Semantic Change Detection dataset, https://captain-whu.github.io/SCD/). The SECOND dataset's license is **unstated on the official page** — it was flagged to and approved by the user before fetching. The CDVQA evaluation labels used for validation are Apache-2.0 (from https://github.com/YZHJessica/CDVQA).

**IMPORTANT**: The SECOND dataset license is unstated. This model and its evaluation should be treated as research-only. For any commercial or public deployment, the SECOND dataset license status must be resolved first.

Additionally, the underlying TinyCD architecture (see `ml/change_model/SELECTION.md`) is released under a **non-commercial/research-only** license per its own README. No formal LICENSE file or SPDX identifier exists for TinyCD.

---

## Files

| File | Description |
|---|---|
| `ml/change_model/train_change.py` | Training script (Siamese CNN + fp16 mixed precision) |
| `ml/change_model/trained/best_change_model.pt` | Best model weights (4.9M params, ~19 MB) |
| `ml/change_model/trained/change_training_log.json` | Training log (per-epoch loss, F1, IoU) |
| `ml/change_model/trained/data/` | Pre-computed .npy arrays (im1, im2, mask, sem1, sem2) |
| `ml/change-worker/` | HTTP worker service (FastAPI + model_provider) |
| `ml/change_model/MODEL_CARD.md` | This file |

---

## Known Limitations

1. **Binary only**: Cannot answer "what changed" — only "did it change."
2. **Fixed input size**: Trained and evaluated on 256×256 images.
3. **Single GPU**: Requires CUDA for reasonable latency (~929ms per inference on RTX 2050). CPU inference is significantly slower.
4. **No georeferencing awareness**: The model treats images as raw pixels; it does not use CRS, bounds, or resolution metadata.
5. **SECOND dataset license unstated**: Training data license status is unclear.
6. **TinyCD non-commercial license**: The underlying architecture is non-commercial/research-only.
