# SatQuery Semantic Change Detection Model — SEMANTIC_MODEL_CARD

## Model Identity

| Field | Value |
|---|---|
| **Registry ID** | `change.semantic_multiclass` |
| **Model Version** | `semantic-cnn-v1` |
| **Artifact** | `ml/change_model/trained/best_semantic_change_model.pt` |
| **Training Script** | `ml/change_model/train_semantic_change.py` |
| **Architecture** | Small Siamese CNN encoder + feature diffusion/fusion + dual per-pixel class heads |
| **Task** | Per-class semantic change detection — predicts land-cover class for both before and after images |
| **Input** | Two co-sized 256×256 RGB images (before / after) |
| **Output** | Per-pixel class logits for both dates (7 classes each, 256×256) |

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
                         SemanticChangeDecoder
                         (512 → 256 → 128 → 64 → 32)
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                    Head 1 (sem1)         Head 2 (sem2)
                  7 class logits         7 class logits
                  (256×256)              (256×256)
```

### SiameseEncoder
- 4 convolutional blocks (3→64→128→256→512 channels), each with BatchNorm + ReLU
- MaxPooling after each block (256→128→64→32→16)
- Shared weights across both temporal branches

### FeatureDiffusionFusion
- Concatenates 4 tensors: absolute difference |f1−f2|, f1, f2, elementwise product f1×f2
- 2×1×1 convolutions with BatchNorm + ReLU to fuse into a single 512-channel tensor

### SemanticChangeDecoder
- Transposed convolution upsampling (512→256→128→64→32)
- Two separate 1×1 convolution heads producing per-pixel class logits for sem1 and sem2

### Parameter Count: ~4,874,670

---

## Hyperparameters

| Hyperparameter | Value |
|---|---|
| **Optimizer** | Adam (lr=1e-4, weight_decay=1e-5) |
| **Loss** | Weighted CrossEntropyLoss (inverse-frequency class weights) |
| **Class Weights** | [0.179, 46.144, 1.675, 2.230, 9.188, 4.423, 219.748] (computed from training class frequencies) |
| **Batch Size** | 8 |
| **Epochs** | 5 |
| **Image Size** | 256×256 |
| **Mixed Precision** | fp16 via `torch.amp.autocast(device_type='cuda', dtype=torch.float16)` |
| **Scaler** | `torch.amp.GradScaler('cuda')` |
| **Random Seed** | 42 |
| **Dataset** | SECOND-derived (1,700 train / 300 val, 512×512→256×256, 7-class semantic maps) |
| **Hardware** | NVIDIA GeForce RTX 2050 (CUDA) |

---

## Honest Binary Scope

This model performs **per-class semantic change detection** — it predicts the land-cover class for each pixel in both before and after images. It does NOT predict a binary change mask directly; binary change is derived as `pred_sem1 != pred_sem2`.

### CDVQA Evaluation Context
The model was trained to predict the SECOND 7-class semantic map for both dates. It does NOT directly answer CDVQA questions (which are about change detection). The derived binary change mask can be used as a fallback, but the primary purpose is semantic segmentation.

---

## Measured Validation Numbers

**Per-class IoU (val set, 300 samples):**

| Class | sem1 (before) IoU | sem2 (after) IoU |
|---|---|---|
| unchanged (0) | 0.4466 | 0.3321 |
| water (1) | 0.0017 | 0.0045 |
| NVG_surface (2) | 0.1973 | 0.1587 |
| low_vegetation (3) | 0.2008 | 0.0239 |
| trees (4) | 0.0452 | 0.0569 |
| buildings (5) | 0.0956 | 0.1949 |
| playgrounds (6) | 0.0145 | 0.0234 |

**Overall mIoU:** 0.3894 (sem1: 0.4466, sem2: 0.3321)

**Derived binary change metrics (where pred_change = pred_sem1 != pred_sem2):**

| Metric | Semantic Model | Binary Model |
|---|---|---|
| **Val F1** | 0.3634 | 0.4677 |
| **Val IoU** | 0.2401 | 0.3268 |

The semantic model's derived binary change metrics are **lower** than the dedicated binary model's. This is expected: the semantic model optimizes per-class cross-entropy, not binary change detection directly.

---

## From-to Transition Confusion Matrix (changed pixels)

For changed pixels (where gt_sem1 != gt_sem2), the model's predicted sem2 class vs the true sem2 class:

- **NVG_surface → buildings**: 3,873 pixels, 89% correct
- **low_vegetation → buildings**: 1,851 pixels, 93% correct
- **NVG_surface → trees**: 594 pixels, 74% correct
- **low_vegetation → trees**: 714 pixels, 79% correct
- **buildings → NVG_surface**: 1,107 pixels, 20% correct
- **water → buildings**: 26 pixels, 96% correct
- **water → NVG_surface**: 101 pixels, 32% correct
- **water → low_vegetation**: 151 pixels, 0% correct

---

## SECOND License Caveat

This model was trained on images from the **SECOND dataset** (Semantic Change Detection dataset, https://captain-whu.github.io/SCD/). The SECOND dataset's license is **unstated on the official page** — it was flagged to and approved by the user before fetching. This model and its evaluation should be treated as research-only. For any commercial or public deployment, the SECOND dataset license status must be resolved first.

**Architecture provenance.** This is an original Siamese CNN written for this project (`train_semantic_change.py`). It reuses the encoder structure from `train_change.py` but adds dual class heads. It is **not** TinyCD, BIT, or ChangeFormer.

---

## Known Limitations

1. **Class imbalance is severe**: Class 0 (unchanged) covers ~80% of pixels. Class 1 (water) is 0.3%, Class 6 (playgrounds) is 0.1%. Despite inverse-frequency class weighting, minority classes have near-zero IoU.
2. **Low per-class IoU for water and playgrounds**: Water (IoU ~0.002-0.005) and playgrounds (IoU ~0.015-0.023) are essentially never predicted correctly. The model collapses to predicting common classes.
3. **Derived binary change underperforms the dedicated binary model**: F1 0.3634 vs 0.4677, IoU 0.2401 vs 0.3268. The semantic model is not optimized for change detection.
4. **Fixed input size**: Trained and evaluated on 256×256 images.
5. **SECOND dataset license unstated**: Training data license status is unclear.
6. **Training domain**: SECOND is 0.5–3 m aerial RGB. The model treats images as raw pixels; it does not use CRS, bounds, or resolution metadata.
7. **No georeferencing awareness**: The model does not use geospatial metadata.
8. **Weighting note**: Class weights are inverse-frequency (total_pixels / (num_classes × class_pixels)) but are extremely skewed (class 6 weight = 219.7). This causes gradient instability for minority classes.

---

## Files

| File | Description |
|---|---|
| `ml/change_model/train_semantic_change.py` | Training script (Siamese CNN + dual class heads + weighted CE loss) |
| `ml/change_model/trained/best_semantic_change_model.pt` | Best checkpoint by mean mIoU (~19.5 MB) |
| `ml/change_model/trained/semantic_training_log.json` | Per-epoch training/validation metrics |
| `ml/change_model/SEMANTIC_MODEL_CARD.md` | This file |
| `ml/change_model/trained/data/` | Pre-computed .npy arrays (im1, im2, sem1, sem2) |
| `ml/change_model/MODEL_CARD.md` | Binary change model card (separate model) |

---

## Known Limitations

1. **Severe class imbalance**: Class 0 (unchanged) covers ~80% of pixels. Class 1 (water) is 0.3%, Class 6 (playgrounds) is 0.1%. Despite inverse-frequency class weighting, minority classes have near-zero IoU.
2. **Low per-class IoU for water and playgrounds**: Water (IoU ~0.002-0.005) and playgrounds (IoU ~0.015-0.023) are essentially never predicted correctly. The model collapses to predicting common classes.
3. **Derived binary change underperforms the dedicated binary model**: F1 0.3634 vs 0.4677, IoU 0.2401 vs 0.3268. The semantic model optimizes per-class cross-entropy, not binary change detection.
4. **Fixed input size**: Trained and evaluated on 256×256 images.
5. **SECOND dataset license unstated**: Training data license status is unclear.
6. **No georeferencing awareness**: The model treats images as raw pixels; it does not use CRS, bounds, or resolution metadata.
7. **Weighting note**: Class weights are inverse-frequency (total_pixels / (num_classes × class_pixels)) but are extremely skewed (class 6 weight = 219.7). This causes gradient instability for minority classes.
8. **Not a direct CDVQA model**: This model predicts semantic class maps for both dates. CDVQA questions require change detection, which must be derived as `pred_sem1 != pred_sem2`.
