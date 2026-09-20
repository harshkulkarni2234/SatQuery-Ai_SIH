"""Build the SECOND-derived change-detection training/validation cache.

For every SECOND_train_set pair NOT in the CDVQA Test split, stores:
  - im1/im2   : 256x256 RGB uint8  (pre/post event optical images, LANCZOS)
  - mask      : 256x256 uint8 0/1  (binary change = semantic label changed,
               area-averaged from the full-res 512x512 label maps)
  - sem1/sem2 : 256x256 uint8 class-id of label1/label2 (SECOND 7-class map,
               decoded with the SECOND colormap verified against the real
               CDVQA answer strings: white=unchanged, blue=water,
               gray=NVG_surface, dark-green=low_vegetation, bright-green=
               trees, dark-red=buildings, red=playgrounds)

Artifacts are written as numpy memmaps under
  ml/change_model/trained/data/  (gitignored).
The CDVQA Test split (968 pairs) is EXCLUDED to avoid leaking into this
model's training/validation (CDVQA eval is the held-out benchmark).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import numpy as np
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
SECOND = os.path.join(REPO, "evaluation", "second_dataset", "raw", "SECOND_train_set")
SIZE = 256
SEED = 42
VAL_FRAC = 0.15

COLOR_TO_IDX = {
    (255, 255, 255): 0, (0, 0, 255): 1, (128, 128, 128): 2,
    (0, 128, 0): 3, (0, 255, 0): 4, (128, 0, 0): 5, (255, 0, 0): 6,
}


def load_class_map(path: str) -> np.ndarray:
    a = np.asarray(Image.open(path).convert("RGB"))
    out = np.zeros(a.shape[:2], dtype=np.uint8)
    for tup, idx in COLOR_TO_IDX.items():
        m = (a[:, :, 0] == tup[0]) & (a[:, :, 1] == tup[1]) & (a[:, :, 2] == tup[2])
        out[m] = idx
    return out


def block_mean_bool(mask512: np.ndarray, block: int = 2) -> np.ndarray:
    """Area-average a full-res boolean mask down by `block` then threshold."""
    h, w = mask512.shape
    rows = h // block * block
    cols = w // block * block
    m = mask512[:rows, :cols].astype(np.float32)
    m = m.reshape(rows // block, block, cols // block, block)
    m = m.mean(axis=(1, 3))
    return (m > 0.5).astype(np.uint8)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default=None, help="Output directory for .npy arrays and names files (default: ml/change_model/trained/data)")
    args = ap.parse_args(argv)

    DATA_DIR = args.output_dir or os.path.join(os.path.dirname(__file__), "trained", "data")

    from evaluation.runners.cdvqa_adapter import load_real_test_set  # noqa: E402 (repo import)

    test_names = {row["file_name"] for row in load_real_test_set("Test")}
    print(f"CDVQA Test file_names: {len(test_names)}")

    all_files = sorted(
        f
        for f in glob.glob(os.path.join(SECOND, "im1", "*.png"))
        if not os.path.basename(f).startswith("._")
    )
    names = [os.path.basename(f) for f in all_files]
    print(f"SECOND pairs available: {len(names)}")

    usable = [n for n in names if n not in test_names]
    excluded = [n for n in names if n in test_names]
    print(f"kept (non-CDVQA-test): {len(usable)}  excluded: {len(excluded)}")

    # All four subfolders must exist for every kept name.
    missing = []
    for n in usable:
        for sub in ("im1", "im2", "label1", "label2"):
            if not os.path.isfile(os.path.join(SECOND, sub, n)):
                missing.append(f"{sub}/{n}")
    if missing:
        raise SystemExit(f"missing files: {missing[:10]}")

    rng = random.Random(SEED)
    rng.shuffle(usable)
    n_val = int(round(len(usable) * VAL_FRAC))
    val_names = sorted(usable[:n_val])
    train_names = sorted(usable[n_val:])
    print(f"train={len(train_names)} val={len(val_names)} (seed={SEED}, val_frac={VAL_FRAC})")

    os.makedirs(DATA_DIR, exist_ok=True)
    meta = {"seed": SEED, "val_frac": VAL_FRAC, "size": SIZE,
            "colormap": {f"{r},{g},{b}": i for (r, g, b), i in COLOR_TO_IDX.items()},
            "train_n": len(train_names), "val_n": len(val_names)}

    for split, split_names in (("train", train_names), ("val", val_names)):
        n = len(split_names)
        shape_im = (n, SIZE, SIZE, 3)
        shape_m = (n, SIZE, SIZE)
        paths = (
            ("im1", np.memmap(os.path.join(DATA_DIR, f"{split}_im1.npy"), dtype=np.uint8, mode="w+", shape=shape_im)),
            ("im2", np.memmap(os.path.join(DATA_DIR, f"{split}_im2.npy"), dtype=np.uint8, mode="w+", shape=shape_im)),
            ("mask", np.memmap(os.path.join(DATA_DIR, f"{split}_mask.npy"), dtype=np.uint8, mode="w+", shape=shape_m)),
            ("sem1", np.memmap(os.path.join(DATA_DIR, f"{split}_sem1.npy"), dtype=np.uint8, mode="w+", shape=shape_m)),
            ("sem2", np.memmap(os.path.join(DATA_DIR, f"{split}_sem2.npy"), dtype=np.uint8, mode="w+", shape=shape_m)),
        )
        for i, name in enumerate(split_names):
            im1 = Image.open(os.path.join(SECOND, "im1", name)).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
            im2 = Image.open(os.path.join(SECOND, "im2", name)).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
            l1 = load_class_map(os.path.join(SECOND, "label1", name))
            l2 = load_class_map(os.path.join(SECOND, "label2", name))
            mask_full = (l1.astype(np.int16) != l2.astype(np.int16))
            paths[0][1][i] = np.asarray(im1)
            paths[1][1][i] = np.asarray(im2)
            paths[2][1][i] = block_mean_bool(mask_full, 2)
            paths[3][1][i] = np.asarray(Image.fromarray(l1).resize((SIZE, SIZE), Image.NEAREST))
            paths[4][1][i] = np.asarray(Image.fromarray(l2).resize((SIZE, SIZE), Image.NEAREST))
            if (i + 1) % 250 == 0:
                print(f"  [{split}] {i+1}/{n}", flush=True)
        for _, mm in paths:
            mm.flush()
            del mm
        with open(os.path.join(DATA_DIR, f"{split}_names.json"), "w") as f:
            json.dump(split_names, f)
        print(f"[{split}] wrote {n} samples -> {DATA_DIR}")

    with open(os.path.join(DATA_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    with open(os.path.join(DATA_DIR, "excluded_cdvqa_test.json"), "w") as f:
        json.dump(excluded, f)
    print("done")


if __name__ == "__main__":
    main()