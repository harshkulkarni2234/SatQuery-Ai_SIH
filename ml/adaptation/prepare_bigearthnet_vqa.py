#!/usr/bin/env python3
"""Build a real question/answer dataset from real BigEarthNet v2.0 labels.

Data provenance (all real, none invented):
- Labels: `ml/adaptation/raw/metadata.parquet`, downloaded from the official
  Zenodo record for BigEarthNet v2.0 (https://zenodo.org/records/10891137,
  file `metadata.parquet`, ~3.6 MB, license CDLA-Permissive-1.0, fetched
  with user approval). The parquet is the official label release covering
  BigEarthNet v2.0 patches. Download script with exact URL + size:
  `ml/adaptation/download_bigearthnet_labels.py`.
- Images: local `testing/` folder, ~4,011 BigEarthNet-style Sentinel-2
  optical patches (tiles T34UDG/2017-07-20 and T35ULA/2017-08-08), already
  present on this machine (see data/manifest.json). They are 120x120 RGB
  PNGs (standard BigEarthNet patch size). NO new image download happens.
- Matching: a patch is used iff its local filename stem
  (`S2A_MSIL2A_20170720T100031_N9999_R122_T34UDG_65_03.png`) equals a
  `patch_id` in the official parquet. This build matches 4008/4011 local
  patches to real official labels. All matched rows are from the official
  parquet's `train` split and country `Lithuania`.

Dataset design (documented so anyone can read label -> question/answer):

Question templates. Two question types, both chosen because they are the
only types that can be derived *honestly* from BigEarthNet multi-label
land-cover vectors, and both match the specialist's gate in
`ml/vqa-worker/model_provider.py`'s `should_use_specialist()` (presence
via "is there", count via "how many"):

1) presence (yes/no), ground truth = label membership:
   - "Is there {SUBJECT} in this image?"
   - "Does the image contain {SUBJECT}?"
   - "Is {SUBJECT} present in this image?"
   answer = "yes" if the patch's label set contains that class, else "no".
   Positive question generated for EVERY present class; negative questions
   generated for a deterministic, seeded sample of absent classes (see
   NEG_SAMPLES_PER_PATCH) so the model also learns "no".

2) count (integer answer), ground truth = number of land-cover labels:
   - "How many distinct land cover types are present in this image?"
   - "How many land cover classes are present?"
   answer = str(len(labels))  (observed range 1..9 in this subset).

Label -> SUBJECT mapping (deterministic, applied verbatim in templates).
Kept close to the official class names; three long official names are
shortened to a fixed natural-language noun so the model sees a clean
subject phrase. This mapping is the single source of truth for both
train and eval:

    Urban fabric                                        -> "urban areas"
    Industrial or commercial units                      -> "industrial or commercial areas"
    Arable land                                         -> "cropland"
    Permanent crops                                     -> "permanent crops"
    Pastures                                            -> "pastures"
    Complex cultivation patterns                        -> "complex cultivation"
    Land principally occupied by agriculture, with
      significant areas of natural vegetation           -> "agricultural land with natural vegetation"
    Broad-leaved forest                                 -> "broad-leaved forest"
    Coniferous forest                                   -> "coniferous forest"
    Mixed forest                                        -> "mixed forest"
    Natural grassland and sparsely vegetated areas      -> "natural grassland or sparsely vegetated areas"
    Moors, heathland and sclerophyllous vegetation      -> "moors or heathland"
    Transitional woodland, shrub                        -> "transitional woodland or shrubland"
    Beaches, dunes, sands                               -> "beaches, dunes or sand"
    Inland wetlands                                     -> "inland wetlands"
    Coastal wetlands                                    -> "coastal wetlands"
    Inland waters                                       -> "inland water"
    Marine waters                                       -> "sea or ocean water"

Splits (train/val/test): the official parquet marks every matched row as
`train`, so there is no official test split for these exact patches. This
script therefore builds its own split over the 4008 matched patches with a
FIXED seed (default 42): shuffled, 80/10/10. This is a real, reproducible
split of a real labeled subset — it is documented as such, NOT claimed to
be the official BigEarthNet test set.

Outputs:
- ml/adaptation/dataset/train.jsonl, val.jsonl, test.jsonl  (one row per
  question/answer pair; image_path is relative to the repo root and points
  into the gitignored local testing/ folder)
- ml/adaptation/dataset/stats.json (real counts per question type, per
  class, per split; seed; source parquet; date; git commit)

Usage:
    python ml/adaptation/prepare_bigearthnet_vqa.py [--parquet ml/adaptation/raw/metadata.parquet]
        [--images-dir testing] [--out ml/adaptation/dataset] [--seed 42]
        [--neg-samples-per-patch 2]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Official BigEarthNet v2.0 19-class nomenclature -> question subject.
CLASS_SUBJECT = {
    "Urban fabric": "urban areas",
    "Industrial or commercial units": "industrial or commercial areas",
    "Arable land": "cropland",
    "Permanent crops": "permanent crops",
    "Pastures": "pastures",
    "Complex cultivation patterns": "complex cultivation",
    "Land principally occupied by agriculture, with significant areas of natural vegetation":
        "agricultural land with natural vegetation",
    "Agro-forestry areas": "agro-forestry areas",
    "Broad-leaved forest": "broad-leaved forest",
    "Coniferous forest": "coniferous forest",
    "Mixed forest": "mixed forest",
    "Natural grassland and sparsely vegetated areas":
        "natural grassland or sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation": "moors or heathland",
    "Transitional woodland, shrub": "transitional woodland or shrubland",
    "Beaches, dunes, sands": "beaches, dunes or sand",
    "Inland wetlands": "inland wetlands",
    "Coastal wetlands": "coastal wetlands",
    "Inland waters": "inland water",
    "Marine waters": "sea or ocean water",
}

PRESENCE_TEMPLATES = [
    "Is there {s} in this image?",
    "Does the image contain {s}?",
    "Is {s} present in this image?",
]
COUNT_TEMPLATES = [
    "How many distinct land cover types are present in this image?",
    "How many land cover classes are present?",
]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def stable_index(key: str, n: int) -> int:
    """Deterministic per-row template selection (no global RNG dependence)."""
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % n


def build_rows(parquet_path: str, images_dir: str, seed: int, neg_per_patch: int) -> dict:
    df = pd.read_parquet(parquet_path)
    targets = set(df["patch_id"])

    local = sorted(
        f[:-4] for f in os.listdir(images_dir)
        if f.endswith(".png") and not f.startswith(".") and not f.startswith("_")
    )
    matched = [p for p in local if p in targets]
    print(f"local png stems: {len(local)}, matched to parquet patch_id: {len(matched)}")

    sub = df[df["patch_id"].isin(matched)].copy()
    matched_ids = set(sub["patch_id"])
    extra_ids = [p for p in matched if p not in matched_ids]
    if extra_ids:
        print(f"  WARNING: {len(extra_ids)} local names matched parquet patch_id "
              f"strings but not parquet rows (dropped): {extra_ids[:5]}")
    matched = [p for p in matched if p in matched_ids]

    # classes actually present in this subset (17 of the 19)
    class_counts = Counter()
    for labels in sub["labels"]:
        for lab in labels:
            class_counts[lab] += 1
    present_classes = sorted(class_counts)
    print(f"classes present in subset: {len(present_classes)}")

    rng = random.Random(seed)
    rng.shuffle(matched)
    n = len(matched)
    n_train = int(n * 0.80)
    n_val = int(n * 0.10)
    split_of_patch = {}
    for i, pid in enumerate(matched):
        if i < n_train:
            split_of_patch[pid] = "train"
        elif i < n_train + n_val:
            split_of_patch[pid] = "val"
        else:
            split_of_patch[pid] = "test"

    row_id = 0
    rows = []
    per_split = Counter()
    per_type = Counter()
    per_class = Counter()  # (class, answer) where answer in {"yes","no"}
    for _, row in sub.iterrows():
        pid = row["patch_id"]
        split = split_of_patch[pid]
        labels = set(row["labels"])
        for lab in labels:
            if lab not in CLASS_SUBJECT:
                print(f"  SKIP unknown class label in parquet: {lab!r}")
                continue
        present = [l for l in sorted(labels) if l in CLASS_SUBJECT]
        absent_pool = [c for c in present_classes if c not in labels]
        if not present:
            continue  # patch has no class we can phrase
        img = os.path.join(os.path.relpath(images_dir, REPO_ROOT), pid + ".png")
        if not os.path.isfile(os.path.join(REPO_ROOT, img)):
            print(f"  SKIP missing image file: {img}")
            continue

        # count question
        qtype = "count"
        answer = str(len(present))
        q = COUNT_TEMPLATES[stable_index(pid + "::count", len(COUNT_TEMPLATES))]
        rows.append({
            "id": row_id, "patch_id": pid, "image_path": img, "split": split,
            "question_type": qtype, "question": q, "answer": answer,
            "labels": sorted(labels), "subject": None,
        })
        row_id += 1
        per_split[split] += 1
        per_type[qtype] += 1

        # presence questions: positive for every present class
        for lab in present:
            subject = CLASS_SUBJECT[lab]
            q = PRESENCE_TEMPLATES[stable_index(pid + "::" + lab, len(PRESENCE_TEMPLATES))]
            rows.append({
                "id": row_id, "patch_id": pid, "image_path": img, "split": split,
                "question_type": "presence", "question": q.replace("{s}", subject),
                "answer": "yes", "labels": sorted(labels), "subject": subject,
            })
            row_id += 1
            per_split[split] += 1
            per_type["presence"] += 1
            per_class[(subject, "yes")] += 1

        # presence questions: deterministic seeded sample of absent classes
        r_per = random.Random(int(hashlib.sha256((pid + "::neg").encode()).hexdigest()[:8], 16))
        for lab in r_per.sample(absent_pool, min(neg_per_patch, len(absent_pool))):
            subject = CLASS_SUBJECT[lab]
            q = PRESENCE_TEMPLATES[stable_index(pid + "::neg::" + lab, len(PRESENCE_TEMPLATES))]
            rows.append({
                "id": row_id, "patch_id": pid, "image_path": img, "split": split,
                "question_type": "presence", "question": q.replace("{s}", subject),
                "answer": "no", "labels": sorted(labels), "subject": subject,
            })
            row_id += 1
            per_split[split] += 1
            per_type["presence"] += 1
            per_class[(subject, "no")] += 1

    return {
        "rows": rows,
        "counts": {"per_split": dict(per_split), "per_type": dict(per_type),
                   "per_class": {f"{k[0]}::{k[1]}": v for k, v in sorted(per_class.items())}},
        "subset_classes": {c: class_counts[c] for c in present_classes},
        "n_matched": len(matched),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", default=os.path.join(REPO_ROOT, "ml", "adaptation", "raw", "metadata.parquet"))
    parser.add_argument("--images-dir", default=os.path.join(REPO_ROOT, "testing"))
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "ml", "adaptation", "dataset"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neg-samples-per-patch", type=int, default=2)
    args = parser.parse_args()

    result = build_rows(args.parquet, args.images_dir, args.seed, args.neg_samples_per_patch)
    os.makedirs(args.out, exist_ok=True)
    for split in ("train", "val", "test"):
        with open(os.path.join(args.out, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for r in result["rows"]:
                if r["split"] == split:
                    f.write(json.dumps(r) + "\n")
    stats = {
        "dataset": "BigEarthNet v2.0 labels (metadata.parquet, Zenodo 10891137) + local testing/ images",
        "license": "CDLA-Permissive-1.0",
        "seed": args.seed,
        "neg_samples_per_patch": args.neg_samples_per_patch,
        "n_matched_patches": result["n_matched"],
        "official_split_of_all_rows": "train (official parquet marks every matched row as train; "
                                      "train/val/test here is our own seeded split of that set)",
        "subset_classes_present": result["subset_classes"],
        "class_subject_mapping": CLASS_SUBJECT,
        "question_templates": {
            "presence": PRESENCE_TEMPLATES,
            "count": COUNT_TEMPLATES,
        },
        "counts": result["counts"],
        "git_commit": git_commit(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(result["counts"], indent=2))
    print(f"Wrote dataset to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())