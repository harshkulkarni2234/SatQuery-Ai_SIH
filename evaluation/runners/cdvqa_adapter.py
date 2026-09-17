"""Dataset adapter for the REAL, official CDVQA test split (Yuan et al.,
"Change Detection Meets Visual Question Answering", IEEE TGRS 2022 —
official repo https://github.com/YZHJessica/CDVQA, Apache-2.0 for the
question/answer labels). CDVQA's images are not shipped in that repo —
they reference the underlying SECOND semantic change detection dataset
(https://captain-whu.github.io/SCD/, license unstated on the official
page — flagged to and approved by the user before fetching).

Raw files expected under:
  - evaluation/cdvqa/raw/Test_{questions,answers,images}.json (from the
    CDVQA repo directly — see evaluation/cdvqa/README.md)
  - evaluation/second_dataset/raw/ (unzipped SECOND download — see
    evaluation/second_dataset/README.md), expected im1/, im2/ subfolders
    of same-named .png files (pre-event / post-event)

This is a bi-temporal (2-image) benchmark with genuine semantic
questions about WHICH land-cover class changed, not just whether pixels
differ — something SatQuery's existing change.deterministic_cv specialist
was never designed to answer (it measures aggregate pixel/area change,
not per-class semantic change). Running this eval is expected to reveal
that gap honestly, not paper over it.
"""

from __future__ import annotations

import json
import os
import random
from typing import Iterator

CDVQA_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "cdvqa", "raw")
SECOND_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "second_dataset", "raw")

DATASET_CITATION = (
    "CDVQA test split (Yuan et al. 2022, https://github.com/YZHJessica/CDVQA, "
    "Apache-2.0 labels) over SECOND dataset images "
    "(https://captain-whu.github.io/SCD/, license unstated)"
)

LAND_COVER_CLASSES = [
    "NVG_surface", "buildings", "low_vegetation", "playgrounds", "trees", "water",
]
CHANGE_RATIO_BUCKETS = [
    "0", "0_to_10", "10_to_20", "20_to_30", "30_to_40", "40_to_50",
    "50_to_60", "60_to_70", "70_to_80", "80_to_90", "90_to_100",
]

YES_NO_TYPES = {"change_or_not", "increase_or_not", "decrease_or_not"}
CATEGORICAL_TYPES = {"smallest_change", "largest_change", "change_to_what"}
BUCKET_TYPES = {"change_ratio", "change_ratio_types"}


def _load_active(filename: str, key: str) -> dict:
    path = os.path.join(CDVQA_RAW_DIR, filename)
    with open(path, "r") as f:
        data = json.load(f)
    return {item["id"]: item for item in data[key] if item.get("active")}


def load_real_test_set(split: str = "Test") -> list[dict]:
    """Returns every real (question, answer, image-pair) row in the given
    split ("Test" or "Test2" — Test2 is a paraphrase-robustness variant
    over the SAME 968 image pairs, not additional data)."""
    questions = _load_active(f"{split}_questions.json", "questions")
    images = _load_active(f"{split}_images.json", "images")
    answers_by_qid = {}
    for a in json.load(open(os.path.join(CDVQA_RAW_DIR, f"{split}_answers.json")))["answers"]:
        if a.get("active"):
            answers_by_qid[a["question_id"]] = a

    rows = []
    for qid, q in questions.items():
        answer = answers_by_qid.get(qid)
        image = images.get(q["img_id"])
        if answer is None or image is None:
            continue
        rows.append({
            "question_id": qid,
            "file_name": image["file_name"],
            "question_type": q["type"],
            "question": q["question"],
            "answer": answer["answer"],
        })
    return rows


def stratified_sample(rows: list[dict], per_type: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = {}
    for row in rows:
        by_type.setdefault(row["question_type"], []).append(row)
    sampled = []
    for qtype, items in by_type.items():
        n = min(per_type, len(items))
        sampled.extend(rng.sample(items, n))
    return sampled


def generate_samples(per_type: int = 30, seed: int = 42, split: str = "Test") -> Iterator[dict]:
    """Yields runner-shaped bi-temporal samples with real ground truth in
    `meta`. Each sample carries the SAME image pair for both `im1`
    (pre-event) and `im2` (post-event) paths — file_name is shared across
    both subfolders."""
    rows = stratified_sample(load_real_test_set(split), per_type=per_type, seed=seed)
    for row in rows:
        before_path = os.path.join(SECOND_IMAGES_DIR, "im1", row["file_name"])
        after_path = os.path.join(SECOND_IMAGES_DIR, "im2", row["file_name"])
        yield {
            "id": f"cdvqa_{split.lower()}_q{row['question_id']}",
            "images": [
                {"path": before_path, "modality": "OPTICAL", "capture_date": None},
                {"path": after_path, "modality": "OPTICAL", "capture_date": None},
            ],
            "query_text": row["question"],
            "meta": {
                "ground_truth_available": True,
                "question_type": row["question_type"],
                "ground_truth_answer": row["answer"],
                "dataset": DATASET_CITATION,
            },
        }
