"""Dataset adapter for the REAL, official RSVQA-LR test split (Lobry et al.,
"RSVQA: Visual Question Answering for Remote Sensing Data", Zenodo record
6344334, CC-BY-4.0 — https://zenodo.org/records/6344334). This is a real,
officially labeled benchmark (unlike testing_folder_adapter.py's unlabeled
smoke test) — ground truth answers ARE available here, so a real accuracy
score can finally be computed.

Raw files expected under evaluation/rsvqa_lr/raw/ (fetched from the Zenodo
record above, not committed to git — see evaluation/rsvqa_lr/README.md):
  - Images_LR/                       (unzipped Images_LR.zip, 773 .tif files)
  - LR_split_test_questions.json
  - LR_split_test_answers.json
  - LR_split_test_images.json

The official test split has 10,004 labeled questions across 100 images.
Running all 10,004 through real (serial, single-inference-slot) VQA model
calls would take on the order of 14 hours — not a smoke run. Per this
project's own established practice (see the plan's own B4 guidance: "if
full test set is too slow, use a documented, seeded, stratified subset and
label it as a subset"), this adapter draws a fixed-seed stratified sample
per question type instead of the full set.
"""

from __future__ import annotations

import json
import os
import random
from typing import Iterator

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "rsvqa_lr", "raw")
IMAGES_DIR = os.path.join(RAW_DIR, "Images_LR")

DATASET_CITATION = (
    "RSVQA-LR test split (Lobry et al. 2020, Zenodo record 6344334, "
    "CC-BY-4.0) — https://zenodo.org/records/6344334"
)


def _load_active(filename: str, key: str) -> dict:
    path = os.path.join(RAW_DIR, filename)
    with open(path, "r") as f:
        data = json.load(f)
    return {item["id"]: item for item in data[key] if item.get("active")}


def load_real_test_set() -> list[dict]:
    """Returns every real (question, answer, image) triple in the official
    test split — no sampling, no filtering beyond what the split itself
    defines. 10,004 real rows."""
    questions = _load_active("LR_split_test_questions.json", "questions")
    answers_by_qid = {}
    for a in json.load(open(os.path.join(RAW_DIR, "LR_split_test_answers.json")))["answers"]:
        if a.get("active"):
            answers_by_qid[a["question_id"]] = a

    rows = []
    for qid, q in questions.items():
        answer = answers_by_qid.get(qid)
        if answer is None:
            continue
        rows.append({
            "question_id": qid,
            "image_id": q["img_id"],
            "question_type": q["type"],
            "question": q["question"],
            "answer": answer["answer"],
        })
    return rows


def stratified_sample(rows: list[dict], per_type: int, seed: int = 42) -> list[dict]:
    """Deterministic, seeded, per-question-type sample — documented here
    and in every result file this produces, not silently substituted for
    the full set."""
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = {}
    for row in rows:
        by_type.setdefault(row["question_type"], []).append(row)
    sampled = []
    for qtype, items in by_type.items():
        n = min(per_type, len(items))
        sampled.extend(rng.sample(items, n))
    return sampled


def generate_samples(per_type: int = 60, seed: int = 42) -> Iterator[dict]:
    """Yields runner-shaped samples (see runners/base.py) with real
    ground truth carried in `meta` — never used to influence the request
    sent to the API, only compared against the response afterward."""
    rows = stratified_sample(load_real_test_set(), per_type=per_type, seed=seed)
    for row in rows:
        image_path = os.path.join(IMAGES_DIR, f"{row['image_id']}.tif")
        yield {
            "id": f"rsvqa_lr_test_q{row['question_id']}",
            "images": [{"path": image_path, "modality": "OPTICAL", "capture_date": None}],
            "query_text": row["question"],
            "meta": {
                "ground_truth_available": True,
                "question_type": row["question_type"],
                "ground_truth_answer": row["answer"],
                "dataset": DATASET_CITATION,
            },
        }
