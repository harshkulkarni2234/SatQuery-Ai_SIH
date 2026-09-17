"""Dataset adapter for the REAL, official VRSBench VQA evaluation split
(Li et al., "VRSBench: A Versatile Vision-Language Benchmark Dataset for
Remote Sensing Image Understanding", NeurIPS 2024 Datasets & Benchmarks —
official HF dataset https://huggingface.co/datasets/xiang709/VRSBench,
CC-BY-4.0).

Raw files expected under evaluation/vrsbench/raw/ (see
evaluation/vrsbench/README.md): VRSBench_EVAL_vqa.json (flat list, real
structure) and Images_val/ (unzipped Images_val.zip).

12 real question types, no small fixed answer vocabulary for most of
them (unlike RSVQA/CDVQA) — see score_vrsbench_vqa.py for how each type
is scored.
"""

from __future__ import annotations

import json
import os
import random
from typing import Iterator

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "vrsbench", "raw")
IMAGES_DIR = os.path.join(RAW_DIR, "Images_val")

DATASET_CITATION = (
    "VRSBench VQA eval split (Li et al. 2024, NeurIPS D&B, "
    "https://huggingface.co/datasets/xiang709/VRSBench, CC-BY-4.0)"
)

NUMERIC_TYPES = {"object quantity"}
YES_NO_TYPES = {"object existence"}
# Everything else (object color/position/category/direction/size/shape,
# scene type, reasoning, image, rural or urban) is open-vocabulary
# free text — scored with contains_ground_truth_accuracy, see
# score_vrsbench_vqa.py.


def load_real_eval_set() -> list[dict]:
    path = os.path.join(RAW_DIR, "VRSBench_EVAL_vqa.json")
    with open(path, "r") as f:
        return json.load(f)


def stratified_sample(rows: list[dict], per_type: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = {}
    for row in rows:
        by_type.setdefault(row["type"], []).append(row)
    sampled = []
    for qtype, items in by_type.items():
        n = min(per_type, len(items))
        sampled.extend(rng.sample(items, n))
    return sampled


def generate_samples(per_type: int = 20, seed: int = 42) -> Iterator[dict]:
    rows = stratified_sample(load_real_eval_set(), per_type=per_type, seed=seed)
    for row in rows:
        image_path = os.path.join(IMAGES_DIR, row["image_id"])
        yield {
            "id": f"vrsbench_vqa_q{row['question_id']}_{row['type'].replace(' ', '_')}",
            "images": [{"path": image_path, "modality": "OPTICAL", "capture_date": None}],
            "query_text": row["question"],
            "meta": {
                "ground_truth_available": True,
                "question_type": row["type"],
                "ground_truth_answer": row["ground_truth"],
                "dataset": DATASET_CITATION,
            },
        }
