"""Dataset adapter for the REAL, official VRSBench captioning evaluation
split. See vrsbench_vqa_adapter.py's module docstring for source/license.

Uses SatQuery's ordinary single-image VQA path with a description-style
prompt (the same "What can you tell me about this image?" framing as the
C9 demo scenario) — this is the one VRSBench task that maps cleanly onto
an existing SatQuery capability with no ontology mismatch, unlike
referring/grounding.
"""

from __future__ import annotations

import json
import os
import random
from typing import Iterator

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "vrsbench", "raw")
IMAGES_DIR = os.path.join(RAW_DIR, "Images_val")

DATASET_CITATION = (
    "VRSBench captioning eval split (Li et al. 2024, NeurIPS D&B, "
    "https://huggingface.co/datasets/xiang709/VRSBench, CC-BY-4.0)"
)


def load_real_eval_set() -> list[dict]:
    path = os.path.join(RAW_DIR, "VRSBench_EVAL_Cap.json")
    with open(path, "r") as f:
        return json.load(f)


def sample(rows: list[dict], n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    return rng.sample(rows, min(n, len(rows)))


def generate_samples(n: int = 60, seed: int = 42) -> Iterator[dict]:
    rows = sample(load_real_eval_set(), n=n, seed=seed)
    for row in rows:
        image_path = os.path.join(IMAGES_DIR, row["image_id"])
        yield {
            "id": f"vrsbench_cap_q{row['question_id']}",
            "images": [{"path": image_path, "modality": "OPTICAL", "capture_date": None}],
            "query_text": row["question"],  # "Describe the image in detail"
            "meta": {
                "ground_truth_available": True,
                "question_type": "caption",
                "ground_truth_answer": row["ground_truth"],
                "dataset": DATASET_CITATION,
            },
        }
