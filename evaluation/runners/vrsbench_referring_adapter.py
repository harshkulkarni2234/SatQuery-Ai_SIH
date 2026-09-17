"""Dataset adapter for the REAL, official VRSBench referring/grounding
evaluation split. See vrsbench_vqa_adapter.py's module docstring for
source/license.

IMPORTANT, verified empirically before this adapter was written (not
assumed): VRSBench's referring expressions ask about DOTA-style
individual objects — vehicle, ship, airplane, harbor, bridge,
storage-tank, etc. (26 real classes; see the real distribution check in
this project's session log). SatQuery's own grounding specialist
(`backend/app/services/grounding.py`) supports exactly 5 land-cover
targets: water, vegetation, built-up, roads, farmland — a completely
different object ontology, zero overlap. A live test with a real
VRSBench-style referring sentence confirmed the query never even reaches
GROUNDING — the planner classifies it as VQA (no recognized target
keyword in the text), and whatever box comes back is VQA's incidental
`visual_evidence` box, unrelated to the actual referred object.

This is scored honestly for exactly what it is: primarily a
task-routing-mismatch measurement (what fraction ever reach GROUNDING at
all — expected near/exactly 0%), with IoU reported only as a secondary,
clearly-caveated statistic. See score_vrsbench_referring.py.
"""

from __future__ import annotations

import json
import os
import random
from typing import Iterator, Optional

from PIL import Image

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "vrsbench", "raw")
IMAGES_DIR = os.path.join(RAW_DIR, "Images_val")

DATASET_CITATION = (
    "VRSBench referring/grounding eval split (Li et al. 2024, NeurIPS D&B, "
    "https://huggingface.co/datasets/xiang709/VRSBench, CC-BY-4.0)"
)


def parse_normalized_box(box_str: str) -> Optional[tuple[float, float, float, float]]:
    """Parses VRSBench's "{<x1><y1><x2><y2>}" ground-truth format (each
    value 0-100, normalized to image width/height) into a plain 4-tuple.
    Returns None on any malformed input rather than guessing."""
    import re
    numbers = re.findall(r"<(\d+(?:\.\d+)?)>", box_str)
    if len(numbers) != 4:
        return None
    return tuple(float(n) for n in numbers)


def _real_image_size(path: str) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size  # (width, height)


def load_real_eval_set() -> list[dict]:
    path = os.path.join(RAW_DIR, "VRSBench_EVAL_referring.json")
    with open(path, "r") as f:
        return json.load(f)


def stratified_sample(rows: list[dict], n: int, seed: int = 42) -> list[dict]:
    """No per-type stratification here (referring has one type, 'ref') —
    just a documented seeded sample by real object class, so the small
    sample still spans a range of the 26 real classes rather than
    clustering on whichever happens to sort first."""
    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = {}
    for row in rows:
        by_class.setdefault(row.get("obj_cls", "unknown"), []).append(row)
    per_class = max(1, n // max(1, len(by_class)))
    sampled = []
    for cls, items in by_class.items():
        sampled.extend(rng.sample(items, min(per_class, len(items))))
    return sampled[:n]


def generate_samples(n: int = 100, seed: int = 42) -> Iterator[dict]:
    rows = stratified_sample(load_real_eval_set(), n=n, seed=seed)
    for row in rows:
        image_path = os.path.join(IMAGES_DIR, row["image_id"])
        gt_box_normalized = parse_normalized_box(row["ground_truth"])
        if gt_box_normalized is None:
            continue
        yield {
            "id": f"vrsbench_ref_q{row['question_id']}",
            "images": [{"path": image_path, "modality": "OPTICAL", "capture_date": None}],
            "query_text": row["question"],
            "meta": {
                "ground_truth_available": True,
                "question_type": "ref",
                "obj_cls": row.get("obj_cls"),
                "ground_truth_box_normalized_0_100": gt_box_normalized,
                "image_path": image_path,
                "dataset": DATASET_CITATION,
            },
        }
